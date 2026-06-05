**Title:** `ggml-backend : sched mutates user gf src pointers in place, leaving dangling pointers across alloc_graph calls`

---

`ggml_backend_sched_split_graph` rewires `node->src[j]` of the user's
graph to point at internal `input_cpy` tensors it allocates in
`sched->ctx`. Those tensors live until the *next* call to
`ggml_backend_sched_split_graph`, which begins with
`ggml_free(sched->ctx); sched->ctx = ggml_init(params)` and
invalidates them. Between sched calls, the user's `gf->nodes[i]->src[j]`
is therefore a dangling pointer.

When the user calls `ggml_backend_sched_alloc_graph` again on the same
gf — a common pattern, e.g. classifier-free guidance running the
same compute graph twice per step with different inputs — the next
`sched_split_graph` reads from those dangling pointers to decide
whether each src needs a new cross-backend copy. The garbage flags
rarely match `GGML_TENSOR_FLAG_INPUT`, so it silently skips creating
the input copy. The consuming GPU kernel then reads whatever stale
data is at the previous `input_cpy`'s offset.

## Repro

Application-level repro using the chatterbox S3Gen UNet (CFM solver
runs the same gf twice per CFM step):

```
# Without the patch: rms 14.6 (garbled audio)
CHATTERBOX_FORCE_GPU=1 CHATTERBOX_S3GEN_UNET_GPU_RESIDENCY=1 \
  ./chatterbox-cli --tts "Hello." --voice prompt.wav --seed 42

# Diagnostic: the dangling src[j] makes split[0] of the SECOND
# graph_compute report n_inputs=0 (the original GGML_TENSOR_FLAG_INPUT
# tensor "unet_input" is invisible because src[j] now points to freed
# memory). The first graph_compute reports n_inputs=1 'unet_input'.
```

A minimal `test-backend-ops`-style repro would be:

1. Build a small gf with one input tensor whose first consumer is on
   a different backend than the input's auto-assigned backend.
2. Call `ggml_backend_sched_alloc_graph + graph_compute` twice with
   different input uploads.
3. The second compute reads stale data, even though the user uploaded
   fresh data via `ggml_backend_tensor_set` between the two calls.

We can prepare one if it helps the upstream review — flag the issue.

## Proposed fix

Track src[j] rewires in a per-sched mutation log; restore each rewire
to the original at the end of `ggml_backend_sched_compute_splits` so
the user's gf is left in its original state between sched calls.
~30 LOC across struct, split_graph, compute_splits, free.

Patch in our fork: `ggml/src/ggml-backend.cpp` (`// CrispASR patch
(#83 r9 follow-up #4)` blocks). Posting the diff below for review:

```c
// (struct ggml_backend_sched_src_mutation declared at file scope)
struct ggml_backend_sched_src_mutation {
    struct ggml_tensor * node;
    struct ggml_tensor * orig_src;
    int j;
};

// inside struct ggml_backend_sched: add fields
struct ggml_backend_sched_src_mutation * src_mutations;
int n_src_mutations;
int src_mutations_capacity;

// in ggml_backend_sched_split_graph, BEFORE node->src[j] = tensor_id_copy(...):
if (sched->n_src_mutations >= sched->src_mutations_capacity) {
    sched->src_mutations_capacity = sched->src_mutations_capacity * 2 + 16;
    sched->src_mutations = realloc(
        sched->src_mutations,
        sched->src_mutations_capacity * sizeof(struct ggml_backend_sched_src_mutation));
    GGML_ASSERT(sched->src_mutations != NULL);
}
sched->src_mutations[sched->n_src_mutations].node = node;
sched->src_mutations[sched->n_src_mutations].orig_src = src;
sched->src_mutations[sched->n_src_mutations].j = j;
sched->n_src_mutations++;
node->src[j] = tensor_id_copy(...);

// at the END of ggml_backend_sched_compute_splits:
for (int i = 0; i < sched->n_src_mutations; i++) {
    const struct ggml_backend_sched_src_mutation * m = &sched->src_mutations[i];
    m->node->src[m->j] = m->orig_src;
}
sched->n_src_mutations = 0;

// in ggml_backend_sched_free:
free(sched->src_mutations);
```

## What this DOES fix

The dangling-pointer bug above. After this patch, repeated
`alloc_graph + graph_compute` on the same gf correctly detects input
tensors on every call and queues the CPU→GPU copies. Verified with
the chatterbox repro: `split[0].n_inputs=1 'unet_input'` on every
CFM-step call (both cond and uncond passes), kernel reads correct
input bytes (verified via inline `ggml_backend_tensor_get` right
before im2col dispatch).

## What this does NOT fix

A separate, residual issue with `unet_input` specifically: even
after the dangling-pointer fix, with `unet_input` on the CPU
backend (auto-assigned) and the sched copy delivering correct
bytes to `MTL0#unet_input#0`, downstream Metal compute still
produces wrong output (chatterbox `vocoder mel rms ~14.6` vs ref
5.1). The kernel sees correct input bytes but the cumulative
compute diverges from the reference.

Application-side workaround (still required in our fork): pin
`unet_input` to the Metal backend via
`ggml_backend_sched_set_tensor_backend(sched, unet_input, c->backend)`.
With both the ggml patch AND the app pin in place, `s3gen_mel
cos_min` goes from 0.940 → 0.999976 (matching the CPU-residency
production path's 0.999980).

The second issue may be a Metal-side correctness bug, a precision
issue with input-copy via the shared-buffer path, or yet another
sched-state issue. We don't have a clean characterisation yet —
filing this PR for the dangling-pointer half because that one is
crisp and reproducible.

---

**Update (2026-05-24, late):** The second issue ("the FIRST split's
sched-copy doesn't always reach the GPU even with this patch") was
diagnosed in R9 follow-up #5 and turned out to be unrelated to the
dangling-pointer issue this PR addresses. It was a Metal cache
coherency issue on shared-storage buffers across back-to-back
command-buffer submissions where the CPU writes between submissions:
the default `[cmd_buf_last waitUntilCompleted]` between-submissions
sync does not invalidate the GPU's L1/L2 cached view of a shared-
storage `MTLBuffer` the CPU just memcpy'd, so the next compute reads
stale data.

The fix for that second issue is to construct the sched with
`ggml_backend_sched_new(..., /*parallel=*/true, ...)`, which uses
`MTLSharedEvent`-backed `ggml_backend_event_record / event_wait`
between submissions; the encoded `encodeSignalEvent` /
`encodeWaitForEvent` commands carry proper GPU cache invalidation.
Once both `parallel=true` and this PR's dangling-pointer patch are in
place, the chatterbox app-side workaround (pinning `unet_input`)
is no longer needed.

This PR's dangling-pointer fix is still independently necessary: the
mutation-log restoration is correct under either `parallel=false` or
`parallel=true`, and other ggml users hitting the dangling-src[j]
behaviour will see correctness problems regardless of `parallel`.
