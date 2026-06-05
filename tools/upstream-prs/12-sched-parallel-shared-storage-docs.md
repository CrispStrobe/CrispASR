**Title:** `ggml-backend / ggml-metal : document that sched parallel=true is required for back-to-back compute calls on the same gf with CPU-side memcpy between them on shared-storage buffers (M1 Metal)`

---

## Background

A user calling
`ggml_backend_sched_new(backends, ..., /*parallel=*/false, /*op_offload=*/false)`
and then doing two `ggml_backend_sched_graph_compute` calls on the
same `ggml_cgraph` — with the user re-uploading new input data to a
`GGML_TENSOR_FLAG_INPUT` tensor on the host backend in between via
`ggml_backend_tensor_set` — observes that the **second** compute on
the consuming Metal backend reads stale data. The first compute is
correct.

A `ggml_backend_tensor_get` on the same buffer offset immediately
before the second compute's first kernel dispatch returns the
correct host bytes the user just wrote. But the kernel itself
produces output consistent with the *previous* compute's leftover
buffer state at that offset. Calling
`ggml_backend_sched_new(..., /*parallel=*/true, ...)` instead fixes
the divergence.

## Why this happens (best understanding from our investigation)

`parallel=false` is the default for many existing ggml users. With
`parallel=false`, `sched->n_copies = 1` and
`ggml_backend_sched_compute_splits` synchronises between split
backends via `ggml_backend_synchronize` (Metal:
`[cmd_buf_last waitUntilCompleted]`).

`waitUntilCompleted` blocks for the prior command buffer's
completion but does NOT invalidate the GPU's L1/L2 cached view of
a shared-storage `MTLBuffer` whose system memory was overwritten by
a host-side `memcpy` between consecutive `commandBuffer` submissions.
The CPU memcpy lands in system memory through SLC, but the GPU
caches retain their warm view from the prior compute submission's
reads, and the next dispatch reads stale data.

With `parallel=true`, `sched->n_copies = GGML_SCHED_MAX_COPIES = 4`
input-copy slots are pre-allocated, the per-call `cur_copy`
alternates, and `compute_splits` uses
`ggml_backend_event_record` / `event_synchronize` for the inter-
submission sync. On Metal the events are backed by
`MTLSharedEvent` whose `encodeSignalEvent` /
`encodeWaitForEvent` commands carry GPU-cache-invalidation
semantics for the buffer they apply to. The next submission reads
fresh data and the compute is correct.

(Whether the actual fix vector is "different slot offset every
other call so no warm cache" or "MTLSharedEvent fences include
cache invalidation" or both is still ambiguous from outside the
Metal driver — but either way `parallel=true` resolves it.)

## Repro shape

Minimal repro pattern (we have an end-to-end application reproducer
in the chatterbox CFM solver but can prepare a test-backend-ops
case if helpful):

1. Create a sched with two backends `[metal, cpu]` and
   `parallel=false`. (Set up so that a Metal split has a
   `GGML_TENSOR_FLAG_INPUT` tensor whose first consumer is on
   the Metal backend.)
2. `ggml_backend_sched_alloc_graph + graph_compute` once with
   `tensor_set(input, host_data_A)` first.
3. `ggml_backend_sched_alloc_graph + graph_compute` AGAIN on the
   SAME gf, with `tensor_set(input, host_data_B)` first.
4. Read back the Metal-resident output of the second compute.
   Compare to a CPU-only or `parallel=true` run as ground truth.

Without `parallel=true`, the second-call output diverges (we saw
cos≈0.21 between the broken and correct outputs in our use case).

## Proposed change

Two options, can be combined:

### A. Documentation-only (low-impact)

Extend the doc comment above the `parallel` parameter of
`ggml_backend_sched_new` in `ggml/include/ggml-backend.h`:

```c
// parallel: if true, sched allocates GGML_SCHED_MAX_COPIES input-
//   copy slots per cross-backend input and uses event-based
//   synchronisation between submissions. Required on Metal if your
//   workflow calls graph_compute back-to-back on the same gf with
//   the host overwriting an input tensor between calls
//   (e.g. classifier-free guidance, iterative solvers re-feeding
//   the same input slot). Without parallel=true, the second
//   compute may read stale data the host writes don't reach due
//   to GPU cache state retained across submissions. See PR
//   ggml-org/ggml#NNN for the analysis.
```

### B. Conservative behaviour change (bigger)

Have `ggml_backend_sched_compute_splits` call
`ggml_backend_event_synchronize` (or equivalent stronger sync) even
in the `parallel=false` path before re-uploading
`GGML_TENSOR_FLAG_INPUT` tensors when the consuming backend is a
GPU device with shared-storage buffers. This is more invasive but
removes the footgun for users who don't know about the `parallel`
flag.

We can prepare either side as a PR. Filing this issue first since
the docs gap was material (it cost us multiple days to track down,
and we eliminated ~10 unrelated hypotheses before finding the
`parallel` flag fix).

## Related

- Issue #NN (PR 10 in our internal queue) reports a separate
  sched-side bug: the dangling `node->src[j]` pointers after
  `sched->ctx` is freed between calls. That fix is independent
  and continues to be required — both fixes ship together in our
  fork.
