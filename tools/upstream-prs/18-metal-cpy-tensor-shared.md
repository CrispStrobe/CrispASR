**Title:** `metal : implement cpy_tensor for shared buffers (avoid host-staging copy)`

**Status (2026-09-07):** Retired. Upstream v0.23 directly copies compatible shared Metal buffers.


**Repo:** ggml-org/ggml (direct — `src/ggml-metal/**`, same as merged PR #04).

---

## Background

`ggml_backend_tensor_copy(src, dst)` copies between two tensors. When
neither buffer reports `is_host`, it delegates to
`dst_buf->iface.cpy_tensor(...)`; if that returns `false` it falls back
to a **malloc + `get_tensor` (device→host) + `set_tensor` (host→device)
+ free** staging round-trip (`ggml-backend.cpp`):

```c
} else if (!ggml_backend_buffer_copy_tensor(src, dst)) {
    size_t nbytes = ggml_nbytes(src);
    void * data = malloc(nbytes);
    ggml_backend_tensor_get(src, data, 0, nbytes);
    ggml_backend_tensor_set(dst, data, 0, nbytes);
    free(data);
}
```

The Metal **shared** (unified-memory) buffer never implements the fast
path — `ggml_backend_metal_buffer_shared_cpy_tensor` returns `false`
unconditionally:

```c
static bool ggml_backend_metal_buffer_shared_cpy_tensor(
        ggml_backend_buffer_t buffer, const ggml_tensor * src, ggml_tensor * dst) {
    ...
    GGML_UNUSED(src);
    GGML_UNUSED(dst);
    return false;   // <-- always
}
```

So every same-backend Metal→Metal `ggml_backend_tensor_copy` allocates a
full-tensor host buffer and bounces the data through it — on unified
memory, where the source and destination are *already the same kind of
host-addressable pointer*. The shared buffer's own `get_tensor` /
`set_tensor` are plain unsynchronized `memcpy`s against
`get_base()`-derived pointers, so a direct `memcpy` between two shared
tensors is both correct and strictly cheaper (no allocation, no double
copy).

## Where it bites

Discovered while porting CrispASR's branched beam-search KV-cache
snapshot to on-device copies (our issue #161). The branched-beam decoder
snapshots/restores the decoder KV cache once per beam per step via
`ggml_backend_tensor_copy`. We expected device-to-device blits; on Metal
the unconditional `false` silently routed every snapshot through the
malloc-bounce path, which (with a recycled snapshot pool) measured
*slower* than the original explicit `tensor_get`/`set` it replaced
(`UNACCOUNTED` 705 → 2168 ms on M1 before we special-cased Metal back to
pooled host buffers). The slowdown is pure allocator + redundant-copy
overhead introduced by the fallback, not the copy itself.

Any caller doing same-backend tensor copies on Metal (KV snapshots, beam
search, graph i/o staging) hits this.

## Fix

Implement the shared-buffer `cpy_tensor`: when the source is also
host-addressable (a Metal shared buffer of the same buffer type, or a
host buffer), copy in place with a single `memcpy`; otherwise return
`false` and let the generic path handle it. Sketch:

```c
static bool ggml_backend_metal_buffer_shared_cpy_tensor(
        ggml_backend_buffer_t buffer, const ggml_tensor * src, ggml_tensor * dst) {
    ggml_metal_buffer_t ctx = (ggml_metal_buffer_t)buffer->context;
    GGML_ASSERT(ggml_metal_buffer_is_shared(ctx));

    // dst is in this shared (unified-memory) buffer. If src is also
    // host-addressable, the copy is a plain memcpy with no staging.
    if (ggml_backend_buffer_is_host(src->buffer) ||
        src->buffer->buft == dst->buffer->buft) {   // both Metal shared
        memcpy(dst->data, src->data, ggml_nbytes(src));
        return true;
    }
    return false;
}
```

Reference patch: `18-metal-cpy-tensor-shared.patch` (1 file, ~10 LOC).

This matches the existing shared `get_tensor`/`set_tensor` contract:
they already `memcpy` against unified pointers without inserting a GPU
sync, so callers are already responsible for ordering w.r.t. in-flight
command buffers (e.g. `ggml_backend_sched_graph_compute` synchronizes
before any host-visible read). `cpy_tensor` inherits exactly that
contract — no new synchronization semantics.

The same pattern can be applied to the **private** buffer iface
(`ggml_backend_metal_buffer_private_cpy_tensor`, also `return false`)
when both src and dst are private Metal buffers of the same type, via a
blit encoder — left out of this PR to keep it to the unified-memory case
that the macOS/iOS/Asahi GPUs actually use.

## Provenance & verification

Ours — found via the CrispASR #161 KV-snapshot work; the fix is a couple
lines mirroring the existing `get_tensor` access pattern. Validate with
`test-backend-ops` on Metal (any op whose graph i/o triggers a
same-backend copy) and a direct two-shared-tensor `ggml_backend_tensor_copy`
round-trip checked bit-exact against the source.

> Filing note: we already have a merged ggml PR (#1477), so the
> first-contributor "1 open PR" cap does not gate us at ggml-org/ggml.
> Implement + run the tests below for real, then fill the actual results
> into the body before opening.

---

## PR body (ready to file — verify test results first)

Title: `metal : implement cpy_tensor for shared buffers`

> Summary
>
> `ggml_backend_tensor_copy()` falls back to a `malloc` + `get_tensor` +
> `set_tensor` + `free` host round-trip whenever the destination buffer's
> `cpy_tensor` returns false. The Metal shared (unified-memory) buffer
> interface never implements it — `ggml_backend_metal_buffer_shared_cpy_tensor`
> returns `false` unconditionally — so a same-backend copy between two shared
> tensors allocates a full-size host staging buffer and copies the data twice,
> on memory that is already host-addressable at both ends.
>
> This adds the fast path: if the source is also host-addressable (a host
> buffer, or another Metal shared buffer of the same type), copy in place with
> a single `memcpy`; otherwise keep returning `false` so the generic path
> handles it.
>
> ```c
> if (ggml_backend_buffer_is_host(src->buffer) ||
>     src->buffer->buft == dst->buffer->buft) {
>     memcpy(dst->data, src->data, ggml_nbytes(src));
>     return true;
> }
> return false;
> ```
>
> The shared-buffer `get_tensor`/`set_tensor` already `memcpy` against
> `get_base()`-derived pointers without inserting a GPU sync, so this inherits
> the exact same ordering contract — callers already synchronize before any
> host-visible read (e.g. `ggml_backend_sched_graph_compute`). No new
> synchronization semantics.
>
> Motivation: I hit this doing per-beam KV-cache snapshots in a downstream
> project. The snapshots use `ggml_backend_tensor_copy` between same-backend
> tensors expecting an in-place copy; on Metal the unconditional `false`
> routed every one through the malloc-bounce, measurably slower than a plain
> `memcpy`.
>
> The private-buffer `cpy_tensor` has the same `return false` and could take a
> blit-encoder implementation later; I kept this PR to the unified-memory case
> that Apple/Asahi GPUs actually use.
>
> Testing
> - `test-backend-ops` on Metal (Apple M1) — no regressions. *(fill exact pass count)*
> - Direct round-trip: allocate two shared tensors, `ggml_backend_tensor_copy(a, b)`,
>   confirm `b` is bit-identical to `a`. *(fill result)*
>
> ---
> AI assistance: an AI coding tool helped me locate the `ggml_backend_tensor_copy`
> fallback path and draft the test harness; the change and this description are
> mine and I've reviewed them. *(keep/adjust to match ggml's disclosure norms)*
