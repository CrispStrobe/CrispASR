**Title:** `ggml-cuda : support per-head additive mask in FLASH_ATTN_EXT`

**Status:** designed, not yet implemented — see "Implementation
scope" below for why this needs a real kernel signature change
across multiple `.cuh` files. Tracking as the next concrete PR
target after #05 lands.

---

`ggml_cuda_get_best_fattn_kernel` in `ggml/src/ggml-cuda/fattn.cu`
unconditionally rejects flash-attn-ext invocations whose mask has
`mask->ne[2] != 1`:

```cuda
if (mask && mask->ne[2] != 1) {
    return BEST_FATTN_KERNEL_NONE;
}
```

This rules out all FA kernels for transformer-XL / FastConformer
style relative-position-bias attention where the additive mask
is *per-head* (typically `(T_kv, T_q, n_heads, 1)` in fp16).
Conformer / Parakeet / Canary all use this construction; on CUDA
they all fall back to CPU, producing one CPU split per layer ×
`n_layers` layers per chunk = the dominant per-chunk cost on
chunked-streaming inference. On parakeet-tdt-0.6b-v3 / RTX A1000
Laptop / Windows-WDDM that's ~15-25 % wallclock.

Inspection of the MMA-F16 kernel (`fattn-mma-f16.cuh`) and its
launchers (`fattn-common.cuh`) confirms the kernel signature
takes a single `mask_h` pointer + `stride_mask` (a 2D mask). The
top-level launcher (line ~1635 of `fattn-mma-f16.cuh`) advances
the pointer by `nb33*(sequence % ne33)` per batch but **not** by
head. WMMA-F16, TILE, and VEC kernels follow the same pattern.
So per-head support requires:

1. Adding a `stride_mask_head` parameter to the kernel signature.
2. Advancing `mask_h` by `head_idx * stride_mask_head` inside
   the per-head iteration (which the kernel already runs over).
3. Plumbing the new parameter through `launch_fattn_*` /
   `ggml_cuda_flash_attn_ext` / `flash_attn_ext_f16_case` and
   the `BEST_FATTN_KERNEL_*` dispatch table.
4. Loosening the `ne[2] != 1` guard in `get_best_fattn_kernel`
   (with a parallel guard ensuring at least one kernel still
   supports the new shape — TILE and WMMA may not gain support
   in the same PR).
5. Adding a `test-backend-ops` case for per-head masks.

## Implementation scope

This is a real kernel-signature change touching at least:

- `fattn.cu` (selector relaxation)
- `fattn-common.cuh` (launcher signatures)
- `fattn-mma-f16.cuh` (per-head pointer advance)
- Optionally `fattn-wmma-f16.cu`, `fattn-tile.cu`, `fattn-vec.cu`
- `test-backend-ops.cpp` (new test case)

Estimated ~100-200 LOC across 4-6 files. Needs full
`test-backend-ops` verification across at least sm_75 (Turing),
sm_86 (Ampere), and sm_89 (Ada) before landing. Not session-scope
work; tracked here so whoever picks it up has the call-chain
analysis already done.

## CrispASR carry strategy

For now: gate behind `GGML_CUDA_CRISPASR_FA_PERHEAD_MASK=OFF`
(default) in `ggml/CMakeLists.txt`. When/if a future commit lands
the actual kernel changes here, flip the default ON for our
internal builds (after a `test-backend-ops` + parakeet WER
verification on the A1000 hardware), keeping OFF as the upstream-
shape default. Same compile-knob pattern as #05.

**Expected speedup on parakeet-tdt-0.6b-v3 / A1000 Laptop:**
~15-25 % wallclock based on the GGML_SCHED_DEBUG split count
(`FLASH_ATTN_EXT` accounts for 24 of the 48 CPU splits per chunk;
each split represents ~1 cudaStreamSynchronize and ~2
cudaMemcpyAsync = ~10-20 % of post-CUDA-Graphs wallclock).
Estimated baseline: 3.063 s long-clip post-#05; expected
post-#06: ~2.5 s. The remaining gap to onnx-fp32 (1.537 s) is
the structural cuDNN-conv advantage discussed in PERFORMANCE.md
"A1000 Ampere CUDA A/B".

## What we did try and abandon

Two client-side workarounds attempted, both regressed:

1. `op_offload=true` in `ggml_backend_sched_new` — +87 % regression
   (re-uploads weights per call).
2. Folding BD into Q before FA — semantic change to the model,
   would need re-training to validate.

The real fix has to land inside ggml-cuda; client-side workarounds
perturb the graph in ways CUDA Graphs and sched's allocator
weren't tuned for.
