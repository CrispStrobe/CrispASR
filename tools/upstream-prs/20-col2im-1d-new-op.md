**Title:** `ggml : add col2im_1d — composable building block for ConvTranspose1d`

**Status (2026-09-07):** Retired as an outbound proposal: upstream added `col2im_1d` in `d962a305`. CrispASR now requests the full signal and crops through a view; the CPU audio-op parity gate passes on v0.23.


---

## What this is

`GGML_OP_COL2IM_1D` (CrispASR-local, PR #160) is the inverse-of-`im2col`
gather that completes a transposed 1-D convolution expressed as a GEMM:

```
ConvTranspose1d(x, W, stride) ==
    col = mul_mat(W_perm, x)        // [K*OC, T_in]   (W pre-permuted to [IC, K*OC])
    y   = col2im_1d(col, stride, OC, crop)   // [T_out, OC],  T_out = (T_in-1)*stride + K - crop
```

ggml today has `im2col` / `im2col_back` and a *monolithic*
`conv_transpose_1d`, but no forward `col2im_1d`, so the decomposition above
isn't expressible upstream. We added it because the decomposed form:

- reuses each backend's heavily-tuned `mul_mat` for the contraction (the
  monolithic `conv_transpose_1d` reimplements a naive accumulation loop —
  the same loop we fixed for perf in ggml#1477 / PRs #04, #17, and taught
  F16 weights in PR #14);
- supports grouped / depthwise transposed conv, which `conv_transpose_1d`
  does not;
- is one cheap kernel: one thread per output element, gathering the
  `K`-tap contributions for its position.

It's the upsample primitive for ~14 neural-vocoder / codec decoders we run
(audioseal, chatterbox-s3gen, openvoice2, tada, kugelaudio, kokoro,
pocket-tts, dia, melotts, voxcpm2, csm, vibevoice, parler, …), via the
`convt1d_decomp` helper in `src/core/conv.h`.

## Proposed op

```c
// ggml.h
GGML_OP_COL2IM_1D,

// col: [K*OC, T_in] F32 — GEMM output (mul_mat of pre-permuted weight)
// out: [T_out, OC]  F32,  T_out = (T_in-1)*s0 + K - p0
// op_params: [s0 = stride, OC, p0 = left_crop]
GGML_API struct ggml_tensor * ggml_col2im_1d(
    struct ggml_context * ctx,
    struct ggml_tensor  * col,
    int                   s0,   // stride
    int                   oc,   // output channels
    int                   p0);  // left offset / pre-crop
```

The gather (identical on every backend):

```
t_abs   = t_out + p0
t_in in [ceil((t_abs-K+1)/s0) .. floor(t_abs/s0)] ∩ [0, T_in)
k       = t_abs - t_in*s0
out[t_out, oc] = Σ col[oc*K + k, t_in]     // F32 accumulator
```

## Type policy — recommend F32-in / F32-out for upstream

Heads-up for whoever files this: our local op grew an inconsistent,
under-tested type matrix that should be collapsed before upstreaming.

| backend | input types accepted |
| --- | --- |
| CPU     | F32 only (asserts) |
| Vulkan  | F32 only (single pipeline) |
| Metal   | F32, F16 |
| CUDA    | F32, F16, BF16 |

The output is **always F32** regardless of input. The F16/BF16 *input*
variants were speculative and shipped a real bug — the Metal and CUDA
kernels templated the *destination* pointer on the source type and wrote
half/bf16 into the F32 buffer (`test_ggml_audio_ops_metal col2im_1d_f16`:
`max_abs=626` vs an exact F32 path; fixed locally 2026-06-19, dst kept
`float`). In practice every caller feeds an F32 `mul_mat` result, so the
clean upstream shape is **F32-in / F32-out only** — drop the F16/BF16 input
paths, match CPU + Vulkan, and the op becomes trivially correct and
reviewer-friendly. (If a maintainer wants low-precision columns, add it
later with the dst-stays-F32 contract and CPU/Vulkan parity in the same PR.)

## Implementation surface (already written, to be re-derived per AI policy)

- `ggml.c` — `ggml_col2im_1d` builder (computes `T_out`, sets F32 result,
  stores `{s0, oc, p0}` op_params).
- `ggml-cpu/ops.cpp` — `ggml_compute_forward_col2im_1d` (F32, threaded over
  `T_out*OC`).
- `ggml-metal/*` — `kernel_col2im_1d` (+ `-impl.h` kargs, `-ops.cpp`
  dispatch, `-device.*` pipeline + `supports_op`).
- `ggml-cuda/col2im-1d.cu` (+ `.cuh`, `ggml-cuda.cu` dispatch + `supports_op`).
- `ggml-vulkan/vulkan-shaders/col2im_1d.comp` (+ `ggml-vulkan.cpp` pipeline,
  push-constants, `supports_op`) — already F32-only; the reference shape.

## Verification plan

1. CPU forward vs a NumPy `col2im` reference for several `(K, OC, T_in,
   stride, crop)` — exact match (integer-indexed gather, F32 accumulate).
2. `test-backend-ops -o COL2IM_1D` CPU↔Metal↔CUDA↔Vulkan, F32, rmsdiff 0
   (the gather is exact; we already carry this as
   `tests/test_ggml_audio_ops_backend.cpp`).
3. Round-trip: decompose a known `conv_transpose_1d` layer both ways
   (monolithic vs `mul_mat`+`col2im_1d`) and assert bit-equality on a TTS
   decoder block.

## Code provenance

CrispASR's own op (PR #160). CPU forward, Metal/CUDA/Vulkan kernels, and the
`convt1d_decomp` callers are ours. No upstream conflict (new op + new enum
slot). Like #07, expect more review surface than a perf patch — open as
`[RFC]` for an op-shape ack first, with cross-backend forwards (or a clean
CPU-fallback via `supports_op`) in scope.

## Notes for the maintainer

- New enum slot: we insert `GGML_OP_COL2IM_1D` between `CONV_TRANSPOSE_1D`
  and `IM2COL`; final placement is the maintainer's call (enum order is ABI
  for serialized graphs).
- Naming: `col2im_1d` mirrors `im2col`. If the maintainer prefers framing it
  as the transpose-conv gradient sibling of `im2col_back`, the op shape is
  the same; happy to rename.
- Relationship to #04/#14/#17: those make the *monolithic* kernel fast and
  F16-capable. This op makes the *decomposed* path a first-class citizen.
  They're complementary; if the decomposition lands upstream, the
  monolithic-kernel patches matter less for our workloads.
