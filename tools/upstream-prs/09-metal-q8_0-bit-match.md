**Title:** `metal : Q8_0 × F32 bit-match mul_mat under GGML_PREC_F32`

---

When an op carries `GGML_PREC_F32` with Q4_K weights × F32 input,
ggml-metal already has a bit-match path that mirrors CPU's
`ggml_vec_dot_q4_K_q8_K_generic` (pre-quantises the F32 input to Q8_K
via `kernel_quantize_q8_K_f32`, then runs a scalar integer-dot
`kernel_mul_mv_q4_K_q8_K`). This PR adds the equivalent path for the
Q8_0 weight × F32 input case.

Without it, `GGML_PREC_F32` on Q8_0 weights routes through
`kernel_mul_mv_ext_q8_0_f32_*`, which uses `float` accumulators and
`dot(float4, float4)`. That's already F32 in form, but the result
differs from CPU's `ggml_vec_dot_q8_0_q8_0_generic` (which quantises
input to Q8_0 then computes an integer dot then scales) by ~1e-3 per
element on Apple Silicon — enough to break apps that need bit-equiv
GPU/CPU comparison (e.g. CrispASR's chatterbox T3 K-projection
diff-bisect workflow, which is the original motivation for the Q4_K
variant).

New kernels:

1. `kernel_quantize_q8_0_f32` — F32 input column → `block_q8_0`
   blocks. One block (32 elements) per simdgroup, one element per
   thread. Mirrors `quantize_row_q8_0` (ARM NEON path):
   `d = amax/127` (F32, stored as F16 scale), `id = 1/d` (uses the
   unrounded F32 `d`, not the F16 round-trip — important for
   bit-equivalence), `qs[j] = rint(x[j] * id)` (round-to-nearest-even
   to match NEON's `vcvtnq_s32_f32`).

2. `kernel_mul_mv_q8_0_q8_0` — Q8_0 weight × Q8_0 input (pre-quantised
   above) → F32 output. Per-output-element scalar: 32-element int8 ×
   int8 → int32 sumi per block, then F32 sumf accumulation. Mirrors
   `ggml_vec_dot_q8_0_q8_0_generic` exactly.

Dispatch in `ggml_metal_op_mul_mat` slots in right after the existing
Q4_K branch, with the same scratch-buffer mechanism
(`ggml_metal_op_mul_mat_extra_q8_0` reserves trailing scratch for the
quantised input; `ggml_metal_op_mul_mat` reads it at dispatch via
`bid_tmp.offs += ggml_nbytes(op)`).

Verified bit-identical to CPU for the chatterbox UNet's 350 Q8_0 ×
F32 mul_mats (350 weights × 10 CFM steps).

Performance note: this is the slow but correct path — only fires when
the op explicitly asks for `GGML_PREC_F32`. The normal Q8_0
`kernel_mul_mm` and `kernel_mul_mv_ext` paths are unchanged.

Patch: `09-metal-q8_0-bit-match.patch` (5 files, ~290 LOC added: 1
header struct pair, 1 dispatch branch, 1 scratch-size function, 1
buffer-allocator hook, 2 kernels).

**Verification.** Run `test-backend-ops` mul_mat cases on Metal with
GGML_PREC_F32 set on Q8_0 weights × F32 input — should match CPU
output bit-for-bit. CrispASR's chatterbox-diff harness:
`s3gen_mel cos_min=0.999...` with weight-residency CPU path; with
this kernel and GPU residency, `cos_min` is dominated by other GPU
op drift (separate issue), not the mul_mat output itself.
