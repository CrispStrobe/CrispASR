**Title:** `ggml-cpu : avoid F16 saturation in MUL_MAT(F16, F32) on ARM NEON`

---

`mul_mat` with F16 weights and F32 inputs converts `src1` (F32) to F16
first via `__fp16` cast, saturating values >65504 to ±Inf. The Inf
propagates through the dot product and the next layer's RMSNorm
produces NaN even with an F32 accumulator. Affects any model whose
intermediate activations exceed 65504 — common in FFN
`silu(gate) * up` products for hidden sizes ≥ ~3072. Q8_0 vec_dot is
unaffected (its inner loop keeps src1 as F32); Apple Metal also
unaffected.

Two coupled changes are required. They land in one PR because the
first alone breaks the second's existing call sites:

**(a) Type traits.** Add `ggml_vec_dot_f16_f32` (F16 weight × F32 input
→ F32 sum, NEON + AVX2/F16C + scalar) and route F16 type traits
through it with `vec_dot_type = F32` so MUL_MAT skips the saturating
quantize. Defence-in-depth: switch existing `ggml_vec_dot_f16` (still
called from e.g. `conv_transpose_1d_f16`) from `vfmaq_f16` to F32
accumulator via `simd-mappings.h`. Touches
`ggml-cpu/{vec.cpp, vec.h, ggml-cpu.c, simd-mappings.h}`.

**(b) Conv graph builders.** With (a) setting `vec_dot_type = F32`
for F16 weights, `ggml_compute_forward_mul_mat`'s
`GGML_ASSERT(src1->type == GGML_TYPE_F32)` rejects any non-F32 src1
that needs conversion. Upstream's `ggml_conv_1d`, `ggml_conv_1d_dw`,
`ggml_conv_2d`, `ggml_conv_2d_dw` hardcode `im2col_type = F16` and
feed the kernel weight in directly, producing `MUL_MAT(F16, F16)` —
which (a) makes unsupported on the CPU backend.

Pick `im2col_type` based on whether either side is F32; when im2col
is F32 and the kernel is non-F32, `ggml_cast` the kernel to F32 so
the resulting MUL_MAT has F32 src1. Touches `ggml.c`. Bandwidth cost
of (b) is real (extra F16 → F32 cast per inference pass) but
unavoidable as long as upstream `mul_mat` rejects non-F32 src1 in the
type-conversion path.

Patch: `01-cpu-f16-saturation.patch` (5 files, +80/-16).

**Verification.** Tested on M1 with downstream consumer (crispasr with kokoro F16 GGUF on
`--gpu-backend cpu` and Qwen3-TTS F16 talker on `--gpu-backend cpu`);
talker emits valid logits and AR loop terminates on `codec_eos`
instead of running to the cap. Without (b), kokoro F16 CPU TTS
aborts at `ggml_backend_sched_split_graph` trying to schedule
`MUL_MAT(F16 reshape, F16 conv1.weight)` from the F0N predictor.
Existing `test-backend-ops` cases unchanged.
