**Title:** `ggml-cpu : F16 mul_mat input saturation on ARM NEON`

---

`mul_mat` with F16 weights and F32 inputs converts `src1` (F32) to F16 first
via `__fp16` cast, saturating values >65504 to ±Inf. The Inf propagates
through the dot product and the next layer's RMSNorm produces NaN even
with an F32 accumulator. Affects any model whose intermediate activations
exceed 65504 — common in FFN `silu(gate) * up` products on wider hidden
sizes.

Repro: any F16 GGUF on `--gpu-backend cpu` with hidden ≥ ~3072 and SiLU
gating (e.g. Qwen2/Qwen3 family). Q8_0 path is unaffected because its
`vec_dot` keeps `src1` as F32 in the inner loop. Apple Metal also
unaffected.

Fix: add `ggml_vec_dot_f16_f32` (F16 weight × F32 input → F32 sum, NEON
+ AVX2/F16C + scalar) and route F16 type traits through it with
`vec_dot_type = F32`. `ggml_compute_forward_mul_mat` then sees `src1`
already matches `vec_dot_type` and skips the saturating quantize step.

Defence-in-depth: switch existing `ggml_vec_dot_f16` (still called from
e.g. `conv_transpose_1d_f16`) from `vfmaq_f16` to F32 accumulator via
`simd-mappings.h` — the F16 register accumulator can also overflow on
long F16×F16 dot products.

Patch: `01-cpu-f16-f32-dot.patch` (4 files, +94/-9).

**Verification.** Tested on M1 / M2 / M3 / M4 with Qwen3-TTS F16 GGUF;
talker emits valid logits and AR loop terminates on `codec_eos` instead
of running to the cap. Existing `test-backend-ops` cases unchanged.
