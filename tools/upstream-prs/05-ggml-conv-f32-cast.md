**Title:** `ggml : cast F16 conv kernel to F32 in conv_1d / conv_1d_dw / conv_2d / conv_2d_dw`

---

`ggml_compute_forward_mul_mat` requires src1 to be F32 when conversion
to vec_dot_type is needed (`GGML_ASSERT(src1->type == GGML_TYPE_F32)`).
The conv graph builders in `ggml.c` hardcode `im2col_type = F16` and
feed the kernel in directly; with F16 weights this produces
`MUL_MAT(F16, F16)` which the CPU backend rejects on any build that
sets a non-F16 `vec_dot_type` for F16 weights (e.g. paired with the
F16-saturation patch that uses F32).

Pick `im2col_type` based on whether either side is F32 (matches the
existing `ggml_conv_1d` behaviour); when im2col is F32 and the kernel
is non-F32, `ggml_cast` the kernel to F32 so the resulting MUL_MAT
has F32 src1.

Patch: `05-ggml-conv-f32-cast.patch` (1 file, +30/-6).

**Pairing.** Designed to ship with `01-cpu-f16-f32-dot.patch`. (01)
sets `vec_dot_type=F32` for F16 weights so `MUL_MAT(F16, F32)` skips
the saturating quantize; (05) makes the conv graph builders produce
F32 src1 so the CPU backend `supports_op` check still admits the
generated MUL_MATs. Without (05), kokoro F16 TTS on `--gpu-backend
cpu` aborts at `ggml_backend_sched_split_graph` trying to schedule
`MUL_MAT(F16 reshape, F16 conv1.weight)`. Recommend bundling both
patches into a single PR or, if maintainers prefer, opening a design
discussion first.

**Verification.** Tested on Apple Silicon (M1/M2/M3/M4) with kokoro
F16 GGUF, `--gpu-backend cpu`. Pre-patch: aborts. Post-patch: valid
WAV output, audio matches Metal reference. No measurable perplexity
or numeric divergence introduced by the F16 → F32 kernel cast (the
cast is exact for F16 operands).
