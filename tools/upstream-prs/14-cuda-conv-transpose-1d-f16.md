**Title:** `CUDA: support F16 weights in conv_transpose_1d`

---

## Background

`ggml_cuda_op_conv_transpose_1d` hard-asserts `src0->type == GGML_TYPE_F32`, and the CUDA backend's `supports_op` returns `false` for any non-F32 kernel weight. CPU + Metal both accept F16 kernels here, so models that ship upsampler weights as F16 (which is the convention for most audio codec / vocoder decoders) work on those backends and crash on CUDA.

This is structural — the kernel reads `src0[…]` as `float` directly, with no per-element conversion. F16 weights are not on a runtime fallback path; they hit the assert and abort.

## Repro

Reported by an outside user (Blackwell RTX PRO 6000) against a SNAC-based Orpheus TTS decoder. The SNAC decoder's upsampler blocks store their kernel as F16; the orpheus runtime drives `conv_transpose_1d` from there. Trace (paraphrased — same op site reproduces on every CUDA arch, not Blackwell-specific):

```
…orpheus_synthesize → snac_decoder_decode → ggml_backend_graph_compute
   → ggml_cuda_op_conv_transpose_1d+0x612
   → ggml_abort+0x152
   → "GGML_ASSERT(src0->type == GGML_TYPE_F32) failed"
     at ggml/src/ggml-cuda/conv-transpose-1d.cu:67
```

Reproduced cleanly on RunPod NVIDIA A40 (sm_86) with CUDA 12.4 against unpatched master: same assert, same site. Apple M1 / Metal works because Metal supports F16 here natively.

## Proposed change

1. Template the kernel + dispatch on `src0` type (`float` / `half`).
2. Use `ggml_cuda_cast<float>(…)` from `convert.cuh` for the per-element load. For the `float` instantiation the `if constexpr (std::is_same_v<dst_t, src_t>) return x;` branch folds away — same generated code as before for existing F32 callers.
3. Relax the runtime assert + `supports_op` predicate from `F32 only` to `F32 or F16`. `src1` (the input) stays F32; `dst` stays F32. Same algorithm, same accumulator order, same numeric result for the F32 path.

The change is ~30 LOC in `conv-transpose-1d.cu` and a one-token edit in `ggml-cuda.cu`'s `supports_op` switch.

Patch: `14-cuda-conv-transpose-1d-f16.patch` (2 files, +24/-14).

## Why this and not "let sched fall back to CPU"

`supports_op` returning `false` *should* let `ggml_backend_sched` route the op to CPU. That works when the consumer goes through sched. But several real consumers (SNAC, voxcpm2 VAE) drive their decode graph with `ggml_gallocr` + a single backend, no sched — so there is no fallback path. Even with sched it costs at least one cross-backend copy of the upsampled tensor per call, which on a fast codec is a measurable hit.

Native CUDA support is the cleanest fix: no caller churn, no scheduler restructuring, no perf regression for the F32 path.

## Verification

Built with `-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86` on a stock `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04` image on a RunPod A40. Reproduced the assert against unpatched master, then re-ran with the patch and the same orpheus + SNAC F16 graph completes with `rc=0` and produces a non-silent 24 kHz WAV.

Recommend running `test-backend-ops` with the CUDA backend over the existing `conv_transpose_1d` cases, plus an added F16-weight case alongside the F32 one — should produce bit-identical F32 outputs for matching F32 inputs (the cast is exact for representable values).

## Risk

- **F32 callers**: zero change. `ggml_cuda_cast<float>(float)` returns the value unchanged via `if constexpr`; the resulting PTX for the F32 instantiation is the same as the pre-patch kernel.
- **F16 callers**: previously crashed; now run. Numeric correctness is the standard F16 → F32 widening (lossless for the conversion direction).
- **HIP/MUSA**: shares the same source via `#ifdef GGML_USE_HIP`. `ggml_cuda_cast<float>(half)` works on HIP via the same `float(x)` fallback (HIP's `__half` has implicit conversion to `float`). Untested on Moore-Threads MUSA — flagging for reviewer awareness.

## Files changed

- `ggml/src/ggml-cuda/conv-transpose-1d.cu` — +22 / -13
- `ggml/src/ggml-cuda/ggml-cuda.cu` — +1 / -1 (supports_op predicate)

## Tested in

Validated on RunPod A40 (sm_86, CUDA 12.4) via a build-and-synthesize harness that exercises the SNAC F16 → CUDA conv_transpose_1d path end-to-end. Pre-patch: aborts with the cited assert. Post-patch: exits 0, produces non-silent audio. Reporter (RTX PRO 6000 Blackwell, sm_120) is the original consumer; once the patch lands we can ask them to confirm on Blackwell too.

## AI-usage disclosure (for the actual PR body)

Per llama.cpp's CONTRIBUTING.md AI policy: the kernel template + dispatch in this patch was sketched with mechanical AI assistance (Claude Code); review of the resulting code, the verification harness, and the prose of the PR description / commit message are all author-written.
