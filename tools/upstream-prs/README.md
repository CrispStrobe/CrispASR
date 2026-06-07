# Upstream PR drafts

Drafts of ggml fork patches we would suggest upstream.
Redacted descriptions in own voice.

| # | Subject | Code provenance | Status |
| - | --- | --- | --- |
| 01 | `ggml-cpu : avoid F16 saturation in MUL_MAT(F16, F32) on ARM NEON` | yours (5eef4e2 + the older conv-cast hunks) — bundles type-traits change with conv-graph kernel cast in one PR | drafted |
| 02 | `CUDA: handle OW > 65535 in im2col (2D and 3D)` | yours (1552434, re-applied in ca6c523) | ggml-org/ggml#1485 closed (wrong repo); refiled at [ggml-org/llama.cpp#22944](https://github.com/ggml-org/llama.cpp/pull/22944) 2026-05-11 |
| 03 | `CUDA: tile cpy_scalar_transpose along grid_y` | AI-authored (2639461) — re-derive yourself before sending | gated on llama.cpp#22944 merge |
| 04 | `metal : tighten input-position loop in kernel_conv_transpose_1d` | yours (4990da8) | ✅ merged [#1477](https://github.com/ggml-org/ggml/pull/1477) 2026-05-10 |
| 05 | `ggml-cuda : per-row-contiguous unary (Phase 1 UAR)` | superseded by `d758fe69` on main (fused norm_affine + siglu removes the strided view entirely) | retired 2026-05-23, WIP branch deleted |
| 06 | `ggml-cuda : per-head mask in flash_attn_ext (MMA-F16 path)` | design + kernel-level patch sketch in `06-cuda-fa-perhead-mask.md` (~45 LOC across `fattn.cu` + `fattn-mma-f16.cuh` + test-backend-ops) | drafted on main 2026-05-23, not yet implemented |
| 07 | `metal : kernel_aa_snake_beta — fused AA SnakeBeta for BigVGAN v2` | drafted from upstream IndexTTS CUDA reference (Apache 2.0) — needs implementation | RFC scope only; new ggml op |
| 08 | `metal : fix cross-simdgroup reduction in kernel_norm / kernel_rms_norm / kernel_l2_norm` | yours — bisected from kokoro short-input audio regression; see [`tests/test_metal_norm_repro.cpp`](../../tests/test_metal_norm_repro.cpp) | drafted, not yet filed |
| 09 | `metal : Q8_0 × F32 bit-match mul_mat under GGML_PREC_F32` | yours (752baec) — Q8_0 counterpart to the existing Q4_K bit-match path | drafted, not yet filed |
| 10 | `metal/ggml-alloc : long F32 GPU graphs accumulate drift sensitive to in-place buffer reuse pattern` | yours — bisected through chatterbox-tts UNet; bug report, no patch | drafted, not yet filed |
| 11 | `metal/sched : mixed CPU+GPU op pinning produces NaN at large input dimensions` | yours — same UNet repro; bug report, no patch | drafted, not yet filed |
| 14 | `CUDA: support F16 weights in conv_transpose_1d` | yours (555deb98) — fixes issue #126 SNAC + orpheus CUDA segfault; templates kernel on src0 type, relaxes F32-only assert + supports_op | validated on RunPod A40 sm_86 2026-05-26, not yet filed |
| 16 | `ggml-cuda : add k-quant support to GET_ROWS (Q2_K–Q6_K)` | ours (3bf9a599) — uses `ggml_get_to_fp32_cuda()` row-dequant, copies indices to host, sequential kernel launch per row; also adds Q2_K–Q6_K to `supports_op` | ✅ validated on RunPod RTX 3090 sm_86 2026-06-05 (mimo-asr Q4_K embed, 2.0× RT, correct JFK); `16-sched-small-graph-cross-backend.md` also documents the scheduler cross-backend routing bug for small graphs |
| 17 | `CUDA: tighten input-position loop in conv_transpose_1d` | ours (f8fc8b8e) — CUDA analog of merged Metal PR #04; i_min/i_max analytical bounds eliminate O(IL) naive loop (was ~200× wasted iterations at TTS codec scale, causing TDR on AMD RX 7900 XTX + NVIDIA RTX 5060 Ti; see #155); bit-identical output | validated locally; CUDA hardware test pending; gated on PR #14 (applies on top of F16 template) — not yet filed |

The `.patch` files are clean diffs; they are reference shape, not
literal `git am` payloads — line numbers are relative to our vendored
ggml master snapshot (fetched 2026-05-05) and may drift before you
open the PR.

`MASTER-AUDIT.md` records the cross-check against `ggml-org/ggml`
master (fetched 2026-05-05): all four still apply in shape; none
have been fixed upstream. Note: `im2col` gained a second target
site (`im2col_3d_kernel`) since v0.10.0; the PR 02 patch covers
both kernels.

## Which repo to file against

CUDA changes belong in **ggml-org/llama.cpp**, not ggml-org/ggml. ggml's
own README says "some of the development is currently happening in the
llama.cpp and whisper.cpp repos," and the `src/ggml-cuda/` commit log
on ggml master is 100% `(llama/NNNNN)` sync commits — i.e., CUDA work
flows llama.cpp → ggml, never the other way. Reviewers (slaren,
JohannesGaessler, …) live in llama.cpp's queue. Confirmed 2026-05-11
when @CISC closed-with-comment on #02 here and redirected us.

| Patch touches | File against | Why |
| --- | --- | --- |
| `src/ggml-cuda/**` | **ggml-org/llama.cpp** | CUDA reviewers concentrate there; auto-syncs back |
| `src/ggml-metal/**` | ggml-org/ggml (direct) or llama.cpp | Both work; PR #1477 landed direct in ggml |
| `src/ggml-vulkan/**` | **ggml-org/llama.cpp** | same pattern as CUDA per commit log |
| `src/ggml-cpu/**`, `src/ggml.c`, type-traits | ggml-org/ggml | core lib stays in ggml |
| Standalone ggml examples / build | ggml-org/ggml | repo-local |

Title convention differs by repo:
- ggml-org/ggml: `<module> : <description>` (space around colon)
- ggml-org/llama.cpp: `CUDA: …` / `vulkan: …` (no space) — see recent master commits

llama.cpp also has a much stricter AI-content policy (AGENTS.md +
CONTRIBUTING.md): prohibits AI-written PR descriptions / commit
messages / reviewer responses, requires disclosure when AI
meaningfully contributed, threatens account bans for repeated
violations. Author your own prose; disclose mechanical AI assistance
explicitly in the Requirements section.

## Sending

Send sequentially, not concurrent (new-contributor cap = 1 open PR per
repo). Order — easiest reviewer call first:

1. ✅ **04** Metal perf → merged at ggml-org/ggml as [#1477](https://github.com/ggml-org/ggml/pull/1477) 2026-05-10
2. 📤 **02** CUDA im2col → filed at ggml-org/llama.cpp as [#22944](https://github.com/ggml-org/llama.cpp/pull/22944) 2026-05-11 (after ggml#1485 was redirected)
3. **03** CUDA cpy → file at ggml-org/llama.cpp after #22944 merges; re-derive the kernel-tiling code yourself first
4. **14** CUDA conv_transpose_1d F16 → file at ggml-org/llama.cpp; small (2 files, ~25 LOC), pure-perf no-regression for existing F32 callers — should be an easy reviewer call
5. **17** CUDA conv_transpose_1d loop → file at ggml-org/llama.cpp after #14 merges; 1 file, +14/−22 on top of #14; CUDA analog of ggml#1477 — reviewer will recognise the pattern
6. **01** CPU F16 → file at ggml-org/ggml (touches ggml-cpu/ + ggml.c); design-discussion expected, consider opening an issue first

Per upstream:

- Squash-merge, title format per repo convention (see above)
- Run `test-backend-ops` against the touched op on at least two backends
- Run local CI from `ci/README.md` if practical

## Workflow

```bash
# pick the right repo per the table above; example for CUDA:
gh repo fork ggml-org/llama.cpp --clone --remote
cd llama.cpp
git checkout -b cuda-<short>              # e.g. cuda-im2col-ow
# apply your re-authored hunk to the file (don't `git am` the .patch
# directly; use it as reference)
git commit -m "CUDA: <description>"       # author your own message
git push -u origin HEAD
gh pr create --web                        # author your own body; fill
                                          # the AI-usage disclosure
                                          # in Requirements honestly
```
