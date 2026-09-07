# Upstream PR drafts

Drafts of ggml fork patches we would suggest upstream.
Redacted descriptions in own voice.

| # | Subject | Code provenance | Status |
| - | --- | --- | --- |
| 01 | `ggml-cpu : avoid F16 saturation in MUL_MAT(F16, F32) on ARM NEON` | yours (5eef4e2 + the older conv-cast hunks) — bundles type-traits change with conv-graph kernel cast in one PR | drafted |
| 02 | `CUDA: handle OW > 65535 in im2col (2D and 3D)` | yours (1552434, re-applied in ca6c523) | ✅ upstreamed by `e79014c1` via [llama.cpp#22944](https://github.com/ggml-org/llama.cpp/pull/22944); retire patch |
| 03 | `CUDA: tile cpy_scalar_transpose along grid_y` | AI-authored (2639461) | retired: superseded by upstream CUDA cpy grid-limit fixes |
| 04 | `metal : tighten input-position loop in kernel_conv_transpose_1d` | yours (4990da8) | ✅ merged [#1477](https://github.com/ggml-org/ggml/pull/1477) 2026-05-10 |
| 05 | `ggml-cuda : per-row-contiguous unary (Phase 1 UAR)` | superseded by `d758fe69` on main (fused norm_affine + siglu removes the strided view entirely) | retired 2026-05-23, WIP branch deleted |
| 06 | `ggml-cuda : per-head mask in flash_attn_ext (MMA-F16 path)` | fork implementation, guarded by `GGML_CUDA_CRISPASR_FA_PERHEAD_MASK` | still absent upstream; re-ported to v0.23, P100 A/B is slower so keep opt-in and validate newer architectures |
| 07 | `metal : kernel_aa_snake_beta — fused AA SnakeBeta for BigVGAN v2` | drafted from upstream IndexTTS CUDA reference (Apache 2.0) — needs implementation | RFC scope only; new ggml op |
| 08 | `metal : fix cross-simdgroup reduction in kernel_norm / kernel_rms_norm / kernel_l2_norm` | yours — bisected from kokoro short-input audio regression; see [`tests/test_metal_norm_repro.cpp`](../../tests/test_metal_norm_repro.cpp) | drafted, not yet filed |
| 09 | `metal : Q8_0 × F32 bit-match mul_mat under GGML_PREC_F32` | yours (752baec) — Q8_0 counterpart to the existing Q4_K bit-match path | drafted, not yet filed |
| 10 | `metal/ggml-alloc : long F32 GPU graphs accumulate drift sensitive to in-place buffer reuse pattern` | yours — bisected through chatterbox-tts UNet; bug report, no patch | drafted, not yet filed |
| 11 | `metal/sched : mixed CPU+GPU op pinning produces NaN at large input dimensions` | yours — same UNet repro; bug report, no patch | drafted, not yet filed |
| 14 | `CUDA: support F16 weights in conv_transpose_1d` | yours (555deb98) — fixes issue #126 SNAC + orpheus CUDA segfault; templates kernel on src0 type, relaxes F32-only assert + supports_op | validated on RunPod A40 sm_86 2026-05-26, not yet filed |
| 16 | `ggml-cuda : add k-quant support to GET_ROWS (Q2_K–Q6_K)` | ours (3bf9a599); scheduler bug report shares the document | GET_ROWS upstreamed by `08130cff`; retire the slower host-sync implementation, retain scheduler issue separately |
| 17 | `CUDA: tighten input-position loop in conv_transpose_1d` | ours (f8fc8b8e), same analytical bounds | ✅ upstreamed by `af684904` / llama.cpp#25310; retire patch |
| 18 | `metal : implement cpy_tensor for shared buffers (avoid host-staging copy)` | ours — found via #161 KV-snapshot work | upstream v0.23 has the direct compatible shared-buffer copy; retire patch |
| 19 | `vulkan: flash_attn_ext coopmat2 path produces wrong output on AMD RDNA4 (RX 9700 XT)` | bug report, no patch — bisected from issue #171 VibeVoice TTS garbage; same graph is verbatim on Metal + Vulkan/MoltenVK (`FA_SCALAR`, `matrix cores: none`), garbles only on RDNA4 (`FA_COOPMAT2`). EOS classifier amplifies the attention drift into over-generation | **draft — awaiting hardware confirmation** (`GGML_VK_DISABLE_COOPMAT2=1`) from #171 reporter; then file at ggml-org/llama.cpp (vulkan) or escalate to Mesa/RADV |
| 20 | `ggml : add col2im_1d — composable building block for ConvTranspose1d` | ours (PR #160), with semantics that differed from the later upstream op | ✅ upstreamed by `d962a305`; retire outbound proposal; the adapted call site passes the v0.23 parity gate |
| 21 | `vulkan: vkResetCommandPool faults on native RADV/NVIDIA under a many-small-graph workload` | bug report, no patch — issue #215 moss-transcribe segfault. The queue-drain hypothesis (`waitIdle` before `resetCommandPool`) was **DISPROVEN** on the reporter's RADV hardware: the guard ran on a valid queue and `resetCommandPool` still faulted, so it is not "reset with pending buffers" — looks like graph-scale state corruption (TADA-#192 signature, native-only, unreproducible on MoltenVK). App-level fix is device-scoped CPU fallback for native Vulkan | **bug report — disproven patch removed; awaiting native-HW isolation before filing** at ggml-org/llama.cpp (vulkan) |
| 22 | `ggml-webgpu : OCR/CNN ops` | ours — originally NORM, IM2COL, POOL_2D, CONV_TRANSPOSE_2D, UPSCALE, ARANGE | partially upstreamed: keep only POOL_2D, CONV_TRANSPOSE_2D and ARANGE; regenerate against current llama.cpp |
| 23 | `metal : fix im2col occupancy for batch-1 inference` | ours (`89a2039d`), measured 2.3× OCR / 1.6× layout / 1.85× vocoder | still needed; re-port from monolithic shader to v0.23 `kernels/conv.metal` and repeat A/B |
| 24 | `ggml : fix ggml_conv_1d output layout for batch N > 1` | ours — the mul_mat result is `[N*OL, OC]` (OC slowest) but the final reshape declares `[OL, OC, N]` (N slowest); identical only at N=1, so every batched conv_1d consumer reads transposed data (repro: N=2 cos 0.41, N=3 cos 0.06 vs direct conv). Fix reshapes to the true `[OL, N, OC]` then permutes; N==1 keeps the zero-copy path bit-identical. Standalone repros `24-conv-1d-batch-reshape.repro.cpp` + `.dw-repro.cpp` | drafted with repros, not yet filed; ggml-org/ggml (core) |

The `.patch` files are clean diffs; they are reference shape, not
literal `git am` payloads — line numbers are relative to our v0.17-era vendored
ggml snapshot and may drift before you
open the PR.

`UPSTREAM-SYNC.md` is the INBOUND counterpart: what we should take FROM
upstream. It exists because nobody was watching that direction — `core_adaln`
failed on ARM for weeks because our vendored ggml predated two upstream SVE
fixes (`6aab1bcb`, `f69bdbb3`, both cherry-picked 2026-07-29). It also records
that upstream now has its own `col2im_1d`, which supersedes draft 20.

`MASTER-AUDIT.md` records the 2026-09-07 cross-check against
`ggml-org/ggml@e91ded11` (v0.23.0). Several patches are now upstream-owned;
do not replay the old patch set wholesale. Note: `im2col` gained a second target
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

We are **not** a first-time contributor — PR **04** is merged at
ggml-org/ggml (#1477) and **02** was merged through ggml-org/llama.cpp (#22944),
so the new-contributor "1 open PR per repo" cap does not gate us at
ggml-org/ggml. Still send sequentially per repo as a courtesy and to keep
reviewer load sane. Order for remaining work, easiest reviewer call first:

1. ✅ **04** and **02** are upstream; archive their local patches.
2. **14** CUDA conv-transpose F16 — rebase and validate on current CUDA.
3. **01** CPU F16 accumulation — validate ARM and open design discussion if needed.
4. **24** conv1d batch layout — rerun standalone repros on v0.23.
5. **22** WebGPU — split out only the three missing ops.
6. **23** Metal im2col — re-port to the modular shader and repeat the published A/B matrix.
7. **06** CUDA per-head FA — keep opt-in until a tensor-core GPU shows a win.

Retired patches **03**, **16 GET_ROWS**, **17**, **18**, and **20** must not be filed or replayed.

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
