# Upstream master audit (ggml v0.23.0, 2026-09-07)

Audited against `ggml-org/ggml@e91ded11` (v0.23.0). The prior May audit is
obsolete: several fork patches have since landed upstream, and Metal moved from
one monolithic shader to one library per operation.

| # | Patch | v0.23 state | Action |
| - | --- | --- | --- |
| 01 | CPU F16 accumulation saturation | Still absent. The fork's F32 accumulation safeguard remains relevant, especially on ARM FP16-vector builds. | Rebase and validate on ARM before filing. |
| 02 | CUDA im2col grid-Y limit | **Upstreamed** by `e79014c1` (`llama.cpp#22944`). | Retire local outbound patch. |
| 03 | CUDA cpy grid limits | **Superseded.** Current `cpy.cu` contains upstream grid-limit fixes (`00504b04`, `b64fb805` and follow-ups). | Retire; do not replay the old patch. |
| 04 | Metal conv-transpose loop bounds | **Merged** as ggml#1477. | Use upstream copy. |
| 06 | CUDA per-head FA mask | Still absent: current selection rejects `mask->ne[2] != 1`. | Rebase on the current MMA selector and validate supported architectures. P100 measurement is slower, so keep opt-in. |
| 07 | AA SnakeBeta op and Metal kernel | Still fork-only. | Keep carried patch; its shader must be a modular `aa_snake_beta` library on v0.23. |
| 10 | scheduler graph mutation on reused allocations | Still present: split construction rewrites `node->src[j]` to copy tensors. | Keep carried alloc-once/compute-many fix and tests. |
| 13 | no-AVX512 build knob | Still applies cleanly. | Rebase/validate before filing. |
| 14 | CUDA conv-transpose F16 weights | Still absent; current CUDA implementation asserts F32 weights. | Rebase and file independently. |
| 16 | CUDA k-quant GET_ROWS | **Upstreamed** by `08130cff`. Upstream dispatch avoids the fork patch's host index copy, sequential launches, and CUDA-graph disable. | Retire fork implementation and use upstream. The scheduler report in the same document remains a separate concern. |
| 17 | CUDA conv-transpose loop bounds | **Upstreamed** by `af684904` (`llama.cpp#25310`). | Retire local outbound patch. |
| 18 | Metal shared-buffer tensor copy | **Upstreamed/superseded.** v0.23 directly copies compatible shared buffers. | Retire local outbound patch. |
| 20 | `col2im_1d` | **Upstreamed** by `d962a305`. | Retire the outbound op proposal; adapt CrispASR call sites only after parity tests because old fork semantics differed. |
| 22 | WebGPU OCR ops | **Partly upstreamed:** NORM, IM2COL and UPSCALE exist; POOL_2D, CONV_TRANSPOSE_2D and ARANGE remain missing. | Split and regenerate a smaller patch against current llama.cpp WebGPU. |
| 23 | Metal im2col occupancy | Still absent, but the monolithic `.metal` patch no longer applies after modularization. | Re-port to `kernels/conv.metal` and current dispatch, then repeat Metal A/B measurements. |
| 24 | conv1d batch layout/depthwise support | Still absent. | Rebase on v0.23 and run the standalone batch repros. |

## Inbound conclusions

The fork can no longer be maintained as a replay of every v0.17 patch. The
candidate branch `CrispStrobe/ggml:sync/upstream-v0.23` merges v0.23, removes the
five upstream-owned implementations above, and ports the remaining custom ops
and diagnostics to the new backend layouts. Its draft review is
CrispStrobe/ggml#3.

Before filing any remaining outbound change, derive a fresh diff from v0.23 (or
current llama.cpp for CUDA/WebGPU), run the backend-specific tests, and use the
old `.patch` only as a behavioral reference.
