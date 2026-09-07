# Inbound ggml sync

**Current audit: 2026-09-07.** The maintained fork was based on v0.17; upstream
is now `e91ded11` / v0.23.0. A consolidation candidate is open as
[CrispStrobe/ggml#3](https://github.com/CrispStrobe/ggml/pull/3) on
`sync/upstream-v0.23`.

The candidate takes upstream as the owner of CUDA im2col limits, CUDA cpy grid
limits, quantized GET_ROWS, CUDA conv-transpose loop bounds, Metal shared copies,
and `col2im_1d`. It keeps the fork features still absent upstream: NORM_AFFINE,
AA_SNAKE_BETA, SiGLU, the CPU F16 safeguard, CUDA per-head masks and F16
conv-transpose weights, Metal profiling/cache/correctness controls, the scheduler
reuse fix, Metal im2col occupancy, WebGPU's remaining OCR ops, and batched
conv1d/depthwise support.

## Merge discipline

A whole-tree `-X ours` resolution is unsafe. During this sync it silently
replaced upstream's SiLU-clamp enum with SiGLU, removed new IQ GET_ROWS cases,
kept a host-synchronizing GET_ROWS implementation, retained a deleted CUDA WMMA
selector, and replaced Metal's new modular library loader. Each overlap must be
resolved at the operation or dispatch-case level.

The validation gate for the fork PR is:

1. CPU x64 and arm64 builds plus the full ggml test suite.
2. CUDA compile with the per-head option both disabled and enabled.
3. Metal source and embedded-library builds, including the modular SnakeBeta
   library and pipeline archive code.
4. Vulkan/lavapipe build and backend tests.
5. CrispASR convert → quantize → reference dump → diff → parity runs for each
   carried custom op before updating the submodule pin.
6. A CUDA Q4 A/B benchmark against the current fork pin using the same audio,
   model, and warm ccache dataset.

`MASTER-AUDIT.md` lists the outbound patch status at this upstream revision.
The older `SYNC-v0.17-CONFLICTS.md` remains historical evidence, not current
instructions.
