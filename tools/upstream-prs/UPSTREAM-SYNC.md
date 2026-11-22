# Inbound ggml sync

**Current audit: 2026-11-22.** Branch `sync/upstream-2026-11` of the fork merges
upstream ggml master `d9fe7bff` (llama.cpp `5676dfbe`) and then the 118 later
ggml commits of llama.cpp up to `1014775c` (`scripts/sync-llama.last`).

Now upstream-owned, our copies dropped: the #38 F16×F32 CPU dot and F32 conv
im2col patches, CUDA per-head FA masks (option `GGML_CUDA_CRISPASR_FA_PERHEAD_MASK`
removed), WebGPU ARANGE and POOL_2D, the Vulkan MMQ A-side zero fill (#589), and
the batched conv_1d layout. Upstream pruned `ggml_mul_mat_set_prec` /
`ggml_flash_attn_ext_set_prec` (use `ggml_prec_set_acc`), ADD1 and
DIAG_MASK_INF/ZERO; CrispASR was ported. Still carried: NORM_AFFINE,
AA_SNAKE_BETA, SIGLU (now in the BF16 Vulkan GLU table), the scheduler rewire
log, lazy UE4M3/FP8 tables, the Vulkan FA mask stride, Metal hp/Q8 kernels, CANN
310P, WebGPU CONV_TRANSPOSE_2D, the conv_1d N==1 zero-copy path, and a new Metal
fusion alias guard (README row 25). `ci/crispasr-patches.txt` lists them.

Earlier audit (2026-09-07): v0.17 → v0.23 consolidation, CrispStrobe/ggml#3.

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
