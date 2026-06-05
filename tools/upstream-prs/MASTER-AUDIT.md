# Upstream master audit (against ggml-org/ggml master, fetched 2026-05-05)

Cross-checked our four upstream-PR drafts against the current upstream
`ggml-org/ggml` master to surface conflicts and any "already fixed
upstream" cases. None are already fixed; all still apply in shape.
Two notes worth keeping in mind:
- `im2col` gained a second target site (`im2col_3d_kernel`) since
  v0.10.0. The PR 02 patch covers both kernels.
- The original audit grep only matched `// CrispASR patch` and missed
  the conv-graph kernel-cast hunks in `ggml.c` (marked
  `// CrispASR fork`). They're now bundled into PR 01 — without them,
  PR 01's type-traits change crashes kokoro F16 CPU at
  `ggml_backend_sched_split_graph`.

| # | Patch | Master state | Action needed at PR time |
| - | --- | --- | --- |
| 01 | F16 mul_mat saturation (CPU type traits + conv graph builders, bundled) | Vulnerable. `ggml-cpu.c:218` still has `vec_dot=ggml_vec_dot_f16, vec_dot_type=GGML_TYPE_F16`. `simd-mappings.h:250,365` still gate on `__ARM_FEATURE_FP16_VECTOR_ARITHMETIC` with `vfmaq_f16`. `ggml.c:4471-4521` (`ggml_conv_1d`, `ggml_conv_1d_dw`) and `ggml.c:4575-4750` (`ggml_conv_2d`, `ggml_conv_2d_dw`) hardcode `im2col_type=GGML_TYPE_F16` with no kernel cast. | Apply as drafted (5 files in one PR). Consider opening a design discussion first. |
| 02 | im2col grid_y > 65535 | Vulnerable, **two sites**. `im2col.cu:54` (existing 2D kernel). New since v0.10.0: `im2col_3d_kernel` at `im2col.cu:118` with `dim3 block_nums(num_blocks, OW, …)` at line 181 and unbounded `iow = blockIdx.y` at line 139. | Apply as drafted — patch covers both kernels. |
| 03 | cpy_scalar_transpose grid_y | Vulnerable. `cpy.cu:222` still has `GGML_ASSERT(grid_y < USHRT_MAX)`. | Apply as drafted (re-derive code first per AI policy). |
| 04 | Metal conv_transpose_1d | Vulnerable / inefficient. `ggml-metal.metal:4860-4861` still iterates full IL with the in-loop `if`. | Apply as drafted. |
| 05 | CUDA per-row-contiguous unary | ~~Vulnerable~~ — solved differently. `d758fe69` (fused `GGML_OP_NORM_AFFINE` + `GGML_GLU_OP_SIGLU`) replaced the strided view that caused the CPU fallback, so the UNARY contiguity gate no longer matters for our workload. | **Retired** 2026-05-23. WIP branch deleted. No upstream PR needed. |
| 06 | CUDA per-head mask in `flash_attn_ext` (MMA-F16 path) | Vulnerable. `fattn.cu:423` still has `if (mask && mask->ne[2] != 1) return BEST_FATTN_KERNEL_NONE;`. `fattn-mma-f16.cuh:1635,1681` advance `mask_h` only by sequence (`nb33 * (sequence % ne33)`), not by head — the kernel already takes `nb32` as a parameter but doesn't use it. | Design + ~45 LOC patch sketched in `06-cuda-fa-perhead-mask.md`. Implementation + `test-backend-ops` validation on sm_75/86/89 pending. |
| 07 | Metal `kernel_aa_snake_beta` (NEW OP) | N/A — adds a new op (`GGML_OP_AA_SNAKE_BETA`). No upstream conflict but expects design ack first. Integrated locally on `main` as `d9ecc9b9` (phase 1 / CPU forward) + `87d0e38f` (phase 2 / Metal kernel). | RFC scope only; do not file before 01/05/06. Other GPU backends fall through to `default: return false` in their `supports_op` switches — sched routes to CPU forward automatically. |
| 14 | CUDA `conv_transpose_1d` F16 weights | Vulnerable. `conv-transpose-1d.cu:67` (`GGML_ASSERT(src0->type == GGML_TYPE_F32)`) is byte-identical to upstream master `ggml-org/ggml@master` and `ggml-org/llama.cpp@master` (verified 2026-05-26 via raw.githubusercontent.com fetch). `ggml-cuda.cu` supports_op switch's `(src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_F32)` predicate also unchanged. No prior or in-flight PR found via `gh api search/issues`. | Apply as drafted. File at ggml-org/llama.cpp per repo routing. |

## What changed in master since v0.10.0 (relevant to our patches)

- `ggml-cuda/im2col.cu` grew an `im2col_3d_kernel` + dispatch (3D conv
  support). Same bug class as the 2D version.
- `ggml-cuda/cpy.cu` line numbers shifted (~6 lines) but the structure
  of `cpy_scalar_transpose` and `ggml_cpy_scalar_cuda` is unchanged.
- `ggml-cpu/{vec.cpp, ggml-cpu.c, simd-mappings.h}` line numbers
  shifted but the F16 vec_dot type traits and the NEON `#if`
  gating pattern are unchanged.
- `ggml-metal.metal` line numbers shifted; `kernel_conv_transpose_1d`
  body is byte-identical to v0.10.0.
- `ggml-cuda/conv-transpose-1d.cu` is byte-identical to upstream
  master as of 2026-05-26 (verified for PR #14 audit) — F32-only
  assert + dispatch unchanged across all CUDA arches.

## Re-verify before PR

The audit is a snapshot. Master moves; before opening any PR:

```bash
mkdir -p /tmp/ggml-master
for f in src/ggml-cuda/im2col.cu src/ggml-cuda/cpy.cu \
         src/ggml-metal/ggml-metal.metal \
         src/ggml-cpu/vec.cpp src/ggml-cpu/vec.h \
         src/ggml-cpu/ggml-cpu.c src/ggml-cpu/simd-mappings.h \
         src/ggml.c; do
  mkdir -p /tmp/ggml-master/$(dirname $f)
  curl -sL "https://raw.githubusercontent.com/ggml-org/ggml/master/$f" \
       -o /tmp/ggml-master/$f
done
```

Then re-grep for the patterns the patches replace:
```bash
grep -n "GGML_ASSERT(grid_y < USHRT_MAX)"            /tmp/ggml-master/src/ggml-cuda/cpy.cu
grep -n "block_nums(num_blocks, OW, "                /tmp/ggml-master/src/ggml-cuda/im2col.cu
grep -n "tgpig\[0\] >= i \* args.s0"                  /tmp/ggml-master/src/ggml-metal/ggml-metal.metal
grep -n "vec_dot_type *= *GGML_TYPE_F16"             /tmp/ggml-master/src/ggml-cpu/ggml-cpu.c
grep -n "ggml_im2col(.*GGML_TYPE_F16)"               /tmp/ggml-master/src/ggml.c
grep -n "GGML_ASSERT(src0->type == GGML_TYPE_F32)"   /tmp/ggml-master/src/ggml-cuda/conv-transpose-1d.cu
```

If any returns no matches, that fix landed upstream and the PR is
unnecessary.
