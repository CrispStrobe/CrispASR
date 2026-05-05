# Upstream master audit (against ggml-org/ggml master, fetched 2026-05-05)

Cross-checked our four upstream-PR drafts against the current upstream
`ggml-org/ggml` master to surface conflicts and any "already fixed
upstream" cases. None are already fixed; all still apply in shape.
Two notes worth keeping in mind:
- `im2col` gained a second target site (`im2col_3d_kernel`) since
  v0.10.0; both need the same fix.
- The original audit grep only matched `// CrispASR patch` and missed
  the conv-graph kernel-cast hunks in `ggml.c` (marked
  `// CrispASR fork`). They're now bundled into PR 01 — without them,
  PR 01's type-traits change crashes kokoro F16 CPU at
  `ggml_backend_sched_split_graph`.

| # | Patch | Master state | Action needed at PR time |
| - | --- | --- | --- |
| 01 | F16 mul_mat saturation (CPU type traits + conv graph builders, bundled) | Vulnerable. `ggml-cpu.c:218` still has `vec_dot=ggml_vec_dot_f16, vec_dot_type=GGML_TYPE_F16`. `simd-mappings.h:250,365` still gate on `__ARM_FEATURE_FP16_VECTOR_ARITHMETIC` with `vfmaq_f16`. `ggml.c:4471-4521` (`ggml_conv_1d`, `ggml_conv_1d_dw`) and `ggml.c:4575-4750` (`ggml_conv_2d`, `ggml_conv_2d_dw`) hardcode `im2col_type=GGML_TYPE_F16` with no kernel cast. | Apply as drafted (5 files in one PR). Consider opening a design discussion first. |
| 02 | im2col grid_y > 65535 | Vulnerable, **two sites**. `im2col.cu:54` (existing 2D kernel — our patch). New since v0.10.0: `im2col_3d_kernel` at `im2col.cu:118` with `dim3 block_nums(num_blocks, OW, …)` at line 181 and unbounded `iow = blockIdx.y` at line 139. | Apply existing 2D fix **plus** mirror the same fix on `im2col_3d_kernel` and its dispatch. Send as one PR (same op, same bug class). |
| 03 | cpy_scalar_transpose grid_y | Vulnerable. `cpy.cu:222` still has `GGML_ASSERT(grid_y < USHRT_MAX)`. | Apply as drafted (re-derive code first per AI policy). |
| 04 | Metal conv_transpose_1d | Vulnerable / inefficient. `ggml-metal.metal:4860-4861` still iterates full IL with the in-loop `if`. | Apply as drafted. |

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
grep -n "GGML_ASSERT(grid_y < USHRT_MAX)" /tmp/ggml-master/src/ggml-cuda/cpy.cu
grep -n "block_nums(num_blocks, OW, "   /tmp/ggml-master/src/ggml-cuda/im2col.cu
grep -n "tgpig\[0\] >= i \* args.s0"     /tmp/ggml-master/src/ggml-metal/ggml-metal.metal
grep -n "vec_dot_type *= *GGML_TYPE_F16" /tmp/ggml-master/src/ggml-cpu/ggml-cpu.c
grep -n "ggml_im2col(.*GGML_TYPE_F16)"   /tmp/ggml-master/src/ggml.c
```

If any returns no matches, that fix landed upstream and the PR is
unnecessary.
