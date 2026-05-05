# Upstream PR drafts

Drafts of four ggml fork patches we would suggest upstream.
Redacted descriptions in own voice.

| # | Subject | Code provenance |
| - | --- | --- |
| 01 | `ggml-cpu : avoid F16 saturation in MUL_MAT(F16, F32) on ARM NEON` | yours (5eef4e2 + the older conv-cast hunks) — bundles type-traits change with conv-graph kernel cast in one PR |
| 02 | `ggml-cuda : handle OW > 65535 in im2col` | yours (1552434, re-applied in ca6c523) |
| 03 | `ggml-cuda : tile cpy_scalar_transpose along grid_y` | AI-authored (2639461) — re-derive yourself before sending |
| 04 | `metal : tighten input-position loop in kernel_conv_transpose_1d` | yours (4990da8) |

The `.patch` files are clean diffs; they are reference shape, not
literal `git am` payloads — line numbers are relative to our vendored
ggml master snapshot (fetched 2026-05-05) and may drift before you
open the PR.

`MASTER-AUDIT.md` records the cross-check against `ggml-org/ggml`
master (fetched 2026-05-05): all four still apply in shape; none
have been fixed upstream. Note: `im2col` gained a second target
site (`im2col_3d_kernel`) since v0.10.0; mirror the fix onto it at
PR time.

## Sending

Send sequentially, not concurrent (new-contributor cap = 1 open PR).
Order — easiest reviewer call first:

1. **04** Metal perf — bit-identical, easy bench
2. **02** CUDA im2col — matches existing binbcast unravel pattern
3. **03** CUDA cpy — only after re-deriving the kernel-tiling code yourself
4. **01** CPU F16 — real correctness bug; one PR but five files (4 in `ggml-cpu/`, plus `ggml.c` for the conv builders that pair with the type-traits change). Design discussion expected; consider opening an issue first to get a maintainer ack on the shape before posting the PR.

Per upstream:

- Squash-merge, title format `<module> : <description>`
- Run `test-backend-ops` against the touched op on at least two backends
- Run local CI from `ci/README.md` if practical

## Workflow

```bash
gh repo fork ggml-org/ggml --clone --remote
cd ggml
git checkout -b <module>-<short>          # e.g. metal-conv-transpose-1d
# apply your re-authored hunk to the file (don't `git am` the .patch
# directly; use it as reference)
git commit -am "<module> : <description>"
git push -u origin HEAD
gh pr create --web                          # write the body in your own voice
```
