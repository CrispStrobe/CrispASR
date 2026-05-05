# Upstream PR drafts

Drafts of five ggml fork patches we would suggest upstream.
Redacted descriptions in own voice.

| # | Subject | Code provenance |
| - | --- | --- |
| 01 | `ggml-cpu : F16 mul_mat input saturation on ARM NEON` | yours (5eef4e2) — **must ship together with 05** |
| 02 | `ggml-cuda : handle OW > 65535 in im2col` | yours (1552434, re-applied in ca6c523) |
| 03 | `ggml-cuda : tile cpy_scalar_transpose along grid_y` | AI-authored (2639461) — re-derive yourself before sending |
| 04 | `metal : tighten input-position loop in kernel_conv_transpose_1d` | yours (4990da8) |
| 05 | `ggml : cast F16 conv kernel to F32 in conv_1d / conv_1d_dw / conv_2d / conv_2d_dw` | yours (predates 0.10.0 bump) — **paired with 01** |

The `.patch` files are clean diffs;
they are reference shape, not literal `git am` payloads — line numbers
are relative to our vendored ggml v0.10.0 and will need rebasing onto
upstream master at PR time.

`MASTER-AUDIT.md` records the cross-check against `ggml-org/ggml`
master (fetched 2026-05-05): all four patches still apply in shape;
none have been fixed upstream. Note: `im2col` gained a second target
site (`im2col_3d_kernel`) since v0.10.0; the same fix needs to be
mirrored onto it at PR time.

## Sending

Send sequentially, not concurrent (new-contributor cap = 1 open PR).
Order — easiest reviewer call first:

1. **04** Metal perf — bit-identical, easy bench
2. **02** CUDA im2col — matches existing binbcast unravel pattern
3. **03** CUDA cpy — only after re-deriving the kernel-tiling code yourself
4. **01 + 05 together** CPU F16 — real correctness bug; bundles type-traits change with conv-graph cast. Send as a single PR or invite a maintainer design call first; design discussion expected.

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
