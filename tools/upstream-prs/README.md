# Upstream PR drafts

Drafts of four ggml fork patches we want upstream. **Re-author the
descriptions in your own voice before sending** — llama.cpp's
contribution policy (which ggml-org/ggml inherits) prohibits
AI-written PR posts.

| # | Subject | Status |
| - | --- | --- |
| 01 | `ggml-cpu : F16 mul_mat input saturation on ARM NEON` | Code: yours (5eef4e2). Description: AI-drafted, redact. |
| 02 | `ggml-cuda : handle OW > 65535 in im2col` | Code: yours (1552434 / ca6c523). Description: AI-drafted, redact. |
| 03 | `ggml-cuda : tile cpy_scalar_transpose along grid_y` | Code: AI-authored (2639461). **Re-derive yourself before upstream.** |
| 04 | `metal : tighten input-position loop in kernel_conv_transpose_1d` | Code: yours (4990da8). Description: AI-drafted, redact. |

## Sending

Send sequentially, not concurrent (new-contributor cap = 1 open PR).
Suggested order — easiest reviewer call first:

1. **04** (Metal perf, bit-identical, easy bench)
2. **02** (CUDA im2col, matches existing binbcast unravel pattern)
3. **01** (CPU F16, real correctness bug but design-discussion risk)
4. **03** (CUDA cpy) — only after re-deriving the kernel-tiling code
   yourself

Per upstream:

- Squash-merge, title format `<module> : <description>`
- Strip the `// CrispASR patch ... MUST RE-APPLY` comment blocks from
  each diff; upstream doesn't need that notice
- Run `test-backend-ops` against the touched op on at least two
  backends before opening
- Run local CI from `ci/README.md` if practical

## Workflow

```bash
gh repo fork ggml-org/ggml --clone --remote
cd ggml
git checkout -b <module>-<short>          # e.g. metal-conv-transpose-1d
git apply ../CrispASR/tools/upstream-prs/NN-*.patch
# strip the // CrispASR patch comments, write your own commit message
git commit -am "<module> : <description>"
git push -u origin HEAD
gh pr create --web                          # write the body in your own voice
```

## Notes

- `01-cpu-f16-f32-dot.patch` is the full mailbox-format export of
  commit `5eef4e2` (your own commit message — keep or rewrite).
- `02-cuda-im2col.patch` exports `ca6c523` (re-application commit;
  the original was `1552434`).
- `03-cuda-cpy.patch` exports `2639461` (this Claude session). The
  comment block in the diff is verbose; trim before sending.
- `04-metal-conv-transpose-1d.patch` is a `git show` slice of
  `4990da8` (the metal hunk only — that commit also touched
  `tools/`, which you don't want to send upstream).
