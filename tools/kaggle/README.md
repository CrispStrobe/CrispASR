# CrispASR — Kaggle notebook kernels

Files in this directory are the **Kaggle-side** half of CrispASR's
regression infrastructure. Pattern A from PLAN §92: a Kaggle-hosted
notebook runs the full backend matrix on a weekly schedule using
Kaggle's native UI scheduler. The script bootstraps itself from
GitHub on every run, so once the kernel is set up the only
maintenance is bumping pinned revisions in `tests/regression/manifest.json`.

## Layout

```
tools/kaggle/
├── README.md                   This file.
├── kernel-metadata.json        Kaggle manifest for the VALIDATE kernel
│                               (id = chr1str/crispasr-regression-suite,
│                                code_file = crispasr-regression.py).
├── crispasr-regression.py      THE notebook script. Clones the repo,
│                               builds CrispASR, runs the regression
│                               suite. Same file powers both kernels.
├── push.sh                     `kaggle kernels push` for the validate kernel.
└── rebake/                     Sibling kernel for AUTO re-bake.
    ├── kernel-metadata.json    id = chr1str/crispasr-auto-rebake-refs,
    │                           code_file = crispasr-rebake.py.
    ├── crispasr-rebake.py      Thin bootstrap: sets MODE=rebake +
    │                           UPLOAD=1, clones `main`, exec's the
    │                           canonical regression script from there.
    └── push.sh                 Push wrapper for the rebake kernel.
```

Two Kaggle kernels, one canonical script — the rebake kernel pulls the
latest `crispasr-regression.py` from `main` on every run, so changes to
the regression logic propagate without re-pushing the bootstrap.

## One-time setup

Required: Kaggle CLI (`pip install kaggle`, ≥ 1.8.0).

**Authentication** — two paths, the CLI auto-detects:

- *Modern (recommended)*: paste the API token from
  [your Kaggle Account page](https://www.kaggle.com/settings/account)
  → "Create New Token" → "API token" into `~/.kaggle/access_token`
  (raw token, ~38 bytes, no JSON wrapper). `kaggle config view`
  reports `auth_method: ACCESS_TOKEN` when this is active.
- *Legacy (still works)*: same page offers the older "API
  credentials" download, which gives a `kaggle.json` file with
  `{"username": ..., "key": ...}`. Drop it at `~/.kaggle/kaggle.json`.

Either way, lock down the file:

```bash
chmod 600 ~/.kaggle/access_token       # modern
chmod 600 ~/.kaggle/kaggle.json        # legacy
```

The CLI prefers the modern access token when both files exist.

1. **Push the kernel** (uploads + triggers a first run):

   ```bash
   ./tools/kaggle/push.sh
   ```

   First push creates the kernel at
   `https://www.kaggle.com/code/<your-kaggle-username>/crispasr-regression-suite`.
   The slug `<username>` comes from the `id` field in
   `kernel-metadata.json` — edit that to your username before the
   first push if you're not `chr1str`.

2. **Wait for the first run to complete cleanly** (poll the URL or
   `kaggle kernels status <id>`). The first run downloads ~1 GB
   of GGUF + ref + builds CrispASR; ~10–15 min total on Kaggle's
   CPU runner.

3. **Add the HF_TOKEN secret** (optional — only needed for
   `MODE=rebake` + `UPLOAD=1`):
   - Open the kernel page → "Add-ons" → "Secrets"
   - Add label `HF_TOKEN`, value = your HF write-scoped token from
     https://huggingface.co/settings/tokens.
   - Validate mode works anonymously against public HF repos; you
     don't need this for the default weekly schedule.

4. **Enable the schedule** (the manual step the CLI can't do —
   Kaggle hasn't exposed scheduling via API):
   - Validate kernel
     ([chr1str/crispasr-regression-suite](https://www.kaggle.com/code/chr1str/crispasr-regression-suite)):
     Settings → "Schedule a notebook run" → Weekly · Sun · 04:00 UTC.
   - Rebake kernel
     ([chr1str/crispasr-auto-rebake-refs](https://www.kaggle.com/code/chr1str/crispasr-auto-rebake-refs)):
     Settings → "Schedule a notebook run" → Monthly · 1st · 04:00 UTC.
     Less often than validate because re-baking is intentional drift
     adoption, not routine checking. The fixtures HF repo gets new
     `ref.gguf` files; the manifest pin in
     `tests/regression/manifest.json` is **NOT** auto-bumped — that
     stays a reviewable commit by a maintainer.

## How a scheduled run works

Each weekly tick, Kaggle re-runs whatever `crispasr-regression.py`
version was last pushed. The script itself does:

1. `git clone --recursive https://github.com/CrispStrobe/CrispASR.git`
   (or `git checkout main` if cached).
2. Reads `tests/regression/manifest.json`.
3. For each backend in the manifest, downloads the pinned-revision
   GGUF + pinned-revision reference dump, runs `crispasr` +
   `crispasr-diff`, compares against the manifest's thresholds.
4. Writes one JSONL per backend with the result.

So **bumping the manifest in GitHub `main` propagates to the next
Kaggle run with zero further action.** You only need to re-push
this kernel when the bootstrap script's own behaviour changes
(new env knob defaults, new pip dependencies, etc.).

## Modes

Driven by env vars on the script, settable in the kernel UI under
"Add-ons → Variables" or by editing `tools/kaggle/crispasr-regression.py`
before a push:

| Env var | Default | Effect |
|---|---|---|
| `CRISPASR_REGRESSION_MODE` | `validate` | `validate` runs the contract check against pinned references. `rebake` regenerates references from the real Python source models. |
| `CRISPASR_REGRESSION_UPLOAD` | `0` | When `rebake` mode AND `=1`, pushes the new ref.gguf files to `cstr/crispasr-regression-suite-fixtures`. Requires `HF_TOKEN` Kaggle secret with write scope. |
| `CRISPASR_REGRESSION_BACKENDS` | `""` (all) | Comma-separated subset of backend names to process. Useful for one-off ad-hoc runs. |
| `CRISPASR_REF` | `main` | Which CrispASR branch / commit / tag to test. |
| `CRISPASR_REGRESSION_BUILD` | `cpu` | `cuda` to JIT-build with `-DGGML_CUDA=ON` for GPU-path validation (requires GPU notebook). |

## Re-baking references

Reference dumps in `cstr/crispasr-regression-suite-fixtures` are the
immutable contract — pinning their revision SHA is what stops an
upstream PyTorch / NeMo / transformers bump from silently shifting
CI's expectations. When something legitimate changes upstream:

1. Run this notebook ad-hoc with `CRISPASR_REGRESSION_MODE=rebake`
   and `CRISPASR_REGRESSION_UPLOAD=1`. Set both via "Add-ons →
   Variables" in the Kaggle UI, not by editing the .py — that way
   the scheduled weekly run continues in validate mode.
2. The script prints the new fixtures-repo commit SHA at the end.
3. Paste that SHA into `tests/regression/manifest.json`'s
   `fixtures.revision`, commit + push. The next weekly validate
   run will pick it up.

## When to re-push this kernel

`./tools/kaggle/push.sh` only when the **bootstrap script itself
needs to change** — i.e. you're editing `crispasr-regression.py`.
For everything else (new backends in the manifest, new C++ commits,
new GGUFs in HF), the running notebook pulls them at runtime.
