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
│                               (id = ${KAGGLE_ACCOUNT}/crispasr-regression-suite,
│                                code_file = crispasr-regression.py).
├── crispasr-regression.py      THE notebook script. Clones the repo,
│                               builds CrispASR, runs the regression
│                               suite. Same file powers both kernels.
├── push.sh                     `kaggle kernels push` for the validate kernel.
└── rebake/                     Sibling kernel for AUTO re-bake.
    ├── kernel-metadata.json    id = ${KAGGLE_ACCOUNT}/crispasr-auto-rebake-refs,
    │                           code_file = crispasr-rebake.py.
    ├── crispasr-rebake.py      Thin bootstrap: sets MODE=rebake +
    │                           UPLOAD=1, clones `main`, exec's the
    │                           canonical regression script from there.
    └── push.sh                 Push wrapper for the rebake kernel.
```

Two Kaggle kernels, one canonical script — the rebake kernel pulls the
latest `crispasr-regression.py` from `main` on every run, so changes to
the regression logic propagate without re-pushing the bootstrap.

## What belongs on Kaggle (and what doesn't)

Kaggle is for work that needs a **real GPU**: CUDA or Vulkan speed and parity on
hardware, and PyTorch references that only run in reasonable time on CUDA.
Everything that runs on CPU goes to GitHub Actions instead, through
`.github/workflows/heavy-cpu.yml` and `tools/ci-heavy/` (see its README). That
covers Python reference dumps, diff harnesses, ASR roundtrips, and
download → convert → quantize → upload. Standard runners are free for this
public repo: 4 vCPU, 16 GB RAM, 6 h per job.

Rules that follow from Kaggle's terms and guidelines (an account was banned on
2026-09-28 for "resource abuse"):

- One account per person. Never switch accounts to get around a quota or the
  2-session GPU cap; wait instead.
- Don't request a GPU session for its RAM, disk or internet, or to build C++
  without using the GPU afterwards.
- Attach only datasets the notebook really uses. No build caches or storage
  datasets (the ccache datasets now live on HF).
- No re-push loops. Push, wait, read `kernels_logs`.

## Account, token and pushing

Nothing account-specific is committed. Kernel metadata (`id`, `dataset_sources`, …)
and kernel code use the placeholder `${KAGGLE_ACCOUNT}`; `tools/kaggle/kpush.py`
renders it from the environment and pushes with the token from the environment:

```bash
# CI: KAGGLE_ACCOUNT / KAGGLE_TOKEN are repository secrets (see kaggle-status.yml).
# Locally: export them from a private env file - never type a token on a command line.
python3 tools/kaggle/kpush.py tools/kaggle/<kernel-dir>
```

`tests/test_no_secrets.py` (Secret scan workflow, every push) fails if a token, a
literal owner in kernel metadata, or the account name itself lands in the repo.

Kaggle's Terms allow ONE account per person and its Community Guidelines ban
"abuse [of] kernel resources such as free storage" and attaching unrelated
datasets: never switch accounts to get around quota or session limits, and do not
keep build caches or tooling as Kaggle datasets (the build caches now live in private
Hugging Face dataset repos).

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
   first push if you're not `$KAGGLE_ACCOUNT`.

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
     ([${KAGGLE_ACCOUNT}/crispasr-regression-suite](https://www.kaggle.com/code/${KAGGLE_ACCOUNT}/crispasr-regression-suite)):
     Settings → "Schedule a notebook run" → Weekly · Sun · 04:00 UTC.
   - Rebake kernel
     ([${KAGGLE_ACCOUNT}/crispasr-auto-rebake-refs](https://www.kaggle.com/code/${KAGGLE_ACCOUNT}/crispasr-auto-rebake-refs)):
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

## Recovering a large output artifact

`recover_output.py` avoids the Kaggle SDK output downloader's `response.content`
buffer. It lists output metadata through the authenticated SDK, selects one exact
`file_name` across all pages, and streams only that artifact in 1 MiB chunks.
Install `kaggle` (including `kagglesdk`) and `requests` in your own environment.
Export `KAGGLE_ACCOUNT` (or `KAGGLE_USERNAME`) and credentials privately.
`KAGGLE_TOKEN` maps to `KAGGLE_API_TOKEN` when the latter is unset; native Kaggle
SDK authentication configuration also works. Account names and signed URLs are
not printed.

```bash
python tools/kaggle/recover_output.py kernel-slug \
  --artifact results/reference.tar \
  --destination "$ARTIFACT_DIR/reference.tar" \
  --expected-bytes "$EXPECTED_BYTES" --sha256 "$EXPECTED_SHA256"
```

Supply the byte count and SHA-256 from a trusted producer receipt, not from the
failed placeholder or the download response. Only the latest session is supported;
there is no version-selection option. The SDK exposes a `version_label` field,
but setting it to a completed kernel's numeric version returned HTTP 404 with
Kaggle 2.2.2; the official output downloader also leaves this field unset. Keep
the expected producer receipt when a kernel is rerun: if the current session
serves different bytes, verification fails and preserves the destination. Use
the full output name including directories, not a basename or regular expression.
The destination parent must already exist on storage with sufficient free space
for a complete temporary copy. A unique sibling `.part` file is flushed and
verified before atomic replacement, including when replacing a zero-byte
placeholder. Errors and interruptions discard the temporary copy and preserve
the destination. A hard process kill can leave an untrusted `.part` file; remove
it before retrying. No extraction or resume is performed. HTTP responses must
be 200 with identity encoding; redirects, partial content and inconsistent
Content-Length are rejected. Filesystems must support atomic same-directory
replacement.

Model-free mocked controls (no network, credentials or weights):

```bash
python -m unittest discover -s tests -p test_kaggle_output_recovery.py -v
```

## One deferred RE-USE GPU diagnostic

`launch_reuse_followup.py` is a task-specific one-shot launcher, not a recurring
scheduler. The frozen candidate is `8c2503d21e2f983254a75e85dbc1e8d0e0e79da0`,
with script SHA-256
`5264912f1d2851b123849c4451e2b62d9029c9f98eef9de8b152cbb81e81452d`.
It reads script/metadata from immutable Git objects, checks their hashes and
runtime source pin, and verifies the cached window-2 reference, native log and
manifest in bounded-memory local reads. It never downloads the large archive.
The script and metadata are staged privately for exactly one submission.

Execution requires `--execute-once`, `--repo`, `--evidence-dir` and `--receipt`.
It refuses outside **2026-11-14 00:00 through 2026-11-15 00:00 UTC**, regardless
of currently available quota. The authenticated username must equal the one
private configured account. Structured SDK quota durations use `.total_seconds()`:
SDK representations can drop `timedelta` days (`1 day + 21600 s` is 30 hours,
not 6). The reviewed SDK datetime serializer strips the API's trailing `Z`,
so its naive quota refresh datetime is explicitly restored to UTC in the SDK
adapter; the quota guard itself rejects unqualified naive timestamps. Allowed
minus used minus reserved must be at least 4500 seconds, and
the weekly quota window must have advanced past the November 14 refresh.
Missing/malformed data, stale quota, API failures and uncertain states refuse.
Only the reviewed Kaggle 2.2.2 / kagglesdk 0.1.37 adapter versions are accepted;
SDK changes require review of submission retry behavior. The submission requests
a **3600-second** session timeout through the documented
SDK argument; platform enforcement is not guaranteed. Existing worker time and
resource guards still apply. This is actual fused CUDA versus reference GPU
work, with no CPU fallback, alternate account, retry, redraw or polling loop.

Two preflight snapshots require the producer to remain COMPLETE at version 1,
with the same ID/timestamp and all three required output filenames. The target
must be absent from the authenticated `mine=True` search for its exact slug;
failed or truncated searches and results from other accounts refuse. A direct
lookup of a nonexistent target returned 403, and broad paginated listings
repeated entries, so neither is used to infer target absence. Producer metadata
uses an exact lookup. Search indexing and a different client submitting after
the last check remain races; the local claim cannot lock Kaggle globally.
Remote metadata exposes filenames,
not archive hashes; local evidence hashes and the worker's mounted-archive check
provide that verification. `kernel_sources` attaches the latest output and has
no supported version pin here: a producer rerun after the last check can race
attachment. Do not rerun the producer while this queued diagnostic is pending;
the worker checks its archive before inference.

A persistent exclusive claim in the repository's **Git common directory** blocks
duplicate manual invocations across its worktrees, even when a different receipt
path is supplied. Once remote preflight starts, any failure or unknown submission
outcome consumes that claim. Inspect the receipt and remote state manually;
never delete the claim to retry automatically. Use one canonical repository for
this job. A private JSONL receipt records safe states and pins, not usernames,
credentials, signed URLs or machine paths. A hard kill preserves the claim and
last flushed receipt event, including submission intent if its outcome is unknown.

An operator may schedule one reviewed private shell job with local `at` after
00:00 UTC on November 14, for example at 00:05. Freeze and hash the launcher and
its `recover_output.py` dependency before queuing; do not execute mutable branch
files on the scheduled date. The private job should set `umask 077`, source its
private environment without tracing, and invoke the frozen launcher under a
300-second local `timeout`, with absolute Python/repository/evidence/receipt
paths. No credentials should be in the shell command, output, or tracked files.
`at` snapshots its environment; enqueue from a clean environment so credential
values are not copied into its spool:

```bash
env -i HOME="$HOME" PATH=/usr/bin:/bin TZ=UTC \
  at -M -t 202611140005 -f "$PRIVATE_JOB"
atq
```

Check the daemon, single queued entry, private job hashes and local receipt
location when scheduling. `atd` must remain available; a late job refuses after
the one-day window. The launcher does not enqueue anything itself, and its
preparation tests neither schedule nor submit a job. Terminal GPU output still
requires separate recovery and review; submission is not model acceptance.

```bash
python -m unittest discover -s tests -p test_kaggle_deferred_launch.py -v
```
