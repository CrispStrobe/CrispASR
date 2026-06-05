# `tools/reference_envs/` — reproducible python envs per ref backend

Each subdirectory matches a `--backend` name from
`tools/dump_reference.py::REGISTERED_BACKENDS` and contains a
`requirements.txt` capturing the python dependencies needed to run
that backend's reference dumper.

The previous practice was a single shared venv (e.g.
`/Volumes/backups/ai/chatterbox-ref-venv`) maintained by hand. When
those venvs got pruned or fell out of sync with the ref backend's
imports, regenerating a reference archive required re-discovering the
correct package set from scratch. The 2026-05-25 chatterbox
investigation hit exactly that: the venv was a stub, the actual
dependency set wasn't documented anywhere, and we had to work around
it by patching the existing archive.

## Bootstrap a backend's env

```bash
tools/bootstrap_ref_env.sh <backend>
```

Creates a venv at `/Volumes/backups/ai/refenvs/<backend>/` (or
`$CRISPASR_REF_ENVS_ROOT/<backend>` if set), installs the pinned
deps, and prints the activate command. Override the install root if
you don't want the env on the external disk (e.g.
`CRISPASR_REF_ENVS_ROOT=$HOME/.crispasr-refenvs`).

After bootstrap, dump a reference archive:

```bash
source /Volumes/backups/ai/refenvs/<backend>/bin/activate
python tools/dump_reference.py --backend <backend> \\
    --model-dir <hf-snapshot-or-local-dir> \\
    --audio samples/jfk.wav \\
    --output /Volumes/backups/ai/<backend>-ref.gguf
```

Use `tools/audit_diff_coverage.py` to see which backends still need
an archive.

## Scope of each `requirements.txt`

Each file is a **scaffold** generated from the actual import
statements in `tools/reference_backends/<backend>.py` (and any
subimports). Two caveats:

1. **PyPI-only.** Packages installed from git (upstream chatterbox,
   custom forks) need their git-URL added manually. The scaffolds
   leave a `# TODO:` line where this is likely needed.

2. **No version pins by default.** Pin versions before relying on a
   given backend's archive for parity work — `pip freeze >
   requirements.lock.txt` after a successful dump is the easiest
   way to capture the resolved set.

## Adding a new backend env

```bash
# After adding tools/reference_backends/<name>.py and wiring it into
# tools/dump_reference.py::REGISTERED_BACKENDS:
python tools/audit_diff_coverage.py --emit-env-scaffolds
# Edit tools/reference_envs/<name>/requirements.txt to add git URLs
# and pin versions, then:
tools/bootstrap_ref_env.sh <name>
```
