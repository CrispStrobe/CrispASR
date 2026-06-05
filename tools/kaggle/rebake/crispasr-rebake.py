# %% [markdown]
# # CrispASR — reference re-bake (Kaggle, compute-only)
#
# Generates fresh reference dumps from the real NeMo / transformers /
# torch source models and writes them to `/kaggle/working/rebake-stage/`.
# **No HF upload here** — the staged files come down via
# `kaggle kernels output` and `hf upload` runs locally where the
# HF write token is already in place. That sidesteps Kaggle Secrets
# entirely (their GetUserSecretByLabel endpoint has flaked on us
# repeatedly — HTTPError 400 even after Attach toggle).
#
# This kernel:
#   1. Clones CrispASR main.
#   2. Sets `MODE=rebake` + `UPLOAD=0` (the latter is the change —
#      previously was 1).
#   3. Execs the canonical `tools/kaggle/crispasr-regression.py`,
#      which runs `tools/dump_reference.py` for every manifest entry
#      and writes the new `ref.gguf` files into the staging dir.
#
# After the run lands, fetch + publish:
#
#   ./tools/kaggle/rebake/fetch-and-upload.sh
#
# That script pulls `/kaggle/working/rebake-stage/` to local
# `/Volumes/backups/ai/crispasr-regression/rebake-out/` then
# `hf upload`s to `cstr/crispasr-regression-fixtures` using
# the proven-good `HF_TOKEN` from `.env`.

# %% [code]
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# ── Unbuffered I/O + tiny progress.jsonl checkpointer ────────────────
# Last rebake run hung 11+h inside cmake build between two consecutive
# compiler-warning lines — invisible because Python's print buffer
# fills slowly and Kaggle only flushes at process exit. Forcing
# line-buffered stdio + writing a JSONL marker at every major bootstrap
# step means the next hang shows up in /kaggle/working/progress.jsonl
# (fetchable via `kaggle kernels output --file-pattern 'progress.jsonl'`
# without waiting for termination).
os.environ["PYTHONUNBUFFERED"] = "1"
try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except (AttributeError, ValueError):
    pass

import json as _json
_PROGRESS = Path("/kaggle/working/progress.jsonl")
_T0 = time.time()


def _step(name: str, **kw):
    rec = {"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "elapsed_s": round(time.time() - _T0, 2), "step": name, **kw}
    try:
        _PROGRESS.parent.mkdir(parents=True, exist_ok=True)
        with _PROGRESS.open("a") as f:
            f.write(_json.dumps(rec) + "\n")
    except Exception:
        pass
    print(f"[boot {rec['elapsed_s']:>6.1f}s] {name}", flush=True)


_step("bootstrap.start")

WORK = Path("/kaggle/working")
REPO = WORK / "CrispASR-bootstrap"

# Defaults for the rebake kernel — overridable from Kaggle's
# Add-ons → Variables UI if you ever want one-off behaviour without
# editing this notebook.
os.environ.setdefault("CRISPASR_REGRESSION_MODE", "rebake")
# UPLOAD=1 → attempt auto-upload via api.upload_folder() at end.
# Works when HF_TOKEN reads cleanly (UI-triggered runs do this).
# Falls back gracefully to staging-only if token is unreadable
# (Kaggle Secrets batch-mode flake) — see preflight_hf() warning;
# pick up with ./tools/kaggle/rebake/fetch-and-upload.sh.
os.environ.setdefault("CRISPASR_REGRESSION_UPLOAD", "1")

# %% [code]
if not REPO.exists():
    _step("git-clone.begin")
    subprocess.check_call([
        "git", "clone", "--recursive", "--depth", "20",
        "https://github.com/CrispStrobe/CrispASR.git", str(REPO),
    ])
    _step("git-clone.done")
else:
    _step("git-clone.skipped", reason="repo-cache-hit")

script = REPO / "tools" / "kaggle" / "crispasr-regression.py"
_step("exec-canonical.begin", script=str(script))
sys.argv[0] = str(script)
exec(compile(script.read_text(), str(script), "exec"))

# %% [code]
# Surface the staged paths in the kernel output so they're easy to
# spot in the Kaggle UI / log + downloadable via `kaggle kernels output`.
import json
staged = list(Path("/kaggle/working/rebake-stage").rglob("*.gguf"))
print(f"\n=== staged {len(staged)} ref.gguf file(s) for local upload ===")
for p in staged:
    print(f"  {p}  ({p.stat().st_size} bytes)")
print("\n=== next step (locally, NOT on Kaggle) ===")
print("  ./tools/kaggle/rebake/fetch-and-upload.sh")
