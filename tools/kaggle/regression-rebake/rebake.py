#!/usr/bin/env python3
"""Bake the missing regression reference dumps (chr1s4).

38 of the 45 entries in tests/regression/manifest.json are `skip_diff: true` —
transcript-only, because their ref dump was never produced. A transcript check
catches a backend that breaks loudly; it cannot catch a stage that drifts. This
runs the canonical suite in rebake mode so those entries can become full
stage-by-stage cosine diffs.

WHY A WRAPPER. `kaggle kernels push` uploads ONLY `code_file` (kaggle_usage.md,
"What gets uploaded" — proven by a ModuleNotFoundError, not inferred), so the
800-line canonical script cannot be shipped as a bundled sibling. It is fetched
from the repo at runtime, which is also how the C++ under test arrives.

The canonical script is driven entirely by environment variables and defaults to
MODE=validate; a kernel push cannot set env vars, so they are set here before it
is executed. That is the whole reason this file exists.
"""
import os, subprocess, sys, pathlib, time

# rebake + upload. UPLOAD=1 needs HF_TOKEN, which the harness resolves from the
# chr1s4 token dataset (see gotcha #13: private datasets are per-account, so
# this kernel MUST be pushed by chr1s4 and reference chr1s4's copies).
os.environ["CRISPASR_REGRESSION_MODE"] = "rebake"
os.environ["CRISPASR_REGRESSION_UPLOAD"] = "1"
os.environ["CRISPASR_REGRESSION_BUILD"] = os.environ.get("CRISPASR_REGRESSION_BUILD", "cpu")

SCRIPT_VERSION = "2026-09-07-rebake-1"
WORK = pathlib.Path("/kaggle/working")

# Clone into a SEPARATE bootstrap dir: the canonical script manages its own
# WORK/CrispASR checkout and pulls it, so pre-populating that path would have
# two owners for one directory.
BOOT = WORK / "_bootstrap"
if not BOOT.exists():
    subprocess.check_call(["git", "clone", "--depth", "1",
                           "https://github.com/CrispStrobe/CrispASR.git", str(BOOT)])

sha = "unknown"
try:
    sha = subprocess.check_output(["git", "-C", str(BOOT), "rev-parse", "--short", "HEAD"],
                                  text=True).strip()
except Exception:
    pass
# Gotcha #24: the kernel script is frozen at the last push while the repo is
# fresh, so a run can score new code with an old harness. Say both out loud.
print(f"[rebake] script_version={SCRIPT_VERSION}  bootstrap_clone={sha}  "
      f"mode={os.environ['CRISPASR_REGRESSION_MODE']} upload={os.environ['CRISPASR_REGRESSION_UPLOAD']}",
      flush=True)

target = BOOT / "tools" / "kaggle" / "crispasr-regression.py"
if not target.is_file():
    raise SystemExit(f"canonical suite not found at {target}")

t0 = time.time()
rc = subprocess.call([sys.executable, str(target)])
print(f"[rebake] canonical suite exited {rc} after {time.time()-t0:.0f}s", flush=True)
sys.exit(rc)
