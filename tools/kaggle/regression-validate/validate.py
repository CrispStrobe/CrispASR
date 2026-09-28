#!/usr/bin/env python3
"""Run the canonical suite (tools/kaggle/crispasr-regression.py) in VALIDATE mode
on a chosen ref + backend subset - the proof step before manifest changes
(diff flips, repins) land on main. Same wrapper pattern as regression-rebake:
a kernel push cannot set env vars, so they are set here. CPU build, like GH.
"""
import os, subprocess, sys, pathlib
REF = "ci/regression-hardening"
BACKENDS = ["wav2vec2-xlsr-en", "hubert-large", "parakeet-tdt-0.6b-en", "mini-omni2",
            "nemotron-3.5-asr-streaming-0.6b"]
os.environ["CRISPASR_REGRESSION_MODE"] = "validate"
os.environ["CRISPASR_REGRESSION_BUILD"] = "cpu"
os.environ["CRISPASR_REF"] = os.environ.get("CRISPASR_REF", REF)
os.environ["CRISPASR_REGRESSION_BACKENDS"] = os.environ.get("CRISPASR_REGRESSION_BACKENDS", ",".join(BACKENDS))
BOOT = pathlib.Path("/kaggle/working/_bootstrap")
if not BOOT.exists():
    subprocess.check_call(["git", "clone", "--depth", "1", "-b", os.environ["CRISPASR_REF"],
                           "https://github.com/CrispStrobe/CrispASR.git", str(BOOT)])
sys.path.insert(0, str(BOOT / "tools" / "kaggle"))
try:
    import kaggle_harness as kh
    _tok = kh.resolve_hf_token()
    if _tok:
        os.environ["HF_TOKEN"] = _tok
except Exception as e:
    print(f"[validate] token resolution failed: {e}", flush=True)
print("[validate] ref", os.environ["CRISPASR_REF"], "backends", os.environ["CRISPASR_REGRESSION_BACKENDS"], flush=True)
rc = subprocess.call([sys.executable, str(BOOT / "tools/kaggle/crispasr-regression.py")])
import shutil  # keep the output listable (see regression-rebake/rebake.py)
for _d in ("CrispASR", "_bootstrap", ".ccache", "hf_cache", "build"):
    shutil.rmtree(pathlib.Path("/kaggle/working") / _d, ignore_errors=True)
sys.exit(rc)
