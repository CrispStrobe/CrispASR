#!/usr/bin/env python3
"""T4 A/B of the FA MMA spill fix: our ggml HEAD (crispstrobe-ops) against the
fa-512-spill-fix branch, with ggml-cuda-fa-ab's alternating timing and
register/spill dump. Ratio = fix / HEAD, so < 1 is faster with the fix."""
import os, runpy, subprocess
from pathlib import Path

CR = Path("/kaggle/temp/CrispASR")
if not CR.exists():
    subprocess.run(["git", "clone", "--depth", "1", "https://github.com/CrispStrobe/CrispASR.git", str(CR)], check=True)
os.environ["AB_A"] = "434c2d79ab00077792f365424b2fd302af3c67f5"   # crispstrobe-ops as pinned by CrispASR main
os.environ["AB_B"] = "7c0a9284329b168cf29642b5c6b4b47abf6094b2"   # + keep unpadded FA expressions
runpy.run_path(str(CR / "tools" / "kaggle" / "ggml-cuda-fa-ab" / "ggml_cuda_fa_ab.py"), run_name="__main__")
