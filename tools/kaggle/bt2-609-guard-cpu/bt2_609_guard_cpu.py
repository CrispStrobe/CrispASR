#!/usr/bin/env python3
"""#609 length guard, CPU-only functional check.

Runs the bt2-609-eos matrix (payloads_r1, both voices, seeds 87/42/0/1) on a
CPU session, which needs no GPU slot. Checks that the guard fires on runaways,
resamples, and leaves normal reads alone. The reporter's CPU case
(000206 Piedmont seed 87) ran 590 frames without the guard.
"""
import os, runpy, subprocess
from pathlib import Path

os.environ.update(BT2_CPU_ONLY="1", N_SEEDS="2", CRISPASR_BRANCH=os.environ.get("CRISPASR_BRANCH", "test/609"))
CR = Path("/kaggle/temp/CrispASR")
CR.parent.mkdir(parents=True, exist_ok=True)
if not CR.exists():
    subprocess.run(["git", "clone", "--depth", "1", "--branch", os.environ["CRISPASR_BRANCH"], "--recurse-submodules",
                    "--shallow-submodules", "https://github.com/CrispStrobe/CrispASR.git", str(CR)], check=True)
runpy.run_path(str(CR / "tools/kaggle/bt2-609-eos/bt2_609_eos.py"), run_name="__main__")
