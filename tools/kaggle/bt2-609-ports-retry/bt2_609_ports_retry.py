#!/usr/bin/env python3
"""#609 ports, second pass: only the arms that failed to set up in the first run
(the C++ ports' vendored ggml hit Kaggle's missing CUDA::cuda_driver; the MLX venv
hit the missing ensurepip). Runs bt2-609-ports/bt2_609_ports.py from main with ARMS
set; "ours" (done in the first pass) is skipped, its build still provides the scorer."""
import os, runpy, subprocess
from pathlib import Path

CR = Path("/kaggle/temp/CrispASR")
if not CR.exists():
    subprocess.run(["git", "clone", "--depth", "1", "--recurse-submodules", "--shallow-submodules",
                    "https://github.com/CrispStrobe/CrispASR.git", str(CR)], check=True)
os.environ.setdefault("ARMS", "hoppou,audiocpp,qwentts,mlxaudio,rishikksh,vanch")
runpy.run_path(str(CR / "tools" / "kaggle" / "bt2-609-ports" / "bt2_609_ports.py"), run_name="__main__")
