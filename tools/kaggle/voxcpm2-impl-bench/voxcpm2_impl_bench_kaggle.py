#!/usr/bin/env python3
"""VoxCPM2 cross-implementation bench on a Kaggle CUDA GPU.

Thin wrapper: clones CrispASR at REF and runs tools/ci-heavy/voxcpm2_impl_bench.py
with device=cuda (crispasr, llama.cpp-omni and VoxCPM.cpp, each built for the
detected GPU arch). Results: /kaggle/working/out/{result.json,summary.md}.
Non-torch, so a P100 (sm_60) draw is still a valid CUDA measurement; the
arch is recorded in the summary.
"""
import os
import subprocess
import sys
from pathlib import Path

REF = "diag/voxcpm2-bench"
REPO = Path("/tmp/CrispASR")
OUT = Path("/kaggle/working/out")
OUT.mkdir(parents=True, exist_ok=True)
subprocess.check_call(["git", "clone", "--depth", "1", "-b", REF, "https://github.com/CrispStrobe/CrispASR.git", str(REPO)])
subprocess.run(["git", "submodule", "update", "--init", "--recursive", "--depth", "1"], cwd=str(REPO))
sys.path.insert(0, str(REPO / "tools" / "kaggle"))
import kaggle_harness as kh  # noqa: E402

kh.init_progress()
tok = kh.resolve_hf_token(require=False)
if tok:
    os.environ["HF_TOKEN"] = tok
kh.install_build_toolchain()
arch = kh.detect_cuda_arch()
kh.cuda_build_flags(arch)  # puts the CUDA stubs on LIBRARY_PATH for all three builds
gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,compute_cap", "--format=csv,noheader"],
                     capture_output=True, text=True).stdout.strip()
(OUT / "gpu.txt").write_text(f"{gpu} arch={arch}\n")
print("GPU", gpu, "arch", arch, flush=True)
env = dict(os.environ, HEAVY_OUT=str(OUT), HEAVY_SCRATCH="/tmp/scr")
with kh.build_heartbeat("voxcpm2-impl-bench"):
    r = subprocess.run([sys.executable, str(REPO / "tools/ci-heavy/voxcpm2_impl_bench.py"),
                        "device=cuda", "runs=5", f"cuda_arch={arch}"], env=env)
print("bench rc", r.returncode, flush=True)
