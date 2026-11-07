#!/usr/bin/env python3
"""Run the manifest `align_backends` regression entries off the GitHub queue.

Clones main (or CRISPASR_REF), builds crispasr-cli on CPU and runs
tests/regression/run_one.py for each forced-aligner entry: Qwen3 0.6B,
Qwen3 Ultra and canary-ctc. The pass/fail rule is the manifest's.
"""
import json, os, subprocess, sys, traceback
from pathlib import Path
OUT = Path("/kaggle/working/out"); OUT.mkdir(parents=True, exist_ok=True)
REPO = Path("/tmp/CrispASR"); G = Path("/tmp/g"); G.mkdir(exist_ok=True)
REF = os.environ.get("CRISPASR_REF", "main")
ENTRIES = ["qwen3-forced-aligner-0.6b-f16", "qwen3-forced-aligner-ultra-f16", "canary-ctc-aligner-q4_k"]
res = {"errors": [], "entries": {}}
def save(): (OUT / "result.json").write_text(json.dumps(res, indent=1))
try:
    subprocess.check_call(["git", "clone", "--depth", "1", "-b", REF, "https://github.com/CrispStrobe/CrispASR.git", str(REPO)])
    subprocess.run(["git", "submodule", "update", "--init", "--recursive", "--depth", "1"], cwd=str(REPO))
    res["commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(REPO), text=True).strip()
    sys.path.insert(0, str(REPO / "tools" / "kaggle")); import kaggle_harness as kh
    tok = kh.resolve_hf_token()
    if tok: os.environ["HF_TOKEN"] = tok
    kh.install_build_toolchain()
    kh.sh(f"cmake -S {REPO} -B {REPO}/build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCRISPASR_OPUS=OFF -DCRISPASR_AMR=OFF "
          + " ".join(kh.cache_and_link_flags()))
    kh.sh(f"cmake --build {REPO}/build -j$(nproc) --target crispasr-cli")
    res["cpu"] = subprocess.run(["lscpu"], capture_output=True, text=True).stdout.split("Model name:")[-1].split("\n")[0].strip()
    env = dict(os.environ, CRISPASR_BIN=str(REPO / "build/bin/crispasr"), WORK_DIR=str(G / "reg"), HF_HOME=str(G / "hf"))
    for name in ENTRIES:
        r = subprocess.run([sys.executable, str(REPO / "tests/regression/run_one.py"), name],
                           capture_output=True, text=True, env=env, cwd=str(REPO), timeout=3600)
        res["entries"][name] = {"rc": r.returncode, "tail": (r.stdout + r.stderr)[-2000:]}
        save()
except BaseException:
    res["errors"].append(traceback.format_exc())
finally:
    save()
