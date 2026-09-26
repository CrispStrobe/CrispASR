#!/usr/bin/env python3
"""CPU A/B for Cohere's hidden F16 pointwise matmuls (PLAN §247)."""

import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

WORK = Path("/kaggle/working")
TEMP = Path("/kaggle/temp") if Path("/kaggle/temp").is_dir() else Path("/tmp")
REPO = WORK / "CrispASR"
BUILD = TEMP / "build"
MODEL = TEMP / "cohere-transcribe-q4_k.gguf"
SHA = "06f0ce7b"

subprocess.check_call(["git", "clone", "--filter=blob:none", "--no-checkout",
                       "https://github.com/CrispStrobe/CrispASR.git", str(REPO)])
subprocess.check_call(["git", "checkout", SHA], cwd=REPO)
subprocess.check_call(["git", "submodule", "update", "--init", "ggml"], cwd=REPO)
sys.path.insert(0, str(REPO / "tools" / "kaggle"))
import kaggle_harness as kh  # noqa: E402

kh.init_progress(hf_progress_repo="cstr/crispasr-kaggle-progress")
step = kh.step
step("script.start", sha=SHA, run="v1")
token = kh.resolve_hf_token("HF_TOKEN")
subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", "huggingface_hub"])
from huggingface_hub import hf_hub_download  # noqa: E402

kh.install_build_toolchain()
ccache_run = TEMP / ".ccache"
warmed = WORK / ".ccache"
if warmed.exists():
    if ccache_run.exists():
        shutil.rmtree(ccache_run)
    shutil.move(str(warmed), str(ccache_run))
ccache_run.mkdir(parents=True, exist_ok=True)
os.environ["CCACHE_DIR"] = str(ccache_run)

BUILD.mkdir(parents=True, exist_ok=True)
flags = " ".join(kh.cache_and_link_flags())
subprocess.check_call(f"cmake -G Ninja -S {REPO} -B {BUILD} -DCMAKE_BUILD_TYPE=Release {flags}", shell=True)
with kh.build_heartbeat("cmake.build"):
    kh.sh_with_progress(f"stdbuf -oL -eL cmake --build {BUILD} --target crispasr-cli -j {kh.safe_build_jobs(False)}")
cli = BUILD / "bin" / "crispasr"
assert cli.is_file()
step("build.done")

downloaded = hf_hub_download("cstr/cohere-transcribe-03-2026-GGUF",
                             "cohere-transcribe-q4_k.gguf", token=token,
                             local_dir=str(TEMP))
if Path(downloaded) != MODEL:
    shutil.copyfile(downloaded, MODEL)
wav = REPO / "samples" / "jfk.wav"


def run(enabled, profile=False):
    env = dict(os.environ)
    env["CRISPASR_COHERE_PW_Q8"] = "1" if enabled else "0"
    if profile:
        env["CRISPASR_SCHED_PROFILE"] = "1"
    cmd = [str(cli), "--no-gpu", "--backend", "cohere", "-m", str(MODEL),
           "-f", str(wav), "-l", "en", "-t", "4"]
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, env=env, text=True, capture_output=True, timeout=900)
    elapsed = time.perf_counter() - t0
    if proc.returncode:
        raise RuntimeError(f"run failed rc={proc.returncode}\n{proc.stderr[-4000:]}")
    transcript = " ".join(line.strip() for line in proc.stdout.splitlines() if line.strip())
    prof_rows = [line.strip() for line in proc.stderr.splitlines() if "sched_profile[cohere]" in line]
    repacked = re.search(r"repacked (\d+) F16 matmul tensors", proc.stderr)
    return {"elapsed_s": elapsed, "transcript": transcript,
            "repacked": int(repacked.group(1)) if repacked else 0,
            "profile": prof_rows}


# Alternate arms after one warm-up each to limit thermal/order bias.
run(False)
run(True)
control, fixed = [], []
for _ in range(3):
    control.append(run(False))
    fixed.append(run(True))
control_prof = run(False, True)
fixed_prof = run(True, True)

same_text = all(x["transcript"] == control[0]["transcript"] for x in control + fixed)
control_med = sorted(x["elapsed_s"] for x in control)[1]
fixed_med = sorted(x["elapsed_s"] for x in fixed)[1]
summary = {
    "sha": SHA,
    "same_transcript": same_text,
    "control_seconds": [x["elapsed_s"] for x in control],
    "fixed_seconds": [x["elapsed_s"] for x in fixed],
    "control_median_s": control_med,
    "fixed_median_s": fixed_med,
    "speedup": control_med / fixed_med,
    "repacked": fixed[0]["repacked"],
    "control_profile": control_prof["profile"],
    "fixed_profile": fixed_prof["profile"],
    "passed": same_text and fixed[0]["repacked"] == 96 and fixed_med < control_med,
}
(WORK / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
step("ab.done", passed=summary["passed"], speedup=summary["speedup"])

# Protocol: export the successful build cache so the matching dataset can be refreshed.
subprocess.check_call(["tar", "cf", str(WORK / "ccache.tar"), "-C", str(TEMP), ".ccache"])
step("ccache.packed")
if not summary["passed"]:
    raise SystemExit(1)
