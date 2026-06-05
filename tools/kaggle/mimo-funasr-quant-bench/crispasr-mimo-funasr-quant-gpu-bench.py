"""
CrispASR — mimo-asr + funasr-mlt-nano: quant x GPU benchmark matrix.

Tests both backends with different quantizations and GPU/CPU paths.
Downloads one model at a time, benchmarks, cleans up before the next.

Known constraints:
  - mimo-asr default path (mixed CPU+CUDA sched) hangs on CUDA —
    only test pure-CPU and FORCE_GPU (GPU-only sched, fix 3ef9f87e).
  - funasr-mlt-nano F16 hits CUDA F16xF32 !-loop (issue #125) —
    use Q8_0 for GPU testing.

Build/report plumbing from the shared harness tools/kaggle/kaggle_harness.py.
enable_gpu=true in kernel-metadata.json.
"""

import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

os.environ["PYTHONUNBUFFERED"] = "1"
try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except (AttributeError, ValueError):
    pass

WORK = Path("/kaggle/working")
REPO = WORK / "CrispASR"
BUILD = WORK / "build"
RESULTS = WORK / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

CRISPASR_REF = os.environ.get("CRISPASR_REF", "main")
CRISPASR_REPO = os.environ.get("CRISPASR_REPO",
                                "https://github.com/CrispStrobe/CrispASR.git")
EXPECTED_JFK = "ask not what your country can do for you"


def run(cmd, check=True, capture=False, env=None, timeout=None):
    print(f"\n$ {' '.join(str(c) for c in cmd)}", flush=True)
    e = os.environ.copy()
    if env:
        e.update(env)
    if capture:
        r = subprocess.run(cmd, env=e, timeout=timeout,
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        print(r.stdout[-4096:] if len(r.stdout) > 4096 else r.stdout, flush=True)
        if check and r.returncode != 0:
            raise SystemExit(f"command failed (rc={r.returncode}): {cmd}")
        return r
    r = subprocess.run(cmd, env=e, timeout=timeout)
    if check and r.returncode != 0:
        raise SystemExit(f"command failed (rc={r.returncode}): {cmd}")
    return r


# ── Clone + CUDA build (standard harness pattern) ──────────────────────────
print(f"[start] ref={CRISPASR_REF}", flush=True)
if REPO.exists():
    shutil.rmtree(REPO)
run(["git", "clone", "--depth", "1", "--branch", CRISPASR_REF,
     "--recursive", CRISPASR_REPO, str(REPO)])

sys.path.insert(0, os.path.join(str(REPO), "tools", "kaggle"))
import kaggle_harness as kh

kh.init_progress()
kh.resolve_hf_token()

sha = subprocess.check_output(
    ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip()
kh.step("cloned", sha=sha, ref=CRISPASR_REF)

run(["nvidia-smi", "-L"])
gpu_name = subprocess.check_output(
    ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
    text=True).strip()
kh.step("gpu", gpu_name=gpu_name)

kh.install_build_toolchain()
arch = kh.detect_cuda_arch()
kh.step("cuda_arch", arch=arch)

BUILD.mkdir(parents=True, exist_ok=True)
cmake_args = (
    ["cmake", "-S", str(REPO), "-B", str(BUILD),
     "-DCMAKE_BUILD_TYPE=Release",
     "-DCRISPASR_BUILD_TESTS=OFF"]
    + kh.cuda_build_flags(arch)
    + kh.cache_and_link_flags()
)
run(cmake_args)
kh.step("cmake_done")

with kh.build_heartbeat("cmake.build"):
    kh.sh_with_progress(
        f"stdbuf -oL -eL cmake --build {BUILD} --target crispasr-cli "
        f"-j{kh.safe_build_jobs(gpu=True)}")

CLI = BUILD / "bin" / "crispasr"
if not CLI.exists():
    cands = [c for c in BUILD.rglob("crispasr")
             if c.is_file() and os.access(c, os.X_OK)]
    assert cands, "crispasr binary not found after build"
    CLI = cands[0]
kh.step("build_done", binary=str(CLI))

# Copy sample audio to a short path (avoids long CLI lines in logs)
JFK = WORK / "jfk.wav"
shutil.copy(REPO / "samples" / "jfk.wav", JFK)

# ── Benchmark harness ──────────────────────────────────────────────────────
results = []


def bench(label, backend, extra_env=None, extra_args=None, timeout=300):
    """Run one config via -m auto --auto-download, then clean cache."""
    print(f"\n{'='*60}", flush=True)
    print(f"  {label}", flush=True)
    print(f"{'='*60}", flush=True)
    kh.step(f"bench.{label}.start")

    cmd = [str(CLI), "--backend", backend, "-m", "auto", "--auto-download",
           "-f", str(JFK), "-t", "4", "--no-prints"]
    if extra_args:
        cmd += extra_args

    env = os.environ.copy()
    env[f"{backend.upper().replace('-','_')}_BENCH"] = "1"
    if extra_env:
        env.update(extra_env)

    print(f"  cmd: {' '.join(cmd)}", flush=True)
    if extra_env:
        print(f"  env: {extra_env}", flush=True)

    t0 = time.time()
    try:
        # Use Popen + line-by-line read to prevent pipe deadlock
        proc = subprocess.Popen(
            cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1)
        output_lines = []
        deadline = t0 + timeout
        while True:
            if time.time() > deadline:
                proc.kill()
                proc.wait()
                raise subprocess.TimeoutExpired(cmd, timeout)
            line = proc.stdout.readline()
            if not line and proc.poll() is not None:
                break
            if line:
                output_lines.append(line)
                # Print live to prevent Kaggle log stall
                sys.stdout.write(f"    [live] {line}")
                sys.stdout.flush()
        rc = proc.returncode
        wall = time.time() - t0
        transcript = output_lines[-1].strip() if output_lines else ""
        ok = rc == 0 and EXPECTED_JFK in transcript.lower()
        rtf = 11.0 / wall if wall > 0 else 0
        status = "PASS" if ok else f"FAIL(rc={rc})"

        # Extract bench line if present
        bench_line = ""
        for ln in output_lines:
            if "_bench:" in ln:
                bench_line = ln.strip()

        print(f"  {status} wall={wall:.1f}s RTx={rtf:.2f}", flush=True)
        if transcript:
            print(f"  transcript: {transcript[:150]}", flush=True)
        if bench_line:
            print(f"  bench: {bench_line}", flush=True)
        results.append({"label": label, "status": "PASS" if ok else "FAIL",
                        "wall_s": round(wall, 1), "rtf": round(rtf, 2),
                        "transcript": transcript[:150], "bench": bench_line})
    except subprocess.TimeoutExpired:
        wall = time.time() - t0
        print(f"  TIMEOUT after {wall:.0f}s", flush=True)
        results.append({"label": label, "status": "TIMEOUT",
                        "wall_s": round(wall, 1)})

    kh.step(f"bench.{label}.done", status=results[-1]["status"])
    # Clean up downloaded models before next config
    cache_dir = os.path.expanduser("~/.cache/crispasr")
    if os.path.isdir(cache_dir):
        freed = 0
        for f in os.listdir(cache_dir):
            fpath = os.path.join(cache_dir, f)
            if os.path.isfile(fpath) and "ggml-tiny" not in f and "silero" not in f:
                freed += os.path.getsize(fpath) / 1024 / 1024
                os.remove(fpath)
        if freed > 10:
            print(f"  Cleaned: {freed:.0f} MB freed", flush=True)
    fg = kh.free_gb()
    if fg is not None:
        print(f"  Disk free: {fg:.1f} GB", flush=True)


# ── mimo-asr: skip default mixed sched (hangs on CUDA), test CPU + GPU-only
bench("mimo Q4_K CPU-forced",
      "mimo-asr",
      extra_env={"GGML_CUDA_DISABLE": "1"},
      timeout=420)

bench("mimo Q4_K GPU-only (FORCE_GPU)",
      "mimo-asr",
      extra_env={"CRISPASR_MIMO_FORCE_GPU": "1"},
      timeout=120)

# ── funasr-mlt-nano: F16 on CPU, Q8_0 on CPU, Q8_0 with LLM on GPU
bench("funasr-mlt Q8_0 CPU",
      "fun-asr-mlt-nano",
      extra_env={"GGML_CUDA_DISABLE": "1"},
      extra_args=["-m", "funasr-mlt-nano-2512-q8_0.gguf"],
      timeout=180)

bench("funasr-mlt Q8_0 GPU (enc GPU, LLM GPU)",
      "fun-asr-mlt-nano",
      extra_env={"FUNASR_LLM_GPU": "1"},
      extra_args=["-m", "funasr-mlt-nano-2512-q8_0.gguf"],
      timeout=60)

bench("funasr-mlt Q4_K CPU",
      "fun-asr-mlt-nano",
      extra_env={"GGML_CUDA_DISABLE": "1"},
      extra_args=["-m", "funasr-mlt-nano-2512-q4_k.gguf"],
      timeout=180)

# ── Summary ────────────────────────────────────────────────────────────────
print(f"\n{'='*60}", flush=True)
print("RESULTS SUMMARY", flush=True)
print(f"{'='*60}", flush=True)
print("| Label | Status | Wall (s) | RTx | Bench |", flush=True)
print("|---|---|---|---|---|", flush=True)
for r in results:
    wall = r.get("wall_s", "-")
    rtf = r.get("rtf", "-")
    b = r.get("bench", "")[:60]
    print(f"| {r['label']} | {r['status']} | {wall} | {rtf} | {b} |", flush=True)

with open(RESULTS / "results.json", "w") as f:
    json.dump(results, f, indent=2)

n_pass = sum(1 for r in results if r["status"] == "PASS")
kh.step("done", n_pass=n_pass, n_total=len(results))
print(f"\nDone. {n_pass}/{len(results)} passed.", flush=True)
