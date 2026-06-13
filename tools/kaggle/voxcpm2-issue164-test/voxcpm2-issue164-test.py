"""
CrispASR issue #164 validation: voxcpm2 VOXCPM2_USE_GRAPH stop predictor + VAE.

Tests:
  A. graph=0 baseline — stop should fire (validates model works).
  B. graph=1 on CUDA — stop should now fire (was broken before fix).
  C. graph=1 on CPU — stop should fire (was broken before fix).

Expected ~20 min on T4 (build + download + 3 TTS runs).
"""

import json
import os
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
CRISPASR_REPO = "https://github.com/CrispStrobe/CrispASR.git"
CRISPASR_REF = os.environ.get("CRISPASR_REF", "main")
MODEL_REPO = "cstr/voxcpm2-GGUF"
MODEL_FILE = "voxcpm2-q4_k.gguf"

_T0 = time.time()
_PROGRESS_PATH = WORK / "progress.jsonl"


def step(name, **kw):
    entry = {"t": round(time.time() - _T0, 1), "step": name, **kw}
    print(json.dumps(entry), flush=True)
    with open(_PROGRESS_PATH, "a") as f:
        f.write(json.dumps(entry) + "\n")


def run(cmd, capture=False, check=True, timeout=600, cwd=None, env=None):
    e = {**os.environ, **(env or {})}
    if capture:
        r = subprocess.run(
            cmd, env=e, cwd=cwd, timeout=timeout,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        )
        print(r.stdout[-4096:] if len(r.stdout) > 4096 else r.stdout, flush=True)
        if check and r.returncode != 0:
            raise SystemExit(f"command failed (rc={r.returncode}): {cmd}")
        return r
    r = subprocess.run(cmd, env=e, cwd=cwd, timeout=timeout)
    if check and r.returncode != 0:
        raise SystemExit(f"command failed (rc={r.returncode}): {cmd}")
    return r


# -- Clone + build --
step("start", ref=CRISPASR_REF)

if REPO.exists():
    import shutil
    shutil.rmtree(REPO)
run(["git", "clone", "--depth", "1", "--branch", CRISPASR_REF, "--recursive",
     CRISPASR_REPO, str(REPO)])

sha = subprocess.check_output(
    ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True
).strip()
step("cloned", sha=sha)

sys.path.insert(0, os.path.join(str(REPO), "tools", "kaggle"))
import kaggle_harness as kh

kh.init_progress()
run(["nvidia-smi", "-L"])
gpu_name = subprocess.check_output(
    ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True
).strip()
step("gpu", gpu_name=gpu_name)

kh.install_build_toolchain()
arch = kh.detect_cuda_arch()
step("cuda_arch", arch=arch)

BUILD.mkdir(parents=True, exist_ok=True)
cmake_args = (
    ["cmake", "-S", str(REPO), "-B", str(BUILD),
     "-DCMAKE_BUILD_TYPE=Release", "-DBUILD_SHARED_LIBS=ON"]
    + kh.cuda_build_flags(arch)
    + kh.cache_and_link_flags()
)
run(cmake_args)
step("cmake_done")

with kh.build_heartbeat("cmake.build"):
    kh.sh_with_progress(
        f"stdbuf -oL -eL cmake --build {BUILD} --target crispasr-cli "
        f"-j{kh.safe_build_jobs(gpu=True)}"
    )
step("build_done")

CLI = BUILD / "bin" / "crispasr"
if not CLI.exists():
    candidates = list(BUILD.rglob("crispasr"))
    candidates = [c for c in candidates if c.is_file() and os.access(c, os.X_OK)]
    if not candidates:
        raise SystemExit("crispasr binary not found after build")
    CLI = candidates[0]
print(f"crispasr binary: {CLI}", flush=True)

LIB_DIR = BUILD / "src"
os.environ["LD_LIBRARY_PATH"] = (
    f"{LIB_DIR}:{os.environ.get('LD_LIBRARY_PATH', '')}"
)

# -- Download model --
MODEL_DIR = WORK / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
model_path = MODEL_DIR / MODEL_FILE

if not model_path.exists():
    token = kh.resolve_hf_token()
    from huggingface_hub import hf_hub_download
    hf_hub_download(
        repo_id=MODEL_REPO, filename=MODEL_FILE,
        local_dir=str(MODEL_DIR), token=token,
    )
step("model_downloaded", model=str(model_path),
     size_mb=round(model_path.stat().st_size / 1e6, 1))


# -- TTS runner --
TEST_TEXT = "Hello world."
OUT_WAV = WORK / "test_output.wav"


def run_tts(label: str, extra_env: dict, timeout: int = 300) -> dict:
    step(f"{label}_start")
    cmd = [
        str(CLI), "-m", str(model_path),
        "--tts", TEST_TEXT,
        "--tts-output", str(OUT_WAV),
        "-t", "2",
        "--verbose", "2",
    ]
    env = {**os.environ, **extra_env}
    t0 = time.time()
    r = subprocess.run(
        cmd, timeout=timeout,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        env=env,
    )
    elapsed = time.time() - t0

    result = {
        "label": label,
        "rc": r.returncode,
        "elapsed": round(elapsed, 1),
    }

    for line in r.stderr.splitlines():
        if "stop=" in line:
            result.setdefault("stop_lines", []).append(line.strip())
        if "stopped at step" in line:
            result["stop_fired"] = True
            result["stop_line"] = line.strip()
        if "hit max_len ceiling" in line:
            result["stop_fired"] = False
            result["ceiling_line"] = line.strip()
        if "replayed" in line:
            result["replayed"] = True

    print(f"  {label}: rc={r.returncode}, elapsed={elapsed:.1f}s, "
          f"stop_fired={result.get('stop_fired', 'unknown')}", flush=True)
    for line in r.stderr.splitlines():
        if "voxcpm2:" in line:
            print(f"    {line.strip()}", flush=True)
    if "stop_lines" in result:
        print(f"  First 10 stop lines:", flush=True)
        for sl in result["stop_lines"][:10]:
            print(f"    {sl}", flush=True)

    step(f"{label}_done", **{k: v for k, v in result.items()
                             if k not in ("stdout", "stderr")})
    return result


# -- Config A: graph=0 (baseline) --
print("\n=== Config A: VOXCPM2_USE_GRAPH=0 (baseline) ===", flush=True)
res_a = run_tts("nograph", {"VOXCPM2_USE_GRAPH": "0"})

# -- Config B: graph=1, CUDA, bucketed (current default — known NaN) --
print("\n=== Config B: VOXCPM2_USE_GRAPH=1, bucketed (CUDA) ===", flush=True)
res_b = run_tts("graph_bucket", {"VOXCPM2_USE_GRAPH": "1"})

# -- Config C: graph=1, CUDA, NO_BUCKET (dynamic graph, ggml_cpy KV) --
print("\n=== Config C: VOXCPM2_USE_GRAPH=1, NO_BUCKET (CUDA) ===", flush=True)
res_c = run_tts("graph_nobucket", {
    "VOXCPM2_USE_GRAPH": "1",
    "VOXCPM2_NO_BUCKET": "1",
})

# -- Summary --
print("\n" + "=" * 60, flush=True)
print("RESULTS SUMMARY", flush=True)
print("=" * 60, flush=True)
all_pass = True
for r in [res_a, res_b, res_c]:
    sf = r.get("stop_fired", "unknown")
    sl = r.get("stop_line", r.get("ceiling_line", ""))
    status = "PASS" if sf is True else ("FAIL" if sf is False else "UNKNOWN")
    if sf is not True:
        all_pass = False
    print(f"  {r['label']:15s}: {status}  rc={r['rc']}  {sl}", flush=True)

print(f"\nOverall: {'ALL PASS' if all_pass else 'SOME FAILED'}", flush=True)
step("DONE", all_pass=all_pass, summary={
    r["label"]: {
        "rc": r["rc"],
        "elapsed": r["elapsed"],
        "stop_fired": r.get("stop_fired"),
    }
    for r in [res_a, res_b, res_c]
})
