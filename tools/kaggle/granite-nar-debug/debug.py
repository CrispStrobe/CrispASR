#!/usr/bin/env python3
"""Debug granite-4.1-nar SIGABRT — capture full stderr."""
import json, os, subprocess, sys, time, shutil
from pathlib import Path

WORK = Path("/kaggle/working")
(WORK / "debug_results.json").write_text("{}")

print("=== granite-nar SIGABRT debug ===", flush=True)

# Clone + build
cdir = WORK / "CrispASR"
if not cdir.exists():
    subprocess.check_call(["git", "clone", "--depth", "1",
        "https://github.com/CrispStrobe/CrispASR.git", str(cdir)])

# HF token
for p in ["/kaggle/input/crispasr-hf-token/hf_token.txt",
          "/kaggle/input/datasets/chr1s4/crispasr-hf-token/hf_token.txt"]:
    if os.path.exists(p):
        os.environ["HF_TOKEN"] = open(p).read().strip()
        break

# Build
if not shutil.which("cmake"):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "cmake", "ninja"], check=False)
subprocess.run("apt-get update -qq && apt-get install -y cmake ninja-build g++ ccache 2>/dev/null || true",
               shell=True, capture_output=True)

bdir = cdir / "build"
cmake_args = ["-DCMAKE_BUILD_TYPE=Debug"]  # Debug build for better backtraces
if shutil.which("ninja"): cmake_args += ["-G", "Ninja"]
subprocess.check_call(["cmake", "-B", str(bdir)] + cmake_args, cwd=str(cdir))
subprocess.check_call(["cmake", "--build", str(bdir), "-j2"], cwd=str(cdir))
print("Build OK (Debug)", flush=True)

# Download model
from huggingface_hub import hf_hub_download
MDIR = WORK / "models"; MDIR.mkdir(exist_ok=True)
mpath = hf_hub_download("cstr/granite-speech-4.1-2b-nar-GGUF",
                         "granite-speech-4.1-2b-nar-q4_k.gguf",
                         cache_dir=str(MDIR/".hf"), local_dir=str(MDIR))
print(f"Model: {mpath}", flush=True)

CLI = str(bdir / "bin" / "crispasr")
JFK = str(cdir / "samples" / "jfk.wav")
results = {}

# Test 1: GPU
print("\n=== Test 1: GPU ===", flush=True)
r = subprocess.run([CLI, "--backend", "granite-4.1-nar", "-m", str(MDIR / "granite-speech-4.1-2b-nar-q4_k.gguf"),
                     "-f", JFK, "-v"], capture_output=True, text=True, timeout=300)
results["gpu"] = {"rc": r.returncode, "stdout": r.stdout[-500:], "stderr": r.stderr[-2000:]}
print(f"GPU rc={r.returncode}", flush=True)
print(f"stderr (last 1000):\n{r.stderr[-1000:]}", flush=True)

# Test 2: CPU only
print("\n=== Test 2: CPU ===", flush=True)
r = subprocess.run([CLI, "--backend", "granite-4.1-nar", "-m", str(MDIR / "granite-speech-4.1-2b-nar-q4_k.gguf"),
                     "-f", JFK, "-v", "--no-gpu"], capture_output=True, text=True, timeout=300)
results["cpu"] = {"rc": r.returncode, "stdout": r.stdout[-500:], "stderr": r.stderr[-2000:]}
print(f"CPU rc={r.returncode}", flush=True)
print(f"stderr (last 1000):\n{r.stderr[-1000:]}", flush=True)

# Test 3: granite-4.1-plus (should work — sanity check)
print("\n=== Test 3: granite-4.1-plus (sanity) ===", flush=True)
try:
    mpath2 = hf_hub_download("cstr/granite-speech-4.1-2b-plus-GGUF",
                              "granite-speech-4.1-2b-plus-q4_k.gguf",
                              cache_dir=str(MDIR/".hf"), local_dir=str(MDIR))
    r = subprocess.run([CLI, "--backend", "granite", "-m", str(MDIR / "granite-speech-4.1-2b-plus-q4_k.gguf"),
                         "-f", JFK, "-np"], capture_output=True, text=True, timeout=300)
    results["plus_sanity"] = {"rc": r.returncode, "transcript": r.stdout.strip()[-200:]}
    print(f"plus rc={r.returncode}: {r.stdout.strip()[-80:]}", flush=True)
except Exception as e:
    results["plus_sanity"] = {"error": str(e)}

(WORK / "debug_results.json").write_text(json.dumps(results, indent=2))
print("\nDONE", flush=True)
