# ─────────────────────────── cell 0 (markdown) ───────────────────────────
# # CrispASR — issue #126 CUDA conv_transpose_1d F16 verify
#
# Verifies the fix for GitHub issue #126: ggml-cuda conv_transpose_1d
# previously asserted `src0->type == GGML_TYPE_F32`, segfaulting on F16
# weights (SNAC decoder up_w, vibevoice generator) when running on CUDA.
#
# What it does:
# 1. Clone CrispASR @ env CRISPASR_REF (default: fix branch).
# 2. Build with -DGGML_CUDA=ON.
# 3. Run `crispasr --tts ... -m auto --backend kartoffel-orpheus-de-natural`
#    on a short German phrase; check output WAV is produced and non-silent.
# 4. Run `crispasr --tts ... -m auto --backend voxcpm2-tts` on a short
#    English phrase; same check.
# 5. Report PASS/FAIL per backend.
#
# Requirements: Kaggle GPU notebook (T4, P100, L4 — anything CUDA-capable).
# Disk: ~10 GB (CrispASR build ~1 GB + orpheus q8_0 ~3.5 GB + SNAC ~200 MB
# + voxcpm2 q4_k ~1.6 GB + deps).

# ─────────────────────────── cell 1 (code) ───────────────────────────
import json
import os
import shutil
import struct
import subprocess
import sys
import time
import wave
from datetime import datetime, timezone
from pathlib import Path

os.environ["PYTHONUNBUFFERED"] = "1"
try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except (AttributeError, ValueError):
    pass

WORK = Path("/kaggle/working")
REPO = WORK / "CrispASR"
BUILD = REPO / "build"
RESULTS = WORK / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

CRISPASR_REF = os.environ.get("CRISPASR_REF", "fix/issue126-cuda-conv-transpose-1d-f16")
CRISPASR_REPO = os.environ.get("CRISPASR_REPO", "https://github.com/CrispStrobe/CrispASR.git")

PROGRESS = RESULTS / "progress.jsonl"
_T0 = time.time()


def step(name, **kv):
    payload = {"t": round(time.time() - _T0, 2), "step": name, **kv}
    line = json.dumps(payload)
    print(f"[step] {line}", flush=True)
    with open(PROGRESS, "a") as f:
        f.write(line + "\n")


def run(cmd, check=True, capture=False, env=None, cwd=None, timeout=None):
    print(f"\n$ {' '.join(str(c) for c in cmd)}", flush=True)
    e = os.environ.copy()
    if env:
        e.update(env)
    if capture:
        r = subprocess.run(cmd, env=e, cwd=cwd, timeout=timeout,
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                           text=True)
        print(r.stdout, flush=True)
        if check and r.returncode != 0:
            raise SystemExit(f"command failed (rc={r.returncode}): {cmd}")
        return r
    r = subprocess.run(cmd, env=e, cwd=cwd, timeout=timeout)
    if check and r.returncode != 0:
        raise SystemExit(f"command failed (rc={r.returncode}): {cmd}")
    return r


# ─────────────────────────── cell 2 (code) — clone + build ───────────────
step("start", ref=CRISPASR_REF)

if REPO.exists():
    shutil.rmtree(REPO)
run(["git", "clone", "--depth", "1", "--branch", CRISPASR_REF, "--recursive",
     CRISPASR_REPO, str(REPO)])

# Harness lives in the cloned repo — import it now that the clone succeeded.
sys.path.insert(0, os.path.join(str(REPO), "tools", "kaggle"))
import kaggle_harness as kh  # noqa: E402

kh.init_progress()

step("cloned", sha=subprocess.check_output(
    ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip())

# Inspect GPU
run(["nvidia-smi", "-L"])
gpu_name = subprocess.check_output(
    ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True).strip()
step("gpu", name=gpu_name)

# Build with CUDA. Install ninja/ccache/mold first so the cache + linker
# launchers are available, then auto-detect the GPU arch (T4→75, A100→80…).
kh.install_build_toolchain()
arch = kh.detect_cuda_arch()
step("cuda_arch", arch=arch)

BUILD.mkdir(parents=True, exist_ok=True)
cmake_args = [
    "cmake", "-S", str(REPO), "-B", str(BUILD),
    "-DCMAKE_BUILD_TYPE=Release",
    "-DBUILD_SHARED_LIBS=ON",
] + kh.cuda_build_flags(arch) + kh.cache_and_link_flags()
run(cmake_args)
step("cmake_done")

# Just need crispasr CLI + libs; skip examples we don't use. Stream the
# build line-by-line with a heartbeat so a hang/OOM is visible mid-run.
with kh.build_heartbeat("cmake.build"):
    kh.sh_with_progress(
        f"stdbuf -oL -eL cmake --build {BUILD} --target crispasr "
        f"-j{kh.safe_build_jobs(gpu=True)}")
step("build_done")

CLI = BUILD / "examples" / "cli" / "crispasr"
if not CLI.exists():
    # Older layouts:
    candidates = list(BUILD.rglob("crispasr"))
    candidates = [c for c in candidates if c.is_file() and os.access(c, os.X_OK)]
    if not candidates:
        raise SystemExit("crispasr binary not found after build")
    CLI = candidates[0]
print(f"crispasr binary: {CLI}", flush=True)
step("cli_found", path=str(CLI))

# Ensure libcrispasr.so is loadable.
LIB_DIR = BUILD / "src"
os.environ["LD_LIBRARY_PATH"] = f"{LIB_DIR}:{os.environ.get('LD_LIBRARY_PATH', '')}"


# ─────────────────────────── cell 3 (code) — synthesis helpers ───────────
def wav_summary(path: Path) -> dict:
    """Return {'duration_s': float, 'rms': float, 'n_samples': int, 'sr': int}."""
    if not path.exists():
        return {"error": "missing"}
    with wave.open(str(path), "rb") as w:
        n = w.getnframes()
        sr = w.getframerate()
        sw = w.getsampwidth()
        ch = w.getnchannels()
        raw = w.readframes(n)
    if sw != 2:
        return {"error": f"sw={sw}"}
    # Sum of squares (mono assumed; if stereo, take channel 0).
    fmt = f"<{n * ch}h"
    pcm = struct.unpack(fmt, raw)
    if ch > 1:
        pcm = pcm[::ch]
    if not pcm:
        return {"duration_s": 0.0, "rms": 0.0, "n_samples": 0, "sr": sr}
    s2 = sum(int(x) * int(x) for x in pcm) / max(1, len(pcm))
    rms = (s2 ** 0.5) / 32768.0
    return {
        "duration_s": round(len(pcm) / sr, 3),
        "rms": round(rms, 6),
        "n_samples": len(pcm),
        "sr": sr,
    }


def try_synthesize(name: str, backend: str, text: str, out_wav: Path,
                   extra_args=None, timeout=900) -> dict:
    extra_args = extra_args or []
    cmd = [
        str(CLI),
        "--backend", backend,
        "-m", "auto",
        "--auto-download",
        "--tts", text,
        "--tts-output", str(out_wav),
        "--no-prints",
    ] + extra_args
    t0 = time.time()
    rc = None
    err_excerpt = ""
    try:
        r = subprocess.run(cmd, timeout=timeout,
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                           text=True)
        rc = r.returncode
        out = r.stdout
        # Save raw log
        (RESULTS / f"{name}.log").write_text(out)
        # Pull the last ~2 KB as a quick excerpt for the summary
        err_excerpt = out[-2048:]
        print(out, flush=True)
    except subprocess.TimeoutExpired as ex:
        rc = -1
        err_excerpt = f"TIMEOUT after {timeout}s\n{ex.stdout or ''}"
        print(err_excerpt, flush=True)

    dt = round(time.time() - t0, 2)
    wav = wav_summary(out_wav) if out_wav.exists() else {"error": "no-wav"}
    return {
        "name": name,
        "backend": backend,
        "rc": rc,
        "elapsed_s": dt,
        "wav": wav,
        "err_excerpt": err_excerpt[-1024:],
    }


def verdict(res: dict) -> str:
    if res["rc"] != 0:
        return "FAIL: crashed/non-zero rc"
    w = res["wav"]
    if "error" in w:
        return f"FAIL: wav {w['error']}"
    if w["duration_s"] < 0.2:
        return f"FAIL: wav too short ({w['duration_s']}s)"
    if w["rms"] < 1e-4:
        return f"FAIL: wav silent (rms={w['rms']})"
    return "PASS"


# ─────────────────────────── cell 4 (code) — run both backends ───────────
results = []

# Orpheus — the backend that crashed in the bug report.
step("orpheus_start")
r = try_synthesize(
    "orpheus",
    "kartoffel-orpheus-de-natural",
    "Hallo, dies ist ein kurzer Test auf der GPU.",
    RESULTS / "orpheus_de.wav",
    extra_args=["--voice", "Alexander", "--temperature", "0.6"],
)
r["verdict"] = verdict(r)
results.append(r)
step("orpheus_done", verdict=r["verdict"], rc=r["rc"], wav=r["wav"])

# Voxcpm2 — the second crashing backend (Q4_K LM, F32 VAE).
step("voxcpm2_start")
r = try_synthesize(
    "voxcpm2",
    "voxcpm2-tts",
    "Hello, this is a short test on the GPU.",
    RESULTS / "voxcpm2_en.wav",
)
r["verdict"] = verdict(r)
results.append(r)
step("voxcpm2_done", verdict=r["verdict"], rc=r["rc"], wav=r["wav"])


# ─────────────────────────── cell 5 (code) — summary ──────────────────────
summary = {
    "ts": datetime.now(timezone.utc).isoformat(),
    "ref": CRISPASR_REF,
    "gpu": gpu_name,
    "results": results,
    "verdicts": {r["name"]: r["verdict"] for r in results},
    "all_pass": all(r["verdict"] == "PASS" for r in results),
}
(RESULTS / "summary.json").write_text(json.dumps(summary, indent=2))
print("\n" + "=" * 60)
print(json.dumps(summary, indent=2))
print("=" * 60)

if not summary["all_pass"]:
    print("\nSOME BACKENDS FAILED — see logs in /kaggle/working/results/", flush=True)
    sys.exit(1)
print("\nALL BACKENDS PASS on", gpu_name, flush=True)
