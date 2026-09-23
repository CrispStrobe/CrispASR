"""
CrispASR — qwen3-tts #337 / #457 CUDA A/B: F16 code-predictor down projection.

The 0.6B-F16 code predictor's SwiGLU intermediate reaches ~156000 (> F16 max
65504). #457 promoted the F16 ffn_down weights to F32 on ROCm only; fix/337-
proper promotes on every backend whose F16 GEMM narrows activations (CUDA,
ROCm, Vulkan, SYCL). This kernel shows whether CUDA really needs it.

Matrix (F16 GGUF, seed 42, same text):
  cpu      -ng                                   reference (never narrows)
  nopromo  CUDA, CRISPASR_QWEN3_TTS_CP_F32_DOWN=0  the pre-fix CUDA path
  promo    CUDA, default policy (promotion on)
Plus the unit test binary. Each WAV: rc, RMS, NaN/Inf-silence check, ASR
roundtrip with parakeet. Also a second text that starts with a short prompt.
"""

import hashlib
import os
import re
import subprocess
import sys
import time
from pathlib import Path

WORK = Path("/kaggle/working")
REPO = WORK / "CrispASR"
BUILD = WORK / "build"
RESULTS = WORK / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

CRISPASR_REF = os.environ.get("CRISPASR_REF", "fix/337-proper")
CRISPASR_REPO = os.environ.get(
    "CRISPASR_REPO", "https://github.com/CrispStrobe/CrispASR.git"
)
TTS_TEXT = "Please call Stella. Ask her to bring these things with her from the store."


def run(cmd, check=True, env=None, timeout=None):
    print(f"\n$ {' '.join(str(c) for c in cmd)}", flush=True)
    e = os.environ.copy()
    if env:
        e.update(env)
    r = subprocess.run(cmd, env=e, timeout=timeout)
    if check and r.returncode != 0:
        raise SystemExit(f"command failed (rc={r.returncode}): {cmd}")
    return r


# ── Clone + CUDA build ──────────────────────────────────────────────
import shutil
print(f"[start] ref={CRISPASR_REF}", flush=True)
print(f"  disk: {shutil.disk_usage('/kaggle/working')}", flush=True)
Path("/kaggle/working/started.txt").write_text("started\n")

if REPO.exists():
    shutil.rmtree(REPO)
run(
    [
        "git", "clone", "--depth", "1", "--branch", CRISPASR_REF,
        "--recursive", CRISPASR_REPO, str(REPO),
    ]
)

sys.path.insert(0, os.path.join(str(REPO), "tools", "kaggle"))
try:
    import kaggle_harness as kh
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import kaggle_harness as kh

kh.init_progress()
kh.resolve_hf_token()

sha = subprocess.check_output(
    ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True
).strip()
kh.step("cloned", sha=sha, ref=CRISPASR_REF)

run(["nvidia-smi", "-L"])
gpu_name = subprocess.check_output(
    ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True
).strip()
kh.step("gpu", gpu_name=gpu_name)

kh.install_build_toolchain()
arch = kh.detect_cuda_arch()
kh.step("cuda_arch", arch=arch)

BUILD.mkdir(parents=True, exist_ok=True)
cmake_args = (
    [
        "cmake", "-S", str(REPO), "-B", str(BUILD),
        "-DCMAKE_BUILD_TYPE=Release", "-DBUILD_SHARED_LIBS=ON",
        "-DCRISPASR_BUILD_TESTS=ON",
    ]
    + kh.cuda_build_flags(arch)
    + kh.cache_and_link_flags()
)
run(cmake_args)
kh.step("cmake_done")
with kh.build_heartbeat("cmake.build"):
    kh.sh_with_progress(
        f"stdbuf -oL -eL cmake --build {BUILD} --target crispasr-cli --target test-qwen3tts-params"
        f" -j{kh.safe_build_jobs(gpu=True)}"
    )

CLI = BUILD / "bin" / "crispasr"
if not CLI.exists():
    cands = [
        c
        for c in BUILD.rglob("crispasr")
        if c.is_file() and os.access(c, os.X_OK)
    ]
    assert cands, "crispasr binary not found after build"
    CLI = cands[0]
os.environ["LD_LIBRARY_PATH"] = (
    f"{BUILD / 'src'}:{os.environ.get('LD_LIBRARY_PATH', '')}"
)
kh.step("build_done", cli=str(CLI))

# ── Download qwen3-tts model + tokenizer + parakeet for ASR roundtrip ──
kh.step("downloading models")
try:
    from huggingface_hub import hf_hub_download
except ImportError:
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q", "huggingface_hub"]
    )
    from huggingface_hub import hf_hub_download

token = os.environ.get("HF_TOKEN")
MODELS = WORK / "models"
MODELS.mkdir(exist_ok=True)

tts_model = Path(hf_hub_download(
    "cstr/qwen3-tts-0.6b-base-GGUF",
    "qwen3-tts-12hz-0.6b-base.gguf",
    cache_dir=str(MODELS), token=token,
))
tts_codec = Path(hf_hub_download(
    "cstr/qwen3-tts-tokenizer-12hz-GGUF",
    "qwen3-tts-tokenizer-12hz.gguf",
    cache_dir=str(MODELS), token=token,
))
asr_model = Path(hf_hub_download(
    "cstr/parakeet-tdt-0.6b-v2-GGUF",
    "parakeet-tdt-0.6b-v2-q4_k.gguf",
    cache_dir=str(MODELS), token=token,
))
kh.step("models_downloaded")


# ── Run TTS under one env config ────────────────────────────────────
def run_tts(label, env_overrides, timeout=900, extra=()):
    """Run qwen3-tts synthesis and return dict with results."""
    kh.step(f"{label}.start")
    out_wav = WORK / f"tts-{label}.wav"
    if out_wav.exists():
        out_wav.unlink()

    env = {"QWEN3_TTS_BENCH": "1"}
    env.update(env_overrides)

    # Use jfk.wav from the repo as voice reference (qwen3-tts requires 24kHz)
    voice_ref_16k = REPO / "samples" / "jfk.wav"
    voice_ref = WORK / "jfk_24k.wav"
    if not voice_ref.exists():
        try:
            import scipy.io.wavfile as swav
            from scipy.signal import resample_poly
            sr_in, data = swav.read(str(voice_ref_16k))
            if sr_in != 24000:
                data_24k = resample_poly(data.astype("float32"), 24000, sr_in)
                swav.write(str(voice_ref), 24000, data_24k.astype("int16"))
            else:
                shutil.copy(str(voice_ref_16k), str(voice_ref))
        except ImportError:
            subprocess.run(["ffmpeg", "-y", "-i", str(voice_ref_16k),
                            "-ar", "24000", str(voice_ref)],
                           capture_output=True, timeout=30)
    ref_text = "And so my fellow Americans, ask not what your country can do for you, ask what you can do for your country."

    cmd = [
        str(CLI), "--backend", "qwen3-tts",
        "-m", str(tts_model),
        "--codec-model", str(tts_codec),
        "--voice", str(voice_ref),
        "--ref-text", ref_text,
        "--i-have-rights",
        "--tts", TTS_TEXT,
        "--tts-output", str(out_wav),
        "--seed", "42",
        "-v",
    ] + list(extra)
    t0 = time.time()
    try:
        r = subprocess.run(
            cmd, env={**os.environ, **env},
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, timeout=timeout,
        )
        rc, stdout, stderr = r.returncode, r.stdout, r.stderr
    except subprocess.TimeoutExpired as ex:
        rc = -1
        stdout = (
            (ex.stdout or b"").decode(errors="replace")
            if isinstance(ex.stdout, bytes)
            else (ex.stdout or "")
        )
        stderr = (
            (ex.stderr or b"").decode(errors="replace")
            if isinstance(ex.stderr, bytes)
            else (ex.stderr or "")
        )
    elapsed = round(time.time() - t0, 1)
    combined = stdout + "\n" + stderr
    (RESULTS / f"{label}_log.txt").write_text(combined)

    wav_exists = out_wav.exists() and out_wav.stat().st_size > 1000
    wav_size = out_wav.stat().st_size if out_wav.exists() else 0
    wav_md5 = (
        hashlib.md5(out_wav.read_bytes()).hexdigest() if wav_exists else None
    )

    bench_lines = [
        ln for ln in combined.splitlines()
        if "qwen3_tts:" in ln and ("ms" in ln or "bench" in ln.lower())
    ]

    ar_match = re.search(
        r"ar_loop\s+([\d.]+)\s+ms\s+\((\d+)\s+frames?,\s+([\d.]+)\s+ms/frame\)",
        combined,
    )
    ms_per_frame = float(ar_match.group(3)) if ar_match else None
    n_frames = int(ar_match.group(2)) if ar_match else None

    # Direct-path health markers
    direct_active = "cp_direct active" in combined
    fell_back = ("using sched path" in combined
                 or "cp_direct compute failed" in combined)

    print(f"\n{'='*64}", flush=True)
    print(
        f"Run: {label}  rc={rc}  elapsed={elapsed}s  "
        f"wav={'OK' if wav_exists else 'MISSING'}  size={wav_size}  "
        f"md5={wav_md5}",
        flush=True,
    )
    if ms_per_frame:
        print(f"  ar_loop: {ms_per_frame:.1f} ms/frame ({n_frames} frames)", flush=True)
    if direct_active:
        print("  cp_direct: ACTIVE", flush=True)
    if fell_back:
        print("  cp_direct/bucket: FELL BACK to sched path", flush=True)
    for bl in bench_lines[-8:]:
        print(f"  {bl.strip()}", flush=True)
    if rc != 0:
        print("  --- output tail ---", flush=True)
        for ln in combined.splitlines()[-30:]:
            print(f"   {ln}", flush=True)

    kh.step(
        f"{label}.done",
        rc=rc, elapsed=elapsed, wav_ok=wav_exists,
        wav_size=wav_size, ms_per_frame=ms_per_frame, md5=wav_md5,
    )
    return {
        "label": label, "rc": rc, "wav_ok": wav_exists,
        "wav_size": wav_size, "ms_per_frame": ms_per_frame,
        "md5": wav_md5, "wav_path": str(out_wav), "elapsed": elapsed,
        "direct_active": direct_active, "fell_back": fell_back,
    }



import json
import numpy as np
import scipy.io.wavfile as swav

ut = subprocess.run([str(BUILD / "bin" / "test-qwen3tts-params")], capture_output=True, text=True)
print("unit tests rc", ut.returncode, ut.stdout[-600:], ut.stderr[-600:], flush=True)

MATRIX = [
    ("cpu", {}, ["-ng"]),
    ("nopromo", {"CRISPASR_QWEN3_TTS_CP_F32_DOWN": "0"}, []),
    ("promo", {}, []),
]
results = {}
for label, env_overrides, extra in MATRIX:
    r = run_tts(label, env_overrides, extra=extra)
    log = (RESULTS / f"{label}_log.txt").read_text()
    r["promoted"] = [ln for ln in log.splitlines() if "promoted" in ln]
    r["backend_lines"] = [ln for ln in log.splitlines() if "backend" in ln.lower()][:6]
    if r["wav_ok"]:
        sr, a = swav.read(r["wav_path"])
        a = a.astype(np.float64) / (32768.0 if a.dtype == np.int16 else 1.0)
        r["rms"] = float(np.sqrt(np.mean(a * a)))
        r["dur"] = len(a) / sr
    results[label] = r


def asr_roundtrip(label, wav_path, timeout=180):
    out_stem = WORK / f"asr-{label}"
    subprocess.run([str(CLI), "--backend", "parakeet", "-m", str(asr_model), "-f", wav_path,
                    "-of", str(out_stem), "-otxt", "--no-prints"], capture_output=True, text=True, timeout=timeout)
    p = out_stem.with_suffix(".txt")
    return p.read_text().strip() if p.exists() else ""


for label, r in results.items():
    r["asr"] = asr_roundtrip(label, r["wav_path"]) if r["wav_ok"] else ""

# PCM comparison of each CUDA run against CPU
def pcm(p):
    return swav.read(p)[1].astype(np.float64)
for label in ("nopromo", "promo"):
    r = results[label]
    if r["wav_ok"] and results["cpu"]["wav_ok"]:
        a, b = pcm(results["cpu"]["wav_path"]), pcm(r["wav_path"])
        n = min(len(a), len(b))
        r["cos_vs_cpu"] = float(np.dot(a[:n], b[:n]) / (np.linalg.norm(a[:n]) * np.linalg.norm(b[:n]) + 1e-12))
        r["len_ratio_vs_cpu"] = len(b) / max(1, len(a))

print("\n" + "=" * 64, flush=True)
print(f"SUMMARY #337 CUDA A/B — {sha[:8]} on {gpu_name}", flush=True)
for label, _, _ in MATRIX:
    r = results[label]
    print(f"  {label:8s} rc={r['rc']} wav={r['wav_ok']} dur={r.get('dur')} rms={r.get('rms')} "
          f"{r['ms_per_frame']} ms/frame cos_vs_cpu={r.get('cos_vs_cpu')} promoted={r['promoted']}", flush=True)
    print(f"           asr={r['asr']!r}", flush=True)
json.dump({"sha": sha, "gpu": gpu_name, "unit_rc": ut.returncode,
           "unit_tail": ut.stdout[-400:], "results": results}, open(RESULTS / "summary.json", "w"), indent=1, default=str)
