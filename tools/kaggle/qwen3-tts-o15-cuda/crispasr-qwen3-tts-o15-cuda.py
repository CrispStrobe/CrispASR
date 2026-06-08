"""
CrispASR — Qwen3-TTS O15 CUDA validation (PLAN #52 step 4)

Tests whether the O15 (persistent code-predictor graph reuse) path works
on CUDA. O15 was reverted to default-OFF in commit 61c42bfb after a
GGML_ASSERT crash in ggml_backend_tensor_set on Jetson sm_87 (issue #56).
The ggml fork has been updated since; this kernel checks if the crash is
resolved on a Kaggle P100.

Plan:
  1. CUDA build of crispasr-cli.
  2. TTS synthesis with O15=OFF (baseline) — verify it produces audio.
  3. TTS synthesis with O15=ON — check for crash or correctness regression.
  4. Compare: both should produce valid WAV; ASR roundtrip both through
     parakeet to verify intelligibility.
  5. Benchmark: O15 ON vs OFF timing (code_pred overhead).

Build/report plumbing from the shared harness tools/kaggle/kaggle_harness.py.
enable_gpu=true in kernel-metadata.json.
"""

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

CRISPASR_REF = os.environ.get("CRISPASR_REF", "main")
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
print(f"[start] ref={CRISPASR_REF}", flush=True)
if REPO.exists():
    import shutil

    shutil.rmtree(REPO)
run(
    [
        "git", "clone", "--depth", "1", "--branch", CRISPASR_REF,
        "--recursive", CRISPASR_REPO, str(REPO),
    ]
)

sys.path.insert(0, os.path.join(str(REPO), "tools", "kaggle"))
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
        "-DCRISPASR_BUILD_TESTS=OFF",
    ]
    + kh.cuda_build_flags(arch)
    + kh.cache_and_link_flags()
)
run(cmake_args)
kh.step("cmake_done")
with kh.build_heartbeat("cmake.build"):
    kh.sh_with_progress(
        f"stdbuf -oL -eL cmake --build {BUILD} --target crispasr-cli"
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
    "qwen3-tts-12hz-0.6b-base-q8_0.gguf",
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


# ── Run TTS with O15=OFF and O15=ON ────────────────────────────────
def run_tts(label, o15_value, extra_env=None, timeout=300):
    """Run qwen3-tts synthesis and return dict with results."""
    kh.step(f"{label}.start")
    out_wav = WORK / f"tts-{label}.wav"
    if out_wav.exists():
        out_wav.unlink()

    env = {
        "QWEN3_TTS_O15": str(o15_value),
        "QWEN3_TTS_BENCH": "1",
    }
    if extra_env:
        env.update(extra_env)

    cmd = [
        str(CLI), "--backend", "qwen3-tts",
        "-m", str(tts_model),
        "--codec-model", str(tts_codec),
        "--tts", TTS_TEXT,
        "--tts-output", str(out_wav),
        "--seed", "42",
        "-v",
    ]
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

    # Extract bench lines
    bench_lines = [
        ln for ln in combined.splitlines()
        if "qwen3_tts:" in ln and ("ms" in ln or "bench" in ln.lower())
    ]

    # Extract ar_loop timing
    ar_match = re.search(
        r"ar_loop\s+([\d.]+)\s+ms\s+\((\d+)\s+frames?,\s+([\d.]+)\s+ms/frame\)",
        combined,
    )
    ms_per_frame = float(ar_match.group(3)) if ar_match else None
    n_frames = int(ar_match.group(2)) if ar_match else None

    # Extract code_pred bench
    cp_match = re.search(
        r"code_pred_kv bench.*?compute=([\d.]+)\s+ms", combined
    )
    cp_compute_ms = float(cp_match.group(1)) if cp_match else None

    print(f"\n{'='*64}", flush=True)
    print(
        f"Run: {label}  rc={rc}  elapsed={elapsed}s  "
        f"wav={'OK' if wav_exists else 'MISSING'}  "
        f"size={wav_size}",
        flush=True,
    )
    if ms_per_frame:
        print(f"  ar_loop: {ms_per_frame:.1f} ms/frame ({n_frames} frames)", flush=True)
    if cp_compute_ms:
        print(f"  code_pred compute: {cp_compute_ms:.1f} ms (15 calls)", flush=True)
    for bl in bench_lines[-10:]:
        print(f"  {bl.strip()}", flush=True)
    if rc != 0:
        print("  --- stderr tail ---", flush=True)
        for ln in combined.splitlines()[-30:]:
            print(f"   {ln}", flush=True)

    kh.step(
        f"{label}.done",
        rc=rc, elapsed=elapsed, wav_ok=wav_exists,
        wav_size=wav_size, ms_per_frame=ms_per_frame,
    )
    return {
        "label": label, "rc": rc, "wav_ok": wav_exists,
        "wav_size": wav_size, "ms_per_frame": ms_per_frame,
        "cp_compute_ms": cp_compute_ms, "wav_path": str(out_wav),
        "elapsed": elapsed,
    }


# Run baseline (O15=OFF) first
off = run_tts("o15_off", 0)

# Run O15=ON — the one that previously crashed on CUDA
on = run_tts("o15_on", 1)


# ── ASR roundtrip: transcribe both WAVs with parakeet ──────────────
def asr_roundtrip(label, wav_path, timeout=120):
    """Transcribe a WAV and return the text."""
    kh.step(f"asr_{label}.start")
    out_stem = WORK / f"asr-{label}"
    cmd = [
        str(CLI), "--backend", "parakeet",
        "-m", str(asr_model),
        "-f", wav_path,
        "-of", str(out_stem), "-otxt",
        "--no-prints",
    ]
    try:
        r = subprocess.run(
            cmd, env=os.environ, capture_output=True, text=True, timeout=timeout
        )
        txt_path = out_stem.with_suffix(".txt")
        text = (
            txt_path.read_text().strip()
            if txt_path.exists() and txt_path.stat().st_size > 0
            else ""
        )
    except subprocess.TimeoutExpired:
        text = ""
    kh.step(f"asr_{label}.done", chars=len(text))
    return text


asr_off = asr_roundtrip("o15_off", off["wav_path"]) if off["wav_ok"] else ""
asr_on = asr_roundtrip("o15_on", on["wav_path"]) if on["wav_ok"] else ""

# ── Summary ────────────────────────────────────────────────────────
print("\n" + "=" * 64, flush=True)
print(f"SUMMARY — qwen3-tts O15 CUDA test — {sha[:8]} on {gpu_name}", flush=True)
print("=" * 64, flush=True)
print(f"  O15=OFF: rc={off['rc']}  wav={'OK' if off['wav_ok'] else 'FAIL'}  "
      f"{off['ms_per_frame'] or '?'} ms/frame  {off['elapsed']}s wall", flush=True)
print(f"  O15=ON:  rc={on['rc']}  wav={'OK' if on['wav_ok'] else 'FAIL'}  "
      f"{on['ms_per_frame'] or '?'} ms/frame  {on['elapsed']}s wall", flush=True)
print(f"  ASR roundtrip OFF: {asr_off[:100]!r}", flush=True)
print(f"  ASR roundtrip ON:  {asr_on[:100]!r}", flush=True)

o15_works = on["rc"] == 0 and on["wav_ok"]
o15_faster = (
    on["ms_per_frame"] is not None
    and off["ms_per_frame"] is not None
    and on["ms_per_frame"] < off["ms_per_frame"]
)

if o15_works and o15_faster:
    print(f"\n  VERDICT: O15 WORKS on CUDA ({gpu_name}). "
          f"Speedup: {off['ms_per_frame']:.1f} -> {on['ms_per_frame']:.1f} ms/frame "
          f"({(1 - on['ms_per_frame']/off['ms_per_frame'])*100:.0f}% faster).", flush=True)
    print("  ACTION: flip QWEN3_TTS_O15 default back to ON.", flush=True)
elif o15_works:
    print(f"\n  VERDICT: O15 runs without crash but no speedup.", flush=True)
elif on["rc"] != 0:
    print(f"\n  VERDICT: O15 STILL CRASHES on CUDA (rc={on['rc']}). "
          f"Keep default OFF.", flush=True)
else:
    print(f"\n  VERDICT: O15 produced no audio. Investigate.", flush=True)

kh.step(
    "summary",
    o15_works=o15_works, o15_faster=o15_faster,
    off_ms=off["ms_per_frame"], on_ms=on["ms_per_frame"],
    sha=sha,
)
kh._push_progress_to_hf(force=True)
kh.step("script.end")
