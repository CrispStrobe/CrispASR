"""
CrispASR — Qwen3-TTS 1.7B small_to_mtp fold, CUDA validation (#161)

The 1.7B talker->code_predictor bridge projection (small_to_mtp) used to run
as 16 separate single-matmul GPU graphs per frame (2 at step-0 + 14 in the
codebook loop), each a full sched_reset + alloc_graph + compute + readback.
On discrete-VRAM GPUs those tiny dispatches/sync dominate the code_pred wall
(excosy's "one CPU core pegged, GPU idle" report).

Commit 816ab541 folds the projection into the code_pred graph (default ON for
1.7B). This kernel measures the win on real CUDA and confirms correctness:

  A: fused   (default)                  — projection inside the code_pred graph
  B: nofuse  (QWEN3_TTS_CP_MTP_NOFUSE=1) — old per-step external projection

For each: ar_breakdown code_pred ms, talker ms, ms/frame, total rtf, and an
ASR roundtrip (parakeet) to confirm the audio is still correct speech.

MUST use the 1.7B model — the 0.6B has no small_to_mtp, so the fold is a no-op
there. Reference-WAV voice path (the 1.7B base ICL path); the baked voice-pack
path is unrelated to this change.
"""

import os
import re
import shutil
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
# Medium-length text → ~80-130 frames, so per-frame dispatch overhead is well
# averaged (excosy's complaint is long inputs).
TTS_TEXT = (
    "Artificial intelligence has transformed the way we live and work, "
    "enabling machines to understand language, recognize images, and make "
    "complex decisions in real time across many different industries."
)


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
print(f"  disk: {shutil.disk_usage('/kaggle/working')}", flush=True)
Path("/kaggle/working/started.txt").write_text("started\n")

if REPO.exists():
    shutil.rmtree(REPO)
run(["git", "clone", "--depth", "1", "--branch", CRISPASR_REF, CRISPASR_REPO, str(REPO)])

sys.path.insert(0, os.path.join(str(REPO), "tools", "kaggle"))
try:
    import kaggle_harness as kh
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import kaggle_harness as kh

kh.init_progress()
kh.resolve_hf_token()

sha = subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip()
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
    cands = [c for c in BUILD.rglob("crispasr") if c.is_file() and os.access(c, os.X_OK)]
    assert cands, "crispasr binary not found after build"
    CLI = cands[0]
os.environ["LD_LIBRARY_PATH"] = f"{BUILD / 'src'}:{os.environ.get('LD_LIBRARY_PATH', '')}"
kh.step("build_done", cli=str(CLI))

# ── Download 1.7B qwen3-tts + tokenizer + parakeet for ASR roundtrip ──
kh.step("downloading models")
try:
    from huggingface_hub import hf_hub_download
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "huggingface_hub"])
    from huggingface_hub import hf_hub_download

token = os.environ.get("HF_TOKEN")
MODELS = WORK / "models"
MODELS.mkdir(exist_ok=True)

tts_model = Path(hf_hub_download(
    "cstr/qwen3-tts-1.7b-base-GGUF", "qwen3-tts-12hz-1.7b-base-q8_0.gguf",
    cache_dir=str(MODELS), token=token,
))
tts_codec = Path(hf_hub_download(
    "cstr/qwen3-tts-tokenizer-12hz-GGUF", "qwen3-tts-tokenizer-12hz.gguf",
    cache_dir=str(MODELS), token=token,
))
asr_model = Path(hf_hub_download(
    "cstr/parakeet-tdt-0.6b-v2-GGUF", "parakeet-tdt-0.6b-v2-q4_k.gguf",
    cache_dir=str(MODELS), token=token,
))
kh.step("models_downloaded")

# Voice reference (24 kHz) from the repo's jfk.wav.
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
        subprocess.run(["ffmpeg", "-y", "-i", str(voice_ref_16k), "-ar", "24000", str(voice_ref)],
                       capture_output=True, timeout=30)
REF_TEXT = ("And so my fellow Americans, ask not what your country can do for you, "
            "ask what you can do for your country.")


# ── Run TTS: fused (default) vs nofuse (QWEN3_TTS_CP_MTP_NOFUSE=1) ──
def run_tts(label, extra_env=None, timeout=600):
    kh.step(f"{label}.start")
    out_wav = WORK / f"tts-{label}.wav"
    if out_wav.exists():
        out_wav.unlink()
    env = {"QWEN3_TTS_BENCH": "1"}
    if extra_env:
        env.update(extra_env)
    cmd = [
        str(CLI), "--backend", "qwen3-tts",
        "-m", str(tts_model), "--codec-model", str(tts_codec),
        "--voice", str(voice_ref), "--ref-text", REF_TEXT,
        "--i-have-rights", "--no-spoken-disclaimer",
        "--tts", TTS_TEXT, "--tts-output", str(out_wav),
        "--seed", "42", "-v",
    ]
    t0 = time.time()
    try:
        r = subprocess.run(cmd, env={**os.environ, **env}, stdout=subprocess.PIPE,
                           stderr=subprocess.PIPE, text=True, timeout=timeout)
        rc, combined = r.returncode, r.stdout + "\n" + r.stderr
    except subprocess.TimeoutExpired as ex:
        rc = -1
        so = ex.stdout or ""
        se = ex.stderr or ""
        combined = (so.decode(errors="replace") if isinstance(so, bytes) else so) + "\n" + \
                   (se.decode(errors="replace") if isinstance(se, bytes) else se)
    elapsed = round(time.time() - t0, 1)
    (RESULTS / f"{label}_log.txt").write_text(combined)

    wav_ok = out_wav.exists() and out_wav.stat().st_size > 1000
    wav_size = out_wav.stat().st_size if out_wav.exists() else 0

    ar = re.search(r"ar_loop\s+([\d.]+)\s+ms\s+\((\d+)\s+frames?,\s+([\d.]+)\s+ms/frame\)", combined)
    ms_per_frame = float(ar.group(3)) if ar else None
    n_frames = int(ar.group(2)) if ar else None
    brk = re.search(r"ar_breakdown.*?code_pred=\s*([\d.]+)\s*ms.*?talker=\s*([\d.]+)\s*ms", combined)
    code_pred_ms = float(brk.group(1)) if brk else None
    talker_ms = float(brk.group(2)) if brk else None
    perf = re.search(r"perf —.*?total=([\d.]+)\s*ms\s+audio=([\d.]+)\s*s\s+rtf=([\d.]+)", combined)
    rtf = float(perf.group(3)) if perf else None
    fused_line = "fused into code_pred graph" in combined

    print(f"\n{'='*64}", flush=True)
    print(f"Run: {label}  rc={rc}  elapsed={elapsed}s  wav={'OK' if wav_ok else 'MISSING'} ({wav_size}B)", flush=True)
    print(f"  fused_log_line={fused_line}  frames={n_frames}  ms/frame={ms_per_frame}", flush=True)
    print(f"  ar_breakdown: code_pred={code_pred_ms} ms  talker={talker_ms} ms   rtf={rtf}", flush=True)
    if rc != 0:
        print("  --- log tail ---", flush=True)
        for ln in combined.splitlines()[-30:]:
            print(f"   {ln}", flush=True)
    kh.step(f"{label}.done", rc=rc, elapsed=elapsed, wav_ok=wav_ok,
            ms_per_frame=ms_per_frame, code_pred_ms=code_pred_ms, rtf=rtf)
    return {
        "label": label, "rc": rc, "wav_ok": wav_ok, "wav_path": str(out_wav),
        "ms_per_frame": ms_per_frame, "n_frames": n_frames,
        "code_pred_ms": code_pred_ms, "talker_ms": talker_ms, "rtf": rtf,
        "fused_line": fused_line, "elapsed": elapsed,
    }


fused = run_tts("fused")  # default ON for 1.7B
nofuse = run_tts("nofuse", {"QWEN3_TTS_CP_MTP_NOFUSE": "1"})


# ── ASR roundtrip both WAVs (parakeet, greedy to avoid #161 beam cost) ──
def asr_roundtrip(label, wav_path, timeout=180):
    kh.step(f"asr_{label}.start")
    out_stem = WORK / f"asr-{label}"
    cmd = [
        str(CLI), "--backend", "parakeet", "-m", str(asr_model),
        "-f", wav_path, "-of", str(out_stem), "-otxt", "-bs", "1", "--no-prints",
    ]
    try:
        subprocess.run(cmd, env=os.environ, capture_output=True, text=True, timeout=timeout)
        txt = out_stem.with_suffix(".txt")
        text = txt.read_text().strip() if txt.exists() and txt.stat().st_size > 0 else ""
    except subprocess.TimeoutExpired:
        text = ""
    kh.step(f"asr_{label}.done", chars=len(text))
    return text


asr_fused = asr_roundtrip("fused", fused["wav_path"]) if fused["wav_ok"] else ""
asr_nofuse = asr_roundtrip("nofuse", nofuse["wav_path"]) if nofuse["wav_ok"] else ""

# ── Summary ─────────────────────────────────────────────────────────
print("\n" + "=" * 64, flush=True)
print(f"SUMMARY — qwen3-tts 1.7B small_to_mtp fold — {sha[:8]} on {gpu_name}", flush=True)
print("=" * 64, flush=True)
for r in (nofuse, fused):
    print(f"  {r['label']:7s}: rc={r['rc']} wav={'OK' if r['wav_ok'] else 'FAIL'} "
          f"frames={r['n_frames']} {r['ms_per_frame']}ms/frame "
          f"code_pred={r['code_pred_ms']}ms talker={r['talker_ms']}ms rtf={r['rtf']} "
          f"({r['elapsed']}s wall)", flush=True)
print(f"  ASR nofuse: {asr_nofuse[:120]!r}", flush=True)
print(f"  ASR fused : {asr_fused[:120]!r}", flush=True)

both_ok = fused["wav_ok"] and nofuse["wav_ok"] and fused["rc"] == 0 and nofuse["rc"] == 0
faster = (fused["code_pred_ms"] is not None and nofuse["code_pred_ms"] is not None
          and fused["code_pred_ms"] < nofuse["code_pred_ms"])
if both_ok and faster:
    cp_gain = (1 - fused["code_pred_ms"] / nofuse["code_pred_ms"]) * 100
    mf_gain = ((1 - fused["ms_per_frame"] / nofuse["ms_per_frame"]) * 100
               if fused["ms_per_frame"] and nofuse["ms_per_frame"] else 0)
    print(f"\n  VERDICT: FOLD FASTER on {gpu_name}. code_pred "
          f"{nofuse['code_pred_ms']:.0f} -> {fused['code_pred_ms']:.0f} ms "
          f"({cp_gain:.0f}% less); ms/frame {mf_gain:.0f}% less.", flush=True)
    print(f"  fused must log the fuse line: {fused['fused_line']}", flush=True)
elif both_ok:
    print("\n  VERDICT: both produce audio but fused not measurably faster — inspect bench.", flush=True)
else:
    print("\n  VERDICT: a run failed — inspect logs.", flush=True)

kh.step("summary", both_ok=both_ok, faster=faster,
        nofuse_cp=nofuse["code_pred_ms"], fused_cp=fused["code_pred_ms"], sha=sha)
kh._push_progress_to_hf(force=True)
kh.step("script.end")
