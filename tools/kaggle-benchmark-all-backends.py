# ─────────────────────────── cell 0 (markdown) ───────────────────────────
# # CrispASR — Comprehensive Backend Benchmark on Kaggle
#
# Tests ALL CrispASR backends on Kaggle GPU/CPU, collecting:
# - Transcription accuracy (WER against reference)
# - Inference speed (realtime factor)
# - Model sizes (F16, Q4_K, Q8_0)
# - Memory usage
# - Output quality comparison
#
# **Requirements:**
# - Kaggle secret `HF_TOKEN` (read access — models are public)
# - Internet ON
# - Any accelerator (CPU, T4, P100 — benchmark adapts)
# - ~30 GB disk
#
# Results are saved as a GitHub Gist via `GH_GIST_TOKEN` secret (optional).

# ─────────────────────────── cell 1 (code) ───────────────────────────
# ── Configuration ──────────────────────────────────────────────────────────
import os, sys, time, json, subprocess, shutil
from datetime import datetime
from pathlib import Path

WORK = "/kaggle/working"
BUILD_DIR = f"{WORK}/CrispASR/build"
CRISPASR = f"{BUILD_DIR}/bin/crispasr"
QUANTIZE = f"{BUILD_DIR}/bin/crispasr-quantize"
RESULTS_DIR = f"{WORK}/results"
SAMPLE_DIR = f"{WORK}/samples"

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(SAMPLE_DIR, exist_ok=True)

# Load secrets
try:
    from kaggle_secrets import UserSecretsClient
    secrets = UserSecretsClient()
    HF_TOKEN = secrets.get_secret("HF_TOKEN")
    GH_GIST_TOKEN = secrets.get_secret("GH_GIST_TOKEN") if "GH_GIST_TOKEN" in dir(secrets) else None
except Exception:
    HF_TOKEN = os.environ.get("HF_TOKEN", "")
    GH_GIST_TOKEN = os.environ.get("GH_GIST_TOKEN", "")

os.environ["HF_TOKEN"] = HF_TOKEN
os.environ["HUGGING_FACE_HUB_TOKEN"] = HF_TOKEN
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

# Reference transcript for jfk.wav
JFK_REF = "and so my fellow americans ask not what your country can do for you ask what you can do for your country"

# All backends to test with their auto-download models
BACKENDS = [
    # (backend, display_name, timeout_seconds, notes)
    ("whisper",           "Whisper (base)",         60,  "ggml-base.bin"),
    ("parakeet",          "Parakeet TDT 0.6B",      60,  "Q4_K"),
    ("moonshine",         "Moonshine Tiny",          30,  "Q4_K, 27M params"),
    ("wav2vec2",          "Wav2Vec2 XLSR-EN",        60,  "Q4_K, 300M params"),
    ("fastconformer-ctc", "FastConformer CTC Large", 30,  "Q4_K, 120M params"),
    ("data2vec",          "Data2Vec Base",            30,  "Q4_K, 95M params"),
    ("hubert",            "HuBERT Large",             60,  "Q4_K, 300M params"),
    ("canary",            "Canary 1B",               120, "Q4_K, 1B params"),
    ("cohere",            "Cohere Transcribe",       120, "Q4_K, 2B params"),
    ("qwen3",             "Qwen3 ASR 0.6B",          60,  "Q4_K"),
    ("omniasr",           "OmniASR CTC 1B v2",      120, "Q4_K, 975M params"),
    ("omniasr-llm",       "OmniASR LLM 300M",       120, "Q4_K, 300M+1.3B params"),
    ("glm-asr",           "GLM ASR Nano",            180, "Q4_K, 1.3B params"),
    ("firered-asr",       "FireRed ASR2 AED",        300, "Q4_K, 900M params"),
    ("kyutai-stt",        "Kyutai STT 1B",           120, "Q4_K, 1B params"),
]

# Slow / large backends (only test if BENCHMARK_SLOW=1)
SLOW_BACKENDS = [
    ("voxtral",           "Voxtral Mini 3B",         300, "Q4_K, 3B params"),
    ("voxtral4b",         "Voxtral 4B Realtime",     300, "Q4_K, 4B params"),
    ("granite",           "Granite Speech 1B",       300, "Q4_K, 2.9B params"),
]

print(f"CrispASR Benchmark — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
print(f"Backends: {len(BACKENDS)} fast + {len(SLOW_BACKENDS)} slow")

# ─────────────────────────── cell 2 (code) ───────────────────────────
# ── Install dependencies ───────────────────────────────────────────────────
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "huggingface_hub", "hf_transfer", "jiwer"], check=True)
print("✓ Dependencies installed")

# ─────────────────────────── cell 3 (code) ───────────────────────────
# ── Clone and build CrispASR ───────────────────────────────────────────────
CRISPASR_DIR = f"{WORK}/CrispASR"

def run(cmd, timeout=600):
    """Run shell command, return (success, stdout, stderr, elapsed)."""
    t0 = time.time()
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True,
                           timeout=timeout)
        elapsed = time.time() - t0
        return r.returncode == 0, r.stdout, r.stderr, elapsed
    except subprocess.TimeoutExpired:
        return False, "", "TIMEOUT", time.time() - t0

if not os.path.isdir(CRISPASR_DIR):
    subprocess.run(f"git clone --depth 1 https://github.com/CrispStrobe/CrispASR.git {CRISPASR_DIR}",
                   shell=True, check=True)
    print("✓ CrispASR cloned")

# Detect GPU
has_gpu = os.path.exists("/usr/local/cuda/bin/nvcc")
gpu_flag = "-DGGML_CUDA=ON" if has_gpu else "-DGGML_CUDA=OFF"
print(f"GPU: {'CUDA detected' if has_gpu else 'CPU only'}")

os.makedirs(BUILD_DIR, exist_ok=True)
subprocess.run(
    f"cmake -S {CRISPASR_DIR} -B {BUILD_DIR} -DCMAKE_BUILD_TYPE=Release {gpu_flag}",
    shell=True, check=True
)
subprocess.run(f"cmake --build {BUILD_DIR} -j$(nproc)", shell=True, check=True)

assert os.path.isfile(CRISPASR), f"Build failed: {CRISPASR} not found"
print(f"✓ CrispASR built ({'GPU' if has_gpu else 'CPU'})")

# Check version
ok, out, err, _ = run(f"{CRISPASR} --help 2>&1 | head -3")
print(out[:200] if out else err[:200])

# ─────────────────────────── cell 4 (code) ───────────────────────────
# ── Download test audio ────────────────────────────────────────────────────
JFK_WAV = f"{SAMPLE_DIR}/jfk.wav"
if not os.path.isfile(JFK_WAV):
    # jfk.wav is in the CrispASR repo
    shutil.copy(f"{CRISPASR_DIR}/samples/jfk.wav", JFK_WAV)
    print(f"✓ jfk.wav copied ({os.path.getsize(JFK_WAV)} bytes)")

# Get audio duration
import wave
with wave.open(JFK_WAV) as wf:
    AUDIO_DURATION = wf.getnframes() / wf.getframerate()
print(f"  Duration: {AUDIO_DURATION:.1f}s")

# ─────────────────────────── cell 5 (code) ───────────────────────────
# ── Helper: compute WER ───────────────────────────────────────────────────
from jiwer import wer as compute_wer
import re

def normalize_text(text):
    """Normalize text for WER: lowercase, remove punctuation, collapse spaces."""
    text = text.lower().strip()
    text = re.sub(r"[^a-z ]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def calc_wer(ref, hyp):
    """Calculate word error rate between reference and hypothesis."""
    ref_norm = normalize_text(ref)
    hyp_norm = normalize_text(hyp)
    if not ref_norm or not hyp_norm:
        return 1.0
    return compute_wer(ref_norm, hyp_norm)

# ─────────────────────────── cell 6 (code) ───────────────────────────
# ── Run benchmark for each backend ─────────────────────────────────────────
import resource

results = []

def _cleanup_cache(backend_name):
    """Remove downloaded model files from cache, keeping whisper-tiny and silero VAD."""
    cache_dir = os.path.expanduser("~/.cache/crispasr")
    if not os.path.isdir(cache_dir):
        return
    freed = 0
    for f in os.listdir(cache_dir):
        fpath = os.path.join(cache_dir, f)
        if not os.path.isfile(fpath):
            continue
        # Keep whisper tiny (LID) and silero VAD — shared across backends
        if "ggml-tiny" in f or "silero" in f or "tokenizer" in f:
            continue
        sz = os.path.getsize(fpath) / 1024 / 1024
        os.remove(fpath)
        freed += sz
    if freed > 0:
        print(f"    Cleaned cache: {freed:.0f} MB freed")

def benchmark_backend(backend, display_name, timeout, notes):
    """Run a single backend benchmark. Returns result dict."""
    print(f"\n{'='*60}")
    print(f"  {display_name} (--backend {backend})")
    print(f"{'='*60}")

    result = {
        "backend": backend,
        "display_name": display_name,
        "notes": notes,
        "status": "UNKNOWN",
        "transcript": "",
        "wer": None,
        "elapsed_s": None,
        "realtime_factor": None,
        "model_size_mb": None,
    }

    # Step 1: Run transcription
    cmd = (f"{CRISPASR} --backend {backend} -m auto --auto-download "
           f"-f {JFK_WAV} --no-prints 2>/dev/null")
    t0 = time.time()
    ok, stdout, stderr, elapsed = run(cmd, timeout=timeout)
    result["elapsed_s"] = round(elapsed, 2)
    result["realtime_factor"] = round(AUDIO_DURATION / elapsed, 2) if elapsed > 0 else 0

    if not ok:
        result["status"] = "TIMEOUT" if "TIMEOUT" in stderr else "CRASH"
        print(f"  ✗ {result['status']} after {elapsed:.1f}s")
        # Still clean up any downloaded model to free disk
        _cleanup_cache(backend)
        return result

    # Step 2: Parse transcript
    transcript = stdout.strip()
    # Remove timestamp prefixes if present (e.g. [00:00:00.000 --> 00:00:11.000])
    transcript = re.sub(r"\[[\d:.]+\s*-->\s*[\d:.]+\]\s*", "", transcript).strip()
    result["transcript"] = transcript

    if not transcript:
        result["status"] = "EMPTY"
        print(f"  ✗ Empty output after {elapsed:.1f}s")
        _cleanup_cache(backend)
        return result

    # Step 3: Compute WER
    w = calc_wer(JFK_REF, transcript)
    result["wer"] = round(w, 4)
    result["status"] = "PASS" if w < 0.3 else "DEGRADED" if w < 0.5 else "FAIL"

    # Step 4: Find model size before cleanup
    cache_dir = os.path.expanduser("~/.cache/crispasr")
    if os.path.isdir(cache_dir):
        for f in sorted(os.listdir(cache_dir)):
            fpath = os.path.join(cache_dir, f)
            if os.path.isfile(fpath) and (f.endswith(".gguf") or f.endswith(".bin")):
                if "ggml-tiny" not in f and "silero" not in f:
                    result["model_size_mb"] = round(os.path.getsize(fpath) / 1024 / 1024, 1)
                    break

    status_icon = {"PASS": "✓", "DEGRADED": "~", "FAIL": "✗"}.get(result["status"], "?")
    sz_str = f"{result['model_size_mb']:.0f}MB" if result["model_size_mb"] else "?"
    print(f"  {status_icon} WER={w:.1%}  RT={result['realtime_factor']:.1f}x  "
          f"Time={elapsed:.1f}s  Model={sz_str}")
    print(f"    Output: {transcript[:100]}")

    # Step 5: Clean up model to free disk for the next backend
    _cleanup_cache(backend)

    return result

# Run all fast backends
for backend, name, timeout, notes in BACKENDS:
    r = benchmark_backend(backend, name, timeout, notes)
    results.append(r)

# Optionally run slow backends
if os.environ.get("BENCHMARK_SLOW", "0") == "1":
    for backend, name, timeout, notes in SLOW_BACKENDS:
        r = benchmark_backend(backend, name, timeout, notes)
        results.append(r)
else:
    print(f"\n⏭ Skipping {len(SLOW_BACKENDS)} slow backends "
          f"(set BENCHMARK_SLOW=1 to include)")

# ─────────────────────────── cell 7 (code) ───────────────────────────
# ── Format results table ───────────────────────────────────────────────────
import platform

# System info
sys_info = {
    "date": datetime.now().strftime("%Y-%m-%d %H:%M UTC"),
    "platform": platform.platform(),
    "cpu": platform.processor() or "unknown",
    "gpu": "CUDA" if has_gpu else "CPU-only",
    "python": platform.python_version(),
    "audio": f"jfk.wav ({AUDIO_DURATION:.1f}s)",
}

# Build markdown table
md_lines = []
md_lines.append("# CrispASR Backend Benchmark Results\n")
md_lines.append(f"**Date:** {sys_info['date']}  ")
md_lines.append(f"**Platform:** {sys_info['platform']}  ")
md_lines.append(f"**GPU:** {sys_info['gpu']}  ")
md_lines.append(f"**Audio:** {sys_info['audio']}  ")
md_lines.append(f"**Reference:** _{JFK_REF}_\n")

md_lines.append("| # | Backend | Status | WER | RT Factor | Time (s) | Model (MB) | Transcript |")
md_lines.append("|---|---|---|---|---|---|---|---|")

for i, r in enumerate(results, 1):
    status = {"PASS": "✅", "DEGRADED": "⚠️", "FAIL": "❌",
              "CRASH": "💥", "TIMEOUT": "⏱️", "EMPTY": "🔇"}.get(r["status"], "❓")
    wer_str = f"{r['wer']:.1%}" if r["wer"] is not None else "—"
    rt_str = f"{r['realtime_factor']:.1f}x" if r["realtime_factor"] else "—"
    time_str = f"{r['elapsed_s']:.1f}" if r["elapsed_s"] else "—"
    sz_str = f"{r['model_size_mb']:.0f}" if r["model_size_mb"] else "—"
    transcript = r["transcript"][:60] + "..." if len(r["transcript"]) > 60 else r["transcript"]
    transcript = transcript.replace("|", "\\|")

    md_lines.append(
        f"| {i} | **{r['display_name']}** | {status} | {wer_str} | {rt_str} | "
        f"{time_str} | {sz_str} | {transcript} |"
    )

# Summary stats
passed = sum(1 for r in results if r["status"] == "PASS")
total = len(results)
md_lines.append(f"\n**Summary:** {passed}/{total} passed, "
                f"{sum(1 for r in results if r['status'] == 'DEGRADED')} degraded, "
                f"{sum(1 for r in results if r['status'] in ('FAIL', 'CRASH', 'TIMEOUT'))} failed\n")

# Speed ranking
speed_results = [r for r in results if r["realtime_factor"] and r["status"] in ("PASS", "DEGRADED")]
if speed_results:
    speed_results.sort(key=lambda r: r["realtime_factor"], reverse=True)
    md_lines.append("## Speed Ranking (fastest first)\n")
    for i, r in enumerate(speed_results, 1):
        md_lines.append(f"{i}. **{r['display_name']}** — {r['realtime_factor']:.1f}x realtime "
                       f"(WER {r['wer']:.1%})")

# Quality ranking
quality_results = [r for r in results if r["wer"] is not None]
if quality_results:
    quality_results.sort(key=lambda r: r["wer"])
    md_lines.append("\n## Quality Ranking (lowest WER first)\n")
    for i, r in enumerate(quality_results, 1):
        md_lines.append(f"{i}. **{r['display_name']}** — WER {r['wer']:.1%} "
                       f"({r['realtime_factor']:.1f}x RT)")

report_md = "\n".join(md_lines)
print(report_md)

# ─────────────────────────── cell 8 (code) ───────────────────────────
# ── Save results ───────────────────────────────────────────────────────────
# Save JSON
json_path = f"{RESULTS_DIR}/benchmark_results.json"
with open(json_path, "w") as f:
    json.dump({"system": sys_info, "results": results}, f, indent=2)
print(f"✓ JSON saved to {json_path}")

# Save Markdown
md_path = f"{RESULTS_DIR}/benchmark_results.md"
with open(md_path, "w") as f:
    f.write(report_md)
print(f"✓ Markdown saved to {md_path}")

# ─────────────────────────── cell 9 (code) ───────────────────────────
# ── Upload results as GitHub Gist (optional) ───────────────────────────────
if GH_GIST_TOKEN:
    import urllib.request

    gist_data = {
        "description": f"CrispASR Benchmark — {sys_info['date']} ({sys_info['gpu']})",
        "public": True,
        "files": {
            "benchmark_results.md": {"content": report_md},
            "benchmark_results.json": {"content": json.dumps(
                {"system": sys_info, "results": results}, indent=2)},
        }
    }

    req = urllib.request.Request(
        "https://api.github.com/gists",
        data=json.dumps(gist_data).encode(),
        headers={
            "Authorization": f"token {GH_GIST_TOKEN}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req) as resp:
            gist_resp = json.loads(resp.read())
            print(f"✓ Gist created: {gist_resp['html_url']}")
    except Exception as e:
        print(f"✗ Gist upload failed: {e}")
else:
    print("⏭ No GH_GIST_TOKEN — skipping gist upload")
    print("  Set a Kaggle secret named GH_GIST_TOKEN to auto-upload results")

# ─────────────────────────── cell 10 (code) ───────────────────────────
# ── Final summary ──────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  CrispASR Benchmark Complete")
print("=" * 60)
print(f"  Backends tested: {len(results)}")
print(f"  Passed: {sum(1 for r in results if r['status'] == 'PASS')}")
print(f"  Fastest: {max((r for r in results if r.get('realtime_factor')), key=lambda r: r['realtime_factor'], default={})}")
print(f"  Best WER: {min((r for r in results if r.get('wer') is not None), key=lambda r: r['wer'], default={})}")
print(f"\n  Results: {RESULTS_DIR}/")
print("=" * 60)
