# %% [markdown]
# # CrispASR — vibevoice-asr F16 vs Q4_K test on Kaggle T4
#
# Three users in HF discussion
# https://huggingface.co/cstr/vibevoice-asr-GGUF/discussions/1
# report Q4_K is bad. Sunknown specifically said "clear high quality
# speech without background noise transcribes to just '[Music]'" on
# CUDA. The F16 was just uploaded; this kernel:
#
#   1. Builds CrispASR-CLI CPU-only (fast; CUDA build takes 10+ h).
#   2. Downloads Q4_K (4.5 GB) first and transcribes samples/jfk.wav.
#   3. Downloads F16 (15.5 GB) if ≥60 min of wall-time remain.
#   4. Prints both transcripts side-by-side vs gold.
#
# CPU result isolates whether Q4_K is fundamentally broken (bad quant)
# vs a CUDA-specific regression.

# %% [code]
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

os.environ["PYTHONUNBUFFERED"] = "1"
os.environ["PYDEVD_DISABLE_FILE_VALIDATION"] = "1"  # suppress pydevd frozen-module noise
try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except (AttributeError, ValueError):
    pass

WORK = Path("/kaggle/working")
_PROGRESS = WORK / "progress.jsonl"
_T0 = time.time()
_HF_LAST_PUSH = 0.0
_HF_REPO = "cstr/crispasr-kaggle-progress"
_HF_PATH = (f"runs/{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
            f"-vibevoice-asr-bench.jsonl")

MAX_WALL_S = 8 * 3600  # Kaggle GPU limit is 9 h; leave 1 h margin


def _push_hf():
    global _HF_LAST_PUSH
    if time.time() - _HF_LAST_PUSH < 30:
        return
    if not os.environ.get("HF_TOKEN") or not _PROGRESS.exists():
        return
    try:
        from huggingface_hub import HfApi
        HfApi(token=os.environ["HF_TOKEN"]).upload_file(
            path_or_fileobj=str(_PROGRESS), path_in_repo=_HF_PATH,
            repo_id=_HF_REPO, repo_type="dataset",
            commit_message=f"progress @ {int(time.time()-_T0)}s")
        _HF_LAST_PUSH = time.time()
    except Exception:
        pass


def step(name, **extra):
    rec = {"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "elapsed_s": round(time.time() - _T0, 2), "step": name, **extra}
    _PROGRESS.parent.mkdir(parents=True, exist_ok=True)
    with _PROGRESS.open("a") as f:
        f.write(json.dumps(rec) + "\n")
    print(f"[step {rec['elapsed_s']:>7.1f}s] {name}" +
          (f"  {extra}" if extra else ""), flush=True)
    _push_hf()


step("script.start")

# Open build-log early so ALL noisy subprocesses (pip, apt, cmake, ninja)
# are redirected there. Unicode box-drawing from pip progress bars and ANSI
# codes from mold corrupt the notebook's cell-output JSON when papermill
# tries to serialize them → the "Invalid \uXXXX" crash on disk-save.
_BUILD_LOG = WORK / "build.log"
_BUILD_LOG.parent.mkdir(parents=True, exist_ok=True)
_blog = open(_BUILD_LOG, "a")

# %% [code]
try:
    from kaggle_secrets import UserSecretsClient
    tok = UserSecretsClient().get_secret("HF_TOKEN")
    os.environ["HF_TOKEN"] = tok
    os.environ["HUGGING_FACE_HUB_TOKEN"] = tok
    step("hf_token.loaded")
except Exception as exc:
    step("hf_token.missing", error=f"{type(exc).__name__}")

# %% [code]
step("install-deps.begin")
subprocess.check_call([
    sys.executable, "-m", "pip", "install", "--quiet",
    "huggingface_hub", "hf_transfer",
], stdout=_blog, stderr=_blog)
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
step("install-deps.done")

# %% [code]
# Download pre-built release binary — skips the 4-minute cmake/ninja build.
RELEASE = "v0.6.6"
TARBALL = f"crispasr-linux-x86_64.tar.gz"
RELEASE_URL = f"https://github.com/CrispStrobe/CrispASR/releases/download/{RELEASE}/{TARBALL}"
BIN_DIR = WORK / "bin"
BIN_DIR.mkdir(exist_ok=True)
CRISPASR = BIN_DIR / "crispasr"

step("binary-download.begin", release=RELEASE)
subprocess.check_call(
    f"wget -q {RELEASE_URL} -O /tmp/crispasr.tar.gz && "
    f"tar -xzf /tmp/crispasr.tar.gz -C {BIN_DIR} --strip-components=1",
    shell=True, stdout=_blog, stderr=_blog)
CRISPASR.chmod(0o755)
assert CRISPASR.is_file(), f"binary missing after download"
step("binary-download.done", binary=str(CRISPASR))

# Clone repo only for the jfk.wav sample (no build needed)
REPO = WORK / "CrispASR"
if not REPO.exists():
    subprocess.check_call(
        "git clone --depth 1 --filter=blob:none --no-checkout "
        f"https://github.com/CrispStrobe/CrispASR.git {REPO} && "
        f"git -C {REPO} checkout HEAD -- samples/",
        shell=True, stdout=_blog, stderr=_blog)
step("samples.ready")

# %% [code]
WAV = REPO / "samples" / "jfk.wav"
assert WAV.is_file(), f"sample WAV missing: {WAV}"
MODELS_DIR = WORK / "models"
MODELS_DIR.mkdir(exist_ok=True)
GOLD = ("And so, my fellow Americans, ask not what your country can do for you, "
        "ask what you can do for your country.")


def transcribe(gguf: Path, label: str) -> dict:
    step(f"transcribe-{label}.begin", gguf=gguf.name)
    t0 = time.time()
    try:
        proc = subprocess.run(
            [str(CRISPASR), "-m", str(gguf), "-f", str(WAV),
             "--backend", "vibevoice"],
            capture_output=True, text=True, timeout=600)
    except subprocess.TimeoutExpired:
        step(f"transcribe-{label}.timeout")
        return {"label": label, "transcript": None, "wallclock_s": 600,
                "stderr_tail": "TIMEOUT"}
    elapsed = time.time() - t0
    text_lines = [ln.strip() for ln in proc.stdout.splitlines() if ln.strip()]
    transcript = text_lines[-1] if text_lines else "(no stdout)"
    step(f"transcribe-{label}.done", wallclock_s=round(elapsed, 1),
         exit=proc.returncode, len_chars=len(transcript))
    return {
        "label": label,
        "transcript": transcript,
        "wallclock_s": round(elapsed, 1),
        "exit": proc.returncode,
        "stderr_tail": proc.stderr[-400:] if proc.returncode else "",
    }


# %% [code]
# ── Q4_K first (4.5 GB) ──────────────────────────────────────────────────────
from huggingface_hub import hf_hub_download

step("download-q4_k.begin")
q4k_path = hf_hub_download(
    repo_id="cstr/vibevoice-asr-GGUF",
    filename="vibevoice-asr-q4_k.gguf",
    local_dir=str(MODELS_DIR),
)
step("download-q4_k.done", size_gb=round(Path(q4k_path).stat().st_size / 1e9, 2))

r_q4k = transcribe(Path(q4k_path), "Q4_K")
results = [r_q4k]
# Print immediately so we have the transcript even if F16 download crashes
print(f"\nQ4_K ({r_q4k['wallclock_s']}s exit={r_q4k.get('exit','?')}): {r_q4k['transcript']!r}", flush=True)
print(f"GOLD: {GOLD!r}\n", flush=True)

# %% [code]
# ── F16 only if time AND disk space allow ────────────────────────────────────
import shutil as _shutil2
import warnings as _warnings

# Delete build object files to recover ~1 GB before F16 download check.
_cmake_files = BUILD / "CMakeFiles"
if _cmake_files.exists():
    _shutil2.rmtree(str(_cmake_files), ignore_errors=True)
    step("cleanup-build-objects")

elapsed_now = time.time() - _T0
remaining = MAX_WALL_S - elapsed_now
free_gb = _shutil2.disk_usage(str(WORK)).free / 1e9
F16_SIZE_GB = 16.7  # vibevoice-asr-f16.gguf is ~16.6 GB
step("time-check", elapsed_s=round(elapsed_now), remaining_s=round(remaining),
     free_gb=round(free_gb, 1))

if remaining >= 60 * 60 and free_gb >= F16_SIZE_GB + 0.5:
    step("download-f16.begin")
    with _warnings.catch_warnings():
        _warnings.simplefilter("ignore")
        f16_path = hf_hub_download(
            repo_id="cstr/vibevoice-asr-GGUF",
            filename="vibevoice-asr-f16.gguf",
            local_dir=str(MODELS_DIR),
        )
    step("download-f16.done", size_gb=round(Path(f16_path).stat().st_size / 1e9, 2))
    results.append(transcribe(Path(f16_path), "F16"))
elif free_gb < F16_SIZE_GB + 0.5:
    step("skip-f16", reason=f"disk full ({free_gb:.1f} GB free < {F16_SIZE_GB+0.5:.1f} GB needed)")
else:
    step("skip-f16", reason="insufficient time remaining")

# %% [code]
step("results.begin")
print("\n=== TRANSCRIPTS ===\n", flush=True)
for r in results:
    print(f"{r['label']:6s}  {r['wallclock_s']}s  exit={r.get('exit','?')}")
    print(f"        {r['transcript']!r}")
    if r.get('exit'):
        print(f"        stderr tail: {r['stderr_tail']}")
    print()

print(f"\nGOLD: {GOLD!r}\n")

RESULTS_PATH = WORK / "vibevoice-asr-bench.json"
RESULTS_PATH.write_text(json.dumps({
    "ts": datetime.now(timezone.utc).isoformat(),
    "results": results,
    "gold": GOLD,
}, indent=2))
print(f"wrote {RESULTS_PATH}")
step("results.done")
