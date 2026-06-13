#!/usr/bin/env python3
"""Verify #89 fix: parakeet-ja auto-chunk at 10s instead of 30s.

Builds CrispASR from the fix/parakeet-ja-chunk branch, downloads
parakeet-tdt-0.6b-ja Q4_K, and tests on reazon_baseball_14s x3 (42s).

Expected: with the fix, auto-chunk fires at 10s (not 30s), recovering
all 3 repetitions of the keyword. Without the fix, the 30s chunk window
loses content because the encoder collapses *inside* each 30s chunk.

Uses chr1s4 account on Kaggle T4/P100.
"""

import os
import subprocess
import sys
from pathlib import Path

WORK = Path("/kaggle/working")
os.chdir(WORK)

# ── Clone + harness ──────────────────────────────────────────────────
CRISPASR_URL = "https://github.com/CrispStrobe/CrispASR.git"
CRISPASR_BRANCH = "fix/parakeet-ja-chunk"
_CRISPASR_DIR = WORK / "CrispASR"

if not _CRISPASR_DIR.exists():
    try:
        subprocess.check_call([
            "git", "clone", "--depth", "1", "-b", CRISPASR_BRANCH,
            CRISPASR_URL, str(_CRISPASR_DIR)
        ])
        sys.path.insert(0, str(_CRISPASR_DIR / "tools" / "kaggle"))
    except Exception:
        pass

if str(_CRISPASR_DIR / "tools" / "kaggle") not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
import kaggle_harness as kh
kh.init_progress()

# ── Build ────────────────────────────────────────────────────────────
kh.install_build_toolchain()
build_dir = _CRISPASR_DIR / "build"
build_dir.mkdir(exist_ok=True)

with kh.build_heartbeat("cmake configure"):
    subprocess.check_call([
        "cmake", "-G", "Ninja", "-B", str(build_dir),
        "-DCMAKE_BUILD_TYPE=Release",
        "-DCMAKE_C_COMPILER_LAUNCHER=ccache",
        "-DCMAKE_CXX_COMPILER_LAUNCHER=ccache",
    ], cwd=str(_CRISPASR_DIR), env={**os.environ, "CCACHE_DIR": str(WORK / ".ccache")})

with kh.build_heartbeat("cmake build"):
    subprocess.check_call([
        "cmake", "--build", str(build_dir), "--target", "crispasr-cli", "-j4",
    ], env={**os.environ, "CCACHE_DIR": str(WORK / ".ccache")})

CRISPASR_BIN = build_dir / "bin" / "crispasr"
assert CRISPASR_BIN.exists(), f"Binary not found at {CRISPASR_BIN}"
print(f"Binary: {CRISPASR_BIN} ({CRISPASR_BIN.stat().st_size / 1e6:.1f} MB)")

# ── HF auth (3-tier: env → Kaggle Secret → dataset file) ────────────
hf_token = kh.resolve_hf_token()
model_dir = WORK / "models"
model_dir.mkdir(exist_ok=True)

MODEL_REPO = "cstr/parakeet-tdt-0.6b-ja-GGUF"
MODEL_FILE = "parakeet-tdt-0.6b-ja-q4_k.gguf"
model_path = model_dir / MODEL_FILE

if not model_path.exists():
    print(f"Downloading {MODEL_REPO}/{MODEL_FILE}...")
    subprocess.check_call([
        "pip", "install", "-q", "huggingface_hub"
    ])
    from huggingface_hub import hf_hub_download
    hf_hub_download(
        repo_id=MODEL_REPO,
        filename=MODEL_FILE,
        local_dir=str(model_dir),
    )
print(f"Model: {model_path} ({model_path.stat().st_size / 1e6:.1f} MB)")

# ── Download test fixture ────────────────────────────────────────────
FIXTURE_REPO = "cstr/crispasr-regression-fixtures"
FIXTURE_FILE = "parakeet-tdt-0.6b-ja/reazon_baseball_14s/audio.wav"
fixture_path = model_dir / "reazon_baseball_14s.wav"

if not fixture_path.exists():
    print(f"Downloading fixture {FIXTURE_FILE}...")
    from huggingface_hub import hf_hub_download
    dl_path = hf_hub_download(
        repo_id=FIXTURE_REPO,
        filename=FIXTURE_FILE,
        local_dir=str(model_dir),
    )
    # Move from nested path to flat
    import shutil
    shutil.copy2(dl_path, str(fixture_path))
print(f"Fixture: {fixture_path} ({fixture_path.stat().st_size / 1e6:.1f} MB)")

# ── Create 42s concatenated clip (3x 14s) ───────────────────────────
import wave
import struct

concat_path = WORK / "reazon_baseball_42s.wav"
if not concat_path.exists():
    with wave.open(str(fixture_path), "rb") as w:
        params = w.getparams()
        frames = w.readframes(w.getnframes())
    concat_frames = frames * 3
    with wave.open(str(concat_path), "wb") as w:
        w.setparams(params._replace(nframes=len(concat_frames) // (params.sampwidth * params.nchannels)))
        w.writeframes(concat_frames)
    duration = len(concat_frames) / (params.sampwidth * params.nchannels * params.framerate)
    print(f"Concatenated: {concat_path} ({duration:.1f}s)")

# ── Test 1: Default (should auto-chunk at 10s with fix) ──────────────
print("\n" + "="*70)
print("TEST 1: Default mode (auto-chunk should fire at 10s for JA model)")
print("="*70)

result = subprocess.run(
    [str(CRISPASR_BIN), "-m", str(model_path), "-f", str(concat_path), "-np"],
    capture_output=True, text=True, timeout=300,
)
print(f"stdout: {result.stdout.strip()}")
print(f"stderr (last 5 lines):")
for line in result.stderr.strip().split("\n")[-5:]:
    print(f"  {line}")

# Count occurrences of keyword
transcript_default = result.stdout.strip()

# ── Test 2: Explicit --chunk-seconds 10 (baseline) ───────────────────
print("\n" + "="*70)
print("TEST 2: Explicit --chunk-seconds 10")
print("="*70)

result = subprocess.run(
    [str(CRISPASR_BIN), "-m", str(model_path), "-f", str(concat_path),
     "--chunk-seconds", "10", "-np"],
    capture_output=True, text=True, timeout=300,
)
print(f"stdout: {result.stdout.strip()}")
transcript_chunk10 = result.stdout.strip()

# ── Test 3: --vad (gold standard) ────────────────────────────────────
print("\n" + "="*70)
print("TEST 3: --vad (gold standard)")
print("="*70)

result = subprocess.run(
    [str(CRISPASR_BIN), "-m", str(model_path), "-f", str(concat_path),
     "--vad", "-np"],
    capture_output=True, text=True, timeout=300,
)
print(f"stdout: {result.stdout.strip()}")
transcript_vad = result.stdout.strip()

# ── Test 4: Explicit --chunk-seconds 30 (old default, should be worse) ─
print("\n" + "="*70)
print("TEST 4: Explicit --chunk-seconds 30 (old default)")
print("="*70)

result = subprocess.run(
    [str(CRISPASR_BIN), "-m", str(model_path), "-f", str(concat_path),
     "--chunk-seconds", "30", "-np"],
    capture_output=True, text=True, timeout=300,
)
print(f"stdout: {result.stdout.strip()}")
transcript_chunk30 = result.stdout.strip()

# ── Test 5: Short clip (14s, should NOT chunk) ───────────────────────
print("\n" + "="*70)
print("TEST 5: Short clip (14s, should not auto-chunk)")
print("="*70)

result = subprocess.run(
    [str(CRISPASR_BIN), "-m", str(model_path), "-f", str(fixture_path), "-np"],
    capture_output=True, text=True, timeout=300,
)
print(f"stdout: {result.stdout.strip()}")
transcript_short = result.stdout.strip()

# ── Summary ──────────────────────────────────────────────────────────
print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"Default (10s auto):  {len(transcript_default):4d} chars")
print(f"Explicit chunk=10:   {len(transcript_chunk10):4d} chars")
print(f"VAD (gold):          {len(transcript_vad):4d} chars")
print(f"Explicit chunk=30:   {len(transcript_chunk30):4d} chars")
print(f"Short (14s, no chk): {len(transcript_short):4d} chars")
print()
print("PASS criteria: default ≈ chunk=10 >> chunk=30")
print(f"Default vs chunk=10 ratio: {len(transcript_default)/max(1,len(transcript_chunk10)):.2f}")
print(f"Default vs chunk=30 ratio: {len(transcript_default)/max(1,len(transcript_chunk30)):.2f}")

# Write results to file for download
with open(WORK / "results.txt", "w") as f:
    f.write(f"default: {transcript_default}\n\n")
    f.write(f"chunk10: {transcript_chunk10}\n\n")
    f.write(f"vad:     {transcript_vad}\n\n")
    f.write(f"chunk30: {transcript_chunk30}\n\n")
    f.write(f"short:   {transcript_short}\n\n")

print("\nDone. Results written to /kaggle/working/results.txt")
