#!/usr/bin/env python3
"""
CrispASR — All-backends regression test (smoke / full).

Goal: verify each shipped backend still produces a usable transcript on
JFK, with a clear pass/fail per backend. Sister to
`tools/macbook-benchmark-all-backends.py` (which is a perf benchmark,
not a regression gate).

Tier model — designed to grow:

  --tier=smoke   (default) — transcript present + WER <= 0.20 vs JFK
                   reference. Fast smoke; one run per backend.
  --tier=full    — same as smoke for now. Future: per-feature checks
                   (timestamps, beam, best-of-N, temperature, VAD,
                   streaming round-trip) gated by --capabilities.

Default model dir: /Volumes/backups/ai/crispasr-models/ (macOS SSD).
Override with --models or set CRISPASR_MODELS_DIR env. Defaults to Q4_K
GGUFs; override per-backend via the registry if needed.

If a model isn't on disk, the script tries `huggingface_hub.hf_hub_download`
unless `--skip-missing` is set. HF_TOKEN env is picked up automatically
by huggingface_hub if present.

Pre-download: checks free disk space on the target volume. Skips if
not enough room (default safety margin: 2 GB above the file size).

Usage:
  python tools/test-all-backends.py
  python tools/test-all-backends.py --skip-missing
  python tools/test-all-backends.py --backends whisper,parakeet,kyutai-stt
  python tools/test-all-backends.py --models ~/.cache/crispasr
  python tools/test-all-backends.py --tier full

Exit code: 0 if all selected backends PASS, non-zero otherwise.
"""

from __future__ import annotations

import argparse
import os
import platform
import re
import shutil
import subprocess
import sys
import time
import wave
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
JFK_WAV = REPO_ROOT / "samples" / "jfk.wav"
JFK_REF = (
    "and so my fellow americans ask not what your country can do for you "
    "ask what you can do for your country"
)


def find_crispasr() -> Path | None:
    """Locate the crispasr binary. Project convention: build-ninja-compile/
    on dev macOS, build/ on most CI / Ubuntu."""
    for rel in (
        "build-ninja-compile/bin/crispasr",
        "build/bin/crispasr",
        "build-release/bin/crispasr",
    ):
        p = REPO_ROOT / rel
        if p.is_file():
            return p
    # Fall back to PATH
    found = shutil.which("crispasr")
    return Path(found) if found else None


# ---------------------------------------------------------------------------
# Backend registry — Q4_K by default. Extend as new backends ship.
# ---------------------------------------------------------------------------


@dataclass
class Backend:
    name: str            # crispasr --backend value
    display: str         # human label for output
    local_file: str      # filename to look for in --models dir
    hf_repo: str         # HF repo id for download fallback
    hf_file: str         # filename within the repo (often == local_file)
    timeout_s: int = 90  # subprocess timeout for one transcribe
    capabilities: tuple[str, ...] = ("transcribe",)
    notes: str = ""
    extra_files: tuple[tuple[str, str, str], ...] = ()  # (local, repo, name)
    # Hint: rough Q4_K size in MB, used for disk-space budgeting before download.
    # Fall back to "ask HF" if not specified.
    approx_size_mb: int | None = None


REGISTRY: tuple[Backend, ...] = (
    Backend("whisper",    "Whisper (base)",      "ggml-base.bin",
            "ggerganov/crispasr", "ggml-base.bin",
            timeout_s=60, approx_size_mb=150),
    Backend("parakeet",   "Parakeet TDT 0.6B",   "parakeet-tdt-0.6b-v3-q4_k.gguf",
            "cstr/parakeet-tdt-0.6b-v3-GGUF", "parakeet-tdt-0.6b-v3-q4_k.gguf",
            timeout_s=60, approx_size_mb=420,
            capabilities=("transcribe", "word-timestamps")),
    Backend("moonshine",  "Moonshine Tiny",      "moonshine-tiny-q4_k.gguf",
            "cstr/moonshine-tiny-GGUF", "moonshine-tiny-q4_k.gguf",
            timeout_s=30, approx_size_mb=30,
            extra_files=(("tokenizer.bin", "cstr/moonshine-tiny-GGUF", "tokenizer.bin"),)),
    Backend("moonshine-streaming", "Moonshine Streaming Tiny",
            "moonshine-streaming-tiny-f16.gguf",
            "cstr/moonshine-streaming-tiny-GGUF", "moonshine-streaming-tiny-f16.gguf",
            timeout_s=60, approx_size_mb=85,
            capabilities=("transcribe", "stream")),
    Backend("wav2vec2",   "Wav2Vec2 XLSR-EN",    "wav2vec2-xlsr-en-q4_k.gguf",
            "cstr/wav2vec2-large-xlsr-53-english-GGUF",
            "wav2vec2-xlsr-en-q4_k.gguf",
            timeout_s=60, approx_size_mb=200),
    Backend("fastconformer-ctc", "FastConformer CTC Large",
            "stt-en-fastconformer-ctc-large-q4_k.gguf",
            "cstr/stt-en-fastconformer-ctc-large-GGUF",
            "stt-en-fastconformer-ctc-large-q4_k.gguf",
            timeout_s=30, approx_size_mb=80),
    Backend("canary",     "Canary 1B",           "canary-1b-v2-q4_k.gguf",
            "cstr/canary-1b-v2-GGUF", "canary-1b-v2-q4_k.gguf",
            timeout_s=120, approx_size_mb=620),
    Backend("cohere",     "Cohere Transcribe",   "cohere-transcribe-q4_k.gguf",
            "cstr/cohere-transcribe-03-2026-GGUF", "cohere-transcribe-q4_k.gguf",
            timeout_s=120, approx_size_mb=1300),
    Backend("qwen3",      "Qwen3 ASR 0.6B",      "qwen3-asr-0.6b-q4_k.gguf",
            "cstr/qwen3-asr-0.6b-GGUF", "qwen3-asr-0.6b-q4_k.gguf",
            timeout_s=60, approx_size_mb=400),
    Backend("omniasr",    "OmniASR CTC 1B v2",   "omniasr-ctc-1b-v2-q4_k.gguf",
            "cstr/omniASR-CTC-1B-v2-GGUF", "omniasr-ctc-1b-v2-q4_k.gguf",
            timeout_s=120, approx_size_mb=620),
    Backend("omniasr-llm", "OmniASR LLM 300M",   "omniasr-llm-300m-v2-q4_k.gguf",
            "cstr/omniasr-llm-300m-v2-GGUF", "omniasr-llm-300m-v2-q4_k.gguf",
            timeout_s=120, approx_size_mb=1100,
            capabilities=("transcribe", "beam", "best-of-n", "temperature")),
    Backend("glm-asr",    "GLM ASR Nano",        "glm-asr-nano-q4_k.gguf",
            "cstr/glm-asr-nano-GGUF", "glm-asr-nano-q4_k.gguf",
            timeout_s=90, approx_size_mb=900,
            capabilities=("transcribe", "beam", "best-of-n", "temperature")),
    Backend("firered-asr", "FireRed ASR2 AED",   "firered-asr2-aed-q4_k.gguf",
            "cstr/firered-asr2-aed-GGUF", "firered-asr2-aed-q4_k.gguf",
            timeout_s=90, approx_size_mb=600),
    Backend("kyutai-stt", "Kyutai STT 1B",       "kyutai-stt-1b-q4_k.gguf",
            "cstr/kyutai-stt-1b-GGUF", "kyutai-stt-1b-q4_k.gguf",
            timeout_s=90, approx_size_mb=700,
            capabilities=("transcribe", "stream", "beam", "best-of-n",
                          "temperature", "word-timestamps")),
    Backend("granite",    "Granite Speech 1B",   "granite-speech-4.0-1b-q4_k.gguf",
            "cstr/granite-speech-4.0-1b-GGUF", "granite-speech-4.0-1b-q4_k.gguf",
            timeout_s=300, approx_size_mb=1700),
    Backend("granite-4.1", "Granite Speech 4.1 2B", "granite-speech-4.1-2b-q4_k.gguf",
            "cstr/granite-speech-4.1-2b-GGUF", "granite-speech-4.1-2b-q4_k.gguf",
            timeout_s=300, approx_size_mb=1500),
    Backend("vibevoice",  "VibeVoice ASR",       "vibevoice-asr-7b-q4_k-fixed.gguf",
            "cstr/vibevoice-asr-GGUF", "vibevoice-asr-q4_k.gguf",
            timeout_s=120, approx_size_mb=4500),
    Backend("voxtral",    "Voxtral Mini 3B",     "voxtral-mini-3b-2507-q4_k.gguf",
            "cstr/voxtral-mini-3b-2507-GGUF", "voxtral-mini-3b-2507-q4_k.gguf",
            timeout_s=300, approx_size_mb=1900),
)


# ---------------------------------------------------------------------------
# Disk space + download helpers
# ---------------------------------------------------------------------------


def free_mb(path: Path) -> int:
    """Free space in MB on the volume containing `path`."""
    p = path if path.exists() else path.parent
    return shutil.disk_usage(p).free // (1024 * 1024)


def fetch_model(b: Backend, models_dir: Path, skip_missing: bool,
                space_margin_mb: int = 2048) -> Path | None:
    """Locate the model on disk; download from HF if missing and allowed."""
    for cand in (b.local_file, b.hf_file):
        p = models_dir / cand
        if p.is_file():
            return p

    if skip_missing:
        return None

    needed_mb = (b.approx_size_mb or 0) + space_margin_mb
    have_mb = free_mb(models_dir)
    if b.approx_size_mb and have_mb < needed_mb:
        print(f"    skip download: need ~{needed_mb} MB, only {have_mb} MB free on {models_dir}")
        return None

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("    huggingface_hub not installed — pip install huggingface_hub hf_xet")
        return None

    print(f"    downloading {b.hf_file} from {b.hf_repo}…", flush=True)
    t0 = time.time()
    try:
        downloaded = hf_hub_download(b.hf_repo, b.hf_file, local_dir=str(models_dir))
    except Exception as e:
        print(f"    download failed: {e}")
        return None
    elapsed = time.time() - t0
    sz_mb = os.path.getsize(downloaded) / 1024 / 1024
    print(f"    ✓ {sz_mb:.0f} MB in {elapsed:.1f}s")

    for ex_local, ex_repo, ex_file in b.extra_files:
        if not (models_dir / ex_local).is_file():
            try:
                hf_hub_download(ex_repo, ex_file, local_dir=str(models_dir))
            except Exception as e:
                print(f"    extra file {ex_file} failed: {e} (continuing)")

    return Path(downloaded)


# ---------------------------------------------------------------------------
# Transcript test
# ---------------------------------------------------------------------------


def normalize(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z ]", "", s.lower())).strip()


def wer(ref: str, hyp: str) -> float | None:
    try:
        from jiwer import wer as compute_wer
    except ImportError:
        return None
    r, h = normalize(ref), normalize(hyp)
    if not r or not h:
        return 1.0
    return compute_wer(r, h)


@dataclass
class Result:
    backend: str
    display: str
    status: str  # PASS / FAIL / SKIP / NO_MODEL / TIMEOUT / CRASH / EMPTY
    wall_s: float = 0.0
    transcript: str = ""
    wer: float | None = None
    detail: str = ""


def run_transcribe(crispasr: Path, model: Path, b: Backend,
                   audio: Path, use_gpu: bool) -> Result:
    cmd = [str(crispasr), "--backend", b.name, "-m", str(model),
           "-f", str(audio), "--no-prints"]
    if not use_gpu:
        cmd.append("-ng")
    t0 = time.time()
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=b.timeout_s)
    except subprocess.TimeoutExpired:
        return Result(b.name, b.display, "TIMEOUT", b.timeout_s,
                      detail=f"subprocess timed out after {b.timeout_s}s")
    elapsed = time.time() - t0
    if r.returncode != 0:
        return Result(b.name, b.display, "CRASH", elapsed,
                      detail=(r.stderr or "")[-300:])
    transcript = re.sub(r"\[[\d:.]+\s*-->\s*[\d:.]+\]\s*", "", r.stdout.strip()).strip()
    if not transcript:
        return Result(b.name, b.display, "EMPTY", elapsed,
                      detail=(r.stderr or "")[-200:])
    w = wer(JFK_REF, transcript)
    return Result(b.name, b.display, "OK", elapsed, transcript=transcript, wer=w)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def smoke(b: Backend, crispasr: Path, models_dir: Path, audio: Path,
          skip_missing: bool, use_gpu: bool, wer_threshold: float) -> Result:
    print(f"\n[{b.name}] {b.display}")
    model = fetch_model(b, models_dir, skip_missing)
    if not model:
        print(f"    SKIP (no model on disk{', --skip-missing set' if skip_missing else ''})")
        return Result(b.name, b.display, "NO_MODEL")

    print(f"    model: {model.name} ({os.path.getsize(model)/1024/1024:.0f} MB)", flush=True)
    r = run_transcribe(crispasr, model, b, audio, use_gpu)
    if r.status != "OK":
        print(f"    FAIL ({r.status}): {r.detail[:160]}")
        return Result(b.name, b.display, "FAIL", r.wall_s, r.transcript, r.wer, r.detail)

    if r.wer is None:
        print(f"    PASS (transcript present, jiwer not installed → no WER)")
        print(f"    out: {r.transcript[:80]}")
        return Result(b.name, b.display, "PASS", r.wall_s, r.transcript, None, "jiwer missing")
    if r.wer > wer_threshold:
        print(f"    FAIL (WER {r.wer:.1%} > {wer_threshold:.0%})")
        print(f"    out: {r.transcript[:80]}")
        return Result(b.name, b.display, "FAIL", r.wall_s, r.transcript, r.wer,
                      f"WER {r.wer:.4f} above threshold {wer_threshold}")
    print(f"    PASS  WER={r.wer:.1%}  wall={r.wall_s:.1f}s")
    print(f"    out: {r.transcript[:80]}")
    return Result(b.name, b.display, "PASS", r.wall_s, r.transcript, r.wer)


def parse_capabilities(s: str | None) -> set[str]:
    if not s:
        return set()
    return {c.strip() for c in s.split(",") if c.strip()}


def select_backends(args) -> list[Backend]:
    if args.backends:
        wanted = {n.strip() for n in args.backends.split(",")}
        sel = [b for b in REGISTRY if b.name in wanted]
        missing = wanted - {b.name for b in sel}
        if missing:
            print(f"WARNING: unknown backends in --backends: {sorted(missing)}", file=sys.stderr)
        return sel
    caps = parse_capabilities(args.capabilities)
    if caps:
        return [b for b in REGISTRY if caps & set(b.capabilities)]
    return list(REGISTRY)


def main() -> int:
    default_models = os.environ.get(
        "CRISPASR_MODELS_DIR",
        "/Volumes/backups/ai/crispasr-models" if platform.system() == "Darwin"
        else str(Path.home() / ".cache" / "crispasr"),
    )

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--models", default=default_models,
                    help=f"Model directory (default: {default_models})")
    ap.add_argument("--audio", default=str(JFK_WAV),
                    help="Audio file to transcribe (default: samples/jfk.wav)")
    ap.add_argument("--backends", default=None,
                    help="Comma-separated subset of backends (default: all in registry)")
    ap.add_argument("--capabilities", default=None,
                    help="Comma-separated capability filter (e.g. stream,beam)")
    ap.add_argument("--tier", choices=("smoke", "full"), default="smoke",
                    help="Test depth (default: smoke; full == smoke for now, expanding later)")
    ap.add_argument("--wer-threshold", type=float, default=0.20,
                    help="WER above this fails (default: 0.20)")
    ap.add_argument("--skip-missing", action="store_true",
                    help="Don't download missing models — skip the backend instead")
    ap.add_argument("--cpu", action="store_true",
                    help="Run with -ng (CPU only)")
    args = ap.parse_args()

    crispasr = find_crispasr()
    if not crispasr:
        print("ERROR: crispasr binary not found in build-ninja-compile/, build/, "
              "build-release/, or PATH. Build it first.", file=sys.stderr)
        return 2
    audio = Path(args.audio)
    if not audio.is_file():
        print(f"ERROR: audio not found: {audio}", file=sys.stderr)
        return 2

    models_dir = Path(args.models)
    if not models_dir.exists():
        try:
            models_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            print(f"ERROR: models dir {models_dir} doesn't exist and can't be created: {e}",
                  file=sys.stderr)
            return 2

    backends = select_backends(args)
    if not backends:
        print("ERROR: no backends selected", file=sys.stderr)
        return 2

    print(f"crispasr:     {crispasr}")
    print(f"models:       {models_dir}  ({free_mb(models_dir)} MB free)")
    print(f"audio:        {audio.name} ({wave.open(str(audio)).getnframes() / 16000:.1f}s)"
          if audio.suffix == ".wav" else f"audio:        {audio.name}")
    print(f"tier:         {args.tier}  (capabilities filter: {args.capabilities or 'none'})")
    print(f"backends:     {len(backends)} selected")
    print(f"download:     {'OFF (--skip-missing)' if args.skip_missing else 'ON'}")
    print(f"backend mode: {'CPU' if args.cpu else 'GPU'}")

    results: list[Result] = []
    for b in backends:
        results.append(smoke(b, crispasr, models_dir, audio,
                             args.skip_missing, not args.cpu, args.wer_threshold))

    # Summary
    print("\n" + "=" * 60)
    print(f"  Summary — tier={args.tier}")
    print("=" * 60)
    pass_ct = sum(1 for r in results if r.status == "PASS")
    fail_ct = sum(1 for r in results if r.status == "FAIL")
    skip_ct = sum(1 for r in results if r.status == "NO_MODEL")
    print(f"  PASS: {pass_ct}    FAIL: {fail_ct}    SKIP (no model): {skip_ct}")
    print()
    for r in results:
        marker = {"PASS": "✓", "FAIL": "✗", "NO_MODEL": "·"}.get(r.status, "?")
        wer_str = f"WER={r.wer:.1%}" if r.wer is not None else ""
        print(f"  {marker} {r.backend:24} {r.status:9} {wer_str}")

    return 1 if fail_ct > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
