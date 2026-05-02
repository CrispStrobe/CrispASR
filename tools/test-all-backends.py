#!/usr/bin/env python3
"""
CrispASR — All-backends regression test.

Sister to `tools/macbook-benchmark-all-backends.py` (which is a perf
benchmark). This is a **regression gate**: pass/fail per backend per
capability, not timing.

Test framework: each backend has a set of advertised capabilities
(transcribe, stream, beam, best-of-n, temperature, word-timestamps,
punctuation, vad, lid). For each capability, three tiers exist:

    ignore  — don't run this test for this backend
    smoke   — quick sanity check (output present + obvious correctness)
    full    — strict regression (e.g. WER threshold, deterministic
              output, timestamp accuracy bounds)

Per-capability tier is set by:

  --profile=smoke (default)   transcribe=smoke, others=ignore
  --profile=full              everything=full
  --profile=feature           transcribe=smoke + every advertised
                              capability=smoke
  --<cap>=ignore|smoke|full   override one capability
                              (e.g. --beam=full --timestamps=smoke)

Selection (subset of backends):

  --backends whisper,parakeet
  --capabilities stream,beam   # backends advertising any of these

Model resolution:

  default --models = /Volumes/backups/ai/crispasr-models on macOS,
  ~/.cache/crispasr elsewhere. CRISPASR_MODELS_DIR env overrides.
  Missing models trigger huggingface_hub.hf_hub_download with HF_TOKEN
  picked up automatically; --skip-missing turns the download off.

Pre-download disk-space check uses each backend's approx_size_mb hint
plus a 2 GB safety margin against shutil.disk_usage(dir).free.

Auto-detects crispasr binary in build-ninja-compile/, build/,
build-release/, or PATH (macOS + Ubuntu both work).

Exit code: 0 if all selected tests PASS or are SKIP, non-zero on FAIL.
"""

from __future__ import annotations

import argparse
import json as _json
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


# ---------------------------------------------------------------------------
# Backend registry
# ---------------------------------------------------------------------------


@dataclass
class Backend:
    name: str            # crispasr --backend value
    display: str         # human label
    local_file: str      # filename to look for in --models
    hf_repo: str         # HF repo id for download fallback
    hf_file: str         # filename within the repo (often == local_file)
    timeout_s: int = 90
    capabilities: tuple[str, ...] = ("transcribe",)
    notes: str = ""
    extra_files: tuple[tuple[str, str, str], ...] = ()
    approx_size_mb: int | None = None


# Capability → ALL backends that advertise it, populated below.
CAPABILITIES_KNOWN = (
    "transcribe", "json-output", "stream", "beam", "best-of-n",
    "temperature", "word-timestamps", "punctuation", "vad", "lid",
)


REGISTRY: tuple[Backend, ...] = (
    Backend("whisper",    "Whisper (base)",      "ggml-base.bin",
            "ggerganov/crispasr", "ggml-base.bin",
            timeout_s=60, approx_size_mb=150,
            capabilities=("transcribe", "json-output", "stream", "lid", "vad")),
    Backend("parakeet",   "Parakeet TDT 0.6B",   "parakeet-tdt-0.6b-v3-q4_k.gguf",
            "cstr/parakeet-tdt-0.6b-v3-GGUF", "parakeet-tdt-0.6b-v3-q4_k.gguf",
            timeout_s=60, approx_size_mb=420,
            capabilities=("transcribe", "json-output", "word-timestamps")),
    Backend("moonshine",  "Moonshine Tiny",      "moonshine-tiny-q4_k.gguf",
            "cstr/moonshine-tiny-GGUF", "moonshine-tiny-q4_k.gguf",
            timeout_s=30, approx_size_mb=30,
            capabilities=("transcribe", "json-output", "beam"),
            extra_files=(("tokenizer.bin", "cstr/moonshine-tiny-GGUF", "tokenizer.bin"),)),
    Backend("moonshine-streaming", "Moonshine Streaming Tiny",
            "moonshine-streaming-tiny-f16.gguf",
            "cstr/moonshine-streaming-tiny-GGUF", "moonshine-streaming-tiny-f16.gguf",
            timeout_s=60, approx_size_mb=85,
            capabilities=("transcribe", "stream", "json-output")),
    Backend("wav2vec2",   "Wav2Vec2 XLSR-EN",    "wav2vec2-xlsr-en-q4_k.gguf",
            "cstr/wav2vec2-large-xlsr-53-english-GGUF",
            "wav2vec2-xlsr-en-q4_k.gguf",
            timeout_s=60, approx_size_mb=200,
            capabilities=("transcribe", "json-output")),
    Backend("fastconformer-ctc", "FastConformer CTC Large",
            "stt-en-fastconformer-ctc-large-q4_k.gguf",
            "cstr/stt-en-fastconformer-ctc-large-GGUF",
            "stt-en-fastconformer-ctc-large-q4_k.gguf",
            timeout_s=30, approx_size_mb=80,
            capabilities=("transcribe", "json-output")),
    Backend("canary",     "Canary 1B",           "canary-1b-v2-q4_k.gguf",
            "cstr/canary-1b-v2-GGUF", "canary-1b-v2-q4_k.gguf",
            timeout_s=120, approx_size_mb=620,
            capabilities=("transcribe", "json-output", "temperature")),
    Backend("cohere",     "Cohere Transcribe",   "cohere-transcribe-q4_k.gguf",
            "cstr/cohere-transcribe-03-2026-GGUF", "cohere-transcribe-q4_k.gguf",
            timeout_s=120, approx_size_mb=1300,
            capabilities=("transcribe", "json-output", "temperature")),
    Backend("qwen3",      "Qwen3 ASR 0.6B",      "qwen3-asr-0.6b-q4_k.gguf",
            "cstr/qwen3-asr-0.6b-GGUF", "qwen3-asr-0.6b-q4_k.gguf",
            timeout_s=60, approx_size_mb=400,
            capabilities=("transcribe", "json-output")),
    Backend("omniasr",    "OmniASR CTC 1B v2",   "omniasr-ctc-1b-v2-q4_k.gguf",
            "cstr/omniASR-CTC-1B-v2-GGUF", "omniasr-ctc-1b-v2-q4_k.gguf",
            timeout_s=120, approx_size_mb=620,
            capabilities=("transcribe", "json-output")),
    Backend("omniasr-llm", "OmniASR LLM 300M",   "omniasr-llm-300m-v2-q4_k.gguf",
            "cstr/omniasr-llm-300m-v2-GGUF", "omniasr-llm-300m-v2-q4_k.gguf",
            timeout_s=120, approx_size_mb=1100,
            capabilities=("transcribe", "json-output", "beam", "best-of-n",
                          "temperature", "punctuation")),
    Backend("glm-asr",    "GLM ASR Nano",        "glm-asr-nano-q4_k.gguf",
            "cstr/glm-asr-nano-GGUF", "glm-asr-nano-q4_k.gguf",
            timeout_s=90, approx_size_mb=900,
            capabilities=("transcribe", "json-output", "beam", "best-of-n",
                          "temperature", "punctuation")),
    Backend("firered-asr", "FireRed ASR2 AED",   "firered-asr2-aed-q4_k.gguf",
            "cstr/firered-asr2-aed-GGUF", "firered-asr2-aed-q4_k.gguf",
            timeout_s=90, approx_size_mb=600,
            capabilities=("transcribe", "json-output")),
    Backend("kyutai-stt", "Kyutai STT 1B",       "kyutai-stt-1b-q4_k.gguf",
            "cstr/kyutai-stt-1b-GGUF", "kyutai-stt-1b-q4_k.gguf",
            timeout_s=90, approx_size_mb=700,
            capabilities=("transcribe", "json-output", "stream", "beam",
                          "best-of-n", "temperature", "word-timestamps")),
    Backend("granite",    "Granite Speech 1B",   "granite-speech-4.0-1b-q4_k.gguf",
            "cstr/granite-speech-4.0-1b-GGUF", "granite-speech-4.0-1b-q4_k.gguf",
            timeout_s=300, approx_size_mb=1700,
            capabilities=("transcribe", "json-output")),
    Backend("granite-4.1", "Granite Speech 4.1 2B", "granite-speech-4.1-2b-q4_k.gguf",
            "cstr/granite-speech-4.1-2b-GGUF", "granite-speech-4.1-2b-q4_k.gguf",
            timeout_s=300, approx_size_mb=1500,
            capabilities=("transcribe", "json-output")),
    Backend("vibevoice",  "VibeVoice ASR",       "vibevoice-asr-7b-q4_k-fixed.gguf",
            "cstr/vibevoice-asr-GGUF", "vibevoice-asr-q4_k.gguf",
            timeout_s=120, approx_size_mb=4500,
            capabilities=("transcribe", "json-output")),
    Backend("voxtral",    "Voxtral Mini 3B",     "voxtral-mini-3b-2507-q4_k.gguf",
            "cstr/voxtral-mini-3b-2507-GGUF", "voxtral-mini-3b-2507-q4_k.gguf",
            timeout_s=300, approx_size_mb=1900,
            capabilities=("transcribe", "json-output", "temperature")),
)


# ---------------------------------------------------------------------------
# crispasr binary + model resolution
# ---------------------------------------------------------------------------


def find_crispasr() -> Path | None:
    for rel in ("build-ninja-compile/bin/crispasr",
                "build/bin/crispasr",
                "build-release/bin/crispasr"):
        p = REPO_ROOT / rel
        if p.is_file():
            return p
    found = shutil.which("crispasr")
    return Path(found) if found else None


def free_mb(path: Path) -> int:
    p = path if path.exists() else path.parent
    return shutil.disk_usage(p).free // (1024 * 1024)


def fetch_model(b: Backend, models_dir: Path, skip_missing: bool,
                space_margin_mb: int = 2048) -> Path | None:
    for cand in (b.local_file, b.hf_file):
        p = models_dir / cand
        if p.is_file():
            return p
    if skip_missing:
        return None
    needed_mb = (b.approx_size_mb or 0) + space_margin_mb
    have_mb = free_mb(models_dir)
    if b.approx_size_mb and have_mb < needed_mb:
        print(f"    skip download: need ~{needed_mb} MB, only {have_mb} MB free")
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
    sz_mb = os.path.getsize(downloaded) / 1024 / 1024
    print(f"    ✓ {sz_mb:.0f} MB in {time.time()-t0:.1f}s")
    for ex_local, ex_repo, ex_file in b.extra_files:
        if not (models_dir / ex_local).is_file():
            try:
                hf_hub_download(ex_repo, ex_file, local_dir=str(models_dir))
            except Exception as e:
                print(f"    extra file {ex_file} failed: {e} (continuing)")
    return Path(downloaded)


# ---------------------------------------------------------------------------
# Test outcome model
# ---------------------------------------------------------------------------


@dataclass
class TestOutcome:
    backend: str
    capability: str
    tier: str            # smoke | full | ignore
    status: str          # PASS | FAIL | SKIP | NO_MODEL | TIMEOUT | CRASH | EMPTY
    detail: str = ""
    wall_s: float = 0.0
    extra: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Test runners — one per capability. Each returns a TestOutcome.
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


def _run_cli(crispasr: Path, b: Backend, model: Path, audio: Path,
             extra_args: list[str], use_gpu: bool,
             timeout_override: int | None = None) -> tuple[int, str, str, float]:
    cmd = [str(crispasr), "--backend", b.name, "-m", str(model),
           "-f", str(audio), "--no-prints", *extra_args]
    if not use_gpu:
        cmd.append("-ng")
    timeout = timeout_override or b.timeout_s
    t0 = time.time()
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return -1, "", f"TIMEOUT after {timeout}s", time.time() - t0
    return r.returncode, r.stdout, r.stderr, time.time() - t0


def parse_transcript(stdout: str) -> str:
    return re.sub(r"\[[\d:.]+\s*-->\s*[\d:.]+\]\s*", "", stdout.strip()).strip()


# ---- transcribe ----------------------------------------------------------------


def test_transcribe(b: Backend, tier: str, ctx: dict) -> TestOutcome:
    crispasr, model, audio, use_gpu, threshold = (
        ctx["crispasr"], ctx["model"], ctx["audio"], ctx["use_gpu"],
        ctx["wer_threshold"])
    rc, out, err, w = _run_cli(crispasr, b, model, audio, [], use_gpu)
    if rc < 0:
        return TestOutcome(b.name, "transcribe", tier, "TIMEOUT", err, w)
    if rc != 0:
        return TestOutcome(b.name, "transcribe", tier, "CRASH",
                           (err or "")[-300:], w)
    transcript = parse_transcript(out)
    if not transcript:
        return TestOutcome(b.name, "transcribe", tier, "EMPTY",
                           (err or "")[-200:], w)
    werv = wer(JFK_REF, transcript)
    extra = {"transcript": transcript[:120], "wer": werv}
    if tier == "smoke":
        # Smoke: transcript must be non-empty + WER <= 2× threshold
        if werv is not None and werv > 2 * threshold:
            return TestOutcome(b.name, "transcribe", tier, "FAIL",
                               f"WER {werv:.1%} > {2*threshold:.0%} (smoke)",
                               w, extra)
        return TestOutcome(b.name, "transcribe", tier, "PASS",
                           f"transcript={len(transcript)} chars; WER={werv:.1%}"
                           if werv is not None else "transcript present (no jiwer)",
                           w, extra)
    # full
    if werv is None:
        return TestOutcome(b.name, "transcribe", tier, "PASS",
                           "transcript present (jiwer missing)", w, extra)
    if werv > threshold:
        return TestOutcome(b.name, "transcribe", tier, "FAIL",
                           f"WER {werv:.1%} > {threshold:.0%}", w, extra)
    return TestOutcome(b.name, "transcribe", tier, "PASS",
                       f"WER={werv:.1%}", w, extra)


# ---- json-output ------------------------------------------------------------


def test_json_output(b: Backend, tier: str, ctx: dict) -> TestOutcome:
    crispasr, model, audio, use_gpu = (
        ctx["crispasr"], ctx["model"], ctx["audio"], ctx["use_gpu"])
    rc, out, err, w = _run_cli(crispasr, b, model, audio, ["-oj"], use_gpu)
    if rc < 0:
        return TestOutcome(b.name, "json-output", tier, "TIMEOUT", err, w)
    if rc != 0:
        return TestOutcome(b.name, "json-output", tier, "CRASH",
                           (err or "")[-200:], w)
    json_path = audio.with_suffix(".json")
    if not json_path.is_file():
        return TestOutcome(b.name, "json-output", tier, "FAIL",
                           f"-oj didn't produce {json_path.name}", w)
    try:
        d = _json.loads(json_path.read_text())
    except Exception as e:
        return TestOutcome(b.name, "json-output", tier, "FAIL",
                           f"invalid JSON: {e}", w)
    segs = d.get("transcription") or []
    if not segs:
        return TestOutcome(b.name, "json-output", tier, "FAIL",
                           "no transcription segments in JSON", w)
    s0 = segs[0]
    if not s0.get("text"):
        return TestOutcome(b.name, "json-output", tier, "FAIL",
                           "first segment has no text", w)
    if tier == "full":
        # Full: timestamps must be present and within audio duration bounds
        offsets = s0.get("offsets") or {}
        t1 = offsets.get("to")
        if t1 is None:
            return TestOutcome(b.name, "json-output", tier, "FAIL",
                               "no offsets.to in first segment", w)
        audio_ms = int(ctx["audio_duration"] * 1000) + 500  # 500ms tolerance
        if t1 > audio_ms:
            return TestOutcome(b.name, "json-output", tier, "FAIL",
                               f"offsets.to={t1}ms exceeds audio {audio_ms}ms", w)
    return TestOutcome(b.name, "json-output", tier, "PASS",
                       f"{len(segs)} segment(s)", w,
                       {"n_segments": len(segs)})


# ---- temperature -----------------------------------------------------------


def test_temperature(b: Backend, tier: str, ctx: dict) -> TestOutcome:
    """T=0 should be deterministic across two runs."""
    crispasr, model, audio, use_gpu = (
        ctx["crispasr"], ctx["model"], ctx["audio"], ctx["use_gpu"])
    rc1, out1, err1, w1 = _run_cli(crispasr, b, model, audio,
                                   ["-tp", "0"], use_gpu)
    if rc1 != 0:
        return TestOutcome(b.name, "temperature", tier, "CRASH",
                           f"T=0 run failed: {(err1 or '')[-200:]}", w1)
    rc2, out2, err2, w2 = _run_cli(crispasr, b, model, audio,
                                   ["-tp", "0"], use_gpu)
    if rc2 != 0:
        return TestOutcome(b.name, "temperature", tier, "CRASH",
                           f"T=0 rerun failed: {(err2 or '')[-200:]}", w1 + w2)
    t1, t2 = parse_transcript(out1), parse_transcript(out2)
    if t1 != t2:
        return TestOutcome(b.name, "temperature", tier, "FAIL",
                           f"T=0 not deterministic across runs", w1 + w2,
                           {"run1": t1[:80], "run2": t2[:80]})
    return TestOutcome(b.name, "temperature", tier, "PASS",
                       "T=0 deterministic across 2 runs", w1 + w2)


# ---- stream (Python wrapper round-trip) ------------------------------------


_STREAM_PY = """
import sys, wave, numpy as np
from crispasr import Session
backend = sys.argv[1]
model = sys.argv[2]
wav = sys.argv[3]
s = Session(model, backend=backend)
with wave.open(wav,'rb') as w:
    pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
out=''; lc=0
with s.stream_open(step_ms=2000, length_ms=15000) as st:
    for i in range(0, len(pcm), 1600):
        rc = st.feed(pcm[i:i+1600])
        if rc == 1:
            d = st.get_text()
            if d['counter'] != lc: lc=d['counter']; out=d['text']
    st.flush()
    d = st.get_text()
    if d['counter'] != lc: out=d['text']
print(out)
"""


def test_stream(b: Backend, tier: str, ctx: dict) -> TestOutcome:
    crispasr, model, audio = ctx["crispasr"], ctx["model"], ctx["audio"]
    # Locate libcrispasr next to the binary.
    libdir = crispasr.parent.parent / "src"
    libname = "libcrispasr.dylib" if platform.system() == "Darwin" else "libcrispasr.so"
    libpath = libdir / libname
    if not libpath.is_file():
        return TestOutcome(b.name, "stream", tier, "SKIP",
                           f"libcrispasr not found at {libpath} (Python wrapper needs it)")
    env = {**os.environ, "CRISPASR_LIB_PATH": str(libpath),
           "PYTHONPATH": str(REPO_ROOT / "python")}
    cmd = [sys.executable, "-c", _STREAM_PY, b.name, str(model), str(audio)]
    t0 = time.time()
    try:
        r = subprocess.run(cmd, capture_output=True, text=True,
                           timeout=b.timeout_s * 2, env=env)
    except subprocess.TimeoutExpired:
        return TestOutcome(b.name, "stream", tier, "TIMEOUT",
                           "Python stream subprocess timed out",
                           time.time() - t0)
    elapsed = time.time() - t0
    if r.returncode != 0:
        return TestOutcome(b.name, "stream", tier, "CRASH",
                           (r.stderr or "")[-300:], elapsed)
    transcript = r.stdout.strip()
    if not transcript:
        return TestOutcome(b.name, "stream", tier, "EMPTY",
                           "stream produced empty transcript", elapsed)
    werv = wer(JFK_REF, transcript)
    extra = {"transcript": transcript[:120], "wer": werv}
    if werv is not None and werv > 0.30:
        return TestOutcome(b.name, "stream", tier, "FAIL",
                           f"stream WER {werv:.1%} > 30%", elapsed, extra)
    return TestOutcome(b.name, "stream", tier, "PASS",
                       f"stream WER={werv:.1%}" if werv is not None else "stream OK",
                       elapsed, extra)


# Capability → test runner. Capabilities not in this map count as
# unimplemented (status SKIP at SMOKE/FULL tier).
RUNNERS = {
    "transcribe":   test_transcribe,
    "json-output":  test_json_output,
    "temperature":  test_temperature,
    "stream":       test_stream,
    # Future: beam, best-of-n, word-timestamps, punctuation, vad, lid
}


# ---------------------------------------------------------------------------
# Profile + tier resolution
# ---------------------------------------------------------------------------


PROFILES = {
    "smoke":   {"transcribe": "smoke"},  # everything else defaults to ignore
    "feature": {c: "smoke" for c in CAPABILITIES_KNOWN},
    "full":    {c: "full" for c in CAPABILITIES_KNOWN},
}


def resolve_tier_per_capability(args) -> dict[str, str]:
    tiers = dict(PROFILES.get(args.profile, {}))
    for c in CAPABILITIES_KNOWN:
        v = getattr(args, c.replace("-", "_"), None)
        if v is not None:
            tiers[c] = v
    return tiers


def select_backends(args) -> list[Backend]:
    if args.backends:
        wanted = {n.strip() for n in args.backends.split(",")}
        sel = [b for b in REGISTRY if b.name in wanted]
        missing = wanted - {b.name for b in sel}
        if missing:
            print(f"WARNING: unknown backends in --backends: {sorted(missing)}",
                  file=sys.stderr)
        return sel
    if args.capabilities:
        caps = {c.strip() for c in args.capabilities.split(",")}
        return [b for b in REGISTRY if caps & set(b.capabilities)]
    return list(REGISTRY)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    default_models = os.environ.get(
        "CRISPASR_MODELS_DIR",
        "/Volumes/backups/ai/crispasr-models" if platform.system() == "Darwin"
        else str(Path.home() / ".cache" / "crispasr"),
    )
    ap = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="See module docstring for the full tier model.",
    )
    ap.add_argument("--models", default=default_models,
                    help=f"Model directory (default: {default_models})")
    ap.add_argument("--audio", default=str(JFK_WAV),
                    help="Audio file (default: samples/jfk.wav)")
    ap.add_argument("--backends", default=None,
                    help="Comma-separated subset of backends (default: all)")
    ap.add_argument("--capabilities", default=None,
                    help="Filter to backends advertising any of these (comma-sep)")
    ap.add_argument("--profile", default="smoke", choices=list(PROFILES.keys()),
                    help="Default tier per capability (overridden by --<cap>=...)")
    ap.add_argument("--wer-threshold", type=float, default=0.10,
                    help="WER above this fails 'transcribe' at full tier (default: 0.10)")
    ap.add_argument("--skip-missing", action="store_true",
                    help="Don't download missing models — skip the backend instead")
    ap.add_argument("--cpu", action="store_true",
                    help="Run with -ng (CPU only)")
    # Per-capability tier overrides
    for c in CAPABILITIES_KNOWN:
        ap.add_argument(f"--{c}", default=None,
                        choices=("ignore", "smoke", "full"),
                        help=f"Override tier for {c} capability")
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
    audio_duration = (wave.open(str(audio)).getnframes() / 16000.0
                      if audio.suffix == ".wav" else 0.0)

    models_dir = Path(args.models)
    models_dir.mkdir(parents=True, exist_ok=True)

    backends = select_backends(args)
    if not backends:
        print("ERROR: no backends selected", file=sys.stderr)
        return 2

    tiers = resolve_tier_per_capability(args)
    active_caps = [c for c, t in tiers.items() if t != "ignore"]

    print(f"crispasr:     {crispasr}")
    print(f"models:       {models_dir}  ({free_mb(models_dir)} MB free)")
    if audio_duration:
        print(f"audio:        {audio.name} ({audio_duration:.1f}s)")
    else:
        print(f"audio:        {audio.name}")
    print(f"profile:      {args.profile}")
    print(f"tiers:        " + ", ".join(f"{c}={tiers[c]}" for c in active_caps)
          + (" (others=ignore)" if len(active_caps) < len(tiers) else ""))
    print(f"backends:     {len(backends)} selected")
    print(f"download:     {'OFF (--skip-missing)' if args.skip_missing else 'ON'}")
    print(f"backend mode: {'CPU' if args.cpu else 'GPU'}")

    outcomes: list[TestOutcome] = []
    for b in backends:
        print(f"\n[{b.name}] {b.display}")
        model = fetch_model(b, models_dir, args.skip_missing)
        if not model:
            print("    SKIP — no model on disk"
                  + (" and --skip-missing set" if args.skip_missing else ""))
            outcomes.append(TestOutcome(b.name, "transcribe", tiers["transcribe"],
                                        "NO_MODEL", "model not on disk"))
            continue
        print(f"    model: {model.name} ({os.path.getsize(model)/1024/1024:.0f} MB)",
              flush=True)
        ctx = {
            "crispasr": crispasr, "model": model, "audio": audio,
            "audio_duration": audio_duration,
            "use_gpu": not args.cpu, "wer_threshold": args.wer_threshold,
        }
        # Run each advertised capability whose tier != ignore.
        for cap in b.capabilities:
            tier = tiers.get(cap, "ignore")
            if tier == "ignore":
                continue
            runner = RUNNERS.get(cap)
            if runner is None:
                outcomes.append(TestOutcome(b.name, cap, tier, "SKIP",
                                            "no runner implemented yet"))
                print(f"    {cap:18} SKIP   (runner not yet implemented)")
                continue
            o = runner(b, tier, ctx)
            outcomes.append(o)
            mark = {"PASS": "✓", "FAIL": "✗", "SKIP": "·", "NO_MODEL": "·"}\
                .get(o.status, "?")
            print(f"    {cap:18} {mark} {o.status:8} ({o.tier:5}) "
                  f"{o.detail[:70]}")

    # Summary
    print("\n" + "=" * 60)
    print(f"  Summary — profile={args.profile}")
    print("=" * 60)
    by_status: dict[str, int] = {}
    for o in outcomes:
        by_status[o.status] = by_status.get(o.status, 0) + 1
    parts = ", ".join(f"{k}: {v}" for k, v in sorted(by_status.items()))
    print(f"  {parts}")
    fails = [o for o in outcomes if o.status == "FAIL"]
    if fails:
        print("\n  Failures:")
        for o in fails:
            print(f"    ✗ {o.backend:20} {o.capability:14} {o.detail[:80]}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
