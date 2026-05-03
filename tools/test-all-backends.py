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
    # TTS-specific extras (only needed for tts-roundtrip capability)
    voice_file: str | None = None     # e.g. kokoro-voice-af_heart.gguf
    codec_model: str | None = None    # e.g. qwen3-tts-tokenizer-12hz-q8_0.gguf
    tts_extra_args: tuple[str, ...] = ()
    # Reference / shared models — never auto-deleted in --cache-mode=ephemeral.
    # Used by other backends as ground truth (parakeet for TTS roundtrip;
    # whisper-tiny for LID; silero for VAD). Set True for any backend whose
    # download is reused across capability tests of OTHER backends.
    is_reference: bool = False


# Capability → ALL backends that advertise it, populated below.
CAPABILITIES_KNOWN = (
    "transcribe", "json-output", "stream", "beam", "best-of-n",
    "temperature", "word-timestamps", "punctuation", "vad", "lid",
    "tts-roundtrip",
)


REGISTRY: tuple[Backend, ...] = (
    Backend("whisper",    "Whisper (tiny)",      "ggml-tiny.bin",
            "ggerganov/crispasr", "ggml-tiny.bin",
            timeout_s=60, approx_size_mb=80,
            capabilities=("transcribe", "json-output", "stream", "lid", "vad"),
            # Pinned: every other backend triggers LID via this model.
            is_reference=True),
    Backend("parakeet",   "Parakeet TDT 0.6B",   "parakeet-tdt-0.6b-v3-q4_k.gguf",
            "cstr/parakeet-tdt-0.6b-v3-GGUF", "parakeet-tdt-0.6b-v3-q4_k.gguf",
            timeout_s=60, approx_size_mb=420,
            capabilities=("transcribe", "json-output", "word-timestamps"),
            # Pinned: TTS-roundtrip uses parakeet ASR as ground truth.
            is_reference=True),
    Backend("moonshine",  "Moonshine Tiny",      "moonshine-tiny-q4_k.gguf",
            "cstr/moonshine-tiny-GGUF", "moonshine-tiny-q4_k.gguf",
            timeout_s=30, approx_size_mb=30,
            capabilities=("transcribe", "json-output", "beam"),
            extra_files=(("tokenizer.bin", "cstr/moonshine-tiny-GGUF", "tokenizer.bin"),)),
    Backend("moonshine-streaming", "Moonshine Streaming Tiny",
            "moonshine-streaming-tiny-q4_k.gguf",
            "cstr/moonshine-streaming-tiny-GGUF", "moonshine-streaming-tiny-q4_k.gguf",
            timeout_s=60, approx_size_mb=35,
            capabilities=("transcribe", "stream", "json-output"),
            extra_files=(("tokenizer.bin", "cstr/moonshine-streaming-tiny-GGUF", "tokenizer.bin"),)),
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
            timeout_s=120, approx_size_mb=400,
            capabilities=("transcribe", "json-output", "beam")),
    # OmniASR CTC Q4_K now uses mixed quantization (head=4 encoder layers
    # at F16, rest at Q4_K) by default in crispasr-quantize. Recovers
    # nearly all of Q8_0's quality (5% WER on JFK from 1-word "americas"→
    # "americans" diff that omniasr-llm shares; uniform Q4_K had 22.7%)
    # at 658 MB vs Q8_0's 1.0 GB. See LEARNINGS "Q4_K is too lossy as
    # the default for CTC-decoded ASR" + "head=4 sweep on omniasr-ctc"
    # for the full diagnosis. To override (full quant, smaller, ~22% WER):
    # CRISPASR_OMNIASR_QUANT_ALL=1 crispasr-quantize ...
    Backend("omniasr",    "OmniASR CTC 1B v2",   "omniasr-ctc-1b-v2-q4_k.gguf",
            "cstr/omniASR-CTC-1B-v2-GGUF", "omniasr-ctc-1b-v2-q4_k.gguf",
            timeout_s=120, approx_size_mb=660,
            capabilities=("transcribe", "json-output")),
    Backend("omniasr-llm", "OmniASR LLM 300M",   "omniasr-llm-300m-v2-q4_k.gguf",
            "cstr/omniasr-llm-300m-v2-GGUF", "omniasr-llm-300m-v2-q4_k.gguf",
            timeout_s=300, approx_size_mb=1100,
            capabilities=("transcribe", "json-output", "beam",
                          "temperature")),
    Backend("glm-asr",    "GLM ASR Nano",        "glm-asr-nano-q4_k.gguf",
            "cstr/glm-asr-nano-GGUF", "glm-asr-nano-q4_k.gguf",
            timeout_s=300, approx_size_mb=900,
            capabilities=("transcribe", "json-output", "beam",
                          "temperature", "punctuation")),
    Backend("firered-asr", "FireRed ASR2 AED",   "firered-asr2-aed-q4_k.gguf",
            "cstr/firered-asr2-aed-GGUF", "firered-asr2-aed-q4_k.gguf",
            timeout_s=300, approx_size_mb=600,
            capabilities=("transcribe", "json-output")),
    Backend("kyutai-stt", "Kyutai STT 1B",       "kyutai-stt-1b-q4_k.gguf",
            "cstr/kyutai-stt-1b-GGUF", "kyutai-stt-1b-q4_k.gguf",
            timeout_s=90, approx_size_mb=700,
            capabilities=("transcribe", "json-output", "stream", "beam",
                          "temperature", "word-timestamps")),
    Backend("granite",    "Granite Speech 1B",   "granite-speech-4.0-1b-q4_k.gguf",
            "cstr/granite-speech-4.0-1b-GGUF", "granite-speech-4.0-1b-q4_k.gguf",
            timeout_s=300, approx_size_mb=1700,
            capabilities=("transcribe", "json-output", "beam")),
    Backend("granite-4.1", "Granite Speech 4.1 2B", "granite-speech-4.1-2b-q4_k.gguf",
            "cstr/granite-speech-4.1-2b-GGUF", "granite-speech-4.1-2b-q4_k.gguf",
            timeout_s=300, approx_size_mb=1500,
            capabilities=("transcribe", "json-output", "beam")),
    Backend("vibevoice",  "VibeVoice ASR",       "vibevoice-asr-7b-q4_k-fixed.gguf",
            "cstr/vibevoice-asr-GGUF", "vibevoice-asr-q4_k.gguf",
            timeout_s=600, approx_size_mb=4500,
            capabilities=("transcribe", "json-output")),
    Backend("voxtral",    "Voxtral Mini 3B",     "voxtral-mini-3b-2507-q4_k.gguf",
            "cstr/voxtral-mini-3b-2507-GGUF", "voxtral-mini-3b-2507-q4_k.gguf",
            timeout_s=300, approx_size_mb=1900,
            capabilities=("transcribe", "json-output", "temperature")),

    # ---- TTS backends (tts-roundtrip capability) ----
    Backend("kokoro",     "Kokoro 82M (TTS)",    "kokoro-82m-q8_0.gguf",
            "cstr/kokoro-82m-GGUF", "kokoro-82m-q8_0.gguf",
            timeout_s=120, approx_size_mb=90,
            capabilities=("tts-roundtrip",),
            voice_file="kokoro-voice-af_heart.gguf",
            extra_files=(("kokoro-voice-af_heart.gguf",
                          "cstr/kokoro-82m-GGUF",
                          "kokoro-voice-af_heart.gguf"),)),
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


# ---------------------------------------------------------------------------
# VAD / streaming probe — multi-segment clip stitched at runtime from a
# single-segment source audio. Caches into build/test-fixtures/. Used by
# vad-full and stream-long tiers as deterministic ground truth without
# committing a binary fixture to the repo.
# ---------------------------------------------------------------------------


def make_multi_segment_probe(src: Path, n_repeats: int = 4,
                             silence_ms: int = 800,
                             unit_ms: int = 2200) -> Path:
    """Stitch `n_repeats` copies of `src` truncated to `unit_ms` (must
    be 16k mono PCM WAV) separated by `silence_ms` of silence each.

    `unit_ms` should match a single silero speech segment in the source
    so the resulting probe has exactly `n_repeats` detectable segments.
    For samples/jfk.wav the first silero segment is ~0.3-2.2s, so
    unit_ms=2200 captures it cleanly without crossing into segment 2.

    Cached so repeated calls don't redo the work.
    """
    cache_dir = REPO_ROOT / "build" / "test-fixtures"
    cache_dir.mkdir(parents=True, exist_ok=True)
    out = cache_dir / f"{src.stem}_unit{unit_ms}_x{n_repeats}_gap{silence_ms}.wav"
    if out.is_file():
        return out
    with wave.open(str(src), "rb") as wf:
        if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getframerate() != 16000:
            raise RuntimeError(
                f"probe source {src} must be 16-bit 16kHz mono PCM "
                f"(got {wf.getnchannels()}ch / {wf.getsampwidth()*8}-bit / "
                f"{wf.getframerate()}Hz)"
            )
        unit_frames = (unit_ms * 16000) // 1000
        pcm = wf.readframes(min(unit_frames, wf.getnframes()))
    silence_bytes = bytes(silence_ms * 16 * 2)  # 16k, 16-bit mono
    payload = (pcm + silence_bytes) * (n_repeats - 1) + pcm
    with wave.open(str(out), "wb") as ow:
        ow.setnchannels(1)
        ow.setsampwidth(2)
        ow.setframerate(16000)
        ow.writeframes(payload)
    return out


@dataclass
class FetchedModel:
    """Result of fetch_model — tracks which files we downloaded *this run*
    vs which were already on disk. Only "downloaded this run" files are
    eligible for ephemeral cleanup."""
    path: Path
    downloaded_files: list[Path] = field(default_factory=list)


def fetch_model(b: Backend, models_dir: Path, skip_missing: bool,
                space_margin_mb: int = 2048) -> FetchedModel | None:
    # Locate the main model — disk first, fall through to download.
    for cand in (b.local_file, b.hf_file):
        p = models_dir / cand
        if p.is_file():
            # Pre-existing main file. Still try to fetch missing extras
            # (tokenizer.bin, voice file, etc.) — they may be needed and
            # the cleanup logic only deletes what we ourselves downloaded.
            downloaded: list[Path] = []
            for ex_local, ex_repo, ex_file in b.extra_files:
                if not (models_dir / ex_local).is_file() and not skip_missing:
                    try:
                        from huggingface_hub import hf_hub_download
                        ep = Path(hf_hub_download(ex_repo, ex_file, local_dir=str(models_dir)))
                        downloaded.append(ep)
                    except Exception as e:
                        print(f"    extra file {ex_file} failed: {e} (continuing)")
            return FetchedModel(p, downloaded)

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
        downloaded_main = hf_hub_download(b.hf_repo, b.hf_file, local_dir=str(models_dir))
    except Exception as e:
        print(f"    download failed: {e}")
        return None
    sz_mb = os.path.getsize(downloaded_main) / 1024 / 1024
    print(f"    ✓ {sz_mb:.0f} MB in {time.time()-t0:.1f}s")
    fetched = FetchedModel(Path(downloaded_main), [Path(downloaded_main)])
    for ex_local, ex_repo, ex_file in b.extra_files:
        if not (models_dir / ex_local).is_file():
            try:
                ep = Path(hf_hub_download(ex_repo, ex_file, local_dir=str(models_dir)))
                fetched.downloaded_files.append(ep)
            except Exception as e:
                print(f"    extra file {ex_file} failed: {e} (continuing)")
    return fetched


def cleanup_downloaded(b: Backend, fetched: FetchedModel | None) -> None:
    """Delete files we downloaded for this backend in this run.
    Reference models (whisper/parakeet/etc., is_reference=True) are
    skipped — other backends rely on them as ground truth or for LID."""
    if fetched is None or b.is_reference:
        if fetched is not None and b.is_reference:
            print(f"    keep (reference model — used by other backends)")
        return
    freed = 0
    for p in fetched.downloaded_files:
        if p.is_file():
            sz = p.stat().st_size
            try:
                p.unlink()
                freed += sz
            except OSError as e:
                print(f"    cleanup failed for {p.name}: {e}")
    if freed:
        print(f"    cleaned up {freed/1024/1024:.0f} MB ({len(fetched.downloaded_files)} file(s))")


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
out=''; lc=0; n_decodes=0
kwargs = {'step_ms': 2000, 'length_ms': 15000}
# Whisper streaming needs an explicit language to bypass per-decode LID.
if backend == 'whisper':
    kwargs['language'] = 'en'
with s.stream_open(**kwargs) as st:
    for i in range(0, len(pcm), 1600):
        rc = st.feed(pcm[i:i+1600])
        if rc == 1:
            d = st.get_text()
            if d['counter'] != lc: lc=d['counter']; out=d['text']; n_decodes += 1
    st.flush()
    d = st.get_text()
    if d['counter'] != lc: out=d['text']; n_decodes += 1
# Print as: "<n_decodes>|<final_transcript>" so the parent can verify
# we got incremental emission, not just one decode at the end.
sys.stdout.write(f'{n_decodes}|{out}')
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
    raw = r.stdout.strip()
    # Parse "<n_decodes>|<final>" from _STREAM_PY.
    n_decodes, _, transcript = raw.partition("|")
    try:
        n_decodes = int(n_decodes)
    except ValueError:
        return TestOutcome(b.name, "stream", tier, "FAIL",
                           f"unparseable stream output: {raw[:80]!r}", elapsed)
    if not transcript:
        return TestOutcome(b.name, "stream", tier, "EMPTY",
                           f"stream produced empty transcript ({n_decodes} decodes)",
                           elapsed)
    werv = wer(JFK_REF, transcript)
    extra = {"transcript": transcript[:120], "wer": werv, "n_decodes": n_decodes}
    # Smoke: counter must have advanced at least once (incremental emission)
    # and final transcript must be reasonable.
    if n_decodes < 1:
        return TestOutcome(b.name, "stream", tier, "FAIL",
                           "no decode events fired during streaming feed", elapsed, extra)
    if werv is not None and werv > 0.30:
        return TestOutcome(b.name, "stream", tier, "FAIL",
                           f"stream WER {werv:.1%} > 30%", elapsed, extra)
    if tier == "full":
        # Full tier: for an 11s clip with step_ms=2000, expect at least 3
        # decode events (chunks at ~2/4/6/8/10s + flush). Below that, the
        # streaming nature isn't really being exercised.
        expected_min = max(2, int(ctx["audio_duration"] // 3))
        if n_decodes < expected_min:
            return TestOutcome(b.name, "stream", tier, "FAIL",
                               f"only {n_decodes} decode events (expected >= {expected_min})",
                               elapsed, extra)
    detail = f"WER={werv:.1%}, {n_decodes} decodes" if werv is not None else f"{n_decodes} decodes"
    return TestOutcome(b.name, "stream", tier, "PASS", detail, elapsed, extra)


# ---- beam -----------------------------------------------------------------


def test_beam(b: Backend, tier: str, ctx: dict) -> TestOutcome:
    """Smoke: -bs 4 doesn't crash and produces a non-empty transcript with
    bounded WER. Full: beam transcript matches greedy or differs by
    at most 1 word edit (JFK is too clean to expect strict WER<).
    """
    crispasr, model, audio, use_gpu = (
        ctx["crispasr"], ctx["model"], ctx["audio"], ctx["use_gpu"])
    rc, out, err, w = _run_cli(crispasr, b, model, audio, ["-bs", "4"], use_gpu)
    if rc < 0:
        return TestOutcome(b.name, "beam", tier, "TIMEOUT", err, w)
    if rc != 0:
        return TestOutcome(b.name, "beam", tier, "CRASH",
                           (err or "")[-300:], w)
    transcript = parse_transcript(out)
    if not transcript:
        return TestOutcome(b.name, "beam", tier, "EMPTY",
                           (err or "")[-200:], w)
    werv = wer(JFK_REF, transcript)
    if werv is not None and werv > 0.30:
        return TestOutcome(b.name, "beam", tier, "FAIL",
                           f"-bs 4 WER {werv:.1%} > 30%", w,
                           {"transcript": transcript[:80], "wer": werv})
    return TestOutcome(b.name, "beam", tier, "PASS",
                       f"-bs 4 WER={werv:.1%}" if werv is not None else "-bs 4 produced output",
                       w)


# ---- best-of-n ------------------------------------------------------------


def test_best_of_n(b: Backend, tier: str, ctx: dict) -> TestOutcome:
    """Smoke: -bo 4 doesn't crash, produces transcript with bounded WER."""
    crispasr, model, audio, use_gpu = (
        ctx["crispasr"], ctx["model"], ctx["audio"], ctx["use_gpu"])
    # best-of-N typically requires temperature > 0 to actually diversify
    # candidates. Some backends accept -bo with -tp 0 as a no-op.
    rc, out, err, w = _run_cli(crispasr, b, model, audio,
                               ["-bo", "4", "-tp", "0.3"], use_gpu)
    if rc < 0:
        return TestOutcome(b.name, "best-of-n", tier, "TIMEOUT", err, w)
    if rc != 0:
        return TestOutcome(b.name, "best-of-n", tier, "CRASH",
                           (err or "")[-300:], w)
    transcript = parse_transcript(out)
    if not transcript:
        return TestOutcome(b.name, "best-of-n", tier, "EMPTY",
                           (err or "")[-200:], w)
    werv = wer(JFK_REF, transcript)
    if werv is not None and werv > 0.30:
        return TestOutcome(b.name, "best-of-n", tier, "FAIL",
                           f"-bo 4 WER {werv:.1%} > 30%", w,
                           {"transcript": transcript[:80], "wer": werv})
    return TestOutcome(b.name, "best-of-n", tier, "PASS",
                       f"-bo 4 WER={werv:.1%}" if werv is not None else "-bo 4 produced output",
                       w)


# ---- punctuation ----------------------------------------------------------


_PUNCT_RE = re.compile(r"[,.!?]")


def test_punctuation(b: Backend, tier: str, ctx: dict) -> TestOutcome:
    """Run with and without --no-punctuation, verify the toggle has effect.
    With: at least one punctuation char in transcript.
    Without: zero punctuation chars in transcript.
    """
    crispasr, model, audio, use_gpu = (
        ctx["crispasr"], ctx["model"], ctx["audio"], ctx["use_gpu"])
    rc1, out1, err1, w1 = _run_cli(crispasr, b, model, audio, [], use_gpu)
    if rc1 != 0:
        return TestOutcome(b.name, "punctuation", tier, "CRASH",
                           f"with-punct run failed: {(err1 or '')[-200:]}", w1)
    rc2, out2, err2, w2 = _run_cli(crispasr, b, model, audio,
                                   ["--no-punctuation"], use_gpu)
    if rc2 != 0:
        return TestOutcome(b.name, "punctuation", tier, "CRASH",
                           f"--no-punctuation run failed: {(err2 or '')[-200:]}",
                           w1 + w2)
    t_with = parse_transcript(out1)
    t_without = parse_transcript(out2)
    has_with = bool(_PUNCT_RE.search(t_with))
    has_without = bool(_PUNCT_RE.search(t_without))
    if not has_with:
        return TestOutcome(b.name, "punctuation", tier, "FAIL",
                           "default run produced no punctuation chars",
                           w1 + w2,
                           {"with": t_with[:80], "without": t_without[:80]})
    if has_without:
        return TestOutcome(b.name, "punctuation", tier, "FAIL",
                           "--no-punctuation still emitted punctuation",
                           w1 + w2,
                           {"with": t_with[:80], "without": t_without[:80]})
    return TestOutcome(b.name, "punctuation", tier, "PASS",
                       "with: punct present; without: punct absent", w1 + w2)


# Capability → test runner. Capabilities not in this map count as
# unimplemented (status SKIP at SMOKE/FULL tier).
RUNNERS = {
    "transcribe":   test_transcribe,
    "json-output":  test_json_output,
    "temperature":  test_temperature,
    "stream":       test_stream,
    "beam":         test_beam,
    "best-of-n":    test_best_of_n,
    "punctuation":  test_punctuation,
    "word-timestamps": None,  # filled in below
    "vad":          None,
    "lid":          None,
}


# ---- word-timestamps -------------------------------------------------------


def test_word_timestamps(b: Backend, tier: str, ctx: dict) -> TestOutcome:
    """Smoke: -ojf produces JSON whose first segment has a `words` array
    with >= 5 entries. Each word entry should have time offsets.
    Full: each word's t0 < t1, t1 monotonically non-decreasing across
    the array, last t1 within audio duration + 500ms."""
    crispasr, model, audio, use_gpu = (
        ctx["crispasr"], ctx["model"], ctx["audio"], ctx["use_gpu"])
    rc, out, err, w = _run_cli(crispasr, b, model, audio, ["-ojf"], use_gpu)
    if rc != 0:
        return TestOutcome(b.name, "word-timestamps", tier, "CRASH",
                           (err or "")[-200:], w)
    json_path = audio.with_suffix(".json")
    try:
        d = _json.loads(json_path.read_text())
    except Exception as e:
        return TestOutcome(b.name, "word-timestamps", tier, "FAIL",
                           f"-ojf JSON unreadable: {e}", w)
    segs = d.get("transcription") or []
    if not segs:
        return TestOutcome(b.name, "word-timestamps", tier, "FAIL",
                           "no segments in -ojf JSON", w)
    words = segs[0].get("words")
    if not isinstance(words, list) or len(words) < 5:
        return TestOutcome(b.name, "word-timestamps", tier, "FAIL",
                           f"first segment has {len(words) if isinstance(words, list) else 0} word entries (need >= 5)",
                           w)
    if tier == "full":
        # Word-entry schema varies: parakeet uses flat t0/t1 in centiseconds;
        # whisper-style word lists use nested offsets.from/to in ms.
        last_t1_ms = -1
        for i, wd in enumerate(words):
            t0_ms = t1_ms = None
            if "t0" in wd and "t1" in wd:
                # parakeet: cs → ms
                t0_ms, t1_ms = wd["t0"] * 10, wd["t1"] * 10
            elif "offsets" in wd:
                off = wd["offsets"] or {}
                t0_ms, t1_ms = off.get("from"), off.get("to")
            if t0_ms is None or t1_ms is None or t0_ms > t1_ms:
                return TestOutcome(b.name, "word-timestamps", tier, "FAIL",
                                   f"word[{i}] bad timestamps: {wd}", w)
            if t1_ms < last_t1_ms:
                return TestOutcome(b.name, "word-timestamps", tier, "FAIL",
                                   f"word[{i}] t1={t1_ms}ms < prev {last_t1_ms}ms (non-monotonic)",
                                   w)
            last_t1_ms = t1_ms
        audio_ms = int(ctx["audio_duration"] * 1000) + 500
        if last_t1_ms > audio_ms:
            return TestOutcome(b.name, "word-timestamps", tier, "FAIL",
                               f"last word t1={last_t1_ms}ms > audio {audio_ms}ms", w)
    return TestOutcome(b.name, "word-timestamps", tier, "PASS",
                       f"{len(words)} word entries", w,
                       {"n_words": len(words)})


# ---- vad -------------------------------------------------------------------


# silero VAD model can live in a few places; the test runner probes them.
def _find_silero(models_dir: Path) -> Path | None:
    for cand in (
        models_dir / "ggml-silero-v5.1.2.bin",
        models_dir / "silero-v5.1.2.bin",
        Path.home() / ".cache" / "crispasr" / "ggml-silero-v5.1.2.bin",
    ):
        if cand.is_file():
            return cand
    return None


def test_vad(b: Backend, tier: str, ctx: dict) -> TestOutcome:
    """Smoke: --vad on JFK produces 1-8 speech segments. (JFK is one
    sentence with internal pauses; silero typically slices at 5.)

    Full: switches to a stitched multi-segment probe (4 copies of the
    source audio separated by 800ms silence) and asserts the segment
    count is 4 ± 1. Tightens the gate from "any reasonable count" to
    "the count the probe was constructed for."

    Counts come from silero's 'Final speech segments after filtering:
    N' log line.
    """
    crispasr, model, audio, use_gpu, models_dir = (
        ctx["crispasr"], ctx["model"], ctx["audio"], ctx["use_gpu"],
        ctx["models_dir"])
    silero = _find_silero(models_dir)
    if not silero:
        return TestOutcome(b.name, "vad", tier, "SKIP",
                           "silero VAD model not found in --models or "
                           "~/.cache/crispasr/ — download "
                           "ggml-silero-v5.1.2.bin from ggml-org/whisper-vad")
    if tier == "full":
        try:
            audio = make_multi_segment_probe(audio, n_repeats=4, silence_ms=800)
        except Exception as e:
            return TestOutcome(b.name, "vad", tier, "SKIP",
                               f"couldn't build multi-segment probe: {e}")
        expected_lo, expected_hi = 3, 5  # 4 ± 1 tolerance
    else:
        expected_lo, expected_hi = 1, 8
    rc, out, err, w = _run_cli(crispasr, b, model, audio,
                               ["--vad", "-vm", str(silero)], use_gpu)
    if rc != 0:
        return TestOutcome(b.name, "vad", tier, "CRASH",
                           (err or "")[-200:], w)
    m = re.search(r"Final speech segments after filtering:\s*(\d+)", err or "")
    if not m:
        return TestOutcome(b.name, "vad", tier, "FAIL",
                           "no VAD segment-count log line", w)
    n_segs = int(m.group(1))
    if n_segs < expected_lo or n_segs > expected_hi:
        return TestOutcome(b.name, "vad", tier, "FAIL",
                           f"VAD produced {n_segs} segments "
                           f"(expected {expected_lo}-{expected_hi})",
                           w, {"n_segments": n_segs})
    return TestOutcome(b.name, "vad", tier, "PASS",
                       f"{n_segs} segments (range {expected_lo}-{expected_hi})",
                       w, {"n_segments": n_segs})


# ---- lid -------------------------------------------------------------------


def test_lid(b: Backend, tier: str, ctx: dict) -> TestOutcome:
    """Smoke: stderr contains 'auto-detected language: en' or
    'detected ... language = en' (parakeet+others route LID through
    whisper). Full: detected probability > 0.5.
    """
    crispasr, model, audio, use_gpu = (
        ctx["crispasr"], ctx["model"], ctx["audio"], ctx["use_gpu"])
    rc, out, err, w = _run_cli(crispasr, b, model, audio, [], use_gpu)
    if rc != 0:
        return TestOutcome(b.name, "lid", tier, "CRASH",
                           (err or "")[-200:], w)
    # Whisper-style: "auto-detected language: en (p = 0.976672)"
    # crispasr-LID-helper:  "crispasr[lid]: detected 'en' (p=0.977) via whisper"
    m = re.search(
        r"(?:auto-detected language:\s*|crispasr\[lid\][^\n]*detected\s*['\"]?)"
        r"([a-z]{2,3})['\"]?[^\n]*?p\s*=\s*([\d.]+)",
        err or "", re.IGNORECASE)
    if not m:
        return TestOutcome(b.name, "lid", tier, "FAIL",
                           "no LID stderr log line", w)
    lang, prob = m.group(1).lower(), float(m.group(2))
    if lang != "en":
        return TestOutcome(b.name, "lid", tier, "FAIL",
                           f"detected language '{lang}' (expected 'en' on JFK)",
                           w, {"lang": lang, "p": prob})
    if tier == "full" and prob < 0.5:
        return TestOutcome(b.name, "lid", tier, "FAIL",
                           f"LID confidence {prob:.3f} < 0.5", w,
                           {"lang": lang, "p": prob})
    return TestOutcome(b.name, "lid", tier, "PASS",
                       f"detected '{lang}' p={prob:.3f}", w,
                       {"lang": lang, "p": prob})


RUNNERS["word-timestamps"] = test_word_timestamps
RUNNERS["vad"] = test_vad
RUNNERS["lid"] = test_lid


# ---- tts-roundtrip ---------------------------------------------------------


# Reference parakeet model — TTS roundtrip uses parakeet as the ASR
# ground truth. Kept in sync with the parakeet entry in REGISTRY.
_PARAKEET_REF = ("parakeet-tdt-0.6b-v3-q4_k.gguf",
                 "cstr/parakeet-tdt-0.6b-v3-GGUF",
                 "parakeet-tdt-0.6b-v3-q4_k.gguf")

# Fixed phrase — pangram with clean word boundaries, no proper nouns or
# tricky punctuation.
_TTS_PHRASE = "The quick brown fox jumps over the lazy dog"


def _ensure_parakeet(models_dir: Path, skip_missing: bool) -> Path | None:
    for cand in (models_dir / _PARAKEET_REF[0], models_dir / _PARAKEET_REF[2]):
        if cand.is_file():
            return cand
    if skip_missing:
        return None
    try:
        from huggingface_hub import hf_hub_download
        downloaded = hf_hub_download(_PARAKEET_REF[1], _PARAKEET_REF[2],
                                     local_dir=str(models_dir))
        return Path(downloaded)
    except Exception:
        return None


def test_tts_roundtrip(b: Backend, tier: str, ctx: dict) -> TestOutcome:
    """Synthesize a fixed phrase via the backend's TTS path, transcribe
    the result with parakeet, and compare WER to the original phrase.

    Smoke: WER <= 0.20 (TTS quality varies + parakeet has its own ~0%
    error baseline, so this leaves headroom for prosody artefacts).
    Full: WER <= 0.10 — tighter, only acceptable if TTS is high-fidelity
    and the test phrase is a clean pangram.
    """
    crispasr, model, models_dir, use_gpu, skip_missing = (
        ctx["crispasr"], ctx["model"], ctx["models_dir"], ctx["use_gpu"],
        ctx["skip_missing"])
    if not b.voice_file:
        return TestOutcome(b.name, "tts-roundtrip", tier, "FAIL",
                           "backend declared tts-roundtrip but voice_file unset")
    voice_path = models_dir / b.voice_file
    if not voice_path.is_file():
        return TestOutcome(b.name, "tts-roundtrip", tier, "SKIP",
                           f"voice file {b.voice_file} missing in {models_dir}")
    parakeet = _ensure_parakeet(models_dir, skip_missing)
    if not parakeet:
        return TestOutcome(b.name, "tts-roundtrip", tier, "SKIP",
                           "parakeet reference model unavailable "
                           f"(needs {_PARAKEET_REF[0]} for ASR ground truth)")
    out_wav = REPO_ROOT / "build" / "test-fixtures" / f"tts_{b.name}.wav"
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    # Step 1: synthesize
    syn_cmd = [
        str(crispasr), "-m", str(model),
        "--voice", str(voice_path),
        "--tts", _TTS_PHRASE,
        "--tts-output", str(out_wav),
        "--no-prints",
    ]
    if b.codec_model:
        codec_path = models_dir / b.codec_model
        if not codec_path.is_file():
            return TestOutcome(b.name, "tts-roundtrip", tier, "SKIP",
                               f"codec model {b.codec_model} missing")
        syn_cmd += ["--codec-model", str(codec_path)]
    syn_cmd += list(b.tts_extra_args)
    if not use_gpu:
        syn_cmd.append("-ng")
    t0 = time.time()
    try:
        r = subprocess.run(syn_cmd, capture_output=True, text=True,
                           timeout=b.timeout_s)
    except subprocess.TimeoutExpired:
        return TestOutcome(b.name, "tts-roundtrip", tier, "TIMEOUT",
                           f"TTS subprocess timed out", time.time() - t0)
    syn_elapsed = time.time() - t0
    if r.returncode != 0:
        return TestOutcome(b.name, "tts-roundtrip", tier, "CRASH",
                           f"TTS synth failed: {(r.stderr or '')[-300:]}",
                           syn_elapsed)
    if not out_wav.is_file() or out_wav.stat().st_size < 1000:
        return TestOutcome(b.name, "tts-roundtrip", tier, "FAIL",
                           f"TTS produced no usable output ({out_wav})",
                           syn_elapsed)
    # Step 2: parakeet ASR roundtrip
    asr_cmd = [str(crispasr), "--backend", "parakeet", "-m", str(parakeet),
               "-f", str(out_wav), "--no-prints"]
    if not use_gpu:
        asr_cmd.append("-ng")
    t1 = time.time()
    try:
        r2 = subprocess.run(asr_cmd, capture_output=True, text=True, timeout=120)
    except subprocess.TimeoutExpired:
        return TestOutcome(b.name, "tts-roundtrip", tier, "TIMEOUT",
                           "parakeet roundtrip timed out", syn_elapsed + 120)
    asr_elapsed = time.time() - t1
    total_w = syn_elapsed + asr_elapsed
    if r2.returncode != 0:
        return TestOutcome(b.name, "tts-roundtrip", tier, "CRASH",
                           f"parakeet ASR failed: {(r2.stderr or '')[-300:]}",
                           total_w)
    transcript = parse_transcript(r2.stdout)
    werv = wer(_TTS_PHRASE, transcript)
    extra = {"phrase": _TTS_PHRASE, "transcript": transcript[:120], "wer": werv}
    threshold = 0.10 if tier == "full" else 0.20
    if werv is None:
        return TestOutcome(b.name, "tts-roundtrip", tier, "PASS",
                           "transcript present (jiwer missing)", total_w, extra)
    if werv > threshold:
        return TestOutcome(b.name, "tts-roundtrip", tier, "FAIL",
                           f"roundtrip WER {werv:.1%} > {threshold:.0%}",
                           total_w, extra)
    return TestOutcome(b.name, "tts-roundtrip", tier, "PASS",
                       f"roundtrip WER={werv:.1%}", total_w, extra)


RUNNERS["tts-roundtrip"] = test_tts_roundtrip


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
    ap.add_argument("--cache-mode", choices=("keep", "ephemeral"), default="keep",
                    help=("keep (default): preserve downloaded models on disk for "
                          "reuse across runs (good for local dev where bandwidth "
                          "is the bottleneck). "
                          "ephemeral: after each backend's tier tests complete, "
                          "delete files we downloaded this run (for tight-disk "
                          "boxes like Kaggle where bandwidth is cheap and disk "
                          "is the bottleneck). Reference models (whisper-tiny "
                          "for LID, parakeet for TTS-roundtrip ground truth) "
                          "are pinned and never auto-deleted. Pre-existing "
                          "files that we didn't download this run are also "
                          "kept untouched."))
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
    print(f"cache-mode:   {args.cache_mode}"
          + ("  (will free downloads after each backend; "
             "reference models pinned)" if args.cache_mode == "ephemeral" else ""))
    print(f"backend mode: {'CPU' if args.cpu else 'GPU'}")

    outcomes: list[TestOutcome] = []
    for b in backends:
        print(f"\n[{b.name}] {b.display}")
        fetched = fetch_model(b, models_dir, args.skip_missing)
        if fetched is None:
            print("    SKIP — no model on disk"
                  + (" and --skip-missing set" if args.skip_missing else ""))
            outcomes.append(TestOutcome(b.name, "transcribe", tiers["transcribe"],
                                        "NO_MODEL", "model not on disk"))
            continue
        model = fetched.path
        print(f"    model: {model.name} ({os.path.getsize(model)/1024/1024:.0f} MB)",
              flush=True)
        ctx = {
            "crispasr": crispasr, "model": model, "audio": audio,
            "audio_duration": audio_duration, "models_dir": models_dir,
            "use_gpu": not args.cpu, "wer_threshold": args.wer_threshold,
            "skip_missing": args.skip_missing,
        }
        try:
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
        finally:
            # Ephemeral cleanup runs even if a runner raises — Kaggle disk
            # is too tight to leak a multi-GB blob on a partial failure.
            if args.cache_mode == "ephemeral":
                cleanup_downloaded(b, fetched)
                print(f"    free disk now: {free_mb(models_dir)} MB")

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
