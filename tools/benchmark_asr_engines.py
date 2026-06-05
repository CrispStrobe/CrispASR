#!/usr/bin/env python3
"""
benchmark_asr_engines.py — head-to-head Parakeet TDT 0.6B v3 benchmark
=====================================================================

Reproduction of the comparison from CrispASR issue #81 (onnx-asr vs
CrispASR/GGUF on the same parakeet-tdt-0.6b-v3 model). Runs both
engines on the same audio under the same window/warmup/runs settings
and reports realtime factor + per-call latency.

A more thorough version of the issue #81 cross-comparison documented
in PERFORMANCE.md "onnx-asr cross-comparison — issue #81 (2026-05-09)":
the existing PERFORMANCE.md run is 1 quant × whole-file × 1 audio; this
script sweeps a 5-quant × 2-mode × 2-audio matrix and emits JSON for
downstream tooling, while keeping the same engine-comparison framing.
Read PERFORMANCE.md first for the headline conclusions, this script for
deeper exploration / CUDA portability.

Two call modes:

  whole    — feed the full audio in one transcribe call. Closest to
             onnx-asr's stock model.recognize(path); fastest because
             the encoder amortizes over the whole utterance.
  chunked  — split into N-second windows (default 4 s, what the
             reporter used) and call transcribe on each chunk. Reports
             p50/p95 per-call latency. This is the latency shape that
             matters for streaming ASR.

Two CrispASR call paths:

  cli       — subprocess ./build/bin/crispasr (one process per run).
              Includes process startup + Metal kernel JIT, so usable
              only with explicit warmup runs and steady-state
              measurement on subsequent runs.
  ctypes    — ctypes.CDLL on libcrispasr.{dylib,so,dll}, calling
              crispasr_parakeet_{init,transcribe,result_text,
              result_free,free}. One process; matches what the issue
              #81 reporter built. This is the apples-to-apples path.

Cross-platform: the script auto-picks libcrispasr.dylib (macOS),
libcrispasr.so (Linux), or crispasr.dll (Windows), and selects
CoreMLExecutionProvider (macOS) or CUDAExecutionProvider (Windows/Linux
+ CUDA) for onnx-asr. Override with --crispasr-lib / --providers.

Usage:

  # Default: both engines, both modes, both audios, both quant cells.
  python tools/benchmark_asr_engines.py

  # Just the onnx side, fp32, 60 s clip, 5 runs:
  python tools/benchmark_asr_engines.py --engine onnx --onnx-quant fp32 \\
      --audio long --runs 5

  # CrispASR Q8_0 only, ctypes, chunked at 2 s windows:
  python tools/benchmark_asr_engines.py --engine crispasr --crispasr-call ctypes \\
      --gguf-quants q8_0 --mode chunked --window-s 2

  # Windows CUDA reproduction (after building libcrispasr with CUDA):
  python tools/benchmark_asr_engines.py --gpu-backend cuda \\
      --providers CUDAExecutionProvider --crispasr-lib path/to/crispasr.dll

The script writes a JSON sidecar with raw results and prints a
markdown-friendly summary table.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import platform
import re
import statistics
import struct
import subprocess
import sys
import time
import wave
from dataclasses import asdict, dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BUILD = REPO_ROOT / "build-ninja-compile"
DEFAULT_CRISPASR_BIN = DEFAULT_BUILD / "bin" / "crispasr"
SAMPLES_JFK = REPO_ROOT / "samples" / "jfk.wav"
DEFAULT_LONG_WAV = REPO_ROOT / "tests" / "fixtures" / "bench_long_60s.wav"

JFK_REF = (
    "and so my fellow americans ask not what your country can do for you "
    "ask what you can do for your country"
)

# GGUF artifacts on the SSD. Falls back to download from HF.
GGUF_DIR = Path("/Volumes/backups/ai/crispasr")
GGUF_FILES = {
    "q4_k": "parakeet-tdt-0.6b-v3-q4_k.gguf",
    "q5_0": "parakeet-tdt-0.6b-v3-q5_0.gguf",
    "q8_0": "parakeet-tdt-0.6b-v3-q8_0.gguf",
    "f16":  "parakeet-tdt-0.6b-v3.gguf",  # repo's non-quant is f16-sized
}
GGUF_HF_REPO = "cstr/parakeet-tdt-0.6b-v3-GGUF"

# ONNX artifacts. snapshot-downloaded on first run.
ONNX_DIR = Path("/Volumes/backups/ai/huggingface-hub/parakeet-tdt-0.6b-v3-onnx")
ONNX_HF_REPO = "istupakov/parakeet-tdt-0.6b-v3-onnx"


# =======================================================================
# helpers
# =======================================================================

def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z ]", "", text.lower())).strip()


def wer_or_none(ref: str, hyp: str) -> float | None:
    try:
        from jiwer import wer as compute_wer  # type: ignore
    except ImportError:
        return None
    r, h = normalize(ref), normalize(hyp)
    if not r or not h:
        return 1.0
    return float(compute_wer(r, h))


def load_wav_pcm_f32(path: Path):
    """Read a 16 kHz mono 16-bit PCM wav into numpy float32.

    Returns (np.ndarray[float32], duration_seconds). Bails on
    non-conforming input rather than silently resampling — the parakeet
    model expects 16 kHz; we want loud failure.
    """
    import numpy as np  # type: ignore
    with wave.open(str(path)) as w:
        if w.getframerate() != 16000:
            raise SystemExit(f"{path}: expected 16000 Hz, got {w.getframerate()}")
        if w.getnchannels() != 1:
            raise SystemExit(f"{path}: expected mono, got {w.getnchannels()} channels")
        if w.getsampwidth() != 2:
            raise SystemExit(f"{path}: expected 16-bit PCM, got sampwidth={w.getsampwidth()}")
        nframes = w.getnframes()
        raw = w.readframes(nframes)
    samples = np.frombuffer(raw, dtype="<i2").astype(np.float32) * (1.0 / 32768.0)
    samples = np.ascontiguousarray(samples)  # ensure contiguous f32 for ctypes pointer
    return samples, nframes / 16000.0


def ensure_long_wav(out_path: Path, target_s: float = 60.0) -> Path:
    """Build a deterministic 60 s clip by tiling jfk.wav and trimming.

    Hermetic: no external download. Caveat: the tiled audio is just
    JFK repeated, so it's only useful for *speed* measurement — WER on
    this clip is uninformative. We still report it in the table for
    sanity but mark it as 'tiled' so nobody mistakes it for a clean
    benchmark of WER quality.
    """
    if out_path.exists():
        return out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(SAMPLES_JFK)) as src:
        params = src.getparams()
        rate = src.getframerate()
        chunk = src.readframes(src.getnframes())
    target_frames = int(target_s * rate)
    bytes_per_frame = params.sampwidth * params.nchannels
    src_frames = len(chunk) // bytes_per_frame
    reps = (target_frames + src_frames - 1) // src_frames
    tiled = (chunk * reps)[: target_frames * bytes_per_frame]
    with wave.open(str(out_path), "wb") as w:
        w.setparams(params)
        w.writeframes(tiled)
    return out_path


def ensure_gguf(quant: str, models_dir: Path) -> Path:
    fname = GGUF_FILES[quant]
    p = models_dir / fname
    if p.is_file():
        return p
    print(f"  ↓ {fname} not on SSD — downloading from {GGUF_HF_REPO}…")
    from huggingface_hub import hf_hub_download
    out = hf_hub_download(GGUF_HF_REPO, fname, local_dir=str(models_dir))
    return Path(out)


def ensure_onnx(onnx_dir: Path) -> Path:
    needed = (onnx_dir / "vocab.txt").is_file() and (
        (onnx_dir / "encoder-model.int8.onnx").is_file()
        or (onnx_dir / "encoder-model.onnx").is_file()
    )
    if needed:
        return onnx_dir
    onnx_dir.mkdir(parents=True, exist_ok=True)
    print(f"  ↓ ONNX models not on SSD — downloading {ONNX_HF_REPO} → {onnx_dir}…")
    from huggingface_hub import snapshot_download
    snapshot_download(ONNX_HF_REPO, local_dir=str(onnx_dir))
    return onnx_dir


def chunk_indices(n_samples: int, window_samples: int) -> list[tuple[int, int]]:
    """Non-overlapping chunks; the trailing partial chunk is kept as-is.

    onnx-asr's recognize() doesn't internally chunk for us at this
    granularity, so we feed identical chunks to both engines.
    """
    out = []
    for s in range(0, n_samples, window_samples):
        out.append((s, min(s + window_samples, n_samples)))
    return out


def host_default_lib() -> Path:
    """Pick the first existing libcrispasr.{dylib,so,dll} candidate.

    Falls back to the canonical location for the host OS even if missing,
    so the script can produce a useful error message at dlopen time
    rather than failing inside argparse-default evaluation.
    """
    sysname = platform.system()
    if sysname == "Darwin":
        candidates = [
            DEFAULT_BUILD / "src" / "libcrispasr.dylib",
            REPO_ROOT / "build" / "src" / "libcrispasr.dylib",
        ]
    elif sysname == "Linux":
        candidates = [
            DEFAULT_BUILD / "src" / "libcrispasr.so",
            REPO_ROOT / "build" / "src" / "libcrispasr.so",
        ]
    elif sysname == "Windows":
        # cmake's default for SHARED is `<name>.dll` (no `lib` prefix), but
        # MinGW and some build configs produce `lib<name>.dll`; honor both.
        candidates = [
            DEFAULT_BUILD / "src" / "crispasr.dll",
            DEFAULT_BUILD / "src" / "libcrispasr.dll",
            REPO_ROOT / "build" / "src" / "crispasr.dll",
            REPO_ROOT / "build" / "src" / "libcrispasr.dll",
        ]
    else:
        candidates = [DEFAULT_BUILD / "src" / "libcrispasr.dylib"]
    for c in candidates:
        if c.is_file():
            return c
    return candidates[0]


def host_default_providers() -> list[str]:
    """Auto-pick a sensible onnxruntime EP per host.

    On macOS we deliberately default to CPU EP, not CoreML — PERFORMANCE.md
    "onnx-asr cross-comparison — issue #81 (2026-05-09)" documents that
    on M1 the CoreML EP is *slower* than CPU EP for parakeet-shaped
    graphs (CTC int8: 1.28 s CoreML vs 0.72 s CPU on the same model),
    and that the upstream parakeet TDT export can't even reach CoreML
    because of `onnxruntime#26355` (external-data + protobuf 2 GB
    ceiling). Use `--providers CoreMLExecutionProvider,...` if you
    want to measure CoreML explicitly.

    On Windows / Linux with a working dGPU EP (CUDA / DirectML / TRT)
    we prefer the GPU EP — that's where ONNX has its real architectural
    advantage and is the shape relevant to issue #81's reporter.
    """
    try:
        import onnxruntime as ort
        avail = ort.get_available_providers()
    except Exception:
        return ["CPUExecutionProvider"]
    # GPU EPs that are actually fast in practice, in preference order.
    for ep in ("CUDAExecutionProvider", "TensorrtExecutionProvider", "DmlExecutionProvider"):
        if ep in avail:
            return [ep, "CPUExecutionProvider"]
    # On macOS, CPU EP outperforms CoreML EP for parakeet — default to CPU.
    return ["CPUExecutionProvider"]


# =======================================================================
# CrispASR — ctypes path
# =======================================================================

class CrispASRParakeet:
    """Thin ctypes wrapper around crispasr_parakeet_* C ABI."""

    def __init__(self, lib_path: Path, model_path: Path, n_threads: int, use_flash: bool):
        self._lib = ctypes.CDLL(str(lib_path))
        L = self._lib
        L.crispasr_parakeet_init.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int]
        L.crispasr_parakeet_init.restype = ctypes.c_void_p
        L.crispasr_parakeet_transcribe.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int,
            ctypes.c_int64,
        ]
        L.crispasr_parakeet_transcribe.restype = ctypes.c_void_p
        L.crispasr_parakeet_result_text.argtypes = [ctypes.c_void_p]
        L.crispasr_parakeet_result_text.restype = ctypes.c_char_p
        L.crispasr_parakeet_result_free.argtypes = [ctypes.c_void_p]
        L.crispasr_parakeet_free.argtypes = [ctypes.c_void_p]

        t0 = time.perf_counter()
        self._ctx = L.crispasr_parakeet_init(
            str(model_path).encode(), int(n_threads), 1 if use_flash else 0
        )
        self.load_s = time.perf_counter() - t0
        if not self._ctx:
            raise RuntimeError(f"crispasr_parakeet_init failed for {model_path}")

    @staticmethod
    def _np_ptr(arr_np):
        # numpy float32 array → POINTER(c_float). Caller must keep arr_np alive.
        return arr_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    def transcribe(self, pcm_np, n: int | None = None) -> str:
        if n is None:
            n = len(pcm_np)
        ptr = self._np_ptr(pcm_np)
        res = self._lib.crispasr_parakeet_transcribe(self._ctx, ptr, int(n), 0)
        if not res:
            return ""
        try:
            txt = self._lib.crispasr_parakeet_result_text(res) or b""
            return txt.decode("utf-8", "ignore")
        finally:
            self._lib.crispasr_parakeet_result_free(res)

    def transcribe_window(self, pcm_full_np, lo: int, hi: int) -> str:
        # Slice without copy via numpy view; pass pointer.
        view = pcm_full_np[lo:hi]
        return self.transcribe(view, hi - lo)

    def close(self):
        if getattr(self, "_ctx", None):
            self._lib.crispasr_parakeet_free(self._ctx)
            self._ctx = None


# =======================================================================
# CrispASR — CLI subprocess path
# =======================================================================

def crispasr_cli_run(
    bin_path: Path, model: Path, audio: Path, gpu_backend: str | None, threads: int,
    timeout: int = 120,
) -> tuple[float, str]:
    cmd = [
        str(bin_path),
        "--backend", "parakeet",
        "-m", str(model),
        "-f", str(audio),
        "--no-prints",
        "-bs", "1",
        "-t", str(threads),
    ]
    if gpu_backend == "cpu":
        cmd.append("-ng")
    elif gpu_backend and gpu_backend != "auto":
        cmd.extend(["--gpu-backend", gpu_backend])
    t0 = time.perf_counter()
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    el = time.perf_counter() - t0
    if r.returncode != 0:
        sys.stderr.write(r.stderr[-2000:] + "\n")
        raise RuntimeError(f"crispasr CLI failed (rc={r.returncode})")
    text = re.sub(r"\[[\d:.]+\s*-->\s*[\d:.]+\]\s*", "", r.stdout.strip()).strip()
    return el, text


# =======================================================================
# bench dataclasses
# =======================================================================

@dataclass
class CallStats:
    n_calls: int
    p50_ms: float
    p95_ms: float
    mean_ms: float

    @classmethod
    def from_seconds(cls, seconds: list[float]) -> "CallStats":
        ms = sorted(s * 1000.0 for s in seconds)
        n = len(ms)
        def pct(p: float) -> float:
            if n == 0:
                return 0.0
            k = max(0, min(n - 1, int(round((p / 100.0) * (n - 1)))))
            return ms[k]
        return cls(
            n_calls=n,
            p50_ms=pct(50),
            p95_ms=pct(95),
            mean_ms=sum(ms) / n if n else 0.0,
        )


@dataclass
class RunResult:
    engine: str
    quant: str
    mode: str         # "whole" | "chunked"
    audio: str        # "short" | "long" | basename
    audio_seconds: float
    load_s: float
    runs_s: list[float] = field(default_factory=list)
    call_stats: CallStats | None = None
    transcript_sample: str = ""
    wer: float | None = None
    extra: dict = field(default_factory=dict)

    @property
    def mean_run_s(self) -> float:
        return statistics.mean(self.runs_s) if self.runs_s else 0.0

    @property
    def realtime_factor(self) -> float:
        m = self.mean_run_s
        return self.audio_seconds / m if m > 0 else 0.0


# =======================================================================
# the actual benchmark
# =======================================================================

def bench_crispasr_ctypes(
    lib_path: Path, model_path: Path, audio_path: Path, audio_label: str, mode: str,
    window_s: float, warmups: int, runs: int, threads: int, use_flash: bool, ref_text: str,
) -> RunResult:
    pcm, dur = load_wav_pcm_f32(audio_path)
    quant = next((q for q, fn in GGUF_FILES.items() if model_path.name == fn), model_path.name)
    eng = CrispASRParakeet(lib_path, model_path, n_threads=threads, use_flash=use_flash)

    out = RunResult(
        engine="crispasr-ctypes",
        quant=quant,
        mode=mode,
        audio=audio_label,
        audio_seconds=dur,
        load_s=eng.load_s,
        extra={"flash_attn": use_flash, "threads": threads, "lib": str(lib_path)},
    )
    try:
        if mode == "whole":
            for _ in range(warmups):
                eng.transcribe(pcm, len(pcm))
            sample = ""
            for _ in range(runs):
                t0 = time.perf_counter()
                txt = eng.transcribe(pcm, len(pcm))
                out.runs_s.append(time.perf_counter() - t0)
                if not sample:
                    sample = txt
            out.transcript_sample = sample
        elif mode == "chunked":
            wsamps = int(window_s * 16000)
            chunks = chunk_indices(len(pcm), wsamps)
            # warm each chunk shape with one full warmup pass
            for _ in range(warmups):
                for lo, hi in chunks:
                    eng.transcribe_window(pcm, lo, hi)
            all_per_call = []
            sample = ""
            for _ in range(runs):
                t0 = time.perf_counter()
                buf = []
                for lo, hi in chunks:
                    c0 = time.perf_counter()
                    buf.append(eng.transcribe_window(pcm, lo, hi))
                    all_per_call.append(time.perf_counter() - c0)
                out.runs_s.append(time.perf_counter() - t0)
                if not sample:
                    sample = " ".join(s.strip() for s in buf if s.strip())
            out.transcript_sample = sample
            out.call_stats = CallStats.from_seconds(all_per_call)
        else:
            raise ValueError(f"unknown mode {mode!r}")
    finally:
        eng.close()

    if audio_label == "short":
        out.wer = wer_or_none(ref_text, out.transcript_sample)
    return out


def bench_crispasr_cli(
    bin_path: Path, model_path: Path, audio_path: Path, audio_label: str,
    gpu_backend: str | None, warmups: int, runs: int, threads: int, ref_text: str,
) -> RunResult:
    pcm_dur = load_wav_pcm_f32(audio_path)[1]
    quant = next((q for q, fn in GGUF_FILES.items() if model_path.name == fn), model_path.name)

    out = RunResult(
        engine="crispasr-cli",
        quant=quant,
        mode="whole",
        audio=audio_label,
        audio_seconds=pcm_dur,
        load_s=0.0,
        extra={"gpu_backend": gpu_backend or "auto", "threads": threads, "bin": str(bin_path)},
    )

    # CLI = process startup every run, no real "load" amortization possible
    # without building a long-lived server. Warmups pay the Metal-JIT cost
    # (cached in ~/Library after first compile), so subsequent runs are
    # closer to steady-state. This intentionally inflates the reported
    # number vs the ctypes path; the JSON keeps both so readers can see.
    sample = ""
    for _ in range(warmups):
        crispasr_cli_run(bin_path, model_path, audio_path, gpu_backend, threads)
    for _ in range(runs):
        el, txt = crispasr_cli_run(bin_path, model_path, audio_path, gpu_backend, threads)
        out.runs_s.append(el)
        if not sample:
            sample = txt
    out.transcript_sample = sample
    if audio_label == "short":
        out.wer = wer_or_none(ref_text, out.transcript_sample)
    return out


def bench_onnx(
    onnx_dir: Path, audio_path: Path, audio_label: str, mode: str, window_s: float,
    warmups: int, runs: int, providers: list[str], quantization: str | None, ref_text: str,
) -> RunResult:
    import onnx_asr  # type: ignore
    import numpy as np  # type: ignore

    # The onnxruntime CoreML EP can't load fp32 ONNX models with
    # external-data sidecars — tracked upstream as `onnxruntime#26355`
    # (closed *not planned*): the external-data initializer + CoreML's
    # subgraph split lose `model_path` along the way; inlining hits
    # protobuf's 2 GB ceiling. See PERFORMANCE.md "onnx-asr
    # cross-comparison — issue #81 (2026-05-09)" for the full table.
    # Workaround for users who want fp32 on CoreML: pre-merge into a
    # single file via `onnx.save(..., save_as_external_data=False)`
    # (one-time 2.4 GB rewrite per repo). We don't do that here
    # automatically; instead we auto-fall-back to CPU EP — which is
    # also the host default on M1 (CPU EP is *faster* than CoreML EP
    # for parakeet on M1, per the same PERFORMANCE.md section) — and
    # record what actually ran in `extra.providers_used` so the JSON
    # makes the substitution auditable.
    actual_providers = list(providers)
    fp32_external_data = (
        quantization is None
        and (onnx_dir / "encoder-model.onnx.data").is_file()
        and "CoreMLExecutionProvider" in providers
    )
    if fp32_external_data:
        actual_providers = [p for p in providers if p != "CoreMLExecutionProvider"]
        if "CPUExecutionProvider" not in actual_providers:
            actual_providers.append("CPUExecutionProvider")

    t0 = time.perf_counter()
    model = onnx_asr.load_model(
        "nemo-parakeet-tdt-0.6b-v3",
        path=onnx_dir,
        quantization=quantization,
        providers=actual_providers,
    )
    load_s = time.perf_counter() - t0

    pcm_np, dur = load_wav_pcm_f32(audio_path)

    out = RunResult(
        engine="onnx-asr",
        quant=quantization or "fp32",
        mode=mode,
        audio=audio_label,
        audio_seconds=dur,
        load_s=load_s,
        extra={
            "providers_requested": providers,
            "providers_used": actual_providers,
            "providers_fallback_reason": (
                # known onnxruntime CoreML EP issue with external-data fp32:
                # workaround is `onnx.save(..., save_as_external_data=False)`
                "fp32 ONNX + external data + CoreML EP fails graph optimization;"
                " fell back to CPU EP" if fp32_external_data else None
            ),
            "onnx_dir": str(onnx_dir),
        },
    )

    if mode == "whole":
        for _ in range(warmups):
            model.recognize(pcm_np)
        sample = ""
        for _ in range(runs):
            t0 = time.perf_counter()
            txt = model.recognize(pcm_np)
            out.runs_s.append(time.perf_counter() - t0)
            if not sample:
                sample = txt if isinstance(txt, str) else str(txt)
        out.transcript_sample = sample
    elif mode == "chunked":
        wsamps = int(window_s * 16000)
        chunks = chunk_indices(len(pcm_np), wsamps)
        for _ in range(warmups):
            for lo, hi in chunks:
                model.recognize(pcm_np[lo:hi])
        all_per_call = []
        sample = ""
        for _ in range(runs):
            t0 = time.perf_counter()
            buf = []
            for lo, hi in chunks:
                c0 = time.perf_counter()
                buf.append(model.recognize(pcm_np[lo:hi]))
                all_per_call.append(time.perf_counter() - c0)
            out.runs_s.append(time.perf_counter() - t0)
            if not sample:
                sample = " ".join(str(s).strip() for s in buf if str(s).strip())
        out.transcript_sample = sample
        out.call_stats = CallStats.from_seconds(all_per_call)
    else:
        raise ValueError(f"unknown mode {mode!r}")

    if audio_label == "short":
        out.wer = wer_or_none(ref_text, out.transcript_sample)
    return out


# =======================================================================
# CLI
# =======================================================================

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Head-to-head Parakeet TDT 0.6B v3 benchmark: onnx-asr vs CrispASR",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--engine", choices=("both", "onnx", "crispasr"), default="both")
    ap.add_argument("--mode", choices=("both", "whole", "chunked"), default="both")
    ap.add_argument("--audio", choices=("both", "short", "long"), default="both")
    ap.add_argument("--audio-path", help="explicit 16 kHz mono wav (overrides --audio)")
    ap.add_argument("--gguf-quants", default="q4_k,q8_0,f16",
                    help="comma list: q4_k,q8_0,f16 (default: all three)")
    ap.add_argument("--onnx-quants", default="int8,fp32",
                    help="comma list: int8,fp32 (default: both)")
    ap.add_argument("--window-s", type=float, default=4.0,
                    help="streaming window for chunked mode (default 4.0 s)")
    ap.add_argument("--warmups", type=int, default=1)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--threads", type=int, default=4)

    ap.add_argument("--crispasr-call", choices=("both", "ctypes", "cli"), default="ctypes",
                    help="how to invoke CrispASR: ctypes (recommended), cli, or both")
    ap.add_argument("--crispasr-bin", default=str(DEFAULT_CRISPASR_BIN))
    ap.add_argument("--crispasr-lib", default=str(host_default_lib()))
    ap.add_argument("--gpu-backend", default=None,
                    help="CrispASR --gpu-backend: auto|metal|cuda|vulkan|cpu (CLI mode only)")
    ap.add_argument("--no-flash-attn", action="store_true", help="disable flash attention (ctypes)")

    ap.add_argument("--providers", default=None,
                    help="comma list of onnxruntime providers; default = host best")
    ap.add_argument("--gguf-dir", default=str(GGUF_DIR))
    ap.add_argument("--onnx-dir", default=str(ONNX_DIR))

    ap.add_argument("--json", default=None, help="path to write raw JSON results")
    ap.add_argument("--prewarm", action="store_true",
                    help=("before the matrix, run one transcribe on each audio shape "
                          "with each crispasr quant so all Metal/CUDA pipeline kernels "
                          "JIT-compile up front. Reduces first-cell variance dramatically."))
    ap.add_argument("--quiet", action="store_true")
    return ap.parse_args()


def expand_audios(args, ref_text: str) -> list[tuple[str, Path]]:
    if args.audio_path:
        return [(Path(args.audio_path).name, Path(args.audio_path))]
    out = []
    if args.audio in ("both", "short"):
        out.append(("short", SAMPLES_JFK))
    if args.audio in ("both", "long"):
        long_p = ensure_long_wav(DEFAULT_LONG_WAV, target_s=60.0)
        out.append(("long", long_p))
    return out


def expand_modes(args) -> list[str]:
    return ["whole", "chunked"] if args.mode == "both" else [args.mode]


def expand_crispasr_calls(args) -> list[str]:
    return ["ctypes", "cli"] if args.crispasr_call == "both" else [args.crispasr_call]


def fmt_table(results: list[RunResult]) -> str:
    rows = []
    rows.append(
        "| engine | quant | mode | audio | dur | load | mean run | RTx | p50 | p95 | calls | WER | sample |"
    )
    rows.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in results:
        cs = r.call_stats
        rows.append(
            "| {engine} | {quant} | {mode} | {audio} | {dur:.1f}s | {load:.2f}s | {mr:.3f}s | "
            "{rt:.1f}× | {p50} | {p95} | {nc} | {wer} | {sample} |".format(
                engine=r.engine,
                quant=r.quant,
                mode=r.mode,
                audio=r.audio,
                dur=r.audio_seconds,
                load=r.load_s,
                mr=r.mean_run_s,
                rt=r.realtime_factor,
                p50=f"{cs.p50_ms:.1f}ms" if cs else "—",
                p95=f"{cs.p95_ms:.1f}ms" if cs else "—",
                nc=cs.n_calls if cs else "—",
                wer=(f"{r.wer:.3f}" if r.wer is not None else "—"),
                sample=(r.transcript_sample[:48].replace("|", "/") + ("…" if len(r.transcript_sample) > 48 else "")),
            )
        )
    return "\n".join(rows)


def main() -> int:
    args = parse_args()
    if args.providers:
        providers = [p.strip() for p in args.providers.split(",") if p.strip()]
    else:
        providers = host_default_providers()
    print(f"host: {platform.system()} {platform.machine()}  python={sys.version.split()[0]}")
    print(f"onnx providers: {providers}")
    print(f"crispasr lib:   {args.crispasr_lib}")
    print(f"crispasr bin:   {args.crispasr_bin}")

    audios = expand_audios(args, JFK_REF)
    modes = expand_modes(args)
    crispasr_calls = expand_crispasr_calls(args)

    gguf_quants = [q.strip() for q in args.gguf_quants.split(",") if q.strip()]
    onnx_quants = [q.strip() for q in args.onnx_quants.split(",") if q.strip()]
    onnx_quants = [None if q == "fp32" else q for q in onnx_quants]

    # Pre-flight: ensure all needed assets exist before timing anything.
    if args.engine in ("both", "crispasr"):
        models = {q: ensure_gguf(q, Path(args.gguf_dir)) for q in gguf_quants}
    else:
        models = {}
    if args.engine in ("both", "onnx"):
        ensure_onnx(Path(args.onnx_dir))

    if args.prewarm and args.engine in ("both", "crispasr"):
        # JIT every (quant × {whole-shape, chunk-shape}) combo before timing.
        # Metal pipeline kernels are shape-specialized: the first transcribe of
        # a new (n_samples) hits 10-30 s of pipeline compile cost. Without this
        # pass, the FIRST cell of the matrix absorbs the entire JIT bill, and
        # comparisons between cells get distorted by where the cache happens
        # to be warm. With it, every timed cell starts on a hot cache.
        print("\n=== prewarm: JIT every shape before timing ===")
        for quant, model in models.items():
            t0 = time.perf_counter()
            eng = CrispASRParakeet(Path(args.crispasr_lib), model,
                                   n_threads=args.threads, use_flash=not args.no_flash_attn)
            try:
                for audio_label, audio_path in audios:
                    pcm_np, _ = load_wav_pcm_f32(audio_path)
                    # whole-audio shape
                    eng.transcribe(pcm_np)
                    # chunk shape
                    wsamps = int(args.window_s * 16000)
                    if len(pcm_np) > wsamps:
                        eng.transcribe_window(pcm_np, 0, wsamps)
            finally:
                eng.close()
            print(f"  prewarmed {quant} in {time.perf_counter() - t0:.1f}s")

    results: list[RunResult] = []

    for audio_label, audio_path in audios:
        print(f"\n=== audio: {audio_label} ({audio_path}) ===")

        def _record(label: str, fn) -> None:
            """Run one bench cell. Catches exceptions so a single broken
            (engine, quant, mode) cell doesn't kill the whole matrix —
            the remaining cells should still get measured and the JSON
            sidecar should still be written."""
            print(f"  {label}", flush=True)
            try:
                r = fn()
            except Exception as exc:
                msg = f"{type(exc).__name__}: {exc}"[:280]
                print(f"    FAIL  {msg}")
                results.append(RunResult(
                    engine=label.split()[0], quant="?", mode="?",
                    audio=audio_label, audio_seconds=0.0, load_s=0.0,
                    extra={"error": msg, "label": label},
                ))
                return
            results.append(r)
            cs = r.call_stats
            line = f"    mean={r.mean_run_s:.3f}s  RT={r.realtime_factor:.1f}×"
            if cs:
                line += f"  p50={cs.p50_ms:.1f}ms  p95={cs.p95_ms:.1f}ms  calls={cs.n_calls}"
            print(line)

        if args.engine in ("both", "crispasr"):
            for quant in gguf_quants:
                model = models[quant]
                for mode in modes:
                    for call in crispasr_calls:
                        if call == "ctypes":
                            _record(
                                f"crispasr-ctypes  {quant:<5} {mode:<7}",
                                lambda quant=quant, mode=mode, model=model: bench_crispasr_ctypes(
                                    Path(args.crispasr_lib), model, audio_path, audio_label, mode,
                                    args.window_s, args.warmups, args.runs, args.threads,
                                    use_flash=not args.no_flash_attn, ref_text=JFK_REF,
                                ),
                            )
                        else:
                            if mode == "chunked":
                                # CLI doesn't expose per-chunk timing without
                                # --stream + stdin plumbing; skip for now.
                                continue
                            _record(
                                f"crispasr-cli     {quant:<5} {mode:<7}",
                                lambda quant=quant, mode=mode, model=model: bench_crispasr_cli(
                                    Path(args.crispasr_bin), model, audio_path, audio_label,
                                    args.gpu_backend, args.warmups, args.runs, args.threads,
                                    ref_text=JFK_REF,
                                ),
                            )

        if args.engine in ("both", "onnx"):
            for q in onnx_quants:
                qlabel = q or "fp32"
                for mode in modes:
                    _record(
                        f"onnx-asr         {qlabel:<5} {mode:<7}",
                        lambda q=q, mode=mode: bench_onnx(
                            Path(args.onnx_dir), audio_path, audio_label, mode, args.window_s,
                            args.warmups, args.runs, providers, q, ref_text=JFK_REF,
                        ),
                    )

    print("\n" + fmt_table(results))

    if args.json:
        payload = {
            "host": {
                "system": platform.system(),
                "machine": platform.machine(),
                "python": sys.version.split()[0],
            },
            "providers": providers,
            "results": [
                {
                    **{k: v for k, v in asdict(r).items() if k != "call_stats"},
                    "call_stats": asdict(r.call_stats) if r.call_stats else None,
                    "mean_run_s": r.mean_run_s,
                    "realtime_factor": r.realtime_factor,
                }
                for r in results
            ],
        }
        Path(args.json).write_text(json.dumps(payload, indent=2))
        print(f"\nwrote {args.json}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
