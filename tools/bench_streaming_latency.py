"""Streaming-latency bench harness (PLAN #7 / #62c validation).

Drives `crispasr_session_stream_open` + feed + flush + get_text on a
voxtral4b / kyutai-stt / moonshine-streaming session, measures per-stage
wall-clock, and (with --check-batch-equality) compares the streaming
transcript byte-for-byte against the single-shot batch transcribe.

Phase 1 of PLAN #7 has decode-on-flush semantics: there's one decode
call inside flush(), and get_text() returns the accumulated transcript
at once. Per-token timing inside the decode loop is exposed via the
`CRISPASR_VOXTRAL4B_BENCH=1` env var (printed to stderr by libcrispasr
when set).
"""
from __future__ import annotations

import argparse
import os
import statistics
import sys
import time
import wave
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "python"))

import crispasr  # noqa: E402


def load_wav_16k_mono(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as w:
        sr = w.getframerate()
        n_ch = w.getnchannels()
        sw = w.getsampwidth()
        n = w.getnframes()
        raw = w.readframes(n)
    if sw != 2:
        raise SystemExit(f"expected 16-bit PCM, got {sw*8}-bit")
    pcm = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if n_ch > 1:
        pcm = pcm.reshape(-1, n_ch).mean(axis=1)
    if sr != 16000:
        # Quick linear resample — bench precision is fine for this; bit-exact
        # gate uses the original sample rate the model expects.
        ratio = 16000 / sr
        new_n = int(pcm.size * ratio)
        idx = np.linspace(0, pcm.size - 1, new_n)
        pcm = np.interp(idx, np.arange(pcm.size), pcm).astype(np.float32)
    return pcm


def repeat_to_duration(pcm: np.ndarray, target_seconds: float) -> np.ndarray:
    cur = pcm.size / 16000.0
    if cur >= target_seconds:
        return pcm[: int(target_seconds * 16000)]
    n_repeats = int(np.ceil(target_seconds / cur))
    return np.tile(pcm, n_repeats)[: int(target_seconds * 16000)]


def percentiles(xs: list[float], ps: list[int]) -> dict[int, float]:
    if not xs:
        return {p: float("nan") for p in ps}
    xs_sorted = sorted(xs)
    out = {}
    for p in ps:
        k = (len(xs_sorted) - 1) * p / 100.0
        f = int(k)
        c = min(f + 1, len(xs_sorted) - 1)
        out[p] = xs_sorted[f] + (xs_sorted[c] - xs_sorted[f]) * (k - f)
    return out


def run_one(*, backend: str, model_path: str, pcm: np.ndarray, chunk_ms: int) -> dict:
    chunk_n = int(chunk_ms * 16)
    t0 = time.perf_counter()
    sess = crispasr.Session(backend=backend, model_path=model_path)
    t_open = time.perf_counter() - t0

    t0 = time.perf_counter()
    stream = sess.stream_open(step_ms=chunk_ms, length_ms=15000)
    t_stream_open = time.perf_counter() - t0

    feed_durations: list[float] = []
    t_feed_start = time.perf_counter()
    for i in range(0, pcm.size, chunk_n):
        chunk = pcm[i : i + chunk_n]
        t = time.perf_counter()
        stream.feed(chunk)
        feed_durations.append(time.perf_counter() - t)
    t_feed_total = time.perf_counter() - t_feed_start

    t = time.perf_counter()
    stream.flush()
    t_flush = time.perf_counter() - t

    t = time.perf_counter()
    out = stream.get_text()
    t_get_text = time.perf_counter() - t

    stream.close()
    sess.close()
    return {
        "transcript": out["text"],
        "t_open": t_open,
        "t_stream_open": t_stream_open,
        "t_feed_total": t_feed_total,
        "t_flush": t_flush,
        "t_get_text": t_get_text,
        "feed_durations": feed_durations,
        "audio_seconds": pcm.size / 16000.0,
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--backend", required=True, help="voxtral4b | kyutai-stt | moonshine-streaming")
    p.add_argument("-m", "--model", required=True, help="GGUF model path")
    p.add_argument(
        "--clip", default=str(REPO_ROOT / "samples" / "jfk.wav"),
        help="Source WAV (default: samples/jfk.wav)"
    )
    p.add_argument("--chunk-ms", type=int, default=80, help="Streaming chunk size (ms)")
    p.add_argument("--clip-duration", type=float, default=0.0,
                   help="If >0, repeat the clip to reach this duration (seconds)")
    p.add_argument("--runs", type=int, default=3, help="Number of warm timing runs")
    p.add_argument("--check-batch-equality", action="store_true",
                   help="Run single-shot transcribe and assert streaming matches.")
    args = p.parse_args()

    pcm = load_wav_16k_mono(Path(args.clip))
    if args.clip_duration > 0:
        pcm = repeat_to_duration(pcm, args.clip_duration)
    print(f"[bench] clip={args.clip} duration={pcm.size/16000.0:.2f}s chunk_ms={args.chunk_ms}")

    if args.check_batch_equality:
        print("[bench] running batch baseline (via CLI for voxtral4b parity)...")
        # The python session.transcribe() path goes through run_voxtral_family,
        # which doesn't pre-pad audio the way the CLI adapter does, so it
        # crashes on arbitrary input sizes. Use the CLI subprocess as the
        # ground-truth batch baseline.
        import subprocess
        cli = REPO_ROOT / "build-ninja-compile" / "bin" / "crispasr"
        if not cli.exists():
            raise SystemExit(f"crispasr CLI not found at {cli}")
        t = time.perf_counter()
        proc = subprocess.run(
            [str(cli), "--backend", args.backend, "-m", args.model, "-f", args.clip],
            capture_output=True, text=True, timeout=300,
        )
        elapsed = time.perf_counter() - t
        if proc.returncode != 0:
            raise SystemExit(f"CLI batch run failed: {proc.stderr}")
        # Last non-empty stdout line is the transcript (per crispasr CLI convention).
        lines = [ln.strip() for ln in proc.stdout.splitlines() if ln.strip()]
        batch_text = lines[-1] if lines else ""
        print(f"[bench] batch transcript: {batch_text!r}")
        print(f"[bench] batch took {elapsed*1000:.0f}ms ({pcm.size/16000.0/elapsed:.2f}x realtime)")

    print(f"[bench] running {args.runs} streaming run(s)...")
    transcripts: list[str] = []
    feed_totals: list[float] = []
    flush_times: list[float] = []
    avg_feed_per_chunk: list[float] = []
    for r in range(args.runs):
        result = run_one(backend=args.backend, model_path=args.model, pcm=pcm, chunk_ms=args.chunk_ms)
        transcripts.append(result["transcript"].strip())
        feed_totals.append(result["t_feed_total"])
        flush_times.append(result["t_flush"])
        if result["feed_durations"]:
            avg_feed_per_chunk.append(statistics.mean(result["feed_durations"]))
        print(f"  run {r}: feed={result['t_feed_total']*1000:.0f}ms "
              f"flush={result['t_flush']*1000:.0f}ms "
              f"transcript={result['transcript'].strip()!r}")

    print(f"[bench] feed-total p50/p95: "
          f"{percentiles(feed_totals, [50, 95])[50]*1000:.0f}/"
          f"{percentiles(feed_totals, [50, 95])[95]*1000:.0f} ms")
    print(f"[bench] flush p50/p95:      "
          f"{percentiles(flush_times, [50, 95])[50]*1000:.0f}/"
          f"{percentiles(flush_times, [50, 95])[95]*1000:.0f} ms")
    if avg_feed_per_chunk:
        print(f"[bench] avg per-chunk feed time: "
              f"{statistics.mean(avg_feed_per_chunk)*1000:.1f} ms (chunk_ms={args.chunk_ms})")

    if args.check_batch_equality:
        print()
        for i, t in enumerate(transcripts):
            ok = (t == batch_text)
            note = "MATCH" if ok else "DIVERGE"
            print(f"[bench] run {i}: {note}")
            if not ok:
                print(f"  batch:    {batch_text!r}")
                print(f"  stream:   {t!r}")
        all_ok = all(t == batch_text for t in transcripts)
        if not all_ok:
            return 1
        print("[bench] BIT-EXACT-BATCH: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
