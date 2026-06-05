#!/usr/bin/env python3
"""
benchmark_vitw_beam.py — beam_size quality + speed benchmark on Voices-in-the-Wild-Bench
==========================================================================================

Uses zhifeixie/Voices-in-the-Wild-Bench (1.75 GB, 8 acoustic conditions ×
real/syn × EN/ZH) to compare beam=1 vs beam=2 vs beam=4 on noisy/degraded
audio where greedy decoding may fail.

The Bench dataset is the evaluation split from the Qwen3 Mega-ASR finetune
family; its 8 conditions (noise, far-field, obstructed, distortion, recording,
echo, dropout, mixed) provide the acoustic diversity needed to expose
beam search quality differences that JFK (clean canonical speech) cannot.

Usage:
    # Default: qwen3-asr, 5 samples/split, real_noise+syn_noise+real_mixed
    python tools/benchmark_vitw_beam.py

    # Full sweep: three backends, all English splits, 10 samples each
    python tools/benchmark_vitw_beam.py --backends qwen3,granite-4.1,voxtral \\
        --splits all --n 10 --beams 1,2,4

    # Quick sanity: just noise, 3 samples, no warmup
    python tools/benchmark_vitw_beam.py --splits real_noise --n 3 --no-warmup

    # Save JSON results
    python tools/benchmark_vitw_beam.py --json vitw_beam_results.json

Environment:
    CRISPASR_BIN    — path to crispasr binary (default: build-ninja-compile/bin/crispasr)
    CRISPASR_QWEN3_MODEL   — override qwen3-asr model path
    CRISPASR_GRANITE_MODEL — override granite-4.1 model path
    CRISPASR_VOXTRAL_MODEL — override voxtral model path
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import wave
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO = Path(__file__).resolve().parents[1]
HF_HOME = "/Volumes/backups/ai/huggingface-hub"
TMPDIR = "/Volumes/backups/ai/tmp/vitw_bench"
MODEL_DIR = Path("/Volumes/backups/ai/crispasr")

SAMPLE_JFK = str(REPO / "samples" / "jfk.wav")

BIN = os.environ.get(
    "CRISPASR_BIN",
    str(REPO / "build-ninja-compile" / "bin" / "crispasr"),
)


def _model(env_var: str, filename: str) -> str:
    from_env = os.environ.get(env_var, "")
    if from_env and Path(from_env).exists():
        return from_env
    p = MODEL_DIR / filename
    return str(p) if p.exists() else ""


BACKENDS: dict[str, dict] = {
    "qwen3": {
        "model": _model("CRISPASR_QWEN3_MODEL", "qwen3-asr-0.6b-q4_k.gguf"),
        "cli_backend": "qwen3",
    },
    "granite-4.1": {
        "model": _model("CRISPASR_GRANITE_MODEL", "granite-speech-4.1-2b-q4_k.gguf"),
        "cli_backend": "granite-4.1",
    },
    "voxtral": {
        "model": _model("CRISPASR_VOXTRAL_MODEL", "voxtral-mini-3b-2507-q4_k.gguf"),
        "cli_backend": "voxtral",
    },
}

ALL_SPLITS = [
    "real_noise", "syn_noise",
    "real_far_field", "syn_far_field",
    "real_obstructed", "syn_obstructed",
    "real_distortion", "syn_distortion",
    "real_recording", "syn_recording",
    "real_echo", "syn_echo",
    "real_dropout", "syn_dropout",
    "real_mixed", "syn_mixed",
]

DEFAULT_SPLITS = ["real_noise", "syn_noise", "real_mixed", "real_far_field"]

# ---------------------------------------------------------------------------
# WER calculation
# ---------------------------------------------------------------------------


def _normalize(text: str) -> list[str]:
    text = text.lower()
    text = re.sub(r"[^\w\s']", " ", text)
    return text.split()


def wer(reference: str, hypothesis: str) -> float:
    r = _normalize(reference)
    h = _normalize(hypothesis)
    if not r:
        return 0.0 if not h else 1.0
    # Standard edit-distance DP
    d = list(range(len(h) + 1))
    for i, rw in enumerate(r):
        prev = d[:]
        d[0] = i + 1
        for j, hw in enumerate(h):
            d[j + 1] = min(prev[j] + (0 if rw == hw else 1),
                           d[j] + 1, prev[j + 1] + 1)
    return d[len(h)] / len(r)


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------


def _is_english(text: str) -> bool:
    # Reject if more than 5% of non-space characters are CJK
    chars = [c for c in text if not c.isspace()]
    if not chars:
        return False
    cjk = sum(1 for c in chars if "一" <= c <= "鿿")
    return cjk / len(chars) < 0.05


def _save_wav(pcm_float: list[float], sr: int, path: str) -> None:
    import struct
    pcm16 = [max(-32768, min(32767, int(s * 32767))) for s in pcm_float]
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(struct.pack(f"<{len(pcm16)}h", *pcm16))


# ---------------------------------------------------------------------------
# Inference runner
# ---------------------------------------------------------------------------


@dataclass
class RunResult:
    backend: str
    beam_size: int
    split: str
    sample_idx: int
    reference: str
    hypothesis: str
    wer_score: float
    elapsed_s: float


def run_crispasr(wav_path: str, backend_name: str, beam_size: int) -> tuple[str, float]:
    """Returns (transcription, elapsed_s).

    The CLI writes transcription to stdout and all init/debug logs to stderr,
    so we use only stdout for the hypothesis.
    """
    cfg = BACKENDS[backend_name]
    cmd = [
        BIN,
        "--backend", cfg["cli_backend"],
        "-m", cfg["model"],
        "--beam-size", str(beam_size),
        wav_path,
    ]
    t0 = time.perf_counter()
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    elapsed = time.perf_counter() - t0
    if r.returncode != 0:
        raise RuntimeError(f"crispasr failed (rc={r.returncode}):\n{r.stderr[:500]}")
    # Stdout is the transcription; stderr is init/debug logs.
    # Strip any blank lines or stray timing annotations the CLI may emit.
    lines = [l for l in r.stdout.splitlines()
             if l.strip() and not l.startswith("[") and not l.startswith("crispasr:")]
    return " ".join(lines).strip(), elapsed


def warmup(backend_name: str) -> float:
    """One greedy run on JFK to compile Metal pipelines. Returns elapsed_s."""
    if not os.path.exists(SAMPLE_JFK):
        print(f"  [warmup] jfk.wav not found at {SAMPLE_JFK}, skipping warmup")
        return 0.0
    print(f"  [warmup] {backend_name} — compiling Metal pipelines (this takes a while)...")
    _, elapsed = run_crispasr(SAMPLE_JFK, backend_name, beam_size=1)
    print(f"  [warmup] done in {elapsed:.1f}s")
    return elapsed


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------


def load_split_samples(split: str, n: int) -> list[dict]:
    """Load up to n English samples from the given split (streaming, no disk cache)."""
    try:
        from datasets import load_dataset
    except ImportError:
        sys.exit("datasets not installed — run: pip install datasets")

    print(f"  Streaming split '{split}'...", end=" ", flush=True)
    ds = load_dataset("zhifeixie/Voices-in-the-Wild-Bench", split=split,
                      streaming=True)
    print("ok", flush=True)

    samples = []
    for row in ds:
        ref = row.get("answer") or row.get("text") or ""
        if not _is_english(ref):
            continue
        audio = row["audio"]
        pcm = audio["array"]
        sr = audio["sampling_rate"]
        if hasattr(pcm, "ndim") and pcm.ndim == 2:
            pcm = pcm.mean(axis=0)
        samples.append({"pcm": pcm.tolist() if hasattr(pcm, "tolist") else list(pcm),
                        "sr": sr, "reference": ref.strip()})
        if len(samples) >= n:
            break

    print(f"  → {len(samples)} English samples selected", flush=True)
    return samples


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--backends", default="qwen3",
                   help="comma-separated backend names (default: qwen3). "
                        "Available: " + ", ".join(BACKENDS))
    p.add_argument("--beams", default="1,2,4",
                   help="comma-separated beam sizes (default: 1,2,4)")
    p.add_argument("--splits", default=",".join(DEFAULT_SPLITS),
                   help="comma-separated split names, or 'all' (default: "
                        + ",".join(DEFAULT_SPLITS) + ")")
    p.add_argument("--n", type=int, default=5,
                   help="English samples per split (default: 5)")
    p.add_argument("--no-warmup", action="store_true",
                   help="skip Metal warmup run (times will include pipeline compilation)")
    p.add_argument("--json", metavar="PATH",
                   help="write raw results JSON to this path")
    return p.parse_args()


def _table_row(label: str, values: list[str], widths: list[int]) -> str:
    cells = [label.ljust(widths[0])] + [v.rjust(widths[i + 1]) for i, v in enumerate(values)]
    return "| " + " | ".join(cells) + " |"


def print_summary(results: list[RunResult], beams: list[int], backends: list[str]) -> None:
    print("\n" + "=" * 72)
    print("RESULTS — WER by condition (lower is better)")
    print("=" * 72)

    splits_seen = list(dict.fromkeys(r.split for r in results))
    beam_headers = [f"beam={b}" for b in beams]

    for backend in backends:
        print(f"\n### {backend}")
        col_w = [22] + [max(8, len(h)) for h in beam_headers] + [8]
        header = _table_row("split", beam_headers + ["avg_t(s)"], col_w)
        sep = "|-" + "-|-".join("-" * w for w in col_w) + "-|"
        print(header)
        print(sep)

        for split in splits_seen:
            row_vals: list[str] = []
            avg_t = 0.0
            count = 0
            for beam in beams:
                rs = [r for r in results
                      if r.backend == backend and r.split == split and r.beam_size == beam]
                if not rs:
                    row_vals.append("—")
                    continue
                avg_wer = sum(r.wer_score for r in rs) / len(rs)
                avg_t += sum(r.elapsed_s for r in rs) / len(rs)
                count += 1
                row_vals.append(f"{avg_wer:.3f}")
            avg_t_str = f"{avg_t / count:.1f}" if count else "—"
            print(_table_row(split, row_vals + [avg_t_str], col_w))

    print("\n" + "=" * 72)
    print("SPEED — mean latency per call (seconds)")
    print("=" * 72)
    for backend in backends:
        print(f"\n### {backend}")
        col_w = [22] + [max(8, len(f"beam={b}")) for b in beams]
        header = _table_row("split", [f"beam={b}" for b in beams], col_w)
        sep = "|-" + "-|-".join("-" * w for w in col_w) + "-|"
        print(header)
        print(sep)
        for split in splits_seen:
            row_vals = []
            for beam in beams:
                rs = [r for r in results
                      if r.backend == backend and r.split == split and r.beam_size == beam]
                row_vals.append(f"{sum(r.elapsed_s for r in rs)/len(rs):.2f}" if rs else "—")
            print(_table_row(split, row_vals, col_w))

    # Sample output comparison for worst WER cases
    print("\n" + "=" * 72)
    print("EXAMPLE TRANSCRIPTIONS (worst WER samples per backend)")
    print("=" * 72)
    for backend in backends:
        brs = [r for r in results if r.backend == backend]
        if not brs:
            continue
        # Find sample with highest WER difference between beam=1 and beam=max
        max_beam = max(beams)
        candidates = []
        for r1 in brs:
            if r1.beam_size != 1:
                continue
            rb = next((r for r in brs if r.beam_size == max_beam
                        and r.split == r1.split and r.sample_idx == r1.sample_idx), None)
            if rb:
                candidates.append((r1.wer_score - rb.wer_score, r1, rb))
        if not candidates:
            continue
        candidates.sort(key=lambda x: x[0], reverse=True)
        # Show top 3 cases where beam helped most
        shown = 0
        for delta, r1, rb in candidates:
            if shown >= 3:
                break
            if abs(delta) < 0.01:
                continue
            print(f"\n  [{backend} | {r1.split} | sample {r1.sample_idx}]"
                  f"  WER beam=1: {r1.wer_score:.3f}  beam={max_beam}: {rb.wer_score:.3f}")
            print(f"  REF: {r1.reference}")
            print(f"  b=1: {r1.hypothesis}")
            print(f"  b={max_beam}: {rb.hypothesis}")
            shown += 1
        if shown == 0:
            print(f"\n  [{backend}] beam search produced no measurable WER improvement on this sample set.")


def main() -> None:
    args = parse_args()

    backends = [b.strip() for b in args.backends.split(",") if b.strip()]
    beams = [int(b) for b in args.beams.split(",") if b.strip()]
    splits = ALL_SPLITS if args.splits == "all" else [s.strip() for s in args.splits.split(",") if s.strip()]

    # Validate
    for b in backends:
        if b not in BACKENDS:
            sys.exit(f"Unknown backend '{b}'. Available: {', '.join(BACKENDS)}")
        if not BACKENDS[b]["model"]:
            sys.exit(f"Model not found for backend '{b}'. "
                     f"Set CRISPASR_{b.upper().replace('-','_')}_MODEL or place in {MODEL_DIR}")
    if not os.path.exists(BIN):
        sys.exit(f"crispasr binary not found at {BIN}. Set CRISPASR_BIN or build first.")

    print(f"Backends : {backends}")
    print(f"Beams    : {beams}")
    print(f"Splits   : {splits}")
    print(f"N/split  : {args.n}")
    print(f"Warmup   : {'no' if args.no_warmup else 'yes'}")
    print()

    # Warmup phase
    if not args.no_warmup:
        for backend in backends:
            warmup(backend)
        print()

    # Load all samples first (download happens here)
    print("Loading dataset samples...")
    split_samples: dict[str, list[dict]] = {}
    for split in splits:
        split_samples[split] = load_split_samples(split, args.n)
    print()

    # Ensure temp dir exists
    os.makedirs(TMPDIR, exist_ok=True)

    results: list[RunResult] = []
    total = len(backends) * len(beams) * sum(len(v) for v in split_samples.values())
    done = 0

    for backend in backends:
        print(f"\n{'─' * 60}")
        print(f"Backend: {backend}")
        print(f"{'─' * 60}")

        for split, samples in split_samples.items():
            print(f"  Split: {split}  ({len(samples)} samples)")

            for beam in beams:
                wers = []
                times = []

                for idx, sample in enumerate(samples):
                    wav_path = os.path.join(TMPDIR, f"vitw_{split}_{idx}.wav")
                    _save_wav(sample["pcm"], sample["sr"], wav_path)

                    try:
                        hyp, elapsed = run_crispasr(wav_path, backend, beam)
                    except Exception as e:
                        print(f"    [WARN] {backend} beam={beam} sample {idx}: {e}")
                        hyp, elapsed = "", 0.0

                    w = wer(sample["reference"], hyp)
                    wers.append(w)
                    times.append(elapsed)

                    results.append(RunResult(
                        backend=backend, beam_size=beam, split=split, sample_idx=idx,
                        reference=sample["reference"], hypothesis=hyp,
                        wer_score=w, elapsed_s=elapsed,
                    ))
                    done += 1
                    print(f"    beam={beam} sample {idx}: WER={w:.3f} t={elapsed:.1f}s  | {hyp[:60]}")

                avg_wer = sum(wers) / len(wers) if wers else 0.0
                avg_t = sum(times) / len(times) if times else 0.0
                print(f"    → beam={beam}  avg WER={avg_wer:.3f}  avg t={avg_t:.1f}s"
                      f"  [{done}/{total}]")

    print_summary(results, beams, backends)

    if args.json:
        with open(args.json, "w") as f:
            json.dump([asdict(r) for r in results], f, indent=2)
        print(f"\nRaw results written to {args.json}")


if __name__ == "__main__":
    main()
