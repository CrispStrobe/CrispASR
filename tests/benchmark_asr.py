#!/usr/bin/env python3
"""benchmark_asr.py — multi-backend ASR benchmark driver for CrispASR.

Runs the crispasr CLI across backends, audio files, and settings
combinations, collecting structured metrics that expose regressions
like issue #89 (parakeet silently losing content on long audio).

Usage::

    # Issue triage — reporter runs this on their audio:
    python tests/benchmark_asr.py --audio myfile.wav --backend parakeet

    # Full corpus matrix:
    python tests/benchmark_asr.py --corpus /mnt/storage/test-audio/corpus.json

    # Specific backend + settings:
    python tests/benchmark_asr.py --audio f.wav --backend parakeet --settings auto,chunk-30

    # Resume interrupted run:
    python tests/benchmark_asr.py --corpus /mnt/storage/test-audio/corpus.json --resume

Results are appended to /mnt/storage/benchmark-results/runs.jsonl and
a human-readable summary table is printed to stdout.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

# Allow importing sibling modules when run from tests/ or repo root
sys.path.insert(0, str(Path(__file__).parent))

from benchmark_metrics import metrics_from_json, CoverageMetrics

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RESULTS_DIR = Path("/mnt/storage/benchmark-results")
RESULTS_FILE = RESULTS_DIR / "runs.jsonl"

SETTINGS_MATRIX = [
    {"label": "auto",        "args": []},
    {"label": "vad-silero",  "args": ["--vad"]},
    {"label": "vad-firered", "args": ["--vad", "--vad-model", "firered"]},
    {"label": "chunk-15",    "args": ["--chunk-seconds", "15"]},
    {"label": "chunk-30",    "args": ["--chunk-seconds", "30"]},
    {"label": "chunk-60",    "args": ["--chunk-seconds", "60"]},
]

# backend → {model_glob patterns to search, languages supported}
BACKEND_MODELS = {
    "parakeet": {
        "globs": ["parakeet-tdt-0.6b-v3/parakeet-tdt-0.6b-v3.gguf",
                   "parakeet-tdt-0.6b-v3.gguf"],
        "langs": ["en", "de"],
    },
    "parakeet-ja": {
        "globs": ["parakeet-tdt-0.6b-ja.gguf"],
        "langs": ["ja"],
    },
    "whisper-small": {
        "globs": ["models/ggml-small.bin", "../whisper.cpp/models/ggml-small.bin"],
        "langs": ["en", "de", "ja", "zh"],
    },
    "moonshine-ja": {
        "globs": ["moonshine-base-ja.gguf"],
        "langs": ["ja"],
    },
    "moonshine-zh": {
        "globs": ["moonshine-base-zh.gguf"],
        "langs": ["zh"],
    },
    "cohere": {
        "globs": ["cohere/cohere-transcribe.gguf", "cohere-transcribe.gguf"],
        "langs": ["en", "de", "ja", "zh"],
    },
    "funasr-mlt": {
        "globs": ["funasr-mlt-nano/funasr-mlt-nano-2512-f16.gguf"],
        "langs": ["en", "ja", "zh"],
    },
    "parakeet-ctc": {
        "globs": ["parakeet-ctc-1.1b/parakeet-ctc-1.1b.gguf"],
        "langs": ["en"],
    },
    "canary": {
        "globs": ["cache/canary-1b-v2-q4_k.gguf", "canary-1b-v2-q4_k.gguf"],
        "langs": ["en", "de", "ja", "zh"],
    },
    "sensevoice": {
        "globs": ["cache/sensevoice-small-q4_k.gguf", "sensevoice-small-q4_k.gguf"],
        "langs": ["en", "de", "ja", "zh"],
    },
    "paraformer": {
        "globs": ["cache/paraformer-zh-q4_k.gguf", "paraformer-zh-q4_k.gguf"],
        "langs": ["zh"],
    },
}

MODELS_DIR = Path(os.environ.get("BENCHMARK_MODELS_DIR", "/mnt/storage"))


# ---------------------------------------------------------------------------
# Binary + model discovery
# ---------------------------------------------------------------------------

def find_binary() -> str | None:
    """Find the crispasr binary."""
    candidates = [
        os.environ.get("CRISPASR_BIN", ""),
        "./build/bin/crispasr",
        "/tmp/build-issue89/bin/crispasr",
        "./build-ninja-compile/bin/crispasr",
    ]
    for c in candidates:
        if c and Path(c).is_file() and os.access(c, os.X_OK):
            return str(Path(c).resolve())
    return None


def find_model(backend: str) -> str | None:
    """Find a model GGUF for the given backend."""
    info = BACKEND_MODELS.get(backend)
    if not info:
        return None
    for g in info["globs"]:
        # Try absolute
        p = MODELS_DIR / g
        if p.exists():
            return str(p)
        # Try from repo root
        repo = Path(__file__).parent.parent
        p = repo / g
        if p.exists():
            return str(p)
    return None


def get_git_sha() -> str:
    """Get current git SHA."""
    try:
        r = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                           capture_output=True, text=True, cwd=Path(__file__).parent.parent)
        return r.stdout.strip() if r.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def get_audio_duration(path: str) -> float:
    """Get audio duration in seconds."""
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "csv=p=0", path],
            capture_output=True, text=True)
        return float(r.stdout.strip()) if r.returncode == 0 else 0.0
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# Run one benchmark
# ---------------------------------------------------------------------------

def run_one(
    binary: str,
    model: str,
    audio: str,
    settings: dict,
    language: str | None = None,
) -> dict:
    """Run crispasr on one (model, audio, settings) combo. Returns result dict."""
    audio_dur = get_audio_duration(audio)
    audio_name = Path(audio).name

    # -of sets the output path *prefix*; crispasr appends ".json" for -ojf.
    with tempfile.NamedTemporaryFile(suffix=".bench", delete=False) as tf:
        out_prefix = tf.name
    json_out = out_prefix + ".json"

    cmd = [binary, "-m", model, "-f", audio, "--no-prints",
           "-ojf", "-of", out_prefix]
    if language:
        cmd += ["-l", language]
    cmd += settings.get("args", [])

    timeout = max(int(audio_dur * 10), 300)  # generous for CPU

    t0 = time.monotonic()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        exit_code = proc.returncode
        error = proc.stderr.strip()[-200:] if proc.returncode != 0 else None
    except subprocess.TimeoutExpired:
        exit_code = -1
        error = f"timeout after {timeout}s"
    wall_s = round(time.monotonic() - t0, 1)

    # Parse metrics from JSON output
    metrics = CoverageMetrics()
    if Path(json_out).exists() and Path(json_out).stat().st_size > 0:
        metrics = metrics_from_json(json_out, audio_dur)

    # Cleanup
    for p in [json_out, out_prefix]:
        try:
            os.unlink(p)
        except OSError:
            pass

    return {
        "ts": datetime.now(timezone.utc).isoformat(),
        "git_sha": get_git_sha(),
        "backend": Path(model).stem,
        "model": Path(model).name,
        "audio": audio_name,
        "audio_duration_s": round(audio_dur, 1),
        "language": language or "auto",
        "settings": {"label": settings["label"]},
        "metrics": metrics.to_dict(),
        "perf": {
            "wall_s": wall_s,
            "rtf": round(audio_dur / wall_s, 2) if wall_s > 0 else 0,
        },
        "exit_code": exit_code,
        "error": error,
    }


# ---------------------------------------------------------------------------
# Result storage
# ---------------------------------------------------------------------------

def append_result(result: dict) -> None:
    """Append one result to the JSONL file."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(result, ensure_ascii=False) + "\n")


def load_results(sha_filter: str | None = None) -> list[dict]:
    """Load results, optionally filtering by git SHA."""
    if not RESULTS_FILE.exists():
        return []
    results = []
    with open(RESULTS_FILE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if sha_filter and r.get("git_sha") != sha_filter:
                continue
            results.append(r)
    return results


def result_exists(git_sha: str, backend: str, audio: str, label: str) -> bool:
    """Check if a result already exists (for --resume)."""
    for r in load_results(sha_filter=git_sha):
        if (r.get("backend") == backend and
            r.get("audio") == audio and
            r.get("settings", {}).get("label") == label):
            return True
    return False


# ---------------------------------------------------------------------------
# Table formatting
# ---------------------------------------------------------------------------

def format_table(results: list[dict]) -> str:
    """Format results as a readable table (reporter's style)."""
    if not results:
        return "No results."

    header = (
        f"{'backend':<28} {'audio':<20} {'settings':<12} "
        f"{'dur':>4} {'words':>6} {'chars':>6} {'first_ts':>9} {'last_ts':>9} "
        f"{'cov%':>6} {'gaps':>5} {'wall_s':>7} {'rtf':>5}"
    )
    sep = "-" * len(header)
    lines = [header, sep]

    for r in results:
        m = r.get("metrics", {})
        p = r.get("perf", {})
        s = r.get("settings", {})
        lines.append(
            f"{r.get('backend', '?'):<28} "
            f"{r.get('audio', '?'):<20} "
            f"{s.get('label', '?'):<12} "
            f"{r.get('audio_duration_s', 0):>4.0f} "
            f"{m.get('word_count', 0):>6} "
            f"{m.get('char_count', 0):>6} "
            f"{m.get('first_ts_s', 0):>9.2f} "
            f"{m.get('last_ts_s', 0):>9.2f} "
            f"{m.get('coverage_pct', 0):>6.1f} "
            f"{m.get('gap_count', 0):>5} "
            f"{p.get('wall_s', 0):>7.1f} "
            f"{p.get('rtf', 0):>5.1f}"
        )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="CrispASR multi-backend benchmark driver",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--audio", type=str, help="Single audio file to test")
    parser.add_argument("--corpus", type=str, help="Path to corpus.json for matrix run")
    parser.add_argument("--backend", type=str, help="Backend name (or comma-separated list)")
    parser.add_argument("--model", type=str, help="Explicit model path (overrides auto-detect)")
    parser.add_argument("--settings", type=str, default="auto",
                        help="Comma-separated settings labels (default: auto)")
    parser.add_argument("--language", "-l", type=str, help="Force language code")
    parser.add_argument("--resume", action="store_true", help="Skip already-completed combos")
    parser.add_argument("--format", choices=["table", "json", "jsonl"], default="table")
    parser.add_argument("--all-settings", action="store_true",
                        help="Run all settings in the matrix")
    args = parser.parse_args()

    # Find binary
    binary = find_binary()
    if not binary:
        print("ERROR: crispasr binary not found. Set CRISPASR_BIN or build first.", file=sys.stderr)
        sys.exit(1)
    print(f"Binary: {binary}", file=sys.stderr)

    git_sha = get_git_sha()
    print(f"Git SHA: {git_sha}", file=sys.stderr)

    # Resolve settings
    if args.all_settings:
        settings_list = SETTINGS_MATRIX
    else:
        labels = [s.strip() for s in args.settings.split(",")]
        settings_list = [s for s in SETTINGS_MATRIX if s["label"] in labels]
        if not settings_list:
            print(f"ERROR: no matching settings for '{args.settings}'", file=sys.stderr)
            sys.exit(1)

    # Resolve backends + models
    backends = []
    if args.model and args.backend:
        backends = [(args.backend, args.model)]
    elif args.backend:
        for b in args.backend.split(","):
            b = b.strip()
            model = find_model(b)
            if model:
                backends.append((b, model))
            else:
                print(f"SKIP: model not found for backend '{b}'", file=sys.stderr)
    else:
        # All available backends
        for b, info in BACKEND_MODELS.items():
            model = find_model(b)
            if model:
                backends.append((b, model))

    if not backends:
        print("ERROR: no backends available. Set --model or check /mnt/storage/", file=sys.stderr)
        sys.exit(1)
    print(f"Backends: {[b[0] for b in backends]}", file=sys.stderr)

    # Resolve audio files
    audio_files = []
    if args.audio:
        audio_files = [(args.audio, args.language)]
    elif args.corpus:
        with open(args.corpus, encoding="utf-8") as f:
            corpus = json.load(f)
        corpus_dir = Path(args.corpus).parent
        for entry in corpus:
            path = corpus_dir / entry["path"]
            if path.exists():
                audio_files.append((str(path), entry.get("language")))
            else:
                print(f"SKIP: {entry['path']} not found", file=sys.stderr)
    else:
        print("ERROR: provide --audio or --corpus", file=sys.stderr)
        sys.exit(1)

    # Run matrix
    all_results = []
    total = len(backends) * len(audio_files) * len(settings_list)
    done = 0

    for backend_name, model_path in backends:
        backend_langs = BACKEND_MODELS.get(backend_name, {}).get("langs", [])
        for audio_path, lang in audio_files:
            # Skip if backend doesn't support this language
            if lang and backend_langs and lang not in backend_langs:
                done += len(settings_list)
                continue
            for settings in settings_list:
                done += 1
                label = settings["label"]
                audio_name = Path(audio_path).name

                if args.resume and result_exists(git_sha, Path(model_path).stem, audio_name, label):
                    print(f"  [{done}/{total}] SKIP (exists): {backend_name} × {audio_name} × {label}",
                          file=sys.stderr)
                    continue

                print(f"  [{done}/{total}] {backend_name} × {audio_name} × {label} ...",
                      file=sys.stderr, end=" ", flush=True)

                result = run_one(binary, model_path, audio_path, settings, language=lang)
                all_results.append(result)
                append_result(result)

                m = result["metrics"]
                status = "OK" if result["exit_code"] == 0 else f"ERR({result['exit_code']})"
                print(f"{status} words={m['word_count']} cov={m['coverage_pct']}% "
                      f"wall={result['perf']['wall_s']}s", file=sys.stderr)

    # Output
    if args.format == "table":
        print()
        print(format_table(all_results))
    elif args.format == "json":
        print(json.dumps(all_results, indent=2, ensure_ascii=False))
    elif args.format == "jsonl":
        for r in all_results:
            print(json.dumps(r, ensure_ascii=False))

    # Summary
    print(f"\n{len(all_results)} runs completed. Results appended to {RESULTS_FILE}",
          file=sys.stderr)


if __name__ == "__main__":
    main()
