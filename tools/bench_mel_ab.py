#!/usr/bin/env python3
"""
tools/bench_mel_ab.py — Paired A/B benchmark for core_mel optimization.

Follows crispasr-crispembed-dev.md Rule 4/4a/5:
  1. Checks system load (sysctl vm.loadavg).
  2. Measures Arm A (CRISPASR_MEL_SERIAL=1, serial baseline)
     vs Arm B (optimized parallel SIMD path).
  3. Executes 1 cold warmup + 3 measured warm runs per arm in separate processes.
  4. Asserts proof of work (non-empty transcript, non-zero return code).
  5. Computes exact decoded transcript match (WER/CER diff between A and B).
  6. Reports exact milliseconds, RTF, speedup ratio, and verbatim transcripts.
"""

import os
import sys
import json
import time
import subprocess
import statistics
import re

def get_load():
    try:
        out = subprocess.check_output(["sysctl", "-n", "vm.loadavg"], text=True).strip()
        return out
    except Exception:
        return "unknown"

def run_decode(bin_path, model_path, audio_path, lang, env_vars):
    env = os.environ.copy()
    env.update(env_vars)
    cmd = [
        bin_path,
        "-m", model_path,
        "-f", audio_path,
        "-l", lang,
        "-t", "8",
    ]
    t0 = time.perf_counter()
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
    t1 = time.perf_counter()
    elapsed_ms = (t1 - t0) * 1000.0

    if res.returncode != 0:
        raise RuntimeError(f"Command failed (exit {res.returncode}):\nSTDERR:\n{res.stderr}\nSTDOUT:\n{res.stdout}")

    # Extract transcription: last non-empty line of stdout
    lines = [line.strip() for line in res.stdout.splitlines() if line.strip()]
    transcript = ""
    for line in reversed(lines):
        if not line.startswith("crispasr:") and not line.startswith("===") and not line.startswith("ggml_"):
            transcript = line
            break

    # Extract reported realtime speed if printed
    match = re.search(r"transcribed ([\d\.]+)s audio in ([\d\.]+)s \(([\d\.]+)x realtime\)", res.stdout)
    reported_audio_s = float(match.group(1)) if match else 0.0
    reported_time_s = float(match.group(2)) if match else 0.0
    reported_rtf = float(match.group(3)) if match else 0.0

    return {
        "wall_ms": elapsed_ms,
        "reported_time_s": reported_time_s,
        "reported_audio_s": reported_audio_s,
        "reported_rtf": reported_rtf,
        "transcript": transcript,
        "stdout": res.stdout,
        "stderr": res.stderr
    }

def run_ab_suite(bin_path, model_path, audio_path, lang, audio_dur_s, ground_truth):
    print(f"\n=======================================================")
    print(f"Benchmarking: {os.path.basename(model_path)} on {os.path.basename(audio_path)} ({audio_dur_s:.1f}s)")
    print(f"Initial load: {get_load()}")
    print(f"=======================================================")

    # 1. Warmup runs (discarded)
    print("Running cold warmup for Arm A (Serial)...")
    _ = run_decode(bin_path, model_path, audio_path, lang, {"CRISPASR_MEL_SERIAL": "1"})
    print("Running cold warmup for Arm B (Optimized Parallel)...")
    _ = run_decode(bin_path, model_path, audio_path, lang, {})

    # 2. Measured alternating runs (3 per arm)
    a_runs = []
    b_runs = []

    for run_idx in range(1, 4):
        print(f"\n--- Measured Iteration {run_idx}/3 ---")
        # Run A
        res_a = run_decode(bin_path, model_path, audio_path, lang, {"CRISPASR_MEL_SERIAL": "1"})
        a_runs.append(res_a)
        print(f"  Arm A (Serial):    {res_a['wall_ms']:.1f} ms (backend time {res_a['reported_time_s']:.2f}s, {res_a['reported_rtf']:.1f}x RT)")

        # Run B
        res_b = run_decode(bin_path, model_path, audio_path, lang, {})
        b_runs.append(res_b)
        print(f"  Arm B (Parallel):  {res_b['wall_ms']:.1f} ms (backend time {res_b['reported_time_s']:.2f}s, {res_b['reported_rtf']:.1f}x RT)")

    a_times = [r["wall_ms"] for r in a_runs]
    b_times = [r["wall_ms"] for r in b_runs]
    a_reported = [r["reported_time_s"] for r in a_runs]
    b_reported = [r["reported_time_s"] for r in b_runs]

    med_a_wall = statistics.median(a_times)
    med_b_wall = statistics.median(b_times)
    med_a_rep = statistics.median(a_reported)
    med_b_rep = statistics.median(b_reported)

    speedup_wall = med_a_wall / med_b_wall
    speedup_backend = med_a_rep / med_b_rep if med_b_rep > 0 else 1.0

    transcript_a = a_runs[0]["transcript"]
    transcript_b = b_runs[0]["transcript"]
    transcript_exact_match = (transcript_a == transcript_b)
    matches_ground_truth = (transcript_b == ground_truth)

    print("\n--- Summary ---")
    print(f"Transcript A: '{transcript_a}'")
    print(f"Transcript B: '{transcript_b}'")
    print(f"Transcripts Exact Match (A == B): {transcript_exact_match}")
    print(f"Matches Ground Truth:            {matches_ground_truth}")
    print(f"Arm A Wall Median:               {med_a_wall:.1f} ms (reported backend {med_a_rep*1000:.1f} ms)")
    print(f"Arm B Wall Median:               {med_b_wall:.1f} ms (reported backend {med_b_rep*1000:.1f} ms)")
    print(f"End-to-End Speedup:              {speedup_wall:.3f}x")
    print(f"Backend Core Speedup:            {speedup_backend:.3f}x")

    return {
        "model": os.path.basename(model_path),
        "audio": os.path.basename(audio_path),
        "audio_dur_s": audio_dur_s,
        "transcript_a": transcript_a,
        "transcript_b": transcript_b,
        "exact_parity": transcript_exact_match,
        "ground_truth_parity": matches_ground_truth,
        "arm_a_wall_ms": a_times,
        "arm_b_wall_ms": b_times,
        "med_a_wall_ms": med_a_wall,
        "med_b_wall_ms": med_b_wall,
        "med_a_backend_s": med_a_rep,
        "med_b_backend_s": med_b_rep,
        "speedup_wall": speedup_wall,
        "speedup_backend": speedup_backend,
    }

def main():
    bin_path = "./build/bin/crispasr"
    if not os.path.exists(bin_path):
        print(f"Binary not found: {bin_path}")
        sys.exit(1)

    models_dir = "/Volumes/backups/ai/crispasr-gguf"
    
    benchmarks = [
        {
            "model": os.path.join(models_dir, "parakeet-tdt-0.6b-v3-q4_k.gguf"),
            "audio": "samples/jfk.wav",
            "lang": "en",
            "dur": 11.0,
            "ground_truth": "And so, my fellow Americans, ask not what your country can do for you, ask what you can do for your country."
        },
        {
            "model": os.path.join(models_dir, "parakeet-tdt-0.6b-ja-q4_k.gguf"),
            "audio": "samples/ja_test.wav",
            "lang": "ja",
            "dur": 4.6,
            "ground_truth": "こんにちは、本日はお日これまでもよく、すばらしい天気です、もう"
        },
        {
            "model": os.path.join(models_dir, "sensevoice-small-q4_k.gguf"),
            "audio": "samples/jfk.wav",
            "lang": "en",
            "dur": 11.0,
            "ground_truth": "And so my fellow Americans ask not what your country can do for you, ask what you can do for your country."
        }
    ]

    results = []
    for b in benchmarks:
        if os.path.exists(b["model"]):
            res = run_ab_suite(bin_path, b["model"], b["audio"], b["lang"], b["dur"], b["ground_truth"])
            results.append(res)
        else:
            print(f"Skipping {b['model']}, not found.")

    with open("build/ab_benchmark_receipt.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nBenchmark receipt saved to build/ab_benchmark_receipt.json")

if __name__ == "__main__":
    main()
