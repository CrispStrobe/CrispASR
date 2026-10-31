#!/usr/bin/env python3
"""Hermetic paired A/B benchmark for CTC forced alignment.

Benchmarks Arm A (serial baseline) vs Arm B (multithreaded chunk-parallel log-softmax + Viterbi bounded DP).
Discards cold warmup and executes 10 warm runs for each arm.
Verifies 100% bit-identical timestamps and reports exact runtime metrics.
"""

import os
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tests'))

from ctc_align_harness import Probe

def run_bench():
    print("=" * 70)
    print("CrispASR CTC Forced Alignment: Paired A/B Benchmark (Serial vs Accelerated)")
    print("=" * 70)

    # Standard ASR configuration (30s audio, 16k vocabulary, realistic sentence)
    T = 1500
    V = 16384
    vocab = [f'tok_{i}' for i in range(V)]
    vocab[0] = '<blank>'
    vocab[1] = '|'
    for i in range(26):
        vocab[i + 2] = chr(ord('a') + i)

    words = [
        'and', 'so', 'my', 'fellow', 'americans', 'ask', 'not', 'what', 'your',
        'country', 'can', 'do', 'for', 'you', 'ask', 'what', 'you', 'can', 'do',
        'for', 'your', 'country'
    ]

    rng = np.random.default_rng(42)
    logits = rng.normal(0, 4, size=(T, V)).astype(np.float32)

    N_RUNS = 10

    # 1. Benchmark Arm A: Serial Baseline (CRISPASR_CTC_ALIGN_SERIAL=1 in subprocess env)
    print(f"\nBenchmarking Arm A (Serial Baseline, {N_RUNS} warm runs)...")
    old_env = os.environ.get('CRISPASR_CTC_ALIGN_SERIAL')
    os.environ['CRISPASR_CTC_ALIGN_SERIAL'] = '1'
    probe_a = Probe('CRISPASR_CTC_ALIGN_OPT', source_root=ROOT)
    # Warmup
    probe_a.run(logits, words, vocab, gate=1)

    times_serial = []
    res_serial = None
    for i in range(N_RUNS):
        res_serial = probe_a.run(logits, words, vocab, gate=1)
        times_serial.append(res_serial['kernel_ns'] / 1e6)
    probe_a.close()

    # 2. Benchmark Arm B: Accelerated (Multithreaded chunk-parallel)
    print(f"Benchmarking Arm B (Accelerated Parallel, {N_RUNS} warm runs)...")
    if 'CRISPASR_CTC_ALIGN_SERIAL' in os.environ:
        del os.environ['CRISPASR_CTC_ALIGN_SERIAL']
    probe_b = Probe('CRISPASR_CTC_ALIGN_OPT', source_root=ROOT)
    # Warmup
    probe_b.run(logits, words, vocab, gate=1)

    times_parallel = []
    res_parallel = None
    for i in range(N_RUNS):
        res_parallel = probe_b.run(logits, words, vocab, gate=1)
        times_parallel.append(res_parallel['kernel_ns'] / 1e6)
    probe_b.close()

    if old_env is not None:
        os.environ['CRISPASR_CTC_ALIGN_SERIAL'] = old_env

    # 3. Parity Verification
    bytes_match = (res_serial['bytes'] == res_parallel['bytes'])
    stamps_match = (res_serial['stamps'] == res_parallel['stamps'])
    allocs_match = (res_serial['allocations'] == res_parallel['allocations'])
    peak_match = (res_serial['peak_heap_bytes'] == res_parallel['peak_heap_bytes'])

    print("\n" + "=" * 70)
    print("PARITY / QUALITY VERIFICATION")
    print("=" * 70)
    print(f"  Exact Wire Byte Stream Parity: {'PASSED (100% bit-for-bit match)' if bytes_match else 'FAILED'}")
    print(f"  Exact Timestamps Parity:       {'PASSED (All words/chars match)' if stamps_match else 'FAILED'}")
    print(f"  Allocations Parity:            {'PASSED (' + str(res_serial['allocations']) + ' allocs)' if allocs_match else 'DIFF'}")
    print(f"  Peak Heap Usage:               {'PASSED (' + str(res_serial['peak_heap_bytes']) + ' bytes)' if peak_match else 'DIFF'}")

    assert bytes_match, "FATAL: Output bytes diverged between serial and parallel arms!"

    # 4. Speedup Statistics
    med_serial = float(np.median(times_serial))
    mean_serial = float(np.mean(times_serial))
    std_serial = float(np.std(times_serial))

    med_parallel = float(np.median(times_parallel))
    mean_parallel = float(np.mean(times_parallel))
    std_parallel = float(np.std(times_parallel))

    speedup = med_serial / med_parallel

    print("\n" + "=" * 70)
    print("EXECUTION TIME & SPEEDUP (T=1500 frames, V=16384 vocabulary)")
    print("=" * 70)
    print(f"  Arm A (Serial Baseline):     {med_serial:.2f} ms median ({mean_serial:.2f} ± {std_serial:.2f} ms)")
    print(f"  Arm B (Accelerated Parallel): {med_parallel:.2f} ms median ({mean_parallel:.2f} ± {std_parallel:.2f} ms)")
    print(f"  SPEEDUP FACTOR:              {speedup:.2f}x faster ({med_serial - med_parallel:.2f} ms savings per call)")
    print("=" * 70)

if __name__ == '__main__':
    run_bench()
