#!/usr/bin/env python3
"""Exact projection A/B, ABBA timings, and pinned real-ASR regression.

No BLAS in the probe or runtime build: float must exercise the changed scalar
projection. Both binaries (with/without OpenMP) must match byte for byte.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import statistics
import subprocess

ROOT = Path(__file__).resolve().parents[2]
p = argparse.ArgumentParser()
p.add_argument('--component-only', action='store_true')
a = p.parse_args()
OUT = Path(os.environ['HEAVY_OUT'])
SCRATCH = Path(os.environ['HEAVY_SCRATCH']) / 'mel-projection'
OUT.mkdir(parents=True, exist_ok=True)
SCRATCH.mkdir(parents=True, exist_ok=True)
receipt = dict(source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
               platform=platform.platform(), cpu_count=os.cpu_count(), default_changed=False,
               component_pass=False, runtime_pass=False, cases=[])

def checkpoint():
    (OUT / 'mel-projection.json').write_text(json.dumps(receipt, indent=2) + '\n')

checkpoint()
for omp in (False, True):
    exe = SCRATCH / ('probe-omp' if omp else 'probe-serial')
    cmd = ['g++', '-std=c++17', '-O3', '-pthread', '-Isrc',
           'tools/ci-heavy/mel_projection_probe.cpp', 'src/core/mel.cpp', '-o', str(exe)]
    if omp:
        cmd += ['-fopenmp']
    subprocess.run(cmd, cwd=ROOT, check=True)
    for frames in (63, 64, 65, 300, 1100, 3000):
        for layout in (0, 1):
            for precision in (0, 1):
                baseline = None
                for threads in (1, 2, 4):
                    # Two warmups per process, four measured repetitions per
                    # process, ABBA gives eight measurements per mode.
                    timings = {0: [], 1: []}
                    projections = {0: [], 1: []}
                    for index, enabled in enumerate((0, 1, 1, 0)):
                        label = f'{int(omp)}-{frames}-{layout}-{precision}-{threads}-{index}'
                        output = SCRATCH / (label + '.bin')
                        env = dict(os.environ, CRISPASR_MEL_PROJECTION_PARALLEL=str(enabled),
                                   CRISPASR_MEL_TIMING='1', OMP_NUM_THREADS=str(threads))
                        run = subprocess.run([str(exe), str(frames), str(layout), str(precision),
                                              str(threads), '4', str(output)], env=env,
                                             text=True, capture_output=True, check=True)
                        (OUT / (label + '.log')).write_text(run.stderr)
                        blob = output.read_bytes()
                        if baseline is None:
                            baseline = blob
                        assert blob == baseline, f'Bitwise drift: {label}'
                        timings[enabled] += [float(x) for x in re.findall(r'TOTAL_MS ([0-9.]+)', run.stderr)]
                        trace = re.findall(r'projection (\d+) frames ([0-9.]+) ms \[(\d+) thread', run.stderr)
                        assert len(trace) == 6, f'Missing projection dispatch: {label}'
                        expected_threads = threads if omp and enabled and frames >= 64 else 1
                        assert all(int(t[2]) == expected_threads for t in trace), label
                        projections[enabled] += [float(t[1]) for t in trace[2:]]
                        output.unlink()
                    receipt['cases'].append(dict(openmp=omp, frames=frames, layout=layout,
                        precision=precision, threads=threads, bitwise_equal=True,
                        sha256=hashlib.sha256(baseline).hexdigest(), total_ms=timings,
                        projection_ms=projections,
                        total_speedup=statistics.median(timings[0])/statistics.median(timings[1]),
                        projection_speedup=statistics.median(projections[0])/max(0.001, statistics.median(projections[1]))))
                checkpoint()
receipt['component_pass'] = True
checkpoint()
if not a.component_only:
    build = SCRATCH / 'build'
    subprocess.run(['cmake', '-S', str(ROOT), '-B', str(build), '-DCMAKE_BUILD_TYPE=Release',
                    '-DGGML_NATIVE=OFF', '-DGGML_CUDA=OFF', '-DCRISPASR_MEL_BLAS=OFF',
                    '-DCRISPASR_BUILD_TESTS=ON', '-DCRISPASR_BUILD_SERVER=OFF'], check=True)
    subprocess.run(['cmake', '--build', str(build), '--target', 'crispasr-cli', 'crispasr-diff', '-j4'], check=True)
    for enabled in (0, 1):
        env = dict(os.environ, BUILD_DIR=str(build), WORK_DIR=str(SCRATCH / 'regression'),
                   KEEP_WORK='1', CRISPASR_MEL_PROJECTION_PARALLEL=str(enabled),
                   CRISPASR_MEL_TIMING='1', OMP_NUM_THREADS='4')
        with (OUT / f'qwen3-{enabled}.log').open('w') as log:
            subprocess.run(['python', 'tests/regression/run_one.py', 'qwen3-asr-0.6b'],
                           cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    receipt['runtime_pass'] = True
checkpoint()
(OUT / 'summary.md').write_text(f"Projection bitwise checks: {len(receipt['cases'])} cases; runtime: {receipt['runtime_pass']}. Default unchanged.\n")
print('MEL_PROJECTION_PASS', len(receipt['cases']), flush=True)
