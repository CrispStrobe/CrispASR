#!/usr/bin/env python3
"""Exact accepted 9B F16, same runtime, pipeline scheduling versus graph reuse.

Run the canonical independent CUDA acceptance first, then isolated AB/BA arms.
No default flip is performed. Three warm calls follow an initial call per clip.
"""
import argparse
import json
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys
import time
import wave

SCRIPT_VERSION = '2026-10-02.1'
# Fill from the successful CI build and private transfer receipt before launch.
SOURCE_COMMIT = 'PENDING'
BUILD_COMMIT = '722f54ba4'  # replace with full immutable build SHA
BUILD_RUN = 37030248844
BUNDLE_REVISION = 'PENDING'
BUNDLE_SHA256 = 'PENDING'
SDK = Path('/kaggle/temp/index-echo-scheduler-sdk')
TEMP = Path('/kaggle/temp/index-echo-validation')
OUT = Path('/kaggle/working')
OUT.mkdir(parents=True, exist_ok=True)
p = argparse.ArgumentParser()
p.add_argument('--worker', choices=['control', 'candidate'])
p.add_argument('--order', choices=['AB', 'BA'])
a = p.parse_args()


def run(*command, **kwargs):
    subprocess.run(list(map(str, command)), check=True, **kwargs)


if not a.worker:
    if any(len(pin) != 40 for pin in (SOURCE_COMMIT, BUILD_COMMIT, BUNDLE_REVISION)) or len(BUNDLE_SHA256) != 64:
        raise RuntimeError('Immutable CI bundle and SDK pins required before launch')
    hardware = subprocess.check_output(['nvidia-smi', '--query-gpu=name,compute_cap,memory.total',
                                        '--format=csv,noheader'], text=True).strip()
    rows = [line.split(',') for line in hardware.splitlines()]
    if len(rows) < 2 or any(row[1].strip() != '7.5' for row in rows):
        (OUT / 'inconclusive.json').write_text(json.dumps(dict(hardware=hardware, conclusive=False,
            reason='Experiment needs two actual SM75 GPUs; no weights downloaded')))
        raise SystemExit(0)
    run('git', 'init', SDK)
    run('git', '-C', SDK, 'remote', 'add', 'origin', 'https://github.com/CrispStrobe/CrispASR.git')
    run('git', '-C', SDK, 'fetch', '--depth=1', 'origin', SOURCE_COMMIT)
    run('git', '-C', SDK, 'checkout', 'FETCH_HEAD')
    sys.path.insert(0, str(SDK / 'tools/kaggle'))
    import kaggle_harness as kh
    kh.init_progress()
    kh.provenance(SCRIPT_VERSION, SDK)
    config = dict(source_commit=SOURCE_COMMIT, build_commit=BUILD_COMMIT, build_run=BUILD_RUN,
                  bundle_revision=BUNDLE_REVISION, bundle_sha256=BUNDLE_SHA256, keep_models=True)
    config_path = OUT / 'runtime-config.json'
    config_path.write_text(json.dumps(config, indent=2) + '\n')
    env = dict(os.environ, INDEX_ECHO_VALIDATION_CONFIG=str(config_path),
               CRISPASR_LLAMA_PIPELINE_DISABLE='1', INDEX_ECHO_BENCH='1')
    validation = SDK / 'tools/kaggle/index-echo-9b-validation/index_echo_9b_validation.py'
    with (OUT / 'candidate-acceptance.log').open('w') as log, kh.build_heartbeat('candidate.acceptance', interval_s=30):
        run(sys.executable, validation, stdout=log, stderr=subprocess.STDOUT, env=env)
    accepted = json.loads((OUT / 'cuda-validation.json').read_text())
    assert accepted['validated'] is True and accepted['full_pipeline_checked'] is True
    results = {}
    for order, arms in [('AB', ['control', 'candidate']), ('BA', ['candidate', 'control'])]:
        results[order] = {}
        for arm in arms:
            env = dict(os.environ, CRISPASR_LLAMA_PIPELINE_DISABLE='1' if arm == 'candidate' else '0',
                       INDEX_ECHO_BENCH='1', OMP_NUM_THREADS='4')
            with (OUT / f'{order}-{arm}.log').open('w') as log, kh.build_heartbeat(order+'.'+arm, interval_s=30):
                run(sys.executable, __file__, '--worker', arm, '--order', order,
                    stdout=log, stderr=subprocess.STDOUT, env=env)
            log = (OUT / f'{order}-{arm}.log').read_text()
            reused = [int(n) for n in re.findall(r'graphs reused\s*=\s*(\d+)', log)]
            assert reused, 'Missing actual decoder graph-reuse counters'
            if arm == 'candidate':
                assert 'pipeline parallelism disabled by CRISPASR_LLAMA_PIPELINE_DISABLE' in log
                assert max(reused) > 0, 'Candidate never reached graph reuse'
            else:
                assert 'pipeline parallelism enabled' in log
                assert max(reused) == 0, 'Control unexpectedly reused graphs'
            results[order][arm] = json.loads((OUT / f'{order}-{arm}.json').read_text())
            results[order][arm]['graphs_reused'] = reused
    ratios = {order: {clip: results[order]['control']['clips'][clip]['warm_median_seconds'] /
                           results[order]['candidate']['clips'][clip]['warm_median_seconds']
                      for clip in results[order]['control']['clips']} for order in results}
    receipt = dict(script_version=SCRIPT_VERSION, hardware=hardware, config=config, orders=results,
                   control_over_candidate=ratios, output_passed=True, acceptance=accepted,
                   faster_in_every_order_and_clip=all(ratio > 1 for order in ratios.values() for ratio in order.values()),
                   default_changed=False)
    (OUT / 'scheduler-ab.json').write_text(json.dumps(receipt, indent=2, ensure_ascii=False) + '\n')
    print('Scheduler A/B complete; default unchanged', ratios, flush=True)
    raise SystemExit(0)

import numpy as np
from gguf import GGUFReader
sys.path.insert(0, str(SDK / 'python'))
from crispasr import Session
sys.path.insert(0, str(SDK / 'tools'))
from index_echo_acceptance import compare_case
bundle = TEMP / 'bundle'
os.environ['LD_LIBRARY_PATH'] = str(bundle) + ':' + os.environ.get('LD_LIBRARY_PATH', '')
# ctypes loads dependencies after the dynamic linker starts; RPATH=$ORIGIN in
# the hosted bundle is the authoritative lookup, independently of this env var.
library = next(bundle.glob('libcrispasr.so*'))
model = TEMP / 'models/index-echo-9b-f16.gguf'
result = dict(arm=a.worker, order=a.order, clips={})
with Session(str(model), lib_path=str(library), n_threads=4) as session:
    for clip, audio in [('jfk', SDK / 'samples/jfk.wav'), ('zh', SDK / 'samples/paraformer_zh.wav'),
                        ('jfk-tail', TEMP / 'references/jfk-tail.wav')]:
        reader = GGUFReader(str(TEMP / 'references' / (clip + '-ref.gguf')))
        text = reader.fields['crispasr.ref.generated_text'].contents()
        del reader
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        assert len(lines) % 3 == 0
        golden = []
        for i in range(0, len(lines), 3):
            m = re.fullmatch(r'\[(\d+):(\d+(?:\.\d+)?)-(\d+):(\d+(?:\.\d+)?)\]', lines[i])
            assert m
            golden.append(dict(start=60*int(m[1])+float(m[2]), end=60*int(m[3])+float(m[4]),
                               text=lines[i+1]+'\n'+lines[i+2]))
        with wave.open(str(audio)) as wav:
            assert wav.getframerate() == 16000 and wav.getnchannels() == 1
            pcm = np.frombuffer(wav.readframes(wav.getnframes()), dtype='<i2').astype(np.float32) / 32768
        iterations = []
        for i in range(4):
            started = time.perf_counter()
            actual = [dict(start=s.start, end=s.end, text=s.text) for s in session.transcribe(pcm)]
            seconds = time.perf_counter() - started
            compare_case(actual, {'independent-source': golden}, 'f16')
            iterations.append(dict(iteration=i, seconds=seconds, segments=actual))
            print(a.worker, clip, i, seconds, flush=True)
        result['clips'][clip] = dict(iterations=iterations, first_seconds=iterations[0]['seconds'],
            warm_median_seconds=statistics.median(row['seconds'] for row in iterations[1:]))
(OUT / f'{a.order}-{a.worker}.json').write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
