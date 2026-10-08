#!/usr/bin/env python3
"""Identical-input CUDA attention probe. Diagnostic only; no ASR acceptance.

Q/K/V come from the pinned official encoder on native conv2 input. Timings
exclude allocation/transfers and use one backend, resident graphs, alternating
order, two warmups and six measured repetitions. Sampled VRAM is a lower bound
on process peak, not a model/session memory benchmark.
"""
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

import numpy as np
from pr492_acceptance import ROOT, digest, metrics, run
from pr492_tokenizer_cuda_precision import FLAG


def main():
    out = Path(os.environ['HEAVY_OUT'])
    capture = out / 'attention'
    capture.mkdir(parents=True, exist_ok=True)
    os.environ.update({FLAG: '1', 'NV_TF32_OVERRIDE': '0', 'MIMO_DIAG_QKV_OUT': str(capture)})
    run([sys.executable, ROOT / 'tools/ci-heavy/pr492_tokenizer_diagnose.py'], out / 'diagnose.log')
    study = json.loads((out / 'tokenizer-diagnosis.json').read_text())
    receipt = dict(scope=__doc__, source=subprocess.check_output(['git', 'rev-parse', 'HEAD'],
                   cwd=ROOT, text=True).strip(), layers={}, full_pr_acceptance=False,
                   tokenizer_exact_codes=study['attention_ab']['q4']['tok_codes'],
                   upstream=study['upstream'], upstream_files=study['upstream_files'])
    def save():
        (out / 'attention-diagnosis.json').write_text(json.dumps(receipt, indent=2) + '\n')
    save()
    for layer in sorted(capture.glob('layer-*')):
        shape = json.loads((layer / 'shape.json').read_text())['shape']
        with (layer / 'native.log').open('w') as log:
            proc = subprocess.Popen([os.environ['MIMO_DIAG_ATTN_PROBE'], str(layer), *map(str, shape)],
                                    stdout=log, stderr=subprocess.STDOUT)
            samples = []
            done = threading.Event()
            def monitor():
                while not done.is_set():
                    try:
                        rows = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,used_memory',
                            '--format=csv,noheader,nounits'], text=True, timeout=10)
                        for row in rows.splitlines():
                            pid, used = row.split(',')
                            if int(pid) == proc.pid:
                                samples.append(dict(seconds=time.monotonic(), mib=int(used)))
                    except (ValueError, subprocess.SubprocessError):
                        pass
                    done.wait(.1)
            thread = threading.Thread(target=monitor, daemon=True)
            thread.start()
            try:
                code = proc.wait(timeout=600)
            finally:
                done.set(); thread.join(timeout=15)
            assert code == 0, (layer, code)
        native = json.loads((layer / 'native.json').read_text())
        assert 'MIMO_DIAG_TILE_F32' in (layer / 'native.log').read_text(), 'No actual TILE dispatch'
        assert 'CUDA' in native['backend']
        arrays = {name: np.fromfile(layer / (name + '.bin'), dtype=np.float32).reshape(shape)
                  for name in ['python', 'python-half', 'python-half-kv', 'flash', 'flash-prec', 'eager', 'half-eager', 'tile-f32']}
        assert all(np.isfinite(x).all() for x in arrays.values())
        comparisons = {name + '_vs_python': metrics(arrays[name], arrays['python'])
                       for name in ['flash', 'flash-prec', 'eager', 'half-eager', 'tile-f32']}
        comparisons.update(tile_vs_half_kv_python=metrics(arrays['tile-f32'], arrays['python-half-kv']),
            kv_half_rounding=metrics(arrays['python-half-kv'], arrays['python']),
            flash_vs_half_python=metrics(arrays['flash'], arrays['python-half']),
            half_eager_vs_half_python=metrics(arrays['half-eager'], arrays['python-half']),
            flash_vs_half_eager=metrics(arrays['flash'], arrays['half-eager']),
            half_rounding=metrics(arrays['python-half'], arrays['python']),
            scale_negative_control=metrics(arrays['python'] * 2, arrays['python']))
        for arm in native['arms'].values():
            assert len(arm['seconds']) == 6 and arm['verified_repetitions'] == 8
            arm['median_seconds'] = float(np.median(arm['seconds']))
        receipt['layers'][layer.name] = dict(shape=shape, native=native, comparisons=comparisons,
            hint_byte_identical=np.array_equal(arrays['flash'], arrays['flash-prec']),
            pid=proc.pid, sampled_vram=samples,
            sampled_peak_mib=max((s['mib'] for s in samples), default=None),
            hashes={p.name: digest(p) for p in layer.glob('*.bin')})
        save()
        # F32 eager is the independent control. Do not gate flash with a tolerance
        # chosen from its observed drift, or turn diagnostic success into acceptance.
        m = comparisons['eager_vs_python']
        assert m['cosine'] >= .9999 and m['relative_l2'] <= .005, layer
        m = comparisons['half_eager_vs_half_python']
        assert m['cosine'] >= .9999 and m['relative_l2'] <= .005, layer
    assert len(receipt['layers']) == 3
    receipt['diagnostic_controls_passed'] = True
    save()
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
