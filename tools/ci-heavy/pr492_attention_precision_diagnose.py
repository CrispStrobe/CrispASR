#!/usr/bin/env python3
"""Compare original and F32 TILE attention with unchanged tokenizer gates.

Same binary, device, original Q4 file and explicit F32 cuBLAS control.
Diagnostics remain separate from full ASR/output or production acceptance.
"""
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
from pr492_acceptance import ROOT, TOK_STAGES, metrics, run
from pr492_attention_precision import FLAG, MARKER


def main():
    out = Path(os.environ['HEAVY_OUT'])
    os.environ.pop(FLAG, None)
    # Baseline tokenizer + identical-input resident graphs including TILE arm.
    run([sys.executable, ROOT / 'tools/ci-heavy/pr492_attention_diagnose.py'], out / 'attention-study.log')
    os.environ.pop('MIMO_DIAG_QKV_OUT', None)
    os.environ[FLAG] = '1'
    # The child attention study sets its own F32 cuBLAS env; set it here too.
    from pr492_tokenizer_cuda_precision import FLAG as CUBLAS_FLAG
    os.environ[CUBLAS_FLAG] = '1'
    tiled = out / 'tile-tokenizer'; tiled.mkdir()
    env = dict(os.environ, HEAVY_OUT=str(tiled))
    run([sys.executable, ROOT / 'tools/ci-heavy/pr492_tokenizer_diagnose.py'],
        out / 'tile-tokenizer.log', env=env)
    study = json.loads((tiled / 'tokenizer-diagnosis.json').read_text())
    receipt = dict(scope=__doc__, source=study['source'], tile_study=study,
                   actual_dispatch={}, baseline_replay={}, full_pr_acceptance=False)
    for arm in ['q4-flash', 'promoted-flash', 'q4-eager', 'promoted-eager']:
        log = (tiled / (arm + '.log')).read_text()
        traces = [line for line in log.splitlines() if MARKER in line]
        assert bool(traces) == arm.endswith('-flash'), arm
        receipt['actual_dispatch'][arm] = traces
        receipt['baseline_replay'][arm] = {}
        for stage in TOK_STAGES:
            a, b = [np.load(base / arm / (stage + '.npy')) for base in [tiled, out]]
            m = (dict(exact=np.array_equal(a, b), mismatches=int(np.count_nonzero(a != b)))
                 if stage == 'tok_codes' else metrics(a, b))
            receipt['baseline_replay'][arm][stage] = m
            if arm.endswith('-eager'):
                assert m['exact'], (arm, stage, 'eager control changed')
    receipt['continuous_passed'] = bool(study['passed'])
    receipt['exact_codes_passed'] = bool(study['attention_ab']['q4']['tok_codes']['exact'])
    receipt['tokenizer_acceptance_passed'] = receipt['continuous_passed'] and receipt['exact_codes_passed']
    (out / 'attention-precision.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps({k: receipt[k] for k in ['continuous_passed', 'exact_codes_passed',
                                            'tokenizer_acceptance_passed', 'full_pr_acceptance']}), flush=True)
    assert receipt['tokenizer_acceptance_passed'], 'Unchanged tokenizer gates failed; inspect attention-precision.json'


if __name__ == '__main__':
    main()
