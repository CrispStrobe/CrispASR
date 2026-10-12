#!/usr/bin/env python3
"""Run Dia F16/Q8/Q4 long controls and the #530 short prompt sequentially."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(os.environ['HEAVY_OUT'])
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--matrix', default='4', help='CPU thread counts passed to the existing full-speech worker')
args = parser.parse_args()
OUT.mkdir(parents=True, exist_ok=True)
worker = ROOT / 'tools/ci-heavy/dia_full_speech.py'
receipt = {
    'source': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
    'worker_sha256': hashlib.sha256(worker.read_bytes()).hexdigest(),
    'model_revision': '3233fbcb32be47761d2e736857b6d1a075b9ba7e',
    'short_prompt': 'We are committed to making it secure.',
    'threads': args.matrix,
    'scope': 'Actual CPU CLI/C ABI complete speech, independent ASR and explicit limits. No GPU or original full-short-speech claim.',
    'runs': [],
}
for quant in ('f16', 'q8_0', 'q4_k'):
    target = OUT / quant
    target.mkdir(exist_ok=True)
    env = dict(os.environ, HEAVY_OUT=str(target))
    command = [sys.executable, str(worker), '--quant', quant, '--matrix', args.matrix,
               '--short-report', '--limits']
    # The shared build/HF scratch is reused; each model process exits before
    # the next starts. A failed quant remains a failure while others are tested.
    with (OUT / f'{quant}-worker.log').open('w') as log:
        proc = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    receipt['runs'].append({'quant': quant, 'returncode': proc.returncode,
                            'passed': proc.returncode == 0, 'output': quant})
    (OUT / 'matrix.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(quant, proc.returncode, (OUT / f'{quant}-worker.log').read_text()[-2500:], flush=True)
assert len(receipt['runs']) == 3 and all(r['passed'] for r in receipt['runs']), receipt
print('DIA_DELAY_FULL_SPEECH_MATRIX_PASS', flush=True)
