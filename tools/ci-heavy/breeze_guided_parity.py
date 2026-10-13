#!/usr/bin/env python3
"""Run original reference then native parity in separate GH CPU processes.

Sequential processes release all original-model RAM before loading native F16.
Both complete receipts, full arrays and logs survive a failure in either half.
"""
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(os.environ['HEAVY_OUT'])
reference = OUT / 'reference'
native = OUT / 'native'
for script, directory, args in (
    ('breeze_guided_reference.py', reference, []),
    ('breeze_guided_native.py', native, ['--reference-dir', str(reference)]),
):
    directory.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env['HEAVY_OUT'] = str(directory)
    with (OUT / (script + '.log')).open('w') as log:
        r = subprocess.run([sys.executable, '-u', str(ROOT / 'tools/ci-heavy' / script), *args],
                           cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    print(script, 'rc=', r.returncode, (OUT / (script + '.log')).read_text()[-4000:], flush=True)
    if r.returncode:
        (OUT / 'summary.md').write_text(f'Breeze guided parity failed in {script}; full evidence retained.\n')
        raise SystemExit(r.returncode)
(OUT / 'summary.md').write_text('Original and native guided parity/context controls PASS. '
                              'Full-speech ASR remains separate.\n')
