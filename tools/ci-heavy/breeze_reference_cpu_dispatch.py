#!/usr/bin/env python3
"""Compare original Breeze default/AVX2 CPU dispatch on one GH host.

Both variants keep the immutable original byte gate. Any failed variant makes
the run fail; this investigation never executes or accepts the native port.
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(os.environ['HEAVY_OUT'])
SCRATCH = Path(os.environ['HEAVY_SCRATCH']) / 'breeze-reference-cpu-dispatch'
ISA_ENV = dict(ATEN_CPU_CAPABILITY='avx2', ONEDNN_MAX_CPU_ISA='AVX2',
               MKL_ENABLE_INSTRUCTIONS='AVX2')


def main():
    import numpy as np
    OUT.mkdir(parents=True, exist_ok=True)
    receipt = dict(source=subprocess.check_output(['git', 'rev-parse', 'HEAD'],
        cwd=ROOT, text=True).strip(), diagnostic_complete=False,
        original_byte_gate_waived=False, native_executed=False,
        initialization_seed=42, variants={}, arrays={})
    try:
        for label in ('default', 'avx2'):
            destination = OUT / label
            destination.mkdir()
            environment = dict(os.environ)
            for name in ISA_ENV:
                environment.pop(name, None)
            if label == 'avx2':
                environment.update(ISA_ENV)
            environment.update(HEAVY_OUT=str(destination),
                HEAVY_SCRATCH=str(SCRATCH / label),
                BREEZE_BACKBONE_BOUNDARIES='0', BREEZE_PRECISION_DECISIONS='0')
            with (destination / 'reference.log').open('w') as log:
                # Set identical initialization RNGs before the unchanged loader.
                # ISA environment is set before the first Torch import.
                bootstrap = ('import random, numpy as np, torch, runpy, sys; '
                    'random.seed(42); np.random.seed(42); torch.manual_seed(42); '
                    'sys.argv=sys.argv[1:]; '
                    'sys.path.insert(0, __import__("pathlib").Path(sys.argv[0]).resolve().parent.as_posix()); '
                    'runpy.run_path(sys.argv[0], run_name="__main__")')
                result = subprocess.run([sys.executable, '-u', '-c', bootstrap,
                    str(Path(__file__).with_name('breeze_depth_probe.py')),
                    '--phase', 'reference'], cwd=ROOT, env=environment,
                    stdout=log, stderr=subprocess.STDOUT, timeout=7200)
            row = dict(exit_status=result.returncode,
                dispatch_env={name: environment.get(name) for name in ISA_ENV})
            for key, name in [('original_control', 'original-equivalence.json'),
                              ('runtime', 'runtime-provenance.json')]:
                path = destination / 'reference' / name
                if path.is_file():
                    row[key] = json.loads(path.read_text())
            receipt['variants'][label] = row
        a, b = (receipt['variants'][name] for name in ('default', 'avx2'))
        if 'runtime' in a and 'runtime' in b:
            receipt['checkpoint_hashes_identical'] = a['runtime']['checkpoint_sha256'] == b['runtime']['checkpoint_sha256']
            receipt['package_versions_identical'] = a['runtime']['packages'] == b['runtime']['packages']
            receipt['torch_capabilities'] = [row['runtime']['torch_cpu_capability'] for row in (a, b)]
        receipt['diagnostic_complete'] = True
        receipt['all_controls_pass'] = all(row['exit_status'] == 0
            and row.get('original_control', {}).get('prompt_byte_exact')
            and row.get('original_control', {}).get('logits_byte_exact')
            and row.get('original_control', {}).get('finite') for row in (a, b))
        receipt['all_controls_pass'] = bool(receipt['all_controls_pass']
            and receipt.get('checkpoint_hashes_identical')
            and receipt.get('package_versions_identical'))
    finally:
        for path in sorted(OUT.rglob('*.npy')):
            value = np.load(path, allow_pickle=False)
            receipt['arrays'][str(path.relative_to(OUT))] = dict(
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                shape=list(value.shape), dtype=str(value.dtype), finite=bool(np.isfinite(value).all()))
        (OUT / 'validation.json').write_text(json.dumps(receipt, indent=2)+'\n')
    assert receipt['all_controls_pass'], 'Original byte-control failure retained; see both dispatch variants'


if __name__ == '__main__':
    main()
