#!/usr/bin/env python3
"""Render the pushed immutable source into the script, then use standard kpush."""
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[3]
FOLDER = Path(__file__).resolve().parent

def main():
    source = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    assert not subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT), 'Commit and push clean source first'
    branch = subprocess.check_output(['git', 'symbolic-ref', '--quiet', '--short', 'HEAD'], cwd=ROOT, text=True).strip()
    remote = subprocess.check_output(['git', 'ls-remote', 'origin', 'refs/heads/' + branch], cwd=ROOT, text=True)
    assert remote.split()[0] == source, 'Remote source does not match kernel source'
    with tempfile.TemporaryDirectory(prefix='dia-cuda-kpush-', dir=os.environ['TMPDIR']) as temporary:
        staging = Path(temporary) / 'kernel'
        staging.mkdir()
        shutil.copyfile(FOLDER / 'kernel-metadata.json', staging / 'kernel-metadata.json')
        code = (FOLDER / 'dia_cuda_cache_control.py').read_text()
        assert code.count('PENDING_SOURCE_PIN') == 1
        (staging / 'dia_cuda_cache_control.py').write_text(code.replace('PENDING_SOURCE_PIN', source))
        print('KERNEL_SOURCE', source, flush=True)
        subprocess.run([sys.executable, ROOT / 'tools/kaggle/kpush.py', staging], check=True)

if __name__ == '__main__':
    main()
