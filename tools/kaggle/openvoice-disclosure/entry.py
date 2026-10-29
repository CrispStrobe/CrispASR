#!/usr/bin/env python3
"""Immutable diagnostic bootstrap; original application acceptance stays false."""
import hashlib
import os
from pathlib import Path
import runpy
import urllib.request
SOURCE = 'f2729e4295844fdc121756de2214b864dee596f9'
SHA256 = '67a7564ee9e71f4ab0c9bda8907f92a7ad133d142b5efd5ad8af277653d5c74e'
print('Disclosure diagnostic source='+SOURCE, 'worker_sha256='+SHA256,
      'bootstrap_sha256='+hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), flush=True)
path=Path('/kaggle/temp/openvoice-disclosure-worker.py')
path.parent.mkdir(parents=True,exist_ok=True)
with urllib.request.urlopen('https://raw.githubusercontent.com/CrispStrobe/CrispASR/'+SOURCE+
        '/tools/kaggle/openvoice-disclosure/run.py',timeout=60) as response:
    code=response.read()
if hashlib.sha256(code).hexdigest()!=SHA256:raise RuntimeError('Pinned diagnostic checksum mismatch')
path.write_bytes(code)
os.environ['CRISPASR_CHECKPOINT_URL'] = '${CRISPASR_CHECKPOINT_URL}'
runpy.run_path(str(path),run_name='__main__')
