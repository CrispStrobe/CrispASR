#!/usr/bin/env python3
"""Immutable diagnostic bootstrap; original application acceptance stays false."""
import hashlib
from pathlib import Path
import runpy
import urllib.request
SOURCE = '0c56d6090badbb4f8a9199e03c9708bf093eb13a'
SHA256 = '9f4e722efcd44e22e25a8c9aaa7cc7c7d6d587409572497d5fd45b4042241de3'
print('Disclosure diagnostic source='+SOURCE, 'worker_sha256='+SHA256,
      'bootstrap_sha256='+hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), flush=True)
path=Path('/kaggle/temp/openvoice-disclosure-worker.py')
path.parent.mkdir(parents=True,exist_ok=True)
with urllib.request.urlopen('https://raw.githubusercontent.com/CrispStrobe/CrispASR/'+SOURCE+
        '/tools/kaggle/openvoice-disclosure/run.py',timeout=60) as response:
    code=response.read()
if hashlib.sha256(code).hexdigest()!=SHA256:raise RuntimeError('Pinned diagnostic checksum mismatch')
path.write_bytes(code)
runpy.run_path(str(path),run_name='__main__')
