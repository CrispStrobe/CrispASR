#!/usr/bin/env python3
"""Immutable full-source CUDA validator bootstrap; no private key or token."""
import hashlib
import os
from pathlib import Path
import runpy
import urllib.request
SOURCE = '91e1e97b399d105924915c0e8d5ea0176359b147'
SHA256 = 'b08f87cdd81e6a1467d8343072970b9d314382d0091f232d31aca5e96d3f11f8'
RECIPIENT = 'af2f46629cfa40da8c9fffdb9923877d1c4eb312d4ce3b4e36a42aa673f8ba47'
print('OpenVoice full-source bootstrap', 'source='+SOURCE, 'worker_sha256='+SHA256,
      'bootstrap_sha256='+hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), flush=True)
path=Path('/kaggle/temp/openvoice-fullsource-worker.py')
path.parent.mkdir(parents=True,exist_ok=True)
with urllib.request.urlopen('https://raw.githubusercontent.com/CrispStrobe/CrispASR/'+SOURCE+
        '/tools/kaggle/openvoice-fullsource/run.py',timeout=60) as response:
    code=response.read()
if hashlib.sha256(code).hexdigest()!=SHA256:raise RuntimeError('Pinned worker checksum mismatch')
path.write_bytes(code)
os.environ.update(CRISPASR_VALIDATION_SOURCE=SOURCE,CRISPASR_AUDIO_RECIPIENT=RECIPIENT)
runpy.run_path(str(path),run_name='__main__')
