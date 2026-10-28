#!/usr/bin/env python3
"""Immutable shared worker bootstrap. Only this script is uploaded by Kaggle."""
import hashlib
import os
from pathlib import Path
import runpy
import urllib.request

WORKER_SOURCE = "18cd593de480d1e9ef268570b1eb556edfd926da"
WORKER_SHA256 = "3162084ed25d9788f6ca28e418eb2b4d8aa1d50eea0f55a0fa3a36feab7cbe21"
print('OpenVoice fixed replay bootstrap v1',
      'bootstrap_sha256=' + hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
      'worker_source=' + WORKER_SOURCE, 'worker_sha256=' + WORKER_SHA256, flush=True)
path = Path('/kaggle/temp/openvoice-fixed-replay-worker.py')
path.parent.mkdir(parents=True, exist_ok=True)
url = ('https://raw.githubusercontent.com/CrispStrobe/CrispASR/' + WORKER_SOURCE +
       '/tools/kaggle/openvoice-bake/openvoice_bake.py')
with urllib.request.urlopen(url, timeout=60) as response:
    code = response.read()
if hashlib.sha256(code).hexdigest() != WORKER_SHA256:
    raise RuntimeError('Pinned shared worker checksum mismatch')
path.write_bytes(code)
os.environ['CRISPASR_OV2_REPLAY'] = '1'
runpy.run_path(str(path), run_name='__main__')
