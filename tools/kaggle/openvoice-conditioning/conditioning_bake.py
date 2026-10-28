#!/usr/bin/env python3
"""Immutable shared worker bootstrap. Only this script is uploaded by Kaggle."""
import hashlib
import os
from pathlib import Path
import runpy
import urllib.request

WORKER_SOURCE = "aa0ae40669456a2ce748348a8f258cb14870ac42"
WORKER_SHA256 = "3cb365e8cab7061ddad03918b56c2b6d92cc6b522dc034dbf27333433a977fa6"
print('OpenVoice conditioning bootstrap v2',
      'bootstrap_sha256=' + hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
      'worker_source=' + WORKER_SOURCE, 'worker_sha256=' + WORKER_SHA256, flush=True)
path = Path('/kaggle/temp/openvoice-conditioning-worker.py')
path.parent.mkdir(parents=True, exist_ok=True)
url = ('https://raw.githubusercontent.com/CrispStrobe/CrispASR/' + WORKER_SOURCE +
       '/tools/kaggle/openvoice-bake/openvoice_bake.py')
with urllib.request.urlopen(url, timeout=60) as response:
    code = response.read()
if hashlib.sha256(code).hexdigest() != WORKER_SHA256:
    raise RuntimeError('Pinned shared worker checksum mismatch')
path.write_bytes(code)
os.environ['CRISPASR_OV2_CONDITIONING'] = '1'
runpy.run_path(str(path), run_name='__main__')
