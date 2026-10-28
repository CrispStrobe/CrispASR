#!/usr/bin/env python3
"""Immutable shared worker bootstrap. Only this script is uploaded by Kaggle."""
import hashlib
import os
from pathlib import Path
import runpy
import urllib.request

WORKER_SOURCE = "d84007612167105cf24c7af80014d1f0ea1bd8a3"
WORKER_SHA256 = "b1fed1e4518ceeb517c749be67ce3eeee9d3243153b5f50043f6edf9c4584f20"
print('OpenVoice fixed replay bootstrap v2',
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
