#!/usr/bin/env python3
"""Immutable shared worker bootstrap. Only this script is uploaded by Kaggle."""
import hashlib
import os
from pathlib import Path
import runpy
import urllib.request

WORKER_SOURCE = "c5287611ef8416704a216d1a91c1a0fea5127a66"
WORKER_SHA256 = "61435c8198ae6de1443c73c36905e4e8b9293a1f54bb17761ead4472011c9179"
print('OpenVoice conditioning bootstrap v4',
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
