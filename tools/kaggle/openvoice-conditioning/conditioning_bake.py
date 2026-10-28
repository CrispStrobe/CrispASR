#!/usr/bin/env python3
"""Immutable shared worker bootstrap. Only this script is uploaded by Kaggle."""
import hashlib
import os
from pathlib import Path
import runpy
import urllib.request

WORKER_SOURCE = "2a9ac6c9f7c058be487710b6891e0f2271013490"
WORKER_SHA256 = "bce35215db0dd9f96a9c9d5b5a5951be9f92b04085adf872ab543dd3240bdb32"
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
