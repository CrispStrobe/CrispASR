#!/usr/bin/env python3
"""Pinned current-SDK validation with real Python Session callers."""
import hashlib
import os
from pathlib import Path
import runpy
import urllib.request

SOURCE = '264ff48a5a4887509888ed0078ddb8012d9d6131'
SHA256 = 'c55c0f5c0c555d0798107f6f3b6b83ea747bd827276aeefdec408e3870305fea'
print('OpenVoice Session bootstrap', 'worker_source=' + SOURCE, 'worker_sha256=' + SHA256,
      'entry_sha256=' + hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), flush=True)
path = Path('/kaggle/temp/openvoice-session-worker.py')
path.parent.mkdir(parents=True, exist_ok=True)
with urllib.request.urlopen('https://raw.githubusercontent.com/CrispStrobe/CrispASR/' + SOURCE +
        '/tools/kaggle/openvoice-current-replay/replay.py', timeout=60) as response:
    code = response.read()
if hashlib.sha256(code).hexdigest() != SHA256:
    raise RuntimeError('Pinned Session worker checksum mismatch')
path.write_bytes(code)
os.environ['CRISPASR_OV2_SESSION'] = '1'
runpy.run_path(str(path), run_name='__main__')
