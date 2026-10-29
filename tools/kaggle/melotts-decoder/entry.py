#!/usr/bin/env python3
"""Immutable diagnostic bootstrap; original application acceptance stays false."""

import hashlib
import os
from pathlib import Path
import runpy
import urllib.request

SOURCE = "59751b4e2388bede93705428b81928d7e778efbc"
SHA256 = "2f21de41c9c5232095ab225483c58c2c35066c0d228b27a1e10983b189557c46"
print(
    "Melo decoder diagnostic source=" + SOURCE,
    "worker_sha256=" + SHA256,
    "bootstrap_sha256=" + hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    flush=True,
)
path = Path("/kaggle/temp/melotts-decoder-worker.py")
path.parent.mkdir(parents=True, exist_ok=True)
with urllib.request.urlopen(
    "https://raw.githubusercontent.com/CrispStrobe/CrispASR/"
    + SOURCE
    + "/tools/kaggle/melotts-decoder/run.py",
    timeout=60,
) as response:
    code = response.read()
if hashlib.sha256(code).hexdigest() != SHA256:
    raise RuntimeError("Pinned diagnostic checksum mismatch")
path.write_bytes(code)
os.environ["CRISPASR_VALIDATION_SOURCE"] = SOURCE
os.environ["CRISPASR_CHECKPOINT_URL"] = "${CRISPASR_CHECKPOINT_URL}"
runpy.run_path(str(path), run_name="__main__")
