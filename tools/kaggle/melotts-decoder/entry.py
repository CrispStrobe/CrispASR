#!/usr/bin/env python3
"""Immutable diagnostic bootstrap; original application acceptance stays false."""

import hashlib
import os
from pathlib import Path
import runpy
import urllib.request

SOURCE = "837e02957585e2b29f29ddfbcbab9a6f54a15f36"
SHA256 = "fb48ce6d1fc371c29d8be62e542fa92f93d2667411d991d12b52e07452d4596d"
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
