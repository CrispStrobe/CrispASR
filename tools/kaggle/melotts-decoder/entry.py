#!/usr/bin/env python3
"""Immutable diagnostic bootstrap; original application acceptance stays false."""

import hashlib
import os
from pathlib import Path
import runpy
import urllib.request

SOURCE = "48e25bddf2319f142fca13bea5fdc07664100868"
SHA256 = "7e37616493e0ebf992edc7c3013584ba5ae3c55681467a91f2098ac4285965b6"
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
