#!/usr/bin/env python3
"""Stage real SCNet weights and a released CPU runtime; never upload or submit."""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import urllib.request
import zipfile

RELEASE_URL = "https://github.com/CrispStrobe/CrispASR/releases/download/v0.8.53/crispasr-linux-x86_64.tar.gz"
RELEASE_SHA = "3396d49473642d1097b667270b54125279bdca740b735e15c3c96c1b32f787c0"
MODEL_SHA = "8441b23c2b6ba47e8357fd863d717af88a5568f1937a3ff328c921c7143bd059"


def stage(output, account, release=None, models=None):
    output.mkdir(parents=True, exist_ok=True)
    runtime = release.read_bytes() if release else urllib.request.urlopen(RELEASE_URL).read()
    assert hashlib.sha256(runtime).hexdigest() == RELEASE_SHA, "release transfer mismatch"
    archive = models.read_bytes() if models else subprocess.check_output([
        "gh", "api", "repos/CrispStrobe/CrispASR/actions/artifacts/13574722861/zip"])
    with zipfile.ZipFile(io.BytesIO(archive)) as bundle:
        weights = bundle.read("scnet-f16.gguf")
        assert hashlib.sha256(weights).hexdigest() == MODEL_SHA, "model transfer mismatch"
        (output / "scnet-f16.gguf").write_bytes(weights)
        (output / "SCNet-LICENSE").write_bytes(bundle.read("LICENSE"))
    (output / "crispasr-linux-x86_64.tar.gz").write_bytes(runtime)
    (output / "dataset-metadata.json").write_text(json.dumps({
        "title": "CrispASR SCNet Cache Model",
        "id": account + "/crispasr-scnet-cache-model",
        "licenses": [{"name": "other"}],
        "description": "Author-MIT SCNet F16 weights and CrispASR v0.8.53 CPU runtime for an offline 60-second graph-cache measurement. See SCNet-LICENSE and runtime licenses."
    }, indent=2) + "\n")
    print("Prepared real-model input dataset; no upload or kernel submission performed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--release", type=Path, help="reuse a downloaded release archive")
    parser.add_argument("--models", type=Path, help="reuse the downloaded qualified-model zip")
    args = parser.parse_args()
    stage(args.output, os.environ["KAGGLE_ACCOUNT"], args.release, args.models)
