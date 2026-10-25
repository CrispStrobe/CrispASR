#!/usr/bin/env python3
"""Run a source-pinned, prebuilt CSM correction on actual Kaggle GPU0."""
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tarfile
import zipfile

PINS = {'source': '2c118384014e2d11c371533bdc6dc3447cb1ba3f',
 'revision': '5ccc9800a530a64bb24da25e1e752b08b6598da3',
 'files': {'runtime/csm-cuda-validation.tar.gz': {'path': 'runs/2c118384014e2d11c371533bdc6dc3447cb1ba3f/runtime/csm-cuda-validation.tar.gz',
                                                  'sha256': '71ab4f8ef5b9ddd029cd3a461c9c6518812ed4735d013efb98509643759fa630',
                                                  'bytes': 230799840},
           'cpu-receipt.json': {'path': 'runs/2c118384014e2d11c371533bdc6dc3447cb1ba3f/cpu-receipt.json',
                                'sha256': '23fe54fe323dfa3c1dc79417eebf99c4a9c5a341787eacaaf866e3c4b81b1061',
                                'bytes': 60012},
           'port-receipt.json': {'path': 'runs/2c118384014e2d11c371533bdc6dc3447cb1ba3f/port-receipt.json',
                                 'sha256': 'a1cb7f0f2841e753d9c2c0fe3a0d926117aaeb5297f4236765e9a1d970f74f3f',
                                 'bytes': 48663},
           'original-rope-reference.zip': {'path': 'runs/2c118384014e2d11c371533bdc6dc3447cb1ba3f/original-rope-reference.zip',
                                           'sha256': '507fbf46343810b1016445cc7058477aac99df6af2607892c3c27e8b82c4e2bf',
                                           'bytes': 796462}}}
SOURCE = PINS["source"]
if not all(re.fullmatch("[0-9a-f]{40}", value) for value in (SOURCE, PINS["revision"])):
    raise RuntimeError("Reviewed immutable worker and input revisions are required")
if not all(re.fullmatch("[0-9a-f]{64}", record["sha256"]) for record in PINS["files"].values()):
    raise RuntimeError("Reviewed immutable artifact hashes are required")
OUT = Path("/kaggle/working")
SCRATCH = Path("/kaggle/temp/csm-original-rope")
SCRATCH.mkdir(parents=True, exist_ok=True)
os.environ.update(TMPDIR=str(SCRATCH), HF_HOME=str(SCRATCH / "hf"),
                  HF_XET_CACHE=str(SCRATCH / "xet"), CUDA_VISIBLE_DEVICES="0")
hardware = subprocess.check_output(["nvidia-smi", "--id=0", "--query-gpu=name,compute_cap,driver_version",
                                    "--format=csv,noheader"], text=True).strip()
if not hardware:
    raise RuntimeError("Actual GPU0 is required")
repo = SCRATCH / "CrispASR"
for argv in (["git", "init", repo], ["git", "-C", repo, "remote", "add", "origin",
                                    "https://github.com/CrispStrobe/CrispASR.git"],
             ["git", "-C", repo, "fetch", "--depth=1", "origin", SOURCE],
             ["git", "-C", repo, "checkout", "FETCH_HEAD"]):
    subprocess.run(list(map(str, argv)), check=True)
if subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip() != SOURCE:
    raise RuntimeError("Worker source differs from reviewed pin")
sys.path.insert(0, str(repo / "tools/kaggle"))
import kaggle_harness as kh
kh.init_progress()
kh.provenance("csm-original-rope-v1", repo)
os.environ["HF_TOKEN"] = kh.kaggle_token_from_dataset("hf_token.txt") or kh.resolve_hf_token(require=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "numpy<3", "huggingface_hub>=0.36,<2"], check=True)
from huggingface_hub import HfApi, hf_hub_download
# The existing private credential is verified before launch. Resolve its owner
# rather than committing private account details in the kernel.
owner = HfApi(token=os.environ["HF_TOKEN"]).whoami()["name"]
staging = owner + "/crispasr-csm-validation"


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


inputs = {}
for name, record in PINS["files"].items():
    path = Path(hf_hub_download(staging, record["path"], revision=PINS["revision"],
                               local_dir=SCRATCH / "inputs"))
    if path.stat().st_size != record["bytes"] or sha(path) != record["sha256"]:
        raise RuntimeError("Pinned input content differs: " + name)
    inputs[name] = path
    kh.step("input.verified", input=name)
reference = SCRATCH / "original-rope-reference"
reference.mkdir()
with zipfile.ZipFile(inputs["original-rope-reference.zip"]) as zipped:
    members = zipped.infolist()
    if (len(members) > 512 or sum(member.file_size for member in members) > 16 * 1024**2 or
            len({member.filename for member in members}) != len(members)):
        raise RuntimeError("Original reference expansion exceeds bounded size or duplicates members")
    for member in members:
        target = reference / member.filename
        if (not target.resolve().is_relative_to(reference.resolve()) or
                (member.external_attr >> 16 & 0o170000) == 0o120000):
            raise RuntimeError("Unsafe original reference archive member")
        if member.is_dir():
            target.mkdir(parents=True, exist_ok=True)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        copied = 0
        with zipped.open(member) as source, target.open("wb") as output:
            while chunk := source.read(1024**2):
                copied += len(chunk)
                if copied > member.file_size or copied > 16 * 1024**2:
                    raise RuntimeError("Original reference member exceeds declared size")
                output.write(chunk)
        if copied != member.file_size:
            raise RuntimeError("Truncated original reference member")
with tarfile.open(inputs["runtime/csm-cuda-validation.tar.gz"]) as archive:
    members = archive.getmembers()
    if len(members) > 8192 or sum(member.size for member in members) > 4 * 1024**3:
        raise RuntimeError("Runtime expansion exceeds bounded size")
    archive.extractall(SCRATCH, filter="data")
bundle = SCRATCH / "bundle"
provenance = json.loads((bundle / "provenance.json").read_text())
if provenance["source"] != SOURCE or not provenance["build_complete"]:
    raise RuntimeError("Runtime compiled source differs from CPU/worker qualification")
(OUT / "entry-provenance.json").write_text(json.dumps(dict(
    pins=PINS, entry_sha256=sha(__file__), hardware=hardware,
    runtime_provenance_sha256=sha(bundle / "provenance.json")), indent=2) + "\n")
command = [sys.executable, repo / "tools/ci-heavy/csm_cuda_acceptance.py", "--bundle", bundle,
           "--cpu-receipt", inputs["cpu-receipt.json"], "--port-receipt", inputs["port-receipt.json"],
           "--reference", reference, "--scratch", SCRATCH / "models", "--out", OUT / "cuda"]
with kh.build_heartbeat("csm.actual.cuda.acceptance"):
    subprocess.run(list(map(str, command)), check=True)
kh.step("csm.actual.cuda.acceptance.complete")
