#!/usr/bin/env python3
"""Run prebuilt Granite5 acceptance on Kaggle after explicit CPU-input pinning."""
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tarfile

VERSION = 'granite5-cuda-acceptance-v4'
SOURCE = '42abcbd62346635b53bc28578493ae1bdb7434de'
CPU_SOURCE = '1bad3b2dbe8b5ce55cc5dfafcec25610d442ee95'
CPU_RUN = 39984477044
CPU_REVISION = 'b99c15b3719518846a90402447cee58331470ed7'
CPU_SHA256 = 'bb8e17e1e4fc81fdb85bbdcab84ae91f01833d30d4da9135f2ea2af4874bfdea'
RUNTIME_SOURCE = '2f73af3d4537ee69417d845d3318186134d0c629'
RUNTIME_REVISION = 'b73f461c80abc95bae34d7cb45a2bed18b433f1d'
RUNTIME_SHA256 = '141ae7661999aaed10c06590b8e6e4e648fd76d6e34473f37cbfc463622420c6'
STAGING = 'cstr/crispasr-granite5-validation-public'
RUNTIME_PATH = f'runtime/{RUNTIME_SOURCE}/granite5-cuda-validation.tar.gz'
# Deliberately unlaunchable until reviewed immutable CPU/runtime pins exist.
for value in (SOURCE, CPU_SOURCE, RUNTIME_SOURCE, CPU_REVISION, RUNTIME_REVISION):
    if not re.fullmatch('[0-9a-f]{40}', value):
        raise RuntimeError('Reviewed immutable CPU and runtime revisions are required before pushing this kernel')
if not all(re.fullmatch('[0-9a-f]{64}', value) for value in (CPU_SHA256, RUNTIME_SHA256)):
    raise RuntimeError('Reviewed CPU receipt hash is required before pushing this kernel')
OUT = Path('/kaggle/working')
SCRATCH = Path('/kaggle/temp/granite5-acceptance')
SCRATCH.mkdir(parents=True, exist_ok=True)
os.environ.update(TMPDIR=str(SCRATCH), HF_HOME=str(SCRATCH / 'hf'), HF_XET_CACHE=str(SCRATCH / 'xet'),
                  OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='4', CUDA_VISIBLE_DEVICES='0')
hardware = subprocess.check_output(['nvidia-smi', '--id=0', '--query-gpu=name,compute_cap,driver_version',
                                    '--format=csv,noheader'], text=True).strip()
if not hardware:
    raise RuntimeError('Actual GPU0 is required')
print(VERSION, SOURCE, hardware, flush=True)
repo = SCRATCH / 'CrispASR'
for command in (['git', 'init', repo], ['git', '-C', repo, 'remote', 'add', 'origin',
                                       'https://github.com/CrispStrobe/CrispASR.git'],
                ['git', '-C', repo, 'fetch', '--depth=1', 'origin', SOURCE],
                ['git', '-C', repo, 'checkout', 'FETCH_HEAD']):
    subprocess.run(list(map(str, command)), check=True)
if subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip() != SOURCE:
    raise RuntimeError('Worker source pin mismatch')
sys.path.insert(0, str(repo / 'tools/kaggle'))
import kaggle_harness as kh
kh.init_progress()
kh.provenance(VERSION, repo)
os.environ['HF_TOKEN'] = kh.resolve_hf_token(require=True)
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'numpy<3', 'gguf==0.17.1',
                'huggingface_hub>=0.36,<2'], check=True)
from huggingface_hub import hf_hub_download


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


cpu_path = f'profiles/{CPU_SOURCE}/cpu-evidence/receipt.json'
cpu = Path(hf_hub_download(STAGING, cpu_path, revision=CPU_REVISION, local_dir=SCRATCH / 'inputs'))
runtime = Path(hf_hub_download(STAGING, RUNTIME_PATH, revision=RUNTIME_REVISION, local_dir=SCRATCH / 'inputs'))
if sha(cpu) != CPU_SHA256 or sha(runtime) != RUNTIME_SHA256:
    raise RuntimeError('Pinned CPU or runtime content mismatch')
evidence = json.loads(cpu.read_text())
if not (evidence['source'] == CPU_SOURCE and evidence['cpu_acceptance'] and evidence['complete']):
    raise RuntimeError('CPU acceptance has not been established')
with tarfile.open(runtime) as archive:
    archive.extractall(SCRATCH, filter='data')
bundle = SCRATCH / 'bundle'
if json.loads((bundle / 'provenance.json').read_text())['source'] != RUNTIME_SOURCE:
    raise RuntimeError('Runtime build source pin mismatch')
(OUT / 'entry-provenance.json').write_text(json.dumps(dict(version=VERSION, source=SOURCE,
    cpu_run=CPU_RUN, cpu_source=CPU_SOURCE, cpu_revision=CPU_REVISION, cpu_sha256=CPU_SHA256,
    runtime_source=RUNTIME_SOURCE, runtime_revision=RUNTIME_REVISION, runtime_sha256=RUNTIME_SHA256,
    entry_sha256=sha(__file__), hardware=hardware), indent=2) + '\n')
command = [sys.executable, repo / 'tools/ci-heavy/granite5_cuda_acceptance.py', '--bundle', bundle,
           '--cpu-receipt', cpu, '--scratch', SCRATCH / 'models', '--out', OUT / 'cuda']
with kh.build_heartbeat('granite5.cuda.acceptance'):
    result = subprocess.run(list(map(str, command)))
summary = json.loads((OUT / 'cuda/receipt.json').read_text())
kh.step('granite5.cuda.result', returncode=result.returncode, complete=summary['complete'],
        cuda_acceptance=summary['cuda_acceptance'])
if not (result.returncode == 0 and summary['complete'] and summary['cuda_acceptance']):
    raise RuntimeError('Actual CUDA acceptance failed; preserve terminal logs before any rerun')
print('GRANITE5_CUDA_ACCEPTED', VERSION, flush=True)
