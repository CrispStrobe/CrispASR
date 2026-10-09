#!/usr/bin/env python3
"""Collect fresh decoder statistics on a pinned disjoint EN/ZH corpus.

Actual GPU inference only; no builds, quantization or model publication.
"""
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

SCRIPT_VERSION = '2026-10-09.1-corpus-v1'
SOURCE_COMMIT = '327be3fda80c44f280bfecab988458f9c3712b08'
BUILD_COMMIT = 'a03820a4fcf32bae0b658b35cf6663e85abcc854'
BUILD_RUN = 37901223622
BUNDLE_REVISION = 'b284cbc8727e1f91904a4c2928e6991bd5bea938'
BUNDLE_SHA256 = '894bf650161f4c672a2992da394cdb8d65cca69af2f17c5b960c52bd4b6826f9'
ROOT = Path('/kaggle/temp/echo-calibration-sdk')
TEMP = Path('/kaggle/temp/echo-calibration-corpus')
OUT = Path('/kaggle/working')
TEMP.mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)
os.environ.update(TMPDIR=str(TEMP), HF_HOME=str(TEMP / 'hf'),
                  HF_XET_CACHE=str(TEMP / 'xet'), OMP_NUM_THREADS='4')
hardware = subprocess.check_output(['nvidia-smi', '--query-gpu=name,compute_cap,memory.total',
                                    '--format=csv,noheader'], text=True).strip()
rows = [line.split(',') for line in hardware.splitlines()]
if (len(rows) != 2 or any(row[1].strip() != '7.5' for row in rows)
        or sum(int(row[-1].strip().split()[0]) for row in rows) < 24 * 1024):
    (OUT / 'inconclusive.json').write_text(json.dumps(dict(hardware=hardware,
        conclusive=False, reason='Pinned SM75 F16 runtime requires two actual T4s; no weights pulled')))
    raise SystemExit(0)
for command in (['git', 'init', ROOT],
                ['git', '-C', ROOT, 'remote', 'add', 'origin', 'https://github.com/CrispStrobe/CrispASR.git'],
                ['git', '-C', ROOT, 'fetch', '--depth=1', 'origin', SOURCE_COMMIT],
                ['git', '-C', ROOT, 'checkout', 'FETCH_HEAD']):
    subprocess.run(list(map(str, command)), check=True)
sys.path.insert(0, str(ROOT / 'tools/kaggle'))
import kaggle_harness as kh
kh.init_progress()
kh.provenance(SCRIPT_VERSION, ROOT)
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'huggingface_hub', 'gguf', 'numpy'], check=True)
os.environ['HF_TOKEN'] = kh.resolve_hf_token(require=True)
from gguf import GGUFReader
import numpy as np

import tarfile
import zipfile
from huggingface_hub import hf_hub_download, snapshot_download

CORPUS_REV = '06258ecef2dfed2be52ced573ec0dffc2443fabc'
CORPUS_SHA = '44680f20e390cc0c6bf5bfcf3e284a1a3feabf118bd1a96e0ab510c682bf2188'
MODEL_REV = 'dffbadf0f173446fee0364a0807803d2b2fb6f49'
receipt = dict(script_version=SCRIPT_VERSION, entry_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    source_commit=SOURCE_COMMIT, build_commit=BUILD_COMMIT, build_run=BUILD_RUN,
    bundle_revision=BUNDLE_REVISION, bundle_sha256=BUNDLE_SHA256, hardware=hardware,
    corpus_revision=CORPUS_REV, corpus_sha256=CORPUS_SHA, model_revision=MODEL_REV,
    calibration_complete=False, q4_accepted=False, defaults_changed=False)


def save():
    (OUT / 'calibration-corpus.json').write_text(json.dumps(receipt, indent=2, ensure_ascii=False) + '\n')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b''):
            h.update(chunk)
    return h.hexdigest()


save()
corpus_zip = hf_hub_download('cstr/crispasr-imatrix-calib',
    'index-echo-en-zh-20261009/index-echo-en-zh-calibration.zip', repo_type='dataset', revision=CORPUS_REV)
assert digest(corpus_zip) == CORPUS_SHA
corpus = TEMP / 'corpus'
with zipfile.ZipFile(corpus_zip) as archive:
    archive.extractall(corpus)
manifest = json.loads((corpus / 'manifest.json').read_text())
assert len(manifest['clips']) == 48 and manifest['heldout_pcm_disjoint']
assert len({clip['pcm_sha256'] for clip in manifest['clips']}) == 48
excluded = set(manifest['heldout_pcm_sha256'].values())
assert not excluded.intersection(clip['pcm_sha256'] for clip in manifest['clips'])
(OUT / 'corpus-manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + '\n')
archive_path = hf_hub_download('cstr/crispasr-index-echo-cuda-validation',
    'index-echo-cuda-validation.tar.gz', repo_type='dataset', revision=BUNDLE_REVISION)
assert digest(archive_path) == BUNDLE_SHA256
with tarfile.open(archive_path) as archive:
    archive.extractall(TEMP, filter='data')
bundle = TEMP / 'bundle'
provenance = json.loads((bundle / 'provenance.json').read_text())
assert provenance['sha'] == BUILD_COMMIT and provenance['architectures'] == [75]
models = Path(snapshot_download('cstr/index-echo-9b-GGUF', revision=MODEL_REV,
    local_dir=TEMP / 'models', allow_patterns=['index-echo-9b-f16.gguf', 'index-echo-9b-decoder-f16.gguf']))
receipt['model_sha256'] = {p.name: digest(p) for p in models.glob('*.gguf')}
save()
library = next(bundle.glob('libcrispasr.so*'))
stats_path = OUT / 'disjoint-en-zh-imatrix.gguf'
assert not stats_path.exists(), 'Fresh corpus must never merge held-out smoke statistics'
# One child keeps the model pair loaded for all 48 clips. Its normal process
# exit flushes the collector before the parent audits the file.
child = TEMP / 'collect.py'
child.write_text(r'''
import hashlib,json,os,subprocess,time,wave
from pathlib import Path
import numpy as np
from crispasr import Session
corpus=Path(os.environ['ECHO_CORPUS'])
manifest=json.loads((corpus/'manifest.json').read_text())
results=[]
with Session(os.environ['ECHO_MODEL'],lib_path=os.environ['ECHO_LIBRARY'],n_threads=4) as session:
 assert session.backend=='index-echo'
 print(subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.used,utilization.gpu','--format=csv,noheader'],text=True),flush=True)
 for clip in manifest['clips']:
  path=corpus/clip['path']
  assert hashlib.sha256(path.read_bytes()).hexdigest()==clip['wav_sha256']
  with wave.open(str(path),'rb') as wav:
   assert wav.getframerate()==16000 and wav.getnchannels()==1 and wav.getsampwidth()==2
   raw=wav.readframes(wav.getnframes())
  assert len(raw)//2==clip['samples'] and hashlib.sha256(raw).hexdigest()==clip['pcm_sha256']
  pcm=np.frombuffer(raw,dtype='<i2').astype(np.float32)/32768
  started=time.monotonic()
  segments=session.transcribe(pcm,sample_rate=16000)
  results.append(dict(path=clip['path'],duration_s=clip['duration_s'],elapsed_s=time.monotonic()-started,
   pcm_sha256=clip['pcm_sha256'],segments=[dict(start=s.start,end=s.end,text=s.text) for s in segments]))
  Path(os.environ['ECHO_RESULTS']).write_text(json.dumps(results,indent=2,ensure_ascii=False)+'\n')
  print('completed',len(results),clip['path'],results[-1]['elapsed_s'],flush=True)
 assert len(results)==48
''')
env = dict(os.environ, PYTHONPATH=str(ROOT / 'python'),
    LD_LIBRARY_PATH=str(bundle) + ':' + os.environ.get('LD_LIBRARY_PATH', ''),
    CRISPASR_IMATRIX_OUT=str(stats_path), CRISPASR_LLAMA_PIPELINE_DISABLE='0',
    ECHO_CORPUS=str(corpus), ECHO_MODEL=str(models / 'index-echo-9b-f16.gguf'),
    ECHO_LIBRARY=str(library), ECHO_RESULTS=str(OUT / 'decoded-corpus.json'))
for name in ('CRISPASR_ACTDUMP_OUT', 'CRISPASR_ACTDUMP_TENSOR'):
    env.pop(name, None)
with (OUT / 'collection.log').open('w') as log, kh.build_heartbeat('corpus.collect', interval_s=30):
    result = subprocess.run([sys.executable, '-u', child], env=env, stdout=log,
        stderr=subprocess.STDOUT, timeout=7200)
receipt['collection_returncode'] = result.returncode
save()
assert result.returncode == 0
log = (OUT / 'collection.log').read_text()
receipt['cuda_assignment_proven'] = 'assigned to device CUDA' in log and 'load_tensors: layer' in log
assert receipt['cuda_assignment_proven'], 'No actual decoder CUDA assignment evidence'
assert stats_path.is_file(), 'No activation statistics emitted at child exit'
stats = GGUFReader(str(stats_path))
assert stats.fields['general.architecture'].contents() == 'crispasr-imatrix'
assert stats.fields['imatrix.version'].contents() == 1
decoder = GGUFReader(str(models / 'index-echo-9b-decoder-f16.gguf'))
weights = {tensor.name: list(map(int, tensor.shape)) for tensor in decoder.tensors}
required = {name for name in weights if re.fullmatch(r'blk\.\d+\.ffn_(gate|up|down)\.weight', name)}
assert len(required) == 96
coverage = {}
for tensor in stats.tensors:
    if tensor.name not in weights:
        continue
    values = np.asarray(tensor.data).reshape(-1)
    key = 'count.' + tensor.name
    count = int(stats.fields[key].contents()) if key in stats.fields else 0
    coverage[tensor.name] = dict(columns=int(values.size), weight_shape=weights[tensor.name], count=count,
        finite=bool(np.isfinite(values).all()), nonnegative=bool((values >= 0).all()),
        nonzero=bool((values > 0).any()), shape_match=int(values.size) == weights[tensor.name][0])
missing = sorted(required - coverage.keys())
invalid = sorted(name for name in required & coverage.keys() if not (
    coverage[name]['count'] > 0 and coverage[name]['finite'] and coverage[name]['nonnegative']
    and coverage[name]['nonzero'] and coverage[name]['shape_match']))
receipt.update(coverage=dict(required_ffn_matrices=96, decoder_matrices=len(coverage),
    missing=missing, invalid=invalid, tensors=coverage), imatrix_sha256=digest(stats_path),
    imatrix_bytes=stats_path.stat().st_size, clips=48,
    duration_s=sum(clip['duration_s'] for clip in manifest['clips']), heldout_pcm_disjoint=True)
save()
assert not missing and not invalid, 'Incomplete or invalid decoder calibration coverage'
receipt['calibration_complete'] = True
save()
kh.step('calibration-corpus.complete', passed=True, clips=48, q4_accepted=False)
