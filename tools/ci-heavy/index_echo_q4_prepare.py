#!/usr/bin/env python3
"""Stream-quantize Echo on hosted CPU; public experimental upload for real GPU acceptance."""
import hashlib
import json
import os
import re
from pathlib import Path
import subprocess
import sys

from huggingface_hub import HfApi, snapshot_download
from gguf import GGUFReader

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(os.environ['HEAVY_OUT'])
TEMP = Path(os.environ['HEAVY_SCRATCH']) / 'echo-q4'
OUT.mkdir(parents=True, exist_ok=True)
TEMP.mkdir(parents=True, exist_ok=True)
os.environ.update(HF_HOME=str(TEMP/'hf'), HF_XET_CACHE=str(TEMP/'xet'), TMPDIR=str(TEMP), OMP_NUM_THREADS='4')
sys.path.insert(0, str(ROOT / 'tools'))
from index_echo_quant_recipes import RECIPES, audit
SOURCE_REVISION = 'dffbadf0f173446fee0364a0807803d2b2fb6f49'
PREPARATION_REPO = 'cstr/index-echo-9b-GGUF'
PREFIX = 'experiments/q4-guards-20261008'
api = HfApi(token=os.environ['HF_TOKEN'])
assert not api.repo_info(PREPARATION_REPO).private, 'Preparation must use the authorized public repository'
receipt = dict(source_revision=SOURCE_REVISION, source_repo='cstr/index-echo-9b-GGUF',
               preparation_repo=PREPARATION_REPO, prefix=PREFIX, validated=False,
               preparation_only=True, recipes={})
receipt['source_commit'] = subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
# An experiment prefix leaves all published model files and registry pins intact.
# This marker explicitly disclaims runtime acceptance; it also proves write
# permission before a large download/build (it cannot prove available quota).
preflight = OUT / 'preflight.json'
preflight.write_text(json.dumps(receipt, indent=2)+'\n')
api.upload_file(repo_id=PREPARATION_REPO, path_or_fileobj=str(preflight),
                path_in_repo=PREFIX+'/preflight.json', commit_message='Unvalidated public Q4 experiment: preparation write preflight')
build = TEMP / 'build'
for args in [
    ['cmake', '-S', ROOT, '-B', build, '-G', 'Ninja', '-DCMAKE_BUILD_TYPE=Release',
     '-DGGML_NATIVE=OFF', '-DGGML_CUDA=OFF', '-DCRISPASR_BUILD_TESTS=OFF', '-DCRISPASR_BUILD_SERVER=OFF'],
    ['cmake','--build',build,'--target','crispasr-quantize','-j4'],
]:
    subprocess.run(list(map(str,args)),check=True)
models = Path(snapshot_download('cstr/index-echo-9b-GGUF', revision=SOURCE_REVISION,
    local_dir=TEMP/'source', allow_patterns=['index-echo-9b-f16.gguf','index-echo-9b-decoder-f16.gguf']))
decoder = models/'index-echo-9b-decoder-f16.gguf'
primary = models/'index-echo-9b-f16.gguf'
source = GGUFReader(str(decoder))
source_types = {t.name: t.tensor_type.name for t in source.tensors}
del source


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024**2),b''):h.update(chunk)
    return h.hexdigest()


receipt['primary_sha256'] = sha256(primary)
receipt['decoder_source_sha256'] = sha256(decoder)
for name, rules in RECIPES.items():
    # Recurrent convolution matrices are F32 in the original decoder. Broad
    # F16 guards must be a precision floor, never a downcast of those weights.
    source_f32 = [re.escape(t) for t, kind in source_types.items() if kind == 'F32']
    if source_f32:
        rules = ['^('+'|'.join(source_f32)+')$=f32'] + rules
    target = TEMP/(name+'.gguf')
    args = [build/'bin/crispasr-quantize',decoder,target,'q4_k']
    for rule in rules:args += ['--tensor-type',rule]
    with (OUT/(name+'-quantize.log')).open('w') as log:
        subprocess.run(list(map(str,args)),check=True,stdout=log,stderr=subprocess.STDOUT,timeout=1800)
    reader = GGUFReader(str(target))
    tensors = [dict(name=t.name,type=t.tensor_type.name,shape=list(map(int,t.shape)),bytes=int(t.n_bytes)) for t in reader.tensors]
    del reader
    assert all(t['type']==source_types[t['name']] for t in tensors if len(t['shape'])<2), 'Small tensors must retain source types'
    assert all(t['type']=='F32' for t in tensors if source_types[t['name']]=='F32'), 'Source-F32 tensors must remain F32'
    recipe = dict(overrides=rules,tensors=tensors,q4_k_bytes=audit(tensors,name),
                  decoder_bytes=target.stat().st_size,primary_bytes=primary.stat().st_size,
                  decoder_sha256=sha256(target),source_revision=SOURCE_REVISION,
                  source_primary_unchanged=True, path=PREFIX+'/'+name+'/decoder.gguf')
    manifest = OUT/(name+'-recipe.json')
    upload = api.upload_file(repo_id=PREPARATION_REPO,path_or_fileobj=str(target),path_in_repo=recipe['path'],
                            commit_message='Experimental '+name+'; GPU acceptance pending')
    recipe['weight_revision'] = upload.oid
    # Verify the immutable remote object before deleting the only local result.
    remote = api.get_paths_info(PREPARATION_REPO, [recipe['path']], revision=upload.oid)[0]
    assert remote.size == recipe['decoder_bytes'], 'Uploaded size mismatch'
    assert remote.lfs and remote.lfs.sha256 == recipe['decoder_sha256'], 'Uploaded SHA256 mismatch'
    recipe['remote_verified'] = True
    manifest.write_text(json.dumps(recipe, indent=2)+'\n')
    api.upload_file(repo_id=PREPARATION_REPO,path_or_fileobj=str(manifest),path_in_repo=PREFIX+'/'+name+'/recipe.json')
    receipt['recipes'][name] = recipe
    (OUT/'q4-preparation.json').write_text(json.dumps(receipt,indent=2)+'\n')
    target.unlink()
    print(name,recipe['decoder_bytes'],'bytes, Q4 bytes',recipe['q4_k_bytes'],flush=True)
final = api.upload_file(repo_id=PREPARATION_REPO,path_or_fileobj=str(OUT/'q4-preparation.json'),
                        path_in_repo=PREFIX+'/preparation.json',commit_message='Completed Q4 preparation; GPU acceptance pending')
receipt['preparation_revision'] = final.oid
(OUT/'q4-preparation.json').write_text(json.dumps(receipt,indent=2)+'\n')
(OUT/'summary.md').write_text('Four mixed-Q4 candidates prepared and physically audited; no runtime acceptance claim.\n')
