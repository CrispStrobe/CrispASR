#!/usr/bin/env python3
"""Publish a clean F16-only repository after independent acceptance.

Experimental model history must stay private. Download, hash and upload weights
on the hosted runner; the shared VPS never holds the paired 19 GB model.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

from huggingface_hub import HfApi, hf_hub_download

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--staging-repo', required=True)
p.add_argument('--staging-revision', required=True)
p.add_argument('--acceptance-revision', required=True)
p.add_argument('--destination', default='cstr/index-echo-9b-GGUF')
a = p.parse_args()
for pin in [a.staging_revision, a.acceptance_revision]:
    if len(pin) != 40 or any(c not in '0123456789abcdef' for c in pin):
        p.error('Immutable source and acceptance revisions are required')
if a.staging_repo == a.destination:
    p.error('Experimental history and clean publication must use separate repositories')
api = HfApi()
if not api.model_info(a.staging_repo).private:
    raise RuntimeError('Experimental staging history must remain private')
out = Path(os.environ['HEAVY_OUT'])
scratch = Path(os.environ['HEAVY_SCRATCH']) / 'index-echo-publication'
out.mkdir(parents=True, exist_ok=True)
scratch.mkdir(parents=True, exist_ok=True)
os.environ['TMPDIR'] = str(scratch)
acceptance = Path(hf_hub_download(a.staging_repo, 'acceptance.json', revision=a.acceptance_revision,
                                 local_dir=scratch / 'metadata'))
receipt = json.loads(acceptance.read_text())
if receipt.get('validated') is not True or receipt.get('accepted_cohort') != 'f16':
    raise RuntimeError('Independent model acceptance has not passed')
if receipt.get('model_revision') != a.staging_revision:
    raise RuntimeError('Acceptance does not identify the requested immutable model revision')
if receipt.get('source_model') != 'IndexTeam/Index-Echo-S2TT-9B':
    raise RuntimeError('Acceptance belongs to a different source model')
for gate in ['cpu', 'cuda', 'pipeline', 'roundtrip']:
    if receipt.get(gate, {}).get('passed') is not True:
        raise RuntimeError('Mandatory acceptance gate missing: ' + gate)
expected = {
    'index-echo-9b-f16.gguf': (1313848736, 'e4b1dd177211ba5d8eb0ea7140ac2f813ddc68b55381f050ecf7ddab3e83172f'),
    'index-echo-9b-decoder-f16.gguf': (17920696992, '98acd9b753295cdc9a9b8773317ab046bf83e466b1ebfc775d59ac8dab9d177a'),
    'LICENSE': (11358, 'c95bae1d1ce0235ecccd3560b772ec1efb97f348a79f0fbe0a634f0c2ccefe2c'),
}
for name, (size, digest) in expected.items():
    artifact = receipt.get('artifacts', {}).get(name, {})
    if artifact.get('bytes') != size or artifact.get('sha256') != digest:
        raise RuntimeError('Acceptance does not cover the exact published artifact: ' + name)
folder = scratch / 'clean'
folder.mkdir(exist_ok=True)
for name, (size, digest) in expected.items():
    path = Path(hf_hub_download(a.staging_repo, name, revision=a.staging_revision, local_dir=folder))
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            h.update(block)
    if path.stat().st_size != size or h.hexdigest() != digest:
        raise RuntimeError('Tested artifact hash mismatch: ' + name)
    print('Verified', name, size, digest, flush=True)
for name in ['README.md', 'acceptance.json']:
    source = hf_hub_download(a.staging_repo, 'publication/' + name, revision=a.acceptance_revision,
                            local_dir=scratch / 'metadata') if name == 'README.md' else acceptance
    shutil.copy2(source, folder / name)
# Only this clean history can become public. Interrupted uploads remain private.
api.create_repo(a.destination, private=True, exist_ok=True)
if not api.model_info(a.destination).private:
    raise RuntimeError('Initial publication target must be private')
allowed = set(expected) | {'README.md', 'acceptance.json', '.gitattributes'}
# An interrupted clean upload can resume. Audit history too: deleting rejected
# files at HEAD would still expose them through their earlier public commits.
for commit in api.list_repo_commits(a.destination):
    if set(api.list_repo_files(a.destination, revision=commit.commit_id)) - allowed:
        raise RuntimeError('Publication target contains experimental history')
for attempt in range(8):
    try:
        api.upload_large_folder(a.destination, folder_path=str(folder), repo_type='model',
                                allow_patterns=list(expected) + ['README.md', 'acceptance.json'], print_report=False)
        break
    except Exception:
        if attempt == 7:
            raise
        print('Retrying resumable validated upload', attempt + 1, flush=True)
        time.sleep(5)
files = set(api.list_repo_files(a.destination))
if files != allowed:
    raise RuntimeError('Published files differ from the validated F16-only set: ' + repr(files))
info = api.model_info(a.destination, files_metadata=True)
if info.card_data is None or info.card_data.get('license') != 'apache-2.0':
    raise RuntimeError('License attribution is missing from the committed card')
for item in info.siblings:
    if item.rfilename in expected:
        size, digest = expected[item.rfilename]
        if item.size != size or (item.lfs and item.lfs.sha256 != digest):
            raise RuntimeError('Remote artifact metadata mismatch: ' + item.rfilename)
api.update_repo_settings(a.destination, private=False)
if api.model_info(a.destination).private or not api.model_info(a.staging_repo).private:
    raise RuntimeError('Publication visibility verification failed')
result = dict(validated=True, repository=a.destination, revision=info.sha,
              staging_repository=a.staging_repo, staging_revision=a.staging_revision,
              acceptance_revision=a.acceptance_revision, artifacts=expected)
(out / 'publication.json').write_text(json.dumps(result, indent=2) + '\n')
(out / 'summary.md').write_text('Clean F16-only Index-Echo 9B publication verified; experimental history remains private.\n')
print(json.dumps(result), flush=True)
