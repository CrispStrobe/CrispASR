#!/usr/bin/env python3
"""Predeclared mixed-Q4 candidates, with unchanged independent acceptance.

CPU compilation, quantization and private uploads run on GitHub. This kernel
only downloads pinned candidates and executes them on actual GPUs. Rejected weights
remain private. No threshold, default, reference or decoded text is changed.
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

SCRIPT_VERSION = '2026-10-02.2'
SOURCE_COMMIT = '1d01260a6165ca26c89bc615afc42a54e158a04a'
BUILD_COMMIT = '012188eedaff3e027ba5b4e903eab3efafe4a46b'
BUILD_RUN = 37040196454
BUNDLE_REVISION = '51158bc93be3e4f2f65a3bde6296bd9a59712968'
BUNDLE_SHA256 = '0a4f882975c802a4ff3a1932fcb3ecf838cac39c3ccecc181240e331d0432303'
MODEL_REVISION = 'dffbadf0f173446fee0364a0807803d2b2fb6f49'
PRIVATE_REPO = 'cstr/index-echo-9b-staging-GGUF'
PREPARATION_RUN = 37044026371
PREPARATION_REVISION = 'PENDING'
PREPARATION_SHA256 = 'PENDING'
PREPARATION_PATH = 'q4-guards-20261002/preparation.json'
SDK = Path('/kaggle/temp/index-echo-q4-sdk')
TEMP = Path('/kaggle/temp/index-echo-q4')
OUT = Path('/kaggle/working')
TEMP.mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)
os.environ.update(TMPDIR=str(TEMP), HF_HOME=str(TEMP / 'hf'),
                  HF_XET_CACHE=str(TEMP / 'xet'), OMP_NUM_THREADS='4')


def run(*command, **kwargs):
    return subprocess.run(list(map(str, command)), check=True, **kwargs)


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b''):
            result.update(chunk)
    return result.hexdigest()


if any(len(pin) != 40 for pin in (SOURCE_COMMIT, BUILD_COMMIT, BUNDLE_REVISION, PREPARATION_REVISION)) or len(BUNDLE_SHA256) != 64 or len(PREPARATION_SHA256) != 64:
    raise RuntimeError('Successful immutable CI bundle and SDK pins required')
hardware = subprocess.check_output(['nvidia-smi', '--query-gpu=name,compute_cap,memory.total',
                                    '--format=csv,noheader'], text=True).strip()
rows = [row.split(',') for row in hardware.splitlines()]
if len(rows) < 2 or any(row[1].strip() != '7.5' for row in rows):
    (OUT / 'inconclusive.json').write_text(json.dumps(dict(hardware=hardware, conclusive=False,
        reason='SM75 CI bundle and F16 control require two actual T4s; no weights pulled')))
    raise SystemExit(0)
run('git', 'init', SDK)
run('git', '-C', SDK, 'remote', 'add', 'origin', 'https://github.com/CrispStrobe/CrispASR.git')
run('git', '-C', SDK, 'fetch', '--depth=1', 'origin', SOURCE_COMMIT)
run('git', '-C', SDK, 'checkout', 'FETCH_HEAD')
sys.path.insert(0, str(SDK / 'tools/kaggle'))
import kaggle_harness as kh
kh.init_progress()
kh.provenance(SCRIPT_VERSION, SDK)
run(sys.executable, '-m', 'pip', 'install', '-q', 'huggingface_hub', 'gguf', 'numpy')
from huggingface_hub import hf_hub_download
os.environ['HF_TOKEN'] = kh.resolve_hf_token(require=True)
kh.step('q4.start', hardware=hardware, build_run=BUILD_RUN)
base_config = dict(source_commit=SOURCE_COMMIT, build_commit=BUILD_COMMIT, build_run=BUILD_RUN,
                   bundle_revision=BUNDLE_REVISION, bundle_sha256=BUNDLE_SHA256,
                   repo_dir=str(SDK), keep_models=True)
validator = SDK / 'tools/kaggle/index-echo-9b-validation/index_echo_9b_validation.py'


def validate(name, additions):
    output = OUT / name
    output.mkdir()
    config = dict(base_config, temp_dir=str(TEMP / name), output_dir=str(output), **additions)
    config_path = output / 'runtime-config.json'
    config_path.write_text(json.dumps(config, indent=2) + '\n')
    env = dict(os.environ, INDEX_ECHO_VALIDATION_CONFIG=str(config_path), INDEX_ECHO_BENCH='1',
               CRISPASR_LLAMA_PIPELINE_DISABLE='0')
    with (output / 'acceptance.log').open('w') as log, kh.build_heartbeat(name+'.acceptance', interval_s=30):
        result = subprocess.run([sys.executable, str(validator)], env=env,
                                stdout=log, stderr=subprocess.STDOUT, timeout=3600)
    receipt_path = output / 'cuda-validation.json'
    if not receipt_path.exists():
        raise RuntimeError(name + ': validator failed before producing a receipt; inspect acceptance.log')
    receipt = json.loads(receipt_path.read_text())
    assert bool(result.returncode == 0) == bool(receipt['validated'])
    assert receipt['full_pipeline_checked'], 'Incomplete acceptance is not a quant result'
    kh.step(name+'.complete', passed=receipt['validated'], failed=receipt['failed'])
    return receipt


# Permission/provenance preflight precedes any model download or execution.
prepared_path = Path(hf_hub_download(PRIVATE_REPO, PREPARATION_PATH,
    revision=PREPARATION_REVISION, local_dir=TEMP / 'preparation'))
assert digest(prepared_path) == PREPARATION_SHA256, 'Preparation receipt checksum mismatch'
prepared = json.loads(prepared_path.read_text())
assert prepared['preparation_only'] and prepared['validated'] is False
assert prepared['source_revision'] == MODEL_REVISION
assert set(prepared['recipes']) == {'q4_k_plain', 'q4_k_sensitive', 'q4_k_ffn_guarded', 'q4_k_middle'}
# Same compiled runtime/hardware must first reproduce the accepted F16 pair.
control = validate('f16-control', {})
assert control['validated'], 'F16 control failed; do not blame or measure quants'
source = TEMP / 'f16-control/models'
results = dict(script_version=SCRIPT_VERSION, source_commit=SOURCE_COMMIT,
               build_commit=BUILD_COMMIT, build_run=BUILD_RUN, hardware=hardware,
               source_model_revision=MODEL_REVISION, private_repo=PRIVATE_REPO,
               preparation_run=PREPARATION_RUN, preparation_revision=PREPARATION_REVISION,
               preparation_sha256=PREPARATION_SHA256,
               control_passed=True, defaults_changed=False, candidates={})
for name, recipe in prepared['recipes'].items():
    models = TEMP / ('models-' + name)
    models.mkdir()
    primary = models / ('index-echo-9b-' + name + '.gguf')
    primary.symlink_to(source / 'index-echo-9b-f16.gguf')
    # Original primary metadata resolves this exact basename. Keep it unchanged
    # while testing the physically audited mixed decoder in an isolated folder.
    decoder = models / 'index-echo-9b-decoder-f16.gguf'
    with kh.build_heartbeat(name+'.download', interval_s=30):
        downloaded = Path(hf_hub_download(PRIVATE_REPO, recipe['path'],
            revision=PREPARATION_REVISION, local_dir=models / 'download'))
    assert downloaded.stat().st_size == recipe['decoder_bytes']
    assert digest(downloaded) == recipe['decoder_sha256']
    decoder.symlink_to(downloaded)
    receipt = validate(name, dict(local_models=str(models), cohorts=[name], model_recipe=recipe))
    results['candidates'][name] = dict(recipe=recipe, validated=receipt['validated'], failed=receipt['failed'])
    (OUT / 'q4-results.json').write_text(json.dumps(results, indent=2, ensure_ascii=False) + '\n')
    decoder.unlink()
    downloaded.unlink()
    kh.step(name+'.weights-released')
results['any_candidate_passed'] = any(row['validated'] for row in results['candidates'].values())
(OUT / 'q4-results.json').write_text(json.dumps(results, indent=2, ensure_ascii=False) + '\n')
if not results['any_candidate_passed']:
    raise RuntimeError('All Q4 candidates rejected; retained full diagnostics; no default or publication change')
