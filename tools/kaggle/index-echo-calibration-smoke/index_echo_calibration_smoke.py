#!/usr/bin/env python3
"""Prove decoder collection on GPUs without using acceptance audio for calibration.

Run unchanged, independent F16 acceptance with collection off and on. Audit
every decoder FFN activation width/count. The resulting imatrix is smoke-test
data ONLY: these are held-out acceptance clips, never a quantization corpus.
No CPU build, quantization, model upload or production default change occurs.
"""
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

SCRIPT_VERSION = '2026-10-09.1'
SOURCE_COMMIT = '327be3fda80c44f280bfecab988458f9c3712b08'
BUILD_COMMIT = 'a03820a4fcf32bae0b658b35cf6663e85abcc854'
BUILD_RUN = 37901223622
BUNDLE_REVISION = 'b284cbc8727e1f91904a4c2928e6991bd5bea938'
BUNDLE_SHA256 = '894bf650161f4c672a2992da394cdb8d65cca69af2f17c5b960c52bd4b6826f9'
ROOT = Path('/kaggle/temp/echo-calibration-sdk')
TEMP = Path('/kaggle/temp/echo-calibration-smoke')
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

receipt = dict(script_version=SCRIPT_VERSION, entry_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    source_commit=SOURCE_COMMIT, build_commit=BUILD_COMMIT, build_run=BUILD_RUN,
    bundle_revision=BUNDLE_REVISION, bundle_sha256=BUNDLE_SHA256, hardware=hardware,
    purpose='held-out F16 acceptance and decoder callback coverage smoke test',
    suitable_for_quantization=False, production_defaults_changed=False,
    q4_accepted=False, validated=False, arms={})


def save():
    (OUT / 'calibration-smoke.json').write_text(json.dumps(receipt, indent=2, ensure_ascii=False) + '\n')


def validate(name, collect):
    output = OUT / name
    output.mkdir()
    config = dict(source_commit=SOURCE_COMMIT, build_commit=BUILD_COMMIT, build_run=BUILD_RUN,
        bundle_revision=BUNDLE_REVISION, bundle_sha256=BUNDLE_SHA256, repo_dir=str(ROOT),
        temp_dir=str(TEMP / name), output_dir=str(output), keep_models=True)
    if collect:
        config['local_models'] = str(TEMP / 'control/models')
    config_path = output / 'runtime-config.json'
    config_path.write_text(json.dumps(config, indent=2) + '\n')
    env = dict(os.environ, INDEX_ECHO_VALIDATION_CONFIG=str(config_path), INDEX_ECHO_BENCH='1',
               CRISPASR_LLAMA_PIPELINE_DISABLE='0')
    for variable in ('CRISPASR_IMATRIX_OUT', 'CRISPASR_ACTDUMP_OUT', 'CRISPASR_ACTDUMP_TENSOR'):
        env.pop(variable, None)
    if collect:
        env['CRISPASR_IMATRIX_OUT'] = str(output / 'heldout-smoke-only.gguf')
    with (output / 'acceptance.log').open('w') as log, kh.build_heartbeat(name + '.acceptance', interval_s=30):
        result = subprocess.run([sys.executable, str(ROOT / 'tools/kaggle/index-echo-9b-validation/index_echo_9b_validation.py')],
            env=env, stdout=log, stderr=subprocess.STDOUT, timeout=7200)
    path = output / 'cuda-validation.json'
    if not path.is_file():
        raise RuntimeError(name + ': no acceptance receipt; inspect archived log')
    accepted = json.loads(path.read_text())
    receipt['arms'][name] = dict(returncode=result.returncode, validated=accepted['validated'],
        full_pipeline_checked=accepted['full_pipeline_checked'], failed=accepted['failed'])
    save()
    if result.returncode or not accepted['validated'] or not accepted['full_pipeline_checked']:
        raise RuntimeError(name + ': unchanged F16 acceptance failed; no quantization conclusion')
    return accepted


save()
control = validate('control', False)
observed = validate('observed', True)
receipt['decoded_receipts_identical'] = control['cohorts'] == observed['cohorts']
save()
if not receipt['decoded_receipts_identical']:
    raise RuntimeError('Collection changed decoded output or acceptance details; retain both arms')
# Both arms independently pass original Python stage/norm, exact bilingual
# outputs/timestamps, complete file pipeline and TTS-ASR roundtrips.
stats_path = OUT / 'observed/heldout-smoke-only.gguf'
if not stats_path.is_file():
    raise RuntimeError('No activation statistics: callback wiring is not proven')
stats = GGUFReader(str(stats_path))
assert stats.fields['general.architecture'].contents() == 'crispasr-imatrix'
assert stats.fields['imatrix.version'].contents() == 1
decoder = GGUFReader(str(TEMP / 'control/models/index-echo-9b-decoder-f16.gguf'))
weights = {tensor.name: list(map(int, tensor.shape)) for tensor in decoder.tensors}
required = {name for name in weights if re.fullmatch(r'blk\.\d+\.ffn_(gate|up|down)\.weight', name)}
assert len(required) == 96, 'Expected three FFN matrices in each of 32 decoder blocks'
coverage = {}
for tensor in stats.tensors:
    if tensor.name not in weights:
        continue  # encoder statistics are not decoder calibration evidence
    values = np.asarray(tensor.data).reshape(-1)
    key = 'count.' + tensor.name
    count = int(stats.fields[key].contents()) if key in stats.fields else 0
    coverage[tensor.name] = dict(columns=int(values.size), weight_shape=weights[tensor.name],
        count=count, finite=bool(np.isfinite(values).all()),
        nonnegative=bool((values >= 0).all()), nonzero=bool((values > 0).any()),
        shape_match=int(values.size) == weights[tensor.name][0])
missing = sorted(required - coverage.keys())
invalid = sorted(name for name in required & coverage.keys() if not (
    coverage[name]['count'] > 0 and coverage[name]['finite'] and coverage[name]['nonnegative']
    and coverage[name]['nonzero'] and coverage[name]['shape_match']))
receipt['coverage'] = dict(required_ffn_matrices=len(required), decoder_matrices=len(coverage),
    all_collected_matrices=len(stats.tensors), missing=missing, invalid=invalid, tensors=coverage,
    not_collected=sorted(weights.keys() - coverage.keys()))
receipt['imatrix_sha256'] = hashlib.sha256(stats_path.read_bytes()).hexdigest()
receipt['imatrix_bytes'] = stats_path.stat().st_size
save()
if missing or invalid:
    raise RuntimeError('Incomplete decoder coverage; retain proof and fix first missing/invalid tensor')
receipt['validated'] = True
save()
kh.step('calibration-smoke.complete', passed=True, decoder_matrices=len(coverage),
        required_ffn_matrices=len(required), suitable_for_quantization=False)
