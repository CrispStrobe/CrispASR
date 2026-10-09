#!/usr/bin/env python3
"""Locate Q4 frontend/subsampling drift under unchanged reference stage gates."""
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess

from gguf import GGUFReader
from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(os.environ['HEAVY_OUT'])
TEMP = Path(os.environ['HEAVY_SCRATCH']) / 'nemotron-q4-boundaries'
OUT.mkdir(parents=True, exist_ok=True)
TEMP.mkdir(parents=True, exist_ok=True)
os.environ.update(HF_HOME=str(TEMP / 'hf'), HF_XET_CACHE=str(TEMP / 'xet'),
                  CRISPASR_NEMOTRON_CONTEXT_PRESET='0', OMP_NUM_THREADS='4')
MODEL = 'cstr/nemotron-3.5-asr-streaming-GGUF'
REV = 'bbd95a9ca5fa0dfca3312a122dfc45a2b578b9c2'
receipt = dict(source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
    diagnostic_complete=False, production_accepted=False, defaults_changed=False,
    model_repo=MODEL, model_revision=REV, models={})


def save():
    (OUT / 'q4-boundaries.json').write_text(json.dumps(receipt, indent=2) + '\n')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def run(command, label):
    with (OUT / (label + '.log')).open('w') as log:
        return subprocess.run(list(map(str, command)), cwd=ROOT, stdout=log,
                              stderr=subprocess.STDOUT, timeout=3600).returncode


save()
build = TEMP / 'build'
assert run(['cmake', '-S', ROOT, '-B', build, '-G', 'Ninja', '-DCMAKE_BUILD_TYPE=Release',
    '-DGGML_NATIVE=OFF', '-DGGML_CUDA=OFF', '-DGGML_BLAS=OFF', '-DCRISPASR_BUILD_TESTS=OFF',
    '-DCRISPASR_BUILD_SERVER=OFF'], 'configure') == 0
assert run(['cmake', '--build', build, '--target', 'crispasr-diff', 'crispasr-quantize', '-j4'], 'build') == 0
manifest = json.loads((ROOT / 'tests/regression/manifest.json').read_text())
entry = next(x for x in manifest['backends'] if x['backend_id'] == 'nemotron')
ref = hf_hub_download(manifest['fixtures']['repo'], entry['fixture_ref_path'],
                      revision=manifest['fixtures']['revision'])
receipt['reference'] = dict(revision=manifest['fixtures']['revision'], path=entry['fixture_ref_path'],
    sha256=sha(ref), thresholds=entry['diff_thresholds'], default_threshold=entry['stage_threshold_default'])
reference = GGUFReader(ref)
receipt['reference']['stages'] = {t.name: list(map(int, t.shape)) for t in reference.tensors}
del reference
for stage in ('mel_spectrogram', 'pre_encode_output', 'encoder_output'):
    assert stage in receipt['reference']['stages']
f16 = Path(hf_hub_download(MODEL, 'nemotron-3.5-asr-streaming-0.6b-f16.gguf', revision=REV))
reader = GGUFReader(str(f16))
source_types = {t.name: t.tensor_type.name for t in reader.tensors}
del reader
head = [r'^joint\.', r'^decoder\.', r'^prompt_kernel\.']
specs = [('f16', None), ('plain-q4', []), ('rnnt-prompt-q4', head),
         ('rnnt-prompt-preout-q4', head + [r'^encoder\.pre\.out\.']),
         ('rnnt-prompt-preall-q4', head + [r'^encoder\.pre\.'])]
baseline = None
for name, patterns in specs:
    path = f16 if patterns is None else TEMP / (name + '.gguf')
    protected = []
    if patterns is not None:
        protected = [key for key in source_types if any(re.search(p, key) for p in patterns)]
        cmd = [build / 'bin/crispasr-quantize', f16, path, 'q4_k']
        for key in protected:
            assert source_types[key] in ('F16', 'F32')
            cmd.extend(['--tensor-type', '^' + re.escape(key) + '$=' + source_types[key].lower()])
        assert run(cmd, name + '-quantize') == 0
    reader = GGUFReader(str(path))
    tensors = {t.name: dict(type=t.tensor_type.name, shape=list(map(int, t.shape)), bytes=int(t.n_bytes),
                           sha256=hashlib.sha256(t.data.tobytes()).hexdigest()) for t in reader.tensors}
    del reader
    for key in protected:
        assert tensors[key]['type'] == source_types[key]
    if name == 'plain-q4':
        baseline = tensors
    if patterns:
        assert all(tensors[k] == v for k, v in baseline.items() if k not in protected)
    result = dict(bytes=path.stat().st_size, sha256=sha(path), protected=protected, tensors=tensors)
    receipt['models'][name] = result
    result['returncode'] = run([build / 'bin/crispasr-diff', 'nemotron', path, ref,
                               ROOT / entry['sample']], name + '-diff')
    log = (OUT / (name + '-diff.log')).read_text()
    result['stages'] = {stage: re.findall(r'^\[(?:PASS|FAIL|ERR )\].*' + stage + r'.*$', log, re.M)
                        for stage in ('mel_spectrogram', 'pre_encode_output', 'encoder_output')}
    save()
    print(name, result['returncode'], result['stages'], flush=True)
    # Stop if the independently pinned F16 reference control fails. Never
    # change layout/shape, pad or relax thresholds to make Q4 look accepted.
    if name == 'f16':
        assert result['returncode'] == 0, 'F16 must pass all three unchanged stage gates first'
        assert all(len(rows) == 1 and rows[0].startswith('[PASS]') for rows in result['stages'].values())
    if patterns is not None:
        path.unlink()
receipt['diagnostic_complete'] = True
save()
(OUT / 'summary.md').write_text('Boundary diagnosis complete; no model/default promotion.\n'
    'Three actual reference stages, exact shapes, original thresholds and per-tensor byte audits retained.\n')
