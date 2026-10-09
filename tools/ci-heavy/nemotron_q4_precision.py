#!/usr/bin/env python3
"""Isolate RNNT precision guards on fixed original-ASR controls; no publication."""
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import zipfile

from gguf import GGUFReader
from huggingface_hub import hf_hub_download
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(os.environ['HEAVY_OUT'])
TEMP = Path(os.environ['HEAVY_SCRATCH']) / 'nemotron-q4-precision'
OUT.mkdir(parents=True, exist_ok=True)
TEMP.mkdir(parents=True, exist_ok=True)
os.environ.update(HF_HOME=str(TEMP / 'hf'), HF_XET_CACHE=str(TEMP / 'xet'),
                  CRISPASR_NEMOTRON_CONTEXT_PRESET='0', OMP_NUM_THREADS='4')
MODEL = 'cstr/nemotron-3.5-asr-streaming-GGUF'
MODEL_REV = 'bbd95a9ca5fa0dfca3312a122dfc45a2b578b9c2'
FIXTURES = 'cstr/crispasr-regression-fixtures'
PROOF_REV = '10e579dedf52a16a159cbc5d35e17f8bffa77190'
PROOF_SHA = '7762f2a73d8ae7dad05d667fbf0bca393a0b3b7ad622b1b1f09473fa8f5d0463'
receipt = dict(source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
    diagnostic_complete=False, production_accepted=False, defaults_changed=False,
    model_repo=MODEL, model_revision=MODEL_REV, original_proof_revision=PROOF_REV,
    original_proof_sha256=PROOF_SHA, models={})


def save():
    (OUT / 'q4-precision.json').write_text(json.dumps(receipt, indent=2) + '\n')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def run(command, label):
    with (OUT / (label + '.log')).open('w') as log:
        result = subprocess.run(list(map(str, command)), cwd=ROOT, stdout=log,
            stderr=subprocess.STDOUT, timeout=3600)
    return result.returncode


def words(text):
    return re.findall('[a-z]+', re.sub(r'<[^>]*>', '', text).lower())


save()
archive = hf_hub_download(FIXTURES, 'tts-asr/nemotron-reference-20261009/proof.zip', revision=PROOF_REV)
assert digest(archive) == PROOF_SHA
reference = TEMP / 'reference'
with zipfile.ZipFile(archive) as zipped:
    zipped.extractall(reference)
prior_path = next(reference.rglob('nemotron-tts-reference.json'))
prior = json.loads(prior_path.read_text())
assert prior['diagnostic_complete'] and len(prior['original']) == 26
cases = {}
for label, case in prior['cases'].items():
    pcm = np.load(prior_path.parent / case['file'], allow_pickle=False)
    assert hashlib.sha256(pcm.tobytes()).hexdigest() == case['sha256']
    assert words(prior['original'][label]['transcript']) == words(prior['native']['f16/fresh/' + label]['transcript'])
    cases[label] = pcm
build = TEMP / 'build'
assert run(['cmake', '-S', ROOT, '-B', build, '-G', 'Ninja', '-DCMAKE_BUILD_TYPE=Release',
    '-DBUILD_SHARED_LIBS=ON', '-DGGML_NATIVE=OFF', '-DGGML_CUDA=OFF', '-DGGML_BLAS=OFF',
    '-DCRISPASR_BUILD_TESTS=OFF', '-DCRISPASR_BUILD_SERVER=OFF'], 'configure') == 0
assert run(['cmake', '--build', build, '--target', 'crispasr-lib', 'crispasr-diff', 'crispasr-quantize', '-j4'], 'build') == 0
sys.path.insert(0, str(ROOT / 'python'))
from crispasr import Session
library = next(build.rglob('libcrispasr.so'))
f16 = Path(hf_hub_download(MODEL, 'nemotron-3.5-asr-streaming-0.6b-f16.gguf', revision=MODEL_REV))
published = Path(hf_hub_download(MODEL, 'nemotron-3.5-asr-streaming-0.6b-q4_k.gguf', revision=MODEL_REV))
for quant, path in (('f16', f16), ('q4_k', published)):
    assert digest(path) == prior['native_models'][quant]['sha256']
source = GGUFReader(str(f16))
source_types = {t.name: t.tensor_type.name for t in source.tensors}
assert source.fields['general.architecture'].contents() == 'nemotron'
del source
manifest = json.loads((ROOT / 'tests/regression/manifest.json').read_text())
entry = next(x for x in manifest['backends'] if x['backend_id'] == 'nemotron')
ref = hf_hub_download(manifest['fixtures']['repo'], entry['fixture_ref_path'],
                      revision=manifest['fixtures']['revision'])
receipt['reference_stage_pin'] = dict(revision=manifest['fixtures']['revision'], path=entry['fixture_ref_path'],
    thresholds=entry['diff_thresholds'], default_threshold=entry['stage_threshold_default'])
# The existing strict stage gate is retained. Q4 transcript recovery alone
# cannot promote a model whose stage/magnitude evidence remains rejected.
specifications = [('f16', f16, None), ('published-q4', published, None),
    ('plain-q4', TEMP / 'plain-q4.gguf', []),
    ('joint-embed-q4', TEMP / 'joint-embed-q4.gguf', [r'^joint\.', r'^decoder\.embed\.']),
    ('rnnt-prompt-q4', TEMP / 'rnnt-prompt-q4.gguf', [r'^joint\.', r'^decoder\.', r'^prompt_kernel\.'])]
baseline = None
for name, path, patterns in specifications:
    protected = []
    if patterns is not None:
        protected = [key for key in source_types if any(re.search(pattern, key) for pattern in patterns)]
        command = [build / 'bin/crispasr-quantize', f16, path, 'q4_k']
        for key in protected:
            assert source_types[key] in ('F16', 'F32'), 'Guard must retain actual source precision'
            command.extend(['--tensor-type', '^' + re.escape(key) + '$=' + source_types[key].lower()])
        assert run(command, name + '-quantize') == 0
    reader = GGUFReader(str(path))
    tensors = {t.name: dict(type=t.tensor_type.name, shape=list(map(int, t.shape)),
        bytes=int(t.n_bytes), sha256=hashlib.sha256(t.data.tobytes()).hexdigest()) for t in reader.tensors}
    del reader
    for key in protected:
        assert tensors[key]['type'] == source_types[key], 'Critical precision was not retained'
    if name == 'plain-q4':
        baseline = tensors
    if patterns:
        assert baseline is not None
        assert all(tensors[key] == value for key, value in baseline.items() if key not in protected), 'Guard changed unrelated encoder weights'
    result = dict(bytes=path.stat().st_size, sha256=digest(path), protected=protected,
        q4_bytes=sum(t['bytes'] for t in tensors.values() if t['type'] == 'Q4_K'),
        tensors=tensors, fresh={}, reused={}, state_changes=[], original_word_matches=0,
        voxcpm_exact=0, voxcpm_cases=22, stage_returncode=None)
    receipt['models'][name] = result
    result['stage_returncode'] = run([build / 'bin/crispasr-diff', 'nemotron', path, ref,
        ROOT / entry['sample']], name + '-diff')
    if name == 'f16':
        assert result['stage_returncode'] == 0, 'F16 strict stage control must pass first'
    with Session(str(path), lib_path=str(library), backend='nemotron', n_threads=4) as session:
        for label, pcm in cases.items():
            text = ' '.join(s.text for s in session.transcribe(pcm, sample_rate=16000, language='en'))
            result['fresh'][label] = text
            result['original_word_matches'] += words(text) == words(prior['original'][label]['transcript'])
            if label.startswith('voxcpm2'):
                result['voxcpm_exact'] += words(text) == words(prior['cases'][label]['expected'])
            save()
    with Session(str(path), lib_path=str(library), backend='nemotron', n_threads=4) as session:
        for label, pcm in reversed(list(cases.items())):
            text = ' '.join(s.text for s in session.transcribe(pcm, sample_rate=16000, language='en'))
            result['reused'][label] = text
            if text != result['fresh'][label]:
                result['state_changes'].append(label)
    result['decoded_matrix_passed'] = result['original_word_matches'] == 26 and result['voxcpm_exact'] == 22 and not result['state_changes']
    result['existing_stage_gate_passed'] = result['stage_returncode'] == 0
    save()
    print(name, 'original words', result['original_word_matches'], '/26, Vox exact',
          result['voxcpm_exact'], '/22, bytes', result['bytes'], 'stage rc', result['stage_returncode'], flush=True)
    if patterns is not None:
        path.unlink()
receipt['diagnostic_complete'] = True
save()
(OUT / 'summary.md').write_text('Completed fixed-audio quantization diagnosis with original/F16 controls.\n'
    'Physical tensor types, byte sizes, raw strict stage/norm logs and fresh/reused decoded outputs retained.\n'
    'No candidate weights published; no defaults, stage gates or TTS acceptance thresholds changed.\n')
