#!/usr/bin/env python3
"""Validate the standalone critical-DAC export against both retained originals.

Startup and complete-utterance code replay are numerical controls. Neither
establishes native free speech, GPU behavior or production-default promotion.
"""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(os.environ['HEAVY_OUT'])
SCRATCH = Path(os.environ['HEAVY_SCRATCH'])
OUT.mkdir(parents=True, exist_ok=True)
SCRATCH.mkdir(parents=True, exist_ok=True)
os.environ['TMPDIR'] = str(SCRATCH)
sys.path.insert(0, str(ROOT / 'tools'))
from dia_dac_export import digest

receipt = dict(source=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
               runs=[], production_defaults_changed=False, speech_acceptance=False)


def run(command, name):
    path = OUT / (name + '.log')
    with path.open('w') as log:
        result = subprocess.run(list(map(str, command)), cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, timeout=7200)
    receipt['runs'].append(dict(name=name, exit=result.returncode, log_sha256=digest(path)))
    (OUT / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(name, result.returncode, path.read_text()[-2000:], flush=True)
    assert result.returncode == 0, name


run([sys.executable, ROOT / 'tests/test_dia_dac_export.py', '-v'], 'tiny-physical-export-contracts')
folders = {}
for label, number in [('startup', 38294496529), ('full', 38285566623)]:
    archive = SCRATCH / label
    run(['gh', 'run', 'download', number, '-R', 'CrispStrobe/CrispASR', '-D', archive], label + '-download')
    folders[label] = next(archive.glob('heavy-*'))
startup, full = folders['startup'], folders['full']
old = json.loads((startup / 'receipt.json').read_text())
assert old['source'] == '778dac9ccbd4dae8592117e8e2cc2f76c608a3b2' and old['codec_diagnostic_pass']
assert digest(full / 'ref.gguf') == '3db6f8923abffd612e6c83d7036debebd10ed28b77c3eb9ec3d91fec1ce1a542'
assert digest(full / 'original-reported.npy') == '5c4331e165afa1e5342cf4cf3f9c50e7100e3a192a4835beb8e7c66d4fa13d6d'
from huggingface_hub import hf_hub_download
repo, revision = 'cstr/dia-1.6b-GGUF', '3233fbcb32be47761d2e736857b6d1a075b9ba7e'
template = hf_hub_download(repo, 'dac-44khz.gguf', revision=revision)
weights = json.loads((startup / 'codec-weight-controls.json').read_text())
assert digest(template) == weights['published_codec_sha256']
candidate = OUT / 'dac-output-f32.gguf'
run([sys.executable, ROOT / 'models/convert-dia-dac-to-gguf.py', '--template', template,
     '--output', candidate, '--precision', 'output-f32', '--receipt', OUT / 'physical-tensor-audit.json'], 'standalone-conversion')
validated = next(row for row in weights['variants'] if row['variant'] == 'output-f32')
assert digest(candidate) == validated['codec_sha256'] == digest(startup / 'dac-output-f32.gguf'), 'Export changed validated bytes'
receipt['exact_validated_candidate_bytes'] = True
receipt['physical_export'] = json.loads((OUT / 'physical-tensor-audit.json').read_text())
build = SCRATCH / 'native-build'
run(['cmake', '-S', ROOT, '-B', build, '-DCMAKE_BUILD_TYPE=Release', '-DBUILD_SHARED_LIBS=ON',
     '-DGGML_NATIVE=OFF', '-DGGML_CUDA=OFF', '-DGGML_VULKAN=OFF', '-DGGML_BLAS=OFF',
     '-DCRISPASR_BUILD_TESTS=OFF', '-DCRISPASR_BUILD_SERVER=OFF', '-DCRISPASR_OPUS=OFF', '-DCRISPASR_AMR=OFF'], 'configure')
run(['cmake', '--build', build, '--target', 'crispasr-diff', '-j4'], 'build-diff')
model = hf_hub_download(repo, 'dia-1.6b-f16.gguf', revision=revision)
os.environ['DIA_CODEC'] = str(candidate)
for label, path in [('startup', startup / 'canonical-codec-ref.gguf'), ('full', full / 'ref.gguf')]:
    reference = OUT / (label + '-original-ref.gguf')
    shutil.copyfile(path, reference)
    run([build / 'bin/crispasr-diff', 'dia', model, reference], label + '-canonical-diff')
    receipt[label] = dict(reference_sha256=digest(reference), canonical_reference_changed=False)
receipt['codec_export_validation_pass'] = True
receipt['scope'] = 'Standalone original critical-output F32 export; exact prior diagnostic bytes and strict startup/full-original-history replay; no native free-speech or GPU acceptance.'
(OUT / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
print('DIA_DAC_STANDALONE_EXPORT_PASS; no production-default or free-speech claim', flush=True)
