#!/usr/bin/env python3
"""Pinned original/native decoder-layer replay at the retained step-978 boundary.

Outputs are diagnostic teacher replay, never accepted native free speech.
Default/late/default capture must leave complete native logits/inputs/PCM exact.
"""
import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(os.environ['HEAVY_OUT'])
SCRATCH = Path(os.environ['HEAVY_SCRATCH'])
OUT.mkdir(parents=True, exist_ok=True)
SCRATCH.mkdir(parents=True, exist_ok=True)
os.environ['TMPDIR'] = str(SCRATCH)
sys.path.insert(0, str(ROOT / 'tools'))
from dia_dac_export import digest
from reference_backends import dia_tts_reference as reference

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--original-child', action='store_true')
parser.add_argument('--canonical', type=Path)
a = parser.parse_args()
STEP = 978
TEXT = '[S1] We are committed to making it secure.'

if a.original_child:
    import torch
    from gguf import GGUFReader
    import dump_reference
    torch.set_num_threads(4)
    assert torch.__version__ == '2.6.0+cpu'
    os.environ.update(DIA_TEXT=TEXT, DIA_MAX_TOKENS='3072', DIA_SEED='42', DIA_TEMPERATURE='1.2',
                      DIA_TOP_P='.95', DIA_TOP_K='45', DIA_CFG_SCALE='3', DIA_CAPTURE_STEP=str(STEP))
    stages = set(reference.BOUNDARY_STAGES)
    stages.update(f'encoder_layer_{i}' for i in range(12))
    stages.update(f'decoder_layer_{i}' for i in range(18))
    result = reference.dump('nari-labs/Dia-1.6B', None, stages=stages)
    original = {t.name: np.array(t.data).reshape(tuple(reversed(t.shape))) for t in GGUFReader(str(a.canonical)).tensors}
    for name in ('decoder_input_tokens', 'decoder_codes_rev'):
        np.testing.assert_array_equal(result[name], original[name])
    assert result['dia_source_pin'] == reference.SOURCE_PIN and result['dia_model_revision'] == reference.MODEL_PIN
    assert result['dia_torch_version'] == '2.6.0+cpu' and result['dia_decoder_capture_step'] == str(STEP)
    assert result['dia_dac_sha256'] == 'a88eed82a7024ccc1facdb1e605c4c2f99281c8118c22c9895ffa846d8fb61aa'
    arrays = {k: v for k, v in result.items() if isinstance(v, np.ndarray)}
    metadata = {k: v for k, v in result.items() if not isinstance(v, np.ndarray)}
    dump_reference.write_gguf_archive(arrays, metadata, OUT / 'original-step978-ref.gguf')
    scaled = dict(arrays, decoder_hidden=arrays['decoder_hidden'] * 2)
    dump_reference.write_gguf_archive(scaled, metadata, OUT / 'scaled-step978-ref.gguf')
    for label, bad in [('negative', '-1'), ('out-of-history', str(len(arrays['decoder_input_tokens'])))]:
        dump_reference.write_gguf_archive(arrays, dict(metadata, dia_decoder_capture_step=bad), OUT / (label + '-step-ref.gguf'))
    np.save(OUT / 'original-late.npy', arrays['decoded_audio'])
    pcm = arrays['decoded_audio'].astype('f8')
    (OUT / 'original-late.json').write_text(json.dumps(dict(phrase=TEXT[5:], seed=42, capacity=3072,
        audio_seconds=len(pcm)/44100, rms=float(np.sqrt(np.mean(pcm**2))), finite=bool(np.isfinite(pcm).all()),
        scope='Actual pinned original generation with step-978 observation; canonical input/code grids exact')) + '\n')
    (OUT / 'original-late-provenance.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print('ORIGINAL_LATE_CAPTURE_CODES_INPUTS_EXACT; layer snapshot is diagnostic', flush=True)
    sys.exit(0)

report = dict(source=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
    worktree_dirty=bool(subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT)),
    worker_sha256=digest(Path(__file__)), step=STEP, runs=[], production_defaults_changed=False,
    speech_acceptance=False, gpu_acceptance=False, q4_acceptance=False,
    scope='Original/native layer and complete-logit teacher replay with default/late/default capture controls')
for name in ('src/dia_tts.cpp', 'src/dia_dump_step.h', 'examples/cli/crispasr_dia_diff.h',
             'tools/reference_backends/dia_tts_reference.py', 'tools/ci-heavy/dia_late_stage_control.py'):
    target = OUT / 'sources' / name
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ROOT / name, target)


def checkpoint():
    (OUT / 'receipt.json').write_text(json.dumps(report, indent=2) + '\n')


def run(command, label, env=None, negative=False):
    log = OUT / (label + '.log')
    with log.open('w') as file:
        p = subprocess.run(list(map(str, command)), cwd=ROOT, env=env, stdout=file,
                           stderr=subprocess.STDOUT, timeout=10800)
    report['runs'].append(dict(name=label, exit=p.returncode, log_sha256=digest(log)))
    checkpoint()
    print(label, p.returncode, log.read_text()[-1500:], flush=True)
    assert (p.returncode != 0) == negative, label


proof = SCRATCH / 'canonical-proof'
run([sys.executable, ROOT / 'tools/dia_proof_archives.py', 'canonical-original', proof], 'canonical-download')
canonical = next(proof.glob('heavy-*')) / 'ref.gguf'
assert digest(canonical) == '3db6f8923abffd612e6c83d7036debebd10ed28b77c3eb9ec3d91fec1ce1a542'
report['canonical_reference_sha256'] = digest(canonical)
shutil.copy2(canonical, OUT / 'canonical-ref.gguf')
source = SCRATCH / 'official-dia'
run(['git', 'clone', '--filter=blob:none', 'https://github.com/nari-labs/dia', source], 'original-source-clone')
run(['git', '-C', source, 'checkout', '--detach', reference.SOURCE_PIN], 'original-source-pin')
reference.verify_source(source)
os.environ.update(DIA_SOURCE=str(source), DIA_DEVICE='cpu')
run([sys.executable, ROOT / 'tests/test_dia_dump_step.py', '-v'], 'capture-index-contracts')
run([sys.executable, ROOT / 'tests/test_dia_reference_capture.py', '-v'], 'original-hook-contracts')
run([sys.executable, __file__, '--original-child', '--canonical', canonical], 'original-late-capture')
from huggingface_hub import hf_hub_download, snapshot_download
from dia_model_precision import WEIGHTS_SHA
location = Path(snapshot_download('nari-labs/Dia-1.6B', revision=reference.MODEL_PIN,
                                allow_patterns=['config.json', 'model.safetensors']))
weights = location / 'model.safetensors'
assert digest(weights) == WEIGHTS_SHA
repo, revision = 'cstr/dia-1.6b-GGUF', '3233fbcb32be47761d2e736857b6d1a075b9ba7e'
template = hf_hub_download(repo, 'dia-1.6b-f16.gguf', revision=revision)
codec_template = hf_hub_download(repo, 'dac-44khz.gguf', revision=revision)
model_dir = SCRATCH / 'native-model'; model_dir.mkdir()
model = model_dir / 'dia-1.6b-f32.gguf'; codec = model_dir / 'dac-44khz.gguf'
run([sys.executable, ROOT / 'tools/ci-heavy/dia_neural_precision_control.py', '--export', 'f32',
     '--weights', weights, '--template', template, '--output', model], 'neural-export')
assert digest(model) == '8128132c538e1aef83f2fb5257b4f71d28a67c2a5a0d54d0a1f0a3b0f5d6b73d'
run([sys.executable, ROOT / 'models/convert-dia-dac-to-gguf.py', '--template', codec_template,
     '--output', codec, '--precision', 'output-f32', '--receipt', OUT / 'physical-codec-audit.json'], 'codec-export')
assert digest(codec) == '6db51da2333868918b6d476008d92036a53400bc816ff7e1d2b1e973499e4135'
run(['sudo', 'apt-get', 'update', '-q'], 'build-dependencies-update')
run(['sudo', 'apt-get', 'install', '-y', 'ninja-build', 'ccache'], 'build-dependencies-install')
build = SCRATCH / 'native-build'
run(['cmake', '-G', 'Ninja', '-S', ROOT, '-B', build, '-DCMAKE_BUILD_TYPE=Release', '-DBUILD_SHARED_LIBS=ON',
     '-DCMAKE_C_COMPILER_LAUNCHER=ccache', '-DCMAKE_CXX_COMPILER_LAUNCHER=ccache',
     '-DGGML_NATIVE=OFF', '-DGGML_CUDA=OFF', '-DGGML_VULKAN=OFF', '-DGGML_BLAS=OFF',
     '-DCRISPASR_BUILD_TESTS=OFF', '-DCRISPASR_BUILD_SERVER=OFF', '-DCRISPASR_OPUS=OFF', '-DCRISPASR_AMR=OFF'], 'configure')
run(['cmake', '--build', build, '--target', 'crispasr-lib', 'crispasr-diff', '-j4'], 'build')
lib = next(build.rglob('libcrispasr.so')); diff = build / 'bin/crispasr-diff'
report.update(library_sha256=digest(lib), diff_sha256=digest(diff), model_sha256=digest(model), codec_sha256=digest(codec))
asr = hf_hub_download('cstr/nemotron-3.5-asr-streaming-GGUF', 'nemotron-3.5-asr-streaming-0.6b-q4_k.gguf',
                     revision='bbd95a9ca5fa0dfca3312a122dfc45a2b578b9c2')
run([sys.executable, ROOT / 'tools/ci-heavy/dia_full_speech.py', '--child', 'recognize', '--model', asr,
     '--lib', lib, '--phrase', 'reported', '--audio-tag', 'original-late'], 'original-asr')
base_env = os.environ.copy(); base_env.update(CRISPASR_DIA_CROSS_CACHE_F32='1', CRISPASR_DIA_PORTABLE_RNG='1',
    CRISPASR_DIA_TTS_GPU='0', DIA_CODEC=str(codec))
for label, ref in [('default', canonical), ('late', OUT / 'original-step978-ref.gguf'), ('repeat-default', canonical)]:
    env = dict(base_env, CRISPASR_DIA_DIFF_DUMP_DIR=str(OUT / label), CRISPASR_DIA_DUMP_STEP='malformed-inherited-control')
    run([diff, 'dia', model, ref], label + '-diff', env)
    log = (OUT / (label + '-diff.log')).read_text()
    assert len(re.findall(r'\[PASS\]', log)) == 40 and '[FAIL]' not in log
    assert (OUT / label / 'capture-step.txt').read_text().strip() == ('978' if label == 'late' else '0')
for name in ('logits.f32', 'feedback.i32', 'decoded_audio.f32'):
    hashes = [digest(OUT / label / name) for label in ('default', 'late', 'repeat-default')]
    assert len(set(hashes)) == 1, (name, hashes)
report['native_capture_off_on_off_complete_arrays_exact'] = True
for label in ('scaled-step978', 'negative-step', 'out-of-history-step'):
    run([diff, 'dia', model, OUT / (label + '-ref.gguf')], label + '-negative', base_env, negative=True)
report['late_layer_diagnostic_pass'] = True
checkpoint()
print('DIA_LATE_LAYER_REPLAY_CONTROL_PASS; no native free-speech or GPU acceptance', flush=True)
