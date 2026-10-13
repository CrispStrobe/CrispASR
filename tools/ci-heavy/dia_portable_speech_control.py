#!/usr/bin/env python3
"""Validate the gated portable Dia sampler with actual neural/PCM generation.

Canonical Torch generation is preserved from its complete, recognized proof.
A distinct control replaces only _sample_next_token with the compiled actual
candidate function with unchanged native filtering. No seed selection, original-code edit or gate relaxation.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(os.environ['HEAVY_OUT'])
SCRATCH = Path(os.environ['HEAVY_SCRATCH'])
OUT.mkdir(parents=True, exist_ok=True)
SCRATCH.mkdir(parents=True, exist_ok=True)
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--child', choices=('original', 'native'))
parser.add_argument('--lib')
parser.add_argument('--model')
a = parser.parse_args()
sys.path.insert(0, str(ROOT / 'tools'))
from reference_backends import dia_tts_reference as reference

TEXT = '[S1] We are committed to making it secure.'
MODE = 'original-neural-with-gated-portable-candidate-sampler'


def digest(path):
    sha = hashlib.sha256()
    with Path(path).open('rb') as file:
        for chunk in iter(lambda: file.read(4 * 1024 * 1024), b''):
            sha.update(chunk)
    return sha.hexdigest()


def save_pcm(tag, pcm, **metadata):
    pcm = np.asarray(pcm, dtype=np.float32).reshape(-1)
    assert pcm.size and np.isfinite(pcm).all(), tag
    np.save(OUT / (tag + '.npy'), pcm)
    result = dict(phrase=TEXT[5:], seed=42, capacity=3072, audio_seconds=pcm.size / 44100,
                  rms=float(np.sqrt(np.mean(pcm.astype(np.float64) ** 2))), finite=True, **metadata)
    (OUT / (tag + '.json')).write_text(json.dumps(result, indent=2) + '\n')


if a.child in ('original', 'native'):
    os.environ['CRISPASR_DIA_PORTABLE_RNG'] = '1'

if a.child == 'original':
    import torch
    from dia_native_sampler import install_native_sampler
    torch.set_num_threads(4)
    source, _ = reference.verify_source(os.environ['DIA_SOURCE'])
    sys.path.insert(0, str(source))
    import dia.model as original_module
    os.environ.update(DIA_TEXT=TEXT, DIA_MAX_TOKENS='3072', DIA_SEED='42',
                      DIA_TEMPERATURE='1.2', DIA_TOP_P='.95', DIA_TOP_K='45', DIA_CFG_SCALE='3')
    with install_native_sampler(original_module, a.lib, seed=42) as sampler:
        captured = reference.dump('nari-labs/Dia-1.6B', None,
            stages={'decoder_input_tokens', 'decoder_step_logits', 'decoder_codes_rev', 'decoded_audio'})
        samples = np.stack(sampler.samples)
    assert captured['dia_source_pin'] == reference.SOURCE_PIN
    assert captured['dia_model_revision'] == reference.MODEL_PIN
    assert captured['dia_dac_package'] == '1.0.0'
    assert captured['dia_dac_sha256'] == 'a88eed82a7024ccc1facdb1e605c4c2f99281c8118c22c9895ffa846d8fb61aa'
    inputs = captured['decoder_input_tokens']
    logits = captured['decoder_step_logits']
    assert inputs.ndim == 2 and inputs.shape[1] == 9 and np.array_equal(inputs, inputs.astype('<i4'))
    assert logits.shape == (len(inputs), 2, 9, 1028) and samples.shape == inputs.shape
    inputs.astype('<i4').tofile(OUT / 'original-control-inputs.i32')
    logits.astype('<f4').tofile(OUT / 'original-control-logits.f32')
    np.save(OUT / 'original-control-pre-override-samples.npy', samples)
    np.save(OUT / 'original-control-codes.npy', captured['decoder_codes_rev'])
    metadata = {name: value for name, value in captured.items() if not isinstance(value, np.ndarray)}
    metadata.update(control_mode=MODE, canonical_torch_reference=False,
                    steps=len(inputs), sampler_header_sha256=digest(ROOT / 'src/dia_sampling.h'))
    (OUT / 'original-control-provenance.json').write_text(json.dumps(metadata, indent=2) + '\n')
    save_pcm('original-native-sampler', captured['decoded_audio'], scope=MODE,
             source_pin=reference.SOURCE_PIN, model_pin=reference.MODEL_PIN)
    sys.exit(0)

if a.child == 'native':
    os.environ.update(CRISPASR_DIA_DUMP_INPUT_TOKENS=str(OUT / 'native-inputs.i32'),
                      CRISPASR_DIA_DUMP_STEPLOGITS=str(OUT / 'native-logits.f32'),
                      CRISPASR_DIA_DUMP_TOKENS='1')
    # This must be free generation, independent of replay-only validation.
    for name in ('FORCE_TOKENS', 'FORCE_OUTPUT_TOKENS', 'DECODE_CODES', 'GREEDY', 'MAX_STEPS'):
        assert not any(os.environ.get(prefix + name) for prefix in ('DIA_', 'CRISPASR_DIA_')), name
    sys.path.insert(0, str(ROOT / 'python'))
    from crispasr import Session
    with Session(a.model, lib_path=a.lib, backend='dia', n_threads=4) as session:
        session.set_temperature(1.2, seed=42)
        pcm = session.synthesize(TEXT)
        assert session.output_sample_rate() == 44100
    save_pcm('native-free', pcm, scope='Actual native C ABI free generation')
    sys.exit(0)

receipt = dict(source=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
               source_worktree_dirty=bool(subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT)),
               worker_sha256=digest(Path(__file__)),
               candidate_header_sha256=digest(ROOT / 'src/dia_exponential_rng.h'),
               runtime_sha256=digest(ROOT / 'src/dia_tts.cpp'),
               control_mode=MODE, canonical_original_run=38285566623,
               production_defaults_changed=False, full_speech_acceptance=False, runs=[])


def run(command, name, allow_failure=False):
    path = OUT / (name + '.log')
    with path.open('w') as log:
        process = subprocess.run(list(map(str, command)), cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, timeout=7200)
    receipt['runs'].append(dict(name=name, exit=process.returncode, log_sha256=digest(path)))
    (OUT / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(name, process.returncode, path.read_text()[-1600:], flush=True)
    if not allow_failure:
        assert process.returncode == 0, name
    return process.returncode


os.environ.pop('CRISPASR_DIA_PORTABLE_RNG', None)

# Preserve and hash the complete existing canonical generation; no replacement.
archive = SCRATCH / 'canonical-proof'
run([sys.executable, ROOT / 'tools/dia_proof_archives.py', 'canonical-original', archive], 'canonical-download')
canonical = next(archive.glob('heavy-*'))
assert digest(canonical / 'ref.gguf') == '3db6f8923abffd612e6c83d7036debebd10ed28b77c3eb9ec3d91fec1ce1a542'
assert digest(canonical / 'original-reported.npy') == '5c4331e165afa1e5342cf4cf3f9c50e7100e3a192a4835beb8e7c66d4fa13d6d'
for suffix in ('.npy', '.json'):
    shutil.copyfile(canonical / ('original-reported' + suffix), OUT / ('canonical-original' + suffix))
canonical_result = json.loads((OUT / 'canonical-original.json').read_text())
assert canonical_result['source_pin'] == reference.SOURCE_PIN and canonical_result['model_pin'] == reference.MODEL_PIN
assert canonical_result['seed'] == 42 and canonical_result['capacity'] == 3072 and canonical_result['wer'] == 0
source = SCRATCH / 'official-dia'
run(['git', 'clone', '--filter=blob:none', 'https://github.com/nari-labs/dia', source], 'source-clone')
run(['git', '-C', source, 'checkout', '--detach', reference.SOURCE_PIN], 'source-pin')
os.environ.update(DIA_SOURCE=str(source), DIA_DEVICE='cpu')
bridge = SCRATCH / 'libdia_sampler_bridge.so'
run(['c++', '-std=c++17', '-O2', '-shared', '-fPIC', ROOT / 'tools/ci-heavy/dia_sampler_bridge.cpp', '-o', bridge],
    'compile-actual-sampler')
os.environ['DIA_SAMPLER_BRIDGE'] = str(bridge)
run([sys.executable, ROOT / 'tests/test_dia_native_sampler.py', '-v'], 'tiny-sampler-contracts')
for path in (ROOT / 'src/dia_exponential_rng.h', ROOT / 'src/dia_sampling.h', ROOT / 'tools/ci-heavy/dia_sampler_bridge.cpp', ROOT / 'tools/dia_native_sampler.py'):
    shutil.copyfile(path, OUT / path.name)
run([sys.executable, __file__, '--child', 'original', '--lib', bridge], 'original-native-sampler-generation')
from dia_dac_export import mapped_gguf
with mapped_gguf(canonical / 'ref.gguf') as reader:
    expected_codes = next(t.data for t in reader.tensors if t.name == 'decoder_codes_rev')
    receipt['original_candidate_codes_exact'] = bool(np.array_equal(expected_codes, np.load(OUT / 'original-control-codes.npy')))
(OUT / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
assert receipt['original_candidate_codes_exact'], 'Actual original neural candidate changed canonical codes'
# Original exits before fresh native loading/build; large model contexts never overlap.
build = SCRATCH / 'dia-native-build'
run(['cmake', '-S', ROOT, '-B', build, '-DCMAKE_BUILD_TYPE=Release', '-DBUILD_SHARED_LIBS=ON',
     '-DGGML_NATIVE=OFF', '-DGGML_CUDA=OFF', '-DGGML_VULKAN=OFF', '-DGGML_BLAS=OFF',
     '-DCRISPASR_BUILD_TESTS=OFF', '-DCRISPASR_BUILD_SERVER=OFF', '-DCRISPASR_OPUS=OFF', '-DCRISPASR_AMR=OFF'], 'configure')
run(['cmake', '--build', build, '--target', 'crispasr-lib', 'crispasr-cli', '-j4'], 'build')
lib = next(build.rglob('libcrispasr.so'))
receipt['native_library_sha256'] = digest(lib)
receipt['native_cli_sha256'] = digest(build / 'bin/crispasr')
from huggingface_hub import hf_hub_download
repo, revision = 'cstr/dia-1.6b-GGUF', '3233fbcb32be47761d2e736857b6d1a075b9ba7e'
model = hf_hub_download(repo, 'dia-1.6b-f16.gguf', revision=revision)
template = hf_hub_download(repo, 'dac-44khz.gguf', revision=revision)
native_dir = SCRATCH / 'native-candidate-model'
native_dir.mkdir()
native_model = native_dir / 'dia-1.6b-f16.gguf'
native_model.symlink_to(model)
codec = native_dir / 'dac-44khz.gguf'
run([sys.executable, ROOT / 'models/convert-dia-dac-to-gguf.py', '--template', template,
     '--output', codec, '--precision', 'output-f32', '--receipt', OUT / 'candidate-codec.json'], 'candidate-codec-export')
assert digest(codec) == '6db51da2333868918b6d476008d92036a53400bc816ff7e1d2b1e973499e4135'
model = native_model
run([sys.executable, __file__, '--child', 'native', '--model', model, '--lib', lib], 'native-free-generation')
import wave
os.environ['CRISPASR_DIA_PORTABLE_RNG'] = '1'
wav = OUT / 'native-cli.wav'
run([build / 'bin/crispasr', '-m', model, '--codec-model', codec, '--tts-output', wav,
     '-l', 'en', '--tts', TEXT, '-t', '4', '--seed', '42', '--temperature', '1.2',
     '--no-spoken-disclaimer', '--accept-marking-responsibility'], 'native-cli-generation')
with wave.open(str(wav), 'rb') as audio:
    assert audio.getnchannels() == 1 and audio.getsampwidth() == 2 and audio.getframerate() == 44100
    cli_pcm = np.frombuffer(audio.readframes(audio.getnframes()), dtype='<i2').astype(np.float32) / 32768.
save_pcm('native-cli', cli_pcm, scope='Actual native CLI free generation with gated portable sampler')
asr = hf_hub_download('cstr/nemotron-3.5-asr-streaming-GGUF', 'nemotron-3.5-asr-streaming-0.6b-q4_k.gguf',
                      revision='bbd95a9ca5fa0dfca3312a122dfc45a2b578b9c2')
failures = []
for tag in ('canonical-original', 'original-native-sampler', 'native-free', 'native-cli'):
    code = run([sys.executable, ROOT / 'tools/ci-heavy/dia_full_speech.py', '--child', 'recognize',
                '--model', asr, '--lib', lib, '--phrase', 'reported', '--audio-tag', tag], tag + '-asr', True)
    receipt[tag] = json.loads((OUT / (tag + '.json')).read_text())
    if code:
        failures.append(tag + '-speech')

original_inputs = np.fromfile(OUT / 'original-control-inputs.i32', dtype='<i4').reshape(-1, 9)
native_inputs = np.fromfile(OUT / 'native-inputs.i32', dtype='<i4').reshape(-1, 9)
from dia_native_sampler import native_sample_rows
native_samples = native_sample_rows((OUT / 'native-free-generation.log').read_text(), len(native_inputs))
np.save(OUT / 'native-pre-override-samples.npy', native_samples)
original_samples = np.load(OUT / 'original-control-pre-override-samples.npy')
assert original_samples.shape == original_inputs.shape
count = min(len(original_inputs), len(native_inputs))
mismatch = np.argwhere(original_inputs[:count] != native_inputs[:count])
prefix = int(mismatch[0, 0]) if len(mismatch) else count
first = None
if len(mismatch):
    step, head = map(int, mismatch[0])
    first = dict(step=step, codebook=head, original_control=int(original_inputs[step, head]), native=int(native_inputs[step, head]))
original_logits = np.fromfile(OUT / 'original-control-logits.f32', dtype='<f4').reshape(len(original_inputs), 2, 9, 1028)
native_logits = np.fromfile(OUT / 'native-logits.f32', dtype='<f4').reshape(len(native_inputs), 2, 9, 1028)
assert prefix > 0, 'No shared original/native input prefix'
ref, actual = original_logits[:prefix].astype(np.float64), native_logits[:prefix].astype(np.float64)
norms_ref, norms_native = np.linalg.norm(ref, axis=-1), np.linalg.norm(actual, axis=-1)
assert np.isfinite(ref).all() and np.isfinite(actual).all() and (norms_ref > 0).all() and (norms_native > 0).all()
cos = np.sum(ref * actual, axis=-1) / (norms_ref * norms_native)
ratio = norms_native / norms_ref
# Preserve all aligned-prefix per-step/codebook magnitudes and cosines.
np.save(OUT / 'aligned-prefix-cosine.npy', cos)
np.save(OUT / 'aligned-prefix-norm-ratio.npy', ratio)
prefix_pass = bool(cos.min() >= .9995 and np.abs(ratio - 1).max() < .02)
receipt['paired_generation'] = dict(original_control_steps=len(original_inputs), native_steps=len(native_inputs),
    aligned_input_prefix_steps=prefix, first_input_divergence=first, minimum_prefix_cosine=float(cos.min()),
    maximum_prefix_norm_error=float(np.abs(ratio - 1).max()), aligned_prefix_neural_pass=prefix_pass,
    complete_input_histories_equal=bool(len(original_inputs) == len(native_inputs) and not len(mismatch)))
sample_mismatch = np.argwhere(original_samples[:count] != native_samples[:count])
first_sample = None
if len(sample_mismatch):
    step, head = map(int, sample_mismatch[0])
    first_sample = dict(step=step, codebook=head, original_control=int(original_samples[step, head]),
                        native=int(native_samples[step, head]), shared_input_history=bool(step < prefix))
receipt['paired_generation'].update(first_pre_override_sample_divergence=first_sample,
    complete_pre_override_samples_equal=bool(len(original_samples) == len(native_samples) and not len(sample_mismatch)))
if not prefix_pass:
    failures.append('aligned-prefix-neural')
receipt['failures'] = failures
receipt['sampler_diagnostic_pass'] = not failures
receipt['scope'] = 'Single fixed short prompt: canonical Torch, original with actual native sampler, and actual native C ABI and CLI free generation with the gated portable sampler; not complete model or GPU acceptance.'
(OUT / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
assert not failures, failures
print('DIA_PORTABLE_SAMPLER_CONTROL_PASS; no production-default or full-model acceptance claim', flush=True)
