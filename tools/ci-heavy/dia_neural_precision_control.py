#!/usr/bin/env python3
"""Fixed original-heads/F32 precision controls for the rejected short Dia speech.

No seed selection, sampling-policy change, forced native generation or model-pin
promotion. Sequential original-coefficient exports and real stage/speech checks
run only on hosted hardware; each failed arm remains a failed control.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import wave

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(os.environ['HEAVY_OUT'])
SCRATCH = Path(os.environ['HEAVY_SCRATCH'])
OUT.mkdir(parents=True, exist_ok=True)
SCRATCH.mkdir(parents=True, exist_ok=True)
os.environ['TMPDIR'] = str(SCRATCH)
sys.path.insert(0, str(ROOT / 'tools'))
from dia_dac_export import digest
from dia_model_precision import export_precision, WEIGHTS_SHA
from reference_backends import dia_tts_reference as reference

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--export', choices=('heads-f32', 'f32'))
parser.add_argument('--weights', type=Path)
parser.add_argument('--template', type=Path)
parser.add_argument('--output', type=Path)
a = parser.parse_args()
if a.export:
    from safetensors import safe_open
    assert digest(a.weights) == WEIGHTS_SHA, 'Original neural checkpoint changed'
    with safe_open(str(a.weights), framework='pt', device='cpu') as weights:
        audit = export_precision(weights.keys(), lambda key: weights.get_tensor(key).float().numpy(),
                                 a.template, a.output, a.export, checkpoint_sha=digest(a.weights))
    assert len(audit['tensors']) == 381
    assert sum(row['selected'] for row in audit['tensors']) == (9 if a.export == 'heads-f32' else 381)
    if a.export == 'heads-f32':
        assert audit['byte_delta'] == 37896192, 'Unexpected critical-head size delta'
    (OUT / ('neural-' + a.export + '-physical-audit.json')).write_text(json.dumps(audit, indent=2) + '\n')
    print(json.dumps({key: value for key, value in audit.items() if key != 'tensors'}), flush=True)
    sys.exit(0)

report = dict(source=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
              source_worktree_dirty=bool(subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT)),
              worker_sha256=digest(Path(__file__)), export_helper_sha256=digest(ROOT / 'tools/dia_model_precision.py'),
              production_defaults_changed=False, production_model_pins_changed=False,
              full_model_acceptance=False, gpu_acceptance=False, q4_acceptance=False, runs=[], arms=[])


def checkpoint():
    (OUT / 'receipt.json').write_text(json.dumps(report, indent=2) + '\n')


def run(command, name, env=None, allowed_failure=False):
    log = OUT / (name + '.log')
    with log.open('w') as file:
        result = subprocess.run(list(map(str, command)), cwd=ROOT, env=env,
                                stdout=file, stderr=subprocess.STDOUT, timeout=7200)
    report['runs'].append(dict(name=name, exit=result.returncode, log_sha256=digest(log)))
    checkpoint()
    print(name, result.returncode, log.read_text()[-1200:], flush=True)
    if not allowed_failure:
        assert result.returncode == 0, name
    return result.returncode


canonical_dir = SCRATCH / 'canonical-proof'
rejected_dir = SCRATCH / 'rejected-proof'
run(['gh', 'run', 'download', '38285566623', '-R', 'CrispStrobe/CrispASR', '-D', canonical_dir], 'canonical-download')
run(['gh', 'run', 'download', '38317327337', '-R', 'CrispStrobe/CrispASR', '-D', rejected_dir], 'rejected-download')
canonical, rejected = next(canonical_dir.glob('heavy-*')), next(rejected_dir.glob('heavy-*'))
ref = canonical / 'ref.gguf'
assert digest(ref) == '3db6f8923abffd612e6c83d7036debebd10ed28b77c3eb9ec3d91fec1ce1a542'
old = json.loads((rejected / 'receipt.json').read_text())
assert old['source'] == '3ba3a317ab5d2996adfde1efef2c23811fc9e049'
assert old['original_candidate_codes_exact'] and old['original-native-sampler']['wer'] == 0
assert old['native-free']['wer'] == old['native-cli']['wer'] == 1
report['rejected_control'] = old
shutil.copyfile(ref, OUT / 'canonical-ref.gguf')
source = SCRATCH / 'official-dia'
run(['git', 'clone', '--filter=blob:none', 'https://github.com/nari-labs/dia', source], 'source-clone')
run(['git', '-C', source, 'checkout', '--detach', reference.SOURCE_PIN], 'source-pin')
reference.verify_source(source)
os.environ.update(DIA_SOURCE=str(source), DIA_DEVICE='cpu')
run([sys.executable, ROOT / 'tests/test_dia_model_precision.py', '-v'], 'tiny-neural-export-contracts')
from huggingface_hub import hf_hub_download, snapshot_download
location = Path(snapshot_download('nari-labs/Dia-1.6B', revision=reference.MODEL_PIN,
                                  allow_patterns=['config.json', 'model.safetensors']))
weights = location / 'model.safetensors'
assert digest(weights) == WEIGHTS_SHA
assert digest(location / 'config.json') == '9140e85fd15b82d7f268c5681a41fb1de68573ec206b87cb3a5862fe5c598d6e'
repo, revision = 'cstr/dia-1.6b-GGUF', '3233fbcb32be47761d2e736857b6d1a075b9ba7e'
template = hf_hub_download(repo, 'dia-1.6b-f16.gguf', revision=revision)
codec_template = hf_hub_download(repo, 'dac-44khz.gguf', revision=revision)
codec = SCRATCH / 'dac-output-f32.gguf'
run([sys.executable, ROOT / 'models/convert-dia-dac-to-gguf.py', '--template', codec_template,
     '--output', codec, '--precision', 'output-f32', '--receipt', OUT / 'physical-codec-audit.json'], 'critical-codec-export')
assert digest(codec) == '6db51da2333868918b6d476008d92036a53400bc816ff7e1d2b1e973499e4135'
report.update(template_sha256=digest(template), original_weights_sha256=WEIGHTS_SHA,
              codec_sha256=digest(codec), canonical_reference_sha256=digest(ref))
checkpoint()
build = SCRATCH / 'native-build'
run(['cmake', '-S', ROOT, '-B', build, '-DCMAKE_BUILD_TYPE=Release', '-DBUILD_SHARED_LIBS=ON',
     '-DGGML_NATIVE=OFF', '-DGGML_CUDA=OFF', '-DGGML_VULKAN=OFF', '-DGGML_BLAS=OFF',
     '-DCRISPASR_BUILD_TESTS=OFF', '-DCRISPASR_BUILD_SERVER=OFF', '-DCRISPASR_OPUS=OFF', '-DCRISPASR_AMR=OFF'], 'configure')
run(['cmake', '--build', build, '--target', 'crispasr-lib', 'crispasr-cli', 'crispasr-diff', '-j4'], 'build')
lib, cli, diff = next(build.rglob('libcrispasr.so')), build / 'bin/crispasr', build / 'bin/crispasr-diff'
report.update(native_library_sha256=digest(lib), native_cli_sha256=digest(cli), native_diff_sha256=digest(diff))
asr = hf_hub_download('cstr/nemotron-3.5-asr-streaming-GGUF', 'nemotron-3.5-asr-streaming-0.6b-q4_k.gguf',
                      revision='bbd95a9ca5fa0dfca3312a122dfc45a2b578b9c2')
for precision in ('heads-f32', 'f32'):
    model_dir = SCRATCH / precision
    model_dir.mkdir()
    model = model_dir / ('dia-1.6b-' + precision + '.gguf')
    (model_dir / 'dac-44khz.gguf').symlink_to(codec)
    # Each exporter child releases its mappings/Torch before native model loading.
    run([sys.executable, __file__, '--export', precision, '--weights', weights,
         '--template', template, '--output', model], precision + '-export')
    audit = json.loads((OUT / ('neural-' + precision + '-physical-audit.json')).read_text())
    assert digest(model) == audit['export_sha256']
    arm_out = OUT / precision
    arm_out.mkdir()
    env = os.environ.copy()
    env.update(HEAVY_OUT=str(arm_out), CRISPASR_DIA_PORTABLE_RNG='1', DIA_CODEC=str(codec))
    for prefix in ('DIA_', 'CRISPASR_DIA_'):
        for suffix in ('FORCE_TOKENS', 'FORCE_OUTPUT_TOKENS', 'DECODE_CODES', 'GREEDY', 'MAX_STEPS'):
            assert not env.get(prefix + suffix), 'Forced free-generation input rejected'
    arm = dict(precision=precision, model_sha256=audit['export_sha256'], model_bytes=model.stat().st_size,
               byte_delta=audit['byte_delta'], strict_diff_exit=run([diff, 'dia', model, ref], precision + '-strict-diff', env, True))
    report['arms'].append(arm)
    checkpoint()
    run([sys.executable, ROOT / 'tools/ci-heavy/dia_portable_speech_control.py', '--child', 'native',
         '--model', model, '--lib', lib], precision + '-native-generation', env)
    arm['native_asr_exit'] = run([sys.executable, ROOT / 'tools/ci-heavy/dia_full_speech.py', '--child', 'recognize',
         '--model', asr, '--lib', lib, '--phrase', 'reported', '--audio-tag', 'native-free'], precision + '-native-asr', env, True)
    arm['native'] = json.loads((arm_out / 'native-free.json').read_text())
    checkpoint()
    wav = arm_out / 'native-cli.wav'
    run([cli, '--no-gpu', '-m', model, '--codec-model', codec, '--tts-output', wav, '-l', 'en',
         '--tts', '[S1] We are committed to making it secure.', '-t', '4', '--seed', '42', '--temperature', '1.2',
         '--no-spoken-disclaimer', '--accept-marking-responsibility'], precision + '-cli-generation', env)
    with wave.open(str(wav), 'rb') as audio:
        assert audio.getnchannels() == 1 and audio.getsampwidth() == 2 and audio.getframerate() == 44100
        pcm = np.frombuffer(audio.readframes(audio.getnframes()), dtype='<i2').astype(np.float32) / 32768.
    np.save(arm_out / 'native-cli.npy', pcm)
    (arm_out / 'native-cli.json').write_text(json.dumps(dict(phrase='We are committed to making it secure.',
        seed=42, capacity=3072, audio_seconds=len(pcm)/44100, rms=float(np.sqrt(np.mean(pcm.astype('f8')**2))),
        finite=bool(np.isfinite(pcm).all()), scope='Actual native CLI free generation in fixed neural precision control')))
    arm['cli_asr_exit'] = run([sys.executable, ROOT / 'tools/ci-heavy/dia_full_speech.py', '--child', 'recognize',
         '--model', asr, '--lib', lib, '--phrase', 'reported', '--audio-tag', 'native-cli'], precision + '-cli-asr', env, True)
    arm['cli'] = json.loads((arm_out / 'native-cli.json').read_text())
    arm['fixed_prompt_pass'] = arm['strict_diff_exit'] == arm['native_asr_exit'] == arm['cli_asr_exit'] == 0
    checkpoint()
    # The audited diagnostic is reproducible from immutable original weights;
    # do not retain multiple large native model controls in hosted scratch.
    model.unlink()
report['precision_controls_pass'] = all(arm['fixed_prompt_pass'] for arm in report['arms'])
report['scope'] = 'Two fixed CPU precision controls with physical original tensor audits, strict canonical stages and actual native CLI/C-ABI ASR; no full-model/GPU/Q4 acceptance or model-pin promotion.'
checkpoint()
assert report['precision_controls_pass'], 'One or more fixed precision controls failed; preserve all results'
print('DIA_NEURAL_PRECISION_CONTROL_PASS; fixed-prompt CPU scope only', flush=True)
