#!/usr/bin/env python3
"""Fixed original-heads/F32 precision controls for the rejected short Dia speech.

No seed selection, sampling-policy change, forced native generation or model-pin
promotion. Sequential original-coefficient exports and real stage/speech checks
run only on hosted hardware; each failed arm remains a failed control.
"""
import argparse
import json
import os
import re
from pathlib import Path
import shutil
import subprocess
import sys
import time
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
parser.add_argument('--cross-cache-control', action='store_true',
                    help='Fixed F32 neural weights, F16/F32/F16-repeat cross-cache isolation')
parser.add_argument('--cuda-control', action='store_true', help='Actual CUDA cache triage using a prebuilt native runtime')
parser.add_argument('--prebuilt', type=Path, help='Existing native build directory; required for CUDA triage')
parser.add_argument('--export', choices=('heads-f32', 'f32'))
parser.add_argument('--weights', type=Path)
parser.add_argument('--template', type=Path)
parser.add_argument('--output', type=Path)
a = parser.parse_args()
if a.cuda_control and (not a.cross_cache_control or not a.prebuilt):
    parser.error('CUDA triage requires --cross-cache-control and --prebuilt')
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
              full_model_acceptance=False, gpu_acceptance=False, q4_acceptance=False,
              cross_cache_control=a.cross_cache_control, cuda_control=a.cuda_control, runs=[], arms=[])
for path in ('src/dia_tts.cpp', 'tools/ci-heavy/dia_neural_precision_control.py',
             'tools/dia_model_precision.py'):
    shutil.copyfile(ROOT / path, OUT / Path(path).name)


def checkpoint():
    (OUT / 'receipt.json').write_text(json.dumps(report, indent=2) + '\n')


def run(command, name, env=None, allowed_failure=False):
    log = OUT / (name + '.log')
    needs_cuda = a.cuda_control and name.endswith(('-strict-diff', '-native-generation', '-cli-generation'))
    peak = 0
    with log.open('w') as file:
        if needs_cuda:
            result = subprocess.Popen(list(map(str, command)), cwd=ROOT, env=env,
                                      stdout=file, stderr=subprocess.STDOUT)
            deadline = time.monotonic() + 7200
            while result.poll() is None:
                rows = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,used_gpu_memory',
                                                '--format=csv,noheader,nounits'], text=True, timeout=30)
                usage = [int(row.split(',')[1].strip()) for row in rows.splitlines()
                         if row.strip() and int(row.split(',')[0].strip()) == result.pid]
                peak = max([peak] + usage)
                if time.monotonic() >= deadline:
                    result.kill(); result.wait()
                    raise TimeoutError(name)
                time.sleep(.5)
        else:
            result = subprocess.run(list(map(str, command)), cwd=ROOT, env=env,
                                    stdout=file, stderr=subprocess.STDOUT, timeout=7200)
    record = dict(name=name, exit=result.returncode, log_sha256=digest(log))
    if needs_cuda:
        record.update(gpu_peak_mib_lower_bound=peak,
                      cuda_scheduler_splits=len(re.findall(r'SPLIT #\d+: CUDA', log.read_text())))
    report['runs'].append(record)
    checkpoint()
    print(name, result.returncode, log.read_text()[-1200:], flush=True)
    if needs_cuda:
        assert peak >= 6000 and record['cuda_scheduler_splits'] > 0, 'Native full-F32 CUDA allocation/compute not proven'
    if not allowed_failure:
        assert result.returncode == 0, name
    return result.returncode


canonical_dir = SCRATCH / 'canonical-proof'
rejected_dir = SCRATCH / 'rejected-proof'
run([sys.executable, ROOT / 'tools/dia_proof_archives.py', 'canonical-original', canonical_dir], 'canonical-download')
run([sys.executable, ROOT / 'tools/dia_proof_archives.py', 'native-speech-rejection', rejected_dir], 'rejected-download')
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
if a.cuda_control:
    # Reuse the byte-verified actual conversion; Kaggle never reinstalls Torch/DAC.
    codec_dir = SCRATCH / 'verified-codec-proof'
    run([sys.executable, ROOT / 'tools/dia_proof_archives.py', 'dac-converter', codec_dir], 'verified-codec-download')
    codec_proof = next(codec_dir.glob('heavy-*'))
    codec = codec_proof / 'dac-output-f32.gguf'
    shutil.copyfile(codec_proof / 'physical-tensor-audit.json', OUT / 'physical-codec-audit.json')
else:
    codec_template = hf_hub_download(repo, 'dac-44khz.gguf', revision=revision)
    codec = SCRATCH / 'dac-output-f32.gguf'
    run([sys.executable, ROOT / 'models/convert-dia-dac-to-gguf.py', '--template', codec_template,
         '--output', codec, '--precision', 'output-f32', '--receipt', OUT / 'physical-codec-audit.json'], 'critical-codec-export')
assert digest(codec) == '6db51da2333868918b6d476008d92036a53400bc816ff7e1d2b1e973499e4135'
report.update(template_sha256=digest(template), original_weights_sha256=WEIGHTS_SHA,
              codec_sha256=digest(codec), canonical_reference_sha256=digest(ref))
checkpoint()
if a.prebuilt:
    build = a.prebuilt.resolve()
    assert build.is_dir(), 'Prebuilt native runtime missing'
else:
    if not shutil.which('ninja') or not shutil.which('ccache'):
        run(['sudo', 'apt-get', 'update', '-q'], 'build-dependencies-update')
        run(['sudo', 'apt-get', 'install', '-y', 'ninja-build', 'ccache'], 'build-dependencies-install')
    build = SCRATCH / 'native-build'
    run(['cmake', '-G', 'Ninja', '-S', ROOT, '-B', build,
         '-DCMAKE_C_COMPILER_LAUNCHER=ccache', '-DCMAKE_CXX_COMPILER_LAUNCHER=ccache', '-DCMAKE_BUILD_TYPE=Release', '-DBUILD_SHARED_LIBS=ON',
         '-DGGML_NATIVE=OFF', '-DGGML_CUDA=OFF', '-DGGML_VULKAN=OFF', '-DGGML_BLAS=OFF',
         '-DCRISPASR_BUILD_TESTS=OFF', '-DCRISPASR_BUILD_SERVER=OFF', '-DCRISPASR_OPUS=OFF', '-DCRISPASR_AMR=OFF'], 'configure')
    run(['cmake', '--build', build, '--target', 'crispasr-lib', 'crispasr-cli', 'crispasr-diff', '-j4'], 'build')
lib, cli, diff = next(build.rglob('libcrispasr.so')), build / 'bin/crispasr', build / 'bin/crispasr-diff'
report.update(native_library_sha256=digest(lib), native_cli_sha256=digest(cli), native_diff_sha256=digest(diff))
asr = hf_hub_download('cstr/nemotron-3.5-asr-streaming-GGUF', 'nemotron-3.5-asr-streaming-0.6b-q4_k.gguf',
                      revision='bbd95a9ca5fa0dfca3312a122dfc45a2b578b9c2')
for precision in (('f32',) if a.cross_cache_control else ('heads-f32', 'f32')):
    model_dir = SCRATCH / precision
    model_dir.mkdir()
    model = model_dir / ('dia-1.6b-' + precision + '.gguf')
    (model_dir / 'dac-44khz.gguf').symlink_to(codec)
    # Each exporter child releases its mappings/Torch before native model loading.
    run([sys.executable, __file__, '--export', precision, '--weights', weights,
         '--template', template, '--output', model], precision + '-export')
    audit = json.loads((OUT / ('neural-' + precision + '-physical-audit.json')).read_text())
    assert digest(model) == audit['export_sha256']
    if a.cross_cache_control:
        assert audit['export_sha256'] == '8128132c538e1aef83f2fb5257b4f71d28a67c2a5a0d54d0a1f0a3b0f5d6b73d', 'Fixed original F32 model changed'
    for cache in (('f16', 'f32', 'f16-repeat') if a.cross_cache_control else ('f16',)):
        label = precision + '-cross-' + cache if a.cross_cache_control else precision
        arm_out = OUT / label
        arm_out.mkdir()
        env = os.environ.copy()
        env.update(HEAVY_OUT=str(arm_out), CRISPASR_DIA_PORTABLE_RNG='1', DIA_CODEC=str(codec))
        env.pop('CRISPASR_DIA_CROSS_CACHE_F32', None)
        env['CRISPASR_DIA_TTS_GPU'] = '1' if a.cuda_control else '0'
        if a.cuda_control:
            env['GGML_SCHED_DEBUG'] = '1'
        if cache == 'f32':
            env['CRISPASR_DIA_CROSS_CACHE_F32'] = '1'
        for prefix in ('DIA_', 'CRISPASR_DIA_'):
            for suffix in ('FORCE_TOKENS', 'FORCE_OUTPUT_TOKENS', 'DECODE_CODES', 'GREEDY', 'MAX_STEPS'):
                assert not env.get(prefix + suffix), 'Forced free-generation input rejected'
        arm = dict(precision=precision, cross_cache_precision='f32' if cache == 'f32' else 'f16', model_sha256=audit['export_sha256'], model_bytes=model.stat().st_size,
                   byte_delta=audit['byte_delta'], strict_diff_exit=run([diff, 'dia', model, ref], label + '-strict-diff', env, True))
        diff_log = (OUT / (label + '-strict-diff.log')).read_text()
        arm['strict_stage_passes'] = len(re.findall(r'\[PASS\]', diff_log))
        if arm['strict_diff_exit'] == 0:
            assert arm['strict_stage_passes'] == 40 and '[FAIL]' not in diff_log, 'Incomplete canonical diff'
        report['arms'].append(arm)
        checkpoint()
        run([sys.executable, ROOT / 'tools/ci-heavy/dia_portable_speech_control.py', '--child', 'native',
             '--model', model, '--lib', lib], label + '-native-generation', env)
        arm['native_asr_exit'] = run([sys.executable, ROOT / 'tools/ci-heavy/dia_full_speech.py', '--child', 'recognize',
             '--model', asr, '--lib', lib, '--phrase', 'reported', '--audio-tag', 'native-free'], label + '-native-asr', env, True)
        arm['native'] = json.loads((arm_out / 'native-free.json').read_text())
        matches = re.findall(r'DIA_CROSS_CACHE: type=(F16|F32) bytes=(\d+) encoder_steps=(\d+) layers=(\d+)',
                             (OUT / (label + '-native-generation.log')).read_text())
        assert len(matches) == 1, 'Missing actual cross-cache allocation receipt'
        kind, size, steps, layers = matches[0]
        assert kind == ('F32' if cache == 'f32' else 'F16')
        arm['cross_cache'] = dict(type=kind, bytes=int(size), encoder_steps=int(steps), layers=int(layers))
        checkpoint()
        wav = arm_out / 'native-cli.wav'
        run([cli, *([] if a.cuda_control else ['--no-gpu']), '-m', model, '--codec-model', codec, '--tts-output', wav, '-l', 'en',
             '--tts', '[S1] We are committed to making it secure.', '-t', '4', '--seed', '42', '--temperature', '1.2',
             '--no-spoken-disclaimer', '--accept-marking-responsibility'], label + '-cli-generation', env)
        with wave.open(str(wav), 'rb') as audio:
            assert audio.getnchannels() == 1 and audio.getsampwidth() == 2 and audio.getframerate() == 44100
            pcm = np.frombuffer(audio.readframes(audio.getnframes()), dtype='<i2').astype(np.float32) / 32768.
        np.save(arm_out / 'native-cli.npy', pcm)
        (arm_out / 'native-cli.json').write_text(json.dumps(dict(phrase='We are committed to making it secure.',
            seed=42, capacity=3072, audio_seconds=len(pcm)/44100, rms=float(np.sqrt(np.mean(pcm.astype('f8')**2))),
            finite=bool(np.isfinite(pcm).all()), scope='Actual native CLI free generation in fixed neural precision control')))
        arm['cli_asr_exit'] = run([sys.executable, ROOT / 'tools/ci-heavy/dia_full_speech.py', '--child', 'recognize',
             '--model', asr, '--lib', lib, '--phrase', 'reported', '--audio-tag', 'native-cli'], label + '-cli-asr', env, True)
        arm['cli'] = json.loads((arm_out / 'native-cli.json').read_text())
        arm['fixed_prompt_pass'] = arm['strict_diff_exit'] == arm['native_asr_exit'] == arm['cli_asr_exit'] == 0
        checkpoint()
    # The audited diagnostic is reproducible from immutable original weights;
    # do not retain multiple large native model controls in hosted scratch.
    model.unlink()
if a.cross_cache_control:
    off, on, repeat = report['arms']
    first, last = OUT / 'f32-cross-f16', OUT / 'f32-cross-f16-repeat'
    repeated = ('native-inputs.i32', 'native-logits.f32', 'native-free.npy', 'native-cli.npy')
    report['cache_off_repeat_exact'] = all(digest(first / name) == digest(last / name) for name in repeated)
    report['negative_cache_f16_retains_rejection'] = all(
        arm['strict_diff_exit'] == 0 and arm['native']['wer'] == arm['cli']['wer'] == 1
        for arm in (off, repeat))
    assert on['cross_cache']['bytes'] == 2 * off['cross_cache']['bytes'] == 2 * repeat['cross_cache']['bytes']
    report['cache_byte_delta'] = on['cross_cache']['bytes'] - off['cross_cache']['bytes']
    report['precision_controls_pass'] = (on['fixed_prompt_pass'] and report['cache_off_repeat_exact']
                                        and (a.cuda_control or report['negative_cache_f16_retains_rejection']))
else:
    report['precision_controls_pass'] = all(arm['fixed_prompt_pass'] for arm in report['arms'])
report['scope'] = ('Actual native CUDA cache triage; CPU/broad acceptance separate; ' if a.cuda_control else '') + ('Fixed F32 neural weights with F16/F32/F16-repeat cache control; ' if a.cross_cache_control else 'Two fixed CPU precision controls; ') + 'Native controls with physical original tensor audits, strict canonical stages and actual native CLI/C-ABI ASR; no full-model/GPU/Q4 acceptance or model-pin promotion.'
checkpoint()
assert report['precision_controls_pass'], 'One or more fixed precision controls failed; preserve all results'
print('DIA_NEURAL_PRECISION_CONTROL_PASS; fixed-prompt CPU scope only', flush=True)
