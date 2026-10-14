#!/usr/bin/env python3
"""Free-running Dia CLI/C-ABI capture transparency; speech is a separate gate."""
import argparse
import json
import os
from pathlib import Path
import re
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
from reference_backends import dia_tts_reference as reference

TEXT = '[S1] We are committed to making it secure.'
ARMS = ('plain', 'logging', 'common', 'boundary', 'plain-repeat')

def arm_environment(surface, arm):
    # Remove inherited diagnostic/forcing flags instead of setting presence flags to zero.
    env = {k: v for k, v in os.environ.items() if not k.startswith('CRISPASR_DIA_')}
    env.update(CRISPASR_DIA_CROSS_CACHE_F32='1', CRISPASR_DIA_PORTABLE_RNG='1',
               CRISPASR_DIA_TTS_GPU='0', CRISPASR_DIA_SOURCE_ROPE='0')
    if arm in ('logging', 'common', 'boundary'):
        env.update(CRISPASR_DIA_DUMP_TOKENS='1',
                   CRISPASR_DIA_DUMP_INPUT_TOKENS=str(OUT / (surface + '-' + arm + '-inputs.i32')),
                   CRISPASR_DIA_DUMP_STEPLOGITS=str(OUT / (surface + '-' + arm + '-logits.f32')))
    if arm in ('common', 'boundary'):
        directory = OUT / (surface + '-' + arm + '-capture')
        directory.mkdir(exist_ok=True)
        env.update(CRISPASR_DIA_DUMP_DIR=str(directory), CRISPASR_DIA_DUMP_STEP='978')
    if arm == 'boundary':
        env['CRISPASR_DIA_DUMP_DECODER_BOUNDARIES'] = '1'
    assert not any('FORCE' in k or 'GREEDY' in k or 'MAX_STEPS' in k for k in env if k.startswith('CRISPASR_DIA_'))
    return env

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--plan-only', action='store_true')
a = parser.parse_args()
if a.plan_only:
    for surface in ('cabi', 'cli'):
        for arm in ARMS:
            env = arm_environment(surface, arm)
            print(surface, arm, json.dumps({k: v for k, v in env.items() if k.startswith('CRISPASR_DIA_')}, sort_keys=True))
    sys.exit(0)

report = dict(source=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
              worktree_dirty=bool(subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT)),
              worker_sha256=digest(Path(__file__)), runs=[], rows=[], step=978,
              scope='No forcing; unmarked/marked CLI and C-ABI free generation under fixed F32-cross portable-RNG source-RoPE-off diagnostic recipe. No new production default, Q4, GPU or performance acceptance.',
              speech_acceptance=False, gpu_acceptance=False, q4_acceptance=False,
              production_defaults_changed=False)
assert not report['worktree_dirty']
if os.environ.get('GITHUB_SHA'):
    assert report['source'] == os.environ['GITHUB_SHA']
for name in ('src/dia_tts.cpp', 'src/dia_dump_step.h', 'tools/ci-heavy/dia_free_capture_control.py',
             'tools/ci-heavy/dia_full_speech.py', 'src/crispasr_c_api.cpp',
             'examples/cli/crispasr_backend_dia.cpp'):
    target = OUT / 'sources' / name
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ROOT / name, target)

def checkpoint():
    temporary = OUT / 'receipt.json.tmp'
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    temporary.replace(OUT / 'receipt.json')

def run(command, label, env=None, allow_failure=False):
    log = OUT / (label + '.log')
    with log.open('w') as file:
        proc = subprocess.run(list(map(str, command)), cwd=ROOT, env=env, stdout=file,
                              stderr=subprocess.STDOUT, timeout=10800)
    report['runs'].append(dict(name=label, exit=proc.returncode, log_sha256=digest(log)))
    checkpoint()
    print(label, proc.returncode, log.read_text()[-1200:], flush=True)
    if not allow_failure:
        assert proc.returncode == 0, label
    return proc.returncode

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
run(['cmake', '--build', build, '--target', 'crispasr-lib', 'crispasr-cli', '-j4'], 'build')
lib = next(build.rglob('libcrispasr.so')); cli = build / 'bin/crispasr'
report.update(library_sha256=digest(lib), cli_sha256=digest(cli), model_sha256=digest(model), codec_sha256=digest(codec))
asr = hf_hub_download('cstr/nemotron-3.5-asr-streaming-GGUF', 'nemotron-3.5-asr-streaming-0.6b-q4_k.gguf',
                     revision='bbd95a9ca5fa0dfca3312a122dfc45a2b578b9c2')
report['asr_sha256'] = digest(Path(asr))
histories = {}
for surface in ('cabi', 'cli'):
    for arm in ARMS:
        tag = surface + '-' + arm
        env = arm_environment(surface, arm)
        if surface == 'cabi':
            command = [sys.executable, ROOT / 'tools/ci-heavy/dia_full_speech.py', '--child', 'generate',
                       '--model', model, '--lib', lib, '--phrase', 'reported', '--threads', '4', '--audio-tag', tag]
            run(command, tag + '-generate', env)
            pcm = np.load(OUT / (tag + '.npy'))
        else:
            wav = OUT / (tag + '.wav')
            command = [cli, '--backend', 'dia', '-m', model, '--codec-model', codec, '--tts-output', wav,
                       '-l', 'en', '--tts', TEXT, '-t', '4', '--seed', '42', '--temperature', '1.2', '--no-gpu',
                       '--no-spoken-disclaimer', '--accept-marking-responsibility']
            run(command, tag + '-generate', env)
            with wave.open(str(wav)) as audio:
                assert (audio.getframerate(), audio.getnchannels(), audio.getsampwidth()) == (44100, 1, 2)
                pcm = np.frombuffer(audio.readframes(audio.getnframes()), dtype='<i2').astype('f4') / 32768
            np.save(OUT / (tag + '.npy'), pcm)
            (OUT / (tag + '.json')).write_text(json.dumps(dict(phrase=TEXT[5:], seed=42, cli=True,
                audio_seconds=len(pcm)/44100, rms=float(np.sqrt(np.mean(pcm.astype('f8')**2))),
                finite=bool(np.isfinite(pcm).all())), indent=2) + '\n')
        assert len(pcm) and np.isfinite(pcm).all()
        log = (OUT / (tag + '-generate.log')).read_text()
        # Session uses the quiet C ABI; only CLI promises the verbose capacity line.
        # Both paths retain the checked model capacity and receive no token cap.
        if surface == 'cli':
            assert 'starting decoder loop (max_gen=3072,' in log, tag
            assert log.count('starting decoder loop (max_gen=3072,') == 1, tag
        assert 'DIA_FORCE_' not in log, tag
        row = dict(surface=surface, arm=arm, pcm_values=len(pcm), pcm_sha256=digest(OUT / (tag + '.npy')),
                   generation_environment={k: v for k, v in env.items() if k.startswith('CRISPASR_DIA_')})
        if arm in ('logging', 'common', 'boundary'):
            tokens = re.findall(r'^DIA_TOK step ([0-9]+):((?: [0-9]+){9})$', log, flags=re.M)
            assert tokens and [int(step) for step, _ in tokens] == list(range(len(tokens))), tag
            codes = np.asarray([[int(x) for x in values.split()] for _, values in tokens], dtype='<i4')
            codes.tofile(OUT / (tag + '-sampled.i32'))
            logits = OUT / (tag + '-logits.f32')
            inputs = OUT / (tag + '-inputs.i32')
            assert logits.stat().st_size == len(tokens)*2*9*1028*4, tag
            assert inputs.stat().st_size == len(tokens)*9*4, tag
            histories[surface, arm] = dict(steps=len(tokens), logits_sha256=digest(logits),
                inputs_sha256=digest(inputs), sampled_sha256=digest(OUT / (tag + '-sampled.i32')))
            row.update(histories[surface, arm])
            if arm in ('common', 'boundary'):
                assert len(tokens) > 978, 'Free run ended before the requested observation'
                capture = OUT / (tag + '-capture')
                snapshots = list(capture.glob('cpp_decoder_block*.f32'))
                assert len(snapshots) == (16 if arm == 'boundary' else 0), (tag, len(snapshots))
                assert not list(capture.glob('cpp_decoder_sa*.f32')), tag
                assert (capture / 'cpp_step978_cond.f32').stat().st_size == 9*1028*4
                assert (capture / 'cpp_step978_uncond.f32').stat().st_size == 9*1028*4
                row['boundary_snapshots'] = len(snapshots)
        else:
            assert 'DIA_TOK step' not in log and 'DIA_DUMP_DIR:' not in log, tag
        report['rows'].append(row)
        checkpoint()
    baseline = np.load(OUT / (surface + '-plain.npy'))
    for arm in ARMS:
        candidate = np.load(OUT / (surface + '-' + arm + '.npy'))
        assert candidate.dtype == baseline.dtype and candidate.shape == baseline.shape
        assert candidate.tobytes() == baseline.tobytes(), (surface, arm, 'complete PCM drift')
    assert histories[surface, 'logging'] == histories[surface, 'common'] == histories[surface, 'boundary'], surface
    report[surface + '_plain_logging_common_boundary_plain_repeat_pcm_exact'] = True
    report[surface + '_logged_complete_logits_inputs_samples_exact'] = True
    checkpoint()
    # Failed ASR is preserved as quality rejection, independent of capture transparency.
    run([sys.executable, ROOT / 'tools/ci-heavy/dia_full_speech.py', '--child', 'recognize',
         '--model', asr, '--lib', lib, '--phrase', 'reported', '--audio-tag', surface + '-plain'],
        surface + '-baseline-asr', allow_failure=True)
    quality = json.loads((OUT / (surface + '-plain.json')).read_text())
    assert 'wer' in quality and 'transcript' in quality, 'ASR must run; a crash is not a quality measurement'
    report[surface + '_baseline_quality'] = quality
    checkpoint()
assert histories['cabi', 'logging'] == histories['cli', 'logging'], 'CLI/C-ABI full history mismatch'
report['cli_cabi_logged_complete_logits_inputs_samples_exact'] = True
report['free_running_capture_transparency_pass'] = True
report['baseline_roundtrip_pass'] = all(report[s + '_baseline_quality']['wer'] <= .2 for s in ('cabi', 'cli'))
report['files'] = [dict(path=p.relative_to(OUT).as_posix(), bytes=p.stat().st_size, sha256=digest(p))
                   for p in sorted(OUT.rglob('*')) if p.is_file() and p.name not in ('receipt.json', 'receipt.json.tmp')]
# This diagnostic does not replace the full speech/quant/platform acceptance matrix.
checkpoint()
print('DIA_FREE_RUNNING_CAPTURE_TRANSPARENCY_PASS; full speech acceptance remains separate', flush=True)
