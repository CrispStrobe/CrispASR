#!/usr/bin/env python3
"""Dia complete-speech acceptance, isolated C ABI generation and ASR processes."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import wave

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(os.environ['HEAVY_OUT'])
SCRATCH = Path(os.environ['HEAVY_SCRATCH'])
PHRASES = [
    ('hello', 42, 'Hello there, how are you doing today? I really hope you are having a wonderful and pleasant time. The weather outside is lovely and bright.'),
    ('fox', 123, 'The quick brown fox jumps over the lazy dog. Please listen carefully, because this is a test of speech synthesis with a chosen number of CPU threads.'),
]
REPORTED_SHORT = ('reported', 42, 'We are committed to making it secure.')
ALL_PHRASES = PHRASES + [REPORTED_SHORT]
p = argparse.ArgumentParser()
p.add_argument('--child', choices=('generate', 'recognize'))
p.add_argument('--model')
p.add_argument('--lib')
p.add_argument('--phrase', choices=('hello', 'fox', 'reported'))
p.add_argument('--short-report', action='store_true', help='also test the exact short prompt from PR #530 through C ABI and CLI')
p.add_argument('--threads', type=int, default=4)
p.add_argument('--steps', type=int, default=0)
p.add_argument('--quant', choices=('f16', 'q8_0', 'q4_k'), default='q8_0')
p.add_argument('--matrix', default='4,1,8')
p.add_argument('--cli-only', action='store_true', help='verify CLI without repeating the C ABI speech matrix')
p.add_argument('--metal', action='store_true', help='require real Metal CLI backend on macOS')
p.add_argument('--audio-tag', help='separate output basename for CLI recognition')
p.add_argument('--limits', action='store_true', help='also verify C ABI setter and CLI explicit/default limits')
a = p.parse_args()
OUT.mkdir(parents=True, exist_ok=True)
SCRATCH.mkdir(parents=True, exist_ok=True)

if a.metal:
    assert sys.platform == 'darwin', 'Metal acceptance requires a macOS runner'
    os.environ['CRISPASR_DIA_TTS_GPU'] = '1'

if a.child:
    import numpy as np
    sys.path.insert(0, str(ROOT / 'python'))
    from crispasr import Session
    key, seed, text = next(c for c in ALL_PHRASES if c[0] == a.phrase)
    tag = a.audio_tag or f'{key}-{a.threads}'
    path = OUT / (tag + '.npy')
    if a.child == 'generate':
        if a.steps:
            os.environ['CRISPASR_DIA_MAX_STEPS'] = str(a.steps)
        with Session(a.model, lib_path=a.lib, backend='dia', n_threads=a.threads) as session:
            session.set_temperature(1.2, seed=seed)
            capped_seconds = None
            if a.limits and a.phrase == 'hello' and a.threads == 4:
                session.set_max_new_tokens(32)
                capped = session.synthesize('[S1] ' + text)
                capped_seconds = len(capped) / 44100
                assert 0 < capped_seconds < .3, capped_seconds
                session.set_max_new_tokens(0)
                session.set_temperature(1.2, seed=seed)
            start = time.perf_counter()
            pcm = session.synthesize('[S1] ' + text)
            elapsed = time.perf_counter() - start
            assert session.output_sample_rate() == 44100
        np.save(path, pcm)
        rms = float(np.sqrt(np.mean(pcm.astype(np.float64) ** 2))) if len(pcm) else 0.
        result = {'phrase': text, 'seed': seed, 'threads': a.threads, 'steps_override': a.steps,
                  'audio_seconds': len(pcm) / 44100, 'generation_seconds': elapsed, 'cabi_limit_32_seconds': capped_seconds,
                  'rms': rms, 'finite': bool(np.isfinite(pcm).all())}
        (OUT / (tag + '.json')).write_text(json.dumps(result, indent=2))
        assert result['finite'] and rms > 1e-4, result
        # Keep the existing long-speech gate. The new short case need not last 3 s.
        assert result['audio_seconds'] > (0 if key == 'reported' else 3), result
    else:
        with Session(a.model, lib_path=a.lib, backend='nemotron', n_threads=4) as session:
            transcript = ' '.join(s.text for s in session.transcribe(np.load(path), sample_rate=44100, language='en'))
        def words(t):
            return re.findall('[a-z]+', re.sub(r'<[^>]*>', '', t).lower())
        ref, actual = words(text), words(transcript)
        row = list(range(len(actual) + 1))
        for i, r in enumerate(ref, 1):
            new = [i]
            for j, x in enumerate(actual, 1):
                new.append(min(new[-1] + 1, row[j] + 1, row[j - 1] + (r != x)))
            row = new
        result = json.loads((OUT / (tag + '.json')).read_text())
        result.update(transcript=transcript, wer=row[-1] / len(ref))
        (OUT / (tag + '.json')).write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps(result), flush=True)
        assert result['wer'] <= .2, result
    sys.exit(0)

from huggingface_hub import hf_hub_download

failures = []

def run(cmd, tag, allow_failure=False):
    with (OUT / (tag + '.log')).open('w') as log:
        proc = subprocess.run(list(map(str, cmd)), cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, timeout=2400)
    print(tag, proc.returncode, (OUT / (tag + '.log')).read_text()[-2200:], flush=True)
    if proc.returncode and allow_failure:
        failures.append(tag)
    else:
        assert proc.returncode == 0, tag
    return proc.returncode

subprocess.run(['uptime'], check=True)
subprocess.run(['vm_stat'] if sys.platform == 'darwin' else ['free', '-h'], check=True)
if a.metal:
    run([sys.executable, ROOT / 'tools/ci-heavy/dia_metal_probe.py'], 'metal-hardware-probe')
build = SCRATCH / 'dia-build'
run(['cmake', '-S', ROOT, '-B', build, '-DCMAKE_BUILD_TYPE=Release', '-DBUILD_SHARED_LIBS=ON',
     '-DGGML_NATIVE=OFF', '-DGGML_CUDA=OFF', '-DGGML_VULKAN=OFF', '-DGGML_BLAS=OFF',
     '-DCRISPASR_BUILD_TESTS=OFF', '-DCRISPASR_BUILD_SERVER=OFF', '-DCRISPASR_OPUS=OFF', '-DCRISPASR_AMR=OFF'], 'configure')
run(['cmake', '--build', build, '--target', 'crispasr-lib', 'crispasr-cli', '-j4'], 'build')
if a.limits:
    assert (build / 'bin/crispasr').is_file(), 'CMake CLI output is bin/crispasr'
lib = next(build.rglob('libcrispasr.dylib' if sys.platform == 'darwin' else 'libcrispasr.so'))
repo, revision = 'cstr/dia-1.6b-GGUF', '3233fbcb32be47761d2e736857b6d1a075b9ba7e'
model = hf_hub_download(repo, f'dia-1.6b-{a.quant}.gguf', revision=revision)
hf_hub_download(repo, 'dac-44khz.gguf', revision=revision)
asr = hf_hub_download('cstr/nemotron-3.5-asr-streaming-GGUF', 'nemotron-3.5-asr-streaming-0.6b-q4_k.gguf', revision='bbd95a9ca5fa0dfca3312a122dfc45a2b578b9c2')
selected = [c for c in ALL_PHRASES if c[0] == a.phrase] if a.phrase else list(PHRASES)
if a.short_report and REPORTED_SHORT not in selected:
    selected.append(REPORTED_SHORT)
for key, _, _ in ([] if a.cli_only else selected):
    for threads in map(int, a.matrix.split(',')):
        for child, weights in (('generate', model), ('recognize', asr)):
            run([sys.executable, __file__, '--child', child, '--model', weights, '--lib', lib,
                 '--phrase', key, '--threads', threads, '--steps', a.steps] + (['--limits'] if a.limits else []) + (['--metal'] if a.metal else []),
                f'{key}-{threads}-{child}', allow_failure=(key == 'reported'))
if a.short_report or a.phrase == 'reported':
    import numpy as np
    key, seed, text = REPORTED_SHORT
    wav = OUT / 'reported-cli.wav'
    command = [build / 'bin/crispasr', '-m', model, '--codec-model', Path(model).parent / 'dac-44khz.gguf',
               '--tts-output', wav, '-l', 'en', '--tts', text, '-t', '4', '--seed', str(seed),
               '--temperature', '1.2', '--no-spoken-disclaimer', '--accept-marking-responsibility']
    if not a.metal:
        command += ['--no-gpu']
    run(command, 'reported-cli')
    if a.metal:
        assert 'dia_tts: GPU backend enabled (MTL' in (OUT / 'reported-cli.log').read_text(), 'Metal fallback rejected'
    with wave.open(str(wav)) as audio:
        assert audio.getframerate() == 44100 and audio.getnchannels() == 1 and audio.getsampwidth() == 2
        pcm = np.frombuffer(audio.readframes(audio.getnframes()), dtype='<i2').astype(np.float32) / 32768
    np.save(OUT / 'reported-cli.npy', pcm)
    rms = float(np.sqrt(np.mean(pcm.astype(np.float64) ** 2))) if len(pcm) else 0.
    result = {'phrase': text, 'seed': seed, 'cli': True, 'audio_seconds': len(pcm) / 44100,
              'rms': rms, 'finite': bool(np.isfinite(pcm).all())}
    (OUT / 'reported-cli.json').write_text(json.dumps(result, indent=2) + '\n')
    assert result['finite'] and rms > 1e-4 and len(pcm) > 0, result
    run([sys.executable, __file__, '--child', 'recognize', '--model', asr, '--lib', lib,
         '--phrase', key, '--audio-tag', 'reported-cli'], 'reported-cli-recognize', allow_failure=True)
if a.limits:
    # CLI must preserve the model default, and honor an explicit short limit.
    import numpy as np
    for limit in (32, 0):
        wav = OUT / f'cli-limit-{limit}.wav'
        command = [build / 'bin/crispasr', '--backend', 'dia', '-m', model,
                   '-t', '4', '--seed', '123', '--temperature', '1.2',
                   '--tts', '[S1] ' + PHRASES[1][2], '--tts-output', wav,
                   '--no-spoken-disclaimer', '--accept-marking-responsibility']
        if not a.metal:
            command += ['--no-gpu']
        if limit:
            command += ['--max-new-tokens', str(limit)]
        run(command, f'cli-limit-{limit}')
        if a.metal:
            assert 'dia_tts: GPU backend enabled (MTL' in (OUT / f'cli-limit-{limit}.log').read_text(), 'Metal fallback rejected'
        with wave.open(str(wav)) as audio:
            assert audio.getframerate() == 44100 and audio.getnchannels() == 1
            assert audio.getsampwidth() == 2
            pcm = np.frombuffer(audio.readframes(audio.getnframes()), dtype='<i2').astype(np.float32) / 32768
        seconds = len(pcm) / 44100
        if limit:
            log = (OUT / f'cli-limit-{limit}.log').read_text()
            assert log.count('starting decoder loop (max_gen=32,') == 1, 'explicit cap repeated per sentence'
            assert 0 < seconds < .3, seconds
        else:
            assert seconds > 3, seconds
            # Reuse the independent ASR acceptance for CLI output as well.
            np.save(OUT / 'cli-default.npy', pcm)
            (OUT / 'cli-default.json').write_text(json.dumps({'phrase': PHRASES[1][2], 'audio_seconds': seconds, 'cli': True}))
            run([sys.executable, __file__, '--child', 'recognize', '--model', asr,
                 '--lib', lib, '--phrase', 'fox', '--threads', '4', '--audio-tag', 'cli-default'], 'cli-default-recognize')
            # Inspect the actual decoder capacity, independent of quantization
            # or sampled speaking rate. Speech must still pass the ASR gate.
            log = (OUT / 'cli-limit-0.log').read_text()
            assert 'starting decoder loop (max_gen=3072,' in log, 'CLI inherited a generic token cap'
            assert log.count('starting decoder loop (max_gen=3072,') == 1, 'Dia was sentence-split'
        (OUT / f'cli-limit-{limit}.json').write_text(json.dumps({'max_new_tokens': limit, 'audio_seconds': seconds}))
assert not failures, f'Short-prompt controls failed: {failures}; retained PCM/metrics/logs are diagnostic evidence, not acceptance'
print('DIA_FULL_SPEECH_PASS', flush=True)
