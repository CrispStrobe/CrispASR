#!/usr/bin/env python3
"""PR 559 hosted short-clip Qwen3 C ABI and CLI context regression."""
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import wave

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(os.environ['HEAVY_OUT'])
SCRATCH = Path(os.environ['HEAVY_SCRATCH']) / 'asr-prompts'
OUT.mkdir(parents=True, exist_ok=True)
SCRATCH.mkdir(parents=True, exist_ok=True)
os.environ['TMPDIR'] = str(SCRATCH)
os.environ['OMP_NUM_THREADS'] = '4'
PINS = {
    'qwen': ('cstr/qwen3-asr-1.7b-GGUF', '674df5d44b50a63e7102a18895ed20e3f91de301', 'qwen3-asr-1.7b-q8_0.gguf'),
    'mimo': ('cstr/mimo-asr-GGUF', 'e2d7dfebf0afd8076771903e92958039c5074eab', 'mimo-asr-q4_k.gguf'),
    'codec': ('cstr/mimo-tokenizer-GGUF', 'fa380f4c49a8e8c62c02c00d0da5e263fc5b0dcf', 'mimo-tokenizer-q4_k.gguf'),
    'piper': ('cstr/piper-en_US-lessac-medium-GGUF', '6b3d385695cd2e91c1af9d37ca5684ee649e2e3d', 'piper-en_US-lessac-medium-f16.gguf'),
}


def run(command, tag, timeout=2400):
    with (OUT / (tag + '.log')).open('w') as log:
        result = subprocess.run(list(map(str, command)), cwd=ROOT, stdout=log,
                                stderr=subprocess.STDOUT, timeout=timeout)
    print(tag, result.returncode, (OUT / (tag + '.log')).read_text()[-1600:], flush=True)
    assert result.returncode == 0, tag


build = SCRATCH / 'build'
run(['cmake', '-S', ROOT, '-B', build, '-G', 'Ninja', '-DCMAKE_BUILD_TYPE=Release',
     '-DBUILD_SHARED_LIBS=ON', '-DGGML_NATIVE=OFF', '-DGGML_BLAS=OFF',
     '-DCRISPASR_BUILD_SERVER=OFF', '-DCRISPASR_BUILD_TESTS=ON'], 'configure')
run(['cmake', '--build', build, '--target', 'crispasr-cli', 'crispasr-lib',
     'test-asr-prompt-contract', 'test-mimoasr-params', 'test_tts_provenance', '-j4'], 'build')
cli = build / 'bin/crispasr'
lib = next(build.rglob('libcrispasr.so'))
for target, spec in [('test-asr-prompt-contract', '[unit]'), ('test-mimoasr-params', '[unit]'),
                     ('test_tts_provenance', '[mp3],[aac]')]:
    run([build / 'bin' / target, spec], target)
for generator in ['gen-feature-matrix.py', 'gen-backend-caps-table.py']:
    run([sys.executable, ROOT / 'tools' / generator, '--crispasr', cli], generator)
for source in ['docs/feature-matrix.md', 'docs/feature-matrix.html', 'src/core/backend_caps_table.h']:
    shutil.copy2(ROOT / source, OUT / Path(source).name)
run(['cmake', '--build', build, '--target', 'crispasr-lib', '-j4'], 'abi-capabilities')
run(['sudo', 'apt-get', 'update'], 'apt-update')
run(['sudo', 'apt-get', 'install', '-y', 'ffmpeg', 'espeak-ng'], 'ffmpeg-install')
from huggingface_hub import hf_hub_download
import numpy as np
sys.path.insert(0, str(ROOT / 'python'))
from crispasr import Session


def download(name):
    repo, revision, file = PINS[name]
    return Path(hf_hub_download(repo, file, revision=revision, local_dir=SCRATCH / name))


def pcm(path):
    with wave.open(str(path)) as wav:
        assert (wav.getframerate(), wav.getnchannels(), wav.getsampwidth()) == (16000, 1, 2)
        return np.frombuffer(wav.readframes(wav.getnframes()), dtype='<i2').astype(np.float32) / 32768


def normalized(text):
    return ' '.join(re.findall('[a-z]+', text.lower()))


results = dict(passed=False, source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
               model_pins=PINS, qwen={}, mimo={}, scope='CPU; synthetic short clips, JFK and SenseVoice Chinese sample; no Vulkan claim')


def save():
    (OUT / 'acceptance.json').write_text(json.dumps(results, indent=2, ensure_ascii=False) + '\n')


piper = download('piper')
shorts = {'hello': 'Hello.', 'thanks': 'Thank you.', 'fox': 'The quick brown fox jumps over the lazy dog.'}
for name, text in shorts.items():
    raw, audio = OUT / (name + '-raw.wav'), OUT / (name + '.wav')
    run([cli, '--backend', 'piper', '-m', piper, '-ng', '-l', 'en', '--seed', '1234', '--no-spoken-disclaimer',
         '--accept-marking-responsibility', '--tts', text, '--tts-output', raw], 'piper-' + name)
    run(['ffmpeg', '-y', '-i', raw, '-ar', '16000', '-ac', '1', audio], 'resample-' + name)
qwen = download('qwen')
with Session(str(qwen), lib_path=str(lib), backend='qwen3', n_threads=4) as session:
    session.set_source_language('en')
    session.set_max_new_tokens(128)
    for name, expected in shorts.items():
        cases = {}
        audio = OUT / (name + '.wav')
        for mode, hotwords, flag in [('off', '', '0'), ('on', 'Hello, CrispASR, thank you', '0'),
                                      ('cleared', '', '0'), ('raw', 'Hello, CrispASR, thank you', '1'),
                                      ('raw-cleared', '', '1')]:
            os.environ['CRISPASR_QWEN3_HOTWORDS_AS_CONTEXT'] = flag
            session.set_hotwords(hotwords)
            text = ' '.join(s.text for s in session.transcribe(pcm(audio)))
            cases[mode] = text
            results['qwen'][name] = dict(expected=expected, transcripts=cases,
                                         audio_sha256=hashlib.sha256(audio.read_bytes()).hexdigest())
            save()
            assert normalized(text) == normalized(expected), (name, mode, text, expected)
        assert cases['off'] == cases['cleared'] == cases['raw-cleared'], (name, cases)
        prefix = OUT / ('qwen-cli-' + name)
        run([cli, '--backend', 'qwen3', '-m', qwen, '-ng', '-t', '4', '-l', 'en',
             '--hotwords', 'Hello, CrispASR, thank you', '-f', audio, '-otxt', '-of', prefix], 'qwen-cli-' + name)
        cli_text = prefix.with_suffix('.txt').read_text().strip()
        assert normalized(cli_text) == normalized(expected), cli_text
        results['qwen'][name]['cli'] = cli_text
        for mode, context in [('raw', 'Hello, CrispASR, thank you'), ('empty', '')]:
            prefix = OUT / ('qwen-cli-' + name + '-' + mode)
            run([cli, '--backend', 'qwen3', '-m', qwen, '-ng', '-t', '4', '-l', 'en',
                 '--context', context, '-f', audio, '-otxt', '-of', prefix], 'qwen-cli-' + name + '-' + mode)
            text = prefix.with_suffix('.txt').read_text().strip()
            assert normalized(text) == normalized(expected), (name, mode, text)
            results['qwen'][name]['cli-' + mode] = text
        save()
results['passed'] = True
save()
(OUT / 'summary.md').write_text('PR 559 Qwen3 CPU: short clips through C ABI hotwords/raw context/clearing and CLI context on/empty pass. No GPU claim.\n')
