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
    'piper': ('cstr/piper-en_US-lessac-medium-GGUF', '6b3d385695cd2e91c1af9d37ca5684ee649e2e3d', 'piper-en_US-lessac-medium-f16.gguf'),
}
PR_HEAD = '6b132720090cbb532f79f544c130381ddf1cf6c9'
# Git blob IDs from the exact PR head. The hosted validation branch is a
# cherry-pick onto current main, so its commit ID alone does not prove that
# the PR's implementation was tested. cli.cpp also has the unrelated #560
# language-detection early-exit insertion; its PR help hunk is checked below.
PR_BLOBS = {
    'examples/cli/crispasr_backend.cpp': 'e995ba32581cc63543f8ccbba330901010e9f317',
    'examples/cli/crispasr_backend_qwen3.cpp': 'c4ae1c02e6604014626bda30d875100c7f6e7c3a',
    'examples/cli/whisper_params.h': 'fd1734889a893e11648481894dbe209849b8df9a',
    'src/core/qwen3_prompt.h': '6db29486030a7228e7613082f6a9caad95dc3bf1',
    'src/crispasr_c_api.cpp': '45d3472ce4d77dd16fce8380fae42594b2061846',
    'tests/test-asr-prompt-contract.cpp': 'd4efe975c315f73d24ce8bc48e771f9fc4bf488f',
}
PR_CLI_BLOB = '8fcbc8c8a41b5be041df4da867cc37714c54cb40'
VALIDATION_CLI_BLOB = 'b55ba8d05910638eac8990d79c374820a032be3b'


def verify_pr_blobs():
    observed = {}
    for source, expected in PR_BLOBS.items():
        actual = subprocess.check_output(['git', 'hash-object', str(ROOT / source)], text=True).strip()
        if actual != expected:
            raise RuntimeError(f'PR head blob mismatch: {source}: {actual} != {expected}')
        observed[source] = actual
    cli_source = ROOT / 'examples/cli/cli.cpp'
    cli_blob = subprocess.check_output(['git', 'hash-object', str(cli_source)], text=True).strip()
    if cli_blob != VALIDATION_CLI_BLOB:
        raise RuntimeError(f'Unexpected validation CLI blob: {cli_blob}')
    cli_text = cli_source.read_text()
    if cli_text.count('(qwen3, vibevoice-asr)') != 1:
        raise RuntimeError('PR CLI context help hunk is missing or duplicated')
    observed['examples/cli/cli.cpp'] = {
        'pr_head_blob': PR_CLI_BLOB, 'validation_blob': cli_blob,
        'pr_context_help_hunk': True, 'overlay': 'unrelated issue #560 language-detection early exit',
    }
    return observed


verified_pr_blobs = verify_pr_blobs()


def run(command, tag, timeout=2400):
    with (OUT / (tag + '.log')).open('w') as log:
        result = subprocess.run(list(map(str, command)), cwd=ROOT, stdout=log,
                                stderr=subprocess.STDOUT, timeout=timeout)
    print(tag, result.returncode, (OUT / (tag + '.log')).read_text()[-1600:], flush=True)
    assert result.returncode == 0, tag


run([sys.executable, ROOT / 'tools/ci-heavy/qwen3_pr559_factory_probe.py'], 'backend-factory-routing')

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
               pr_head=PR_HEAD, verified_pr_blobs=verified_pr_blobs, model_pins=PINS, qwen={},
               scope='CPU; Piper-generated synthetic English short clips and prompt routing; no Vulkan claim')


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
# A test-only ELF interposer records the actual tokenizer inputs produced by
# the C ABI. Use only synthetic clips and context; no user audio is involved.
trace_lib = SCRATCH / 'qwen3-prompt-trace.so'
run(['cc', '-shared', '-fPIC', '-O2', ROOT / 'tools/ci-heavy/qwen3_prompt_trace.c',
     '-ldl', '-o', trace_lib], 'build-prompt-trace')
run(['env', 'LD_PRELOAD=' + str(trace_lib), 'CRISPASR_TEST_QWEN_LIB=' + str(lib), sys.executable,
     ROOT / 'tools/ci-heavy/qwen3_pr559_trace.py', lib, qwen, OUT / 'hello.wav', OUT, cli],
    'c-abi-prompt-routing')
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
