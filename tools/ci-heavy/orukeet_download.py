#!/usr/bin/env python3
"""Issue #491: real CLI download/cache routing, old-release control, and C ABI.

This changes no model math: compare the same Q4 weights through the short-name,
explicit-path and anonymous-filename session surfaces on JFK.
"""
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tarfile
import urllib.request
import wave

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(os.environ['HEAVY_OUT'])
SCRATCH = Path(os.environ['HEAVY_SCRATCH']) / 'orukeet-download'
OUT.mkdir(parents=True, exist_ok=True)
SCRATCH.mkdir(parents=True, exist_ok=True)
os.environ['TMPDIR'] = str(SCRATCH)
os.environ['CRISPASR_CACHE_DIR'] = str(SCRATCH / 'cache')
os.environ.pop('CRISPASR_MODELS_DIR', None)
REVISION = '2a93474c2771a6ca2228a7e001bd06ce0c97f880'
SHA256 = '769f346e3960de76afabcb82670b9d0d78c91051bc030be82e77eac45a7075be'


def run(command, tag, expected_success=True, timeout=2400):
    log_path = OUT / (tag + '.log')
    with log_path.open('w') as log:
        result = subprocess.run(list(map(str, command)), cwd=ROOT, stdin=subprocess.DEVNULL,
                                stdout=log, stderr=subprocess.STDOUT, timeout=timeout)
    text = log_path.read_text()
    print(tag, result.returncode, text[-1800:], flush=True)
    if expected_success:
        assert result.returncode == 0, tag
    return result, text


build = SCRATCH / 'build'
run(['cmake', '-S', ROOT, '-B', build, '-G', 'Ninja', '-DCMAKE_BUILD_TYPE=Release',
     '-DBUILD_SHARED_LIBS=ON', '-DGGML_NATIVE=OFF', '-DGGML_BLAS=OFF',
     '-DCRISPASR_BUILD_SERVER=OFF', '-DCRISPASR_BUILD_TESTS=ON'], 'configure')
run(['cmake', '--build', build, '--target', 'crispasr-cli', 'crispasr-lib', 'test-registry', '-j4'], 'build')
run([build / 'bin/test-registry', '[registry]'], 'registry')
cli = build / 'bin/crispasr'
cache = SCRATCH / 'cache'
cache.mkdir(exist_ok=True)
model = cache / 'orukeet-q4_k.gguf'
assert not model.exists(), 'First run must exercise the actual downloader'
common = ['-ng', '-t', '4', '-l', 'en', '-f', ROOT / 'samples/jfk.wav', '--cache-dir', cache]
# Acceptance is deliberately required for CC-BY-SA. Verify the refusal
# without fetching weights, then explicitly accept for this validation run.
os.environ.pop('CRISPASR_ACCEPT_LICENSE', None)
refused, refusal_log = run([cli, '-m', 'orukeet', *common, '--auto-download'],
                           'license-refusal', expected_success=False)
assert refused.returncode != 0 and '--accept-license cc-by-sa-4.0' in refusal_log
assert 'NON-COMMERCIAL' not in refusal_log, 'CC-BY-SA is not a noncommercial licence'
assert not model.exists(), 'Unaccepted licence must not fetch the model'
texts = {}
for name, arg, extra in [('download', 'orukeet', ['--auto-download', '--accept-license', 'cc-by-sa-4.0']),
                         ('cached', 'orukeet', []), ('filename', model.name, []),
                         ('explicit', model, [])]:
    prefix = OUT / name
    _, log = run([cli, '-m', arg, *common, *extra, '-otxt', '-of', prefix], name)
    assert 'failed to initialize whisper' not in log.lower(), name
    assert 'NON-COMMERCIAL' not in log, name
    texts[name] = prefix.with_suffix('.txt').read_text().strip()
    assert texts[name], name
    if name == 'download':
        assert model.stat().st_size == 402226496
        with model.open('rb') as f:
            assert hashlib.file_digest(f, 'sha256').hexdigest() == SHA256, 'Public artifact changed'
        assert 'downloading' in log.lower(), 'Downloader was not exercised'
    else:
        assert 'auto-downloading' not in log.lower(), 'Cached model should not download'
assert len(set(texts.values())) == 1, texts
words = re.findall('[a-z]+', texts['download'].lower())
assert all(w in words for w in ['americans', 'country', 'ask']), texts

# The released binary fails with exactly the same model in its configured cache.
archive = SCRATCH / 'old-release.tar.gz'
url = 'https://github.com/CrispStrobe/CrispASR/releases/download/v0.8.41/crispasr-linux-x86_64.tar.gz'
urllib.request.urlretrieve(url, archive)
old_dir = SCRATCH / 'old-release'
with tarfile.open(archive) as tar:
    tar.extractall(old_dir, filter='data')
old_cli = next(p for p in old_dir.rglob('crispasr') if p.is_file())
old, old_log = run([old_cli, '-m', 'orukeet', *common], 'baseline', expected_success=False)
assert old.returncode != 0 and 'failed to initialize whisper context' in old_log, 'Control did not reproduce #491'

import numpy as np
sys.path.insert(0, str(ROOT / 'python'))
from crispasr import Session
lib = next(build.rglob('libcrispasr.so'))
anonymous = SCRATCH / 'anonymous.gguf'
shutil.copyfile(model, anonymous)
with wave.open(str(ROOT / 'samples/jfk.wav')) as wav:
    assert (wav.getframerate(), wav.getnchannels(), wav.getsampwidth()) == (16000, 1, 2)
    pcm = np.frombuffer(wav.readframes(wav.getnframes()), dtype='<i2').astype(np.float32) / 32768
with Session(str(anonymous), lib_path=str(lib), n_threads=4) as session:
    assert session.backend == 'parakeet', session.backend
    session.set_source_language('en')
    texts['abi_anonymous'] = ' '.join(s.text for s in session.transcribe(pcm)).strip()
assert texts['abi_anonymous'] == texts['explicit'], texts
receipt = dict(passed=True, source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
               model_repo='cstr/orukeet-GGUF', model_revision=REVISION, model_sha256=SHA256,
               baseline_release='v0.8.41', baseline_reproduced=True, license_refusal_verified=True,
               accepted_license='cc-by-sa-4.0', transcripts=texts,
               scope='CPU; same-Q4 cross-surface routing; no new numerical/performance claim')
(OUT / 'acceptance.json').write_text(json.dumps(receipt, indent=2) + '\n')
(OUT / 'summary.md').write_text('Orukeet download, cached short-name reuse, filename/path CLI, and anonymous C ABI PASS; v0.8.41 control reproduces #491.\n')
