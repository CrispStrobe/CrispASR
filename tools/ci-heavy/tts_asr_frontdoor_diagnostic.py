#!/usr/bin/env python3
"""Cross-recognize fixed TTS PCM and isolate resampling; not speech acceptance."""
import ctypes as C
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import zipfile

import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly
from huggingface_hub import HfApi, hf_hub_download

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(os.environ['HEAVY_OUT'])
TEMP = Path(os.environ['HEAVY_SCRATCH']) / 'tts-frontdoor-diagnostic'
OUT.mkdir(parents=True, exist_ok=True)
TEMP.mkdir(parents=True, exist_ok=True)
TEXTS = {'short': 'Hello, this is a short test sentence.',
         'long': 'The sun is shining today. Hello, this is a short test sentence.'}
PROOFS = {
    'omnivoice': ('omnivoice/clone-diagnostic-20261009/proof.zip',
                 '3f7d258695ae58be8e0b7b717f4273b8241609a2',
                 'f69d5a1b7e9257c78e3cfc42736ebdcbb22227acbabfb00561c56b0602870b9b'),
    'voxcpm2-original': ('voxcpm2/original-controls-20261009/proof.zip',
                        '78790b006e7ee65f6e8975d437000858fb5dfe7a',
                        '2684b0da6981bb36423c9fc3c007d743069462860c7bba5f994bae364d39d234'),
    'voxcpm2-gpu': ('voxcpm2/path-diagnostic-20261009-v6/proof.zip',
                   '449ed355010eddb8dcedef6c2ab0fa62431ebeb7',
                   'f52e5fbc49a0654bdcd13da6660c8e0841fbe87e80eb0cc158d74801339a5abc'),
}
MODELS = {
    'nemotron': ('cstr/nemotron-3.5-asr-streaming-GGUF',
                 'bbd95a9ca5fa0dfca3312a122dfc45a2b578b9c2',
                 'nemotron-3.5-asr-streaming-0.6b-q4_k.gguf'),
    'parakeet': ('cstr/parakeet-tdt-0.6b-v3-GGUF',
                 '815cc1bcb5cfb92f9cd0bbd22c7bc2f6e2d4ec13',
                 'parakeet-tdt-0.6b-v3-q4_k.gguf'),
    'qwen3': ('cstr/qwen3-asr-0.6b-GGUF',
              'ad086c22597ed47af05cc159dd61c98bd6e945f9',
              'qwen3-asr-0.6b-q4_k.gguf'),
}
receipt = dict(passed=False, diagnostic_complete=False, scope=__doc__,
               source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
               models={}, proofs=PROOFS, cases={}, alias_controls={}, transcripts={})


def save():
    (OUT / 'frontdoor-diagnostic.json').write_text(json.dumps(receipt, indent=2) + '\n')


def run(command, tag):
    with (OUT / (tag + '.log')).open('w') as log:
        subprocess.run(list(map(str, command)), cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                       check=True, timeout=7200)


save()
build = TEMP / 'build'
run(['cmake', '-S', ROOT, '-B', build, '-G', 'Ninja', '-DCMAKE_BUILD_TYPE=Release',
     '-DBUILD_SHARED_LIBS=ON', '-DGGML_NATIVE=OFF', '-DGGML_CUDA=OFF',
     '-DCRISPASR_BUILD_TESTS=OFF', '-DCRISPASR_BUILD_SERVER=OFF'], 'configure')
run(['cmake', '--build', build, '--target', 'crispasr-lib', '-j4'], 'build')
lib_path = next(build.rglob('libcrispasr.so'))
lib = C.CDLL(str(lib_path))
lib.crispasr_audio_load.argtypes = [C.c_char_p, C.POINTER(C.POINTER(C.c_float)),
                                   C.POINTER(C.c_int), C.POINTER(C.c_int)]
lib.crispasr_audio_load.restype = C.c_int
lib.crispasr_audio_free.argtypes = [C.POINTER(C.c_float)]
lib.crispasr_audio_free.restype = None


def native_resample(pcm, rate, label):
    path = TEMP / (label + '.wav')
    wavfile.write(path, rate, np.asarray(pcm, dtype=np.float32))
    data = C.POINTER(C.c_float)()
    count, output_rate = C.c_int(), C.c_int()
    assert lib.crispasr_audio_load(os.fsencode(path), C.byref(data), C.byref(count), C.byref(output_rate)) == 0
    try:
        assert output_rate.value == 16000 and count.value > 0
        return np.ctypeslib.as_array(data, shape=(count.value,)).copy()
    finally:
        lib.crispasr_audio_free(data)


def linear_resample(pcm, rate):
    # Exactly the existing Python Session/Whisper ndarray front door.
    indices = np.linspace(0, len(pcm) - 1, int(len(pcm) * 16000 / rate))
    return np.interp(indices, np.arange(len(pcm)), pcm).astype(np.float32)


for rate in (24000, 48000):
    tone = (.5 * np.sin(2 * np.pi * 10000 * np.arange(rate) / rate)).astype(np.float32)
    outputs = {'python-linear': linear_resample(tone, rate),
               'native-file-loader': native_resample(tone, rate, 'tone-' + str(rate))}
    rows = {name: float(np.sqrt(np.mean(pcm[320:-320].astype(np.float64) ** 2)))
            for name, pcm in outputs.items()}
    receipt['alias_controls'][str(rate)] = dict(input_hz=10000, target_nyquist_hz=8000,
        rms=rows, linear_negative_control_met=rows['python-linear'] > .1,
        file_loader_alias_gate_met=rows['native-file-loader'] < .001,
        interpretation='C ABI file loader uses miniaudio, not core Kaiser polyphase; record failures without skipping fixed-audio transcripts')
    save()
    assert rows['python-linear'] > .1, rows
save()
cases = {}
for cohort, (filename, revision, checksum) in PROOFS.items():
    path = hf_hub_download('cstr/crispasr-regression-fixtures', filename, revision=revision)
    assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == checksum
    dest = TEMP / cohort
    with zipfile.ZipFile(path) as archive:
        archive.extractall(dest)
    for p in sorted(dest.rglob('*.npy')):
        if p.name.endswith('-latent.npy'):
            continue
        pcm = np.load(p, allow_pickle=False).astype(np.float32).reshape(-1)
        assert np.isfinite(pcm).all() and len(pcm) > 24000
        label = cohort + ':' + str(p.relative_to(dest).with_suffix('')).replace('/', ':')
        rate = 24000 if cohort == 'omnivoice' else 48000
        expected = 'The quick brown fox jumps over the lazy dog.' if rate == 24000 else TEXTS[p.stem.split('-')[-1]]
        common = math.gcd(rate, 16000)
        signals = {'python-linear': pcm,
                   'native-file-loader': native_resample(pcm, rate, str(len(cases))),
                   'scipy-polyphase': resample_poly(pcm, 16000 // common, rate // common).astype(np.float32)}
        cases[label] = (rate, expected, signals)
        receipt['cases'][label] = dict(sample_rate=rate, expected=expected,
            source_sha256=hashlib.sha256(pcm.tobytes()).hexdigest(),
            resampled={name: dict(samples=len(signal), sha256=hashlib.sha256(signal.tobytes()).hexdigest())
                       for name, signal in signals.items() if name != 'python-linear'})
save()
assert len(cases) == 26, len(cases)
sys.path.insert(0, str(ROOT / 'python'))
from crispasr import Session


def wer(expected, actual):
    words = lambda text: re.findall('[a-z]+', re.sub(r'<[^>]*>', '', text).lower())
    ref, hyp = words(expected), words(actual)
    row = list(range(len(hyp) + 1))
    for i, word in enumerate(ref, 1):
        nxt = [i]
        for j, candidate in enumerate(hyp, 1):
            nxt.append(min(nxt[-1] + 1, row[j] + 1, row[j - 1] + (word != candidate)))
        row = nxt
    return row[-1] / len(ref)


api = HfApi()
for backend, (repo, revision, filename) in MODELS.items():
    path = hf_hub_download(repo, filename, revision=revision)
    fingerprint = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    info = api.model_info(repo, revision=revision, files_metadata=True)
    remote = next(f for f in info.siblings if f.rfilename == filename)
    assert remote.lfs.sha256 == fingerprint and remote.size == Path(path).stat().st_size
    receipt['models'][backend] = dict(repo=repo, revision=revision, file=filename, sha256=fingerprint)
    with Session(path, lib_path=str(lib_path), backend=backend, n_threads=4) as session:
        for label, (rate, expected, signals) in cases.items():
            for sampler, pcm in signals.items():
                transcript = ' '.join(seg.text for seg in session.transcribe(pcm,
                    sample_rate=rate if sampler == 'python-linear' else 16000, language='en'))
                error = wer(expected, transcript)
                receipt['transcripts'][backend + '/' + sampler + '/' + label] = dict(
                    transcript=transcript, wer=error, existing_gate_met=error <= .2 if rate == 24000 else error == 0)
                save()
                print('FIXED_AUDIO_ASR', backend, sampler, label, error, transcript, flush=True)
assert len(receipt['transcripts']) == 234
receipt['diagnostic_complete'] = True
save()
print('FRONTDOOR_DIAGNOSTIC_COMPLETE', flush=True)
