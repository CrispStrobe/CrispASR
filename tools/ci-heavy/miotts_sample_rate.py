#!/usr/bin/env python3
"""Prove MioTTS model rate through native getter, session ABI, CLI WAV and ASR.

The 24 kHz and missing-key copies test rate metadata dispatch only: they do
not represent valid legacy codec audio. Speech acceptance uses the unmodified,
immutable public 44.1 kHz model and its actual English speaker preset.
"""
import ctypes
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
SCRATCH = Path(os.environ['HEAVY_SCRATCH']) / 'miotts-rate'
OUT.mkdir(parents=True, exist_ok=True)
SCRATCH.mkdir(parents=True, exist_ok=True)
os.environ['TMPDIR'] = str(SCRATCH)
TEXT = 'The quick brown fox jumps over the lazy dog.'
REVISION = 'fab3cdbef802fb5f1047708c27378c6e34ec230f'


def run(command, tag):
    with (OUT / (tag + '.log')).open('w') as log:
        result = subprocess.run(list(map(str, command)), cwd=ROOT, stdout=log,
                                stderr=subprocess.STDOUT, timeout=2400)
    print(tag, result.returncode, (OUT / (tag + '.log')).read_text()[-2500:], flush=True)
    assert result.returncode == 0, tag


build = SCRATCH / 'build'
run(['cmake', '-S', ROOT, '-B', build, '-G', 'Ninja', '-DCMAKE_BUILD_TYPE=Release',
     '-DBUILD_SHARED_LIBS=ON', '-DGGML_NATIVE=OFF', '-DGGML_BLAS=OFF',
     '-DCRISPASR_BUILD_SERVER=OFF', '-DCRISPASR_BUILD_TESTS=ON'], 'configure')
run(['cmake', '--build', build, '--target', 'crispasr-cli', 'crispasr-lib',
     'test-miotts-live', '-j4'], 'build')
from huggingface_hub import hf_hub_download
from gguf import GGUFReader
import numpy as np
sys.path.insert(0, str(ROOT / 'python'))
from crispasr import Session
model_dir = SCRATCH / 'model'
model = Path(hf_hub_download('cstr/miotts-0.6b-GGUF', 'miotts-0.6b-q4_k.gguf',
                           revision=REVISION, local_dir=model_dir))
for name in ('tokenizer.json', 'en_female.emb.gguf'):
    hf_hub_download('cstr/miotts-0.6b-GGUF', name, revision=REVISION, local_dir=model_dir)
reader = GGUFReader(str(model))
field = reader.fields['miotts.codec.sample_rate']
assert field.contents() == 44100
value = field.parts[field.data[0]]
value_offset = value.ctypes.data - reader.data.ctypes.data
key_part = field.parts[1]
key_offset = key_part.ctypes.data - reader.data.ctypes.data
assert bytes(key_part) == b'miotts.codec.sample_rate'
del value, key_part, field, reader
lib = next(build.rglob('libcrispasr.dylib' if sys.platform == 'darwin' else 'libcrispasr.so'))
os.environ['CRISPASR_MODEL_MIOTTS'] = str(model)
run([build / 'bin/test-miotts-live', '[miotts]'], 'native-live')
# Native null context is defined independently of loaded metadata.
abi = ctypes.CDLL(str(lib))
abi.miotts_get_sample_rate.argtypes = [ctypes.c_void_p]
abi.miotts_get_sample_rate.restype = ctypes.c_int
assert abi.miotts_get_sample_rate(None) == 24000
rates = {}
for name, rate in [('public-v2', 44100), ('metadata-24k', 24000), ('metadata-missing', 24000)]:
    path = model if name == 'public-v2' else model_dir / (name + '.gguf')
    if path != model:
        shutil.copyfile(model, path)
        with path.open('r+b') as stream:
            stream.seek(value_offset if name == 'metadata-24k' else key_offset)
            stream.write(np.uint32(24000).tobytes() if name == 'metadata-24k' else b'miotts.codec.sample_rata')
    with Session(str(path), lib_path=str(lib), backend='miotts', n_threads=4) as session:
        rates[name] = session.output_sample_rate()
        assert rates[name] == rate, (name, rates[name], rate)
        if name == 'public-v2':
            session.set_voice(str(model_dir / 'en_female.emb.gguf'))
            session.set_temperature(0, seed=42)
            pcm = session.synthesize(TEXT)
            assert np.isfinite(pcm).all() and len(pcm) / rate > 1
            np.save(OUT / 'session.npy', pcm)
    if path != model:
        path.unlink()
cli_wav = OUT / 'cli.wav'
run([build / 'bin/crispasr', '--backend', 'miotts', '-m', model, '--no-gpu',
     '-t', '4', '--temperature', '0', '--voice', model_dir / 'en_female.emb.gguf',
     '--tts', TEXT, '--tts-output', cli_wav], 'cli-speech')
with wave.open(str(cli_wav)) as wav:
    assert wav.getframerate() == 44100 and wav.getnchannels() == 1 and wav.getsampwidth() == 2
    cli_pcm = np.frombuffer(wav.readframes(wav.getnframes()), dtype='<i2').astype(np.float32) / 32768
    cli_frames = wav.getnframes()
assert np.isfinite(cli_pcm).all() and cli_frames / 44100 > 1
asr = hf_hub_download('cstr/nemotron-3.5-asr-streaming-GGUF', 'nemotron-3.5-asr-streaming-0.6b-q4_k.gguf',
                     revision='bbd95a9ca5fa0dfca3312a122dfc45a2b578b9c2')
results = {}
with Session(asr, lib_path=str(lib), backend='nemotron', n_threads=4) as session:
    for name, pcm in [('session', np.load(OUT / 'session.npy')), ('cli', cli_pcm)]:
        actual = ' '.join(seg.text for seg in session.transcribe(pcm, sample_rate=44100, language='en'))
        words = lambda s: re.findall('[a-z]+', s.lower())
        ref, hyp = words(TEXT), words(actual)
        row = list(range(len(hyp) + 1))
        for i, word in enumerate(ref, 1):
            new = [i]
            for j, got in enumerate(hyp, 1):
                new.append(min(new[-1] + 1, row[j] + 1, row[j - 1] + (word != got)))
            row = new
        results[name] = dict(transcript=actual, wer=row[-1] / len(ref), samples=len(pcm),
                             sample_rate=44100, duration_seconds=len(pcm) / 44100)
        (OUT / 'roundtrips.json').write_text(json.dumps(results, indent=2) + '\n')
        assert results[name]['wer'] <= .2, results[name]
receipt = dict(passed=True, source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
               model_revision=REVISION, metadata_rates=rates, roundtrips=results,
               metadata_copy_scope='dispatch only; no legacy codec speech claim', platform=sys.platform)
(OUT / 'acceptance.json').write_text(json.dumps(receipt, indent=2) + '\n')
(OUT / 'summary.md').write_text('MioTTS native rate, session ABI metadata dispatch, 44.1 kHz CLI WAV and both speech roundtrips PASS.\n')
print('MIOTTS_RATE_AND_SPEECH_PASS', flush=True)
