#!/usr/bin/env python3
"""Pinned official full generation and native precision controls; diagnostic only."""
import gc
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

import numpy as np
from huggingface_hub import hf_hub_download, snapshot_download

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(os.environ['HEAVY_OUT'])
TEMP = Path(os.environ['HEAVY_SCRATCH']) / 'voxcpm2-original-controls'
OUT.mkdir(parents=True, exist_ok=True)
TEMP.mkdir(parents=True, exist_ok=True)
UPSTREAM = 'cce58a5b59303c9bd63d12f23afe6a49a4a80c59'
MODEL = '32279effe8c19989596f05d353d1447f51d9e915'
GGUF = '25b5cf03fdbf20011dad9a77def6112023cc0fe3'
TEXTS = {'short': 'Hello, this is a short test sentence.',
         'long': 'The sun is shining today. Hello, this is a short test sentence.'}
receipt = dict(passed=False, diagnostic_complete=False, scope=__doc__, seed=2, steps=10,
               cfg=2.0, upstream_source=UPSTREAM, upstream_model=MODEL, gguf_revision=GGUF,
               native_source=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
               generation={}, roundtrips={}, comparisons={})


def save():
    (OUT / 'original-controls.json').write_text(json.dumps(receipt, indent=2) + '\n')


def run(command, tag):
    with (OUT / (tag + '.log')).open('w') as log:
        subprocess.run(list(map(str, command)), cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                       check=True, timeout=7200)


def record(label, key, pcm, seconds):
    pcm = np.asarray(pcm, dtype=np.float32).reshape(-1)
    assert len(pcm) > 24000 and np.isfinite(pcm).all()
    np.save(OUT / (label + '-' + key + '.npy'), pcm)
    receipt['generation'][label + '/' + key] = dict(samples=len(pcm), sample_rate=48000,
        sha256=hashlib.sha256(pcm.tobytes()).hexdigest(), seconds=seconds,
        norm=float(np.linalg.norm(pcm.astype(np.float64))), peak=float(np.max(np.abs(pcm))))
    save()


save()
source = TEMP / 'upstream'
run(['git', 'init', source], 'upstream-init')
run(['git', '-C', source, 'remote', 'add', 'origin', 'https://github.com/OpenBMB/VoxCPM.git'], 'upstream-origin')
run(['git', '-C', source, 'fetch', '--depth', '1', 'origin', UPSTREAM], 'upstream-fetch')
run(['git', '-C', source, 'checkout', '--detach', 'FETCH_HEAD'], 'upstream-checkout')
assert subprocess.check_output(['git', '-C', source, 'rev-parse', 'HEAD'], text=True).strip() == UPSTREAM
sys.path.insert(0, str(source / 'src'))
import torch
from voxcpm.model.voxcpm2 import VoxCPM2Model

torch.set_num_threads(4)
torch.set_num_interop_threads(1)
receipt['torch'] = torch.__version__
model_dir = snapshot_download('openbmb/VoxCPM2', revision=MODEL, local_dir=TEMP / 'original',
    allow_patterns=['*.json', '*.safetensors', 'audiovae.pth', 'tokenizer.model'])
model = VoxCPM2Model.from_local(model_dir, optimize=False, device='cpu')
assert model.sample_rate == 48000 and model.config.dtype == 'bfloat16'
receipt['python_parameter_dtype'] = str(next(model.base_lm.parameters()).dtype)
assert next(model.base_lm.parameters()).dtype == torch.bfloat16
decode = model.audio_vae.decode
captured = {}


def capture(latent):
    captured['latent'] = latent.detach().cpu().float().numpy().copy()
    return decode(latent)


model.audio_vae.decode = capture
for key, text in TEXTS.items():
    previous = None
    previous_latent = None
    for repeat in range(2):
        start = time.monotonic()
        # Explicit seed avoids the upstream default which materializes a new seed.
        pcm = model.generate(target_text=text, seed=2, inference_timesteps=10,
                             cfg_value=2.0, retry_badcase=False).float().numpy().reshape(-1).copy()
        elapsed = time.monotonic() - start
        latent = captured['latent'].copy()
        assert np.isfinite(latent).all()
        if previous is not None:
            assert np.array_equal(previous, pcm) and np.array_equal(previous_latent, latent), 'Original seeded repeat drift'
        previous, previous_latent = pcm, latent
        print('ORIGINAL_GENERATED', key, repeat, len(pcm), elapsed, flush=True)
    np.save(OUT / ('python-bf16-' + key + '-latent.npy'), previous_latent)
    record('python-bf16', key, previous, elapsed)
del captured, previous, previous_latent, latent, pcm, model, decode
gc.collect()

build = TEMP / 'build'
run(['cmake', '-S', ROOT, '-B', build, '-G', 'Ninja', '-DCMAKE_BUILD_TYPE=Release',
     '-DBUILD_SHARED_LIBS=ON', '-DGGML_NATIVE=OFF', '-DGGML_CUDA=OFF',
     '-DCRISPASR_BUILD_TESTS=OFF', '-DCRISPASR_BUILD_SERVER=OFF'], 'configure')
run(['cmake', '--build', build, '--target', 'crispasr-lib', '-j4'], 'build')
sys.path.insert(0, str(ROOT / 'python'))
from crispasr import Session

lib = next(build.rglob('libcrispasr.so'))
os.environ.update(CRISPASR_VOXCPM2_INFERENCE_STEPS='10', CRISPASR_VOXCPM2_BENCH='1')
for label, filename in [('native-f16', 'voxcpm2-f16.gguf'), ('native-q8', 'voxcpm2-q8_0.gguf')]:
    path = hf_hub_download('cstr/voxcpm2-GGUF', filename, revision=GGUF, local_dir=TEMP / 'native')
    with Session(path, lib_path=str(lib), backend='voxcpm2', n_threads=4) as session:
        session.set_tts_steps(10)
        assert session._lib.crispasr_session_set_tts_steps(session._handle, 10) == 0
        session.accept_marking_responsibility('Matched original full-generation diagnostic')
        assert session.output_sample_rate() == 48000
        for key, text in TEXTS.items():
            previous = None
            for repeat in range(2):
                session.set_tts_seed(2)
                start = time.monotonic()
                pcm = np.asarray(session.synthesize_raw(text), dtype=np.float32)
                elapsed = time.monotonic() - start
                if previous is not None:
                    assert np.array_equal(previous, pcm), 'Native seeded repeat drift'
                previous = pcm.copy()
            record(label, key, previous, elapsed)
    gc.collect()

asr = hf_hub_download('cstr/nemotron-3.5-asr-streaming-GGUF',
    'nemotron-3.5-asr-streaming-0.6b-q4_k.gguf', revision='bbd95a9ca5fa0dfca3312a122dfc45a2b578b9c2')
with Session(asr, lib_path=str(lib), backend='nemotron', n_threads=4) as session:
    for label in ('python-bf16', 'native-f16', 'native-q8'):
        for key, expected in TEXTS.items():
            pcm = np.load(OUT / (label + '-' + key + '.npy'))
            transcript = ' '.join(seg.text for seg in session.transcribe(pcm, sample_rate=48000, language='en'))
            normalize = lambda value: re.findall('[a-z]+', value.lower())
            receipt['roundtrips'][label + '/' + key] = dict(transcript=transcript, expected=expected,
                                                          exact=normalize(transcript) == normalize(expected))
            baseline = np.load(OUT / ('python-bf16-' + key + '.npy')).astype(np.float64)
            actual = pcm.astype(np.float64)
            comparison = dict(samples=len(actual), reference_samples=len(baseline), aligned=False)
            if actual.shape == baseline.shape:
                comparison.update(cosine=float(np.dot(actual, baseline) / (np.linalg.norm(actual) * np.linalg.norm(baseline))),
                    relative_l2=float(np.linalg.norm(actual - baseline) / np.linalg.norm(baseline)),
                    mine_norm=float(np.linalg.norm(actual)), ref_norm=float(np.linalg.norm(baseline)))
            receipt['comparisons'][label + '/' + key] = comparison
            save()
assert len(receipt['roundtrips']) == 6
receipt['diagnostic_complete'] = True
save()
print('ORIGINAL_CONTROLS_COMPLETE', receipt['roundtrips'], flush=True)
