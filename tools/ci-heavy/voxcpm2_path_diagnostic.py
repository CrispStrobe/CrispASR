#!/usr/bin/env python3
"""Isolate existing GPU CFM paths; diagnostic completion is not speech acceptance."""
import ctypes as C
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

import numpy as np
from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(os.environ['HEAVY_OUT'])
BUILD = Path(os.environ['VOX_PROFILE_BUILD'])
TEMP = Path(os.environ['HEAVY_SCRATCH'])
PIN = '25b5cf03fdbf20011dad9a77def6112023cc0fe3'
TEXTS = {'short': 'Hello, this is a short test sentence.',
         'long': 'The sun is shining today. Hello, this is a short test sentence.'}
ARMS = {'vulkan-fused': ('vulkan', '1', '1'),
        'vulkan-step-batch': ('vulkan', '0', '1'),
        'vulkan-step-split': ('vulkan', '0', '0'),
        'cuda-fused': ('cuda', '1', '1')}
sys.path.insert(0, str(ROOT / 'python'))
from crispasr import Session


class Params(C.Structure):
    _fields_ = [(n, C.c_int) for n in
               ('abi_version', 'n_threads', 'use_gpu', 'verbosity', 'flash_attn', 'n_gpu_layers')] + [('reserved', C.c_int * 6)]


def session(model, backend, device):
    s = Session.__new__(Session)
    s._lib = C.CDLL(str(next(BUILD.rglob('libcrispasr.so'))))
    s._handle, s._progress_cb_holder = None, None
    s._setup_session_signatures()
    s._lib.crispasr_set_gpu_backend.argtypes = [C.c_char_p]
    s._lib.crispasr_set_gpu_backend(device.encode())
    p = Params(2, 4, 1, 1, 1, -1)
    s._handle = s._lib.crispasr_session_open_with_params(os.fsencode(model), backend.encode(), C.byref(p))
    assert s._handle
    s.backend, s._n_threads = backend, 4
    return s


if '--arm' in sys.argv:
    arm = sys.argv[sys.argv.index('--arm') + 1]
    device, fused, batch = ARMS[arm]
    os.environ.update(CRISPASR_VOXCPM2_CFM_FUSED=fused, CRISPASR_VOXCPM2_CFG_BATCH=batch,
                      CRISPASR_VOXCPM2_INFERENCE_STEPS='10', CRISPASR_VOXCPM2_BENCH='1')
    rows = {}
    for cohort, name in [('q8', 'voxcpm2-q8_0.gguf'), ('mixed', 'voxcpm2-q8_0-locdit-f16.gguf')]:
        model = hf_hub_download('cstr/voxcpm2-GGUF', name, revision=PIN, local_dir=TEMP / 'models')
        with session(model, 'voxcpm2', device) as s:
            s.accept_marking_responsibility('GPU path diagnostic, private raw waveform comparison')
            # Exercise a changed solver/cache key, then restore the ten-step recipe.
            assert s._lib.crispasr_session_set_tts_steps(s._handle, 11) == 0
            assert s._lib.crispasr_session_set_tts_steps(s._handle, 10) == 0
            for key, text in TEXTS.items():
                original = None
                for rep in range(2):
                    s.set_tts_seed(2)
                    pcm = np.asarray(s.synthesize_raw(text), dtype=np.float32)
                    assert s.output_sample_rate() == 48000 and len(pcm) > 24000
                    assert np.isfinite(pcm).all() and np.max(np.abs(pcm)) > .01
                    if original is not None:
                        assert np.array_equal(original, pcm), (arm, cohort, key, 'seeded repeat drift')
                    original = pcm.copy()
                np.save(OUT / (cohort + '-' + key + '.npy'), original)
                rows[cohort + '-' + key] = dict(samples=len(original), sample_rate=48000,
                    sha256=hashlib.sha256(original.tobytes()).hexdigest())
    (OUT / 'arm.json').write_text(json.dumps(dict(arm=arm, device=device, fused=fused, batch=batch, outputs=rows), indent=2) + '\n')
    raise SystemExit(0)

receipt = dict(source=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
               model_revision=PIN, scope=__doc__, steps=10, seed=2, passed=False,
               diagnostic_complete=False, defaults_changed=False, arms={}, roundtrips={}, comparisons={})


def save():
    (OUT / 'voxcpm2-vulkan-profile.json').write_text(json.dumps(receipt, indent=2) + '\n')


save()
for arm, (device, _, _) in ARMS.items():
    dest = OUT / arm
    dest.mkdir(exist_ok=True)
    with (OUT / (arm + '.log')).open('w') as log:
        subprocess.run([sys.executable, __file__, '--arm', arm], cwd=ROOT,
            env=dict(os.environ, HEAVY_OUT=str(dest)), stdout=log, stderr=subprocess.STDOUT,
            check=True, timeout=3600)
    trace = (OUT / (arm + '.log')).read_text()
    assert 'voxcpm2: backend = ' + ('Vulkan' if device == 'vulkan' else 'CUDA') in trace
    assert set(re.findall(r'voxcpm2\[bench\]: cfm.steps=(\d+)', trace)) == {'10'}
    assert 'falling back to CPU vae_decode' not in trace
    receipt['arms'][arm] = json.loads((dest / 'arm.json').read_text())
    save()

asr = hf_hub_download('cstr/nemotron-3.5-asr-streaming-GGUF', 'nemotron-3.5-asr-streaming-0.6b-q4_k.gguf',
                      revision='bbd95a9ca5fa0dfca3312a122dfc45a2b578b9c2')
with session(asr, 'nemotron', 'cuda') as s:
    for arm in ARMS:
        for key in receipt['arms'][arm]['outputs']:
            pcm = np.load(OUT / arm / (key + '.npy'))
            text = ' '.join(seg.text for seg in s.transcribe(pcm, sample_rate=48000, language='en'))
            expected = TEXTS[key.split('-')[-1]]
            normalize = lambda t: re.findall('[a-z]+', t.lower())
            receipt['roundtrips'][arm + '/' + key] = dict(transcript=text, expected=expected,
                                                        exact=normalize(text) == normalize(expected))
            baseline = np.load(OUT / 'vulkan-fused' / (key + '.npy')).astype(np.float64)
            actual = pcm.astype(np.float64)
            row = dict(baseline_samples=len(baseline), samples=len(actual), exact=np.array_equal(baseline, actual))
            if actual.shape == baseline.shape:
                row.update(relative_l2=float(np.linalg.norm(actual - baseline) / np.linalg.norm(baseline)),
                    mine_norm=float(np.linalg.norm(actual)), ref_norm=float(np.linalg.norm(baseline)),
                    cosine=float(np.dot(actual, baseline) / (np.linalg.norm(actual) * np.linalg.norm(baseline))))
            receipt['comparisons'][arm + '/' + key] = row
            save()
assert len(receipt['roundtrips']) == 16
receipt['diagnostic_complete'] = True
receipt['speech_exact_by_arm'] = {arm: all(receipt['roundtrips'][arm + '/' + key]['exact']
    for key in receipt['arms'][arm]['outputs']) for arm in ARMS}
save()
print('VOX_GPU_PATH_DIAGNOSTIC_COMPLETE', receipt['speech_exact_by_arm'], flush=True)
