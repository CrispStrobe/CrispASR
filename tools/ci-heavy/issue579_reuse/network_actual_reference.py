#!/usr/bin/env python3
"""Pinned original RE-USE CPU reference for a short public microphone excerpt."""
import json
from math import gcd
from pathlib import Path
import sys
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
import torch
from safetensors.torch import load_file

torch.set_num_threads(2)
torch.set_grad_enabled(False)
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(1, str(Path('reuse-model').resolve()))
from models.generator_SEMamba_time_d4 import SEMamba
from models.stfts import mag_phase_stft, mag_phase_istft

root = Path('reuse-model')
config = json.loads((root / 'config.json').read_text())
audio, source_rate = sf.read(root / 'noisy_audio/mic_test2.wav', dtype='float32', always_2d=True)
assert audio.shape[1] == 1
factor = gcd(source_rate, 8000)
audio = resample_poly(audio[:, 0], 8000 // factor, source_rate // factor).astype(np.float32)
length = 800
assert len(audio) >= length
positions = range(0, len(audio) - length + 1, length // 2)
start = max(positions, key=lambda offset: float(np.sum(audio[offset:offset + length] ** 2)))
waveform = np.ascontiguousarray(audio[start:start + length])
assert np.isfinite(waveform).all() and np.linalg.norm(waveform) > 1e-5
waveform.astype('<f4').tofile('reuse-network-input.f32le')
model = SEMamba(config).eval()
model.load_state_dict(load_file(str(root / 'model.safetensors')), strict=True)
stages = {}
def hook(name):
    def capture(_module, _args, result):
        stages[name] = result.detach().float().cpu().contiguous().numpy()
    return capture
model.dense_encoder.register_forward_hook(hook('encoder'))
for index in (0, 14, 29):
    model.TSMamba[index].register_forward_hook(hook('block' + str(index)))
t = torch.from_numpy(waveform[None])
mag, phase, _ = mag_phase_stft(t, 320, 40, 320, 'relu_log1p', addeps=False)
amplitude, phase, _ = model(mag, phase)
zero_fraction = (torch.expm1(torch.relu(amplitude)) == 0).sum(1) / amplitude.shape[1]
amplitude[:, :, (zero_fraction > 0.5)[0]] = 0
result = mag_phase_istft(amplitude, phase, 320, 40, 320, 'relu_log1p').numpy()[0]
assert len(result) == length and np.isfinite(result).all() and np.linalg.norm(result) > 1e-6
result.astype('<f4').tofile('reuse-network-expected.f32le')
np.savez_compressed('reuse-network-reference-stages.npz', **stages)
print('REUSE_NETWORK_REFERENCE', json.dumps({'samples':length, 'start_sample':start,
      'output_rms':float(np.sqrt(np.mean(result ** 2))), 'stages':{k:list(v.shape) for k,v in stages.items()}}), flush=True)
