"""RE-USE upstream network with a non-fused Mamba-1 CPU reference adapter."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly
from safetensors.torch import load_file

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(1, str(Path('reuse-model').resolve()))
from models.generator_SEMamba_time_d4 import SEMamba
from models.stfts import mag_phase_stft, mag_phase_istft
from utils.util import pad_or_trim_to_match

torch.set_num_threads(2)
torch.set_grad_enabled(False)
root = Path('reuse-model')
config = json.loads((root / 'config.json').read_text())
model = SEMamba(config)
model.load_state_dict(load_file(str(root / 'model.safetensors')), strict=True)
model.eval()
audio, source_rate = sf.read(root / 'noisy_audio/mic_test2.wav', dtype='float32', always_2d=True)
assert audio.shape[1] == 1
from math import gcd
factor = gcd(source_rate, 8000)
waveform = resample_poly(audio[:, 0], 8000 // factor, source_rate // factor).astype('float32')[:4000]
sf.write('reuse-input.wav', waveform, 8000, subtype='FLOAT')
t = torch.from_numpy(waveform[None])
stft = config['stft_cfg']
n_fft, hop, win = [int(stft[k] * 8000 // stft['sampling_rate']) for k in ('n_fft','hop_size','win_size')]
n_fft, hop, win = [n + n % 2 for n in (n_fft, hop, win)]
compression = config['model_cfg']['compress_factor']
captures = {}
def hook(name):
    def record(_m, _i, out):
        values = out if isinstance(out, (tuple,list)) else [out]
        for i, value in enumerate(values):
            if isinstance(value, torch.Tensor): captures[f'{name}_{i}'] = value.float().numpy()
    return record
for i in (0, 14, 29): model.TSMamba[i].register_forward_hook(hook(f'tsmamba_{i}'))
model.dense_encoder.register_forward_hook(hook('encoder'))
mag, phase, _ = mag_phase_stft(t, n_fft, hop, win, compression, addeps=False)
amp, phase, _ = model(mag, phase)
zeros = (torch.expm1(torch.relu(amp)) == 0).sum(1) / amp.shape[1]
amp[:, :, (zeros > 0.5)[0]] = 0
result = pad_or_trim_to_match(t, mag_phase_istft(amp, phase, n_fft, hop, win, compression), pad_value=1e-8)
out = result.numpy()[0]
assert np.isfinite(out).all() and np.sqrt(np.mean(out ** 2)) > 1e-6
sf.write('reuse-reference.wav', out, 8000, subtype='FLOAT')
np.savez_compressed('reuse-stages.npz', **captures)
print('CPU_REFERENCE_PASS', json.dumps({'samples':len(out), 'rms':float(np.sqrt(np.mean(out**2))),
    'model_sha256':hashlib.sha256((root/'model.safetensors').read_bytes()).hexdigest(),
    'stage_shapes':{k:list(v.shape) for k,v in captures.items()}}), flush=True)
