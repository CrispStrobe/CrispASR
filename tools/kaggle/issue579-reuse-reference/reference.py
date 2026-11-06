"""Generate independent upstream RE-USE GPU reference and selected activations."""
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

WORK = Path('/kaggle/working')
REV = '31f82f2efad1095034492f236e8dac9239ba713b'
REPO = 'nvidia/NVIDIA-NemotronLabs-RE-USE'

def run(cmd):
    print('+', ' '.join(str(x) for x in cmd), flush=True)
    subprocess.run(cmd, check=True)

run(['nvidia-smi'])
run([sys.executable, '-m', 'pip', 'install', 'huggingface_hub', 'safetensors', 'einops', 'soundfile', 'ninja', 'packaging'])
try:
    import mamba_ssm
    import causal_conv1d
except ImportError:
    run([sys.executable, '-m', 'pip', 'install', '--no-build-isolation', 'causal-conv1d', 'mamba-ssm'])
from huggingface_hub import snapshot_download
root = Path(snapshot_download(REPO, revision=REV, local_dir='/kaggle/temp/reuse',
                              allow_patterns=['*.json', '*.py', '*.safetensors', 'models/*', 'utils/*', 'noisy_audio/mic_test2.wav']))
sys.path.insert(0, str(root))
import numpy as np
import soundfile as sf
import torch
from safetensors.torch import load_file
from models.generator_SEMamba_time_d4 import SEMamba
from models.stfts import mag_phase_stft, mag_phase_istft
from utils.util import pad_or_trim_to_match

assert torch.cuda.is_available(), 'CUDA device required'
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.set_grad_enabled(False)
config = json.loads((root / 'config.json').read_text())
model = SEMamba(config)
model.load_state_dict(load_file(str(root / 'model.safetensors')), strict=True)
model = model.cuda().eval()
audio, rate = sf.read(root / 'noisy_audio/mic_test2.wav', dtype='float32', always_2d=True)
waveform = torch.from_numpy(audio.T.copy()).cuda()
assert waveform.shape[0] == 1
stft = config['stft_cfg']
sizes = [int(stft[key] * rate // stft['sampling_rate']) for key in ('n_fft','hop_size','win_size')]
n_fft, hop, win = [n + n % 2 for n in sizes]
compression = config['model_cfg']['compress_factor']
captured = {}
def hook(name):
    def save(_module, _inputs, output):
        values = output if isinstance(output, (tuple, list)) else [output]
        for i, value in enumerate(values):
            if isinstance(value, torch.Tensor):
                captured[f'{name}_{i}'] = value.detach().float().cpu().numpy()
    return save
handles = [model.dense_encoder.register_forward_hook(hook('encoder'))]
for i in (0, 14, 29):
    handles.append(model.TSMamba[i].register_forward_hook(hook(f'tsmamba_{i}')))
for name in ('mask_decoder','phase_decoder'):
    handles.append(getattr(model, name).register_forward_hook(hook(name)))
mag, phase, _ = mag_phase_stft(waveform, n_fft, hop, win, compression, addeps=False)
captured['input_magnitude'] = mag.float().cpu().numpy()
captured['input_phase'] = phase.float().cpu().numpy()
amp, phase, _ = model(mag, phase)
zeros = (torch.expm1(torch.relu(amp)) == 0).sum(1) / amp.shape[1]
amp[:, :, (zeros > 0.5)[0]] = 0
enhanced = mag_phase_istft(amp, phase, n_fft, hop, win, compression)
enhanced = pad_or_trim_to_match(waveform, enhanced, pad_value=1e-8)
output = enhanced.float().cpu().numpy()[0]
assert np.isfinite(output).all() and np.sqrt(np.mean(output ** 2)) > 1e-6
sf.write(WORK / 'reference.wav', output, rate, subtype='FLOAT')
np.savez_compressed(WORK / 'stages.npz', **captured)
def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
report = {'revision': REV, 'rate': rate, 'samples': len(output), 'output_rms': float(np.sqrt(np.mean(output ** 2))),
          'model_sha256': digest(root / 'model.safetensors'), 'audio_sha256': digest(root / 'noisy_audio/mic_test2.wav'),
          'reference_sha256': digest(WORK / 'reference.wav'), 'stage_names': sorted(captured),
          'stage_shapes': {k: list(v.shape) for k,v in captured.items()}, 'device': torch.cuda.get_device_name(0)}
(WORK / 'report.json').write_text(json.dumps(report, indent=2))
print('REFERENCE_PASS', json.dumps(report), flush=True)
