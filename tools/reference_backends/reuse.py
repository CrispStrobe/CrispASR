"""Original NVIDIA RE-USE per-stage reference for crispasr-diff.

The CPU non-fused Mamba adapter is the independently checked audit adapter.
Archives contain activations for a bounded public input, never model weights.
"""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

DEFAULT_STAGES = (['raw_audio', 'frontend_magnitude', 'frontend_phase'] +
                  ['frontend_window', 'frontend_real', 'frontend_imaginary'] +
                  [f'encoder_layer_{i}' for i in range(6)] +
                  [name for i in range(30) for name in (f'block_time_{i}', f'block_{i}')] +
                  ['amplitude', 'output_phase', 'waveform'])


def capture_frontend(torch, mag_phase_stft, audio, captures):
    """Observe the original STFT call, including its actual generated window.

    Reconstructing complex values from phase would lose signed zeros and the
    small imaginary residuals that determine which side of atan2's cut wins.
    The observer calls the original function once and returns its exact result.
    """
    from unittest.mock import patch
    original = torch.stft

    def observe(*args, **kwargs):
        result = original(*args, **kwargs)
        captures['frontend_window'] = kwargs['window'].detach().float().cpu().numpy().copy()
        captures['frontend_real'] = result[0].real.T.detach().float().cpu().contiguous().numpy().copy()
        captures['frontend_imaginary'] = result[0].imag.T.detach().float().cpu().contiguous().numpy().copy()
        return result

    with patch.object(torch, 'stft', observe):
        return mag_phase_stft(torch.from_numpy(audio[None]), 640, 80, 640, 'relu_log1p', addeps=False)


def _check_file(path, expected):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    if digest.hexdigest() != expected:
        raise ValueError(f'Unexpected pinned RE-USE source digest for {path.name}')


def prepare_audio(audio):
    """Return a bounded 16 kHz diagnostic input without silently padding it."""
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim != 1:
        raise ValueError('RE-USE reference requires mono audio')
    if len(audio) > 16000:
        raise ValueError("RE-USE reference input exceeds the 16000-sample diagnostic limit; select an explicit window")
    audio = np.ascontiguousarray(audio)
    if len(audio) <= 320 or len(audio) % 80 or not np.isfinite(audio).all():
        raise ValueError('RE-USE reference requires finite 16 kHz mono audio longer than 320 samples and a multiple of the 80-sample hop')
    return audio


def dump(model_dir, audio, stages, **_kwargs):
    unknown = set(stages) - set(DEFAULT_STAGES)
    if unknown:
        raise ValueError(f'Unknown RE-USE reference stages: {sorted(unknown)}')
    audio = prepare_audio(audio)
    import torch
    from safetensors.torch import load_file

    model_dir = Path(model_dir).resolve()
    _check_file(model_dir / 'config.json', 'e69986483774f01893d2727503addc49ef201d19e6e1a266b8d4e971e28c478d')
    _check_file(model_dir / 'model.safetensors', '06badaf4029b297f97f7fe48d62990b48d5482608210df577354e0d4e0ed70e1')
    # The reference's package imports mamba_ssm; use the audited CPU adapter.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'ci-heavy' / 'issue579_reuse'))
    sys.path.insert(1, str(model_dir))
    from models.generator_SEMamba_time_d4 import SEMamba
    from models.stfts import mag_phase_stft, mag_phase_istft

    torch.set_num_threads(2)
    torch.set_grad_enabled(False)
    model = SEMamba(json.loads((model_dir / 'config.json').read_text())).eval()
    model.load_state_dict(load_file(str(model_dir / 'model.safetensors')), strict=True)
    captures = {'raw_audio': audio.copy()}

    def hook(name):
        def capture(_module, _args, result):
            if name in stages:
                captures[name] = result[0].detach().float().cpu().contiguous().numpy().copy()
        return capture

    enc = model.dense_encoder
    layers = [enc.dense_conv_1, *enc.dense_block.dense_block, enc.dense_conv_2]
    for i, layer in enumerate(layers):
        layer.register_forward_hook(hook(f'encoder_layer_{i}'))
    def time_hook(index):
        def capture(_module, args):
            name = f'block_time_{index}'
            if name in stages:
                # freq_mamba sees [time, freq, channel] for batch size one.
                captures[name] = args[0].permute(2, 0, 1).detach().float().cpu().contiguous().numpy().copy()
        return capture

    for i, block in enumerate(model.TSMamba):
        block.freq_mamba.register_forward_pre_hook(time_hook(i))
        block.register_forward_hook(hook(f'block_{i}'))
    if any(name in stages for name in ('frontend_window', 'frontend_real', 'frontend_imaginary')):
        mag, phase, _ = capture_frontend(torch, mag_phase_stft, audio, captures)
    else:
        mag, phase, _ = mag_phase_stft(torch.from_numpy(audio[None]), 640, 80, 640, 'relu_log1p', addeps=False)
    captures['frontend_magnitude'] = mag[0].T.contiguous().numpy().copy()
    captures['frontend_phase'] = phase[0].T.contiguous().numpy().copy()
    amp, output_phase, _ = model(mag, phase)
    captures['amplitude'] = amp[0].T.contiguous().numpy().copy()
    captures['output_phase'] = output_phase[0].T.contiguous().numpy().copy()
    zero_fraction = (torch.expm1(torch.relu(amp)) == 0).sum(1) / amp.shape[1]
    amp[:, :, (zero_fraction > 0.5)[0]] = 0
    captures['waveform'] = mag_phase_istft(amp, output_phase, 640, 80, 640, 'relu_log1p').numpy()[0].copy()
    if len(captures['waveform']) != len(audio):
        raise ValueError('RE-USE input length must be a multiple of the 80-sample hop')
    missing = set(stages) - captures.keys()
    if missing:
        raise ValueError(f'Missing RE-USE reference stages: {sorted(missing)}')
    return {name: captures[name] for name in stages}
