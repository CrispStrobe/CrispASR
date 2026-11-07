#!/usr/bin/env python3
"""Compare RE-USE outputs for two captured phase inputs (diagnostic only).

This tool never treats canonicalized phase as an upstream-parity pass. Inputs
are an NPZ with audio, local_phase, and kaggle_phase; both phases must originate
from the same 16 kHz audio. Model assets must be the pinned original revision.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

HASHES = {
    'config.json': 'e69986483774f01893d2727503addc49ef201d19e6e1a266b8d4e971e28c478d',
    'model.safetensors': '06badaf4029b297f97f7fe48d62990b48d5482608210df577354e0d4e0ed70e1',
}


def canonical_first_frame(phase, tolerance=1e-4):
    """Diagnostic candidate: choose +pi near either branch cut in frame 0."""
    value = np.asarray(phase, dtype=np.float32)
    if value.ndim != 2 or not value.size or not np.isfinite(value).all() or not 0 < tolerance < 0.01:
        raise ValueError('invalid finite phase matrix or tolerance')
    result = value.copy()
    mask = np.abs(np.abs(result[0]) - np.pi) < tolerance
    result[0, mask] = np.float32(np.pi)
    return result


def compare(a, b):
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('waveforms must have equal shape and finite values')
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na <= 1e-6 or nb <= 1e-6:
        raise ValueError('waveforms must be non-silent')
    return {'relative_rmse': float(np.linalg.norm(a-b)/na), 'norm_ratio': float(nb/na),
            'cosine': float(np.dot(a,b)/(na*nb)), 'max_abs': float(np.max(np.abs(a-b)))}


def mutually_compatible(metrics, tolerance):
    """Necessary bound for one waveform to pass both relative-error gates."""
    return metrics['relative_rmse'] <= tolerance * (1.0 + metrics['norm_ratio'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir', required=True, type=Path)
    parser.add_argument('--input-npz', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    for name, expected in HASHES.items():
        digest = hashlib.sha256()
        with (args.model_dir/name).open('rb') as stream:
            for block in iter(lambda: stream.read(1024*1024), b''):
                digest.update(block)
        if digest.hexdigest() != expected:
            raise ValueError(f'invalid pinned RE-USE {name} digest')
    with np.load(args.input_npz, allow_pickle=False) as data:
        audio = np.asarray(data['audio'], dtype=np.float32)
        first = np.asarray(data['local_phase'], dtype=np.float32)
        second = np.asarray(data['kaggle_phase'], dtype=np.float32)
    if audio.shape != (1600,) or first.shape != (21,321) or second.shape != first.shape or not np.isfinite(audio).all():
        raise ValueError('expected 1600 finite audio samples and two phase matrices of shape (21, 321)')
    first_stable, second_stable = canonical_first_frame(first), canonical_first_frame(second)
    import torch
    from safetensors.torch import load_file
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    sys.path.insert(1, str(args.model_dir.resolve()))
    from models.generator_SEMamba_time_d4 import SEMamba
    from models.stfts import mag_phase_stft, mag_phase_istft
    torch.set_num_threads(2)
    torch.set_grad_enabled(False)
    model = SEMamba(json.loads((args.model_dir/'config.json').read_text())).eval()
    model.load_state_dict(load_file(str(args.model_dir/'model.safetensors')), strict=True)
    magnitude, _, _ = mag_phase_stft(torch.from_numpy(audio[None]), 640, 80, 640, 'relu_log1p', addeps=False)
    def forward(phase):
        amplitude, output_phase, _ = model(magnitude, torch.from_numpy(np.ascontiguousarray(phase.T))[None])
        zeros = (torch.expm1(torch.relu(amplitude)) == 0).sum(1) / amplitude.shape[1]
        amplitude[:, :, (zeros > 0.5)[0]] = 0
        return mag_phase_istft(amplitude, output_phase, 640, 80, 640, 'relu_log1p').numpy()[0]
    a, b, ca, cb = map(forward, (first, second, first_stable, second_stable))
    upstream = compare(a,b)
    result = {'input_sha256': hashlib.sha256(audio.tobytes()).hexdigest(),
              'upstream_first_frame_branch_flips': int(np.count_nonzero(np.abs(first[0]-second[0]) > 1)),
              'upstream_waveform': upstream,
              'could_both_pass_0_5_percent': mutually_compatible(upstream, 0.005),
              'candidate_local_vs_original': compare(a,ca), 'candidate_kaggle_vs_original': compare(b,cb),
              'candidate_cross_reference': compare(ca,cb),
              'candidate_note': 'Experimental diagnostic only; not an upstream parity pass.'}
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print('REUSE_PHASE_STABILITY', json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
