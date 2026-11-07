#!/usr/bin/env python3
"""Capture/replay the original 16 kHz STFT across CPU environments.

Diagnostic only: no phase canonicalization and no model acceptance decision.
Run --audio input.wav --output-npz local.npz --report local.json on each
environment, then repeat with --compare-npz remote.npz to isolate window/FFT
differences on byte-identical audio. Optional --native-f32 consumes the probe
layout: window[640], then real, imaginary, log-magnitude, phase[frames,321].
"""
import argparse
import hashlib
import json
from pathlib import Path
import wave

import numpy as np


def metrics(actual, expected):
    actual, expected = np.asarray(actual), np.asarray(expected)
    if actual.shape != expected.shape or not actual.size or not np.isfinite(actual).all() or not np.isfinite(expected).all():
        raise ValueError('expected equal nonempty finite arrays')
    delta = actual.astype(np.float64) - expected
    return {'differing': int(np.count_nonzero(actual != expected)),
            'max_abs': float(np.max(np.abs(delta))),
            'relative_rmse': float(np.linalg.norm(delta) / max(np.linalg.norm(expected.astype(np.float64)), 1e-30))}


def branches(actual, expected):
    actual, expected = np.asarray(actual), np.asarray(expected)
    metrics(actual, expected)
    if actual.ndim != 2:
        raise ValueError('expected time-major phase matrices')
    flips = np.abs(actual.astype(np.float64) - expected) > 1
    return {'all': int(np.count_nonzero(flips)), 'first_frame': int(np.count_nonzero(flips[0]))}


def validate(capture, audio):
    if (capture['audio'].shape != audio.shape or capture['audio'].dtype != np.float32 or
            capture['audio'].tobytes() != audio.tobytes()):
        raise ValueError('capture audio must match exact float32 input bytes')
    frames = 1 + len(audio) // 80
    shapes = [('window', (640,)), ('phase', (frames, 321)), ('first_complex', (321, 2))]
    shapes += [(name, (frames, 321)) for name in ('real', 'imaginary') if name in capture]
    for name, shape in shapes:
        value = capture[name]
        if value.shape != shape or value.dtype != np.float32 or not np.isfinite(value).all():
            raise ValueError(f'invalid capture: {name}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audio', required=True, type=Path)
    parser.add_argument('--output-npz', required=True, type=Path)
    parser.add_argument('--report', required=True, type=Path)
    parser.add_argument('--compare-npz', type=Path)
    parser.add_argument('--native-f32', type=Path)
    args = parser.parse_args()
    with wave.open(str(args.audio), 'rb') as stream:
        if stream.getframerate() != 16000 or stream.getnchannels() != 1 or stream.getsampwidth() != 2:
            raise ValueError('expected mono PCM16 16 kHz WAV')
        audio = np.frombuffer(stream.readframes(stream.getnframes()), dtype='<i2').astype(np.float32) / 32768
    if not 320 < len(audio) <= 16000 or len(audio) % 80:
        raise ValueError('expected 400..16000 samples, divisible by 80')
    import torch
    torch.set_num_threads(2)
    x = torch.from_numpy(audio)
    window = torch.hann_window(640)
    def stft(w):
        return torch.stft(x, n_fft=640, hop_length=80, win_length=640, window=w, center=True,
                          pad_mode='reflect', normalized=False, return_complex=True)
    spectrum = stft(window)
    local = {'audio': audio, 'window': window.numpy(), 'real': spectrum.real.T.contiguous().numpy(),
             'imaginary': spectrum.imag.T.contiguous().numpy(), 'phase': torch.angle(spectrum).T.contiguous().numpy(),
             'first_complex': torch.view_as_real(spectrum[:, 0]).numpy()}
    validate(local, audio)
    np.savez(args.output_npz, **local)
    report = {'torch': str(torch.__version__), 'cpu_capability': torch.backends.cpu.get_cpu_capability(),
              'input_sha256': hashlib.sha256(audio.tobytes()).hexdigest(), 'samples': len(audio),
              'note': 'Diagnostic only. Raw phase remains a required model-input parity gate.'}
    if args.compare_npz:
        with np.load(args.compare_npz, allow_pickle=False) as saved:
            remote = {name: saved[name] for name in saved.files}
        validate(remote, audio)
        replay = torch.angle(stft(torch.from_numpy(remote['window']))).T.numpy()
        report['local_vs_remote_window'] = metrics(local['window'], remote['window'])
        report['local_vs_remote_phase_branches'] = branches(local['phase'], remote['phase'])
        report['local_with_remote_window_vs_remote_phase_branches'] = branches(replay, remote['phase'])
        report['local_vs_remote_first_complex'] = metrics(local['first_complex'], remote['first_complex'])
        for name in ('real', 'imaginary'):
            if name in remote:
                report['local_vs_remote_' + name] = metrics(local[name], remote[name])
    if args.native_f32:
        native = np.fromfile(args.native_f32, dtype='<f4')
        count = local['phase'].size
        if native.size != 640 + 4 * count or not np.isfinite(native).all():
            raise ValueError('invalid native diagnostic layout')
        nw = native[:640]
        nr, ni, _nm, phase = native[640:].reshape(4, *local['phase'].shape)
        frame = np.r_[320:0:-1, 0:320]
        native_product = audio[frame] * nw
        torch_product = (torch.from_numpy(audio[frame]) * torch.from_numpy(nw)).numpy()
        report['native_vs_local_window'] = metrics(nw, local['window'])
        report['same_window_float_multiply'] = metrics(native_product, torch_product)
        report['native_vs_local_real'] = metrics(nr, local['real'])
        report['native_vs_local_imaginary'] = metrics(ni, local['imaginary'])
        report['native_vs_local_phase_branches'] = branches(phase, local['phase'])
        if args.compare_npz:
            report['native_vs_remote_phase_branches'] = branches(phase, remote['phase'])
    args.report.write_text(json.dumps(report, indent=2) + '\n')
    print('REUSE_FRONTEND_STABILITY', json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
