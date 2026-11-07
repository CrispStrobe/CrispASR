#!/usr/bin/env python3
"""Capture the RE-USE PyTorch STFT contract on real and synthetic audio."""
import json
import wave
import numpy as np
import torch


def capture(audio, rate, output):
    nfft = int(320 * rate // 8000)
    hop = int(40 * rate // 8000)
    nfft += nfft % 2
    hop += hop % 2
    x = torch.from_numpy(audio.copy())
    z = torch.stft(x, n_fft=nfft, hop_length=hop, win_length=nfft, window=torch.hann_window(nfft),
                   center=True, pad_mode='reflect', normalized=False, return_complex=True)
    mag = torch.log1p(torch.abs(z)).T.contiguous().numpy()
    phase = torch.angle(z).T.contiguous().numpy()
    result = np.stack((mag, phase)).astype('<f4')
    result.tofile(output)
    print('REUSE_STFT_REFERENCE_PASS', json.dumps({'samples':len(audio), 'rate':rate, 'nfft':nfft, 'hop':hop,
                                                   'shape':list(result.shape)}), flush=True)


with wave.open('samples/jfk.wav', 'rb') as f:
    assert f.getnchannels() == 1 and f.getsampwidth() == 2 and f.getframerate() == 16000
    audio = np.frombuffer(f.readframes(1600), dtype='<i2').astype(np.float32) / 32768.0
assert len(audio) == 1600
capture(audio, 16000, 'reuse-stft-reference-f32le.bin')
synthetic = (((np.arange(4400, dtype=np.int64) * 37) % 2001) - 1000).astype(np.float32) / 32768.0
capture(synthetic, 44100, 'reuse-stft-44100-reference-f32le.bin')

# Independent inverse-STFT reference, with model-like amplitude and phase heads.
# Frame 3 has mostly nonpositive magnitudes and exercises the upstream gate.
frames, bins, stride = 11, 321, 322
idx = torch.arange(frames * stride, dtype=torch.int64).reshape(frames, stride)
amp = ((idx % 29).float() - 3) / 50
amp[3, :] = -1
phase = ((idx % 101).float() - 50) / 20
real, imag = torch.cos(phase), torch.sin(phase)
amp = amp[:, :bins]
mag = torch.expm1(torch.relu(amp))
mag[(amp <= 0).sum(1) * 2 > bins] = 0
z = torch.polar(mag, torch.atan2(imag[:, :bins], real[:, :bins])).T.contiguous()
output = torch.istft(z, n_fft=640, hop_length=80, win_length=640, window=torch.hann_window(640), center=True)
np.concatenate((output.numpy(), np.full(3, 1e-8, dtype=np.float32))).astype('<f4').tofile('reuse-istft-reference-f32le.bin')
