#!/usr/bin/env python3
"""Capture the official RE-USE PyTorch STFT contract on a public test clip."""
import json
import wave
import numpy as np
import torch

with wave.open('samples/jfk.wav', 'rb') as f:
    assert f.getnchannels() == 1 and f.getsampwidth() == 2 and f.getframerate() == 16000
    audio = np.frombuffer(f.readframes(1600), dtype='<i2').astype(np.float32) / 32768.0
assert len(audio) == 1600
nfft, hop = 640, 80
x = torch.from_numpy(audio.copy())
z = torch.stft(x, n_fft=nfft, hop_length=hop, win_length=nfft, window=torch.hann_window(nfft),
               center=True, pad_mode='reflect', normalized=False, return_complex=True)
mag = torch.log1p(torch.abs(z)).T.contiguous().numpy()
phase = torch.angle(z).T.contiguous().numpy()
assert mag.shape == phase.shape == (21,321)
out = np.stack((mag,phase)).astype('<f4')
out.tofile('reuse-stft-reference-f32le.bin')
print('REUSE_STFT_REFERENCE_PASS', json.dumps({'samples':len(audio),'rate':16000,'nfft':nfft,'hop':hop,
                                               'shape':list(out.shape)}), flush=True)
