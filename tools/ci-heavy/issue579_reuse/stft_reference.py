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
