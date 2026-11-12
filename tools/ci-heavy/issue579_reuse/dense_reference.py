#!/usr/bin/env python3
"""Independent PyTorch fixture for the RE-USE dense encoder contract."""
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

torch.set_num_threads(1)
channels, time, freq = 64, 5, 7

def values(n, mod, shift, divisor):
    return ((torch.arange(n, dtype=torch.int64) % mod - shift).float() / divisor)

def layer(x, inputs, kt, kf, stride=(1, 1), dilation=(1, 1), padding=(0, 0)):
    weights = values(channels * inputs * kt * kf, 17, 8, 128).reshape(channels, inputs, kt, kf)
    bias = values(channels, 7, 3, 64)
    norm_weight = 1 + values(channels, 9, 4, 64)
    norm_bias = values(channels, 11, 5, 128)
    slope = 0.25 + values(channels, 5, 2, 64)
    x = F.conv2d(x, weights, bias, stride=stride, padding=padding, dilation=dilation)
    x = F.instance_norm(x, weight=norm_weight, bias=norm_bias, eps=1e-5)
    return F.prelu(x, slope)

x = values(2 * time * freq, 31, 15, 32).reshape(1, 2, time, freq)
x = F.pad(x, (0, 2, 0, 2))
x = layer(x, 2, 1, 1)
outputs = [x]
skip = x
for i in range(4):
    dilation = 2 ** i
    x = layer(skip, channels * (i + 1), 3, 3, dilation=(dilation, 1), padding=(dilation, 1))
    outputs.append(x)
    skip = torch.cat((x, skip), dim=1)
x = layer(x, channels, 1, 3, stride=(4, 2))
outputs.append(x)
Path('tests/fixtures/reuse-dense-encoder-f32le.bin').write_bytes(
    b''.join(v.detach().contiguous().numpy().astype('<f4').tobytes() for v in outputs))
print('REUSE_DENSE_REFERENCE', [list(x.shape) for x in outputs])
