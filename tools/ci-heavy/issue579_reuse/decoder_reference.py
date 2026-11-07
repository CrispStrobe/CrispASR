#!/usr/bin/env python3
"""Independent PyTorch fixture for RE-USE decoder stages (synthetic weights)."""
from pathlib import Path
import torch
import torch.nn.functional as F

torch.set_num_threads(1)
def values(n, mod, shift, divisor, offset=0):
    return (torch.arange(n, dtype=torch.int64) % mod - shift).float() / divisor + offset

channels, time, freq = 4, 2, 3
x = values(channels * time * freq, 19, 9, 16).reshape(1, channels, time, freq)
skip = x
for i in range(4):
    dilation = 2 ** i
    inputs = channels * (i + 1)
    weight = values(channels * inputs * 3 * 3, 17, 8, 128).reshape(channels, inputs, 3, 3)
    bias = values(channels, 7, 3, 64)
    norm_weight = values(channels, 9, 4, 64, 1)
    norm_bias = values(channels, 11, 5, 128)
    slope = values(channels, 5, 2, 64, 0.25)
    x = F.conv2d(skip, weight, bias, padding=(dilation, 1), dilation=(dilation, 1))
    x = F.prelu(F.instance_norm(x, weight=norm_weight, bias=norm_bias, eps=1e-5), slope)
    skip = torch.cat((x, skip), dim=1)
stages = [x]

def upsample(x, scale):
    weight = values(scale * channels * channels * 3, 13, 6, 64).reshape(scale * channels, channels, 1, 3)
    bias = values(scale * channels, 7, 3, 64)
    norm_weight = values(channels, 9, 4, 64, 1)
    norm_bias = values(channels, 11, 5, 128)
    slope = values(channels, 5, 2, 64, 0.25)
    y = F.conv2d(F.pad(x, (1, 1, 0, 0)), weight, bias)
    _, _, t, f = y.shape
    y = y.view(1, scale, channels, t, f).permute(0, 2, 3, 4, 1).contiguous().view(1, channels, t, f * scale)
    return F.prelu(F.instance_norm(y, weight=norm_weight, bias=norm_bias, eps=1e-5), slope)

x = upsample(x, 2)
stages.append(x)
x = upsample(x.transpose(2, 3), 4).transpose(2, 3).contiguous()
stages.append(x)
for shift in (2, 3, 4):
    weight = values(channels, 9, shift, 32).reshape(1, channels, 1, 1)
    stages.append(F.conv2d(x, weight, torch.tensor([0.125])))
Path('tests/fixtures/reuse-decoder-f32le.bin').write_bytes(
    b''.join(value.contiguous().numpy().astype('<f4').tobytes() for value in stages))
print('REUSE_DECODER_REFERENCE', [list(value.shape) for value in stages])
