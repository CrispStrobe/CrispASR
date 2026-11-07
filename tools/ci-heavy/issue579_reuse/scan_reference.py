#!/usr/bin/env python3
"""Independent PyTorch selective-scan fixture for small synthetic inputs."""
from pathlib import Path
import torch
import torch.nn.functional as F

def values(n, mod, shift, divisor, offset=0):
    return (torch.arange(n, dtype=torch.int64) % mod - shift).float() / divisor + offset

batch, channels, times, states = 2, 3, 4, 2
u = values(batch * channels * times, 11, 5, 8).reshape(batch, channels, times)
delta = values(batch * channels * times, 7, 0, 16, 0.125).reshape(batch, channels, times)
a = -values(channels * states, 5, 0, 8, 0.125).reshape(channels, states)
b = values(batch * times * states, 9, 4, 8).reshape(batch, times, states)
c = values(batch * times * states, 7, 3, 8).reshape(batch, times, states)
d = values(channels, 5, 2, 8)
z = values(batch * channels * times, 13, 6, 8).reshape(batch, channels, times)
state = torch.zeros((batch, channels, states))
output = []
for t in range(times):
    dt, x = delta[:, :, t], u[:, :, t]
    state = torch.exp(dt.unsqueeze(-1) * a) * state + (dt * x).unsqueeze(-1) * b[:, t, :].unsqueeze(1)
    output.append(((state * c[:, t, :].unsqueeze(1)).sum(-1) + d * x) * F.silu(z[:, :, t]))
output = torch.stack(output, dim=2)
Path('tests/fixtures/reuse-scan-f32le.bin').write_bytes(
    output.numpy().astype('<f4').tobytes() + state.numpy().astype('<f4').tobytes())
print('REUSE_SCAN_REFERENCE', list(output.shape), list(state.shape))
