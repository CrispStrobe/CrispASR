#!/usr/bin/env python3
"""Independent PyTorch Mamba-1 block fixture with synthetic weights."""
from pathlib import Path
import torch
import torch.nn.functional as F

torch.set_num_threads(1)
def values(n, mod, shift, divisor, offset=0):
    return (torch.arange(n, dtype=torch.int64) % mod - shift).float() / divisor + offset

batch, times, model, inner, rank, states, kernel = 2, 5, 4, 8, 1, 2, 3
x = values(batch * times * model, 17, 8, 16).reshape(batch, times, model)
in_weight = values(2 * inner * model, 13, 6, 64).reshape(2 * inner, model)
conv_weight = values(inner * kernel, 11, 5, 32).reshape(inner, 1, kernel)
conv_bias = values(inner, 7, 3, 64)
x_weight = values((rank + 2 * states) * inner, 17, 8, 64).reshape(rank + 2 * states, inner)
dt_weight = values(inner * rank, 7, 3, 32).reshape(inner, rank)
dt_bias = values(inner, 9, 4, 32)
a_log = values(inner * states, 5, 2, 16).reshape(inner, states)
d = values(inner, 7, 3, 32)
out_weight = values(model * inner, 11, 5, 64).reshape(model, inner)
def forward(x):
    batch_size, length, _ = x.shape
    u, z = F.linear(x, in_weight).transpose(1, 2).chunk(2, dim=1)
    u = F.silu(F.conv1d(u, conv_weight, conv_bias, padding=kernel - 1, groups=inner)[..., :length])
    factors = F.linear(u.transpose(1, 2), x_weight)
    dt, b, c = torch.split(factors, (rank, states, states), dim=-1)
    delta = F.softplus(F.linear(dt, dt_weight, dt_bias)).transpose(1, 2)
    a = -torch.exp(a_log)
    state = torch.zeros(batch_size, inner, states)
    out = []
    for t in range(length):
        state = torch.exp(delta[:, :, t, None] * a) * state + (delta[:, :, t] * u[:, :, t])[:, :, None] * b[:, t, None, :]
        y = (state * c[:, t, None, :]).sum(-1) + d * u[:, :, t]
        out.append(y * F.silu(z[:, :, t]))
    return F.linear(torch.stack(out, dim=2).transpose(1, 2), out_weight)

y = forward(x)
Path('tests/fixtures/reuse-mamba-f32le.bin').write_bytes(y.contiguous().numpy().astype('<f4').tobytes())
projection = values(model * 2 * model, 13, 6, 32).reshape(model, 2 * model)
bias = values(model, 7, 3, 32)
norm_weight = values(model, 5, 2, 32, 1)
norm_bias = values(model, 9, 4, 64)
def bidirectional(value):
    backward = forward(value.flip(1)).flip(1)
    return F.layer_norm(F.linear(torch.cat((forward(value) + value, backward + value), dim=-1), projection, bias),
                        (model,), norm_weight, norm_bias, eps=1e-5)

bidir = bidirectional(x)
Path('tests/fixtures/reuse-bimamba-f32le.bin').write_bytes(bidir.contiguous().numpy().astype('<f4').tobytes())
time, freq = 5, 3
tf = values(model * time * freq, 19, 9, 16).reshape(1, model, time, freq)
time_input = tf.permute(0, 3, 2, 1).contiguous().view(freq, time, model)
time_output = bidirectional(time_input) + time_input
freq_input = time_output.view(1, freq, time, model).permute(0, 2, 1, 3).contiguous().view(time, freq, model)
freq_output = bidirectional(freq_input) + freq_input
tf_output = freq_output.view(1, time, freq, model).permute(0, 3, 1, 2).contiguous()
Path('tests/fixtures/reuse-tf-mamba-f32le.bin').write_bytes(tf_output.numpy().astype('<f4').tobytes())
print('REUSE_MAMBA_REFERENCE', list(y.shape))
