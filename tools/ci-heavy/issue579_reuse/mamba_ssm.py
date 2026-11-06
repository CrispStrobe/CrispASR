"""CPU-only Mamba-1 reference for the pinned RE-USE model.

Implements the non-fused selective-scan equations of state-spaces/mamba's
mamba_simple.py and selective_scan_ref. This is an audit adapter, not a fast
kernel or evidence of CUDA behavior.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F


def selective_scan(u, delta, A, B, C, D, z):
    """Inputs u, delta, z: [batch, channel, time]; B/C: [batch, time, state]."""
    state = u.new_zeros((u.shape[0], u.shape[1], A.shape[1]))
    out = []
    for t in range(u.shape[2]):
        dt = delta[:, :, t]
        x = u[:, :, t]
        state = torch.exp(dt.unsqueeze(-1) * A) * state + (dt * x).unsqueeze(-1) * B[:, t, :].unsqueeze(1)
        y = (state * C[:, t, :].unsqueeze(1)).sum(-1) + D * x
        out.append(y * F.silu(z[:, :, t]))
    return torch.stack(out, dim=2)


class Mamba(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, **kwargs):
        super().__init__()
        inner = int(expand * d_model)
        self.d_state = d_state
        self.dt_rank = math.ceil(d_model / 16)
        self.in_proj = nn.Linear(d_model, 2 * inner, bias=False)
        self.conv1d = nn.Conv1d(inner, inner, d_conv, groups=inner, padding=d_conv - 1)
        self.x_proj = nn.Linear(inner, self.dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, inner, bias=True)
        self.A_log = nn.Parameter(torch.empty(inner, d_state))
        self.D = nn.Parameter(torch.empty(inner))
        self.out_proj = nn.Linear(inner, d_model, bias=False)

    def forward(self, hidden_states):
        length = hidden_states.shape[1]
        x, z = self.in_proj(hidden_states).transpose(1, 2).chunk(2, dim=1)
        x = F.silu(self.conv1d(x)[..., :length])
        factors = self.x_proj(x.transpose(1, 2))
        dt, B, C = torch.split(factors, (self.dt_rank, self.d_state, self.d_state), dim=-1)
        delta = F.softplus(self.dt_proj(dt)).transpose(1, 2)
        A = -torch.exp(self.A_log.float())
        y = selective_scan(x, delta, A, B, C, self.D.float(), z)
        return self.out_proj(y.transpose(1, 2))
