#!/usr/bin/env python3
"""Capture original-weight Mamba scan inputs using the CPU reference adapter."""
import sys
import json
from pathlib import Path
import numpy as np
import torch
from safetensors.torch import load_file

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mamba_ssm

torch.set_num_threads(1)
torch.set_grad_enabled(False)
prefix = 'TSMamba.0.time_mamba.forward_blocks.'
weights = load_file('reuse-model/model.safetensors')
block = mamba_ssm.Mamba(d_model=64, d_state=16, d_conv=4, expand=4).eval()
block.load_state_dict({name.removeprefix(prefix): value for name, value in weights.items()
                       if name.startswith(prefix)}, strict=True)
original = mamba_ssm.selective_scan

def capture(u, delta, a, b, c, d, z):
    result = original(u, delta, a, b, c, d, z)
    assert list(u.shape) == [2, 256, 3] and list(b.shape) == [2, 3, 16]
    with open('reuse-scan-original-oracle.bin', 'wb') as out:
        for tensor in (u, delta, a, b, c, d, z, result):
            data = tensor.detach().float().contiguous().numpy()
            assert np.isfinite(data).all()
            out.write(data.astype('<f4').tobytes())
    return result

mamba_ssm.selective_scan = capture
hidden = ((torch.arange(2 * 3 * 64, dtype=torch.int64) % 31 - 15).float() / 32).reshape(2, 3, 64)
result = block(hidden)
assert torch.isfinite(result).all()
with open('reuse-mamba-original-oracle.bin', 'wb') as out:
    for tensor in (block.in_proj.weight, block.conv1d.weight, block.conv1d.bias, block.x_proj.weight,
                   block.dt_proj.weight, block.dt_proj.bias, block.A_log, block.D, block.out_proj.weight, result):
        data = tensor.detach().float().contiguous().numpy()
        assert np.isfinite(data).all()
        out.write(data.astype('<f4').tobytes())
sys.path.insert(1, str(Path('reuse-model').resolve()))
from models.mamba_block2_SEMamba import MambaBlock, TFMambaBlock
mamba_ssm.selective_scan = original
config = json.loads(Path('reuse-model/config.json').read_text())
bimamba = MambaBlock(d_model=64, cfg=config).eval()
prefix = 'TSMamba.0.time_mamba.'
bimamba.load_state_dict({name.removeprefix(prefix): value for name, value in weights.items()
                         if name.startswith(prefix)}, strict=True)
bidir = bimamba(hidden)
assert torch.isfinite(bidir).all()
with open('reuse-bimamba-original-oracle.bin', 'wb') as out:
    def save(tensor):
        data = tensor.detach().float().contiguous().numpy()
        assert np.isfinite(data).all()
        out.write(data.astype('<f4').tobytes())
    for part in (bimamba.forward_blocks, bimamba.backward_blocks):
        for tensor in (part.in_proj.weight, part.conv1d.weight, part.conv1d.bias, part.x_proj.weight,
                       part.dt_proj.weight, part.dt_proj.bias, part.A_log, part.D, part.out_proj.weight):
            save(tensor)
    for tensor in (bimamba.output_proj.weight, bimamba.output_proj.bias, bimamba.norm.weight,
                   bimamba.norm.bias, bidir):
        save(tensor)
tf_block = TFMambaBlock(config).eval()
prefix = 'TSMamba.0.'
tf_block.load_state_dict({name.removeprefix(prefix): value for name, value in weights.items()
                          if name.startswith(prefix)}, strict=True)
tf_input = ((torch.arange(64 * 3 * 4, dtype=torch.int64) % 19 - 9).float() / 16).reshape(1, 64, 3, 4)
tf_output = tf_block(tf_input)
assert torch.isfinite(tf_output).all()
with open('reuse-tf-mamba-original-oracle.bin', 'wb') as out:
    def save_tf(tensor):
        data = tensor.detach().float().contiguous().numpy()
        assert np.isfinite(data).all()
        out.write(data.astype('<f4').tobytes())
    for direction in (tf_block.time_mamba, tf_block.freq_mamba):
        for part in (direction.forward_blocks, direction.backward_blocks):
            for tensor in (part.in_proj.weight, part.conv1d.weight, part.conv1d.bias, part.x_proj.weight,
                           part.dt_proj.weight, part.dt_proj.bias, part.A_log, part.D, part.out_proj.weight):
                save_tf(tensor)
        for tensor in (direction.output_proj.weight, direction.output_proj.bias, direction.norm.weight,
                       direction.norm.bias):
            save_tf(tensor)
    save_tf(tf_output)
print('REUSE_ORIGINAL_SCAN_ORACLE', list(result.shape), flush=True)
