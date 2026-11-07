#!/usr/bin/env python3
"""Run the pinned original dense encoder and prepare a local-only C++ oracle."""
import json
from pathlib import Path
import sys
import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file

torch.set_num_threads(1)
torch.set_grad_enabled(False)
sys.path.insert(0, str(Path('reuse-model').resolve()))
from models.codec_module_time_d4 import DenseEncoder

config = json.loads(Path('reuse-model/config.json').read_text())
encoder = DenseEncoder(config).eval()
weights = load_file('reuse-model/model.safetensors')
encoder.load_state_dict({k.removeprefix('dense_encoder.'): v for k, v in weights.items()
                         if k.startswith('dense_encoder.')}, strict=True)
features = ((torch.arange(2 * 5 * 7, dtype=torch.int64) % 31 - 15).float() / 32).reshape(1, 2, 5, 7)
x = F.pad(features, (0, 2, 0, 2))
out = encoder.dense_conv_1(x)
stages = [out]
skip = out
for block in encoder.dense_block.dense_block:
    out = block(skip)
    stages.append(out)
    skip = torch.cat((out, skip), dim=1)
out = encoder.dense_conv_2(out)
stages.append(out)
modules = [encoder.dense_conv_1, *encoder.dense_block.dense_block, encoder.dense_conv_2]
with open('reuse-dense-original-oracle.bin', 'wb') as f:
    def write(tensor):
        f.write(tensor.detach().contiguous().numpy().astype('<f4').tobytes())
    for module in modules:
        for value in (module[0].weight, module[0].bias, module[1].weight, module[1].bias, module[2].weight):
            write(value)
    for value in stages:
        write(value)
print('REUSE_ORIGINAL_DENSE_ORACLE', [list(x.shape) for x in stages], flush=True)
