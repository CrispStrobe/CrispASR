#!/usr/bin/env python3
"""Capture RE-USE decoder stages from pinned original weights on synthetic input."""
import json
from pathlib import Path
import sys
import numpy as np
import torch
from safetensors.torch import load_file

torch.set_num_threads(1)
torch.set_grad_enabled(False)
sys.path.insert(0, str(Path('reuse-model').resolve()))
from models.codec_module_time_d4 import MagDecoder, PhaseDecoder

config = json.loads(Path('reuse-model/config.json').read_text())
weights = load_file('reuse-model/model.safetensors')
x = ((torch.arange(64 * 2 * 3, dtype=torch.int64) % 19 - 9).float() / 16).reshape(1, 64, 2, 3)
with open('reuse-decoder-original-oracle.bin', 'wb') as out:
    def save(tensor):
        data = tensor.detach().float().contiguous().numpy()
        assert np.isfinite(data).all()
        out.write(data.astype('<f4').tobytes())
    for prefix, cls in [('mask_decoder.', MagDecoder), ('phase_decoder.', PhaseDecoder)]:
        decoder = cls(config).eval()
        decoder.load_state_dict({name.removeprefix(prefix): value for name, value in weights.items()
                                 if name.startswith(prefix)}, strict=True)
        for module in decoder.dense_block.dense_block:
            for tensor in (module[0].weight, module[0].bias, module[1].weight, module[1].bias, module[2].weight):
                save(tensor)
        for module in (decoder.up_conv1, decoder.up_conv2):
            for tensor in (module[0].conv.weight, module[0].conv.bias, module[1].weight, module[1].bias,
                           module[2].weight):
                save(tensor)
        heads = [decoder.final_conv] if prefix == 'mask_decoder.' else [decoder.phase_conv_r, decoder.phase_conv_i]
        for head in heads:
            save(head.weight)
            save(head.bias)
        dense = decoder.dense_block(x)
        first = decoder.up_conv1(dense)
        final = decoder.up_conv2(first.transpose(2, 3)).transpose(2, 3).contiguous()
        for value in (dense, first, final, *(head(final) for head in heads)):
            save(value)
print('REUSE_ORIGINAL_DECODER_ORACLE', flush=True)
