#!/usr/bin/env python3
"""Prepare hosted-only malformed GGUF controls and original tensor probes."""
import gc
import json
from pathlib import Path
import gguf
import numpy as np
from safetensors import safe_open

inventory = json.loads(Path('tests/fixtures/reuse-inventory.json').read_text())
revision = '31f82f2efad1095034492f236e8dac9239ba713b'
target = 'dense_encoder.dense_conv_1.0.weight'

def write(name, arch, tensors):
    writer = gguf.GGUFWriter(name, arch=arch)
    writer.add_string('reuse.source_revision', revision)
    for tensor_name, tensor in tensors:
        writer.add_tensor(tensor_name, tensor)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

write('reuse-bad-arch.gguf', 'wrong', [('dummy', np.zeros(1, dtype=np.float32))])
write('reuse-missing.gguf', 'reuse', [('dummy', np.zeros(1, dtype=np.float32))])
for mode in ('shape', 'dtype'):
    arrays = []
    for name, shape in inventory.items():
        shape = list(shape)
        dtype = np.float32
        if name == target and mode == 'shape':
            shape[-1] += 1
        if name == target and mode == 'dtype':
            dtype = np.float16
        arrays.append((name, np.zeros(shape, dtype=dtype)))
    write('reuse-bad-' + mode + '.gguf', 'reuse', arrays)
    del arrays
    gc.collect()

names = ['dense_encoder.dense_conv_1.0.weight', 'TSMamba.0.time_mamba.forward_blocks.in_proj.weight',
         'TSMamba.29.freq_mamba.backward_blocks.A_log', 'mask_decoder.up_conv2.0.conv.weight',
         'phase_decoder.phase_conv_i.weight']
with safe_open('reuse-model/model.safetensors', framework='numpy') as original, open('reuse-loader-probe.bin', 'wb') as out:
    for name in names:
        value = original.get_tensor(name)
        assert value.dtype == np.float32 and list(value.shape) == inventory[name]
        out.write(np.ascontiguousarray(value, dtype='<f4').tobytes())
print('REUSE_LOADER_FIXTURES_READY', flush=True)
