#!/usr/bin/env python3
"""Prepare the pinned upstream local-head weights and saved input/logit oracle.

This binary is a temporary test fixture, not a model format or release artifact.
"""
import hashlib
from pathlib import Path
import numpy as np
from huggingface_hub import hf_hub_download
from safetensors.numpy import load_file

path = Path(hf_hub_download('17slever17/Qwen3-ForcedAligner-Ultra', 'local_head/model.safetensors',
                            revision='765dc1d714e4c80196782a5df5bfbcb3e1416736', local_dir='head-model'))
assert hashlib.sha256(path.read_bytes()).hexdigest() == '50a70d6ed969446770fb9dc1063f5f4e6e4c86eac05fd3efb637492407d0e2ac'
weights = load_file(str(path))
shapes = [
    ('audio.0.weight', (128,91,5)), ('audio.0.bias', (128,)),
    ('audio.2.weight', (128,128,5)), ('audio.2.bias', (128,)),
    ('context_projection.weight', (128,1024)), ('context_projection.bias', (128,)),
    ('role_embedding.weight', (4,128)), ('endpoint_embedding.weight', (2,128)),
    ('context_attention.in_proj_weight', (384,128)), ('context_attention.in_proj_bias', (384,)),
    ('context_attention.out_proj.weight', (128,128)), ('context_attention.out_proj.bias', (128,)),
    ('audio_attention.in_proj_weight', (384,128)), ('audio_attention.in_proj_bias', (384,)),
    ('audio_attention.out_proj.weight', (128,128)), ('audio_attention.out_proj.bias', (128,)),
    ('query_norm.weight', (128,)), ('query_norm.bias', (128,)),
    ('audio_norm.weight', (128,)), ('audio_norm.bias', (128,)),
    ('activity_bias.weight', (1,128)), ('activity_bias.bias', (1,)),
    ('boundary_bias.weight', (1,128)), ('boundary_bias.bias', (1,)),
    ('derivative_scale', ()), ('qwen_prior', (13,)),
]
assert set(weights) == {name for name, _ in shapes}
stages = np.load('local-head-stages.npz')
assert stages['features'].shape == (44,41,91) and stages['context'].shape == (44,4,1024)
with open('ultra-head-oracle.bin', 'wb') as out:
    for name, shape in shapes:
        value = weights[name]
        assert value.shape == shape and np.isfinite(value).all(), name
        out.write(np.asarray(value, dtype='<f4').tobytes())
    for i in range(44):
        for name in ('features', 'context', 'context_mask', 'endpoints', 'boundary_logits'):
            out.write(np.asarray(stages[name][i], dtype='<f4').tobytes())
print('HEAD_ORACLE_PREPARED', Path('ultra-head-oracle.bin').stat().st_size, flush=True)
