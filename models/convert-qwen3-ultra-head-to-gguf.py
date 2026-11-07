#!/usr/bin/env python3
"""Convert the pinned Qwen3-ForcedAligner-Ultra local head to an F32 GGUF sidecar.

This does not convert the main aligner and does not enable the local head in
CrispASR. It preserves the separately licensed upstream checkpoint exactly so
individual port stages can be compared against the captured reference.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from safetensors.numpy import load_file
import gguf

REVISION = '765dc1d714e4c80196782a5df5bfbcb3e1416736'
HASHES = {
    'config.json': 'd3bb91b1c29742f4d5e71507d61e52754d8ee237dc3b2e92500e61c4aa0e3f8e',
    'normalizer.json': '24f6d4fa08dd78e12f7d15a75b0d8b0cd3172bbdfdaa7ae055ca2c64db660ae7',
    'model.safetensors': '50a70d6ed969446770fb9dc1063f5f4e6e4c86eac05fd3efb637492407d0e2ac',
}
SHAPES = {
    'audio.0.weight': (128,91,5), 'audio.0.bias': (128,),
    'audio.2.weight': (128,128,5), 'audio.2.bias': (128,),
    'context_projection.weight': (128,1024), 'context_projection.bias': (128,),
    'role_embedding.weight': (4,128), 'endpoint_embedding.weight': (2,128),
    'context_attention.in_proj_weight': (384,128), 'context_attention.in_proj_bias': (384,),
    'context_attention.out_proj.weight': (128,128), 'context_attention.out_proj.bias': (128,),
    'audio_attention.in_proj_weight': (384,128), 'audio_attention.in_proj_bias': (384,),
    'audio_attention.out_proj.weight': (128,128), 'audio_attention.out_proj.bias': (128,),
    'query_norm.weight': (128,), 'query_norm.bias': (128,), 'audio_norm.weight': (128,), 'audio_norm.bias': (128,),
    'activity_bias.weight': (1,128), 'activity_bias.bias': (1,),
    'boundary_bias.weight': (1,128), 'boundary_bias.bias': (1,),
    'derivative_scale': (), 'qwen_prior': (13,),
}


def validate(weights, normalizer):
    if set(weights) != set(SHAPES):
        raise ValueError('local-head tensor inventory mismatch')
    for name, shape in SHAPES.items():
        value = np.asarray(weights[name])
        if value.shape != shape or value.dtype != np.float32 or not np.isfinite(value).all():
            raise ValueError(f'invalid local-head tensor: {name}')
    if set(normalizer) != {'mean', 'std'}:
        raise ValueError('normalizer inventory mismatch')
    result = {}
    for name in ('mean', 'std'):
        value = np.asarray(normalizer[name], dtype=np.float32)
        if value.shape != (91,) or not np.isfinite(value).all() or (name == 'std' and np.any(value <= 0)):
            raise ValueError(f'invalid normalizer: {name}')
        result[name] = value
    return result


def write_gguf(weights, normalizer, output):
    normalizer = validate(weights, normalizer)
    writer = gguf.GGUFWriter(str(output), arch='qwen3_ultra_head')
    writer.add_name('Qwen3 ForcedAligner Ultra local head')
    writer.add_string('qwen3_ultra_head.source_revision', REVISION)
    for name in sorted(SHAPES):
        writer.add_tensor(name, np.ascontiguousarray(weights[name], dtype=np.float32).reshape(SHAPES[name] or (1,)))
    for name, value in normalizer.items():
        writer.add_tensor('normalizer.' + name, value)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, type=Path, help='pinned local_head directory')
    parser.add_argument('--output', required=True, type=Path, help='new GGUF sidecar path')
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('refusing to overwrite existing output')
    for name, expected in HASHES.items():
        if hashlib.sha256((args.input / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f'unexpected pinned input hash: {name}')
    weights = load_file(str(args.input / 'model.safetensors'))
    normalizer = json.loads((args.input / 'normalizer.json').read_text())
    try:
        write_gguf(weights, normalizer, args.output)
    except Exception:
        args.output.unlink(missing_ok=True)
        raise
    print(f'Wrote {len(SHAPES)} F32 head tensors and two normalizer arrays to {args.output}')


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError) as error:
        sys.exit(str(error))
