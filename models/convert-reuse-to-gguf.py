#!/usr/bin/env python3
"""Convert the pinned NVIDIA NemotronLabs RE-USE checkpoint to F32 GGUF.

This converter produces a research artifact for the in-progress port. It does
not make RE-USE available through CrispASR's CLI or C ABI.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import gguf
import numpy as np
from safetensors import safe_open

REVISION = '31f82f2efad1095034492f236e8dac9239ba713b'
HASHES = {
    'config.json': 'e69986483774f01893d2727503addc49ef201d19e6e1a266b8d4e971e28c478d',
    'model.safetensors': '06badaf4029b297f97f7fe48d62990b48d5482608210df577354e0d4e0ed70e1',
}
EXPECTED_MODEL = {'compress_factor': 'relu_log1p', 'd_conv': 4, 'd_state': 16, 'expand': 4,
                  'hid_feature': 64, 'num_tfmamba': 30, 'input_channel': 2, 'output_channel': 1}
EXPECTED_STFT = {'n_fft': 320, 'hop_size': 40, 'win_size': 320, 'sampling_rate': 8000}


def validate_config(config):
    if not isinstance(config, dict):
        raise ValueError('config must be an object')
    for section, expected in (('model_cfg', EXPECTED_MODEL), ('stft_cfg', EXPECTED_STFT)):
        value = config.get(section)
        if not isinstance(value, dict) or any(value.get(k) != v for k, v in expected.items()):
            raise ValueError(f'incompatible {section}')


def validate_tensor(name, value):
    data = np.asarray(value)
    if not name or data.dtype != np.float32 or data.ndim < 1 or data.ndim > 4 or any(x <= 0 for x in data.shape):
        raise ValueError(f'invalid tensor metadata: {name}')
    if not np.isfinite(data).all():
        raise ValueError(f'nonfinite tensor: {name}')
    return np.ascontiguousarray(data)


def write_gguf(tensors, config, output):
    validate_config(config)
    writer = gguf.GGUFWriter(str(output), arch='reuse')
    writer.add_name('NVIDIA NemotronLabs RE-USE')
    writer.add_string('reuse.source_revision', REVISION)
    writer.add_string('reuse.license', 'OpenMDW-1.1')
    writer.add_string('reuse.model_config_json', json.dumps(config['model_cfg'], sort_keys=True))
    writer.add_string('reuse.stft_config_json', json.dumps(config['stft_cfg'], sort_keys=True))
    for name in sorted(tensors):
        writer.add_tensor(name, validate_tensor(name, tensors[name]))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


def file_hash(path):
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, type=Path, help='pinned upstream model directory')
    parser.add_argument('--output', required=True, type=Path, help='new F32 GGUF path')
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('refusing to overwrite existing output')
    for name, expected in HASHES.items():
        if file_hash(args.input / name) != expected:
            raise ValueError(f'unexpected pinned input hash: {name}')
    config = json.loads((args.input / 'config.json').read_text())
    validate_config(config)
    with safe_open(str(args.input / 'model.safetensors'), framework='numpy') as source:
        keys = list(source.keys())
        if len(keys) != 1416:
            raise ValueError('unexpected tensor count')
        tensors = {name: source.get_tensor(name) for name in keys}
    try:
        write_gguf(tensors, config, args.output)
    except Exception:
        args.output.unlink(missing_ok=True)
        raise
    print(f'Converted {len(tensors)} F32 tensors from pinned RE-USE revision to {args.output}')


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError) as error:
        sys.exit(str(error))
