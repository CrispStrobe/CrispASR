#!/usr/bin/env python3
"""Check raw, unnormalized RE-USE stages and waveform against the original."""
import json
import numpy as np

def compare(name, expected, actual, limit):
    expected = np.asarray(expected, dtype=np.float64).reshape(-1)
    actual = np.asarray(actual, dtype=np.float64).reshape(-1)
    assert actual.shape == expected.shape and np.isfinite(actual).all() and np.isfinite(expected).all()
    expected_norm, actual_norm = np.linalg.norm(expected), np.linalg.norm(actual)
    assert expected_norm > 1e-6 and actual_norm > 1e-6
    relative = float(np.linalg.norm(actual - expected) / expected_norm)
    cosine = float(np.dot(actual, expected) / (actual_norm * expected_norm))
    ratio = float(actual_norm / expected_norm)
    print('REUSE_NETWORK_PARITY', name, json.dumps({'relative_rmse':relative, 'cosine':cosine,
                                                     'norm_ratio':ratio}), flush=True)
    assert relative <= limit and cosine >= 0.999 and 0.98 <= ratio <= 1.02
    return {'relative_rmse':relative, 'cosine':cosine, 'norm_ratio':ratio}

def main():
    with np.load('reuse-network-reference-stages.npz') as reference_stages:
        for name in ('encoder', 'block0', 'block14', 'block29'):
            compare(name, reference_stages[name], np.fromfile('reuse-trace-' + name + '.f32le', dtype='<f4'), 0.002)
    compare('waveform', np.fromfile('reuse-network-expected.f32le', dtype='<f4'),
            np.fromfile('reuse-network-actual.f32le', dtype='<f4'), 0.005)

if __name__ == '__main__':
    main()
