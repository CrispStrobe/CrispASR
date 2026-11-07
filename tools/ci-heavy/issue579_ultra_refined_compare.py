#!/usr/bin/env python3
"""Compare experimental native Ultra refined timestamps with upstream."""
import json
import numpy as np
reference = json.load(open('local-head-reference.json'))
expected = np.array([[w['start'] * 1000, w['end'] * 1000] for w in reference])
actual = np.fromfile('ultra-refined.bin', dtype='<f8').reshape(expected.shape)
assert np.isfinite(actual).all() and np.all(actual[:,0] <= actual[:,1])
error = np.abs(actual - expected)
print('ULTRA_REFINED_PARITY', json.dumps({'words':len(reference), 'max_error_ms':float(error.max()),
                                         'mean_error_ms':float(error.mean())}), flush=True)
assert float(error.max()) <= 2.0 and float(error.mean()) <= 0.5
