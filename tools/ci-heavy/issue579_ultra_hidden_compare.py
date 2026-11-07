#!/usr/bin/env python3
"""Compare captured timestamp-marker hidden states to upstream Python."""
import json
import numpy as np
ref = np.load('local-head-stages.npz')['context'][:, 0, :]
actual = np.fromfile('ultra-hidden.bin', dtype='<f4').reshape(ref.shape)
assert np.isfinite(actual).all() and np.isfinite(ref).all()
cos = np.sum(ref * actual, axis=1) / (np.linalg.norm(ref, axis=1) * np.linalg.norm(actual, axis=1))
relative = float(np.linalg.norm(actual-ref) / np.linalg.norm(ref))
print('ULTRA_HIDDEN_PARITY', json.dumps({'endpoints':len(cos), 'min_cosine':float(cos.min()),
                                        'relative_rmse':relative, 'max_abs':float(np.max(np.abs(actual-ref)))}), flush=True)
assert float(cos.min()) >= 0.999 and relative <= 0.02
