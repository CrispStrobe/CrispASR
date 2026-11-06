"""Compare two unaligned, unnormalized float32 waveforms."""
import json
import numpy as np
import soundfile as sf
ref, sr = sf.read('reuse-reference.wav', dtype='float64')
actual, actual_sr = sf.read('reuse-native.wav', dtype='float64')
assert sr == actual_sr and ref.shape == actual.shape and ref.ndim == 1
assert np.isfinite(actual).all()
cos = float(np.dot(ref, actual) / (np.linalg.norm(ref) * np.linalg.norm(actual)))
rel = float(np.linalg.norm(ref - actual) / np.linalg.norm(ref))
print('REUSE_CPU_PARITY', json.dumps({'samples':len(ref), 'sample_rate':sr, 'cosine':cos, 'relative_rmse':rel}), flush=True)
assert cos >= 0.995 and rel <= 0.1
