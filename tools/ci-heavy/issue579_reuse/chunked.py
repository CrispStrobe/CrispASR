"""Research-only overlap-add harness for bounded RE-USE stage comparisons.

The default is an experimental 1 s diagnostic policy. Explicitly opting into
a larger bound permits up to 5 s windows for further experiments. Neither
policy establishes NVIDIA production equivalence or a streaming interface.
"""
import numpy as np


def chunked(audio, enhance, window_samples=16000, hop_samples=8000, *, max_window_samples=16000):
    """Overlap-add bounded windows; pad to the 80-sample grid and anchor the tail.

    Five-second experiments must explicitly supply window_samples=80000,
    hop_samples=40000 and max_window_samples=80000. Raising the limit alone
    preserves the default window and hop. The returned waveform is cropped
    to the original input length, including for short or unaligned input.
    """
    if type(max_window_samples) is not int or not 400 <= max_window_samples <= 80000 or max_window_samples % 80:
        raise ValueError('max_window_samples must be an 80-sample multiple from 400 to 80000')
    samples = np.asarray(audio, dtype=np.float32)
    if samples.ndim != 1 or not len(samples) or not np.isfinite(samples).all():
        raise ValueError('audio must be a nonempty finite mono array')
    if (not isinstance(window_samples, int) or not isinstance(hop_samples, int)
            or window_samples <= 320 or window_samples > max_window_samples
            or window_samples % 80 or hop_samples <= 0 or hop_samples > window_samples
            or hop_samples % 80):
        raise ValueError('window and hop must be valid 80-sample multiples within the diagnostic limit')
    n = len(samples)
    padded_n = max(window_samples, ((n + 79) // 80) * 80)
    padded = np.pad(samples, (0, padded_n - n))
    starts = list(range(0, padded_n - window_samples + 1, hop_samples))
    if starts[-1] != padded_n - window_samples:
        starts.append(padded_n - window_samples)
    window = np.maximum(np.hanning(window_samples + 1)[:-1], 1e-7).astype(np.float64)
    out = np.zeros(padded_n, dtype=np.float64)
    weights = np.zeros(padded_n, dtype=np.float64)
    for i, start in enumerate(starts):
        result = np.asarray(enhance(padded[start:start + window_samples].copy(), i, start), dtype=np.float32)
        if result.shape != (window_samples,) or not np.isfinite(result).all():
            raise ValueError('enhancer must return one finite value per input sample')
        out[start:start + window_samples] += result * window
        weights[start:start + window_samples] += window
    if not np.all(weights > 0):
        raise ValueError('chunk plan leaves uncovered samples')
    return (out[:n] / weights[:n]).astype(np.float32), starts
