"""Audio comparison helpers for TTS diff harnesses.

Three uses today (vibevoice-tts, qwen3-tts pipeline test, future TTS
backends):

  - sample-wise cosine at zero shift (catches everything that's wildly
    wrong);
  - sample-wise cosine at the cross-correlation peak (catches things
    that are bit-correct modulo a leading-silence trim or a
    causal-padding offset);
  - per-band spectral power table (catches systematic activation /
    weight-scaling errors that a noise-pinned cos still misses when
    diffusion stochasticity drives most of the variance).

`load_wav_mono` reads 16-bit PCM WAV in float64 [-1, 1]. WAV is the
common output format across crispasr's TTS backends. If a backend ever
needs float32 .raw files, route them through a small wrapper.

The audio cos/xcorr math is plain numpy + scipy.signal.correlate. Avoids
adding a new dependency: scipy is already pulled in by the reference
backends (transformers indirectly).
"""

from __future__ import annotations

import wave
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np


# Speech bands (Hz). Roughly: F0 fundamentals; vowel formants; consonant
# resonances; fricative onsets; sibilance / breath. Choose ordered ranges
# disjoint, contiguous, and ending at 12 kHz (Nyquist for 24 kHz audio).
DEFAULT_BANDS: Tuple[Tuple[int, int], ...] = (
    (0, 500),
    (500, 1500),
    (1500, 3000),
    (3000, 6000),
    (6000, 12000),
)


def load_wav_mono(path: str | Path) -> Tuple[np.ndarray, int]:
    """Read a mono 16-bit PCM WAV. Returns (samples in [-1, 1], sample_rate).

    Stereo files are downmixed to mono. Returns float64 — the absolute
    magnitudes are well within float64 precision and downstream code
    (FFT, correlate) works at float64 by default anyway.
    """
    with wave.open(str(path), "rb") as w:
        n = w.getnframes()
        sr = w.getframerate()
        nch = w.getnchannels()
        sw = w.getsampwidth()
        raw = w.readframes(n)
    if sw != 2:
        raise ValueError(f"{path}: expected 16-bit PCM, got sample width {sw}")
    data = np.frombuffer(raw, dtype=np.int16).astype(np.float64) / 32768.0
    if nch > 1:
        data = data.reshape(-1, nch).mean(axis=1)
    return data, sr


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    return float(a @ b / (na * nb + 1e-12))


def cos_at_zero_shift(a: np.ndarray, b: np.ndarray) -> Tuple[float, int]:
    """Cosine with zero alignment, truncated to the shorter of the two.
    Returns (cos, n_samples_compared)."""
    n = min(len(a), len(b))
    if n == 0:
        return 0.0, 0
    return _cos(a[:n], b[:n]), n


def cos_at_xcorr_peak(
    a: np.ndarray,
    b: np.ndarray,
    *,
    window_samples: int | None = 120000,
    max_abs_lag: int | None = None,
) -> Tuple[float, int, int]:
    """Find the best integer-sample alignment between `a` and `b` via
    cross-correlation, then return the cos at that alignment.

    Tunables:
      - `window_samples` caps the segment fed into correlate (default
        ~5 s at 24 kHz). Cross-correlation cost is O(N log N); capping
        keeps it fast and avoids the FFT going wild on very long inputs.
      - `max_abs_lag` clamps the search range (e.g. 0.5 s = 12000 at
        24 kHz) when you know the offset can't be huge — guards against
        a spurious far-out peak.

    Returns (cos, lag_samples, n_samples_compared). `lag_samples > 0`
    means `a` leads `b` (drop `lag_samples` from the front of `a` to
    align). `lag_samples < 0` means the opposite.
    """
    from scipy.signal import correlate

    m = min(len(a), len(b), window_samples or len(a))
    if m == 0:
        return 0.0, 0, 0
    xc = correlate(a[:m], b[:m], mode="full", method="fft")
    lags = np.arange(-m + 1, m)
    if max_abs_lag is not None:
        mask = np.abs(lags) <= max_abs_lag
        peak_idx = int(np.argmax(np.abs(xc[mask])))
        lag = int(lags[mask][peak_idx])
    else:
        lag = int(lags[int(np.argmax(np.abs(xc)))])

    if lag > 0:
        sa = a[lag:]
        sb = b[: len(sa)]
    elif lag < 0:
        sb = b[-lag:]
        sa = a[: len(sb)]
    else:
        sa, sb = a, b
    n = min(len(sa), len(sb))
    if n == 0:
        return 0.0, lag, 0
    return _cos(sa[:n], sb[:n]), lag, n


def spectral_band_powers(
    data: np.ndarray, sr: int, bands: Sequence[Tuple[int, int]] = DEFAULT_BANDS,
) -> List[float]:
    """Power-spectrum percentages per `(lo, hi)` band.

    Bands are inclusive-exclusive `[lo, hi)` Hz. Total power normalises
    to 100% (which won't sum to exactly 100 if the bands don't cover the
    entire spectrum). Returns one float per band, in band order.
    """
    spec = np.abs(np.fft.rfft(data))
    sp = spec * spec
    f = np.fft.rfftfreq(len(data), 1.0 / sr)
    total = float(sp.sum())
    if total <= 0:
        return [0.0] * len(bands)
    return [100.0 * float(sp[(f >= lo) & (f < hi)].sum()) / total for lo, hi in bands]


def hf_floor(data: np.ndarray, sr: int, *, lo_hz: int = 8000) -> float:
    """Median magnitude of the spectrum above `lo_hz` — a rough proxy
    for high-frequency noise. Useful when comparing TTS outputs from
    diffusion samplers (where most of the audio matches in spectrum but
    the silent regions can have varying hiss)."""
    spec = np.abs(np.fft.rfft(data))
    f = np.fft.rfftfreq(len(data), 1.0 / sr)
    sel = spec[f >= lo_hz]
    if sel.size == 0:
        return 0.0
    return float(np.median(sel))


def summary_row(label: str, data: np.ndarray, sr: int,
                bands: Sequence[Tuple[int, int]] = DEFAULT_BANDS) -> str:
    """Single-line summary used by `format_table` and ad-hoc prints."""
    rms = float(np.sqrt((data ** 2).mean()))
    pcts = spectral_band_powers(data, sr, bands)
    nf = hf_floor(data, sr)
    band_str = "/".join(f"{p:5.1f}" for p in pcts)
    band_labels = "/".join(
        f"{'≥' if hi >= 1000 else ''}{lo // 1000 if lo >= 1000 else lo}-"
        f"{hi // 1000 if hi >= 1000 else hi}{' kHz' if hi >= 1000 else ' Hz'}"
        for lo, hi in bands
    )  # noqa: F841 (unused — kept for future header rendering)
    return (f"{label:30s}  dur={len(data) / sr:5.2f}s  rms={rms:.4f}  "
            f"bands(% per band): {band_str}  hf_floor={nf:.3f}")


def format_table(
    pairs: Iterable[Tuple[str, str | Path]],
    *,
    bands: Sequence[Tuple[int, int]] = DEFAULT_BANDS,
) -> str:
    """Render a one-line-per-WAV table of duration / RMS / band-percent /
    hf-floor. Bands list is reused for the band-percent string. The
    label column is fixed-width 30 chars."""
    lines = []
    for label, path in pairs:
        data, sr = load_wav_mono(path)
        lines.append(summary_row(label, data, sr, bands))
    return "\n".join(lines)


def cos_report(
    a_path: str | Path,
    b_path: str | Path,
    *,
    label_a: str = "A",
    label_b: str = "B",
    window_samples: int | None = 120000,
    max_abs_lag: int | None = None,
) -> str:
    """Multi-line report comparing two WAVs:
      - file metadata (samples, duration, RMS) for each;
      - cos at zero shift;
      - cos at cross-correlation peak (and the lag in samples + ms).

    Suitable for printing from CLI scripts or pasting into PR
    descriptions. Returns the formatted string; the caller decides
    whether to print or capture it.
    """
    a, sr_a = load_wav_mono(a_path)
    b, sr_b = load_wav_mono(b_path)
    if sr_a != sr_b:
        raise ValueError(f"sample-rate mismatch: {sr_a} ({a_path}) vs {sr_b} ({b_path})")
    sr = sr_a

    cos0, n0 = cos_at_zero_shift(a, b)
    cos_xc, lag, n_xc = cos_at_xcorr_peak(a, b, window_samples=window_samples,
                                          max_abs_lag=max_abs_lag)
    rms_a = float(np.sqrt((a ** 2).mean()))
    rms_b = float(np.sqrt((b ** 2).mean()))
    return "\n".join([
        f"{label_a}: {len(a)} samples = {len(a)/sr:.2f}s  rms={rms_a:.4f}",
        f"{label_b}: {len(b)} samples = {len(b)/sr:.2f}s  rms={rms_b:.4f}",
        f"cos at zero shift  = {cos0:.4f}  (n={n0})",
        f"cos at xcorr peak  = {cos_xc:.4f}  lag={lag} samples = {lag/sr*1000:.1f} ms  (n={n_xc})",
    ])
