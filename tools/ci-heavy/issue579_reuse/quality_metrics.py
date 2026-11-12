#!/usr/bin/env python3
"""Read-only quality diagnostics for aligned research waveform artifacts.

These objective metrics on a speech proxy with synthetic noise do not establish
perceptual quality, generalization, or an acceptance threshold.
"""
import argparse
import json
from pathlib import Path
import numpy as np


def quality(reference, estimate):
    ref = np.asarray(reference, dtype=np.float64)
    est = np.asarray(estimate, dtype=np.float64)
    if (ref.ndim != 1 or est.shape != ref.shape or len(ref) == 0
            or not np.isfinite(ref).all() or not np.isfinite(est).all()):
        raise ValueError('reference and estimate must be nonempty, aligned, finite mono arrays')
    energy = float(np.dot(ref, ref))
    centered = ref - ref.mean()
    centered_energy = float(np.dot(centered, centered))
    if energy <= 1e-20 or centered_energy <= 1e-20:
        raise ValueError('reference must have nonzero energy and variance')
    noise = ref - est
    snr = 10 * np.log10(energy / max(float(np.dot(noise, noise)), 1e-30))
    est_centered = est - est.mean()
    projection = centered * (float(np.dot(est_centered, centered)) / centered_energy)
    error = est_centered - projection
    signal_power = float(np.dot(projection, projection))
    # With zero projection SI-SDR is undefined (rather than a misleading 0 dB).
    si_sdr = None if signal_power <= 1e-30 else float(
        10 * np.log10(signal_power / max(float(np.dot(error, error)), 1e-30)))
    return {'snr_db': float(snr), 'si_sdr_db': si_sdr}


def main():
    import soundfile as sf
    from pystoi import stoi
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--reference-whole', type=Path, required=True)
    parser.add_argument('--reference-chunked', type=Path, required=True)
    parser.add_argument('--native-chunked', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    ref, rate = sf.read(args.reference, dtype='float32', always_2d=True)
    if rate != 16000 or ref.shape[1] != 1:
        raise ValueError('expected a 16 kHz mono reference')
    ref = ref[:, 0]
    results = {}
    for label, path in (('input', args.input), ('reference_whole', args.reference_whole),
                        ('reference_chunked', args.reference_chunked), ('native_chunked', args.native_chunked)):
        signal, sample_rate = sf.read(path, dtype='float32', always_2d=True)
        if sample_rate != rate or signal.shape != (len(ref), 1):
            raise ValueError(f'{label}: sample rate or channel/length mismatch')
        signal = signal[:, 0]
        stats = quality(ref, signal)
        stats['stoi'] = float(stoi(ref, signal, rate, extended=False))
        if not np.isfinite(stats['stoi']):
            raise ValueError(f'{label}: STOI is not finite')
        results[label] = stats
    args.output.write_text(json.dumps({'sample_rate': rate, 'samples': len(ref), 'results': results,
                                      'scope': 'single speech proxy plus synthetic noise; diagnostics only'}, indent=2) + '\n')
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
