"""Audit retained RE-USE waveforms without loading or executing a model."""
import argparse
import hashlib
import json
from pathlib import Path
import tempfile
from typing import Callable, Mapping

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import signal

RATE = 16000
SAMPLES = 160000
WINDOW = 80000
STARTS = (0, 40000, 80000)
INPUT_NAMES = ('input-proxy.wav', 'input-noisy.wav', 'output-reference-chunked.wav')
ESTOI_SEED = 579


def metrics(reference: ArrayLike, estimate: ArrayLike, *, stoi_scorer=None) -> dict:
    """Aligned waveform metrics; reject undefined references, retain zero outputs.

    STOI/ESTOI are optional. Their injected scorer runs with a fixed seed while
    preserving the caller's NumPy RNG state (ESTOI adds normalization jitter).
    """
    ref = np.asarray(reference, dtype=np.float64)
    est = np.asarray(estimate, dtype=np.float64)
    if (ref.ndim != 1 or not ref.size or est.shape != ref.shape
            or not np.isfinite(ref).all() or not np.isfinite(est).all()):
        raise ValueError('reference and estimate must be nonempty aligned finite mono arrays')
    energy = float(np.dot(ref, ref))
    centered = ref - ref.mean()
    centered_energy = float(np.dot(centered, centered))
    if energy <= 1e-20 or centered_energy <= 1e-20:
        raise ValueError('reference must have nonzero energy and variance')
    gain = np.dot(centered, est - est.mean()) / centered_energy
    projection = gain * centered
    error = est - est.mean() - projection
    projection_energy = float(np.dot(projection, projection))
    nperseg = min(512, len(ref))
    noverlap = min(384, nperseg - 1)
    _, _, ref_spectrum = signal.stft(ref, RATE, nperseg=nperseg, noverlap=noverlap)
    _, _, est_spectrum = signal.stft(est, RATE, nperseg=nperseg, noverlap=noverlap)
    result = {
        'snr_db': float(10 * np.log10(energy / max(float(np.dot(ref-est, ref-est)), 1e-30))),
        'si_sdr_db': None if projection_energy <= 1e-30 else float(
            10 * np.log10(projection_energy / max(float(np.dot(error, error)), 1e-30))),
        'gain_projection': float(gain),
        'rms_ratio': float(np.linalg.norm(est) / np.linalg.norm(ref)),
        'spectral_convergence': float(
            np.linalg.norm(abs(ref_spectrum)-abs(est_spectrum)) / np.linalg.norm(abs(ref_spectrum))),
    }
    if stoi_scorer is not None:
        state = np.random.get_state()
        try:
            np.random.seed(ESTOI_SEED)
            result['stoi'] = float(stoi_scorer(ref, est, RATE))
            result['estoi'] = float(stoi_scorer(ref, est, RATE, extended=True))
        finally:
            np.random.set_state(state)
        if not np.isfinite([result['stoi'], result['estoi']]).all():
            raise ValueError('STOI scorer returned nonfinite metrics')
    return result


def metrics_with_intelligibility(reference: ArrayLike, estimate: ArrayLike) -> dict:
    from pystoi import stoi
    return metrics(reference, estimate, stoi_scorer=stoi)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read_audio(path: Path) -> NDArray[np.float64]:
    import soundfile as sf
    audio, rate = sf.read(path, dtype='float64')
    if rate != RATE or audio.shape != (SAMPLES,) or not np.isfinite(audio).all():
        raise ValueError(f'{path.name}: expected {SAMPLES} finite mono samples at {RATE} Hz')
    return audio


def read_reference_window(path: Path) -> tuple[np.ndarray, np.ndarray]:
    # Import lazily: unit tests and numerical metrics need no GGUF dependency.
    from gguf import GGUFReader
    reader = GGUFReader(path)
    tensors = {}
    for tensor in reader.tensors:
        if tensor.name in ('raw_audio', 'waveform'):
            if tensor.name in tensors:
                raise ValueError(f'{path.name}: duplicate {tensor.name} tensor')
            tensors[tensor.name] = np.array(tensor.data, copy=True)
    if set(tensors) != {'raw_audio', 'waveform'}:
        raise ValueError(f'{path.name}: missing raw_audio or waveform tensor')
    return tensors['raw_audio'], tensors['waveform']


def validate_reference_window(raw: ArrayLike, waveform: ArrayLike,
                              expected_input: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    arrays = [np.asarray(value) for value in (raw, waveform)]
    for name, array in zip(('raw_audio', 'waveform'), arrays):
        if (array.shape != (WINDOW,) or array.dtype != np.dtype('float32')
                or not np.isfinite(array).all()):
            raise ValueError(f'{name}: expected {WINDOW} finite float32 mono samples')
    expected = np.asarray(expected_input, dtype=np.float32)
    if expected.shape != (WINDOW,) or not np.array_equal(arrays[0], expected):
        raise ValueError('reference raw_audio differs from the matching noisy input window')
    return tuple(np.ascontiguousarray(array) for array in arrays)


def atomic_write(path: Path, writer: Callable[[Path], None]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix='.'+path.name, suffix='.tmp', delete=False) as stream:
        temporary = Path(stream.name)
    try:
        writer(temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def write_json(path: Path, value: dict) -> None:
    text = json.dumps(value, indent=2, allow_nan=False) + '\n'
    atomic_write(path, lambda temporary: temporary.write_text(text))


def write_audio(path: Path, audio: np.ndarray) -> None:
    import soundfile as sf
    atomic_write(path, lambda temporary: sf.write(temporary, audio, RATE, format='WAV', subtype='FLOAT'))


def audit(input_dir: Path, output_dir: Path, *, expected_input_hashes: Mapping[str, str] | None = None,
          reference_loader: Callable = read_reference_window,
          metric_function: Callable = metrics_with_intelligibility) -> dict:
    """Validate inputs, score saved windows, and atomically checkpoint the report."""
    input_dir, output_dir = Path(input_dir), Path(output_dir)
    hashes = {name: sha256_file(input_dir/name) for name in INPUT_NAMES}
    if expected_input_hashes is not None:
        if set(expected_input_hashes) != set(INPUT_NAMES):
            raise ValueError('expected input hashes must identify exactly the three input WAVs')
        for name, digest in hashes.items():
            if digest != expected_input_hashes[name]:
                raise ValueError(f'{name}: input SHA-256 mismatch')
    clean, noisy, stitched = [read_audio(input_dir/name) for name in INPUT_NAMES]
    report = {
        'estoi_normalization_seed': ESTOI_SEED,
        'input_hashes': hashes,
        'scope': 'Saved waveform audit; one JFK speech proxy, no new model runs or independent voices',
        'segments': [], 'windows': [], 'overlaps': [],
    }
    segments = ((0,160000), (0,80000), (80000,160000), (0,40000),
                (40000,80000), (80000,120000), (120000,160000))
    for start, end in segments:
        report['segments'].append({
            'start': start, 'end': end,
            'input': metric_function(clean[start:end], noisy[start:end]),
            'output': metric_function(clean[start:end], stitched[start:end]),
        })
    waves = []
    for index, start in enumerate(STARTS):
        raw, wave = reference_loader(input_dir/f'window-{index}-reference.gguf')
        raw, wave = validate_reference_window(raw, wave, noisy[start:start+WINDOW])
        waves.append(wave)
        write_audio(output_dir/f'window-{index}-reference.wav', wave)
        report['windows'].append({
            'index': index, 'start': start,
            'input': metric_function(clean[start:start+WINDOW], raw),
            'output': metric_function(clean[start:start+WINDOW], wave),
            'raw_input_matches': True,
            'raw_audio_sha256': hashlib.sha256(raw.tobytes()).hexdigest(),
            'waveform_sha256': hashlib.sha256(wave.tobytes()).hexdigest(),
        })
    for index in range(2):
        left, right = waves[index][40000:], waves[index+1][:40000]
        clean_region = clean[(index+1)*40000:(index+2)*40000]
        left_metrics = metric_function(clean_region, left)
        right_metrics = metric_function(clean_region, right)
        norm = np.linalg.norm(clean_region)
        if norm <= 1e-10:
            raise ValueError('overlap reference must have nonzero energy')
        report['overlaps'].append({
            'left_index': index, 'right_index': index+1,
            'relative_disagreement': float(np.linalg.norm(left-right) / norm),
            'left': left_metrics, 'right': right_metrics,
        })
    write_json(output_dir/'saved-waveform-audit.json', report)
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--expected-receipt', type=Path,
                        help='Prior audit receipt whose three input WAV hashes must match')
    args = parser.parse_args(argv)
    expected = None
    if args.expected_receipt:
        expected = json.loads(args.expected_receipt.read_text())['input_hashes']
    report = audit(args.input_dir, args.output_dir, expected_input_hashes=expected)
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
