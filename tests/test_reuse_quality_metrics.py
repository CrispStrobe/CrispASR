"""Independent controls for RE-USE's aligned waveform quality diagnostics."""
import importlib.util
import json
import subprocess
import sys
import tempfile
from pathlib import Path
import unittest
try:
    import numpy as np
except ImportError:
    raise SystemExit(77)

path = Path(__file__).resolve().parents[1] / 'tools/ci-heavy/issue579_reuse/quality_metrics.py'
spec = importlib.util.spec_from_file_location('reuse_quality', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class QualityTests(unittest.TestCase):
    def test_identity_and_scale_have_expected_snr_and_scale_invariance(self):
        x = np.array([-1.0, -0.5, 0.5, 1.0])
        identity = module.quality(x, x)
        scaled = module.quality(x, x * 0.5)
        self.assertGreater(identity['snr_db'], 200)
        self.assertAlmostEqual(scaled['snr_db'], 6.020599913, places=6)
        self.assertGreater(scaled['si_sdr_db'], 200)

    def test_cli_measures_four_aligned_waveforms(self):
        try:
            import soundfile as sf
            import pystoi  # noqa: F401
        except ImportError:
            self.skipTest('soundfile and pystoi are needed for the CLI control')
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            t = np.arange(16000, dtype=np.float64) / 16000
            x = (0.2 * np.sin(2 * np.pi * 190 * t) + 0.1 * np.sin(2 * np.pi * 390 * t)).astype(np.float32)
            for name in ('ref', 'input', 'whole', 'chunk', 'native'):
                sf.write(root / f'{name}.wav', x, 16000, subtype='FLOAT')
            output = root / 'report.json'
            subprocess.run([sys.executable, str(path), '--reference', str(root / 'ref.wav'),
                            '--input', str(root / 'input.wav'), '--reference-whole', str(root / 'whole.wav'),
                            '--reference-chunked', str(root / 'chunk.wav'), '--native-chunked', str(root / 'native.wav'),
                            '--output', str(output)], check=True, capture_output=True, text=True)
            report = json.loads(output.read_text())
            self.assertEqual(set(report['results']), {'input', 'reference_whole', 'reference_chunked', 'native_chunked'})
            self.assertGreater(report['results']['native_chunked']['stoi'], 0.99)

    def test_noise_reduces_si_sdr_and_bad_input_fails(self):
        x = np.array([-1.0, -0.5, 0.5, 1.0])
        self.assertLess(module.quality(x, x + np.array([0.2, -0.2, 0.2, -0.2]))['si_sdr_db'], 30)
        self.assertIsNone(module.quality(x, np.zeros_like(x))['si_sdr_db'])
        for ref, est in (([], []), ([0, 0], [0, 0]), ([1, 1], [1, 1]),
                         ([1, 2], [1]), ([1, float('nan')], [1, 2])):
            with self.subTest(ref=ref), self.assertRaises(ValueError):
                module.quality(ref, est)


if __name__ == '__main__':
    unittest.main()
