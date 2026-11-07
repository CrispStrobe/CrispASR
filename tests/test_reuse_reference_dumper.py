"""Model-free controls for the RE-USE crispasr-diff reference contract."""
import importlib.util
from pathlib import Path
import unittest
import numpy as np

path = Path(__file__).resolve().parents[1] / 'tools/reference_backends/reuse.py'
spec = importlib.util.spec_from_file_location('reuse_reference_dumper', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

class ReferenceContract(unittest.TestCase):
    def test_all_model_boundaries_are_registered_once(self):
        stages = module.DEFAULT_STAGES
        self.assertEqual(len(stages), len(set(stages)))
        for index in range(6):
            self.assertIn(f'encoder_layer_{index}', stages)
        for index in range(30):
            self.assertIn(f'block_time_{index}', stages)
            self.assertIn(f'block_{index}', stages)
        for name in ('raw_audio', 'frontend_magnitude', 'frontend_phase', 'amplitude', 'output_phase', 'waveform'):
            self.assertIn(name, stages)

    def test_invalid_audio_and_unknown_stage_fail_before_loading_model(self):
        for audio in (np.zeros(320, np.float32), np.zeros(401, np.float32), np.full(400, np.nan, np.float32), np.zeros((400, 2), np.float32)):
            with self.subTest(length=len(audio)), self.assertRaises(ValueError):
                module.dump('/nonexistent', audio, {'waveform'})
        with self.assertRaisesRegex(ValueError, 'Unknown'):
            module.dump('/nonexistent', np.zeros(400, np.float32), {'invented'})

if __name__ == '__main__':
    unittest.main()
