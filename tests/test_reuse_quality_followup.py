"""Weight-free controls for the final-window fused diagnostic candidate."""
import importlib.util
import json
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT/'tools/kaggle/reuse-quality-followup-20261112/qualification.py'
spec = importlib.util.spec_from_file_location('quality_followup', PATH)
job = importlib.util.module_from_spec(spec)
spec.loader.exec_module(job)


class Controls(unittest.TestCase):
    def test_exact_global_noise_then_final_window(self):
        clean, noisy, block = job.exact_inputs(ROOT)
        self.assertEqual(block.shape, (80000,))
        np.testing.assert_array_equal(block, noisy[80000:160000])
        self.assertEqual(job.hashlib.sha256(block.tobytes()).hexdigest(), job.WINDOW_HASH)
        # Renormalizing noise on a cropped 5s window is a different experiment.
        locally_generated = np.random.default_rng(579).standard_normal(80000).astype(np.float32)
        self.assertFalse(np.array_equal(block-clean[80000:], locally_generated))

    def test_changed_source_input_rejected(self):
        # A plausible shape cannot bypass the original final-window digest.
        import soundfile
        with patch.object(soundfile, 'read', return_value=(np.ones((160000, 1), dtype=np.float32), 16000)):
            with self.assertRaisesRegex(ValueError, 'reconstruction mismatch'):
                job.exact_inputs(ROOT)

    def test_scale_mismatch_visible_despite_cosine(self):
        result = job.compare_arrays(np.ones((2, 3)), np.ones((2, 3))*2)
        self.assertAlmostEqual(result['cosine'], 1)
        self.assertAlmostEqual(result['rel_rms'], 1)
        self.assertAlmostEqual(result['actual_norm']/result['reference_norm'], 2)

    def test_missing_shape_or_nonfinite_rejected(self):
        for actual in (np.zeros(4), np.array([0, 0, float('nan')])):
            with self.assertRaises(ValueError):
                job.compare_arrays(np.ones(3), actual)

    def test_all_72_stages_unique(self):
        self.assertEqual(len(job.STAGES), 72)
        self.assertEqual(len(set(job.STAGES)), 72)
        self.assertIn('block_time_29', job.STAGES)
        self.assertIn('waveform', job.STAGES)

    def test_frozen_sources_and_existing_producer_attachment(self):
        self.assertEqual(job.SOURCE_REVISION, '529e073f8c9c99b688be5823e1be78443d50763e')
        metadata = json.loads((PATH.parent/'kernel-metadata.json').read_text())
        self.assertEqual(metadata['kernel_sources'], ['${KAGGLE_ACCOUNT}/crispasr-reuse-ten-second-20261112'])
        self.assertEqual(metadata['enable_gpu'], 'true')
        self.assertEqual(set(job.EXPECTED_WHEELS), {'mamba_ssm-2.2.4-cp313-cp313-linux_x86_64.whl', 'causal_conv1d-1.5.0.post8-cp313-cp313-linux_x86_64.whl'})


if __name__ == '__main__':
    unittest.main()
