"""Weight-free controls for the final-window fused diagnostic candidate."""
import importlib.util
import json
from pathlib import Path
import unittest
import tempfile
from collections import namedtuple
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

    def test_live_process_tree_includes_descendants_only(self):
        with tempfile.TemporaryDirectory(dir=ROOT.parent) as directory:
            proc = Path(directory)
            for pid, parent, rss in [(10, 1, 20), (11, 10, 30), (12, 11, 40), (99, 1, 500)]:
                path = proc/str(pid)
                path.mkdir()
                (path/'status').write_text(f'PPid: {parent}\nVmRSS: {rss} kB\n')
            self.assertEqual(job.process_tree(proc, 10), {10: 20, 11: 30, 12: 40})
            with self.assertRaises(RuntimeError):
                job.process_tree(proc, 1234)

    def test_resource_limits_and_retained_outputs(self):
        usage = namedtuple('Usage', 'total used free')
        with tempfile.TemporaryDirectory(dir=ROOT.parent) as directory:
            root = Path(directory)
            preserved = root/'retained-reference.gguf'
            preserved.write_bytes(b'preserve')
            with patch.object(job, 'process_tree', return_value={1: 10, 2: 20}), patch.object(job.shutil, 'disk_usage', return_value=usage(100, 1, job.MIN_FREE_BYTES)):
                self.assertEqual(job.check_budget(root, root)['tree_rss_kib'], 30)
                with patch.object(job, 'MAX_TREE_RSS_KIB', 29):
                    with self.assertRaisesRegex(job.BudgetExceeded, 'live_tree_rss_limit'):
                        job.check_budget(root, root)
                with patch.object(job, 'MAX_OUTPUT_BYTES', 7):
                    with self.assertRaisesRegex(job.BudgetExceeded, 'output_size_limit'):
                        job.check_budget(root, root)
                with patch.object(job, 'MIN_FREE_BYTES', job.MIN_FREE_BYTES+1):
                    with self.assertRaisesRegex(job.BudgetExceeded, 'free_disk_reserve'):
                        job.check_budget(root, root)
            self.assertEqual(preserved.read_bytes(), b'preserve')

    def test_private_exception_and_helper_chatter_not_persisted(self):
        error = RuntimeError('https://private.example/signed?token=secret /private/account/path')
        rendered = json.dumps(job.failure_receipt(error))
        for private in ('private.example', 'secret', '/private'):
            self.assertNotIn(private, rendered)
        def noisy_helper():
            print('private account mount')
            return 'resolved-value'
        self.assertEqual(job.quiet_call(noisy_helper), 'resolved-value')

    def test_frozen_sources_and_existing_producer_attachment(self):
        self.assertEqual(job.SOURCE_REVISION, '529e073f8c9c99b688be5823e1be78443d50763e')
        metadata = json.loads((PATH.parent/'kernel-metadata.json').read_text())
        self.assertEqual(metadata['kernel_sources'], ['${KAGGLE_ACCOUNT}/crispasr-reuse-ten-second-20261112'])
        self.assertEqual(metadata['enable_gpu'], 'true')
        self.assertEqual(set(job.EXPECTED_WHEELS), {'mamba_ssm-2.2.4-cp313-cp313-linux_x86_64.whl', 'causal_conv1d-1.5.0.post8-cp313-cp313-linux_x86_64.whl'})


if __name__ == '__main__':
    unittest.main()
