"""Model-free controls for strict ten-second qualification result accounting."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

path = Path(__file__).resolve().parents[1]/'tools/kaggle/reuse-ten-second-20261112/qualification.py'
spec = importlib.util.spec_from_file_location('reuse_ten_second', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def log():
    return '\n'.join(f'mkl-avx2{suffix}/{stage:<22} PASS rel_rms=0 max_abs=0 cos_min=1 norm_ratio=1'
                     for suffix in ('', '-conditioned') for stage in module.EXPECTED_STAGES)


class QualificationControls(unittest.TestCase):
    def test_both_complete_arms_pass_with_diagnostic_rows_ignored(self):
        self.assertEqual(module.parse_rows('diagnostic/frontend_real DIAGNOSTIC rms=0\n'+log()),
                         {'unconditioned': 71, 'conditioned': 71})

    def test_missing_conditioned_arm_cannot_be_replaced_with_duplicates(self):
        unconditioned = '\n'.join(line for line in log().splitlines() if '-conditioned/' not in line)
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            module.parse_rows(unconditioned+'\n'+unconditioned)
        with self.assertRaisesRegex(ValueError, 'conditioned'):
            module.parse_rows(unconditioned)

    def test_missing_unknown_and_failed_stages_are_rejected(self):
        for bad in (log().replace('frontend_magnitude', 'unknown_stage', 1),
                    '\n'.join(log().splitlines()[1:]), log().replace(' PASS ', ' FAIL ', 1)):
            with self.subTest(bad=bad[:70]), self.assertRaises(ValueError):
                module.parse_rows(bad)

    def test_atomic_manifest_rejects_nan_without_overwriting_previous(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory)/'manifest.json'
            module.save_json(target, {'kept': 1})
            original = target.read_bytes()
            with self.assertRaises(ValueError):
                module.save_json(target, {'bad': float('nan')})
            self.assertEqual(target.read_bytes(), original)

    def test_live_memory_includes_children_and_grandchildren(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for pid, parent, rss in ((100, 1, 10), (101, 100, 20), (102, 101, 30), (103, 1, 999)):
                (root/str(pid)).mkdir()
                (root/str(pid)/'status').write_text(f'PPid:\t{parent}\nVmRSS:\t{rss} kB\n')
            self.assertEqual(module.live_tree_rss_kib(root, root_pid=100), 60)
            with self.assertRaisesRegex(RuntimeError, 'inspect'):
                module.live_tree_rss_kib(root, root_pid=999)

    def test_low_disk_budget_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.object(module.shutil, 'disk_usage', return_value=type('Usage', (), {'free': 1})()):
                with self.assertRaisesRegex(OSError, 'reserve'):
                    module.check_budget(root, root)

    def test_unfrozen_source_fails_before_execution(self):
        with patch.object(module, 'SOURCE_REVISION', 'UNFROZEN'), patch.object(module, 'run') as run:
            with self.assertRaisesRegex(ValueError, 'Freeze'):
                module.main()
            run.assert_not_called()


if __name__ == '__main__':
    unittest.main()
