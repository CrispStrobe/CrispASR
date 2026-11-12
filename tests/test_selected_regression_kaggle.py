"""Controls for the selected-regression Kaggle artifact cleanup."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

path = Path(__file__).resolve().parents[1] / 'tools/kaggle/selected-regression-20261112/selected.py'
spec = importlib.util.spec_from_file_location('selected_regression', path)
selected = importlib.util.module_from_spec(spec)
spec.loader.exec_module(selected)


class CleanupTest(unittest.TestCase):
    def test_removes_large_intermediates_and_preserves_results(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ('CrispASR', '_bootstrap', '.ccache', 'hf_cache', 'build', 'rebake-stage', 'results'):
                (root / name).mkdir()
                (root / name / 'artifact').write_text('data')
            (root / 'progress.jsonl').write_text('progress')
            selected.cleanup_workdir(root)
            self.assertEqual({p.name for p in root.iterdir()}, {'results', 'progress.jsonl'})
            self.assertEqual((root / 'results/artifact').read_text(), 'data')


if __name__ == '__main__':
    unittest.main()
