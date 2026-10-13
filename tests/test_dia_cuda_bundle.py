#!/usr/bin/env python3
"""Weight-free guards against compiling the wrong Dia candidate for hardware tests."""
import importlib.util
import os
from pathlib import Path
import shutil
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('dia_cuda_bundle', ROOT / 'tools/ci-heavy/dia_cuda_bundle.py')
bundle = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bundle)


class RuntimePinTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(dir=os.environ['TMPDIR'])
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.names = tuple(bundle.pinned_runtime_sources(ROOT))
        for name in self.names:
            target = self.root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / name, target)

    def test_actual_candidate_sources_are_accepted(self):
        self.assertEqual(bundle.pinned_runtime_sources(self.root), bundle.pinned_runtime_sources(ROOT))

    def test_changed_actual_cache_precision_is_rejected(self):
        path = self.root / 'src/dia_tts.cpp'
        original = path.read_text()
        needle = 'cross_f32 ? GGML_TYPE_F32 : GGML_TYPE_F16'
        self.assertIn(needle, original)
        path.write_text(original.replace(needle, 'GGML_TYPE_F16', 1))
        with self.assertRaisesRegex(ValueError, 'candidate bytes'):
            bundle.pinned_runtime_sources(self.root)

    def test_changed_capture_dispatch_is_rejected(self):
        path = self.root / 'examples/cli/crispasr_dia_diff.h'
        original = path.read_text()
        needle = 'const bool capture_sa = capture_flag && std::strcmp(capture_flag, "1") == 0;'
        self.assertIn(needle, original)
        path.write_text(original.replace(needle, 'const bool capture_sa = true;', 1))
        with self.assertRaisesRegex(ValueError, 'candidate bytes'):
            bundle.pinned_runtime_sources(self.root)

    def test_missing_sampler_source_is_rejected(self):
        (self.root / 'src/dia_exponential_rng.h').unlink()
        with self.assertRaises(FileNotFoundError):
            bundle.pinned_runtime_sources(self.root)


if __name__ == '__main__':
    unittest.main()
