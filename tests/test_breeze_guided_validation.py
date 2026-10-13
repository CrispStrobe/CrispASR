"""Deletion controls for the original-reference guided acceptance matrix."""
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('breeze_reference', ROOT / 'tools/reference_backends/breeze_tts_2.py')
reference = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reference)
CASES = ('plain', 'clone', 'design-s0', 'design-s1', 'design-s4', 'direct-s0', 'direct-s1', 'direct-s4')


class GuidedValidationTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.ref, self.cpp = self.root / 'ref', self.root / 'cpp'
        self.ref.mkdir()
        (self.ref / 'meta.json').write_text(json.dumps({'guided': {case: {} for case in CASES}}))
        for case in CASES:
            stages = {'prompt_input_ids': np.array([1, 2], dtype=np.int32),
                      'backbone_logits_frame0': np.ones(2052, dtype=np.float32),
                      'codes': np.zeros((24, 16), dtype=np.int32)}
            if case.endswith(('s0', 's4')):
                stages['neg_prompt_input_ids'] = np.array([1], dtype=np.int32)
            for side in (self.ref, self.cpp):
                directory = side / 'guided' / case
                directory.mkdir(parents=True)
                for name, values in stages.items():
                    np.save(directory / (name + '.npy'), values)
        self.env = patch.dict(os.environ, BREEZE_FIXTURE_DIR=str(self.ref))
        self.env.start()

    def tearDown(self):
        self.env.stop()
        self.tmp.cleanup()

    def compare(self):
        with contextlib.redirect_stdout(io.StringIO()):
            return reference.compare_guided(self.cpp, download=False)

    def test_complete_matrix_passes(self):
        self.assertEqual(self.compare(), 0)

    def test_missing_reference_case_fails(self):
        directory = self.ref / 'guided/direct-s4'
        for p in directory.iterdir():
            p.unlink()
        directory.rmdir()
        self.assertNotEqual(self.compare(), 0)

    def test_missing_native_stages_fail(self):
        for p in (self.cpp / 'guided/plain').iterdir():
            p.unlink()
        self.assertNotEqual(self.compare(), 0)

    def test_missing_reference_stage_fails(self):
        (self.ref / 'guided/direct-s4/backbone_logits_frame0.npy').unlink()
        self.assertNotEqual(self.compare(), 0)

    def test_missing_negative_prompt_fails(self):
        for side in (self.ref, self.cpp):
            (side / 'guided/design-s4/neg_prompt_input_ids.npy').unlink()
        self.assertNotEqual(self.compare(), 0)

    def test_empty_codes_fail_even_when_equal(self):
        for side in (self.ref, self.cpp):
            np.save(side / 'guided/plain/codes.npy', np.empty((0, 16), dtype=np.int32))
        self.assertNotEqual(self.compare(), 0)

    def test_partial_reference_generation_fails(self):
        (self.ref / 'meta.json').write_text(json.dumps({'guided_errors': {'direct-s4': 'generation failed'}}))
        self.assertNotEqual(self.compare(), 0)

    def test_numeric_and_code_mismatches_fail(self):
        np.save(self.cpp / 'guided/design-s4/backbone_logits_frame0.npy', np.full(2052, 2, dtype=np.float32))
        self.assertNotEqual(self.compare(), 0)
        np.save(self.cpp / 'guided/plain/codes.npy', np.ones((24, 16), dtype=np.int32))
        self.assertNotEqual(self.compare(), 0)


if __name__ == '__main__':
    unittest.main()
