"""Actual public buffer versus original constructor; no model/GPU simulation."""
import os
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from dia_model_precision import original_timescales
from dia_timescale_proof import fetch, original_buffer


class OriginalBufferProofTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(dir=os.environ['TMPDIR'])
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.root = Path(cls.temporary.name)
        cls.archive = fetch(cls.root / 'actual-original.zip')

    def test_actual_buffer_and_metadata_are_identical_to_original_constructor(self):
        source = os.environ['DIA_SOURCE']
        expected, metadata = original_timescales(source, 128)
        actual, provenance = original_buffer(self.archive, source, 128)
        np.testing.assert_array_equal(actual, expected)
        self.assertEqual(actual.tobytes(), expected.tobytes())
        self.assertEqual(actual.dtype, np.float32)
        self.assertEqual(provenance, metadata)
        self.assertEqual(list(provenance.items()), list(metadata.items()))

    def test_changed_archive_is_rejected(self):
        path = self.root / 'changed.zip'
        data = bytearray(self.archive.read_bytes()); data[100] ^= 1
        path.write_bytes(data)
        with self.assertRaisesRegex(ValueError, 'size/SHA'):
            original_buffer(path, os.environ['DIA_SOURCE'], 128)

    def test_other_head_dimension_is_rejected(self):
        for head in (64, 256, 128.0, True):
            with self.subTest(head=head):
                with self.assertRaisesRegex(ValueError, 'dimension'):
                    original_buffer(self.archive, os.environ['DIA_SOURCE'], head)

    def test_fetch_preserves_existing_path(self):
        before = self.archive.read_bytes()
        with self.assertRaisesRegex(ValueError, 'new destination'):
            fetch(self.archive)
        self.assertEqual(self.archive.read_bytes(), before)


if __name__ == '__main__':
    unittest.main()
