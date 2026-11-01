#!/usr/bin/env python3
"""Model-free checks for bounded reference archives and their failure controls."""

import importlib.util
import math
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from reference_backends.mini_bs_roformer import DEFAULT_STAGES, stage_shapes

HAS_ARRAY_TOOLS = all(importlib.util.find_spec(name) is not None for name in ("numpy", "gguf"))


class LayoutTests(unittest.TestCase):
    def test_full_chunk_has_all_original_boundaries(self):
        config = SimpleNamespace(stft_n_fft=4096, stft_n_fft_out=2048, stft_hop_length=512,
                                 time_conv_length=4, freqs_per_bands=[1] * 80,
                                 num_input_channels=2, hidden_size=512, register_token_num=4,
                                 num_stems=4, band_proj_size=256, num_hidden_layers=6)
        shapes = stage_shapes(config, 352800)
        self.assertEqual(set(shapes), set(DEFAULT_STAGES))
        self.assertEqual(len(shapes), 25)
        self.assertEqual(shapes["stft_input"], (690, 8196))
        self.assertEqual(shapes["layer5_freq"], (177, 84, 512))
        self.assertEqual(shapes["mask_raw"], (4, 690, 4100))
        self.assertEqual(sum(math.prod(shape) * 4 for shape in shapes.values()), 572_284_864)
        with self.assertRaises(ValueError):
            stage_shapes(config, 2048)
        with self.assertRaises(ValueError):
            stage_shapes(config, 352800, ["unknown"])


@unittest.skipUnless(HAS_ARRAY_TOOLS, "numpy and gguf are needed for archive round trips")
class ArchiveTests(unittest.TestCase):
    def setUp(self):
        import numpy as np
        from reference_backends._streaming_gguf import StreamingGGUFArchive

        self.np = np
        self.writer = StreamingGGUFArchive
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)

    def tearDown(self):
        self.directory.cleanup()

    def test_stream_matches_existing_writer_byte_for_byte(self):
        from dump_reference import write_gguf_archive

        np = self.np
        captures = {
            "z_strided": np.arange(120, dtype=np.float32).reshape(4, 5, 6).transpose(2, 0, 1),
            "a_half": np.linspace(-2, 2, 21, dtype=np.float16).reshape(3, 7),
            "m_wide": np.arange(48, dtype=np.float64).reshape(2, 24) / 17,
        }
        metadata = {"backend": "mini-bs-roformer", "n_samples": 32, "verified": True, "score": 1.0}
        expected, actual = self.root / "expected.gguf", self.root / "actual.gguf"
        write_gguf_archive(captures, metadata, expected)
        with self.writer(actual, {name: x.shape for name, x in captures.items()}, metadata,
                         block_bytes=28) as archive:
            for name, value in captures.items():
                archive.capture(name, value)
            self.assertLessEqual(archive.max_block_bytes, 28)
        self.assertEqual(actual.read_bytes(), expected.read_bytes())
        self.assertEqual(list(self.root.glob("*.partial")), [])

    def test_missing_stage_cannot_publish(self):
        output = self.root / "incomplete.gguf"
        with self.assertRaisesRegex(RuntimeError, "missing stages"):
            with self.writer(output, {"a": (2,), "b": (2,)}, {}) as archive:
                archive.capture("a", self.np.zeros(2, dtype=self.np.float32))
        self.assertFalse(output.exists())
        self.assertEqual(list(self.root.iterdir()), [])

    def test_invalid_stage_preserves_existing_archive(self):
        for kind in ("unknown", "shape", "duplicate", "nonfinite"):
            with self.subTest(kind=kind):
                output = self.root / "existing.gguf"
                output.write_bytes(b"original complete artifact")
                with self.assertRaises(ValueError):
                    with self.writer(output, {"a": (2,)}, {}) as archive:
                        if kind == "unknown":
                            archive.capture("unknown", self.np.zeros(2))
                        elif kind == "shape":
                            archive.capture("a", self.np.zeros(3))
                        elif kind == "duplicate":
                            archive.capture("a", self.np.zeros(2))
                            archive.capture("a", self.np.zeros(2))
                        else:
                            archive.capture("a", self.np.array([0.0, self.np.nan]))
                self.assertEqual(output.read_bytes(), b"original complete artifact")
                self.assertEqual(list(self.root.iterdir()), [output])


if __name__ == "__main__":
    unittest.main()
