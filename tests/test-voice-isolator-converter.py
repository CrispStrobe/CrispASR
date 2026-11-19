#!/usr/bin/env python3
"""Model-free guards for models/convert-voice-isolator-to-gguf.py (#602)."""

import importlib.util
import json
import struct
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("voice_isolator_converter",
                                              ROOT / "models" / "convert-voice-isolator-to-gguf.py")
CONVERTER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CONVERTER)

# Released config of inworld/voice-isolator denoiser.safetensors (pinned values).
RELEASED = {
    "band_plan": [[100, 1000], [250, 4000], [500, 8000], [1000, 16000], [2000, 24000]],
    "n_fft": 2048, "sample_rate": 48000,
}


class BandEdges(unittest.TestCase):

    def test_released_plan_has_46_contiguous_bands(self):
        edges = CONVERTER.band_edges(RELEASED)
        self.assertEqual(len(edges), 46)  # band_embedding is [46, 512] in both checkpoints
        self.assertEqual(edges[0], (0, 4))  # 100 Hz / 23.4375 Hz per bin -> 4 bins
        self.assertEqual(edges[-1][1], 1025)
        for (a, b), (c, _) in zip(edges, edges[1:]):
            self.assertLess(a, b)
            self.assertEqual(b, c)  # every bin exactly once

    def test_widths_sum_to_encoder_inputs(self):
        # sum of 2 * band width = 2 * bins: the encoders consume (re, im) of every bin once
        self.assertEqual(sum(2 * (b - a) for a, b in CONVERTER.band_edges(RELEASED)), 2 * 1025)


class Refusals(unittest.TestCase):

    def _write(self, path: Path, config: dict) -> None:
        header = {"__metadata__": {"config": json.dumps(config)},
                  "band_embedding": {"dtype": "F32", "shape": [1, 1], "data_offsets": [0, 4]}}
        raw = json.dumps(header).encode()
        path.write_bytes(struct.pack("<Q", len(raw)) + raw + b"\0" * 4)

    def test_unsupported_fields_are_refused(self):
        import sys
        for field, value in (("dual_mode", True), ("mode_film", True), ("n_fft", 1024), ("exclusive_axes", "both")):
            with tempfile.TemporaryDirectory() as d:
                src = Path(d) / "x.safetensors"
                self._write(src, dict(RELEASED, **{field: value}, block_types=["mamba"]))
                argv = sys.argv
                sys.argv = ["convert", "--input", str(src), "--output", str(Path(d) / "o.gguf")]
                try:
                    with self.assertRaises(SystemExit) as cm:
                        CONVERTER.main()
                    self.assertIn(field, str(cm.exception))
                finally:
                    sys.argv = argv


if __name__ == "__main__":
    unittest.main()
