"""convert-silero-vad-to-ggml.py version parsing (PR #612: silero-vad 6.3 reports "6.3")."""
import importlib.util
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "convert_silero", Path(__file__).resolve().parents[1] / "models" / "convert-silero-vad-to-ggml.py")
conv = importlib.util.module_from_spec(spec)
spec.loader.exec_module(conv)  # silero_vad is imported lazily, so this needs no torch


class ParseVersion(unittest.TestCase):
    def test_three_parts(self):
        self.assertEqual(conv.parse_version("6.2.0"), (6, 2, 0))

    def test_two_parts(self):
        self.assertEqual(conv.parse_version("6.3"), (6, 3, 0))

    def test_one_part_and_suffix(self):
        self.assertEqual(conv.parse_version("7"), (7, 0, 0))
        self.assertEqual(conv.parse_version("6.3.1rc1"), (6, 3, 1))

    def test_garbage_rejected(self):
        with self.assertRaises(ValueError):
            conv.parse_version("dev")


if __name__ == "__main__":
    unittest.main()
