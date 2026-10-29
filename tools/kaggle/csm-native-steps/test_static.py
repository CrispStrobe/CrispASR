#!/usr/bin/env python3
"""Model-free checks of the hosted CSM profiler receipt gate."""
import importlib.util
from pathlib import Path
import sys
import unittest

sys.dont_write_bytecode = True
spec = importlib.util.spec_from_file_location("csm_native_steps", Path(__file__).with_name("run.py"))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def sample():
    header = "csm_step_profile: complete=1 attempts=2 accepted=1 eos=1 codebooks=32\n"
    counts = (("backbone_prefill", 1), ("backbone_step", 1), ("depth_prefill", 2), ("depth_step", 60))
    return header + "\n".join(f"csm_step_profile: {name} calls={count} total_us=123" for name, count in counts)


class ProfileGateTest(unittest.TestCase):
    def test_complete_profile(self):
        result = module.parse_profile(sample())
        self.assertEqual((result["attempts"], result["accepted"], result["eos"]), (2, 1, 1))
        self.assertEqual(result["stages"]["depth_step"]["calls"], 60)

    def test_rejects_partial_wrong_count_zero_time_and_duplicate(self):
        original = sample()
        for value in (original.replace("complete=1", "complete=0"),
                      original.replace("calls=60", "calls=59"),
                      original.replace("total_us=123", "total_us=0", 1),
                      original + "\n" + original,
                      original.replace("accepted=1", "accepted=2")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                module.parse_profile(value)


if __name__ == "__main__":
    unittest.main()
