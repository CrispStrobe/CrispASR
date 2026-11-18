#!/usr/bin/env python3
"""Weight-free native base-TTS selection versus the serving masked argmax."""
import ctypes
import math
from pathlib import Path
import random
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
VALID = (151643, 151652, 151653, 151654)


class TokenConstraint(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        compiler = shutil.which("c++")
        if compiler is None:
            raise unittest.SkipTest("C++ compiler unavailable")
        cls.temp = tempfile.TemporaryDirectory(prefix="vibevoice-token-")
        root = Path(cls.temp.name)
        source = root / "probe.cpp"
        source.write_text('''#include "core/vibevoice_tts_generation.h"
extern "C" int select_token(const float* x, size_t n, bool constrain) {
    return core_vibevoice_tts::select_base_token(x, n, constrain);
}
''')
        lib = root / "probe.so"
        subprocess.run([compiler, "-std=c++17", "-shared", "-fPIC", "-I", str(ROOT / "src"),
                        str(source), "-o", str(lib)], check=True, capture_output=True)
        cls.lib = ctypes.CDLL(str(lib))
        cls.select = cls.lib.select_token
        cls.select.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_size_t, ctypes.c_bool]
        cls.select.restype = ctypes.c_int

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def pick(self, values, constrain=True):
        data = (ctypes.c_float * len(values))(*values)
        return self.select(data, len(values), constrain)

    def test_non_speech_peak_is_masked(self):
        values = [-30.] * 152064
        values[17] = 100.
        values[151654] = -1.
        self.assertEqual(self.pick(values, False), 17)
        self.assertEqual(self.pick(values), 151654)

    def test_each_control_can_win(self):
        for winner in VALID:
            with self.subTest(winner=winner):
                values = [10.] * 152064  # all disallowed rows beat controls
                for token in VALID:
                    values[token] = -20.
                values[winner] = -2.
                self.assertEqual(self.pick(values), winner)

    def test_ties_choose_lowest_vocabulary_id(self):
        values = [0.] * 152064
        self.assertEqual(self.pick(values), min(VALID))
        values[151643] = -1.
        self.assertEqual(self.pick(values), 151652)
        self.assertEqual(self.pick(values, False), 0)

    def test_full_mask_oracle(self):
        rng = random.Random(589)
        for _ in range(12):
            values = [rng.uniform(-10., 10.) for _ in range(152064)]
            # Serving processor: scores + {0 for controls, -inf otherwise}.
            masked = [value if i in VALID else -math.inf for i, value in enumerate(values)]
            self.assertEqual(self.pick(values), max(range(len(masked)), key=masked.__getitem__))

    def test_no_vocabulary_is_rejected(self):
        self.assertEqual(self.pick([]), -1)
        self.assertEqual(self.pick([0.] * 10), 0)

    def test_masked_nonfinite_scores_follow_argmax(self):
        values = [-math.inf] * 152064
        self.assertEqual(self.pick(values), 0)
        values[151654] = 0.
        self.assertEqual(self.pick(values), 151654)
        values[5] = math.nan
        self.assertEqual(self.pick(values), 5)
        values[4] = math.inf  # +inf plus the -inf mask is NaN
        self.assertEqual(self.pick(values), 4)


if __name__ == "__main__":
    unittest.main()
