"""Controls for Ultra alignment parity comparison."""
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('ultra_audit', Path(__file__).resolve().parents[1] /
                                              'tools/ci-heavy/issue579_ultra_live.py')
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


class UltraAuditTest(unittest.TestCase):
    def test_accepts_matching_words_and_timestamps(self):
        ref = [{'word': 'Hello,', 'start': 0.0, 'end': 0.48}]
        got = [{'word': 'hello', 'start': 0.0, 'end': 0.48}]
        self.assertEqual(audit.compare(ref, got)['words'], 1)

    def test_rejects_missing_wrong_and_nonfinite_results(self):
        ref = [{'word': 'hello', 'start': 0.0, 'end': 0.48}]
        for got in ([], [{'word':'goodbye','start':0,'end':0.48}],
                    [{'word':'hello','start':0,'end':float('nan')}],
                    [{'word':'hello','start':0,'end':1.0}]):
            with self.subTest(got=got), self.assertRaises(AssertionError):
                audit.compare(ref, got)


if __name__ == '__main__':
    unittest.main()
