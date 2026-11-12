"""Coverage outcomes must distinguish an intentional deferral from execution."""
import importlib.util
from pathlib import Path
import unittest
import tempfile
import os
from contextlib import redirect_stdout
from io import StringIO
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('extended_report', ROOT / 'tools/extended_regression_report.py')
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)


class ExtendedRegressionOutcome(unittest.TestCase):
    def test_busy_queue_is_explicit_zero_coverage_not_a_failure(self):
        result = report.classify('success', 'deferred', '[]', 'skipped')
        self.assertEqual(result['outcome'], 'deferred')
        self.assertEqual(result['selected_count'], 0)
        self.assertTrue(result['ok'])
        self.assertIn('No backends were tested', result['summary'])

    def test_selected_matrix_requires_actual_success(self):
        for regression in ('failure', 'cancelled', 'skipped', ''):
            with self.subTest(regression=regression):
                result = report.classify('success', 'selected', '["a", "b"]', regression)
                self.assertEqual(result['outcome'], 'failed')
                self.assertFalse(result['ok'])
        result = report.classify('success', 'selected', '["a", "b"]', 'success')
        self.assertEqual(result['outcome'], 'passed')
        self.assertEqual(result['selected_count'], 2)
        self.assertEqual(result['backends'], ['a', 'b'])
        self.assertIn('a, b', result['summary'])
        self.assertTrue(result['ok'])

    def test_selection_failure_malformed_or_contradictory_matrix_fails_closed(self):
        for args in [('failure', '', '', 'skipped'), ('success', 'selected', '[]', 'skipped'),
                     ('success', 'selected', 'invalid', 'success'),
                     ('success', 'selected', '[1]', 'success'),
                     ('success', 'selected', '["a", "a"]', 'success'),
                     ('success', 'deferred', '["a"]', 'skipped'),
                     ('success', 'deferred', '[]', 'success')]:
            with self.subTest(args=args):
                self.assertFalse(report.classify(*args)['ok'])


    def test_github_step_writes_machine_and_human_outcomes(self):
        with tempfile.TemporaryDirectory() as temp:
            output, summary = Path(temp) / 'output', Path(temp) / 'summary'
            variables = {'SELECTION_RESULT': 'success', 'COVERAGE_SELECTION': 'deferred',
                         'COVERAGE_MATRIX': '[]', 'REGRESSION_RESULT': 'skipped',
                         'GITHUB_OUTPUT': str(output), 'GITHUB_STEP_SUMMARY': str(summary)}
            with patch.dict(os.environ, variables, clear=True), redirect_stdout(StringIO()) as stdout:
                self.assertEqual(report.main(), 0)
            self.assertIn('outcome=deferred', output.read_text())
            self.assertIn('selected_count=0', output.read_text())
            self.assertIn('No backends were tested', summary.read_text())
            self.assertIn('::notice::', stdout.getvalue())
            variables.update({'COVERAGE_SELECTION': 'selected', 'COVERAGE_MATRIX': '["tada"]',
                              'REGRESSION_RESULT': 'skipped'})
            with patch.dict(os.environ, variables, clear=True), redirect_stdout(StringIO()) as stdout:
                self.assertEqual(report.main(), 1)
            self.assertIn('::error::', stdout.getvalue())



if __name__ == '__main__':
    unittest.main()
