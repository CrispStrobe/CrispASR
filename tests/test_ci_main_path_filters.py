"""Contract for ordered main-push paths in the heavy CI workflow."""
from fnmatch import fnmatchcase
from pathlib import Path
import re
import unittest

WORKFLOW = Path(__file__).resolve().parents[1] / '.github/workflows/ci.yml'


def paths_for(event):
    text = WORKFLOW.read_text()
    block = re.search(r'^  ' + re.escape(event) + r':\n(?P<body>(?:^(?:    .*|\s*)\n)+)', text, re.M)
    if not block:
        raise AssertionError(f'missing {event} event')
    paths = re.search(r'^    paths:\n(?P<body>(?:^      .*\n)+)', block['body'], re.M)
    if not paths:
        raise AssertionError(f'missing {event} paths')
    return re.findall(r"^      - '([^']+)'", paths['body'], re.M)


def selected(paths, changed):
    # The patterns in this workflow use ** only as a broad wildcard; for
    # these representative cases fnmatchcase has the same ordered behavior.
    for filename in changed:
        match = False
        for pattern in paths:
            excluded = pattern.startswith('!')
            if fnmatchcase(filename, pattern[1:] if excluded else pattern):
                match = not excluded
        if match:
            return True
    return False


class MainPathControls(unittest.TestCase):
    def test_docs_receipts_and_lanes_alone_are_excluded(self):
        paths = paths_for('push')
        for name in ('docs/historical/new-evidence.json', 'docs/models/a.json', 'docs/guide.md',
                     'lanes/plan.json', 'lanes/work/native.cpp', 'PLAN.md'):
            with self.subTest(path=name):
                self.assertFalse(selected(paths, [name]))

    def test_actual_fixture_inputs_and_mixed_code_are_retained(self):
        paths = paths_for('push')
        for name in ('docs/historical/data2vec-avx512-controls-2026-10-28.json',
                     'docs/historical/musa-sdk52-2026-10-26.json', 'src/whisper.cpp',
                     '.github/workflows/ci.yml', 'tests/test_ci_main_path_filters.py'):
            with self.subTest(path=name):
                self.assertTrue(selected(paths, [name]))
        self.assertTrue(selected(paths, ['docs/historical/new-evidence.json', 'src/whisper.cpp']))

    def test_pr_receipt_coverage_remains_broad(self):
        self.assertTrue(selected(paths_for('pull_request'), ['docs/historical/new-evidence.json']))


if __name__ == '__main__':
    unittest.main()
