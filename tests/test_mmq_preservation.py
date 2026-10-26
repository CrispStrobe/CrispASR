"""Archive safety and no-public-promotion controls; no network or binaries."""
import copy
import importlib.util
import io
from pathlib import Path
import unittest
import zipfile

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('preserve', ROOT / 'tools/ci-heavy/preserve_cuda126_mmq.py')
PRESERVE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PRESERVE)


class PreservationTest(unittest.TestCase):
    def archive(self, entries):
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, 'w') as archive:
            for name, data in entries:
                archive.writestr(name, data)
        buffer.seek(0)
        return zipfile.ZipFile(buffer)

    def test_windows_member_separators_preserve_safe_identity(self):
        with self.archive([('package\\LICENSE', b'notice')]) as archive:
            entries = PRESERVE.members(archive, 100)
            self.assertEqual(PRESERVE.safe_name(entries[0].filename), 'package/LICENSE')

    def test_parent_absolute_and_drive_paths_are_rejected(self):
        for name in ('../outside', 'pkg/../../outside', '/absolute', 'C:\\outside', '\\\\host\\share'):
            with self.subTest(name=name), self.assertRaises(ValueError):
                PRESERVE.safe_name(name)

    def test_normalized_duplicate_paths_are_rejected(self):
        with self.archive([('pkg/LICENSE', b'one'), ('pkg\\LICENSE', b'two')]) as archive:
            with self.assertRaises(AssertionError):
                PRESERVE.members(archive, 100)

    def test_uncompressed_archive_limit_is_enforced(self):
        with self.archive([('payload', b'x' * 100)]) as archive:
            with self.assertRaises(AssertionError):
                PRESERVE.members(archive, 10)

    def test_only_owned_unpublished_experiment_draft_is_accepted(self):
        good = dict(draft=True, prerelease=True, target_commitish=PRESERVE.SOURCE, body=PRESERVE.MARKER)
        PRESERVE.require_draft(good)
        for key, value in [('draft', False), ('prerelease', False),
                           ('target_commitish', 'different-source'), ('body', 'unrelated draft'), ('body', None)]:
            bad = copy.deepcopy(good)
            bad[key] = value
            with self.subTest(key=key, value=value), self.assertRaises(RuntimeError):
                PRESERVE.require_draft(bad)


if __name__ == '__main__':
    unittest.main()
