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

    def test_startup_load_must_recover_before_transfers(self):
        samples = iter([dict(load=(6.13, 19, 12), available_kib=3 * 1024**2,
                             disk_free=100 * 1024**3, cpus=3),
                        dict(load=(5, 19, 12), available_kib=3 * 1024**2,
                             disk_free=100 * 1024**3, cpus=3)])
        receipt, pauses, saves = {}, [], []
        PRESERVE.wait_for_resources(Path('.'), receipt, lambda: saves.append(True),
                                    probe=lambda _: next(samples), pause=pauses.append, attempts=2)
        self.assertEqual(pauses, [10])
        self.assertEqual(len(saves), 2)
        self.assertEqual(receipt['resources']['load'][0], 5)

    def test_persistent_load_or_memory_pressure_stops_at_wait_bound(self):
        for load, memory in [(6.13, 3 * 1024**2), (1, 1024**2)]:
            with self.subTest(load=load, memory=memory):
                pauses, receipt = [], {}
                with self.assertRaises(RuntimeError):
                    PRESERVE.wait_for_resources(Path('.'), receipt, lambda: None,
                        probe=lambda _: dict(load=(load, 0, 0), available_kib=memory,
                                             disk_free=100 * 1024**3, cpus=3),
                        pause=pauses.append, attempts=3)
                self.assertEqual(pauses, [10, 10])
                self.assertEqual(len(receipt['resource_samples']), 3)

    def test_insufficient_disk_stops_without_waiting_or_cleanup(self):
        pauses = []
        with self.assertRaises(RuntimeError):
            PRESERVE.wait_for_resources(Path('.'), {}, lambda: None,
                probe=lambda _: dict(load=(1, 0, 0), available_kib=3 * 1024**2,
                                     disk_free=7 * 1024**3, cpus=3), pause=pauses.append)
        self.assertEqual(pauses, [])


if __name__ == '__main__':
    unittest.main()
