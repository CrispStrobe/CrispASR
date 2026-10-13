#!/usr/bin/env python3
"""Guard archive corruption, unsafe members and incomplete false successes."""
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from dia_proof_archives import extract_verified, fetch


class ProofArchiveTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=os.environ['TMPDIR'])
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.archive, self.destination = self.root / 'proof.zip', self.root / 'result'

    def make(self, member='heavy-test/data.bin', data=b'original', digest=None, duplicate=False, extra=False):
        row = dict(path='data.bin', bytes=len(data), sha256=digest or hashlib.sha256(data).hexdigest())
        with zipfile.ZipFile(self.archive, 'w') as zipped:
            zipped.writestr(member, data)
            if duplicate:
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore'); zipped.writestr(member, data)
            if extra: zipped.writestr('unexpected.txt', b'junk')
            zipped.writestr('ARCHIVE-MANIFEST.json', json.dumps(dict(files=[row])))
            zipped.writestr('tracked-evidence.json', '{}')
        return dict(bytes=self.archive.stat().st_size, sha256=hashlib.sha256(self.archive.read_bytes()).hexdigest(),
                    source_files=1, archive_member_prefix='heavy-test/', proof_folder='heavy-test')

    def test_correct_payload_and_old_destination_preserved(self):
        spec = self.make()
        result = extract_verified(self.archive, self.destination, spec)
        self.assertEqual((result / 'data.bin').read_bytes(), b'original')
        with self.assertRaisesRegex(ValueError, 'already exists'):
            extract_verified(self.archive, self.destination, spec)
        self.assertEqual((result / 'data.bin').read_bytes(), b'original')

    def test_corrupted_archive_rejected_before_extraction(self):
        spec = self.make(); data = bytearray(self.archive.read_bytes()); data[40] ^= 1; self.archive.write_bytes(data)
        with self.assertRaisesRegex(ValueError, 'immutable pin'):
            extract_verified(self.archive, self.destination, spec)
        self.assertFalse(self.destination.exists())

    def test_wrong_size_rejected(self):
        spec = self.make(); spec['bytes'] += 1
        with self.assertRaisesRegex(ValueError, 'immutable pin'):
            extract_verified(self.archive, self.destination, spec)
        self.assertFalse(self.destination.exists())

    def test_unsafe_path_rejected_even_with_matching_whole_zip_hash(self):
        spec = self.make(member='../outside')
        with self.assertRaisesRegex(ValueError, 'Unsafe'):
            extract_verified(self.archive, self.destination, spec)
        self.assertFalse((self.root / 'outside').exists())

    def test_duplicate_and_extra_members_rejected(self):
        for values in (dict(duplicate=True), dict(extra=True)):
            spec = self.make(**values)
            with self.assertRaises(ValueError):
                extract_verified(self.archive, self.destination, spec)
            self.assertFalse(self.destination.exists())

    def test_missing_manifest_rejected(self):
        spec = self.make()
        with zipfile.ZipFile(self.archive, 'w') as zipped:
            zipped.writestr('heavy-test/data.bin', b'original')
        spec['bytes'] = self.archive.stat().st_size
        spec['sha256'] = hashlib.sha256(self.archive.read_bytes()).hexdigest()
        with self.assertRaises(KeyError):
            extract_verified(self.archive, self.destination, spec)
        self.assertFalse(self.destination.exists())

    def test_payload_manifest_hash_rejected(self):
        spec = self.make(digest='0' * 64)
        with self.assertRaisesRegex(ValueError, 'payload'):
            extract_verified(self.archive, self.destination, spec)
        self.assertFalse(self.destination.exists())

    def test_unknown_proof_rejected_without_network(self):
        with self.assertRaisesRegex(ValueError, 'Unknown'):
            fetch('unknown', self.destination)
        self.assertFalse(self.destination.exists())


if __name__ == '__main__':
    unittest.main()
