"""Small-file controls for independently retained RE-USE reference archives."""
import hashlib
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

path = Path(__file__).resolve().parents[1] / 'tools/ci-heavy/issue579_reuse/retain_reference.py'
spec = importlib.util.spec_from_file_location('reuse_reference_retention', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class ReferenceRetentionTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.source = self.root / 'source.gguf'
        self.destination = self.root / 'reference.gguf'
        self.payload = b'independent small reference\x00\xff'
        self.source.write_bytes(self.payload)
        self.destination.write_bytes(b'previous checkpoint')

    def assert_unchanged(self):
        self.assertEqual(self.source.read_bytes(), self.payload)
        self.assertEqual(self.destination.read_bytes(), b'previous checkpoint')
        self.assertEqual(sorted(p.name for p in self.root.iterdir()), ['reference.gguf', 'source.gguf'])

    def test_success_verifies_before_atomic_publish_and_preserves_source(self):
        replace = module.os.replace
        digest = hashlib.sha256(self.payload).hexdigest()
        def publish(temporary, destination):
            self.assertEqual(temporary.parent, destination.parent)
            self.assertEqual(temporary.read_bytes(), self.payload)
            self.assertEqual(module.sha256(temporary), digest)
            self.assertEqual(destination.read_bytes(), b'previous checkpoint')
            replace(temporary, destination)
        with patch.object(module.os, 'replace', side_effect=publish) as atomic:
            result = module.retain_reference(self.source, self.destination)
        atomic.assert_called_once()
        self.assertEqual(result, {'name': 'reference.gguf', 'sha256': digest, 'bytes': len(self.payload)})
        self.assertEqual(self.destination.read_bytes(), self.payload)
        self.assertEqual(self.source.read_bytes(), self.payload)
        self.assertEqual(len(list(self.root.iterdir())), 2)

    def test_insufficient_space_does_not_start_copy(self):
        with patch.object(module.shutil, 'disk_usage', return_value=SimpleNamespace(free=len(self.payload) + 63)), \
                patch.object(module.shutil, 'copyfile') as copy:
            with self.assertRaisesRegex(OSError, 'insufficient output space'):
                module.retain_reference(self.source, self.destination, reserve_bytes=64)
        copy.assert_not_called()
        self.assert_unchanged()

    def test_copy_failure_cleans_partial_and_keeps_checkpoint(self):
        def failed_copy(source, temporary):
            temporary.write_bytes(b'incomplete')
            raise OSError('simulated full disk during copy')
        with patch.object(module.shutil, 'copyfile', side_effect=failed_copy):
            with self.assertRaisesRegex(OSError, 'simulated full disk'):
                module.retain_reference(self.source, self.destination)
        self.assert_unchanged()

    def test_equal_size_corruption_fails_hash_and_keeps_checkpoint(self):
        def corrupt_copy(source, temporary):
            temporary.write_bytes(b'x' * len(self.payload))
        with patch.object(module.shutil, 'copyfile', side_effect=corrupt_copy), \
                patch.object(module.os, 'replace') as atomic:
            with self.assertRaisesRegex(ValueError, 'SHA-256 verification'):
                module.retain_reference(self.source, self.destination)
        atomic.assert_not_called()
        self.assert_unchanged()

    def test_hash_read_failure_cleans_partial_and_keeps_checkpoint(self):
        with patch.object(module, 'sha256', side_effect=[hashlib.sha256(self.payload).hexdigest(), OSError('hash read failed')]):
            with self.assertRaisesRegex(OSError, 'hash read failed'):
                module.retain_reference(self.source, self.destination)
        self.assert_unchanged()

    def test_publish_failure_keeps_previous_checkpoint(self):
        with patch.object(module.os, 'replace', side_effect=OSError('publish failed')):
            with self.assertRaisesRegex(OSError, 'publish failed'):
                module.retain_reference(self.source, self.destination)
        self.assert_unchanged()


if __name__ == '__main__':
    unittest.main()
