"""Real recipient/tamper controls; model-free, no CI or external writes."""
import io
from pathlib import Path
import sys
import tarfile
import tempfile
import unittest

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools/validation'))
from private_evidence import open_archive, seal_directory


class EvidenceEnvelope(unittest.TestCase):
    def test_only_recipient_can_read_and_tampering_fails(self):
        key = X25519PrivateKey.generate()
        private = key.private_bytes(serialization.Encoding.Raw, serialization.PrivateFormat.Raw,
                                    serialization.NoEncryption())
        public = key.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); source = root / 'private'; source.mkdir()
            (source / 'audio.wav').write_bytes(b'private validation audio')
            envelope = root / 'sealed.bin'
            seal_directory(source, envelope, public.hex())
            data = envelope.read_bytes()
            self.assertNotIn(b'private validation audio', data)
            with tarfile.open(fileobj=io.BytesIO(open_archive(data, private))) as archive:
                self.assertEqual(archive.getnames(), ['audio.wav'])
                self.assertEqual(archive.extractfile('audio.wav').read(), b'private validation audio')
            other = X25519PrivateKey.generate().private_bytes(serialization.Encoding.Raw,
                serialization.PrivateFormat.Raw, serialization.NoEncryption())
            with self.assertRaises(InvalidTag):
                open_archive(data, other)
            with self.assertRaises(InvalidTag):
                open_archive(data[:-1] + bytes([data[-1] ^ 1]), private)


if __name__ == '__main__':
    unittest.main()
