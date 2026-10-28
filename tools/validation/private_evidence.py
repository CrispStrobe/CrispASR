"""Seal validation audio for the maintainer; no private key enters CI."""
import hashlib
import io
import os
from pathlib import Path
import tarfile

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey, X25519PublicKey
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

MAGIC = b'CRISPASR-EVIDENCE-1\n'
INFO = b'CrispASR private validation audio v1'


def key_material(shared):
    return HKDF(algorithm=hashes.SHA256(), length=32, salt=None, info=INFO).derive(shared)


def seal_directory(directory, destination, recipient_hex):
    recipient = bytes.fromhex(recipient_hex)
    public = X25519PublicKey.from_public_bytes(recipient)
    files = sorted(Path(directory).rglob('*'))
    if any(p.is_symlink() for p in files):
        raise ValueError('Private evidence cannot contain symlinks')
    files = [p for p in files if p.is_file()]
    if len(files) > 128 or sum(p.stat().st_size for p in files) > 256 * 1024**2:
        raise ValueError('Private evidence exceeds bounded validation envelope')
    archive = io.BytesIO()
    with tarfile.open(fileobj=archive, mode='w:gz') as output:
        for path in files:
            output.add(path, arcname=str(path.relative_to(directory)), recursive=False)
    ephemeral = X25519PrivateKey.generate()
    nonce = os.urandom(12)
    header = MAGIC + ephemeral.public_key().public_bytes(
        serialization.Encoding.Raw, serialization.PublicFormat.Raw) + nonce
    payload = header + AESGCM(key_material(ephemeral.exchange(public))).encrypt(nonce, archive.getvalue(), header)
    Path(destination).write_bytes(payload)
    return dict(bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest(),
                recipient_sha256=hashlib.sha256(recipient).hexdigest(), files=len(files))


def open_archive(payload, private_key):
    if not payload.startswith(MAGIC) or len(payload) < len(MAGIC) + 60:
        raise ValueError('Invalid private evidence envelope')
    offset = len(MAGIC)
    public = X25519PublicKey.from_public_bytes(payload[offset:offset+32])
    nonce = payload[offset+32:offset+44]
    header = payload[:offset+44]
    key = X25519PrivateKey.from_private_bytes(private_key)
    return AESGCM(key_material(key.exchange(public))).decrypt(nonce, payload[len(header):], header)
