"""Checkpoint a diagnostic reference before a later comparison can fail."""
import hashlib
import os
from pathlib import Path
import shutil
import tempfile


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def retain_reference(source, destination, reserve_bytes=64 * 1024 * 1024):
    """Copy, verify, then atomically publish; leave existing output intact on error.

    The destination filesystem must have room for the copy plus a small reserve.
    The original remains available if copying, verification or publication fails.
    """
    source, destination = Path(source), Path(destination)
    size = source.stat().st_size
    if shutil.disk_usage(destination.parent).free < size + reserve_bytes:
        raise OSError('insufficient output space for reference archive and reserve')
    expected = sha256(source)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=destination.parent,
                                         prefix=f'.{destination.name}.', suffix='.partial',
                                         delete=False) as stream:
            temporary = Path(stream.name)
        shutil.copyfile(source, temporary)
        if temporary.stat().st_size != size or sha256(temporary) != expected:
            raise ValueError('reference archive copy failed SHA-256 verification')
        os.replace(temporary, destination)
        return {'name': destination.name, 'sha256': expected, 'bytes': size}
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
