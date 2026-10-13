"""Reuse the actual pinned Torch 2.6 rotary buffer without regenerating it."""
import json
from pathlib import Path
import tempfile
import urllib.request

import numpy as np

from dia_proof_archives import digest, extract_verified
from reference_backends import dia_tts_reference as reference

REVISION = '13cfd81146f17b746a47ce154013603aeabef658'
URL = ('https://huggingface.co/datasets/cstr/crispasr-regression-fixtures/resolve/'
       + REVISION + '/dia-20261013-cache/rotary-coefficient-probe.zip')
SPEC = dict(bytes=15308, sha256='e063c7158c8a98cb6ccbd17b8cb22135f3493bfbfe49d1a4f8e9c6e7f44064ea',
            source_files=7, archive_member_prefix='rotary-probe/', proof_folder='rotary-probe')
BUFFER_SHA = '0f38c9ea2be251a9e979929d91834881afa4ce2a70209326e4a419b1630c0a2b'
LAYERS_SHA = '9b77d590101783a7e570bd9889afb09b85912d05072313c32a984dbfe9323024'


def fetch(destination):
    destination = Path(destination)
    if destination.exists() or destination.is_symlink():
        raise ValueError('Original timescale proof requires a new destination')
    destination.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.build_opener().open(URL, timeout=60) as response:
        data = response.read(SPEC['bytes'] + 1)
    if len(data) != SPEC['bytes']:
        raise ValueError('Original timescale proof size differs')
    destination.write_bytes(data)
    if digest(destination) != SPEC['sha256']:
        raise ValueError('Original timescale proof SHA differs')
    return destination


def original_buffer(archive, source, head_dim):
    """Validate source, immutable ZIP, every payload and original provenance.

    Return exactly the CPU constructor's metadata: the resulting GGUF must be
    byte-identical to its Torch 2.6 export regardless of the reader's Torch.
    This does not invoke an original neural model or a new constructor.
    """
    if type(head_dim) is not int or head_dim != 128:
        raise ValueError('Pinned original buffer requires head dimension 128')
    _, hashes = reference.verify_source(source)
    if hashes['dia/layers.py'] != LAYERS_SHA:
        raise ValueError('Original rotary source differs')
    archive = Path(archive)
    with tempfile.TemporaryDirectory(prefix='dia-timescale-', dir=archive.parent) as temporary:
        folder = extract_verified(archive, Path(temporary) / 'proof', SPEC)
        evidence = json.loads((folder.parent / 'tracked-evidence.json').read_text())
        if (evidence['head_dim'] != head_dim or evidence['torch_version'] != '2.6.0+cpu'
                or evidence['original_source_sha256'] != LAYERS_SHA):
            raise ValueError('Original rotary provenance differs')
        path = folder / 'original-timescales.f32'
        if path.stat().st_size != 256 or digest(path) != BUFFER_SHA:
            raise ValueError('Actual original timescale payload differs')
        value = np.fromfile(path, dtype='<f4')
    if value.shape != (64,) or not np.isfinite(value).all() or not (value > 0).all():
        raise ValueError('Invalid original timescale values')
    metadata = dict(source_pin=reference.SOURCE_PIN, source_sha256=LAYERS_SHA,
                    torch_version='2.6.0+cpu', head_dim=head_dim)
    return value, metadata


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('destination', type=Path)
    args = parser.parse_args()
    path = fetch(args.destination)
    print(json.dumps(dict(url=URL, revision=REVISION, bytes=path.stat().st_size,
                          sha256=digest(path), buffer_sha256=BUFFER_SHA)))
