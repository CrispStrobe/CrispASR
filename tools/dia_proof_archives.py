"""Retrieve immutable public Dia proofs with whole-archive and payload checks."""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import stat
import tempfile
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
INDEX = ROOT / 'docs/historical/dia-public-proof-archives-2026-10-13.json'


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def safe_name(name):
    path = PurePosixPath(name)
    if not name or '\\' in name or path.is_absolute() or '..' in path.parts:
        raise ValueError('Unsafe archive path')
    return path


def extract_verified(archive, destination, spec):
    """Accept only the pinned ZIP and complete manifest; never replace a path."""
    archive, destination = Path(archive), Path(destination)
    if destination.exists() or destination.is_symlink():
        raise ValueError('Proof destination already exists')
    if archive.stat().st_size != spec['bytes'] or digest(archive) != spec['sha256']:
        raise ValueError('Archive size/SHA differs from immutable pin')
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='dia-proof-', dir=destination.parent) as scratch:
        root = Path(scratch) / 'contents'
        root.mkdir()
        with zipfile.ZipFile(archive) as zipped:
            entries = zipped.infolist()
            names = [entry.filename for entry in entries]
            if len(names) != len(set(names)):
                raise ValueError('Duplicate archive members')
            for entry in entries:
                safe_name(entry.filename)
                mode = stat.S_IFMT(entry.external_attr >> 16)
                if mode not in (0, stat.S_IFREG, stat.S_IFDIR):
                    raise ValueError('Nonregular archive member')
            manifest = json.loads(zipped.read('ARCHIVE-MANIFEST.json'))
            rows = manifest['files']
            expected = {spec['archive_member_prefix'] + row['path'] for row in rows}
            for name in expected:
                safe_name(name)
            if len(rows) != spec['source_files'] or len(expected) != len(rows):
                raise ValueError('Manifest file count differs')
            if set(names) != expected | {'ARCHIVE-MANIFEST.json', 'tracked-evidence.json'}:
                raise ValueError('Archive coverage differs from manifest')
            zipped.extractall(root)
        for row in rows:
            path = root / (spec['archive_member_prefix'] + row['path'])
            if path.stat().st_size != row['bytes'] or digest(path) != row['sha256']:
                raise ValueError('Extracted payload size/SHA differs')
        folder = root / spec['proof_folder']
        if not folder.is_dir() or destination.exists() or destination.is_symlink():
            raise ValueError('Proof folder missing or destination changed')
        root.rename(destination)
    return destination / spec['proof_folder']


def fetch(name, destination):
    index = json.loads(INDEX.read_text())
    matches = [row for row in index['archives'] if row['name'] == name]
    if len(matches) != 1:
        raise ValueError('Unknown pinned Dia proof')
    spec = matches[0]
    destination = Path(destination)
    if destination.exists() or destination.is_symlink():
        raise ValueError('Proof destination already exists')
    destination.parent.mkdir(parents=True, exist_ok=True)
    pin = index['publication']
    url = (f"https://huggingface.co/datasets/{pin['repo']}/resolve/{pin['revision']}/"
           f"{pin['folder']}/{spec['path']}")
    with tempfile.TemporaryDirectory(prefix='dia-download-', dir=destination.parent) as scratch:
        archive = Path(scratch) / 'proof.zip'
        # Fresh standard-library opener: no authenticated HF SDK/session/token.
        request = urllib.request.Request(url, headers={'User-Agent': 'CrispASR-public-proof'})
        with urllib.request.build_opener().open(request, timeout=60) as response, archive.open('wb') as output:
            size = 0
            for block in iter(lambda: response.read(4 * 1024 * 1024), b''):
                size += len(block)
                if size > spec['bytes']:
                    raise ValueError('Download exceeds pinned size')
                output.write(block)
                output.flush()
                os.fsync(output.fileno())
        return extract_verified(archive, destination, spec)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('name', choices=('canonical-original', 'portable-rng', 'native-speech-rejection', 'dac-converter'))
    parser.add_argument('destination', type=Path)
    args = parser.parse_args()
    print(fetch(args.name, args.destination))
