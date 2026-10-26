#!/usr/bin/env python3
"""Recover bounded evidence from the first replay without copying its model links.

Run on Actions only. The first replay placed opaque model symlinks in HEAVY_OUT;
this collector retains every non-model member and records exclusions explicitly.
It neither runs inference nor changes acceptance or deletes producer artifacts.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import zipfile


def api(path):
    return json.loads(subprocess.check_output(['gh', 'api', path], text=True))


def sha(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''):
            value.update(block)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=int, default=40377479500)
    args = parser.parse_args()
    assert os.environ.get('GITHUB_ACTIONS') == 'true', 'Large artifact recovery belongs on Actions'
    out = Path(os.environ['HEAVY_OUT'])
    scratch = Path(os.environ['HEAVY_SCRATCH'])
    out.mkdir(parents=True, exist_ok=True)
    available = next(int(line.split()[1]) for line in Path('/proc/meminfo').read_text().splitlines()
                     if line.startswith('MemAvailable:'))
    assert available >= 1024**2 and os.getloadavg()[0] < 2 * (os.cpu_count() or 1)
    assert shutil.disk_usage(scratch).free >= 8 * 1024**3
    base = 'repos/CrispStrobe/CrispASR/actions'
    run = api(f'{base}/runs/{args.run}')
    assert run['status'] == 'completed'
    assert run['head_sha'] == '6143b9aa2ff0457092a2f35b0dcb2a4aa42a5ff8'
    artifacts = api(f'{base}/runs/{args.run}/artifacts')['artifacts']
    selected = [a for a in artifacts if a['name'] == f'heavy-{args.run}' and not a['expired']]
    assert len(selected) == 1
    artifact = selected[0]
    assert artifact['size_in_bytes'] <= 3 * 1024**3
    archive_path = scratch / 'original-replay.zip'
    with archive_path.open('wb') as stream:
        subprocess.run(['gh', 'api', f'{base}/artifacts/{artifact["id"]}/zip'], stdout=stream, check=True)
    receipt = dict(producer_run=args.run, producer_source=run['head_sha'],
                   producer_conclusion=run['conclusion'], artifact_id=artifact['id'],
                   archive_bytes=archive_path.stat().st_size, archive_sha256=sha(archive_path),
                   retained={}, excluded={}, inference_executed=False)
    with zipfile.ZipFile(archive_path) as archive:
        members = archive.infolist()
        assert len({m.filename for m in members}) == len(members), 'Duplicate evidence names'
        assert all(not Path(m.filename).is_absolute() and '..' not in Path(m.filename).parts
                   and '\\' not in m.filename for m in members)
        ordinary = [m for m in members if not m.is_dir() and not m.filename.lower().endswith('.gguf')]
        assert sum(m.file_size for m in ordinary) <= 256 * 1024**2, 'Unexpectedly large non-model evidence'
        for member in members:
            if member.is_dir():
                continue
            if member.filename.lower().endswith('.gguf'):
                assert Path(member.filename).name == 'opaque.gguf', 'Unexpected model payload'
                receipt['excluded'][member.filename] = dict(bytes=member.file_size, crc32=member.CRC,
                    reason='temporary auto-detection model link; immutable model identity is in replay.json')
                continue
            dest = out / 'cpu-evidence' / member.filename
            dest.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member) as source, dest.open('wb') as target:
                shutil.copyfileobj(source, target)
            receipt['retained'][member.filename] = dict(bytes=dest.stat().st_size, sha256=sha(dest))
    assert 'replay.json' in receipt['retained']
    # The numerical receipt is copied byte-for-byte; no generated acceptance.
    (out / 'collection.json').write_text(json.dumps(receipt, indent=2) + '\n')
    (out / 'summary.md').write_text('Original replay evidence retained byte-for-byte; opaque model links excluded.\n'
                                  'No inference, threshold change or producer artifact deletion.\n')
    print('Retained', len(receipt['retained']), 'evidence files; excluded', len(receipt['excluded']), 'model links')


if __name__ == '__main__':
    main()
