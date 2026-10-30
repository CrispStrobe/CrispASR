#!/usr/bin/env python3
"""Preserve the original issue 483 pair as exact archives in a draft release.

Actions-only archive work: no compiler, SDK installation, executable loading or
inference. Public promotion is deliberately absent. Both original archives are
wrapped with additive vendor terms, original notices and provenance.
"""
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import time
import zipfile

ROOT = Path(__file__).resolve().parents[2]
REPOSITORY = 'CrispStrobe/CrispASR'
SOURCE = '04ded753e909da019a2cd391f651f51f5299b6f5'
PRODUCER = 37733919241
TAG = 'archive-issue483-mmq-20261008'
MARKER = f'Original issue #483 pair, producer {PRODUCER}, source {SOURCE}.'
ARTIFACTS = {
    'OFF': (11532751916, 520620096, 'ddaf82ddf80992c9d5b8f86263cc23317e5b92f6377dcf9c365926fbb00fe244'),
    'ON': (11532004516, 520659350, '84a6b98729aba3412642959f09e60f41b029f66fe545375293b252b39768bfa1'),
}


def command(argv, **kwargs):
    return subprocess.run(list(map(str, argv)), check=True, **kwargs)


def api(path):
    return json.loads(subprocess.check_output(['gh', 'api', f'repos/{REPOSITORY}/' + path], text=True))


def sha_stream(stream):
    value = hashlib.sha256()
    for chunk in iter(lambda: stream.read(1024**2), b''):
        value.update(chunk)
    return value.hexdigest()


def sha(path):
    with path.open('rb') as stream:
        return sha_stream(stream)


def safe_name(name):
    # PowerShell ZIPs can contain Windows separators; normalize before checking
    # or using any path, without changing the immutable original archive.
    path = PurePosixPath(name.replace('\\', '/'))
    if path.is_absolute() or '..' in path.parts or ':' in str(path):
        raise ValueError('Unsafe archive member path')
    return str(path)


def require_draft(release):
    if not (release['draft'] and release['prerelease'] and
            release['target_commitish'] == SOURCE and MARKER in (release.get('body') or '')):
        raise RuntimeError('Only the matching maintainer preservation draft may be used')


def members(archive, limit):
    entries = archive.infolist()
    assert len(entries) < 1000 and len({safe_name(m.filename).lower() for m in entries}) == len(entries)
    assert sum(m.file_size for m in entries) <= limit
    return [m for m in entries if not m.is_dir()]


def resource_snapshot(scratch):
    import psutil

    return dict(load=os.getloadavg(), available_kib=psutil.virtual_memory().available // 1024,
                disk_free=shutil.disk_usage(scratch).free, cpus=os.cpu_count() or 1)


def wait_for_resources(scratch, receipt, save, *, probe=resource_snapshot, pause=time.sleep, attempts=31):
    """Allow runner startup load to settle; never lower the transfer gates."""
    receipt['resource_samples'] = []
    receipt['resource_wait_interval_s'] = 10
    for attempt in range(attempts):
        row = dict(probe(scratch), wait_attempt=attempt)
        receipt['resources'] = row
        receipt['resource_samples'].append(row)
        save()
        if row['disk_free'] < 8 * 1024**3:
            raise RuntimeError('Need 8 GiB free for archive scratch')
        if row['available_kib'] >= 2 * 1024**2 and row['load'][0] < 2 * row['cpus']:
            return
        if attempt + 1 < attempts:
            print('Waiting for archive headroom:', row, flush=True)
            pause(10)
    raise RuntimeError('Archive load/memory headroom did not recover within the bounded wait')


def preserve(mode, out, scratch, pin):
    artifact_id, expected_bytes, expected_sha = ARTIFACTS[mode]
    info = api(f'actions/artifacts/{artifact_id}')
    assert info['name'] == f'cuda126-ptx-mmq-{mode}' and not info['expired']
    assert info['size_in_bytes'] == expected_bytes and info['digest'] == 'sha256:' + expected_sha
    assert info['workflow_run']['id'] == PRODUCER and info['workflow_run']['head_sha'] == SOURCE
    directory = scratch / mode
    directory.mkdir()
    original = directory / f'original-artifact-{artifact_id}.zip'
    with original.open('wb') as stream:
        command(['gh', 'api', f'repos/{REPOSITORY}/actions/artifacts/{artifact_id}/zip'], stdout=stream)
    assert original.stat().st_size == expected_bytes and sha(original) == expected_sha
    with zipfile.ZipFile(original) as archive:
        entries = members(archive, 768 * 1024**2)
        packages = [m for m in entries if m.filename.endswith('.zip')]
        manifests = [m for m in entries if PurePosixPath(safe_name(m.filename)).name == 'manifest.json']
        assert len(packages) == len(manifests) == 1
        manifest_raw = archive.read(manifests[0])
        manifest = json.loads(manifest_raw.decode('utf-8-sig'))
        assert manifest == pin['manifests'][mode], 'Original uploaded manifest changed'
        package = directory / PurePosixPath(safe_name(packages[0].filename)).name
        with archive.open(packages[0]) as source, package.open('wb') as target:
            shutil.copyfileobj(source, target)
        original_inventory = {m.filename: dict(bytes=m.file_size, crc32=m.CRC) for m in entries}
    notices = out / mode / 'original-notices'
    notices.mkdir(parents=True)
    inventory = {}
    with zipfile.ZipFile(package) as archive:
        entries = members(archive, 3 * 1024**3)
        for member in entries:
            with archive.open(member) as stream:
                inventory[member.filename] = dict(bytes=member.file_size, sha256=sha_stream(stream))
            name = PurePosixPath(safe_name(member.filename)).name
            if any(term in name.upper() for term in ('LICENSE', 'NOTICE', 'COPYRIGHT', 'EULA')):
                assert member.file_size <= 1024**2
                target = notices / safe_name(member.filename)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(archive.read(member))
        assert any(PurePosixPath(safe_name(m.filename)).name == 'crispasr.exe' for m in entries)
        notice_names = {p.name for p in notices.rglob('*') if p.is_file()}
        assert {'LICENSE', 'THIRD_PARTY_NOTICES.txt'}.issubset(notice_names)
        inside = [m for m in entries if PurePosixPath(safe_name(m.filename)).name == 'manifest.json']
        assert len(inside) == 1 and json.loads(archive.read(inside[0]).decode('utf-8-sig')) == manifest
        runtime = {PurePosixPath(safe_name(name)).name: value['sha256'] for name, value in inventory.items()
                   if PurePosixPath(safe_name(name)).name in manifest['runtime_sha256']}
        assert runtime == manifest['runtime_sha256']
        # No driver or SDK installer is part of the known experiment payload.
        assert not any(PurePosixPath(safe_name(name)).name.lower() in {'nvcuda.dll', 'nvcc.exe', 'setup.exe'} for name in inventory)
    vendor = ROOT / 'tools/ci-heavy/mmq-preservation'
    vendor_origin = json.loads((vendor / 'nvidia-origin.json').read_text())
    assert sha(vendor / 'NVIDIA-CUDA-12.6.3-LICENSE.txt') == vendor_origin['text_sha256']
    notice = directory / 'ADDITIONAL-CUDA-NOTICE.txt'
    notice.write_text('Original experiment archives are preserved byte-for-byte.\n'
        'The NVIDIA cudart64_12.dll, cublas64_12.dll and cublasLt64_12.dll remain\n'
        'subject to NVIDIA CUDA 12.6.3 terms and applicable third-party notices;\n'
        'the application MIT license does not relicense those DLLs.\n'
        'The complete vendor agreement/supplement/attachments accompany this\n'
        'archive to retain the cuBLAS third-party notices omitted from the\n'
        'historical project notice. Retain these additive notices with extracted\n'
        'binaries. This does not grant standalone SDK redistribution rights.\n'
        'Intended CUDA execution requires NVIDIA hardware.\n')
    row = dict(mode=mode, producer_artifact_id=artifact_id, producer_expires_at=info['expires_at'],
        original_bytes=expected_bytes, original_sha256=expected_sha,
        original_members=original_inventory, package_sha256=sha(package), package_members=inventory,
        manifest=manifest, vendor_origin=vendor_origin, rebuilt=False, gpu_executed=False)
    (out / mode / 'inventory.json').write_text(json.dumps(row, indent=2) + '\n')
    wrapper = scratch / f'cuda126-ptx-mmq-{mode}-preserved.zip'
    with zipfile.ZipFile(wrapper, 'w', compression=zipfile.ZIP_STORED) as archive:
        archive.write(original, original.name)
        archive.write(vendor / 'NVIDIA-CUDA-12.6.3-LICENSE.txt', 'notices/NVIDIA-CUDA-12.6.3-LICENSE.txt')
        archive.write(vendor / 'nvidia-origin.json', 'notices/nvidia-origin.json')
        archive.write(notice, 'notices/ADDITIONAL-CUDA-NOTICE.txt')
        for path in sorted(p for p in notices.rglob('*') if p.is_file()):
            archive.write(path, 'notices/original/' + str(path.relative_to(notices)))
        archive.write(out / mode / 'inventory.json', 'inventory.json')
    row.update(preservation_file=wrapper.name, preservation_bytes=wrapper.stat().st_size,
               preservation_sha256=sha(wrapper))
    return row, wrapper


def main():
    assert os.environ.get('GITHUB_ACTIONS') == 'true' and os.environ.get('GITHUB_REPOSITORY') == REPOSITORY
    out, scratch = Path(os.environ['PRESERVE_OUT']), Path(os.environ['PRESERVE_SCRATCH'])
    out.mkdir(parents=True, exist_ok=True)
    scratch.mkdir(parents=True, exist_ok=True)
    assert not scratch.resolve().is_relative_to(out.resolve())
    receipt = dict(complete=False, published=False, draft=True, gpu_executed=False, rebuilt=False,
                   producer_run=PRODUCER, original_source=SOURCE, modes={})

    def save():
        (out / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')

    save()
    try:
        wait_for_resources(scratch, receipt, save)
        producer = api(f'actions/runs/{PRODUCER}')
        assert producer['conclusion'] == 'success' and producer['head_sha'] == SOURCE
        pin = json.loads((ROOT / 'docs/historical/cuda126-mmq-experiment-2026-10-08.json').read_text())
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            jobs = {mode: pool.submit(preserve, mode, out, scratch, pin) for mode in ARTIFACTS}
            results = {mode: job.result() for mode, job in jobs.items()}
        for mode, (row, _) in results.items():
            receipt['modes'][mode] = row
        off, on = (receipt['modes'][mode]['manifest'] for mode in ('OFF', 'ON'))
        for key in ('commit', 'ggml_commit', 'toolkit', 'runtime_version', 'architectures', 'cpu_floor', 'runtime_sha256'):
            assert off[key] == on[key]
        assert off['force_mmq'] == 'OFF' and on['force_mmq'] == 'ON'
        save()
        body = scratch / 'release-notes.md'
        body.write_text(MARKER + '\n\n'
            'Maintainer preservation draft: exact original artifacts wrapped with\n'
            'additive CUDA terms/notices and per-member inventories. No rebuild,\n'
            'public promotion, production-default change or GPU performance claim.\n'
            'The source is archived at archive/issue483-mmq-source-20261008.\n'
            'Compare OFF and ON on the same GTX1660/MX150, with identical explicit\n'
            'model/audio/settings. Retain startup logs/transcripts and distinguish\n'
            'cold PTX-JIT startup from repeated warm timings. Actual device results\n'
            'remain required. Keep original and additive vendor notices together.\n')
        releases = api('releases?per_page=100')
        existing = [r for r in releases if r['tag_name'] == TAG]
        if existing:
            assert len(existing) == 1
            require_draft(existing[0])
            release = existing[0]
        else:
            payload = scratch / 'draft-release.json'
            payload.write_text(json.dumps(dict(tag_name=TAG, target_commitish=SOURCE,
                name='Archive: original CUDA 12.6 MMQ OFF/ON experiment', body=body.read_text(),
                draft=True, prerelease=True, make_latest='false')) + '\n')
            release = json.loads(subprocess.check_output(['gh', 'api', '--method', 'POST',
                f'repos/{REPOSITORY}/releases', '--input', str(payload)], text=True))
        require_draft(release)
        receipt['draft_release_id'], receipt['draft_release_tag'] = release['id'], TAG
        for mode, (row, wrapper) in results.items():
            assets = api(f'releases/{release["id"]}/assets?per_page=100')
            found = [a for a in assets if a['name'] == wrapper.name]
            if found:
                assert len(found) == 1 and found[0]['size'] == row['preservation_bytes']
                if found[0].get('digest'):
                    assert found[0]['digest'] == 'sha256:' + row['preservation_sha256']
            else:
                command(['gh', 'release', 'upload', TAG, wrapper, '--repo', REPOSITORY])
            assets = api(f'releases/{release["id"]}/assets?per_page=100')
            asset = next(a for a in assets if a['name'] == wrapper.name)
            assert asset['size'] == row['preservation_bytes']
            if asset.get('digest'):
                assert asset['digest'] == 'sha256:' + row['preservation_sha256']
            downloaded = scratch / (mode + '-readback.zip')
            with downloaded.open('wb') as stream:
                command(['gh', 'api', '-H', 'Accept: application/octet-stream',
                         f'repos/{REPOSITORY}/releases/assets/{asset["id"]}'], stdout=stream)
            assert sha(downloaded) == row['preservation_sha256']
            row.update(draft_asset_id=asset['id'], full_draft_asset_readback=True)
            save()
        final = api(f'releases/{release["id"]}')
        require_draft(final)
        assert all(p.suffix.lower() not in {'.zip', '.dll', '.exe'} for p in out.rglob('*') if p.is_file())
        receipt['complete'] = True
        (out / 'summary.md').write_text('Original OFF/ON archives preserved and fully read back in a maintainer draft.\n'
            'All source/configuration/runtime identities match; vendor notice supplement included.\n'
            'No rebuild, public promotion or GTX1660/MX150 performance acceptance.\n')
        print('MMQ_ORIGINAL_PAIR_PRESERVED_IN_DRAFT', release['id'], flush=True)
    finally:
        save()


if __name__ == '__main__':
    main()
