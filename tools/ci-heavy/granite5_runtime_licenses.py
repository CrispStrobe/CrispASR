#!/usr/bin/env python3
"""Inventory Debian runtime files copied into a validation capsule.

Call snapshot_dependencies before rewriting RUNPATH or copying the dependency
closure, then collect_licenses after packaging. The hosted builder can also call
collect_corresponding_sources to retain exact Debian source packets. This
records evidence, not legal clearance, and makes no source-offer promise.
"""
import hashlib
import json
from email.parser import Parser
from pathlib import Path
import re
import shutil
import subprocess


def _run(args):
    return subprocess.check_output(list(map(str, args)), text=True, stderr=subprocess.STDOUT)


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _elf(path):
    if not path.is_file():
        return False
    with path.open('rb') as stream:
        return stream.read(4) == b'\x7fELF'


def _soname(path):
    return _run(['patchelf', '--print-soname', path]).strip()


def snapshot_dependencies(bundle, build):
    """Snapshot ldd's original paths before closure/RUNPATH mutation.

    The initial ELF files must be the caller's owned build outputs. Unknown
    dependencies may remain unresolved here: the closure/check-bundled-deps
    gates decide which dependencies are intentionally host-provided.
    """
    bundle, build = Path(bundle).resolve(), Path(build).resolve()
    owned = sorted(p.name for p in bundle.iterdir() if _elf(p))
    external = {}
    source_records = {}
    evidence = {}
    for name in owned:
        result = subprocess.run(['ldd', str(bundle / name)], text=True,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
        evidence[name] = dict(returncode=result.returncode, output=result.stdout)
        for line in result.stdout.splitlines():
            match = re.match(r'\s*(\S+)\s+=>\s+(/\S+)\s+\(', line)
            if not match:
                continue
            needed, original = match.groups()
            source = Path(original).resolve(strict=True)
            if source.is_relative_to(build) or source.is_relative_to(bundle):
                continue
            if source not in source_records:
                source_records[source] = dict(source_path=str(source),
                                              original_sha256=_sha(source), original_soname=_soname(source))
            item = dict(source_records[source], ldd_path=original)
            # Both ldd's requested name and resolved basename can be used by
            # the closure script; retain aliases without losing provenance.
            for alias in {needed, Path(original).name, source.name}:
                if alias in external and external[alias]['source_path'] != str(source):
                    raise RuntimeError(f'Ambiguous runtime provenance for {alias}')
                external[alias] = item
    return dict(schema=1, owned_elf_names=owned, dependencies=external, ldd=evidence)


def _owner(item):
    source = Path(item['source_path'])
    candidates = {str(source), item['ldd_path']}
    # dpkg may retain pre-usrmerge paths even when realpath resolves to /usr.
    for candidate in list(candidates):
        if candidate.startswith('/usr/lib/'):
            candidates.add(candidate[4:])
        elif candidate.startswith('/lib/'):
            candidates.add('/usr' + candidate)
    owners = set()
    for candidate in sorted(candidates):
        result = subprocess.run(['dpkg-query', '-S', candidate], text=True,
                                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=False)
        if result.returncode == 0:
            for line in result.stdout.splitlines():
                if ': ' in line:
                    owner, reported = line.rsplit(': ', 1)
                    if reported == candidate:
                        owners.add(owner)
    if len(owners) != 1:
        raise RuntimeError(f'Expected one Debian owner for {source}, found {sorted(owners)}')
    return owners.pop()


def collect_licenses(bundle, snapshot):
    """Collect package notices for every externally copied DSO; fail closed.

    Original and final hashes intentionally differ when patchelf changes ELF
    metadata. Package ownership refers to the unchanged original source file.
    """
    bundle = Path(bundle).resolve()
    destination = bundle / 'licenses' / 'runtime-packages'
    destination.mkdir(parents=True, exist_ok=True)
    manifest = dict(schema=1, libraries={}, packages={},
                    distribution=Path('/etc/os-release').read_text(),
                    corresponding_source_collected=False,
                    corresponding_source_availability_verified=False,
                    limitations=[
                        'Installed package identity and local APT policy are provenance evidence only.',
                        'Corresponding source payloads were not collected or availability verified.',
                        'This inventory provides no legal clearance or new source-offer promise.',
                    ])
    for path in sorted(bundle.iterdir()):
        if not _elf(path) or path.name in snapshot['owned_elf_names']:
            continue
        item = snapshot['dependencies'].get(path.name)
        if item is None:
            raise RuntimeError(f'No pre-closure provenance for bundled ELF {path.name}')
        if _sha(item['source_path']) != item['original_sha256']:
            raise RuntimeError(f'Original dependency changed: {item["source_path"]}')
        package = _owner(item)
        if package not in manifest['packages']:
            fields = _run(['dpkg-query', '-W', '-f',
                           '${binary:Package}\t${Version}\t${Architecture}\t'
                           '${source:Package}\t${source:Version}\n', package]).strip().split('\t')
            if len(fields) != 5 or not all(fields):
                raise RuntimeError(f'Incomplete Debian package identity: {package}')
            binary, version, architecture, source_package, source_version = fields
            package_dir = destination / binary.replace(':', '_')
            package_dir.mkdir(exist_ok=True)
            copyright_path = Path('/usr/share/doc') / binary.split(':')[0] / 'copyright'
            if not copyright_path.is_file():
                raise RuntimeError(f'Missing package copyright/license: {copyright_path}')
            text = copyright_path.read_text()
            if not text.strip():
                raise RuntimeError(f'Empty package copyright/license: {copyright_path}')
            shutil.copy2(copyright_path, package_dir / 'copyright')
            notices = {'copyright': _sha(package_dir / 'copyright')}
            for name in sorted(set(re.findall(r'/usr/share/common-licenses/([A-Za-z0-9.+_-]+)', text))):
                license_path = Path('/usr/share/common-licenses') / name.rstrip('.')
                if not license_path.is_file():
                    raise RuntimeError(f'Missing referenced common license: {license_path}')
                shutil.copy2(license_path, package_dir / license_path.name)
                notices[license_path.name] = _sha(package_dir / license_path.name)
            policy = _run(['apt-cache', 'policy', binary])
            (package_dir / 'apt-policy.txt').write_text(policy)
            notices['apt-policy.txt'] = _sha(package_dir / 'apt-policy.txt')
            manifest['packages'][package] = dict(
                binary_package=binary, binary_version=version, architecture=architecture,
                source_package=source_package, source_version=source_version,
                source_identity_origin='installed Debian dpkg database; see copyright and apt-policy.txt',
                directory=str(package_dir.relative_to(bundle)), files=notices)
        bundled_sha = _sha(path)
        manifest['libraries'][path.name] = dict(
            **item, package=package, bundled_sha256=bundled_sha, bundled_soname=_soname(path),
            bytes_modified_by_packaging=bundled_sha != item['original_sha256'])
    (destination / 'manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
    (destination / 'pre-closure-ldd.json').write_text(json.dumps(snapshot, indent=2, sort_keys=True) + '\n')
    return manifest


def _verify_source_packet(directory, source_package, source_version):
    """Verify an APT-downloaded source packet without extracting its archives."""
    directory = Path(directory)
    descriptors = list(directory.glob('*.dsc'))
    if len(descriptors) != 1:
        raise RuntimeError(f'Expected exactly one source descriptor in {directory}')
    descriptor = descriptors[0]
    if descriptor.is_symlink() or not descriptor.is_file():
        raise RuntimeError(f'Not a regular source descriptor: {descriptor}')
    text = descriptor.read_text()
    if text.startswith('-----BEGIN PGP SIGNED MESSAGE-----'):
        # APT authenticates the repository's source index and downloaded files.
        # Decode the cleartext envelope solely to inspect the Debian fields.
        try:
            text = text.split('\n\n', 1)[1].split('\n-----BEGIN PGP SIGNATURE-----', 1)[0]
        except IndexError as exc:
            raise RuntimeError(f'Malformed signed source descriptor: {descriptor}') from exc
        text = '\n'.join(line[2:] if line.startswith('- ') else line for line in text.splitlines())
    fields = Parser().parsestr(text)
    for key, expected in [('Source', source_package), ('Version', source_version)]:
        if len(fields.get_all(key, [])) != 1 or fields[key].strip() != expected:
            raise RuntimeError(f'Source descriptor {key} does not match {expected}: {descriptor}')
    if len(fields.get_all('Checksums-Sha256', [])) != 1:
        raise RuntimeError(f'Missing or duplicate Checksums-Sha256: {descriptor}')
    members = {}
    for line in fields['Checksums-Sha256'].splitlines():
        if not line.strip():
            continue
        match = re.fullmatch(r'\s*([0-9a-fA-F]{64})\s+([0-9]+)\s+(\S+)\s*', line)
        if not match:
            raise RuntimeError(f'Malformed source checksum entry: {line!r}')
        digest, size, name = match.groups()
        if name in {'.', '..', descriptor.name} or Path(name).name != name or name in members:
            raise RuntimeError(f'Unsafe or duplicate source member: {name}')
        members[name] = dict(sha256=digest.lower(), bytes=int(size))
    if not members:
        raise RuntimeError(f'Empty source packet: {descriptor}')
    expected_names = set(members) | {descriptor.name}
    actual_names = {path.name for path in directory.iterdir()}
    if actual_names != expected_names:
        raise RuntimeError(f'Source packet members differ: missing={expected_names - actual_names}, '
                           f'extra={actual_names - expected_names}')
    for name, expected in members.items():
        path = directory / name
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(f'Not a regular source member: {path}')
        if path.stat().st_size != expected['bytes'] or _sha(path) != expected['sha256']:
            raise RuntimeError(f'Source member checksum/size mismatch: {path}')
    members[descriptor.name] = dict(sha256=_sha(descriptor), bytes=descriptor.stat().st_size)
    return members


def collect_corresponding_sources(bundle, manifest):
    """Download and ship the exact source packets of inventoried Debian DSOs.

    Intended for the hosted builder with authenticated deb-src indexes enabled.
    This does not extract/build sources, create a source offer, or establish
    legal clearance. No success flags change until every packet verifies.
    """
    bundle = Path(bundle).resolve()
    destination = bundle / 'licenses' / 'runtime-sources'
    destination.mkdir(parents=True, exist_ok=True)
    identities = sorted({(p['source_package'], p['source_version'])
                         for p in manifest['packages'].values()})
    packets = []
    for package, version in identities:
        if not re.fullmatch(r'[a-z0-9][a-z0-9+.-]*', package) or not re.fullmatch(r'[0-9A-Za-z.+:~_-]+', version):
            raise RuntimeError(f'Invalid source package identity: {package}={version}')
        identity = f'{package}={version}'
        safe_name = re.sub(r'[^0-9A-Za-z.+_-]', '_', identity)
        safe_name += '-' + hashlib.sha256(identity.encode()).hexdigest()[:12]
        directory = destination / safe_name
        directory.mkdir()  # Refuse to mix payloads from earlier attempts.
        log = destination / (safe_name + '.apt.log')
        command = ['apt-get', '-o', 'Acquire::AllowInsecureRepositories=false',
                   '-o', 'APT::Get::AllowUnauthenticated=false',
                   'source', '--download-only', '--only-source', '-y', identity]
        with log.open('w') as stream:
            subprocess.run(command, cwd=directory, stdout=stream, stderr=subprocess.STDOUT,
                           check=True, timeout=1800)
        members = _verify_source_packet(directory, package, version)
        packets.append(dict(source_package=package, source_version=version,
                            directory=str(directory.relative_to(bundle)), files=members,
                            apt_command=command, apt_log=str(log.relative_to(bundle)),
                            apt_log_sha256=_sha(log),
                            authentication='APT authenticated source index and source download; '
                                           'descriptor identity and SHA256 member hashes verified'))
    manifest['source_packets'] = packets
    manifest['corresponding_source_collected'] = True
    manifest['corresponding_source_availability_verified'] = True
    manifest['limitations'] = [
        'Exact Debian source packets accompany the inventoried external runtime binaries.',
        'Source identity, APT download and source packet hashes are verified; sources were not rebuilt.',
        'This inventory provides no legal clearance or new source-offer promise.',
    ]
    manifest_path = bundle / 'licenses' / 'runtime-packages' / 'manifest.json'
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
    return manifest
