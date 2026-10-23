#!/usr/bin/env python3
"""Complete CUDA/CUB notices without changing a single compiled ELF payload.

Streaming repack only; no extraction, compiler, model loading or inference.
"""
import argparse
import copy
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import subprocess
import tarfile

ROOT = Path(__file__).resolve().parents[2]
CUB_SHA = '2e15eba2e130f3e80d9e91724c7dce3781ce99316324681ff54b00c1fc9759b3'
CUB_URL = 'https://raw.githubusercontent.com/NVIDIA/cccl/v2.3.2/cub/LICENSE.TXT'
SDK_URL = 'https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-cccl-12-4_12.4.127-1_amd64.deb'
SDK_SHA = '33a176b9f1b4c779d9b265e6049702490a31b4374a581bac5160ce9675af2485'


def digest(data):
    return dict(sha256=hashlib.sha256(data).hexdigest(), bytes=len(data))


def file_digest(path):
    h, count = hashlib.sha256(), 0
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
            count += len(block)
    return dict(sha256=h.hexdigest(), bytes=count)


class HashReader:
    def __init__(self, source):
        self.source, self.hash, self.count, self.magic = source, hashlib.sha256(), 0, b''
    def read(self, size):
        block = self.source.read(size)
        if self.count < 4:
            self.magic += block[:4 - self.count]
        self.hash.update(block)
        self.count += len(block)
        return block
    def record(self):
        return dict(sha256=self.hash.hexdigest(), bytes=self.count)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--cub-license', type=Path, required=True)
    args = parser.parse_args()
    original = json.loads((args.input_dir / 'validation.json').read_text())
    if original.get('cuda') != '12.4.1' or original.get('license_amendment'):
        raise ValueError('Only original pinned CUDA12.4.1 capsules may be amended')
    source = args.input_dir / original['archive']['name']
    if file_digest(source) != {k: original['archive'][k] for k in ('sha256', 'bytes')}:
        raise ValueError('Original qualified archive hash/size mismatch')
    cub = args.cub_license.read_bytes()
    if digest(cub)['sha256'] != CUB_SHA:
        raise ValueError('CUB license differs from the verified NVIDIA2.3.2 source')
    notices = (ROOT / 'THIRD_PARTY_NOTICES.txt').read_bytes()
    if cub.strip() not in notices or b'NVIDIA CUB 2.3.2' not in notices:
        raise ValueError('Complete CUB notice is absent from project notices')
    packaging_source = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    changes = {'THIRD_PARTY_NOTICES.txt': notices, 'licenses/CUB-2.3.2.txt': cub,
               'licenses/CUDA-compiler-components.txt': (
                   'CUDA12.4.1 compiler/device redistributables retain their NVIDIA SDK terms.\n'
                   'https://docs.nvidia.com/cuda/archive/12.4.1/eula/index.html\n'
                   'No NVIDIA shared-library files or SDK headers are included.\n'
                   'Compiled CUB2.3.2 implementations retain BSD-3-Clause; see CUB-2.3.2.txt.\n'
                   'The exact official cuda-cccl package/version was inspected, rather than\n'
                   'relying on the release-note table which names CUB2.3.1.\n').encode()}
    args.out.mkdir(parents=True, exist_ok=True)
    destination = args.out / 'granite5-cuda-validation.tar.gz'
    if destination.exists():
        raise ValueError('Refusing to mix an existing amended capsule')
    temporary = destination.with_suffix('.partial')
    hashes, links, elfs = {}, {}, {}
    provenance = None
    with tarfile.open(source, 'r|gz') as incoming, tarfile.open(
            temporary, 'w|gz', compresslevel=6, copybufsize=1024 * 1024) as outgoing:
        for member in incoming:
            path = PurePosixPath(member.name)
            if path.is_absolute() or '..' in path.parts or path.parts[0] != 'bundle':
                raise ValueError('Unsafe capsule member')
            name = str(path.relative_to('bundle'))
            if path.name.startswith(('libcuda', 'libcublas', 'libnv')) and '.so' in path.name:
                raise ValueError('This amendment does not authorize NVIDIA shared-library redistribution')
            if member.isdir():
                outgoing.addfile(member)
                continue
            if member.issym():
                target = PurePosixPath(member.linkname)
                if target.is_absolute() or '..' in target.parts:
                    raise ValueError('Unsafe capsule alias')
                links[name] = str(PurePosixPath(name).parent / target)
                outgoing.addfile(member)
                continue
            if not member.isfile():
                raise ValueError('Unexpected capsule member type')
            stream = incoming.extractfile(member)
            if name in {'provenance.json', 'THIRD_PARTY_NOTICES.txt', 'README.txt'}:
                data = stream.read()
                hashes[name] = digest(data)
                if name == 'provenance.json':
                    provenance = json.loads(data)
                elif name == 'THIRD_PARTY_NOTICES.txt':
                    if not notices.startswith(data.rstrip()):
                        raise ValueError('Existing attribution would be removed')
                else:
                    changes[name] = data.replace(b'NVIDIA libraries', b'NVIDIA shared-library files') + (
                        b'\nCUB license completion: compiled ELF payloads are unchanged. Preserve\n'
                        b'THIRD_PARTY_NOTICES.txt and licenses/CUB-2.3.2.txt with this package.\n')
                continue
            reader = HashReader(stream)
            outgoing.addfile(member, reader)
            hashes[name] = reader.record()
            if reader.magic == b'\x7fELF':
                elfs[name] = reader.record()
        for name, target in links.items():
            visited = {name}
            while target in links:
                if target in visited:
                    raise ValueError('Cyclic capsule alias')
                visited.add(target)
                target = links[target]
            hashes[name] = hashes[target]
        if (not provenance or set(hashes) - {'provenance.json'} != set(provenance['files']) or
                any(hashes[name] != row for name, row in provenance['files'].items())):
            raise ValueError('Original capsule payload differs from its manifest')
        amended = copy.deepcopy(provenance)
        component = amended.setdefault('compiler_header_components', {}).setdefault('CUB', {})
        if component.get('version', '2.3.2') != '2.3.2':
            raise ValueError('Existing measured CUB version differs')
        component.update(version='2.3.2',
            notice='licenses/CUB-2.3.2.txt', license='BSD-3-Clause', license_source=CUB_URL,
            license_sha256=CUB_SHA, sdk_package=SDK_URL, sdk_package_sha256=SDK_SHA,
            version_evidence='Actual package cub/version.cuh defines CUB_VERSION200302')
        amended['license_amendment'] = dict(packaging_source=packaging_source,
            parent_archive=original['archive'], unchanged_elf_payloads=elfs,
            compiler_output_terms='https://docs.nvidia.com/cuda/archive/12.4.1/eula/index.html')
        for name, data in changes.items():
            amended['files'][name] = digest(data)
        changes['provenance.json'] = (json.dumps(amended, indent=2) + '\n').encode()
        for name, data in changes.items():
            member = tarfile.TarInfo('bundle/' + name)
            member.mode, member.size, member.mtime = 0o644, len(data), 0
            outgoing.addfile(member, io.BytesIO(data))
    temporary.rename(destination)
    receipt = dict(amended, archive=dict(name=destination.name, **file_digest(destination)))
    (args.out / 'validation.json').write_text(json.dumps(receipt, indent=2) + '\n')
    (args.out / 'license-completion.json').write_text(json.dumps(dict(
        complete=True, packaging_source=packaging_source, compiled_source=amended['source'],
        elf_payloads_unchanged=len(elfs), compiler_header_components=amended['compiler_header_components'],
        parent_archive=original['archive'], archive=receipt['archive'], model_acceptance=False), indent=2) + '\n')
    print(json.dumps(dict(license_completion=True, unchanged_elfs=len(elfs), archive=receipt['archive'])), flush=True)


if __name__ == '__main__':
    main()
