"""Prepare a reviewed, self-contained script; never uploads or submits."""
import argparse
import base64
import gzip
import lzma
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile
import zipfile

SOURCE = '8d3b6aa393046b048643709555efc76719ee0bcc'
GGML = '1b972bb754aa5f83e2d872ccf37a335717f6658d'
GGUF_WHEEL_SHA256 = 'af93f7ef198a265cbde5fa6a6b3101528bca285903949ab0a3e591cd993a1864'


def gguf_files(path):
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != GGUF_WHEEL_SHA256:
        raise ValueError('expected the exact official gguf 0.18.0 wheel')
    with zipfile.ZipFile(io.BytesIO(data)) as wheel:
        metadata = wheel.read('gguf-0.18.0.dist-info/METADATA').decode()
        license_data = wheel.read('gguf-0.18.0.dist-info/licenses/LICENSE')
        if 'Version: 0.18.0\n' not in metadata or b'MIT License' not in license_data:
            raise ValueError('unexpected gguf wheel version/license')
        result = {'vendor/'+name:wheel.read(name) for name in wheel.namelist()
                  if name.startswith('gguf/') and name.endswith('.py')}
        result['vendor/GGUF-LICENSE'] = license_data
        result['vendor/GGUF-METADATA'] = metadata.encode()
        result['vendor/GGUF-PROVENANCE.json'] = json.dumps(dict(version='0.18.0',wheel_sha256=GGUF_WHEEL_SHA256,
            source='https://pypi.org/project/gguf/0.18.0/')).encode()
        return result


def archive_files(repository, revision, paths, prefix):
    data = subprocess.check_output(['git', '-C', str(repository), 'archive', revision, *paths])
    result = {}
    with tarfile.open(fileobj=io.BytesIO(data)) as archive:
        for item in archive:
            if item.isfile():
                result[prefix+item.name] = archive.extractfile(item).read()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--native', type=Path, required=True, help='reviewed GCC11 frontend DSO/probe/profile directory')
    parser.add_argument('--gguf-wheel', type=Path, required=True, help='exact official gguf 0.18.0 wheel')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    files = archive_files(root, SOURCE, ['LICENSE', 'src/core', 'cmake/CrispasrReuseMkl.cmake',
        'cmake/CrispasrReuseCuda.cmake', 'examples/cli/crispasr_diff.h', 'examples/cli/crispasr_diff.cpp',
        'examples/cli/crispasr_backend.h', 'examples/cli/crispasr_speaker_identity_models.h',
        'examples/cli/crispasr_speaker_identity.h',
        'examples/cli/reuse_diff.h', 'examples/cli/reuse_frontend_profile.h',
        'tools/ci-heavy/issue579_reuse', 'tools/reference_backends/reuse.py', 'tools/dump_reference.py',
        'tools/kaggle/kaggle_harness.py', 'tools/kaggle/reuse-quality-cpu-20261112/quality_cpu.py',
        'models/convert-reuse-to-gguf.py', 'tests/test-reuse-diff-controls.cpp',
        'tests/test_reuse_mkl_probe.py', 'tests/test-reuse-frontend-profile.cpp'], 'source/')
    files.update(archive_files(root/'ggml', GGML,
                              ['CMakeLists.txt', 'LICENSE', 'cmake', 'src', 'include', 'ggml.pc.in'], 'source/ggml/'))
    def needed(name):
        if name.startswith('source/src/core/'):
            return Path(name).name.startswith(('reuse_', 'gguf_loader', 'ggml_cpu_backend', 'wav_reader'))
        if name.startswith('source/ggml/src/'):
            relative = name.removeprefix('source/ggml/src/')
            if '/' not in relative:
                return True
            if not relative.startswith('ggml-cpu/'):
                return False
            cpu = relative.removeprefix('ggml-cpu/')
            if cpu.startswith('arch/'):
                return cpu.startswith('arch/x86/')
            return not cpu.startswith(('kleidiai/', 'spacemit/'))
        return True
    files = {name:data for name,data in files.items() if needed(name)}
    # The original standalone CMake defines all its test targets at configure
    # time, so include their exact sources even though only two targets build.
    test_names = subprocess.check_output(['git', '-C', str(root), 'ls-tree', '-r', '--name-only', SOURCE, 'tests'], text=True).splitlines()
    test_names = [name for name in test_names if name.startswith('tests/test-reuse-') and name.endswith('.cpp')]
    files.update(archive_files(root, SOURCE, test_names, 'source/'))
    files.update(gguf_files(args.gguf_wheel))
    for name in ('libreuse-frontend.so', 'reuse-frontend-mkl-probe', 'reuse-profile-mkl'):
        files['native/'+name] = (args.native/name).read_bytes()
    # The native frontend source is unmodified; build commands and ELF floor
    # checks travel with the generated payload for review/reproduction.
    files['native/build-receipt.json'] = (args.native/'build-receipt.json').read_bytes()
    manifest = {name:hashlib.sha256(data).hexdigest() for name,data in sorted(files.items())}
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode='w') as archive:
        for name, data in sorted(files.items()):
            item = tarfile.TarInfo(name)
            item.size, item.mtime, item.mode = len(data), 0, 0o644
            archive.addfile(item, io.BytesIO(data))
    data = stream.getvalue()
    payload = dict(sha256=hashlib.sha256(data).hexdigest(), manifest=manifest,
                   data=base64.b85encode(lzma.compress(data, preset=9)).decode())
    runner = Path(__file__).with_name('qualification.py').read_text()
    runner = runner.replace('PAYLOAD = None  # replaced in the generated single-file submission', 'PAYLOAD = '+repr(payload))
    if args.output.exists():
        raise ValueError('refusing to overwrite a prepared submission')
    args.output.write_text(runner)
    args.output.with_suffix('.manifest.json').write_text(json.dumps(dict(source=SOURCE,ggml=GGML,
        script_sha256=hashlib.sha256(runner.encode()).hexdigest(), script_bytes=len(runner.encode()),
        payload_sha256=payload['sha256'], members=manifest),indent=2)+'\n')
    print(json.dumps(dict(script_bytes=len(runner), members=len(files), script_sha256=hashlib.sha256(runner.encode()).hexdigest())))


if __name__ == '__main__':
    main()
