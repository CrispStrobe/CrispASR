#!/usr/bin/env python3
"""Build Granite5 CUDA validation on GitHub; real GPU acceptance belongs on Kaggle.

NVIDIA user-space libraries are host-provided, never included in this capsule.
"""
import hashlib
import json
import os
import re
from pathlib import Path
import shutil
import subprocess
import tarfile
import urllib.request
from granite5_runtime_licenses import collect_corresponding_sources, collect_licenses, snapshot_dependencies

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'granite5-cuda-out'
BUILD = ROOT / 'granite5-cuda-build'
CUDA = Path(os.environ.get('CUDA_PATH', '/usr/local/cuda'))
OUT.mkdir(exist_ok=True)


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def run(command, label):
    print(label, flush=True)
    meminfo = Path('/proc/meminfo').read_text()
    snapshot = dict(load_average=os.getloadavg(), cpus=os.cpu_count(), meminfo=meminfo)
    (OUT / (label + '-resources.json')).write_text(json.dumps(snapshot, indent=2) + '\n')
    available = int(next(line.split()[1] for line in meminfo.splitlines() if line.startswith('MemAvailable:')))
    if available < 6 * 1024 * 1024 or snapshot['load_average'][0] > 2 * snapshot['cpus']:
        raise RuntimeError(f'Insufficient hosted resource headroom before {label}')
    with (OUT / (label + '.log')).open('w') as log:
        subprocess.run(list(map(str, command)), cwd=ROOT, stdout=log,
                       stderr=subprocess.STDOUT, check=True, timeout=7000)


def main():
    source = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    ggml = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT / 'ggml', text=True).strip()
    receipt = dict(source=source, ggml=ggml, cuda='12.4.1', architectures=[60, 75],
                   issue=552,
                   build_complete=False, gpu_executed=False, model_acceptance=False)
    (OUT / 'validation.json').write_text(json.dumps(receipt, indent=2) + '\n')
    stubs = list(CUDA.rglob('stubs/libcuda.so'))
    assert stubs, 'Toolkit driver stub required for GPU-less CI linker'
    links = BUILD / 'link-stubs'
    links.mkdir(parents=True, exist_ok=True)
    (links / 'libcuda.so.1').symlink_to(stubs[0])
    # Keep the production DL package configuration; inject only these probes.
    # The full test CMake file contains unrelated direct CPU MODULE links.
    proof = BUILD / 'proof-targets.cmake'
    proof.write_text(f"""add_executable(granite5-stage-probe "{ROOT}/tools/ci-heavy/granite5_stage_probe.cpp")
    target_include_directories(granite5-stage-probe PRIVATE "{ROOT}/src")
    target_link_libraries(granite5-stage-probe PRIVATE granite_speech ggml-base)
    target_compile_features(granite5-stage-probe PRIVATE cxx_std_17)
    add_executable(granite5-rope-probe "{ROOT}/tools/ci-heavy/granite5_rope_probe.cpp")
    target_include_directories(granite5-rope-probe PRIVATE "{ROOT}/src")
    target_link_libraries(granite5-rope-probe PRIVATE ggml ggml-base crispasr_link_ggml_cpu)
    target_compile_features(granite5-rope-probe PRIVATE cxx_std_17)
    set_target_properties(granite5-stage-probe granite5-rope-probe PROPERTIES RUNTIME_OUTPUT_DIRECTORY "{BUILD}/bin")
    """)
    run(['cmake', '-G', 'Ninja', '-S', ROOT, '-B', BUILD,
         f'-DCMAKE_PROJECT_crispasr_INCLUDE={proof}',
         '-DCMAKE_BUILD_TYPE=Release', '-DBUILD_SHARED_LIBS=ON',
         '-DGGML_BACKEND_DL=ON', '-DGGML_CUDA=ON', '-DCMAKE_CUDA_ARCHITECTURES=60;75',
         '-DGGML_NATIVE=OFF', '-DGGML_BLAS=OFF', '-DGGML_OPENBLAS=OFF',
         '-DGGML_VULKAN=OFF', '-DGGML_METAL=OFF',
         '-DCRISPASR_BUILD_TESTS=OFF', '-DCRISPASR_BUILD_SERVER=OFF',
         '-DCRISPASR_BUILD_EXAMPLES=ON',
         f'-DCMAKE_EXE_LINKER_FLAGS=-Wl,-rpath-link,{links}',
         '-DCMAKE_C_COMPILER_LAUNCHER=ccache', '-DCMAKE_CXX_COMPILER_LAUNCHER=ccache',
         '-DCMAKE_CUDA_COMPILER_LAUNCHER=ccache'], 'configure')
    run(['cmake', '--build', BUILD, '--target', 'crispasr-cli', 'crispasr-lib',
         'granite5-stage-probe', 'granite5-rope-probe', '-j2'], 'build')
    bundle = OUT / 'bundle'
    bundle.mkdir()
    for name in ['crispasr']:
        shutil.copy2(BUILD / 'bin' / name, bundle / name)
    for name in ['granite5-stage-probe', 'granite5-rope-probe']:
        shutil.copy2(BUILD / 'bin' / name, bundle / name)
    for path in BUILD.rglob('*.so*'):
        if path.is_file() and not path.name.startswith('libcuda.so'):
            soname = subprocess.check_output(['patchelf', '--print-soname', str(path)], text=True).strip()
            dest = bundle / (soname or path.name)
            if dest.exists():
                assert sha(dest) == sha(path), f'Conflicting library {dest.name}'
            else:
                shutil.copy2(path, dest)
            # Explicit C ABI paths and dlopen discovery need the original
            # aliases as well as the ELF SONAME (e.g. libcrispasr.so -> .so.1).
            alias = bundle / path.name
            if path.name != dest.name and not alias.exists():
                alias.symlink_to(dest.name)
    dependencies = snapshot_dependencies(bundle, BUILD)
    # Kaggle supplies its own CUDA runtime. Do not redistribute NVIDIA blobs.
    assert not any(p.name.startswith(('libcuda', 'libcublas', 'libnv')) for p in bundle.iterdir())
    for path in bundle.iterdir():
        subprocess.run(['patchelf', '--set-rpath', '$ORIGIN', str(path)], check=True)
    run(['bash', ROOT / 'scripts/bundle-linux-runtime.sh', bundle], 'runtime-closure')
    run(['python3', ROOT / 'scripts/check-bundled-deps.py', bundle,
         '--allow', 'libcuda.so.*', '--allow', 'libcudart.so.*',
         '--allow', 'libcublas*.so.*', '--allow', 'libnv*.so.*'], 'dependency-check')
    assert not any(p.name.startswith(('libcuda', 'libcublas', 'libnv')) for p in bundle.iterdir())
    assert (bundle / 'libggml-cuda.so').is_file()
    assert (bundle / 'libcrispasr.so').is_file()
    # The binding must be the same composition as the compiled C ABI.
    shutil.copytree(ROOT / 'python/crispasr', bundle / 'python/crispasr',
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '*.so*'))
    shutil.copy2(ROOT / 'LICENSE', bundle)
    shutil.copy2(ROOT / 'THIRD_PARTY_NOTICES.txt', bundle)
    licenses = bundle / 'licenses'
    licenses.mkdir()
    for source_path, name in [
        (ROOT / 'ggml/LICENSE', 'ggml-MIT.txt'),
        (ROOT / 'third_party/c2pa-audio/LICENSE', 'c2pa-audio.txt'),
        (Path('/usr/share/common-licenses/GPL-3'), 'GPL-3.txt'),
        (Path('/usr/share/doc/libgomp1/copyright'), 'libgomp-copyright.txt'),
    ]:
        assert source_path.is_file(), source_path
        shutil.copy2(source_path, licenses / name)
    (licenses / 'libgomp-package.txt').write_text(subprocess.check_output(
        ['dpkg-query', '-W', '-f', '${Package} ${Version}\n${source:Package} ${source:Version}\n', 'libgomp1'],
        text=True))
    runtime_licenses = collect_licenses(bundle, dependencies)
    runtime_licenses = collect_corresponding_sources(bundle, runtime_licenses)
    (bundle / 'README.txt').write_text(
        'Internal Granite5 validation capsule, not a model-support release.\n'
        'No model weights or NVIDIA libraries are included. Kaggle must provide\n'
        'the NVIDIA driver and CUDA 12 runtime/cuBLAS libraries. Build completion\n'
        'does not establish CPU or CUDA model acceptance. Preserve LICENSE and\n'
        'THIRD_PARTY_NOTICES.txt for the compiled code and its dependencies.\n')
    receipt['runtime_sources'] = {
        str(p.relative_to(ROOT)): sha(p)
        for folder in [ROOT / 'src', ROOT / 'examples/cli']
        for p in folder.rglob('*') if p.is_file() and p.suffix in {'.cpp', '.c', '.h', '.hpp'}
    }
    cub_headers = [p for p in CUDA.rglob('version.cuh') if p.parent.name == 'cub']
    cub_versions = {}
    for header in cub_headers:
        match = re.search(r'^\s*#\s*define\s+CUB_VERSION\s+(\d+)', header.read_text(), re.MULTILINE)
        if match:
            cub_versions[str(header.relative_to(CUDA))] = dict(version=int(match.group(1)), sha256=sha(header))
    assert cub_versions and all(v['version'] == 200302 for v in cub_versions.values()), cub_versions
    assert 'NVIDIA CUB 2.3.2' in (bundle / 'THIRD_PARTY_NOTICES.txt').read_text()
    cub_url = 'https://raw.githubusercontent.com/NVIDIA/cccl/v2.3.2/cub/LICENSE.TXT'
    cub_sha = '2e15eba2e130f3e80d9e91724c7dce3781ce99316324681ff54b00c1fc9759b3'
    with urllib.request.urlopen(cub_url, timeout=120) as response:
        cub_license = response.read()
    assert hashlib.sha256(cub_license).hexdigest() == cub_sha
    (bundle / 'licenses/CUB-2.3.2.txt').write_bytes(cub_license)
    receipt['compiler_header_components'] = dict(CUB=dict(version='2.3.2', headers=cub_versions,
        notice='licenses/CUB-2.3.2.txt', license='BSD-3-Clause',
        license_source=cub_url, license_sha256=cub_sha))
    receipt['probe_sources'] = {
        name: sha(ROOT / 'tools/ci-heavy' / name)
        for name in ('granite5_stage_probe.cpp', 'granite5_rope_probe.cpp')
    }
    receipt['host_provided_nvidia_libraries'] = True
    receipt['external_runtime_packages'] = runtime_licenses['packages']
    receipt['build_complete'] = True
    receipt['files'] = {str(p.relative_to(bundle)): dict(sha256=sha(p), bytes=p.stat().st_size)
                        for p in bundle.rglob('*') if p.is_file()}
    (bundle / 'provenance.json').write_text(json.dumps(receipt, indent=2) + '\n')
    archive = OUT / 'granite5-cuda-validation.tar.gz'
    with tarfile.open(archive, 'w:gz') as tar:
        tar.add(bundle, arcname='bundle')
    receipt['archive'] = dict(name=archive.name, sha256=sha(archive), bytes=archive.stat().st_size)
    (OUT / 'validation.json').write_text(json.dumps(receipt, indent=2) + '\n')
    shutil.rmtree(bundle)
    print(json.dumps(receipt['archive']), flush=True)


if __name__ == '__main__':
    main()
