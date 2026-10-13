#!/usr/bin/env python3
"""Build the composed Dia cache candidate on GH; execute speech only on a GPU.

Probe-only package: includes server support, diff, shared ABI and bindings.
A green build accepts no model, generation, quantization or pending PR.
"""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'dia-cuda-out'
BUILD = ROOT / 'dia-cuda-build'
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
    with (OUT / (label + '.log')).open('w') as log:
        subprocess.run(list(map(str, command)), cwd=ROOT, stdout=log,
                       stderr=subprocess.STDOUT, check=True, timeout=7000)


def pinned_runtime_sources(root):
    """Reject a different neural/cache implementation before expensive builds."""
    expected_runtime = {
        'src/dia_tts.cpp': 'fc91490e1f4e9c842e208f07ed7335e10cfa18d43d4eef67cf697652088d973d',
        'src/dia_sampling.h': 'ad23c213ed9fde411f4ce6648cd3383470dc4fb4504195794bce96b2c655f6cf',
        'src/dia_exponential_rng.h': 'e2d42a64b8b4d0bbc3948c4da29b8395d42f20b4df10b80bec1b101c41b7df0e',
        'examples/cli/crispasr_dia_diff.h': 'd8e896fe52982107da26fa1fb8d857bd312113417484e6475172ca6d0249d364',
    }
    actual = {name: sha(root / name) for name in expected_runtime}
    if actual != expected_runtime:
        raise ValueError('Pinned Dia candidate bytes changed')
    return actual


def main():
    source = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    ggml = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT / 'ggml', text=True).strip()
    receipt = dict(source=source, ggml=ggml, cuda='12.4.1', architectures=[60, 75],
                   cpu_control_source='2205d920d879108b73651ebb9f3eb9499da4e5f5',
                   cpu_control_run=38357576078, cpu_control_accepted=False,
                   scope='CUDA compilation/dependency closure; explicit GPU replay request repaired; separate model diagnostics remain unaccepted',
                   cpu_diff_dispatch_sha256='d2b5d639f35f4c9b5e5430f881692c90268812a0eee7b25f7b25f57de2701f2e',
                   diff_device_dispatch_changed=True,
                   build_complete=False, gpu_executed=False, model_acceptance=False)
    receipt['tracked_worktree_dirty'] = bool(subprocess.check_output(
        ['git', 'status', '--porcelain', '--untracked-files=no'], cwd=ROOT))
    assert not receipt['tracked_worktree_dirty'], 'Tracked build source changed'
    receipt['nvcc_version'] = subprocess.check_output(['nvcc', '--version'], text=True).strip()
    assert 'release 12.4,' in receipt['nvcc_version'], 'Unexpected CUDA compiler'
    sources = OUT / 'sources'
    sources.mkdir()
    runtime = ('src/dia_tts.cpp', 'src/dia_sampling.h', 'src/dia_exponential_rng.h',
               'examples/cli/crispasr_dia_diff.h')
    receipt['runtime_source_sha256'] = pinned_runtime_sources(ROOT)
    for name in runtime + ('src/crispasr_c_api.cpp', 'python/crispasr/_binding.py',
                           'tools/ci-heavy/dia_cuda_bundle.py'):
        dest = sources / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, dest)
    receipt['source_files'] = {str(p.relative_to(sources)): dict(sha256=sha(p), bytes=p.stat().st_size)
                               for p in sources.rglob('*') if p.is_file()}
    (OUT / 'validation.json').write_text(json.dumps(receipt, indent=2) + '\n')
    stubs = list(CUDA.rglob('stubs/libcuda.so'))
    assert stubs, 'Toolkit driver stub required for GPU-less CI linker'
    links = BUILD / 'link-stubs'
    links.mkdir(parents=True, exist_ok=True)
    (links / 'libcuda.so.1').symlink_to(stubs[0])
    run(['cmake', '-G', 'Ninja', '-S', ROOT, '-B', BUILD,
         '-DCMAKE_BUILD_TYPE=Release', '-DBUILD_SHARED_LIBS=ON',
         '-DGGML_BACKEND_DL=ON', '-DGGML_CUDA=ON', '-DCMAKE_CUDA_ARCHITECTURES=60;75',
         '-DGGML_NATIVE=OFF', '-DGGML_BLAS=OFF', '-DGGML_OPENBLAS=OFF',
         '-DGGML_VULKAN=OFF', '-DGGML_METAL=OFF',
         '-DCRISPASR_BUILD_TESTS=OFF', '-DCRISPASR_BUILD_SERVER=ON',
         '-DCRISPASR_BUILD_EXAMPLES=ON',
         f'-DCMAKE_EXE_LINKER_FLAGS=-Wl,-rpath-link,{links}',
         '-DCMAKE_C_COMPILER_LAUNCHER=ccache', '-DCMAKE_CXX_COMPILER_LAUNCHER=ccache',
         '-DCMAKE_CUDA_COMPILER_LAUNCHER=ccache'], 'configure')
    run(['cmake', '--build', BUILD, '--target', 'crispasr-cli', 'crispasr-lib',
         'crispasr-diff', 'crispasr-quantize', 'crispasr-server', '-j2'], 'build')
    bundle = OUT / 'bundle'
    bundle.mkdir()
    for name in ['crispasr', 'crispasr-diff', 'crispasr-quantize', 'crispasr-server']:
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
    # Copy toolkit user-space runtime, never driver or compatibility shim.
    for name in ['libcudart.so.12', 'libcublas.so.12', 'libcublasLt.so.12']:
        paths = list(CUDA.rglob(name))
        assert paths, name
        shutil.copy2(paths[0], bundle / name)
    for path in bundle.iterdir():
        subprocess.run(['patchelf', '--set-rpath', '$ORIGIN', str(path)], check=True)
    run(['bash', ROOT / 'scripts/bundle-linux-runtime.sh', bundle], 'runtime-closure')
    run(['python3', ROOT / 'scripts/check-bundled-deps.py', bundle,
         '--allow', 'libcuda.so.*'], 'dependency-check')
    assert (bundle / 'libggml-cuda.so').is_file()
    assert (bundle / 'libcrispasr.so').is_file()
    # The binding must be the same composition as the compiled C ABI.
    shutil.copytree(ROOT / 'python/crispasr', bundle / 'python/crispasr',
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '*.so*'))
    shutil.copy2(ROOT / 'LICENSE', bundle)
    shutil.copy2(ROOT / 'THIRD_PARTY_NOTICES.txt', bundle)
    receipt['build_complete'] = True
    receipt['files'] = {str(p.relative_to(bundle)): dict(sha256=sha(p), bytes=p.stat().st_size)
                        for p in bundle.rglob('*') if p.is_file()}
    (bundle / 'provenance.json').write_text(json.dumps(receipt, indent=2) + '\n')
    archive = OUT / 'dia-cuda-speech-validation.tar.gz'
    with tarfile.open(archive, 'w:gz') as tar:
        tar.add(bundle, arcname='bundle')
    receipt['archive'] = dict(name=archive.name, sha256=sha(archive), bytes=archive.stat().st_size)
    (OUT / 'validation.json').write_text(json.dumps(receipt, indent=2) + '\n')
    shutil.rmtree(bundle)
    print(json.dumps(receipt['archive']), flush=True)


if __name__ == '__main__':
    main()
