#!/usr/bin/env python3
"""Profile unchanged MiMo GPU decode graph phases with exact CLI/session A/B."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

SCRIPT_VERSION = 'mimo-graph-profile-v1'
SOURCE = 'b8ab3249df7a791afefbde2541c41646fd0aac23'
GGML = 'c36dab89b662838f0f5d4826c399198c0b90bbfc'
CACHE = {'repo': 'cstr/crispasr-ccache', 'file': 'mimo-pr492/sm75-full-asr-v3.tar', 'revision': '0480221f9f7e5b16773ad8ad7e673cfc1afcdc94', 'sha256': '5cf9eb772a1190c2a058d9ec682ea0b08378b0d723bfbbef7235d80293833344', 'bytes': 78080000}
WORK = Path('/kaggle/working')
SCRATCH = Path('/kaggle/temp/mimo-graph-profile')
REPO = SCRATCH / 'CrispASR'
OUT = WORK / 'mimo-asr'


def main():
    SCRATCH.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    os.environ.update(PYTHONUNBUFFERED='1', TMPDIR=str(SCRATCH),
                      HF_HOME=str(SCRATCH / 'hf'), KAGGLE_KERNEL_REF='mimo-graph-profile-v1')
    devices = subprocess.check_output(['nvidia-smi', '--query-gpu=name,compute_cap,memory.total',
                                       '--format=csv'], text=True)
    print(SCRIPT_VERSION, devices, flush=True)
    subprocess.run(['git', 'clone', '--depth=1', '--branch', 'perf/mimo-voxcpm-profile',
                    'https://github.com/CrispStrobe/CrispASR.git', REPO], check=True, timeout=600)
    subprocess.run(['git', 'fetch', '--depth=1', 'origin', SOURCE], cwd=REPO, check=True, timeout=600)
    subprocess.run(['git', 'checkout', '--detach', SOURCE], cwd=REPO, check=True)
    subprocess.run(['git', 'submodule', 'update', '--init', '--recursive'], cwd=REPO,
                   check=True, timeout=1200)
    assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip() == SOURCE
    assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO / 'ggml', text=True).strip() == GGML
    # CLI consent hashes require c2pa-audio even with C2PA signing disabled.
    assert (REPO / 'third_party/c2pa-audio/src/sha256.h').is_file()
    submodules = subprocess.check_output(['git', 'submodule', 'status', '--recursive'],
                                        cwd=REPO, text=True)
    assert all(line.startswith(' ') for line in submodules.splitlines()), submodules
    os.chdir(REPO)
    sys.path.insert(0, str(REPO / 'tools/kaggle'))
    import kaggle_harness as kh
    kh.init_progress(WORK / 'progress.jsonl')
    kh.resolve_hf_token()
    kh.provenance(SCRIPT_VERSION, REPO)
    kh.install_build_toolchain()
    arch = kh.detect_cuda_arch()
    # Reuse only actual Kaggle compiler output, with an immutable HF pin/hash.
    # The existing private cache repo holds build objects, never model weights.
    if str(arch) == '75':
        import tarfile
        from huggingface_hub import hf_hub_download
        archive = Path(hf_hub_download(CACHE['repo'], CACHE['file'], repo_type='dataset',
                           revision=CACHE['revision'], local_dir=SCRATCH / 'build-cache'))
        assert hashlib.sha256(archive.read_bytes()).hexdigest() == CACHE['sha256']
        destination = Path(os.environ['CCACHE_DIR']).parent
        with tarfile.open(archive) as tf:
            assert all(m.name == '.ccache' or m.name.startswith('.ccache/') for m in tf)
            tf.extractall(destination, filter='data')
        kh.step('cuda.cache.warmed', **CACHE)
    else:
        kh.step('cuda.cache.skipped', arch=arch, cached_arch='75')
    # CUDA performs the native inference. Do not install or use PyTorch CUDA.
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'numpy', 'gguf',
                    'huggingface_hub', 'soundfile'], check=True, timeout=1200)
    subprocess.run(['apt-get', '-o', 'Acquire::Retries=2', '-o', 'Acquire::http::Timeout=30',
                    '-o', 'Acquire::https::Timeout=30', 'install', '-y', 'ffmpeg'], check=True, timeout=600)
    build = SCRATCH / 'build'
    flags = ['-DCMAKE_BUILD_TYPE=Release', '-DBUILD_SHARED_LIBS=ON', '-DGGML_NATIVE=OFF',
             '-DGGML_BLAS=OFF', '-DCRISPASR_MEL_BLAS=OFF', '-DCRISPASR_BUILD_TESTS=OFF',
             '-DCRISPASR_BUILD_EXAMPLES=ON', '-DCRISPASR_BUILD_SERVER=OFF',
             *kh.cuda_build_flags(arch), *kh.cache_and_link_flags()]
    import shlex
    (OUT / 'hardware.json').write_text(json.dumps(dict(script_version=SCRIPT_VERSION, source=SOURCE,
        ggml=GGML, devices=devices, arch=arch, cmake_flags=flags,
        kernel_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        cache=CACHE, submodules=submodules, scope=__doc__), indent=2) + '\n')
    with kh.build_heartbeat('cuda-configure'):
        kh.sh_with_progress(shlex.join(['cmake', '-S', str(REPO), '-B', str(build), '-G', 'Ninja', *flags]))
    with kh.build_heartbeat('cuda-build'):
        kh.sh_with_progress(shlex.join(['cmake', '--build', str(build), '--target', 'crispasr-lib', 'crispasr-cli',
                                       '-j' + kh.safe_build_jobs(gpu=True)]))
    assert next(build.rglob('libcrispasr.so')).is_file()
    assert (build / 'bin/crispasr').is_file()
    (OUT / 'CMakeCache.txt').write_bytes((build / 'CMakeCache.txt').read_bytes())
    os.environ.update(HEAVY_OUT=str(OUT), HEAVY_SCRATCH=str(SCRATCH),
                      MIMO_PROFILE_BUILD=str(build), CUDA_VISIBLE_DEVICES='0')
    kh.step('asr.validation.begin', source=SOURCE, arch=arch)
    try:
        with kh.build_heartbeat('asr-cuda-validation'):
            result = subprocess.run([sys.executable, REPO / 'tools/ci-heavy/mimo_graph_profile.py'],
                                    cwd=REPO, timeout=7200)
        kh.step('asr.validation.end', returncode=result.returncode)
    finally:
        kh.export_ccache_tar(OUT / 'ccache.tar')
        kh._push_progress_to_hf(force=True)
    assert result.returncode == 0, 'CUDA graph-phase profile/decoded-output acceptance failed; inspect saved speech/logs'


if __name__ == '__main__':
    main()
