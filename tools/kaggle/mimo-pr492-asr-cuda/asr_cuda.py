#!/usr/bin/env python3
"""Full CUDA MiMo CLI/session speech acceptance before repeated precision A/B.

The original pinned Q4 LM/codec run on one actual GPU; numerical overrides
are diagnostic only, restricted to tokenizer weights and unmasked D64/H20.
No production default or model change; CANN/full PR acceptance remains separate.
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

SCRIPT_VERSION = 'mimo-pr492-asr-cuda-v3'
SOURCE = '6b90353678f525e634c1a359355ae15c2c815ace'
GGML = 'c36dab89b662838f0f5d4826c399198c0b90bbfc'
CACHE = {'repo': 'cstr/crispasr-ccache', 'file': 'mimo-pr492/sm75-full-asr-v2.tar', 'revision': '2d1f1ff707b75c962f369da2e3730296dee4e61e', 'sha256': 'a3184b5c746ac7d5a3f5b4bebda9339b63c10ee5454a8d59ae63d6188b75c233', 'bytes': 76267520, 'source': 'fee98afbc80296e4bb7b63ccf6b8d8952d5433f2'}
WORK = Path('/kaggle/working')
SCRATCH = Path('/kaggle/temp/mimo-pr492-asr')
REPO = SCRATCH / 'CrispASR'
OUT = WORK / 'mimo-asr'


def main():
    SCRATCH.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    os.environ.update(PYTHONUNBUFFERED='1', TMPDIR=str(SCRATCH),
                      HF_HOME=str(SCRATCH / 'hf'), KAGGLE_KERNEL_REF='mimo-pr492-asr-cuda-v3')
    devices = subprocess.check_output(['nvidia-smi', '--query-gpu=name,compute_cap,memory.total',
                                       '--format=csv'], text=True)
    print(SCRIPT_VERSION, devices, flush=True)
    subprocess.run(['git', 'clone', '--depth=1', '--branch', 'review/pr492-acceptance',
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
    sys.path.insert(0, str(REPO / 'tools/ci-heavy'))
    from pr492_tokenizer_cuda_precision import patch_cuda
    patch = patch_cuda(REPO, OUT, encoder_only=True)
    kh.step('cuda.precision.patch', **patch)
    from pr492_attention_precision import patch_attention
    kh.step('cuda.attention.patch', **patch_attention(REPO, OUT))
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
                      MIMO_ASR_DIAG_BUILD=str(build), CUDA_VISIBLE_DEVICES='0')
    kh.step('asr.validation.begin', source=SOURCE, arch=arch)
    try:
        with kh.build_heartbeat('asr-cuda-validation'):
            result = subprocess.run([sys.executable, REPO / 'tools/ci-heavy/pr492_asr_cuda.py'],
                                    cwd=REPO, timeout=7200)
        kh.step('asr.validation.end', returncode=result.returncode)
    finally:
        kh.export_ccache_tar(OUT / 'ccache.tar')
        kh._push_progress_to_hf(force=True)
    assert result.returncode == 0, 'CUDA full-ASR acceptance/profile failed; inspect saved speech/logs'


if __name__ == '__main__':
    main()
