#!/usr/bin/env python3
"""MiMo PR492 identical-input CUDA attention diagnostic; no full-ASR claim.

Uses the actual native tokenizer, including device weights/RVQ, and the pinned
official Python encoder on the GPU's own conv2 input. PyTorch is CPU-only here:
ggml compiles and executes CUDA for the detected GPU, including a P100. Pascal
results do not establish integer MMQ support. Same-weight promotion is a
diagnostic, not recovered original precision or full ASR acceptance.
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

SCRIPT_VERSION = 'mimo-pr492-tokenizer-cuda-v5'
SOURCE = '534a929f75e61febf4143a96fdaa88dbdb811528'
GGML = 'c36dab89b662838f0f5d4826c399198c0b90bbfc'
CACHE = {'repo': 'cstr/crispasr-ccache', 'file': 'mimo-pr492/sm75-v3.tar', 'revision': '7bac5a6c0e5cae845d8ded349a2bcb7d0a23050e', 'sha256': 'a60bf4b1ea7d0db9099ead0383133c2ac22b0f5c6c8c0117857dc1ef59035b95', 'bytes': 51097600, 'source': '389c3c712081f8612817e20b7f9e784d03a7b703'}
WORK = Path('/kaggle/working')
SCRATCH = Path('/kaggle/temp/mimo-pr492')
REPO = SCRATCH / 'CrispASR'
OUT = WORK / 'mimo-tokenizer'


def main():
    SCRATCH.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    os.environ.update(PYTHONUNBUFFERED='1', TMPDIR=str(SCRATCH),
                      HF_HOME=str(SCRATCH / 'hf'), KAGGLE_KERNEL_REF='mimo-pr492-tokenizer-cuda-v5')
    devices = subprocess.check_output(['nvidia-smi', '--query-gpu=name,compute_cap,memory.total',
                                       '--format=csv'], text=True)
    print(SCRIPT_VERSION, devices, flush=True)
    subprocess.run(['git', 'clone', '--depth=1', '--branch', 'review/pr492-acceptance',
                    'https://github.com/CrispStrobe/CrispASR.git', REPO], check=True, timeout=600)
    subprocess.run(['git', 'fetch', '--depth=1', 'origin', SOURCE], cwd=REPO, check=True, timeout=600)
    subprocess.run(['git', 'checkout', '--detach', SOURCE], cwd=REPO, check=True)
    subprocess.run(['git', 'submodule', 'update', '--init', '--recursive', 'ggml'], cwd=REPO,
                   check=True, timeout=1200)
    assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip() == SOURCE
    assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO / 'ggml', text=True).strip() == GGML
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
                    'huggingface_hub', 'soundfile', 'transformers==4.57.6'], check=True, timeout=1200)
    subprocess.run(['apt-get', '-o', 'Acquire::Retries=2', '-o', 'Acquire::http::Timeout=30',
                    '-o', 'Acquire::https::Timeout=30', 'install', '-y', 'ffmpeg'], check=True, timeout=600)
    sys.path.insert(0, str(REPO / 'tools/ci-heavy'))
    from pr492_tokenizer_cuda_precision import patch_cuda
    patch = patch_cuda(REPO, OUT)
    kh.step('cuda.precision.patch', **patch)
    from pr492_attention_precision import patch_attention
    kh.step('cuda.attention.patch', **patch_attention(REPO, OUT))
    wrapper = SCRATCH / 'wrapper'
    wrapper.mkdir()
    (wrapper / 'entry.cpp').write_text('''#include "mimo_tokenizer.h"
extern "C" int mimo_probe_threads() { return mimo_tokenizer_context_default_params().n_threads; }
''')
    (wrapper / 'add-probe.cmake').write_text(f'''function(mimo_diag_add_probe)
  add_library(mimo_tokenizer_probe SHARED "{wrapper / 'entry.cpp'}")
  target_link_libraries(mimo_tokenizer_probe PRIVATE mimo_tokenizer)
  add_executable(mimo_attention_probe "{REPO / 'tools/ci-heavy/pr492_attention_probe.cpp'}")
  target_link_libraries(mimo_attention_probe PRIVATE ggml)
  file(GENERATE OUTPUT "{wrapper / 'probe-path.txt'}" CONTENT "$<TARGET_FILE:mimo_attention_probe>")
endfunction()
cmake_language(DEFER CALL mimo_diag_add_probe)
''')
    build = SCRATCH / 'build'
    flags = ['-DCMAKE_BUILD_TYPE=Release', '-DBUILD_SHARED_LIBS=ON', '-DGGML_NATIVE=OFF',
             '-DGGML_BLAS=OFF', '-DCRISPASR_MEL_BLAS=OFF', '-DCRISPASR_BUILD_TESTS=OFF',
             '-DCRISPASR_BUILD_EXAMPLES=OFF', '-DCRISPASR_BUILD_SERVER=OFF',
             f"-DCMAKE_PROJECT_crispasr_INCLUDE={wrapper / 'add-probe.cmake'}",
             *kh.cuda_build_flags(arch), *kh.cache_and_link_flags()]
    import shlex
    with kh.build_heartbeat('cuda-configure'):
        kh.sh_with_progress(shlex.join(['cmake', '-S', str(REPO), '-B', str(build), '-G', 'Ninja', *flags]))
    with kh.build_heartbeat('cuda-build'):
        kh.sh_with_progress(shlex.join(['cmake', '--build', str(build), '--target', 'mimo_tokenizer_probe', 'mimo_attention_probe',
                                       '-j' + kh.safe_build_jobs(gpu=True)]))
    library = build / 'libmimo_tokenizer_probe.so'
    assert library.is_file()
    probe = Path((wrapper / 'probe-path.txt').read_text().strip()).resolve(strict=True)
    assert probe.is_file() and os.access(probe, os.X_OK), probe
    (OUT / 'CMakeCache.txt').write_bytes((build / 'CMakeCache.txt').read_bytes())
    (OUT / 'hardware.json').write_text(json.dumps(dict(script_version=SCRIPT_VERSION, source=SOURCE,
        ggml=GGML, devices=devices, arch=arch, cmake_flags=flags,
        kernel_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        attention_probe=str(probe), cache=CACHE, scope=__doc__), indent=2) + '\n')
    os.environ.update(HEAVY_OUT=str(OUT), HEAVY_SCRATCH=str(SCRATCH),
                      MIMO_TOKENIZER_DIAG_GPU='1', MIMO_TOKENIZER_DIAG_LIB=str(library),
                      MIMO_DIAG_ATTN_PROBE=str(probe))
    kh.step('tokenizer.diagnosis.begin', source=SOURCE, arch=arch)
    with kh.build_heartbeat('tokenizer-cuda-diagnosis'):
        result = subprocess.run([sys.executable, REPO / 'tools/ci-heavy/pr492_attention_precision_diagnose.py'],
                                cwd=REPO, timeout=7200)
    kh.step('tokenizer.diagnosis.end', returncode=result.returncode)
    kh.export_ccache_tar(OUT / 'ccache.tar')
    kh._push_progress_to_hf(force=True)
    assert result.returncode == 0, 'CUDA identical-input diagnostic controls failed; inspect saved metrics/logs'


if __name__ == '__main__':
    main()
