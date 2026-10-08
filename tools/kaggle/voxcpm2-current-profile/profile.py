#!/usr/bin/env python3
"""Profile ten-step VoxCPM2 on actual NVIDIA Vulkan; cold/warm setup and speech gates."""
import hashlib
import json
import os
import re
from pathlib import Path
import subprocess
import sys

SCRIPT_VERSION = 'voxcpm2-current-profile-v4'
SOURCE = '8fb4d1b0b19fea3f15c71247121d2c56bed8ebba'
GGML = 'c36dab89b662838f0f5d4826c399198c0b90bbfc'
CACHE = {'repo': 'cstr/crispasr-ccache', 'file': 'voxcpm2-vulkan/sm75-v3-build.tar', 'revision': 'df9e11b0b66d1c87f1e0be5552d2974590e4ae20', 'sha256': 'a75cd84e3a122271386f68044117167840545696fd83866406b8380e8bac5dc8', 'bytes': 99747840}
WORK = Path('/kaggle/working')
SCRATCH = Path('/kaggle/temp/voxcpm2-current-profile')
REPO = SCRATCH / 'CrispASR'
OUT = WORK / 'voxcpm2-profile'


def main():
    SCRATCH.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    os.environ.update(PYTHONUNBUFFERED='1', TMPDIR=str(SCRATCH),
                      HF_HOME=str(SCRATCH / 'hf'), KAGGLE_KERNEL_REF='voxcpm2-current-profile-v4')
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
    # Vulkan is the measured TTS backend; CUDA runs the ASR quality gate.
    # Every package operation is checked; CPU llvmpipe is not accepted.
    subprocess.run(['apt-get','update','-qq'],check=True,timeout=600)
    subprocess.run(['apt-get','install','-y','-qq','libvulkan1','libvulkan-dev',
                    'vulkan-tools','glslc','glslang-tools','spirv-tools','spirv-headers'],check=True,timeout=900)
    vk = subprocess.run(['vulkaninfo','--summary'],capture_output=True,text=True,timeout=60)
    if not any(word in vk.stdout for word in ('NVIDIA','Tesla','GeForce')):
        driver = subprocess.check_output(['nvidia-smi','--query-gpu=driver_version','--format=csv,noheader'],text=True).splitlines()[0].split('.')[0]
        subprocess.run(['apt-get','install','-y','-qq','libnvidia-gl-'+driver],check=True,timeout=900)
        vk = subprocess.run(['vulkaninfo','--summary'],capture_output=True,text=True,timeout=60)
    (OUT/'vulkaninfo.log').write_text(vk.stdout+'\n'+vk.stderr)
    assert vk.returncode == 0 and any(word in vk.stdout for word in ('NVIDIA','Tesla','GeForce')), 'No actual NVIDIA Vulkan device'
    kh.step('vulkan.device.ready',summary=vk.stdout)
    build = SCRATCH / 'build'
    flags = ['-DCMAKE_BUILD_TYPE=Release', '-DBUILD_SHARED_LIBS=ON', '-DGGML_NATIVE=OFF',
             '-DGGML_BLAS=OFF', '-DCRISPASR_MEL_BLAS=OFF', '-DCRISPASR_BUILD_TESTS=OFF',
             '-DCRISPASR_BUILD_EXAMPLES=ON', '-DCRISPASR_BUILD_SERVER=OFF',
             *kh.cuda_build_flags(arch), '-DGGML_VULKAN=ON', *kh.cache_and_link_flags()]
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
                      VOX_PROFILE_BUILD=str(build), CUDA_VISIBLE_DEVICES='0')
    kh.step('asr.validation.begin', source=SOURCE, arch=arch)
    try:
        with kh.build_heartbeat('asr-cuda-validation'):
            with (OUT/'vulkan-worker.log').open('w') as log:
                result = subprocess.run([sys.executable, REPO / 'tools/ci-heavy/voxcpm2_vulkan_profile.py'],
                                        cwd=REPO, stdout=log, stderr=subprocess.STDOUT, timeout=7200)
        if result.returncode == 0:
            perf = OUT/'per-op'
            perf.mkdir(exist_ok=True)
            with kh.build_heartbeat('vulkan-per-op-capture'):
                with (OUT/'vulkan-per-op.log').open('w') as log:
                    capture = subprocess.run([sys.executable, REPO/'tools/ci-heavy/voxcpm2_vulkan_profile.py', '--perf-only'],
                        cwd=REPO, env=dict(os.environ,HEAVY_OUT=str(perf)),
                        stdout=log,stderr=subprocess.STDOUT,timeout=2400)
            assert capture.returncode == 0, 'Vulkan per-op capture failed'
            per_op = (OUT/'vulkan-per-op.log').read_text()
            assert 'MUL_MAT' in per_op and 'SIN' in per_op, 'Missing LocDiT/VAE per-op evidence'
        kh.step('asr.validation.end', returncode=result.returncode)
    finally:
        kh.export_ccache_tar(OUT / 'ccache.tar')
        kh._push_progress_to_hf(force=True)
    assert result.returncode == 0, 'Vulkan profile/speech acceptance failed'
    log = (OUT/'vulkan-worker.log').read_text()
    assert re.search(r'voxcpm2: backend = Vulkan',log), 'VoxCPM2 did not use Vulkan'
    assert 'falling back to CPU vae_decode' not in log and 'using CPU' not in log, 'VAE CPU fallback invalidates profile'
    assert 'vae.wn_init' in log and 'vae.compute' in log, 'Missing VAE phase evidence'
    steps = re.findall(r'voxcpm2\[bench\]: cfm.steps=(\d+)',log)
    assert steps and set(steps)=={'10'}, 'Native solver did not execute ten steps'
    receipt = json.loads((OUT/'voxcpm2-vulkan-profile.json').read_text())
    assert receipt['passed'] and receipt['steps']==10



if __name__ == '__main__':
    main()
