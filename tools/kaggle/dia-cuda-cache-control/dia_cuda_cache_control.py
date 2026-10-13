#!/usr/bin/env python3
"""Actual native CUDA cache/stage/speech triage, preserving fixed CPU truth."""
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys

VERSION = '2026-10-13.dia-cuda-cache.1'
# Filled from the pushed composed source before kernel submission.
SOURCE = os.environ.get('DIA_CUDA_SOURCE', 'PENDING_SOURCE_PIN')
OUT = Path('/kaggle/working')
SCRATCH = Path('/tmp/dia-cuda-cache')
ROOT = SCRATCH / 'CrispASR'
BUILD = SCRATCH / 'native-build'
SCRATCH.mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)
os.environ.update(TMPDIR=str(SCRATCH), HF_HOME=str(SCRATCH / 'hf'), HF_XET_CACHE=str(SCRATCH / 'xet'),
                  HEAVY_OUT=str(OUT), HEAVY_SCRATCH=str(SCRATCH),
                  OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='4')
assert re.fullmatch('[0-9a-f]{40}', SOURCE), 'Immutable source pin required before kernel launch'
hardware = subprocess.check_output(['nvidia-smi', '--query-gpu=name,compute_cap,memory.total,driver_version',
                                   '--format=csv,noheader'], text=True).strip()
assert hardware, 'Actual CUDA device required'
arch = min(int(float(row.split(',')[1].strip()) * 10 + .1) for row in hardware.splitlines())
nvcc = subprocess.check_output(['/usr/local/cuda/bin/nvcc', '--version'], text=True).strip()
compiler_major = int(re.search(r'release (\d+)\.', nvcc).group(1))
preflight = dict(version=VERSION, source=SOURCE, hardware=hardware, nvcc=nvcc,
                 torch_reinstalled=False, gpu_executed=False, model_acceptance=False)
(OUT / 'hardware-preflight.json').write_text(json.dumps(preflight, indent=2) + '\n')
# CUDA 13 dropped Pascal. Retain an inconclusive receipt and stop before downloads.
if arch < 75 and compiler_major >= 13:
    preflight['inconclusive'] = 'Installed CUDA compiler does not support this physical architecture'
    (OUT / 'hardware-preflight.json').write_text(json.dumps(preflight, indent=2) + '\n')
    raise RuntimeError(preflight['inconclusive'])
assert arch in (60, 75), 'This diagnostic package targets actual SM60/SM75'
for command in (['git', 'init', ROOT], ['git', '-C', ROOT, 'remote', 'add', 'origin', 'https://github.com/CrispStrobe/CrispASR.git'],
                ['git', '-C', ROOT, 'fetch', '--depth=1', 'origin', SOURCE], ['git', '-C', ROOT, 'checkout', 'FETCH_HEAD'],
                ['git', '-C', ROOT, 'submodule', 'update', '--init', '--recursive']):
    subprocess.run(list(map(str, command)), check=True)
assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip() == SOURCE
sys.path.insert(0, str(ROOT / 'tools/kaggle'))
import kaggle_harness as kh
kh.init_progress()
kh.provenance(VERSION, ROOT)
os.environ['HF_TOKEN'] = kh.resolve_hf_token(require=True)
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'gguf==0.18.0', 'pydantic==2.11.3',
                'safetensors', 'huggingface_hub'], check=True)
kh.step('dia.cuda.toolchain', source=SOURCE, hardware=hardware)
kh.install_build_toolchain()
assert kh.detect_cuda_arch() == str(arch), 'Detected GPU architecture changed'
configure = ['cmake', '-G', 'Ninja', '-S', ROOT, '-B', BUILD, '-DCMAKE_BUILD_TYPE=Release',
             '-DBUILD_SHARED_LIBS=ON', '-DGGML_NATIVE=OFF', '-DGGML_BLAS=OFF', '-DGGML_VULKAN=OFF',
             '-DCRISPASR_BUILD_TESTS=OFF', '-DCRISPASR_BUILD_SERVER=OFF', '-DCRISPASR_OPUS=OFF', '-DCRISPASR_AMR=OFF',
             *kh.cuda_build_flags(kh.detect_cuda_arch()), *kh.cache_and_link_flags()]
with kh.build_heartbeat('dia.cuda.configure'):
    kh.sh_with_progress(shlex.join(list(map(str, configure))))
with kh.build_heartbeat('dia.cuda.build'):
    kh.sh_with_progress(shlex.join(['cmake', '--build', str(BUILD), '--target', 'crispasr-cli',
                                    'crispasr-lib', 'crispasr-diff', '-j' + kh.safe_build_jobs(gpu=True)]))
# Small source guards prevent a stale/wrong runtime from consuming model/GPU time.
subprocess.run([sys.executable, ROOT / 'tests/test_dia_cuda_bundle.py', '-v'], check=True)
command = [sys.executable, '-u', ROOT / 'tools/ci-heavy/dia_neural_precision_control.py',
           '--cross-cache-control', '--cuda-control', '--prebuilt', BUILD]
kh.step('dia.cuda.model-controls.begin', source=SOURCE)
try:
    with kh.build_heartbeat('dia.cuda.stage-speech-controls'):
        subprocess.run(list(map(str, command)), cwd=ROOT, check=True, timeout=14400)
finally:
    # Pin both delivery halves and preserve all failed/partial results.
    executed = Path(__file__).resolve()
    (OUT / 'executed-kernel.py').write_bytes(executed.read_bytes())
    preflight['kernel_sha256'] = hashlib.sha256(executed.read_bytes()).hexdigest()
    preflight['worker_sha256'] = hashlib.sha256((ROOT / 'tools/ci-heavy/dia_neural_precision_control.py').read_bytes()).hexdigest()
    if (OUT / 'receipt.json').is_file():
        controls = json.loads((OUT / 'receipt.json').read_text())
        preflight['gpu_executed'] = any(row.get('cuda_scheduler_splits', 0) > 0
                                       and row.get('gpu_peak_mib_lower_bound', 0) >= 6000
                                       and row['exit'] == 0 for row in controls['runs'])
        preflight['fixed_prompt_control_pass'] = controls.get('precision_controls_pass', False)
    (OUT / 'hardware-preflight.json').write_text(json.dumps(preflight, indent=2) + '\n')
    kh.step('dia.cuda.model-controls.finished', receipt=str(OUT / 'receipt.json'))
print('DIA_FIXED_PROMPT_CUDA_CACHE_CONTROL_PASS; broader CPU/GPU/Q4 acceptance separate', flush=True)
