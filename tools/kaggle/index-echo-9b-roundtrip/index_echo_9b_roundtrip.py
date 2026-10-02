#!/usr/bin/env python3
"""Real CUDA F16/Q8 stage/cache parity and native TTS -> ASR round-trips.

CPU synthesis already ran on GitHub. Consume its pinned WAVs; never use a GPU
session for CPU synthesis or pretend these tests prove the full-file pipeline.
"""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

SCRIPT_VERSION = '2026-10-02.1'
SOURCE_COMMIT = '0026d9b13dde167d5509a15972f5fe259f3bf8f2'
MODEL_REVISION = 'd1cc752e82bb97842052f2a5cd335512da984f96'
REFERENCE_REVISION = 'fac990e86eb1163280e62f66a989da3b6432b4bd'
AUDIO_REVISION = 'd0a7d7a8be318a5841dfdbe6ad37d3acf75523e3'
ROOT = Path('/kaggle/temp/index-echo-roundtrip-repo')
TEMP = Path('/kaggle/temp/index-echo-roundtrip')
OUT = Path('/kaggle/working')
TEMP.mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)
os.environ['TMPDIR'] = str(TEMP)


def run(*command, **kw):
    subprocess.run(list(map(str, command)), check=True, **kw)


hardware = subprocess.check_output(['nvidia-smi', '--query-gpu=name,compute_cap,memory.total',
                                    '--format=csv,noheader'], text=True).strip()
print('actual GPU:', hardware, flush=True)
rows = [line.split(',') for line in hardware.splitlines()]
if sum(int(row[-1].strip().split()[0]) for row in rows) < 24 * 1024:
    raise RuntimeError('Inconclusive: F16 pair requires at least 24 GiB aggregate VRAM; no model pull')
run('git', 'init', ROOT)
run('git', '-C', ROOT, 'remote', 'add', 'origin', 'https://github.com/CrispStrobe/CrispASR.git')
run('git', '-C', ROOT, 'fetch', '--depth=1', 'origin', SOURCE_COMMIT)
run('git', '-C', ROOT, 'checkout', 'FETCH_HEAD')
run('git', '-C', ROOT, 'submodule', 'update', '--init', '--recursive', '--depth=1')
sys.path.insert(0, str(ROOT / 'tools/kaggle'))
import kaggle_harness as kh
kh.init_progress()
kh.provenance(SCRIPT_VERSION, ROOT)
run(sys.executable, '-m', 'pip', 'install', '-q', 'huggingface_hub', 'gguf', 'numpy')
from huggingface_hub import hf_hub_download, snapshot_download
os.environ['HF_TOKEN'] = kh.resolve_hf_token(require=True)
fixture_repo = 'cstr/crispasr-regression-fixtures'
manifest_path = hf_hub_download(fixture_repo, 'index-echo-9b/roundtrip-piper/roundtrip-audio.json',
                                revision=AUDIO_REVISION, local_dir=TEMP / 'fixtures')
manifest = json.loads(Path(manifest_path).read_text())
for item in manifest['cases'].values():
    path = hf_hub_download(fixture_repo, 'index-echo-9b/roundtrip-piper/' + item['audio'],
                           revision=AUDIO_REVISION, local_dir=TEMP / 'fixtures')
    import hashlib
    if hashlib.sha256(Path(path).read_bytes()).hexdigest() != item['sha256']:
        raise RuntimeError('Synthetic fixture checksum mismatch')
    shutil.copy2(path, OUT / item['audio'])
reference = hf_hub_download(fixture_repo, 'index-echo-9b-f32-generation/jfk_11s/ref.gguf',
                            revision=REFERENCE_REVISION, local_dir=TEMP / 'fixtures')
kh.install_build_toolchain()
arch = kh.detect_cuda_arch()
build = TEMP / 'build'
flags = ['-DCMAKE_BUILD_TYPE=Release', '-DBUILD_SHARED_LIBS=ON', '-DCRISPASR_BUILD_TESTS=OFF', '-DCRISPASR_BUILD_SERVER=OFF']
flags += kh.cuda_build_flags(arch) + kh.cache_and_link_flags()
run('cmake', '-S', ROOT, '-B', build, '-G', 'Ninja', *flags)
with kh.build_heartbeat('build.cuda', interval_s=30):
    kh.sh_with_progress(f'cmake --build {build} --target crispasr-cli crispasr-lib crispasr-diff -j{kh.safe_build_jobs(gpu=True)}', cwd=str(ROOT))
sys.path.insert(0, str(ROOT / 'tools/ci-heavy'))
from index_echo_roundtrip import check_roundtrips
library = next(build.rglob('libcrispasr.so'))
receipt = dict(script_version=SCRIPT_VERSION, source_commit=SOURCE_COMMIT,
               model_revision=MODEL_REVISION, reference_revision=REFERENCE_REVISION,
               audio_revision=AUDIO_REVISION, hardware=hardware, cuda_arch=arch,
               full_pipeline_checked=False, cohorts={})
failed = []
for cohort in ['f16', 'q8_0']:
    models = Path(snapshot_download('cstr/index-echo-9b-GGUF', revision=MODEL_REVISION,
                  local_dir=TEMP / 'models', allow_patterns=[f'index-echo-9b-{cohort}.gguf', f'index-echo-9b-decoder-{cohort}.gguf']))
    primary = models / f'index-echo-9b-{cohort}.gguf'
    with (OUT / f'{cohort}-jfk-diff.log').open('w') as log, kh.build_heartbeat(cohort + '.diff', interval_s=30):
        diff = subprocess.run([str(build / 'bin/crispasr-diff'), 'index-echo', str(primary),
                               reference, str(ROOT / 'samples/jfk.wav')],
                              stdout=log, stderr=subprocess.STDOUT, timeout=3600)
    with kh.build_heartbeat(cohort + '.roundtrip', interval_s=30):
        roundtrip_failures = check_roundtrips(ROOT, OUT, build / 'bin/crispasr', library, primary, manifest, use_gpu=True)
    receipt['cohorts'][cohort] = dict(stage_rc=diff.returncode, roundtrip_failures=roundtrip_failures)
    if diff.returncode or roundtrip_failures:
        failed.append(cohort)
    receipt['failed'] = failed
    (OUT / 'cuda-roundtrip.json').write_text(json.dumps(receipt, indent=2) + '\n')
    primary.unlink()
    (models / f'index-echo-9b-decoder-{cohort}.gguf').unlink()
receipt['stage_and_roundtrip_passed'] = not failed
(OUT / 'cuda-roundtrip.json').write_text(json.dumps(receipt, indent=2) + '\n')
if failed:
    raise RuntimeError('CUDA stage/cache or roundtrip gate failed: ' + ', '.join(failed))
