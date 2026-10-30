#!/usr/bin/env python3
"""Inference-only Nemotron replay; pin reviewed CPU evidence before dispatch."""
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tarfile
import urllib.request

SOURCE = 'cc191e718595aa4be90d62fcd66b66360d44e426'
CPU_RUN = 40377479500
CPU_SOURCE = '6143b9aa2ff0457092a2f35b0dcb2a4aa42a5ff8'
CPU_RECEIPT = 'docs/historical/nemotron-default-replay-2026-10-26-cpu.json'
CPU_SHA256 = 'f4e078e83504da76b7a0b2bf85847d21b14c9232ab8bacd112b52b3e81bc62fb'
ASSET = 'crispasr-linux-x86_64-cuda.tar.gz'
ASSET_SHA256 = '90e4fcf737b714dfbb899f426cdd20781600dc8e993c05f1b068d6cdcf4cc1e9'
VERSION = 'nemotron-default-replay-v2'


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''):
            value.update(block)
    return value.hexdigest()


def main():
    if not (re.fullmatch('[0-9a-f]{40}', SOURCE) and re.fullmatch('[0-9a-f]{64}', CPU_SHA256)):
        raise RuntimeError('Reviewed complete CPU receipt and immutable worker source required before dispatch')
    overrides = sorted(key for key in os.environ if key.startswith(('CRISPASR_', 'GGML_')))
    assert not overrides, 'Inherited runtime overrides invalidate the controlled replay: ' + ', '.join(overrides)
    out = Path('/kaggle/working/nemotron')
    scratch = Path('/kaggle/temp/nemotron-default-replay')
    out.mkdir(parents=True, exist_ok=True)
    scratch.mkdir(parents=True, exist_ok=True)
    os.environ.update(TMPDIR=str(scratch), HF_HOME=str(scratch / 'hf'),
                      HF_XET_CACHE=str(scratch / 'xet'), CUDA_VISIBLE_DEVICES='0',
                      HEAVY_OUT=str(out), HEAVY_SCRATCH=str(scratch),
                      OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='4')
    subprocess.run(['nvidia-smi', '--id=0'], check=True)
    repo = scratch / 'CrispASR'
    for command in (['git', 'init', repo], ['git', '-C', repo, 'remote', 'add', 'origin',
                     'https://github.com/CrispStrobe/CrispASR.git'],
                    ['git', '-C', repo, 'fetch', '--depth=1', 'origin', SOURCE],
                    ['git', '-C', repo, 'checkout', 'FETCH_HEAD']):
        subprocess.run(list(map(str, command)), check=True)
    assert subprocess.check_output(['git', '-C', repo, 'rev-parse', 'HEAD'], text=True).strip() == SOURCE
    sys.path.insert(0, str(repo / 'tools/kaggle'))
    import kaggle_harness as kh
    kh.init_progress()
    kh.provenance(VERSION, repo)
    os.environ['HF_TOKEN'] = kh.resolve_hf_token(require=True)
    cpu_path = repo / CPU_RECEIPT
    assert sha(cpu_path) == CPU_SHA256
    cpu = json.loads(cpu_path.read_text())
    assert cpu['source'] == CPU_SOURCE and cpu['device'] == 'cpu' and cpu['complete']
    assert cpu['profiles']['f16']['accepted'], 'Original/F16 CPU control must pass first'
    assert cpu['profiles']['registry-q4']['complete'], 'Require complete default diagnosis'
    (out / 'cpu-input.json').write_text(json.dumps(dict(run=CPU_RUN, source=CPU_SOURCE,
        sha256=CPU_SHA256, receipt=cpu), indent=2) + '\n')
    inventory = json.loads((repo / 'docs/release-publication-2026-10-26/native-assets.json').read_text())[ASSET]
    assert inventory['sha256'] == ASSET_SHA256
    archive_path = scratch / ASSET
    urllib.request.urlretrieve(inventory['url'], archive_path)
    assert archive_path.stat().st_size == inventory['bytes'] and sha(archive_path) == ASSET_SHA256
    with tarfile.open(archive_path) as archive:
        files = {m.name: m for m in archive.getmembers() if not m.isdir()}
        assert set(files) == set(inventory['members']), 'Release archive inventory changed'
        assert all(m.isfile() and not Path(m.name).is_absolute() and '..' not in Path(m.name).parts
                   for m in files.values()), 'Unexpected archive member'
        assert sum(m.size for m in files.values()) < 1024**3
        archive.extractall(scratch, filter='data')
    for name, expected in inventory['members'].items():
        path = scratch / name
        assert path.stat().st_size == expected['bytes'] and sha(path) == expected['sha256'], name
    bundle = scratch / ASSET.removesuffix('.tar.gz')
    sys.path.insert(0, str(repo / 'tools/ci-heavy'))
    from granite5_cuda_preflight import run_preflight
    preflight = run_preflight(bundle, out / 'cuda-loader.json')
    assert preflight['runtime_preflight_pass']
    assert int(preflight['gpu0']['memory_free_mib']) >= 3072, 'Insufficient GPU memory before inference'
    os.environ['LD_LIBRARY_PATH'] = preflight['ld_library_path']
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'numpy<3',
                    'gguf==0.17.1', 'huggingface_hub>=0.36,<2'], check=True)
    (out / 'entry-provenance.json').write_text(json.dumps(dict(version=VERSION, source=SOURCE,
        entry_sha256=sha(__file__), cpu_run=CPU_RUN, cpu_sha256=CPU_SHA256,
        runtime_tag='v0.8.45', runtime_source='e58bb48b52cf1fe957c19586297c83a2ed0d9fe4',
        archive_sha256=ASSET_SHA256, members=inventory['members']), indent=2) + '\n')
    command = [sys.executable, '-B', repo / 'tools/ci-heavy/nemotron_default_replay.py',
               '--device', 'cuda', '--library', bundle / 'libcrispasr.so.1', '--cli', bundle / 'crispasr']
    with kh.build_heartbeat('nemotron.cuda.inference'):
        result = subprocess.run(list(map(str, command)), cwd=repo)
    receipt = json.loads((out / 'replay.json').read_text())
    assert result.returncode == 0 and receipt['complete'], 'Preserve incomplete replay before any rerun'
    assert receipt['profiles']['f16']['accepted']
    for profile in receipt['profiles'].values():
        assert profile['cuda_backend_selection_verified'] and profile['actual_cuda']['mapped_backend']
        assert profile['decode_gpu_activity']['verified']
    kh.step('nemotron.cuda.complete', current_default_accepted=receipt['current_default_accepted'])
    print('NEMOTRON_CUDA_DIAGNOSIS_COMPLETE', receipt['current_default_accepted'], flush=True)


if __name__ == '__main__':
    main()
