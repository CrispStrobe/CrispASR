#!/usr/bin/env python3
"""Pinned delivered runtime and original capture; actual GPU observation only."""
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tarfile
import zipfile

VERSION = '2026-10-13.dia-cuda-attention.5'
SOURCE = 'PENDING_SOURCE_PIN'
INPUT_REVISION = 'PENDING_INPUT_REVISION'
MANIFEST_SHA256 = 'PENDING_MANIFEST_SHA256'
MANIFEST_PATH = 'dia-20261013-attention/input-manifest.json'
SCRATCH = Path('/tmp/dia-cuda-attention')
OUT = Path('/kaggle/working')
ROOT = SCRATCH / 'CrispASR'


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def safe_path(value):
    path = Path(value)
    if path.is_absolute() or '..' in path.parts or '\\' in value:
        raise ValueError('Unsafe delivered archive path')
    return path


def main():
    assert re.fullmatch('[0-9a-f]{40}', SOURCE) and re.fullmatch('[0-9a-f]{40}', INPUT_REVISION)
    assert re.fullmatch('[0-9a-f]{64}', MANIFEST_SHA256), 'Pin complete audited inputs before submission'
    OUT.mkdir(exist_ok=True); SCRATCH.mkdir(parents=True, exist_ok=True)
    os.environ.update(TMPDIR=str(SCRATCH), HEAVY_OUT=str(OUT), HEAVY_SCRATCH=str(SCRATCH),
                      HF_HOME=str(SCRATCH / 'hf'), HF_XET_CACHE=str(SCRATCH / 'xet'),
                      HF_XET_CHUNK_CACHE_SIZE_BYTES='0', OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='1',
                      MKL_NUM_THREADS='4')
    hardware = subprocess.check_output(['nvidia-smi', '--query-gpu=name,compute_cap,memory.total,driver_version',
                                       '--format=csv,noheader'], text=True).strip()
    preflight = dict(version=VERSION, source=SOURCE, input_revision=INPUT_REVISION,
                     manifest_sha256=MANIFEST_SHA256, hardware=hardware, kernel_sha256=digest(__file__),
                     torch_reinstalled=False, gpu_executed=False, model_acceptance=False)
    shutil.copyfile(__file__, OUT / 'executed-kernel.py')
    (OUT / 'hardware-preflight.json').write_text(json.dumps(preflight, indent=2) + '\n')
    assert hardware and all(int(round(float(row.split(',')[1]) * 10)) in (60, 75)
                            for row in hardware.splitlines()), 'Actual SM60/75 required; no hardware fishing'
    for command in (['git', 'init', ROOT], ['git', '-C', ROOT, 'remote', 'add', 'origin',
                    'https://github.com/CrispStrobe/CrispASR.git'],
                    ['git', '-C', ROOT, 'fetch', '--depth=1', 'origin', SOURCE],
                    ['git', '-C', ROOT, 'checkout', 'FETCH_HEAD']):
        subprocess.run(list(map(str, command)), check=True)
    assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip() == SOURCE
    sys.path.insert(0, str(ROOT / 'tools/kaggle'))
    import kaggle_harness as kh
    kh.init_progress(); kh.provenance(VERSION, ROOT)
    # No native build, compiler-cache warmup or Torch installation on Kaggle.
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'gguf==0.18.0', 'pydantic==2.11.3',
                    'safetensors', 'huggingface_hub'], check=True)
    from huggingface_hub import hf_hub_download
    repo = 'cstr/crispasr-regression-fixtures'

    def download(facts):
        safe_path(facts['file'])
        path = Path(hf_hub_download(repo, facts['file'], revision=INPUT_REVISION, repo_type='dataset', token=False))
        assert path.stat().st_size == facts['bytes'] and digest(path) == facts['sha256'], 'Delivered input changed'
        return path

    manifest_file = Path(hf_hub_download(repo, MANIFEST_PATH, revision=INPUT_REVISION, repo_type='dataset', token=False))
    assert digest(manifest_file) == MANIFEST_SHA256
    manifest = json.loads(manifest_file.read_text())
    shutil.copy2(manifest_file, OUT / 'input-manifest.json')
    capture_archive, package_archive, package_ledger = [download(manifest[k]) for k in
                                                       ('capture', 'package', 'package_ledger')]
    capture_root = SCRATCH / 'capture'; capture_root.mkdir()
    with zipfile.ZipFile(capture_archive) as z:
        assert z.testzip() is None
        seen = set()
        for m in z.infolist():
            safe_path(m.filename)
            assert m.filename not in seen and (m.external_attr >> 16) & 0o170000 != 0o120000
            seen.add(m.filename)
        z.extractall(capture_root)
    capture_root /= safe_path(manifest['capture']['root'])
    bundle_root = SCRATCH / 'runtime'; bundle_root.mkdir()
    with tarfile.open(package_archive) as t:
        seen = set()
        for m in t.getmembers():
            name = safe_path(m.name)
            assert name.parts[0] == 'bundle' and m.name not in seen
            seen.add(m.name)
            assert m.isfile() or m.isdir() or m.issym() or m.islnk()
            if m.issym() or m.islnk():
                link = safe_path(m.linkname)
                target = name.parent / link if m.issym() else link
                assert target.parts[0] == 'bundle'
        t.extractall(bundle_root, filter='data')
    bundle = bundle_root / 'bundle'
    # Small raw ZIP delivery does not guarantee Unix executable mode.
    for name in ('crispasr', 'crispasr-diff', 'crispasr-quantize', 'crispasr-server'):
        (bundle / name).chmod(0o755)
    ledger = capture_root / safe_path(manifest['capture']['ledger'])
    command = [sys.executable, '-u', ROOT / 'tools/ci-heavy/dia_attention_cuda_control.py',
               '--capture', capture_root, '--capture-ledger', ledger,
               '--capture-ledger-sha256', manifest['capture']['ledger_sha256'], '--bundle', bundle,
               '--package-ledger', package_ledger, '--package-ledger-sha256', manifest['package_ledger']['sha256']]
    kh.step('dia.cuda.attention.controls.begin', source=SOURCE, input_revision=INPUT_REVISION)
    try:
        with kh.build_heartbeat('dia.cuda.attention.actual-gpu-controls'):
            subprocess.run(list(map(str, command)), cwd=ROOT, check=True, timeout=14400)
    finally:
        preflight['worker_sha256'] = digest(ROOT / 'tools/ci-heavy/dia_attention_cuda_control.py')
        if (OUT / 'receipt.json').is_file():
            result = json.loads((OUT / 'receipt.json').read_text())
            preflight['gpu_executed'] = any(r['exit'] == 0 and r['gpu_peak_mib_lower_bound'] >= 6000
                                           and r['cuda_scheduler_splits'] > 0 for r in result['runs'])
            preflight['cuda_attention_diagnostic_pass'] = result.get('cuda_late_attention_diagnostic_capture_pass', False)
        (OUT / 'hardware-preflight.json').write_text(json.dumps(preflight, indent=2) + '\n')
        kh.step('dia.cuda.attention.controls.finished', receipt=str(OUT / 'receipt.json'))


if __name__ == '__main__':
    main()
