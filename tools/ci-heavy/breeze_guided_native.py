#!/usr/bin/env python3
"""Strict actual native F16 Breeze guidance parity against a verified CPU oracle."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(os.environ['HEAVY_OUT'])
SCRATCH = Path(os.environ['HEAVY_SCRATCH']) / 'breeze-native'


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    import numpy as np
    from huggingface_hub import hf_hub_download
    ap = argparse.ArgumentParser(description=__doc__)
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument('--reference-dir', type=Path)
    group.add_argument('--reference-proof', action='store_true',
                       help='Reuse the pinned, independently verified matched-budget original archive')
    args = ap.parse_args()
    if args.reference_proof:
        archive = Path(hf_hub_download('cstr/crispasr-regression-fixtures',
            'breeze-guided-20261011/matched-budget-failed-parity-proof.zip', repo_type='dataset',
            revision='27896a1fab69d45c8b246eb659823f92a5e0e458'))
        assert digest(archive) == '13d5eed11955e07f2e5fc2f681df123c1b2b35d2f11998416571dbf962d54f33'
        proof = SCRATCH / 'reused-proof'
        proof.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive) as z:
            z.extractall(proof)
        ref = proof / 'reference'
    else:
        ref = args.reference_dir.resolve()
    oracle = json.loads((ref / 'validation.json').read_text())
    assert oracle['complete'] and oracle['max_frames'] == 24
    assert oracle['script_version'] == '2026-10-11.breeze-guided-gh-cpu.2'
    assert oracle['upstream'] == '58ec70ce5fa4cc361bdebf77ec40d1365da00ab2'
    assert oracle['model_revision'] == '3e28c5151381a722f1d8661b4118c298caa77aa4'
    fixture = ref / 'breeze-dumps'
    for name, row in oracle['arrays'].items():
        p = fixture / name
        assert digest(p) == row['sha256'], name
        a = np.load(p, allow_pickle=False)
        assert list(a.shape) == row['shape'] and str(a.dtype) == row['dtype']
        assert a.size and np.isfinite(a).all(), name
    OUT.mkdir(parents=True, exist_ok=True)
    SCRATCH.mkdir(parents=True, exist_ok=True)
    receipt = dict(source=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                   ggml=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT / 'ggml', text=True).strip(),
                   reference_sha256=digest(ref / 'validation.json'), device='CPU; GPU backends compiled OFF',
                   native_precision='F16 backbone and codec', original_arrays_verified=len(oracle['arrays']),
                   native_dump_pass=False, guided_parity_pass=False, complete=False,
                   full_speech_asr_tested=False)
    def run(command, name, env=None):
        with (OUT / (name + '.log')).open('w') as log:
            r = subprocess.run(list(map(str, command)), cwd=ROOT, env=env, stdout=log,
                               stderr=subprocess.STDOUT, timeout=7200)
        print(name, 'rc=', r.returncode, (OUT / (name + '.log')).read_text()[-3000:], flush=True)
        return r.returncode
    try:
        assert run(['sudo', 'bash', 'tools/ci-apt.sh', 'update'], 'apt-update') == 0
        assert run(['sudo', 'bash', 'tools/ci-apt.sh', 'install', '-y', 'cmake', 'ninja-build',
                    'g++', 'libopenblas-dev'], 'apt-install') == 0
        build = SCRATCH / 'build'
        assert run(['cmake', '-S', ROOT, '-B', build, '-G', 'Ninja', '-DCMAKE_BUILD_TYPE=Release',
                    '-DBUILD_SHARED_LIBS=OFF', '-DGGML_NATIVE=OFF', '-DGGML_OPENBLAS=ON',
                    '-DGGML_CUDA=OFF', '-DGGML_VULKAN=OFF', '-DGGML_METAL=OFF',
                    '-DCRISPASR_BUILD_SERVER=OFF', '-DCRISPASR_BUILD_TESTS=OFF'], 'configure') == 0
        assert run(['cmake', '--build', build, '--target', 'crispasr-diff', '-j4'], 'build') == 0
        receipt['native_binary_sha256'] = digest(build / 'bin/crispasr-diff')
        models = []
        for repo, revision, filename in (
            ('cstr/breeze-tts-2-GGUF', '5314fe9ae3ec6eac1a9238c4f6888fdc9fda0e83', 'breeze-tts-2-f16.gguf'),
            ('cstr/qwen3-tts-tokenizer-12hz-GGUF', '344e279551177654762e5c7d39e217834325f222',
             'qwen3-tts-tokenizer-12hz.gguf')):
            p = Path(hf_hub_download(repo, filename, revision=revision))
            models.append(p)
            receipt.setdefault('models', []).append(dict(repo=repo, revision=revision, filename=filename,
                                                          bytes=p.stat().st_size, sha256=digest(p)))
        env = os.environ.copy()
        env.update(BREEZE_CODEC=str(models[1]), BREEZE_FIXTURE_DIR=str(fixture), OPENBLAS_NUM_THREADS='1')
        native = OUT / 'native-dumps'
        native.mkdir()
        base = OUT / 'native-boundaries'
        base.mkdir()
        receipt['base_dump_returncode'] = run(
            [build / 'bin/crispasr-diff', 'bt2-tts', models[0], fixture, base], 'native-boundaries', env)
        # These are oracle-fed boundaries. Full generation/codec output is
        # separately retained and never relabelled as stage acceptance.
        required = {Path(name).stem for name in oracle['arrays']
                    if '/' not in name and (name.startswith(('te_', 'backbone_', 'prompt_', 'dd_logits_'))
                                             or name == 'dd_codes_frame0_stepwise.npy')}
        code = ('import sys; from pathlib import Path; '
                'from tools.reference_backends.breeze_tts_2 import compare; '
                'raise SystemExit(compare(Path(sys.argv[1]), download=False, '
                'only=set(sys.argv[2:]), strict=True))')
        receipt['boundary_comparison_returncode'] = run(
            [sys.executable, '-c', code, base, *sorted(required)], 'boundary-comparison', env)
        receipt['boundary_required_stages'] = sorted(required)
        receipt['boundary_pass'] = (receipt['base_dump_returncode'] == 0 and
                                    receipt['boundary_comparison_returncode'] == 0)
        dump_rc = run([build / 'bin/crispasr-diff', 'bt2-tts-guided', models[0], fixture, native],
                      'native-guided', env)
        # Score all available stages even when the native context controls fail.
        compare_rc = run([sys.executable, 'tools/reference_backends/breeze_tts_2.py',
                          '--guided', '--no-download', '--cpp-dump', native], 'guided-comparison', env)
        receipt['native_dump_returncode'] = dump_rc
        receipt['comparison_returncode'] = compare_rc
        receipt['native_dump_pass'] = dump_rc == 0
        receipt['guided_parity_pass'] = compare_rc == 0
        receipt['arrays'] = {}
        for p in sorted(native.rglob('*.npy')):
            a = np.load(p, allow_pickle=False)
            assert a.size and np.isfinite(a).all(), str(p)
            receipt['arrays'][str(p.relative_to(native))] = dict(
                shape=list(a.shape), dtype=str(a.dtype), sha256=digest(p))
        receipt['boundary_arrays'] = {}
        for p in sorted(base.rglob('*.npy')):
            a = np.load(p, allow_pickle=False)
            assert a.size and np.isfinite(a).all(), str(p)
            receipt['boundary_arrays'][str(p.relative_to(base))] = dict(
                shape=list(a.shape), dtype=str(a.dtype), sha256=digest(p))
        receipt['complete'] = dump_rc == 0 and compare_rc == 0 and receipt['boundary_pass']
        assert receipt['complete'], 'Strict native guided parity/context controls failed; inspect retained logs/arrays'
    except BaseException as exc:
        receipt['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        (OUT / 'validation.json').write_text(json.dumps(receipt, indent=2) + '\n')
        (OUT / 'summary.md').write_text(
            f"Breeze F16 guided parity: complete={receipt['complete']}. Full-speech ASR remains separate.\n")


if __name__ == '__main__':
    main()
