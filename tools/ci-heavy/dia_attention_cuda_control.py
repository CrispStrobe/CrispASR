#!/usr/bin/env python3
"""Actual CUDA late-history observation against an audited original CPU dump.

No reference inference or native compilation runs here. Successful observation
does not accept free-running speech, quantization, performance or defaults.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

CAPTURE_SOURCE = '42b111c5e6006e7dcc82afff2b1b43da7c15861c'
PACKAGE_SOURCE = '2328f15ce6f73c05c8aab44fa629a49e2d42c44d'
GGML_SOURCE = 'c36dab89b662838f0f5d4826c399198c0b90bbfc'
ROOT = Path(__file__).resolve().parents[2]
RUNTIME_FILES = {'src/dia_tts.cpp', 'src/dia_sampling.h', 'src/dia_exponential_rng.h',
                 'examples/cli/crispasr_dia_diff.h', 'src/crispasr_c_api.cpp', 'python/crispasr/_binding.py'}


def runtime_environment(bundle, parent=None):
    """Keep host driver lookup while preferring the audited user-space bundle."""
    env = dict(os.environ if parent is None else parent)
    inherited = env.get('LD_LIBRARY_PATH', '')
    env['LD_LIBRARY_PATH'] = os.pathsep.join([str(bundle)] + ([inherited] if inherited else []))
    env['GGML_BACKEND_PATH'] = str(bundle / 'libggml-cuda.so')
    return env


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_checkpoint(path, report):
    # Serialize before touching the previous checkpoint, then replace atomically.
    payload = json.dumps(report, indent=2, allow_nan=False) + '\n'
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(payload)
    temporary.replace(path)


def attention_row(layer, part, tensor, file):
    import numpy as np
    # GGUF dimensions are NumPy uint64 on some reader versions. Keep shapes
    # explicitly portable in JSON without coercing unrelated report values.
    shape = tuple(int(n) for n in reversed(tensor.shape))
    expected = np.asarray(tensor.data).reshape(shape)
    assert file.stat().st_size == expected.size * 4
    actual = np.fromfile(file, dtype='<f4').reshape(shape)
    assert np.isfinite(expected).all() and np.isfinite(actual).all()
    x, y = expected.astype('f8').reshape(-1, 128), actual.astype('f8').reshape(-1, 128)
    xn, yn = np.linalg.norm(x, axis=1), np.linalg.norm(y, axis=1)
    assert (xn > 0).all() and (yn > 0).all()
    return dict(layer=layer, part=part, original_shape=list(shape), values=int(expected.size),
                cosine_min=float(np.min(np.sum(x*y, axis=1)/(xn*yn))),
                max_row_norm_error=float(np.max(abs(yn/xn-1))), max_abs=float(abs(x-y).max()),
                relative_l2=float(np.linalg.norm(x-y)/np.linalg.norm(x)),
                changed_values=int(np.count_nonzero(expected != actual)), file_sha256=digest(file))


def checked_file(root, name, facts):
    relative = Path(name)
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError('Unsafe input ledger path')
    path = root / relative
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError('Input payload escapes its audited root')
    if path.stat().st_size != facts['bytes'] or digest(path) != facts['sha256']:
        raise ValueError('Input payload differs: ' + name)
    return path


def capture_inputs(root, ledger_path, expected_sha):
    if digest(ledger_path) != expected_sha:
        raise ValueError('Unpinned capture verification ledger')
    ledger = json.loads(ledger_path.read_text())
    if ledger['source'] != CAPTURE_SOURCE:
        raise ValueError('Unexpected original capture source')
    names = ('receipt.json', 'canonical-ref.gguf', 'original-step978-ref.gguf',
             'scaled-step978-ref.gguf', 'negative-step-ref.gguf', 'out-of-history-step-ref.gguf')
    rows = {r['path']: r for r in ledger['files']}
    if len(rows) != len(ledger['files']):
        raise ValueError('Duplicate capture ledger path')
    files = {name: checked_file(root, name, rows[name]) for name in names}
    receipt = json.loads(files['receipt.json'].read_text())
    if receipt['source'] != CAPTURE_SOURCE or receipt['worktree_dirty'] or receipt['step'] != 978:
        raise ValueError('Unexpected capture receipt')
    for name in ('original_capture_off_on_off_all_common_arrays_exact',
                 'native_capture_off_on_off_complete_arrays_exact', 'late_attention_diagnostic_capture_pass'):
        if receipt.get(name) is not True:
            raise ValueError('Capture transparency not proved: ' + name)
    if digest(files['canonical-ref.gguf']) != '3db6f8923abffd612e6c83d7036debebd10ed28b77c3eb9ec3d91fec1ce1a542':
        raise ValueError('Canonical original reference changed')
    return files, receipt


def package_inputs(bundle, receipt_path, expected_sha, root=ROOT):
    if digest(receipt_path) != expected_sha:
        raise ValueError('Unpinned CUDA package verification ledger')
    verification = json.loads(receipt_path.read_text())
    receipt = verification['receipt']
    if receipt['source'] != PACKAGE_SOURCE or receipt['ggml'] != GGML_SOURCE:
        raise ValueError('Unexpected CUDA package source')
    if (receipt['attention_capture_source'] != CAPTURE_SOURCE or not receipt['build_complete']
            or receipt['tracked_worktree_dirty'] or receipt['architectures'] != [60, 75]
            or receipt['cuda'] != '12.4.1' or receipt['gpu_executed'] or receipt['model_acceptance']):
        raise ValueError('Unexpected CUDA package provenance')
    embedded = json.loads((bundle / 'provenance.json').read_text())
    if embedded != {k: v for k, v in receipt.items() if k != 'archive'}:
        raise ValueError('Embedded package provenance changed')
    actual = {p.relative_to(bundle).as_posix() for p in bundle.rglob('*') if p.is_file()}
    if actual != set(receipt['files']) | {'provenance.json'}:
        raise ValueError('Missing or unlisted package payload')
    for name, facts in receipt['files'].items():
        if Path(name).name.startswith('libcuda.so'):
            raise ValueError('Driver/stub must not be packaged')
        checked_file(bundle, name, facts)
    runtime = receipt['runtime_source_sha256']
    if set(runtime) != RUNTIME_FILES or any(digest(root / name) != expected for name, expected in runtime.items()):
        raise ValueError('Worker runtime differs from compiled capture runtime')
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture', type=Path, required=True)
    parser.add_argument('--capture-ledger', type=Path, required=True)
    parser.add_argument('--capture-ledger-sha256', required=True)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--package-ledger', type=Path, required=True)
    parser.add_argument('--package-ledger-sha256', required=True)
    a = parser.parse_args()
    for value in (a.capture_ledger_sha256, a.package_ledger_sha256):
        if not re.fullmatch('[0-9a-f]{64}', value):
            parser.error('Verification ledgers require immutable SHA256 pins')
    files, cpu_receipt = capture_inputs(a.capture, a.capture_ledger, a.capture_ledger_sha256)
    package_inputs(a.bundle, a.package_ledger, a.package_ledger_sha256)
    out = Path(os.environ['HEAVY_OUT']); out.mkdir(parents=True, exist_ok=True)
    scratch = Path(os.environ['HEAVY_SCRATCH']); scratch.mkdir(parents=True, exist_ok=True)
    hardware = subprocess.check_output(['nvidia-smi', '--query-gpu=name,compute_cap,memory.total,driver_version',
                                       '--format=csv,noheader'], text=True).strip()
    if not hardware or any(int(round(float(row.split(',')[1]) * 10)) not in (60, 75)
                           for row in hardware.splitlines()):
        raise ValueError('This package requires an actual SM60/75 device')
    report = dict(source=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                  tracked_worktree_dirty=bool(subprocess.check_output(
                      ['git', 'status', '--porcelain', '--untracked-files=no'], cwd=ROOT)),
                  worker_sha256=digest(__file__), hardware=hardware, capture_source=CAPTURE_SOURCE,
                  capture_ledger_sha256=a.capture_ledger_sha256, package_source=PACKAGE_SOURCE,
                  package_ledger_sha256=a.package_ledger_sha256, runs=[], step=978,
                  production_defaults_changed=False, speech_acceptance=False, gpu_acceptance=False,
                  q4_acceptance=False, torch_reinstalled=False,
                  scope='Actual CUDA observation/replay diagnostic; native free speech remains independently unaccepted')
    if report['tracked_worktree_dirty']:
        raise ValueError('Worker source must be clean')
    for name in ('src/dia_tts.cpp', 'examples/cli/crispasr_dia_diff.h',
                 'tools/ci-heavy/dia_attention_cuda_control.py'):
        path = out / 'sources' / name; path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, path)

    def checkpoint():
        write_checkpoint(out / 'receipt.json', report)

    def run(command, label, env=None, negative=False, cuda=False):
        log = out / (label + '.log')
        peak = 0
        with log.open('w') as f:
            child = subprocess.Popen(list(map(str, command)), cwd=ROOT, env=env,
                                     stdout=f, stderr=subprocess.STDOUT)
            deadline = time.monotonic() + 7200
            try:
                while child.poll() is None:
                    if cuda:
                        usage = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,used_gpu_memory',
                                                         '--format=csv,noheader,nounits'], text=True, timeout=30)
                        peak = max([peak] + [int(row.split(',')[1]) for row in usage.splitlines()
                                             if row.strip() and int(row.split(',')[0]) == child.pid])
                    if time.monotonic() > deadline:
                        raise TimeoutError(label)
                    time.sleep(.5)
            finally:
                if child.poll() is None:
                    child.kill()
                child.wait()
                record = dict(name=label, exit=child.returncode, log_sha256=digest(log),
                              gpu_peak_mib_lower_bound=peak,
                              cuda_scheduler_splits=len(re.findall(r'SPLIT #\d+: CUDA', log.read_text())))
                report['runs'].append(record); checkpoint()
        print(label, child.returncode, log.read_text()[-1200:], flush=True)
        if (child.returncode != 0) != negative:
            raise AssertionError(label)
        if cuda and (peak < 6000 or record['cuda_scheduler_splits'] <= 0):
            raise AssertionError('Native full-F32 CUDA allocation/compute not proved')
        return record

    checkpoint()
    # CUDA drivers are host-provided, deliberately outside the delivered
    # toolkit bundle. Reject missing/unusable drivers before downloading weights.
    driver_check = '''import ctypes, sys
driver = ctypes.CDLL('libcuda.so.1')
driver.cuInit.argtypes = [ctypes.c_uint]
driver.cuInit.restype = ctypes.c_int
assert driver.cuInit(0) == 0, 'CUDA driver initialization failed'
count = ctypes.c_int()
driver.cuDeviceGetCount.argtypes = [ctypes.POINTER(ctypes.c_int)]
driver.cuDeviceGetCount.restype = ctypes.c_int
assert driver.cuDeviceGetCount(ctypes.byref(count)) == 0 and count.value > 0
ctypes.CDLL(sys.argv[1])
print('ACTUAL_CUDA_DRIVER_AND_PACKAGE_PLUGIN_LOAD_PASS', count.value)
'''
    run([sys.executable, '-c', driver_check, a.bundle / 'libggml-cuda.so'],
        'cuda-driver-plugin-preflight', runtime_environment(a.bundle))
    report['cuda_driver_plugin_preflight_pass'] = True
    checkpoint()
    sys.path.insert(0, str(ROOT / 'tools'))
    import numpy as np
    from gguf import GGUFReader
    from huggingface_hub import hf_hub_download, snapshot_download
    from reference_backends import dia_tts_reference as reference
    from dia_model_precision import WEIGHTS_SHA
    reader = GGUFReader(str(files['original-step978-ref.gguf']))
    tensors = {t.name: t for t in reader.tensors}
    canonical = {t.name: t for t in GGUFReader(str(files['canonical-ref.gguf'])).tensors}
    for name in ('decoder_input_tokens', 'decoder_codes_rev'):
        np.testing.assert_array_equal(tensors[name].data, canonical[name].data)
    pcm = tensors['decoded_audio'].data.reshape(-1).copy()
    assert pcm.dtype == np.float32 and np.isfinite(pcm).all()
    np.save(out / 'original-late.npy', pcm)
    (out / 'original-late.json').write_text(json.dumps(dict(phrase='We are committed to making it secure.',
        seed=42, capacity=3072, audio_seconds=len(pcm)/44100,
        rms=float(np.sqrt(np.mean(pcm.astype('f8')**2))), finite=True)) + '\n')
    source = scratch / 'official-dia'
    run(['git', 'clone', '--filter=blob:none', 'https://github.com/nari-labs/dia', source], 'original-source-clone')
    run(['git', '-C', source, 'checkout', '--detach', reference.SOURCE_PIN], 'original-source-pin')
    reference.verify_source(source)
    os.environ.update(DIA_SOURCE=str(source), DIA_DEVICE='cpu')
    location = Path(snapshot_download('nari-labs/Dia-1.6B', revision=reference.MODEL_PIN,
                                     allow_patterns=['config.json', 'model.safetensors']))
    weights = location / 'model.safetensors'; assert digest(weights) == WEIGHTS_SHA
    assert digest(location / 'config.json') == '9140e85fd15b82d7f268c5681a41fb1de68573ec206b87cb3a5862fe5c598d6e'
    repo, revision = 'cstr/dia-1.6b-GGUF', '3233fbcb32be47761d2e736857b6d1a075b9ba7e'
    template = hf_hub_download(repo, 'dia-1.6b-f16.gguf', revision=revision)
    model_dir = scratch / 'native-model'; model_dir.mkdir()
    model, codec = model_dir / 'dia-1.6b-f32.gguf', model_dir / 'dac-44khz.gguf'
    run([sys.executable, ROOT / 'tools/ci-heavy/dia_neural_precision_control.py', '--export', 'f32',
         '--weights', weights, '--template', template, '--output', model], 'neural-export')
    assert digest(model) == cpu_receipt['model_sha256'] == '8128132c538e1aef83f2fb5257b4f71d28a67c2a5a0d54d0a1f0a3b0f5d6b73d'
    # Reuse the complete audited original DAC export; no DAC/Torch install or
    # original CPU generation belongs in this actual-GPU session.
    codec_dir = scratch / 'verified-codec-proof'
    run([sys.executable, ROOT / 'tools/dia_proof_archives.py', 'dac-converter', codec_dir], 'verified-codec-download')
    codec_proof = next(codec_dir.glob('heavy-*'))
    shutil.copy2(codec_proof / 'dac-output-f32.gguf', codec)
    shutil.copy2(codec_proof / 'physical-tensor-audit.json', out / 'physical-codec-audit.json')
    assert digest(codec) == cpu_receipt['codec_sha256'] == '6db51da2333868918b6d476008d92036a53400bc816ff7e1d2b1e973499e4135'
    report.update(model_sha256=digest(model), codec_sha256=digest(codec), original_weights_sha256=WEIGHTS_SHA)
    lib, diff = a.bundle / 'libcrispasr.so', a.bundle / 'crispasr-diff'
    env = dict(runtime_environment(a.bundle),
               GGML_SCHED_DEBUG='1', CRISPASR_DIA_TTS_GPU='1', CRISPASR_DIA_CROSS_CACHE_F32='1',
               CRISPASR_DIA_PORTABLE_RNG='1', CRISPASR_DIA_SOURCE_ROPE='0', DIA_CODEC=str(codec))
    asr = hf_hub_download('cstr/nemotron-3.5-asr-streaming-GGUF', 'nemotron-3.5-asr-streaming-0.6b-q4_k.gguf',
                         revision='bbd95a9ca5fa0dfca3312a122dfc45a2b578b9c2')
    positive = run([sys.executable, ROOT / 'tools/ci-heavy/dia_full_speech.py', '--child', 'recognize',
                   '--model', asr, '--lib', lib, '--phrase', 'reported', '--audio-tag', 'original-late'],
                  'original-gpu-asr', env)
    assert positive['cuda_scheduler_splits'] > 0, 'Original-audio ASR did not execute on CUDA'
    report['original_gpu_asr_positive'] = json.loads((out / 'original-late.json').read_text())
    for label, enabled in [('attention-off', False), ('attention-on', True), ('attention-repeat-off', False)]:
        arm_env = dict(env, CRISPASR_DIA_DIFF_DUMP_DIR=str(out / label),
                       CRISPASR_DIA_DUMP_STEP='malformed-inherited-control',
                       CRISPASR_DIA_DUMP_SELF_ATTN='1' if enabled else '0')
        run([diff, 'dia', model, files['original-step978-ref.gguf']], label + '-diff', arm_env, cuda=True)
        log = (out / (label + '-diff.log')).read_text().replace('DIA_DECODE_CODES: 1210 frames\n', '')
        assert len(re.findall(r'\[PASS\]', log)) == 40 and '[FAIL]' not in log
        assert (out / label / 'capture-step.txt').read_text().strip() == '978'
        assert len(list((out / label).glob('cpp_decoder_sa*.f32'))) == (72 if enabled else 0)
    for name in ('logits.f32', 'feedback.i32', 'decoded_audio.f32'):
        assert len({digest(out / label / name) for label in ('attention-off', 'attention-on', 'attention-repeat-off')}) == 1, name
    report['native_capture_off_on_off_complete_arrays_exact'] = True; checkpoint()
    negative_env = dict(env, CRISPASR_DIA_DUMP_SELF_ATTN='0')
    for label in ('scaled-step978', 'negative-step', 'out-of-history-step'):
        run([diff, 'dia', model, files[label + '-ref.gguf']], label + '-negative', negative_env, negative=True)
    rows = []
    for layer in range(18):
        for part in ('q', 'k', 'v', 'attention'):
            name = f'decoder_sa{layer}_{part}'
            file = out / 'attention-on' / ('cpp_' + name + '.f32')
            rows.append(attention_row(layer, part, tensors[name], file))
            report['attention_history_rows'] = rows; checkpoint()
    report['cuda_late_attention_diagnostic_capture_pass'] = True; checkpoint()
    print('DIA_CUDA_LATE_ATTENTION_DIAGNOSTIC_PASS; native free speech is not accepted', flush=True)


if __name__ == '__main__':
    main()
