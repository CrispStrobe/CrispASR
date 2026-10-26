#!/usr/bin/env python3
"""Replay the registry's real Nemotron default against immutable original controls.

Diagnostic only: no model publication or acceptance-gate/default changes.
CPU builds run on Actions; --library/--cli permits a prebuilt actual-CUDA replay.
"""
import argparse
from contextlib import contextmanager
import ctypes as ct
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[2]
PROOF_REV = '10e579dedf52a16a159cbc5d35e17f8bffa77190'
PROOF_SHA = '7762f2a73d8ae7dad05d667fbf0bca393a0b3b7ad622b1b1f09473fa8f5d0463'
FIXTURES = 'cstr/crispasr-regression-fixtures'
MODELS = {
    'f16': ('cstr/nemotron-3.5-asr-streaming-GGUF', 'bbd95a9ca5fa0dfca3312a122dfc45a2b578b9c2',
            'nemotron-3.5-asr-streaming-0.6b-f16.gguf',
            '14cbc98df495d8ab452c6a4ac367136d94ab5da55e73afaa96ba4f749dd95d09', 1286368480),
    'registry-q4': ('cstr/nemotron-3.5-asr-streaming-0.6b-GGUF', 'f6f18e48c30035807b6ca0cdb965654fa02dbf69',
                    'nemotron-3.5-asr-streaming-0.6b-q4_k.gguf',
                    '22480dd283a3bf1b39e9d2ab86c048b0c1654b2fc280e46627a4cbfded4b46fd', 408557984),
}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''):
            h.update(block)
    return h.hexdigest()


def words(text):
    return re.findall('[a-z]+', re.sub(r'<[^>]*>', '', text).lower())


def runtime_default(library):
    """Ask the tested C ABI, rather than assuming the historical Q4 is default."""
    lib = ct.CDLL(str(library))
    lib.crispasr_registry_lookup_abi.argtypes = [ct.c_char_p, ct.c_char_p, ct.c_int32,
                                               ct.c_char_p, ct.c_int32, ct.c_char_p, ct.c_int32]
    lib.crispasr_registry_lookup_abi.restype = ct.c_int
    filename, url, size = ct.create_string_buffer(256), ct.create_string_buffer(1024), ct.create_string_buffer(64)
    assert lib.crispasr_registry_lookup_abi(b'nemotron', filename, len(filename), url, len(url), size, len(size)) == 0
    row = dict(file=filename.value.decode(), url=url.value.decode(), size=size.value.decode())
    row['profile'] = next(name for name, model in MODELS.items() if model[2] == row['file'])
    return row


def active_gpu_samples(text, pid):
    """NVIDIA pmon -s u: gpu, pid, type, SM%, memory%, ... ."""
    rows = [line.split() for line in text.splitlines() if not line.lstrip().startswith('#')]
    return [row for row in rows if len(row) >= 5 and row[0] == '0'
            and row[1] == str(pid) and row[3].isdigit() and int(row[3]) > 0]


@contextmanager
def gpu_decode_activity(out, result, save):
    """Capture process-specific SM activity during the reused decode sequence.

    The model is already open; CUDA context/weight initialization occurs before
    this interval. No claim is made that every graph operator executes on CUDA.
    See https://docs.nvidia.com/deploy/nvidia-smi/index.html#process-monitoring
    """
    path = out / 'decode-pmon.txt'
    with path.open('w') as stream:
        monitor = subprocess.Popen(['nvidia-smi', 'pmon', '-i', '0', '-s', 'u', '-d', '1'],
                                   stdout=stream, stderr=subprocess.STDOUT)
        try:
            yield
        finally:
            monitor.terminate()
            try:
                monitor.wait(timeout=10)
            except subprocess.TimeoutExpired:
                monitor.kill()
                monitor.wait()
    samples = active_gpu_samples(path.read_text(), os.getpid())
    result['decode_gpu_activity'] = dict(process_id=os.getpid(), active_samples=len(samples),
                                        evidence=path.name, verified=bool(samples))
    save()
    assert samples, 'No own-process SM activity established during decoding'


def resources():
    available = next(int(x.split()[1]) for x in Path('/proc/meminfo').read_text().splitlines()
                     if x.startswith('MemAvailable:'))
    row = dict(load=list(os.getloadavg()), available_KiB=available)
    print('RESOURCES', row, flush=True)
    assert row['load'][0] < 2 * (os.cpu_count() or 1), 'Host overloaded'
    assert available >= 8 * 1024**2, 'Need 8 GiB available before build/inference'
    return row


def run(command, label, out, timeout=3600):
    with (out / (label + '.log')).open('w') as log:
        result = subprocess.run(list(map(str, command)), cwd=ROOT, stdout=log,
                                stderr=subprocess.STDOUT, timeout=timeout)
    return result.returncode


def metric(actual, reference):
    import numpy as np
    assert actual.shape == reference.shape
    a, b = actual.astype('float64'), reference.astype('float64')
    assert np.isfinite(a).all() and np.isfinite(b).all()
    a, b = a.reshape(-1, a.shape[-1]), b.reshape(-1, b.shape[-1])
    an, bn = np.linalg.norm(a, axis=1), np.linalg.norm(b, axis=1)
    denominator = an * bn
    cos = np.divide((a * b).sum(axis=1), denominator, out=np.zeros_like(an), where=denominator > 1e-20)
    cos[(an <= 1e-20) & (bn <= 1e-20)] = 1
    relative = float(np.linalg.norm(a - b) / max(float(np.linalg.norm(b)), 1e-20))
    return dict(cosine_min=float(cos.min()), native_norm=float(np.linalg.norm(a)),
                reference_norm=float(np.linalg.norm(b)), relative_l2=relative,
                passed=bool(cos.min() >= .999 and relative <= .01), shape=list(actual.shape))


def stages(library, model, ref_file, out, gpu):
    """Use existing native diagnostic entry points; retain complete native arrays."""
    import numpy as np
    from gguf import GGUFReader
    import wave
    with wave.open(str(ROOT / 'samples/jfk.wav')) as wav:
        assert wav.getframerate() == 16000 and wav.getnchannels() == 1 and wav.getsampwidth() == 2
        pcm = np.frombuffer(wav.readframes(wav.getnframes()), dtype='<i2').astype('float32') / 32768
    references = {t.name: np.array(t.data, dtype='float32', copy=True) for t in GGUFReader(str(ref_file)).tensors}
    lib = ct.CDLL(str(library))
    ptr = ct.POINTER(ct.c_float)
    lib.crispasr_nemotron_init.argtypes = [ct.c_char_p, ct.c_int, ct.c_int]
    lib.crispasr_nemotron_init.restype = ct.c_void_p
    lib.crispasr_nemotron_free.argtypes = [ct.c_void_p]
    lib.crispasr_nemotron_transcribe.argtypes = [ct.c_void_p, ptr, ct.c_int, ct.c_int64]
    lib.crispasr_nemotron_transcribe.restype = ct.c_void_p
    lib.nemotron_result_free.argtypes = [ct.c_void_p]
    lib.nemotron_compute_mel.argtypes = [ct.c_void_p, ptr, ct.c_int, ct.POINTER(ct.c_int), ct.POINTER(ct.c_int)]
    lib.nemotron_compute_mel.restype = ptr
    for name in ('nemotron_run_preencode_ext', 'nemotron_run_encoder_ext'):
        fn = getattr(lib, name)
        fn.argtypes = [ct.c_void_p, ptr, ct.c_int, ct.c_int, ct.POINTER(ct.c_int), ct.POINTER(ct.c_int)]
        fn.restype = ptr
    libc = ct.CDLL(None)
    libc.free.argtypes = [ct.c_void_p]
    ctx = lib.crispasr_nemotron_init(str(model).encode(), 4, int(gpu))
    assert ctx
    result = lib.crispasr_nemotron_transcribe(ctx, pcm.ctypes.data_as(ptr), len(pcm), 0)
    assert result
    mel = None
    try:
        nm, tm = ct.c_int(), ct.c_int()
        mel = lib.nemotron_compute_mel(ctx, pcm.ctypes.data_as(ptr), len(pcm), ct.byref(nm), ct.byref(tm))
        assert mel
        arrays = {'mel_spectrogram': np.ctypeslib.as_array(mel, shape=(tm.value * nm.value,)).reshape(tm.value, nm.value).T.copy()}
        for name, function in [('pre_encode_output', 'nemotron_run_preencode_ext'),
                               ('encoder_output', 'nemotron_run_encoder_ext')]:
            rows, width = ct.c_int(), ct.c_int()
            value = getattr(lib, function)(ctx, mel, nm, tm, ct.byref(rows), ct.byref(width))
            assert value, name
            arrays[name] = np.ctypeslib.as_array(value, shape=(rows.value * width.value,)).reshape(rows.value, width.value).copy()
            libc.free(value)
        reports = {}
        for name, array in arrays.items():
            np.save(out / (name + '.npy'), array, allow_pickle=False)
            reports[name] = metric(array, references[name])
        return reports
    finally:
        if mel:
            libc.free(mel)
        lib.nemotron_result_free(result)
        lib.crispasr_nemotron_free(ctx)


def replay(args):
    import numpy as np
    sys.path.insert(0, str(ROOT / 'python'))
    from crispasr import Session
    out, data = Path(args.output), Path(args.reference)
    prior = json.loads(data.read_text())
    model = Path(args.model)
    result = dict(profile=args.profile, device=args.device, model_sha256=digest(model), fresh={}, reused={})

    def save():
        (out / 'result.json').write_text(json.dumps(result, indent=2) + '\n')

    def pcm(label):
        info = prior['cases'][label]
        value = np.load(data.parent / info['file'], allow_pickle=False)
        assert value.dtype == np.float32 and value.ndim == 1
        assert digest_bytes(value.tobytes()) == info['sha256']
        assert len(value) == info['samples']
        return value

    def session(backend='nemotron'):
        return Session(str(model), lib_path=args.library, backend=backend, n_threads=4,
                       use_gpu=args.device == 'cuda', verbosity=1 if args.device == 'cuda' else 0)

    resources()
    labels = list(prior['cases'])
    assert len(labels) == 26
    for label in labels:
        with session() as s:
            assert s.backend == 'nemotron'
            if args.device == 'cuda' and 'actual_cuda' not in result:
                mapped = sorted({line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines()
                                 if 'libggml-cuda.so' in line})
                allocations = subprocess.check_output(
                    ['nvidia-smi', '--id=0', '--query-compute-apps=pid,used_gpu_memory',
                     '--format=csv,noheader,nounits'], text=True)
                result['actual_cuda'] = dict(mapped_backend=mapped, process_allocations=allocations,
                                             process_id=os.getpid(),
                                             minimum_process_allocation_mib=(model.stat().st_size + 1024**2 - 1) // 1024**2,
                                             allocation_scope='own-process total CUDA memory; not isolated weight bytes')
                save()
                assert mapped, 'Actual CUDA backend is not mapped'
                sizes = [float(line.split(',')[-1].strip()) for line in allocations.splitlines()
                         if re.fullmatch(r'\s*\d+\s*,\s*[0-9.]+\s*', line)
                         and int(line.split(',')[0].strip()) == os.getpid()]
                assert sizes and max(sizes) >= result['actual_cuda']['minimum_process_allocation_mib'], (
                    'Own-process CUDA allocation is smaller than the pinned model file')
            result['fresh'][label] = ' '.join(x.text for x in s.transcribe(pcm(label), language='en-US'))
        save()
    with session() as s:
        def reused():
            for label in reversed(labels):
                result['reused'][label] = ' '.join(x.text for x in s.transcribe(pcm(label), language='en-US'))
                save()
        if args.device == 'cuda':
            with gpu_decode_activity(out, result, save):
                reused()
        else:
            reused()
    # Keep model links in scratch: a native abort can bypass Python cleanup,
    # and artifact collectors may follow symlinks in the evidence directory.
    with tempfile.TemporaryDirectory(prefix='nemotron-auto-', dir=os.environ['HEAVY_SCRATCH']) as auto_dir:
        opaque = Path(auto_dir) / 'opaque.gguf'
        opaque.symlink_to(model)
        with Session(str(opaque), lib_path=args.library, n_threads=4, use_gpu=args.device == 'cuda') as s:
            assert s.backend == 'nemotron'
            result['cabi_auto'] = ' '.join(x.text for x in s.transcribe(pcm(labels[0]), language='en-US'))
    result['auto_matches_explicit'] = result['cabi_auto'] == result['fresh'][labels[0]]
    result['original_word_matches'] = sum(words(text) == words(prior['original'][label]['transcript'])
                                          for label, text in result['fresh'].items())
    result['voxcpm_exact'] = sum(words(result['fresh'][label]) == words(prior['cases'][label]['expected'])
                                 for label in labels if label.startswith('voxcpm2'))
    result['state_changes'] = [label for label in labels if result['fresh'][label] != result['reused'][label]]
    cli_label = next(label for label in labels if label.startswith('voxcpm2') and label.endswith('short'))
    audio = pcm(cli_label).astype('<f4').tobytes()
    wav = out / 'cli-control.wav'
    wav.write_bytes(b'RIFF' + struct.pack('<I', 36 + len(audio)) + b'WAVEfmt ' +
                    struct.pack('<IHHIIHH', 16, 3, 1, 16000, 64000, 4, 32) +
                    b'data' + struct.pack('<I', len(audio)) + audio)
    command = [args.cli, '--backend', 'nemotron', '-m', str(model), '-f', str(wav),
               '-t', '4', '-l', 'en-US', '-otxt', '-of', str(out / 'cli-control')]
    if args.device == 'cpu':
        command.append('-ng')
    result['cli_returncode'] = run(command, 'cli', out, 600)
    if args.device == 'cuda':
        cli_log = (out / 'cli.log').read_text()
        result['cli_cuda_backend_selection_verified'] = ('nemotron: backend = CUDA0' in cli_log and
                                                         'nemotron: backend = CPU' not in cli_log)
        save()
        assert result['cli_cuda_backend_selection_verified'], 'CLI CUDA request fell back to CPU'
    textfile = out / 'cli-control.txt'
    result['cli_transcript'] = textfile.read_text() if textfile.exists() else ''
    result['cli_original_match'] = (result['cli_returncode'] == 0 and
        words(result['cli_transcript']) == words(prior['original'][cli_label]['transcript']))
    result['stages'] = stages(args.library, model, args.ref_gguf, out, args.device == 'cuda')
    result['accepted'] = (result['original_word_matches'] == 26 and result['voxcpm_exact'] == 22
                          and not result['state_changes'] and result['auto_matches_explicit'] and result['cli_original_match']
                          and all(x['passed'] for x in result['stages'].values()))
    result['complete'] = True
    save()


def digest_bytes(value):
    return hashlib.sha256(value).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--library')
    parser.add_argument('--cli')
    parser.add_argument('--child', action='store_true')
    for name in ('profile', 'model', 'reference', 'ref-gguf', 'output'):
        parser.add_argument('--' + name)
    args = parser.parse_args()
    os.environ.update(CRISPASR_NEMOTRON_CONTEXT_PRESET='0', OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='1')
    if args.child:
        replay(args)
        return
    import numpy as np
    from huggingface_hub import hf_hub_download
    out = Path(os.environ['HEAVY_OUT'])
    scratch = Path(os.environ['HEAVY_SCRATCH']) / 'nemotron-default-replay'
    out.mkdir(parents=True, exist_ok=True)
    scratch.mkdir(parents=True, exist_ok=True)
    assert metric(np.eye(3), np.eye(3))['passed']
    assert not metric(np.eye(3) * 2, np.eye(3))['passed'], 'Scale-blind gate'
    assert not metric(-np.eye(3), np.eye(3))['passed']
    receipt = dict(complete=False, source=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                   device=args.device, original_reference_revision=PROOF_REV, original_reference_sha256=PROOF_SHA,
                   context_preset=0, language='en-US', profiles={}, defaults_changed=False,
                   stage_gates=dict(cosine_min=.999, relative_l2_max=.01), stage_attention='native diagnostic default',
                   session_attention='existing Session default', original_model_rerun=False)

    def save():
        (out / 'replay.json').write_text(json.dumps(receipt, indent=2) + '\n')

    save()
    if not args.library:
        assert args.device == 'cpu', 'CUDA requires prebuilt package; do not build on Kaggle'
        resources()
        assert run(['sudo', 'apt-get', 'update'], 'apt-update', out, 600) == 0
        assert run(['sudo', 'apt-get', 'install', '-y', 'ninja-build', 'ccache', 'libssl-dev'], 'apt-install', out, 600) == 0
        build = scratch / 'build'
        flags = ['-G', 'Ninja', '-DCMAKE_BUILD_TYPE=Release', '-DBUILD_SHARED_LIBS=ON',
                 '-DCMAKE_C_COMPILER_LAUNCHER=ccache', '-DCMAKE_CXX_COMPILER_LAUNCHER=ccache',
                 '-DGGML_NATIVE=OFF', '-DGGML_CUDA=OFF', '-DGGML_BLAS=OFF', '-DGGML_VULKAN=OFF',
                 '-DGGML_AVX2=ON', '-DGGML_FMA=ON', '-DGGML_F16C=ON', '-DGGML_BMI2=ON',
                 '-DGGML_SSE42=ON', '-DGGML_AVX=ON', '-DGGML_AVX512=OFF', '-DCRISPASR_BUILD_TESTS=OFF']
        receipt['build_flags'] = flags
        assert run(['cmake', '-S', ROOT, '-B', build] + flags, 'configure', out) == 0
        resources()
        assert run(['cmake', '--build', build, '--target', 'crispasr-lib', 'crispasr-cli', 'crispasr-diff', '-j2'], 'build', out) == 0
        args.library = str(next(build.rglob('libcrispasr.so')))
        args.cli = str(build / 'bin/crispasr')
    receipt['library_sha256'] = digest(args.library)
    receipt['cli_sha256'] = digest(args.cli)
    receipt['runtime_registry_default'] = runtime_default(args.library)
    proof = hf_hub_download(FIXTURES, 'tts-asr/nemotron-reference-20261009/proof.zip', revision=PROOF_REV)
    assert digest(proof) == PROOF_SHA
    reference = scratch / 'reference'
    with zipfile.ZipFile(proof) as archive:
        assert sum(x.file_size for x in archive.infolist()) <= 512 * 1024**2
        assert all(not Path(x.filename).is_absolute() and '..' not in Path(x.filename).parts for x in archive.infolist())
        archive.extractall(reference)
    prior_path = next(reference.rglob('nemotron-tts-reference.json'))
    prior = json.loads(prior_path.read_text())
    assert prior['diagnostic_complete'] and len(prior['original']) == len(prior['cases']) == 26
    for label in prior['cases']:
        assert words(prior['original'][label]['transcript']) == words(prior['native']['f16/fresh/' + label]['transcript'])
    manifest = json.loads((ROOT / 'tests/regression/manifest.json').read_text())
    entry = next(x for x in manifest['backends'] if x['backend_id'] == 'nemotron')
    assert entry['stage_threshold_default'] == .999 and entry['diff_thresholds']['encoder_output'] == .999
    ref = hf_hub_download(FIXTURES, entry['fixture_ref_path'], revision=manifest['fixtures']['revision'])
    receipt['reference_gguf'] = dict(revision=manifest['fixtures']['revision'], path=entry['fixture_ref_path'], sha256=digest(ref))
    for profile, (repo, revision, filename, sha, size) in MODELS.items():
        model = hf_hub_download(repo, filename, revision=revision)
        assert Path(model).stat().st_size == size and digest(model) == sha
        dest = out / profile
        dest.mkdir(exist_ok=True)
        code = run([sys.executable, __file__, '--child', '--profile', profile, '--device', args.device,
                    '--model', model, '--reference', prior_path, '--ref-gguf', ref, '--output', dest,
                    '--library', args.library, '--cli', args.cli], profile + '-session', out, 7200)
        row = dict(repo=repo, revision=revision, file=filename, sha256=sha, bytes=size, returncode=code)
        if (dest / 'result.json').exists():
            row.update(json.loads((dest / 'result.json').read_text()))
        if args.device == 'cuda':
            log = (out / (profile + '-session.log')).read_text()
            row['cuda_backend_selection_verified'] = ('nemotron: backend = CUDA0' in log and
                                                       'nemotron: backend = CPU' not in log)
            assert row['cuda_backend_selection_verified'], 'CUDA request fell back to CPU'
        receipt['profiles'][profile] = row
        save()
        print('NEMOTRON_PROFILE=' + json.dumps({key: row.get(key) for key in
              ('profile', 'device', 'returncode', 'complete', 'accepted', 'original_word_matches', 'voxcpm_exact')}),
              flush=True)
        if profile == 'f16':
            assert code == 0 and row.get('accepted'), 'Pinned F16/original control must pass before interpreting Q4'
    receipt['complete'] = all(x.get('complete') for x in receipt['profiles'].values())
    receipt['legacy_registry_q4_accepted'] = receipt['profiles']['registry-q4'].get('accepted', False)
    receipt['current_default_accepted'] = receipt['profiles'][receipt['runtime_registry_default']['profile']].get('accepted', False)
    save()
    assert receipt['complete'], 'Replay incomplete; raw failure retained'
    lines = ['Complete 26-case original-reference replay; see replay.json.',
             'Diagnostic completion does not approve Q4 or change defaults.', '',
             '| Profile | Accepted | Original transcript matches | VoxCPM prompt matches |',
             '| --- | --- | --- | --- |']
    for profile, row in receipt['profiles'].items():
        lines.append(f'| {profile} | {row["accepted"]} | {row["original_word_matches"]}/26 | {row["voxcpm_exact"]}/22 |')
    (out / 'summary.md').write_text('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
