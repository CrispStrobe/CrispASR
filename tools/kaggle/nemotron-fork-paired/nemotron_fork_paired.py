#!/usr/bin/env python3
"""Private, pinned, inference-only T4 comparison of Nemotron F16 runtimes."""

import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import statistics
import struct
import subprocess
import sys
import tarfile
import threading
import time
import unicodedata
import urllib.request
import wave
import zipfile

import numpy as np


VERSION = 'nemotron-f16-fork-paired-v2'
OUT = Path('/kaggle/working/nemotron-fork-paired-v2')
SCRATCH = Path('/kaggle/temp/nemotron-fork-paired-v2')
LICENSE = dict(
    name='NVIDIA OpenMDW-1.1',
    upstream='https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b',
    crisp_notice_sha256='728c1751d295f5bf6856c20ffbda1805c5f6a14cb2e36f0a2e5c09c091bd9cc6',
    crisp_license_sha256='be21b29b1c69af5b4a4e60a6bd7bc933eec374e52be29d1ebb9e6a17a6588491',
    runtime_license='transcribe.cpp MIT; NVIDIA model terms apply to converted GGUFs',
)
CRISP = dict(
    runtime='CrispASR', tag='v0.8.47', source='5f87a56265e4bbd7e2adc7052254e576d5774abb',
    archive_url='https://github.com/CrispStrobe/CrispASR/releases/download/v0.8.47/crispasr-linux-x86_64-cuda.tar.gz',
    archive_sha256='90e23cffb44f4a80d929bf88cf3abb2d526ec4c0b95d88a82d1a8bf33135722d',
    archive_bytes=225497671,
    model_repo='cstr/nemotron-3.5-asr-streaming-GGUF',
    model_revision='bbd95a9ca5fa0dfca3312a122dfc45a2b578b9c2',
    model_file='nemotron-3.5-asr-streaming-0.6b-f16.gguf',
    model_sha256='14cbc98df495d8ab452c6a4ac367136d94ab5da55e73afaa96ba4f749dd95d09',
    model_bytes=1286368480,
)
TRANSCRIBE = dict(
    runtime='transcribe.cpp', tag='v0.4.1', source='4d7c1ccefd39e5a9ea36cca5f8fefc6dcc14b99b',
    archive_url='https://github.com/handy-computer/transcribe.cpp/releases/download/v0.4.1/transcribe-native-0.4.1-linux-x86_64-cuda.tar.gz',
    archive_sha256='c4bdf9c4798bac1868ecdcaa04f27eb7712fb43125268313ce14d508c70d247d',
    archive_bytes=203199021,
    python_wheel_url='https://github.com/handy-computer/transcribe.cpp/releases/download/v0.4.1/transcribe_cpp-0.4.1-py3-none-any.whl',
    python_wheel_sha256='f373b6e20a4d897ed0dcaa2fe7fae183379777432b3a76641c42506c77b442ad',
    python_wheel_bytes=47639,
    model_repo='handy-computer/nemotron-3.5-asr-streaming-0.6b-gguf',
    model_revision='53ce288f69326b415aeef030cfcb02a08a8096dd',
    model_file='nemotron-3.5-asr-streaming-0.6b-F16.gguf',
    model_sha256='f21a0cea64d232981def7f8f2b7ab322459703a2e89359962c042c57159755b1',
    model_bytes=1277750240,
    upstream_revision='24b151a851dd15909e1fc611b11bb2da52b9fc81',
    upstream_artifact='nemotron-3.5-asr-streaming-0.6b.nemo',
    upstream_artifact_sha256='79766c070eed987b43ee595fff7bd21fe49aae6ee26e881f51b86d8e662e713d',
)
PROOF = dict(
    url='https://huggingface.co/cstr/crispasr-regression-fixtures/resolve/10e579dedf52a16a159cbc5d35e17f8bffa77190/tts-asr/nemotron-reference-20261009/proof.zip',
    sha256='7762f2a73d8ae7dad05d667fbf0bca393a0b3b7ad622b1b1f09473fa8f5d0463',
    bytes=17083344,
    original_revision='ea30d66debe3740a08b573244286791d423d6b3e',
    original_sha256='9eebdd6590289cb3030f310858f3df93256600a800a3e8200c5993d5f967e174',
)
JFK = dict(
    url='https://raw.githubusercontent.com/CrispStrobe/CrispASR/cc114d2b4a5a9413738f390093e9131576743cb6/samples/jfk.wav',
    sha256='59dfb9a4acb36fe2a2affc14bacbee2920ff435cb13cc314a08c13f66ba7860e',
    text='and so my fellow americans ask not what your country can do for you ask what you can do for your country',
)


def progress(event, **fields):
    row = dict(event=event, time_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), **fields)
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / 'progress.jsonl').open('a') as f:
        f.write(json.dumps(row, sort_keys=True) + '\n')
    print(json.dumps(row, sort_keys=True), flush=True)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def download(url, dest, expected_hash, expected_size):
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size == expected_size and sha256(dest) == expected_hash:
        return dest
    partial = dest.with_suffix(dest.suffix + '.part')
    partial.unlink(missing_ok=True)
    req = urllib.request.Request(url, headers={'User-Agent': 'nemotron-f16-private-benchmark/1'})
    h = hashlib.sha256()
    size = 0
    with urllib.request.urlopen(req, timeout=120) as response, partial.open('wb') as f:
        while chunk := response.read(4 * 1024 * 1024):
            h.update(chunk)
            f.write(chunk)
            size += len(chunk)
    assert size == expected_size and h.hexdigest() == expected_hash, f'Asset pin mismatch: {dest.name}'
    partial.replace(dest)
    progress('download.done', file=dest.name, bytes=size, sha256=expected_hash)
    return dest


def safe_archive_member(name):
    p = PurePosixPath(name)
    return bool(name) and not p.is_absolute() and '..' not in p.parts


def unpack_release(spec, stem):
    archive = download(spec['archive_url'], SCRATCH / (stem + '.tar.gz'),
                       spec['archive_sha256'], spec['archive_bytes'])
    root = SCRATCH / stem
    root.mkdir(exist_ok=True)
    with tarfile.open(archive, 'r:gz') as tf:
        members = tf.getmembers()
        assert members and len(members) < 500 and sum(m.size for m in members) < 1024**3
        assert all(safe_archive_member(m.name) and (m.isfile() or m.isdir()) for m in members)
        tf.extractall(root, filter='data')
    candidate = 'crispasr' if stem == 'crisp' else 'libtranscribe.so'
    binaries = [p for p in root.rglob(candidate) if p.is_file()]
    assert len(binaries) == 1, f'Expected one {candidate}, got {binaries}'
    binary = binaries[0]
    if stem == 'crisp':
        assert os.access(binary, os.X_OK), f'Not executable: {binary}'
    libdirs = sorted({str(p.parent) for p in root.rglob('*.so*') if p.is_file()})
    assert libdirs, f'No shared libraries in {stem} release'
    return binary, libdirs


def model(spec, stem):
    url = ('https://huggingface.co/' + spec['model_repo'] + '/resolve/' +
           spec['model_revision'] + '/' + spec['model_file'])
    return download(url, SCRATCH / stem / spec['model_file'],
                    spec['model_sha256'], spec['model_bytes'])


def gguf_context(path):
    """Read only the GGUF header KVs needed to prove offline attention settings."""
    scalars = {0: 'B', 1: 'b', 2: 'H', 3: 'h', 4: 'I', 5: 'i',
               6: 'f', 7: '?', 10: 'Q', 11: 'q', 12: 'd'}
    with Path(path).open('rb') as f:
        def exact(n):
            data = f.read(n)
            assert len(data) == n, 'Truncated GGUF metadata'
            return data

        def number(fmt):
            return struct.unpack('<' + fmt, exact(struct.calcsize(fmt)))[0]

        def value(t):
            if t in scalars:
                return number(scalars[t])
            if t == 8:
                n = number('Q')
                assert n < 1024 * 1024
                return exact(n).decode('utf-8')
            if t == 9:
                item_type, n = number('I'), number('Q')
                assert n < 1000000
                return [value(item_type) for _ in range(n)]
            raise AssertionError(f'Unsupported GGUF metadata type {t}')

        assert exact(4) == b'GGUF'
        version = number('I')
        assert version == 3, f'Unexpected GGUF version {version}'
        _n_tensors, n_kv = number('Q'), number('Q')
        assert n_kv < 100000
        wanted = ('stt.parakeet.encoder.att_context_left',
                  'stt.parakeet.encoder.att_context_right',
                  'stt.parakeet.encoder.att_context_style')
        found = {}
        for _ in range(n_kv):
            key = value(8)
            t = number('I')
            entry = value(t)
            if key in wanted:
                found[key.rsplit('.', 1)[-1]] = entry
        assert found == dict(att_context_left=56, att_context_right=13,
                             att_context_style='chunked_limited'), f'Unmatched transcribe GGUF context: {found}'
        return found


def normalized_words(s):
    return re.findall(r'\w+', unicodedata.normalize('NFKC', s).casefold(), re.UNICODE)


def errors(ref, hyp):
    a, b = normalized_words(ref), normalized_words(hyp)
    assert a, 'Empty reference'
    prev = list(range(len(b) + 1))
    for i, word in enumerate(a, 1):
        cur = [i]
        for j, got in enumerate(b, 1):
            cur.append(min(cur[-1] + 1, prev[j] + 1, prev[j - 1] + (word != got)))
        prev = cur
    return dict(errors=prev[-1], words=len(a), wer=prev[-1] / len(a))


def prepare_audio():
    proof = download(PROOF['url'], SCRATCH / 'proof.zip', PROOF['sha256'], PROOF['bytes'])
    audio_dir = SCRATCH / 'audio'
    audio_dir.mkdir(exist_ok=True)
    cases = []
    with zipfile.ZipFile(proof) as z:
        names = z.namelist()
        assert all(safe_archive_member(n) for n in names)
        proof_json = [n for n in names if n.endswith('/nemotron-tts-reference.json')]
        assert len(proof_json) == 1
        reference = json.loads(z.read(proof_json[0]))
        assert reference['original_model']['revision'] == PROOF['original_revision']
        assert reference['original_model']['sha256'] == PROOF['original_sha256']
        assert reference['native_models']['f16']['sha256'] == CRISP['model_sha256']
        assert len(reference['cases']) == len(reference['original']) == 26
        for key, case in sorted(reference['cases'].items(), key=lambda kv: kv[1]['file']):
            assert key in reference['original'] and case['sample_rate'] == 16000
            matches = [n for n in names if n.endswith('/' + case['file'])]
            assert len(matches) == 1
            arr = np.load(io.BytesIO(z.read(matches[0])), allow_pickle=False)
            assert arr.dtype == np.dtype('float32') and arr.ndim == 1 and arr.size == case['samples']
            assert hashlib.sha256(arr.tobytes()).hexdigest() == case['sha256']
            pcm = np.clip(np.rint(arr.astype(np.float64) * 32768.0), -32768, 32767).astype('<i2')
            path = audio_dir / case['file'].replace('.npy', '.wav')
            with wave.open(str(path), 'wb') as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(16000)
                w.writeframes(pcm.tobytes())
            cases.append(dict(id=key, file=str(path), wav_sha256=sha256(path),
                              original_float_sha256=case['sha256'], original_text=reference['original'][key]['transcript'],
                              intended_text=case['expected'], seconds=arr.size / 16000))
    assert len(cases) == 26
    jfk = audio_dir / 'jfk.wav'
    # JFK's source size is checked via its pinned SHA; it is already PCM16/16 kHz.
    if not jfk.exists() or sha256(jfk) != JFK['sha256']:
        with urllib.request.urlopen(JFK['url'], timeout=30) as r, jfk.open('wb') as f:
            shutil.copyfileobj(r, f)
    assert sha256(jfk) == JFK['sha256']
    with wave.open(str(jfk), 'rb') as w:
        assert (w.getnchannels(), w.getsampwidth(), w.getframerate()) == (1, 2, 16000)
        seconds = w.getnframes() / w.getframerate()
    cases.append(dict(id='jfk', file=str(jfk), wav_sha256=sha256(jfk),
                      original_text=JFK['text'], intended_text=JFK['text'], seconds=seconds))
    return cases


def gpu_identity():
    q = subprocess.check_output(['nvidia-smi', '--query-gpu=name,compute_cap,memory.total',
                                 '--format=csv,noheader'], text=True, timeout=20).strip().splitlines()
    assert q and re.search(r'\bT4\b', q[0]), f'T4 required on device 0; no allocation retry: {q}'
    return dict(device_0=q[0], visible_devices=q)


def gpu_monitor(pid, stop, samples):
    while not stop.is_set():
        try:
            out = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,used_gpu_memory',
                                           '--format=csv,noheader,nounits'], text=True, timeout=10)
            for line in out.splitlines():
                parts = [x.strip() for x in line.split(',')]
                if len(parts) >= 2 and parts[0] == str(pid):
                    samples['memory_mib'].append(int(parts[1]))
        except (subprocess.SubprocessError, ValueError):
            pass
        try:
            out = subprocess.check_output(['nvidia-smi', 'pmon', '-c', '1', '-s', 'u'],
                                          text=True, timeout=10, stderr=subprocess.DEVNULL)
            for line in out.splitlines():
                fields = line.split()
                if len(fields) >= 4 and fields[1] == str(pid) and fields[3].isdigit():
                    samples['sm_percent'].append(int(fields[3]))
        except subprocess.SubprocessError:
            pass


def run_process(command, env, label, timeout=900):
    assert all(isinstance(x, str) for x in command)
    proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True, cwd=SCRATCH, env=env)
    samples = dict(memory_mib=[], sm_percent=[])
    stop = threading.Event()
    monitor = threading.Thread(target=gpu_monitor, args=(proc.pid, stop, samples), daemon=True)
    monitor.start()
    start = time.monotonic()
    timed_out = False
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        stdout, stderr = proc.communicate()
        timed_out = True
    finally:
        stop.set()
        monitor.join(timeout=12)
    elapsed = time.monotonic() - start
    (OUT / (label + '.stdout.txt')).write_text(stdout)
    (OUT / (label + '.stderr.txt')).write_text(stderr)
    assert not timed_out, f'{label} timed out; stdout/stderr saved'
    assert proc.returncode == 0, f'{label} returned {proc.returncode}'
    assert samples['memory_mib'] and max(samples['memory_mib']) > 0, f'{label} no own-PID GPU allocation'
    if label.endswith('-quality'):
        assert samples['sm_percent'] and max(samples['sm_percent']) > 0, f'{label} no own-PID SM activity'
    return stdout, stderr, elapsed, dict(max_memory_mib=max(samples['memory_mib']),
                                         max_sm_percent=max(samples['sm_percent'], default=0),
                                         memory_samples=len(samples['memory_mib']),
                                         sm_samples=len(samples['sm_percent']))


def clean_env(libdirs, preset=None):
    assert not any(k.startswith(('CRISPASR_', 'GGML_', 'TRANSCRIBE_')) for k in os.environ), 'Inherited runtime override'
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES='0', OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='1',
               LD_LIBRARY_PATH=':'.join(libdirs + [env.get('LD_LIBRARY_PATH', '')]))
    if preset is not None:
        env['CRISPASR_NEMOTRON_CONTEXT_PRESET'] = str(preset)
    return env


def crisp_run(binary, libdirs, model_path, cases, arm, label):
    for case in cases:
        Path(case['file'] + '.txt').unlink(missing_ok=True)
    preset = 3 if arm == 'crisp_r13' else None
    env = clean_env(libdirs, preset)
    env['CRISPASR_NEMOTRON_BENCH'] = '1'
    command = [str(binary), '--backend', 'nemotron', '-m', str(model_path),
               '-l', 'en-US', '-t', '4', '-bs', '1', '-otxt', '--gpu-backend', 'cuda']
    for case in cases:
        command.extend(['-f', case['file']])
    stdout, stderr, elapsed, gpu = run_process(command, env, label)
    assert re.search(r'nemotron: backend = CUDA', stderr, re.I), 'Crisp CUDA backend not selected'
    expected_r = 13 if preset == 3 else 3
    observed = re.findall(r'nemotron: streaming attention L=(\d+) R=(\d+)', stderr)
    assert observed == [('56', str(expected_r))] * len(cases), f'{arm}: offline attention mask not verified'
    assert 'running streaming chunked encoder path' not in stderr
    if preset == 3:
        assert stderr.count('nemotron: context preset 3 (L=56, R=13)') == 1
    else:
        assert 'nemotron: context preset' not in stderr, 'R3 must be the production default'
    stages = {}
    for name in ('mel', 'encoder', 'prompt', 'rnnt_decode'):
        vals = [float(x) for x in re.findall(r'^\s*nemotron_bench:\s+' + name + r'\s+([0-9.]+) ms$',
                                             stderr, re.M)]
        assert len(vals) == len(cases), f'{arm}: expected {len(cases)} {name} timings, got {len(vals)}'
        stages[name + '_ms'] = vals
    transcripts = []
    for case in cases:
        path = Path(case['file'] + '.txt')
        assert path.is_file(), f'Missing Crisp transcript {path}'
        transcript = path.read_text().strip()
        assert normalized_words(transcript), f'Empty Crisp transcript for {case["id"]}'
        transcripts.append(transcript)
    return dict(texts=transcripts, stages=stages, wall_s=elapsed, gpu=gpu,
                context=dict(left=56, right=expected_r, offline=True, source='native runtime mask log'))


def transcribe_worker():
    """Use the pinned Python API with the pinned native library in one GPU PID."""
    assert len(sys.argv) == 4
    import transcribe_cpp

    assert transcribe_cpp.__version__ == transcribe_cpp.native_version() == '0.4.1'
    assert transcribe_cpp.library_path() == os.environ['TRANSCRIBE_LIBRARY']
    with transcribe_cpp.Model(sys.argv[2], backend='cuda') as model:
        assert model.device.kind == 'cuda', f'Unexpected device {model.device.kind}: {model.backend}'
        with model.session(n_threads=4) as session:
            for index, file in enumerate(Path(sys.argv[3]).read_text().splitlines()):
                with wave.open(file, 'rb') as wav:
                    assert (wav.getnchannels(), wav.getsampwidth(), wav.getframerate()) == (1, 2, 16000)
                    pcm = np.frombuffer(wav.readframes(wav.getnframes()), dtype='<i2').astype(np.float32) / 32768.0
                result = session.run(pcm, language='en-US', timestamps='none')
                if index == 0:
                    print(json.dumps(dict(type='batch_header', load_ms=result.timings.load_ms)), flush=True)
                print(json.dumps(dict(file=file, text=result.text,
                                      mel_ms=result.timings.mel_ms,
                                      encode_ms=result.timings.encode_ms,
                                      decode_ms=result.timings.decode_ms)), flush=True)


def transcribe_run(binary, libdirs, model_path, cases, label, python_wheel):
    batch_file = SCRATCH / (label + '.batch.txt')
    batch_file.write_text(''.join(case['file'] + '\n' for case in cases))
    env = clean_env(libdirs)
    env['TRANSCRIBE_LIBRARY'] = str(binary)
    env['PYTHONPATH'] = str(python_wheel)
    command = [sys.executable, str(Path(__file__).resolve()), '--transcribe-worker',
               str(model_path), str(batch_file)]
    stdout, stderr, elapsed, gpu = run_process(command, env, label)
    rows = [json.loads(line) for line in stdout.splitlines() if line.strip()]
    assert len(rows) == len(cases) + 1 and rows[0].get('type') == 'batch_header'
    assert isinstance(rows[0].get('load_ms'), (int, float))
    stages = {k: [] for k in ('mel_ms', 'encode_ms', 'decode_ms')}
    texts = []
    for case, row in zip(cases, rows[1:]):
        assert row.get('file') == case['file'], f'transcribe.cpp batch order mismatch: {row}'
        assert not row.get('error') and normalized_words(row.get('text', ''))
        for k in stages:
            value = row.get(k)
            assert isinstance(value, (int, float)) and value >= 0, f'Missing {k}'
            stages[k].append(value)
        texts.append(row['text'])
    return dict(texts=texts, stages=stages, load_ms=rows[0]['load_ms'],
                wall_s=elapsed, gpu=gpu,
                context=dict(left=56, right=13, offline=True,
                             source='pinned v0.4.1 model default; single-utterance Python API offline path'))


def score(cases, texts):
    assert len(cases) == len(texts)
    rows = []
    for case, text in zip(cases, texts):
        rows.append(dict(id=case['id'], transcript=text,
                         against_original=errors(case['original_text'], text),
                         against_intended=errors(case['intended_text'], text)))
    original = [row['against_original'] for row in rows if row['id'] != 'jfk']
    intended = [row['against_intended'] for row in rows if row['id'] != 'jfk']
    total = lambda values: sum(v['errors'] for v in values) / sum(v['words'] for v in values)
    return dict(cases=rows, original_26_wer=total(original), intended_26_wer=total(intended),
                jfk_wer=rows[-1]['against_intended']['wer'])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    SCRATCH.mkdir(parents=True, exist_ok=True)
    assert not (OUT / 'receipt.json').exists(), 'Preserve existing receipt; do not overwrite a run'
    receipt = dict(version=VERSION, status='incomplete', pins=dict(crisp=CRISP, transcribe=TRANSCRIBE,
                                                                    proof=PROOF, jfk=JFK),
                   license=LICENSE, script_sha256=sha256(__file__), arms={})
    (OUT / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    try:
        receipt['gpu'] = gpu_identity()
        progress('gpu.preflight', gpu=receipt['gpu'])
        crisp_bin, crisp_libdirs = unpack_release(CRISP, 'crisp')
        tc_bin, tc_libdirs = unpack_release(TRANSCRIBE, 'transcribe')
        tc_wheel = download(TRANSCRIBE['python_wheel_url'], SCRATCH / 'transcribe_cpp-0.4.1-py3-none-any.whl',
                            TRANSCRIBE['python_wheel_sha256'], TRANSCRIBE['python_wheel_bytes'])
        crisp_model = model(CRISP, 'crisp')
        tc_model = model(TRANSCRIBE, 'transcribe')
        receipt['transcribe_gguf_context'] = gguf_context(tc_model)
        cases = prepare_audio()
        receipt['audio'] = cases
        progress('inputs.ready', n_cases=len(cases))
        configs = dict(crisp_r3=(crisp_bin, crisp_libdirs, crisp_model),
                       crisp_r13=(crisp_bin, crisp_libdirs, crisp_model),
                       transcribe_r13=(tc_bin, tc_libdirs, tc_model))
        for arm in ('crisp_r3', 'crisp_r13', 'transcribe_r13'):
            binary, libdirs, model_path = configs[arm]
            if arm.startswith('crisp'):
                result = crisp_run(binary, libdirs, model_path, cases, arm, arm + '-quality')
            else:
                result = transcribe_run(binary, libdirs, model_path, cases, arm + '-quality', tc_wheel)
            result['quality'] = score(cases, result['texts'])
            receipt['arms'][arm] = dict(quality_run=result)
            (OUT / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
            progress('quality.done', arm=arm, original_wer=result['quality']['original_26_wer'],
                     jfk_wer=result['quality']['jfk_wer'])
        jfk = cases[-1]
        perf_cases = []
        for name in ('warm', 'measured'):
            path = SCRATCH / 'audio' / (name + '-jfk.wav')
            shutil.copyfile(jfk['file'], path)
            assert sha256(path) == jfk['wav_sha256']
            perf_cases.append(dict(jfk, id=name, file=str(path)))
        order = ('crisp_r3', 'crisp_r13', 'transcribe_r13', 'transcribe_r13', 'crisp_r13', 'crisp_r3') * 2
        for i, arm in enumerate(order):
            binary, libdirs, model_path = configs[arm]
            label = f'perf-{i:02d}-{arm}'
            if arm.startswith('crisp'):
                result = crisp_run(binary, libdirs, model_path, perf_cases, arm, label)
            else:
                result = transcribe_run(binary, libdirs, model_path, perf_cases, label, tc_wheel)
            assert normalized_words(result['texts'][1]) == normalized_words(receipt['arms'][arm]['quality_run']['texts'][-1])
            measured = {name: values[1] for name, values in result['stages'].items()}
            receipt['arms'][arm].setdefault('measured_repeats', []).append(dict(
                stages_ms=measured, load_ms=result.get('load_ms'),
                process_wall_s=result['wall_s'], gpu=result['gpu']))
            (OUT / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
            progress('perf.done', arm=arm, repeat=len(receipt['arms'][arm]['measured_repeats']))
        for arm, data in receipt['arms'].items():
            repeats = data['measured_repeats']
            assert len(repeats) == 4
            data['median_native_stage_ms'] = {
                name: statistics.median(row['stages_ms'][name] for row in repeats)
                for name in repeats[0]['stages_ms']}
            data['quality_gate'] = (data['quality_run']['quality']['original_26_wer'] <= 0.15 and
                                    data['quality_run']['quality']['jfk_wer'] <= 0.25)
        a = receipt['arms']['crisp_r13']['quality_run']['quality']['original_26_wer']
        b = receipt['arms']['transcribe_r13']['quality_run']['quality']['original_26_wer']
        receipt['matched_r13_quality_gap_wer'] = abs(a - b)
        receipt['matched_r13_quality_compatible'] = abs(a - b) <= 0.10
        receipt['status'] = 'complete'
        receipt['accepted'] = (all(a['quality_gate'] for a in receipt['arms'].values()) and
                               receipt['matched_r13_quality_compatible'])
        progress('complete', accepted=receipt['accepted'], matched_r13_quality_gap_wer=abs(a - b))
    except Exception as e:
        receipt['error'] = str(e)
        progress('failed', error=str(e))
        raise
    finally:
        (OUT / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--transcribe-worker':
        transcribe_worker()
    else:
        main()
