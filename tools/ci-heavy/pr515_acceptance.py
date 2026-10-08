#!/usr/bin/env python3
"""PR #515: real empty-cache bundles and C ABI streaming, on a hosted CPU.

No GPU speed or new model-parity claim. Record each completed check before
moving to the next model. Numerical regressions remain covered separately by
regression.yml's pinned crispasr-diff fixtures.
"""
import ctypes as C
import hashlib
import json
import os
import platform
import re
from pathlib import Path
import subprocess
import sys
import wave

import gguf
import numpy as np
from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(os.environ['HEAVY_OUT'])
SCRATCH = Path(os.environ['HEAVY_SCRATCH']) / 'pr515'
OUT.mkdir(parents=True, exist_ok=True)
SCRATCH.mkdir(parents=True, exist_ok=True)
os.environ['TMPDIR'] = str(SCRATCH)
os.environ['HF_HOME'] = str(SCRATCH / 'hf')
os.environ.pop('CRISPASR_MODELS_DIR', None)
results = {'commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
           'scope': 'CPU download/bundle/C ABI packet/flush acceptance; no GPU performance claim',
           'architecture': platform.machine(), 'cases': []}


def save():
    (OUT / 'acceptance.json').write_text(json.dumps(results, indent=2, ensure_ascii=False) + '\n')


def run(command, tag, timeout=2400):
    with (OUT / (tag + '.log')).open('w') as log:
        r = subprocess.run(list(map(str, command)), cwd=ROOT, stdin=subprocess.DEVNULL,
                           stdout=log, stderr=subprocess.STDOUT, timeout=timeout)
    text = (OUT / (tag + '.log')).read_text()
    print(tag, r.returncode, text[-1200:], flush=True)
    assert r.returncode == 0, tag
    return text


sdk = SCRATCH / 'ort'
run(['bash', ROOT / 'scripts/fetch_onnxruntime.sh', sdk], 'sdk')
run(['bash', ROOT / 'scripts/fetch_onnxruntime.sh', sdk], 'sdk-cache')
build = SCRATCH / 'build'
run(['cmake', '-S', ROOT, '-B', build, '-G', 'Ninja', '-DCMAKE_BUILD_TYPE=Release',
     '-DBUILD_SHARED_LIBS=ON', '-DGGML_NATIVE=OFF', '-DGGML_BLAS=OFF',
     '-DCRISPASR_BUILD_SERVER=OFF', '-DCRISPASR_BUILD_TESTS=ON',
     '-DCRISPASR_ONNXRUNTIME_ROOT=' + str(sdk)], 'configure')
run(['cmake', '--build', build, '--target', 'crispasr-cli', 'crispasr-lib',
     'crispasr-diff', 'test-moonshine-tokenizer', 'test-qwen3-stream', 'test-nemotron-params', 'test-registry', '-j4'], 'build')
for target in ['test-moonshine-tokenizer', 'test-qwen3-stream', 'test-nemotron-params', 'test-registry']:
    run([build / 'bin' / target], target)
run([sys.executable, ROOT / 'tools/check-backend-wiring.py', '--crispasr', build / 'bin/crispasr', '--require-lib'], 'wiring')

manifest = json.loads((ROOT / 'tests/regression/manifest.json').read_text())
# This later fixture is absent from the global regression revision. Pin it
# independently; do not advance every backend's oracle to fetch one audio clip.
fixture_revision = 'e9a1bd899e6b46f8a624aec05fc605c5461b13db'
fixture = hf_hub_download(manifest['fixtures']['repo'], 'orukeet/de/ref.gguf',
                         repo_type='dataset', revision=fixture_revision,
                         local_dir=SCRATCH / 'fixture')
assert hashlib.sha256(Path(fixture).read_bytes()).hexdigest() == (
    '09fb97f8a2ed2ea4eb6a513177602fb16949a3ca9ddca923007922240c199a5d')
reader = gguf.GGUFReader(fixture)
pcm = next(np.array(t.data, dtype=np.float32).reshape(-1) for t in reader.tensors if t.name == 'raw_audio')
del reader
wav = OUT / 'german.wav'
with wave.open(str(wav), 'wb') as f:
    f.setparams((1, 2, 16000, 0, 'NONE', 'not compressed'))
    f.writeframes((np.clip(pcm, -1, 1) * 32767).astype('<i2').tobytes())
# Feed exactly the same PCM as the CLI's PCM16 WAV reader.
pcm = (np.frombuffer(wav.read_bytes()[44:], dtype='<i2').astype(np.float32) / 32768).copy()
results['audio_sha256'] = hashlib.sha256(wav.read_bytes()).hexdigest()
results['fixture_revision'] = fixture_revision

sys.path.insert(0, str(ROOT / 'python'))
from crispasr import Session
library = next(build.rglob('libcrispasr.so'))
lib = C.CDLL(str(library))
P, I, F = C.c_void_p, C.c_int, C.POINTER(C.c_float)
for name, restype, args in [
    ('crispasr_session_stream_kind', I, [P]),
    ('crispasr_session_stream_open', P, [P, I, I, I, I, C.c_char_p, I]),
    ('crispasr_stream_feed', I, [P, F, I]),
    ('crispasr_stream_flush', I, [P]),
    ('crispasr_stream_get_text', I, [P, P, I, C.POINTER(C.c_double), C.POINTER(C.c_double), C.POINTER(C.c_int64)]),
    ('crispasr_stream_close', None, [P]),
]:
    fn = getattr(lib, name)
    fn.restype, fn.argtypes = restype, args


def stream_text(session, packet):
    stream = lib.crispasr_session_stream_open(session._handle, 4, 640, 20000, 0, b'de', 0)
    assert stream
    drafts, counters = [], []
    t0, t1, counter = C.c_double(), C.c_double(), C.c_int64()

    def read():
        buf = C.create_string_buffer(16)
        n = lib.crispasr_stream_get_text(stream, buf, len(buf), C.byref(t0), C.byref(t1), C.byref(counter))
        assert n >= 0
        if n >= len(buf):
            buf = C.create_string_buffer(n + 1)
            assert lib.crispasr_stream_get_text(stream, buf, len(buf), C.byref(t0), C.byref(t1), C.byref(counter)) == n
        return buf.value.decode('utf-8')

    try:
        for off in range(0, len(pcm), packet):
            chunk = pcm[off:off + packet]
            assert lib.crispasr_stream_feed(stream, chunk.ctypes.data_as(F), len(chunk)) >= 0
            text = read()
            if text and (not counters or counter.value != counters[-1]):
                drafts.append(text)
                counters.append(counter.value)
        assert lib.crispasr_stream_flush(stream) >= 0
        final = read()
        assert lib.crispasr_stream_flush(stream) == 0
        assert read() == final
        assert lib.crispasr_stream_feed(stream, pcm.ctypes.data_as(F), 1) < 0
        assert drafts, 'Must emit before flush'
        assert final.strip(), 'Final text is empty'
        return final.strip(), drafts
    finally:
        lib.crispasr_stream_close(stream)


variants = [
    ('moonshine-streaming-tiny-de-onnx', 2), ('moonshine-streaming-small-de-onnx', 2),
    ('moonshine-streaming-tiny-de-onnx-f32', 2), ('moonshine-streaming-small-de-onnx-f32', 2),
    ('moonshine-tiny-de-phreak87-onnx', 0), ('moonshine-tiny-de-dattazigzag', 0),
    ('nemotron', 2), ('qwen3', 3),
]
cache = SCRATCH / 'cache'
for variant, expected_kind in variants:
    case = {'variant': variant, 'passed': False}
    results['cases'].append(case)
    try:
        prefix = OUT / variant
        native = variant in ('nemotron', 'qwen3')
        if native:
            entry = next(x for x in manifest['backends'] if x['name'] ==
                         ('nemotron-3.5-asr-streaming-0.6b' if variant == 'nemotron' else 'qwen3-asr-0.6b'))
            pin = entry['gguf']
            model_arg = hf_hub_download(pin['repo'], pin['file'], revision=pin['revision'],
                                       local_dir=SCRATCH / variant)
            case['model_pin'] = pin
        else:
            model_arg = variant
        log = run([build / 'bin/crispasr', '-m', model_arg, '--auto-download', '--cache-dir', cache,
                   '-ng', '-t', '4', '-l', 'de', '-f', wav, '-otxt', '-of', prefix], variant)
        cli_text = prefix.with_suffix('.txt').read_text().strip()
        assert cli_text
        recognized = set(re.findall(r'[^\W\d_]+', cli_text.lower()))
        assert len(recognized & {'morgen', 'sitzung', 'heute', 'neun', 'großen', 'saal'}) >= 4, cli_text
        # Find the exact primary from the model-specific cache, not a generic filename.
        if native:
            model = Path(model_arg)
        elif variant.endswith('dattazigzag'):
            model = cache / 'moonshine-tiny-de-dattazigzag-q4_k/moonshine-tiny-de-dattazigzag-q4_k.gguf'
        elif variant.endswith('phreak87-onnx'):
            model = cache / variant / 'onnx/encoder_model.onnx'
        else:
            model = cache / variant / ('encoder.onnx' if variant.endswith('f32') else 'encoder_int8.onnx')
        if variant.startswith('moonshine-streaming'):
            reference = OUT / (variant + '-ref.gguf')
            run([sys.executable, ROOT / 'tools/dump_reference.py', '--backend', 'moonshine-onnx',
                 '--model-dir', model, '--audio', wav, '--output', reference], variant + '-reference')
            run([build / 'bin/crispasr-diff', 'moonshine-onnx', model, reference, wav], variant + '-diff')
            case['graph_diff'] = 'passed; exact input/shape, cosine >= 0.99999, relative L2 <= 1e-4, text exact'
            if variant == 'moonshine-streaming-tiny-de-onnx':
                # A uniform scale defect keeps cosine perfect. Prove the new
                # magnitude gate rejects it using the actual native diff path.
                sys.path.insert(0, str(ROOT / 'tools'))
                from dump_reference import write_gguf_archive
                r = gguf.GGUFReader(str(reference))
                tensors = {t.name: np.array(t.data, dtype=np.float32, copy=True) for t in r.tensors}
                key = next(k for k in tensors if k.startswith('encoder.'))
                tensors[key] *= 2
                bad = OUT / 'scale-control-ref.gguf'
                # Preserve the original text oracle: only the selected encoder
                # tensor changes, so rejection must be solely numerical.
                write_gguf_archive(tensors, {
                    'backend': 'moonshine-onnx',
                    'generated_text': r.fields['crispasr.ref.generated_text'].contents(),
                }, bad)
                with (OUT / 'scale-control.log').open('w') as log_file:
                    control = subprocess.run([str(build / 'bin/crispasr-diff'), 'moonshine-onnx',
                                              str(model), str(bad), str(wav)], cwd=ROOT,
                                             stdout=log_file, stderr=subprocess.STDOUT, timeout=600)
                control_log = (OUT / 'scale-control.log').read_text()
                rejected = re.findall(r'relative_l2=([0-9.eE+\-]+) FAIL', control_log)
                assert control.returncode != 0 and len(rejected) == 1, control_log
                assert abs(float(rejected[0]) - 0.5) < 0.01, control_log
                assert '[PASS] generated_text:' in control_log, control_log
                assert 'summary: 15 pass, 1 fail, 0 skip' in control_log, control_log
                case['scale_failure_rejected'] = True
        with Session(str(model), lib_path=str(library), n_threads=4) as session:
            kind = lib.crispasr_session_stream_kind(session._handle)
            assert session.stream_kind() == kind
            assert kind == expected_kind, (kind, expected_kind)
            batch = ' '.join(x.text for x in session.transcribe(pcm, language='de')).strip()
            assert batch == cli_text, (batch, cli_text)
            case.update(batch=batch, streaming_kind=kind)
            if kind >= 2:
                a, drafts_a = stream_text(session, 1777)
                b, drafts_b = stream_text(session, 5120)
                assert a == b, (a, b)
                # This backend's final flush is full BOS decoding of the same encoder.
                if variant.startswith('moonshine-streaming'):
                    assert a == batch, (a, batch)
                words = set(re.findall(r'[^\W\d_]+', a.lower()))
                assert len(words & {'morgen', 'sitzung', 'heute', 'neun', 'großen', 'saal'}) >= 4, a
                case.update(final=a, updates=[len(drafts_a), len(drafts_b)])
        case['passed'] = True
    except Exception as e:
        case['error'] = repr(e)
        print(variant, repr(e), flush=True)
    save()
results['passed'] = all(x['passed'] for x in results['cases'])
save()
(OUT / 'summary.md').write_text('PR #515 CPU bundle/C ABI acceptance: ' + ('PASS' if results['passed'] else 'FAIL') + '\n')
raise SystemExit(0 if results['passed'] else 1)
