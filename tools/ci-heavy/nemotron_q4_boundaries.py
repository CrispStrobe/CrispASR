#!/usr/bin/env python3
"""Locate Q4 frontend/subsampling drift under unchanged reference stage gates."""
import ctypes
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import wave

import numpy as np

from gguf import GGUFReader
from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(os.environ['HEAVY_OUT'])
TEMP = Path(os.environ['HEAVY_SCRATCH']) / 'nemotron-q4-boundaries'
OUT.mkdir(parents=True, exist_ok=True)
TEMP.mkdir(parents=True, exist_ok=True)
os.environ.update(HF_HOME=str(TEMP / 'hf'), HF_XET_CACHE=str(TEMP / 'xet'),
                  CRISPASR_NEMOTRON_CONTEXT_PRESET='0', OMP_NUM_THREADS='4')
MODEL = 'cstr/nemotron-3.5-asr-streaming-GGUF'
REV = 'bbd95a9ca5fa0dfca3312a122dfc45a2b578b9c2'
receipt = dict(source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
    diagnostic_complete=False, production_accepted=False, defaults_changed=False,
    model_repo=MODEL, model_revision=REV, models={})


def save():
    (OUT / 'q4-boundaries.json').write_text(json.dumps(receipt, indent=2) + '\n')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def run(command, label):
    with (OUT / (label + '.log')).open('w') as log:
        return subprocess.run(list(map(str, command)), cwd=ROOT, stdout=log,
                              stderr=subprocess.STDOUT, timeout=3600).returncode


save()
build = TEMP / 'build'
assert run(['cmake', '-S', ROOT, '-B', build, '-G', 'Ninja', '-DCMAKE_BUILD_TYPE=Release',
    '-DBUILD_SHARED_LIBS=ON', '-DGGML_NATIVE=OFF', '-DGGML_CUDA=OFF', '-DGGML_BLAS=OFF', '-DCRISPASR_BUILD_TESTS=OFF',
    '-DCRISPASR_BUILD_SERVER=OFF'], 'configure') == 0
assert run(['cmake', '--build', build, '--target', 'crispasr-lib', 'crispasr-diff', 'crispasr-quantize', '-j4'], 'build') == 0
manifest = json.loads((ROOT / 'tests/regression/manifest.json').read_text())
entry = next(x for x in manifest['backends'] if x['backend_id'] == 'nemotron')
ref = hf_hub_download(manifest['fixtures']['repo'], entry['fixture_ref_path'],
                      revision=manifest['fixtures']['revision'])
receipt['reference'] = dict(revision=manifest['fixtures']['revision'], path=entry['fixture_ref_path'],
    sha256=sha(ref), thresholds=entry['diff_thresholds'], default_threshold=entry['stage_threshold_default'])
reference = GGUFReader(ref)
receipt['reference']['stages'] = {t.name: list(map(int, t.shape)) for t in reference.tensors}
del reference
for stage in ('mel_spectrogram', 'pre_encode_output', 'encoder_output'):
    assert stage in receipt['reference']['stages']
f16 = Path(hf_hub_download(MODEL, 'nemotron-3.5-asr-streaming-0.6b-f16.gguf', revision=REV))
reader = GGUFReader(str(f16))
source_types = {t.name: t.tensor_type.name for t in reader.tensors}
del reader
# Capture full native and reference arrays before judging the newly exposed
# boundaries. Preserve every sample/frame: no crop, padding or gate change.
class Params(ctypes.Structure):
    _fields_ = [('n_threads', ctypes.c_int), ('use_flash', ctypes.c_bool),
                ('verbosity', ctypes.c_int), ('use_gpu', ctypes.c_bool)]


lib = ctypes.CDLL(str(next(build.rglob('libcrispasr.so'))))
fp = ctypes.POINTER(ctypes.c_float)
ip = ctypes.POINTER(ctypes.c_int)
lib.nemotron_context_default_params.argtypes = []
lib.nemotron_context_default_params.restype = Params
lib.nemotron_init_from_file.argtypes = [ctypes.c_char_p, Params]
lib.nemotron_init_from_file.restype = ctypes.c_void_p
lib.nemotron_free.argtypes = [ctypes.c_void_p]
lib.nemotron_free.restype = None
lib.nemotron_compute_mel.argtypes = [ctypes.c_void_p, fp, ctypes.c_int, ip, ip]
lib.nemotron_compute_mel.restype = fp
for name in ('nemotron_run_preencode_ext', 'nemotron_run_encoder_ext'):
    fn = getattr(lib, name)
    fn.argtypes = [ctypes.c_void_p, fp, ctypes.c_int, ctypes.c_int, ip, ip]
    fn.restype = fp
allocator = ctypes.CDLL(None)
allocator.free.argtypes = [ctypes.c_void_p]
allocator.free.restype = None
with wave.open(str(ROOT / entry['sample']), 'rb') as wav:
    assert wav.getframerate() == 16000 and wav.getnchannels() == 1 and wav.getsampwidth() == 2
    pcm = np.frombuffer(wav.readframes(wav.getnframes()), dtype='<i2').astype(np.float32) / 32768
params = lib.nemotron_context_default_params()
params.n_threads, params.use_flash, params.verbosity, params.use_gpu = 4, False, 0, False
ctx = lib.nemotron_init_from_file(str(f16).encode(), params)
assert ctx
arrays = {}
try:
    nm, tm = ctypes.c_int(), ctypes.c_int()
    mel = lib.nemotron_compute_mel(ctx, pcm.ctypes.data_as(fp), len(pcm), ctypes.byref(nm), ctypes.byref(tm))
    assert mel
    try:
        arrays['native_mel_spectrogram'] = np.ctypeslib.as_array(mel, shape=(tm.value * nm.value,)).copy().reshape(tm.value, nm.value).T.copy()
        for stage, api_name in [('pre_encode_output', 'nemotron_run_preencode_ext'),
                                ('encoder_output', 'nemotron_run_encoder_ext')]:
            te, dm = ctypes.c_int(), ctypes.c_int()
            output = getattr(lib, api_name)(ctx, mel, nm.value, tm.value, ctypes.byref(te), ctypes.byref(dm))
            assert output
            try:
                arrays['native_' + stage] = np.ctypeslib.as_array(output, shape=(te.value * dm.value,)).copy().reshape(te.value, dm.value)
            finally:
                allocator.free(output)
    finally:
        allocator.free(mel)
finally:
    lib.nemotron_free(ctx)
reference = GGUFReader(ref)
for tensor in reference.tensors:
    arrays['reference_' + tensor.name] = np.asarray(tensor.data).copy().reshape(tuple(reversed(tensor.shape)))
del reference
np.savez_compressed(OUT / 'f16-boundary-arrays.npz', **arrays)
receipt['boundary_capture'] = dict(path='f16-boundary-arrays.npz', sha256=sha(OUT / 'f16-boundary-arrays.npz'),
    pcm_sha256=hashlib.sha256(pcm.tobytes()).hexdigest(), layout='Full Python logical shapes; no frame removal', stages={})
for stage in ('mel_spectrogram', 'pre_encode_output', 'encoder_output'):
    native, target = arrays['native_' + stage], arrays['reference_' + stage]
    assert native.shape == target.shape
    error = np.abs(native.astype(np.float64) - target)
    frame_error = error.max(axis=0 if stage == 'mel_spectrogram' else 1)
    receipt['boundary_capture']['stages'][stage] = dict(shape=list(native.shape), max_abs=float(error.max()),
        rms=float(np.sqrt(np.mean(error**2))), per_frame_max_abs=frame_error.tolist(),
        worst_index=list(map(int, np.unravel_index(error.argmax(), error.shape))))
save()
head = [r'^joint\.', r'^decoder\.', r'^prompt_kernel\.']
specs = [('f16', None), ('plain-q4', []), ('rnnt-prompt-q4', head),
         ('rnnt-prompt-preout-q4', head + [r'^encoder\.pre\.out\.']),
         ('rnnt-prompt-preall-q4', head + [r'^encoder\.pre\.'])]
baseline = None
for name, patterns in specs:
    path = f16 if patterns is None else TEMP / (name + '.gguf')
    protected = []
    if patterns is not None:
        protected = [key for key in source_types if any(re.search(p, key) for p in patterns)]
        cmd = [build / 'bin/crispasr-quantize', f16, path, 'q4_k']
        for key in protected:
            assert source_types[key] in ('F16', 'F32')
            cmd.extend(['--tensor-type', '^' + re.escape(key) + '$=' + source_types[key].lower()])
        assert run(cmd, name + '-quantize') == 0
    reader = GGUFReader(str(path))
    tensors = {t.name: dict(type=t.tensor_type.name, shape=list(map(int, t.shape)), bytes=int(t.n_bytes),
                           sha256=hashlib.sha256(t.data.tobytes()).hexdigest()) for t in reader.tensors}
    del reader
    for key in protected:
        assert tensors[key]['type'] == source_types[key]
    if name == 'plain-q4':
        baseline = tensors
    if patterns:
        assert all(tensors[k] == v for k, v in baseline.items() if k not in protected)
    result = dict(bytes=path.stat().st_size, sha256=sha(path), protected=protected, tensors=tensors)
    receipt['models'][name] = result
    result['returncode'] = run([build / 'bin/crispasr-diff', 'nemotron', path, ref,
                               ROOT / entry['sample']], name + '-diff')
    log = (OUT / (name + '-diff.log')).read_text()
    result['stages'] = {stage: re.findall(r'^\[(?:PASS|FAIL|ERR )\].*' + stage + r'.*$', log, re.M)
                        for stage in ('mel_spectrogram', 'pre_encode_output', 'encoder_output')}
    save()
    print(name, result['returncode'], result['stages'], flush=True)
    # Stop if the independently pinned F16 reference control fails. Never
    # change layout/shape, pad or relax thresholds to make Q4 look accepted.
    if name == 'f16':
        assert result['returncode'] == 0, 'F16 must pass all three unchanged stage gates first'
        assert all(len(rows) == 1 and rows[0].startswith('[PASS]') for rows in result['stages'].values())
    if patterns is not None:
        path.unlink()
receipt['diagnostic_complete'] = True
save()
(OUT / 'summary.md').write_text('Boundary diagnosis complete; no model/default promotion.\n'
    'Three actual reference stages, exact shapes, original thresholds and per-tensor byte audits retained.\n')
