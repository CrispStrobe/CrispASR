#!/usr/bin/env python3
"""Capture native encoder layers with byte-exact capture OFF/ON controls."""
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
TEMP = Path(os.environ['HEAVY_SCRATCH']) / 'nemotron-encoder-layers'
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
    (OUT / 'encoder-layers.json').write_text(json.dumps(receipt, indent=2) + '\n')


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
class Params(ctypes.Structure):
    _fields_ = [('n_threads', ctypes.c_int), ('use_flash', ctypes.c_bool),
                ('verbosity', ctypes.c_int), ('use_gpu', ctypes.c_bool)]
lib = ctypes.CDLL(str(next(build.rglob('libcrispasr.so'))))
fp, ip = ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_int)
lib.nemotron_context_default_params.argtypes = []
lib.nemotron_context_default_params.restype = Params
lib.nemotron_init_from_file.argtypes = [ctypes.c_char_p, Params]
lib.nemotron_init_from_file.restype = ctypes.c_void_p
lib.nemotron_free.argtypes = [ctypes.c_void_p]
lib.nemotron_free.restype = None
lib.nemotron_compute_mel.argtypes = [ctypes.c_void_p, fp, ctypes.c_int, ip, ip]
lib.nemotron_compute_mel.restype = fp
lib.nemotron_run_encoder_ext.argtypes = [ctypes.c_void_p, fp, ctypes.c_int, ctypes.c_int, ip, ip]
lib.nemotron_run_encoder_ext.restype = fp
lib.nemotron_run_encoder_layers_ext.argtypes = [ctypes.c_void_p, fp, ctypes.c_int, ctypes.c_int, ip, ip, ip]
lib.nemotron_run_encoder_layers_ext.restype = fp
allocator = ctypes.CDLL(None)
allocator.free.argtypes = [ctypes.c_void_p]
allocator.free.restype = None
with wave.open(str(ROOT / entry['sample']), 'rb') as wav:
    assert wav.getframerate() == 16000 and wav.getnchannels() == 1 and wav.getsampwidth() == 2
    pcm = np.frombuffer(wav.readframes(wav.getnframes()), dtype='<i2').astype(np.float32) / 32768
params = lib.nemotron_context_default_params()
params.n_threads, params.use_flash, params.verbosity, params.use_gpu = 4, False, 0, False
receipt['pcm_sha256'] = hashlib.sha256(pcm.tobytes()).hexdigest()
source = GGUFReader(str(f16))
source_tensors = {t.name: dict(type=t.tensor_type.name, bytes=int(t.n_bytes),
    sha256=hashlib.sha256(t.data.tobytes()).hexdigest()) for t in source.tensors}
del source
common_guards = [r'^joint\.', r'^decoder\.', r'^prompt_kernel\.', r'^encoder\.pre\.out\.']
specs = [('f16', f16, None), ('plain-q4', TEMP / 'plain-q4.gguf', []),
    ('rnnt-preout-q4', TEMP / 'rnnt-preout-q4.gguf', common_guards),
    ('attention-source-q4', TEMP / 'attention-source-q4.gguf',
        common_guards + [r'^encoder\.layers\.[0-9]+\.attn\.']),
    ('ffn-source-q4', TEMP / 'ffn-source-q4.gguf',
        common_guards + [r'^encoder\.layers\.[0-9]+\.ff[12]\.'])]
all_arrays = {'pcm': pcm}
plain_tensors = None
for name, path, patterns in specs:
    protected = []
    if patterns is not None:
        protected = [key for key in source_types if any(re.search(p, key) for p in patterns)]
        command = [build / 'bin/crispasr-quantize', f16, path, 'q4_k']
        for key in protected:
            assert source_types[key] in ('F16', 'F32')
            command += ['--tensor-type', '^' + re.escape(key) + '$=' + source_types[key].lower()]
        assert run(command, name + '-quantize') == 0
    reader = GGUFReader(str(path))
    tensors = {t.name: dict(type=t.tensor_type.name, bytes=int(t.n_bytes),
        sha256=hashlib.sha256(t.data.tobytes()).hexdigest()) for t in reader.tensors}
    del reader
    for key in protected:
        assert tensors[key] == source_tensors[key], 'Guard must preserve actual source bytes'
    if name == 'plain-q4': plain_tensors = tensors
    if patterns:
        assert all(tensors[k] == v for k, v in plain_tensors.items() if k not in protected)
    q4_bytes = sum(t['bytes'] for t in tensors.values() if t['type'].startswith('Q4'))
    if patterns is not None:
        assert q4_bytes > 0, 'Diagnostic candidate must retain real Q4 tensors'
    result = dict(bytes=path.stat().st_size, q4_bytes=q4_bytes, sha256=sha(path), protected=protected,
        tensors=tensors, stage_returncode=run([build / 'bin/crispasr-diff', 'nemotron', path, ref,
            ROOT / entry['sample']], name + '-diff'))
    receipt['models'][name] = result
    if name == 'f16': assert result['stage_returncode'] == 0, 'F16 original control failed'
    ctx = lib.nemotron_init_from_file(str(path).encode(), params)
    assert ctx
    try:
        nm, tm = ctypes.c_int(), ctypes.c_int()
        mel = lib.nemotron_compute_mel(ctx, pcm.ctypes.data_as(fp), len(pcm), ctypes.byref(nm), ctypes.byref(tm))
        assert mel
        try:
            encs = []
            for capture in [False, True, False]:
                te, dm, nl = ctypes.c_int(), ctypes.c_int(), ctypes.c_int()
                if capture:
                    data = lib.nemotron_run_encoder_layers_ext(ctx, mel, nm.value, tm.value,
                        ctypes.byref(nl), ctypes.byref(te), ctypes.byref(dm))
                    assert nl.value == 24
                    shape = (nl.value, te.value, dm.value)
                else:
                    data = lib.nemotron_run_encoder_ext(ctx, mel, nm.value, tm.value, ctypes.byref(te), ctypes.byref(dm))
                    shape = (te.value, dm.value)
                assert data
                try: array = np.ctypeslib.as_array(data, shape=(int(np.prod(shape)),)).copy().reshape(shape)
                finally: allocator.free(data)
                assert np.isfinite(array).all()
                if capture:
                    all_arrays[name + '-layers'] = array
                    encs.append(array[-1].copy())
                else: encs.append(array)
            assert np.array_equal(encs[0], encs[1]) and np.array_equal(encs[0], encs[2]), 'Capture OFF/ON/OFF changed final encoder bytes'
            result['capture_final_exact'] = True
            result['shape'] = list(all_arrays[name + '-layers'].shape)
            save()
        finally: allocator.free(mel)
    finally: lib.nemotron_free(ctx)
    if patterns is not None: path.unlink()
archive = OUT / 'native-encoder-layers.npz'
np.savez_compressed(archive, **all_arrays)
receipt.update(diagnostic_complete=True, archive_sha256=sha(archive),
    original_layer_parity_tested=False, production_accepted=False)
save()
