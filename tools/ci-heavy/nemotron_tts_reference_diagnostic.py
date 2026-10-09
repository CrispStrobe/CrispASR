#!/usr/bin/env python3
"""Original NVIDIA ASR vs native F16/Q4 on immutable TTS PCM; diagnostic only."""
import gc
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import zipfile

import numpy as np
from scipy.signal import resample_poly
from huggingface_hub import HfApi, hf_hub_download, snapshot_download

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(os.environ['HEAVY_OUT'])
TEMP = Path(os.environ['HEAVY_SCRATCH']) / 'nemotron-tts-reference'
OUT.mkdir(parents=True, exist_ok=True)
TEMP.mkdir(parents=True, exist_ok=True)
MODEL = 'nvidia/nemotron-3.5-asr-streaming-0.6b'
MODEL_REV = 'ea30d66debe3740a08b573244286791d423d6b3e'
GGUF_REV = 'bbd95a9ca5fa0dfca3312a122dfc45a2b578b9c2'
FRONTDOOR = ('tts-asr/frontdoor-diagnostic-20261009/proof.zip',
             '0b434ca1ea6c34c9b462be4d3e689ba1713381e9',
             '3dd2f988e327dd30bf3ae608edc7611e52aedf3a8374a3ea6e2f288342110b98')
receipt = dict(passed=False, diagnostic_complete=False, scope=__doc__,
    source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
    original_model=dict(repo=MODEL, revision=MODEL_REV), native_models={},
    fixed_audio_proof=FRONTDOOR, cases={}, original={}, native={}, state_comparisons={})


def save():
    (OUT / 'nemotron-tts-reference.json').write_text(json.dumps(receipt, indent=2) + '\n')


def digest(path):
    sha = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            sha.update(block)
    return sha.hexdigest()


def download_proof(name, revision, sha, dest):
    path = hf_hub_download('cstr/crispasr-regression-fixtures', name, revision=revision)
    assert digest(path) == sha
    with zipfile.ZipFile(path) as archive:
        archive.extractall(dest)


def run(command, tag):
    with (OUT / (tag + '.log')).open('w') as log:
        subprocess.run(list(map(str, command)), cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                       check=True, timeout=7200)


def wer(expected, actual):
    words = lambda text: re.findall('[a-z]+', re.sub(r'<[^>]*>', '', text).lower())
    ref, hyp = words(expected), words(actual)
    row = list(range(len(hyp) + 1))
    for i, word in enumerate(ref, 1):
        nxt = [i]
        for j, candidate in enumerate(hyp, 1):
            nxt.append(min(nxt[-1] + 1, row[j] + 1, row[j - 1] + (word != candidate)))
        row = nxt
    return row[-1] / len(ref)


save()
front = TEMP / 'frontdoor'
download_proof(*FRONTDOOR, front)
prior = json.loads(next(front.rglob('frontdoor-diagnostic.json')).read_text())
assert prior['diagnostic_complete'] and len(prior['transcripts']) == 234
cases = {}
for cohort, proof in prior['proofs'].items():
    dest = TEMP / cohort
    download_proof(*proof, dest)
    for p in sorted(dest.rglob('*.npy')):
        if p.name.endswith('-latent.npy'):
            continue
        label = cohort + ':' + str(p.relative_to(dest).with_suffix('')).replace('/', ':')
        old = prior['cases'][label]
        pcm = np.load(p, allow_pickle=False).astype(np.float32).reshape(-1)
        assert hashlib.sha256(pcm.tobytes()).hexdigest() == old['source_sha256']
        rate = old['sample_rate']
        common = math.gcd(rate, 16000)
        signal = resample_poly(pcm, 16000 // common, rate // common).astype(np.float32)
        sha = hashlib.sha256(signal.tobytes()).hexdigest()
        assert sha == old['resampled']['scipy-polyphase']['sha256']
        name = f'audio-{len(cases):02d}'
        np.save(OUT / (name + '-16k.npy'), signal)
        cases[label] = (signal, old['expected'], name)
        receipt['cases'][label] = dict(samples=len(signal), sample_rate=16000,
            sha256=sha, expected=old['expected'], original_case=old, file=name + '-16k.npy')
assert len(cases) == 26
save()

# The original model-card offline API, fixed F32 CPU arithmetic and pinned
# official package/model. No model-weight dequantization masquerades as gold.
import torch
from transformers import AutoModelForRNNT, AutoProcessor
assert importlib.metadata.version('transformers') == '5.19.0'
torch.set_num_threads(4)
torch.set_num_interop_threads(1)
receipt['packages'] = {name: importlib.metadata.version(name)
    for name in ('torch', 'transformers', 'numpy', 'scipy', 'huggingface_hub')}
model_dir = snapshot_download(MODEL, revision=MODEL_REV,
    allow_patterns=['*.json', '*.safetensors'])
weight = Path(model_dir) / 'model.safetensors'
assert digest(weight) == '9eebdd6590289cb3030f310858f3df93256600a800a3e8200c5993d5f967e174'
receipt['original_model'].update(file='model.safetensors', sha256=digest(weight), bytes=weight.stat().st_size)
processor = AutoProcessor.from_pretrained(model_dir, local_files_only=True)
model = AutoModelForRNNT.from_pretrained(model_dir, local_files_only=True, torch_dtype=torch.float32)
model.eval()
assert next(model.parameters()).device.type == 'cpu'
processor.set_num_lookahead_tokens(3)  # native context preset 0: left56/right3
receipt['original_model'].update(dtype=str(next(model.parameters()).dtype),
    device='cpu', lookahead_tokens=3, language='en-US', generation='model-card offline generate',
    captured_encoder_scope='stock encoder before language prompt fusion/projector')
# Capture actual original encoder output before RNNT decoding when exposed by
# the stock model, retaining full features and sequences for later stage diffs.
encoders = [(name, module) for name, module in model.named_modules()
            if name == 'encoder' or name == 'model.encoder']
assert len(encoders) == 1, [name for name, _ in encoders]
current = {}


def capture_encoder(module, args, output):
    tensor = getattr(output, 'last_hidden_state', None)
    if tensor is None:
        tensor = output[0] if isinstance(output, tuple) else output
    assert isinstance(tensor, torch.Tensor)
    array = tensor.detach().cpu().float().numpy()
    assert np.isfinite(array).all()
    np.save(OUT / (current['name'] + '-original-encoder.npy'), array)
    current['encoder_shape'] = list(array.shape)


hook = encoders[0][1].register_forward_hook(capture_encoder)
with torch.inference_mode():
    for label, (signal, expected, name) in cases.items():
        current.clear()
        current['name'] = name
        inputs = processor(signal, sampling_rate=16000, language='en-US', return_tensors='pt')
        inputs = inputs.to(model.device, dtype=model.dtype)
        for key, value in inputs.items():
            if isinstance(value, torch.Tensor):
                np.save(OUT / (name + '-original-input-' + key + '.npy'), value.cpu().numpy())
        start = time.perf_counter()
        output = model.generate(**inputs, return_dict_in_generate=True, max_new_tokens=512)
        seconds = time.perf_counter() - start
        sequence = output.sequences.cpu().numpy()
        np.save(OUT / (name + '-original-sequence.npy'), sequence)
        text = processor.batch_decode(output.sequences, skip_special_tokens=True)[0]
        assert isinstance(text, str) and sequence.size < 512
        receipt['original'][label] = dict(transcript=text, wer=wer(expected, text),
            seconds=seconds, encoder_shape=current.get('encoder_shape'),
            sequence_sha256=hashlib.sha256(sequence.tobytes()).hexdigest())
        save()
        print('ORIGINAL', label, repr(text), flush=True)
hook.remove()
del model, processor, inputs, output
gc.collect()

# Native controls use the identical archived SciPy-resampled PCM, not a file
# loader or newly generated voice. Reverse reused order to detect session-state
# leakage separately from model quantization or the original blueprint.
build = TEMP / 'build'
run(['cmake', '-S', ROOT, '-B', build, '-G', 'Ninja', '-DCMAKE_BUILD_TYPE=Release',
     '-DBUILD_SHARED_LIBS=ON', '-DGGML_NATIVE=OFF', '-DGGML_CUDA=OFF', '-DGGML_BLAS=OFF',
     '-DCRISPASR_BUILD_TESTS=OFF', '-DCRISPASR_BUILD_SERVER=OFF'], 'configure')
run(['cmake', '--build', build, '--target', 'crispasr-lib', '-j4'], 'build')
lib_path = next(build.rglob('libcrispasr.so'))
sys.path.insert(0, str(ROOT / 'python'))
from crispasr import Session
os.environ['CRISPASR_NEMOTRON_CONTEXT_PRESET'] = '0'
api = HfApi()
for quant in ('f16', 'q4_k'):
    filename = 'nemotron-3.5-asr-streaming-0.6b-' + quant + '.gguf'
    path = hf_hub_download('cstr/nemotron-3.5-asr-streaming-GGUF', filename, revision=GGUF_REV)
    sha = digest(path)
    remote = next(f for f in api.model_info('cstr/nemotron-3.5-asr-streaming-GGUF',
        revision=GGUF_REV, files_metadata=True).siblings if f.rfilename == filename)
    assert remote.lfs.sha256 == sha and remote.size == Path(path).stat().st_size
    receipt['native_models'][quant] = dict(repo='cstr/nemotron-3.5-asr-streaming-GGUF',
        revision=GGUF_REV, file=filename, sha256=sha, bytes=remote.size)
    for label, (signal, expected, name) in cases.items():
        with Session(path, lib_path=str(lib_path), backend='nemotron', n_threads=4) as session:
            text = ' '.join(seg.text for seg in session.transcribe(signal, sample_rate=16000, language='en'))
        receipt['native'][quant + '/fresh/' + label] = dict(transcript=text, wer=wer(expected, text))
        save()
    with Session(path, lib_path=str(lib_path), backend='nemotron', n_threads=4) as session:
        for label, (signal, expected, name) in reversed(list(cases.items())):
            text = ' '.join(seg.text for seg in session.transcribe(signal, sample_rate=16000, language='en'))
            receipt['native'][quant + '/reused/' + label] = dict(transcript=text, wer=wer(expected, text))
            receipt['state_comparisons'][quant + '/' + label] = dict(
                decoded_equal=receipt['native'][quant + '/fresh/' + label]['transcript'] == text)
            save()
receipt['diagnostic_complete'] = True
save()
(OUT / 'summary.md').write_text('Completed original NVIDIA and native F16/Q4 fixed-audio ASR diagnosis. '
    'This captures evidence; it does not promote a recognizer, relax speech gates or certify TTS.\n')
print('NEMOTRON_TTS_REFERENCE_DIAGNOSTIC_COMPLETE', flush=True)
