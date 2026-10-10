#!/usr/bin/env python3
"""Original OmniVoice generation control; no native acceptance or seed search."""
import gc
import hashlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import random
import re
import subprocess
import types
import wave

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(os.environ['HEAVY_OUT'])
TEMP = Path(os.environ['HEAVY_SCRATCH']) / 'omnivoice-original-clone'
OUT.mkdir(parents=True, exist_ok=True)
TEMP.mkdir(parents=True, exist_ok=True)
os.environ.update(HF_HOME=str(TEMP / 'hf'), HF_XET_CACHE=str(TEMP / 'xet'), OMP_NUM_THREADS='4')
import numpy as np
import torch
import torchaudio
from huggingface_hub import hf_hub_download, snapshot_download
from omnivoice import OmniVoice
from transformers import AutoModelForRNNT, AutoProcessor

UPSTREAM = '08be0b4ccbac3e13e374e86fbfead4b4cac343e2'
MODEL_REV = 'c5fdb5ccb189668d56333f77ba2629f4cd7535f4'
FIXTURE_REV = 'd0a7d7a8be318a5841dfdbe6ad37d3acf75523e3'
ASR_REV = 'ea30d66debe3740a08b573244286791d423d6b3e'
TEXT = 'The quick brown fox jumps over the lazy dog.'
torch.set_num_threads(4)
torch.set_num_interop_threads(1)
assert importlib.metadata.version('transformers') == '5.19.0'
receipt = dict(diagnostic_complete=False, native_accepted=False, defaults_changed=False,
    source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
    upstream_revision=UPSTREAM, model_revision=MODEL_REV, fixture_revision=FIXTURE_REV,
    asr_revision=ASR_REV, dtype='torch.float32', device='cpu', seed=42, steps=32,
    text=TEXT, arms={}, packages={name: importlib.metadata.version(name) for name in
        ['omnivoice', 'torch', 'torchaudio', 'transformers', 'numpy', 'huggingface_hub']})


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def save():
    (OUT / 'original-clone.json').write_text(json.dumps(receipt, indent=2) + '\n')


def seed():
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)


def words(text):
    return re.findall('[a-z]+', re.sub(r'<[^>]*>', '', text).lower())


def wer(actual):
    ref, hyp = words(TEXT), words(actual)
    row = list(range(len(hyp) + 1))
    for i, token in enumerate(ref, 1):
        new = [i]
        for j, other in enumerate(hyp, 1):
            new.append(min(new[-1] + 1, row[j] + 1, row[j-1] + (token != other)))
        row = new
    return row[-1] / len(ref)


save()
manifest = hf_hub_download('cstr/crispasr-regression-fixtures',
    'index-echo-9b/roundtrip-piper/roundtrip-audio.json', revision=FIXTURE_REV)
reference = json.loads(Path(manifest).read_text())['cases']['fox']
ref_wav = hf_hub_download('cstr/crispasr-regression-fixtures',
    'index-echo-9b/roundtrip-piper/' + reference['audio'], revision=FIXTURE_REV)
assert sha(ref_wav) == reference['sha256']
receipt['reference'] = reference
model_dir = snapshot_download('k2-fsa/OmniVoice', revision=MODEL_REV,
    allow_patterns=['*.json', '*.safetensors', '*.jinja', 'audio_tokenizer/*.json', 'audio_tokenizer/*.safetensors'])
receipt['model_files'] = {str(p.relative_to(model_dir)): dict(bytes=p.stat().st_size, sha256=sha(p))
    for p in Path(model_dir).rglob('*') if p.is_file()}
model = OmniVoice.from_pretrained(model_dir, device_map='cpu', dtype=torch.float32).eval()
assert model.device.type == 'cpu' and model.dtype == torch.float32
assert model.sampling_rate == 24000
(OUT / 'omnivoice-blueprint.py').write_text(inspect.getsource(type(model)))
receipt['blueprint_sha256'] = sha(OUT / 'omnivoice-blueprint.py')
# The original unprocessed arm is a control, not a claim that it duplicates
# the native legacy RMS-restore path. Do not tune duration/language or seeds.
outputs = {}
for label, preprocess, postprocess, ref_text in [
        ('unprocessed', False, False, reference['text'].rstrip('.')),
        ('full-cleanup', True, True, reference['text'])]:
    kwargs = dict(text=TEXT, ref_audio=str(ref_wav), ref_text=ref_text,
        num_step=32, preprocess_prompt=preprocess, postprocess_output=postprocess)
    seed()
    with torch.inference_mode():
        control = np.asarray(model.generate(**kwargs)[0], dtype=np.float32)
    captured = []
    original = model._decode_and_post_process

    def capture(self, tokens, ref_rms, config):
        captured.append(tokens.detach().clone().cpu().numpy())
        return original(tokens, ref_rms, config)

    model._decode_and_post_process = types.MethodType(capture, model)
    try:
        seed()
        with torch.inference_mode():
            pcm = np.asarray(model.generate(**kwargs)[0], dtype=np.float32)
    finally:
        model._decode_and_post_process = original
    assert len(captured) == 1 and np.array_equal(control, pcm), 'Capture changed original PCM'
    assert len(pcm) > 24000 and np.isfinite(pcm).all()
    np.save(OUT / (label + '.npy'), pcm)
    np.save(OUT / (label + '-codes.npy'), captured[0])
    outputs[label] = pcm
    receipt['arms'][label] = dict(preprocess_prompt=preprocess, postprocess_output=postprocess,
        ref_text=ref_text, capture_pcm_exact=True, samples=len(pcm),
        pcm_sha256=hashlib.sha256(pcm.tobytes()).hexdigest(), codes_shape=list(captured[0].shape),
        codes_sha256=hashlib.sha256(captured[0].tobytes()).hexdigest())
    save()
    print('ORIGINAL_GENERATION_COMPLETE', label, len(pcm), flush=True)
# Release the TTS weights before loading the independent original ASR model.
del original, capture, model, captured
gc.collect()
asr_dir = snapshot_download('nvidia/nemotron-3.5-asr-streaming-0.6b', revision=ASR_REV,
    allow_patterns=['*.json', '*.safetensors'])
assert sha(Path(asr_dir) / 'model.safetensors') == '9eebdd6590289cb3030f310858f3df93256600a800a3e8200c5993d5f967e174'
processor = AutoProcessor.from_pretrained(asr_dir, local_files_only=True)
asr = AutoModelForRNNT.from_pretrained(asr_dir, local_files_only=True, torch_dtype=torch.float32).eval()
processor.set_num_lookahead_tokens(3)
assert asr.device.type == 'cpu' and asr.dtype == torch.float32
with wave.open(str(ROOT / 'samples/jfk.wav')) as wav:
    assert wav.getframerate() == 16000 and wav.getnchannels() == 1 and wav.getsampwidth() == 2
    jfk = np.frombuffer(wav.readframes(wav.getnframes()), dtype='<i2').astype(np.float32) / 32768


def transcribe(pcm):
    inputs = processor(pcm, sampling_rate=16000, language='en-US', return_tensors='pt').to(asr.device, dtype=asr.dtype)
    with torch.inference_mode():
        output = asr.generate(**inputs, return_dict_in_generate=True, max_new_tokens=512)
    return processor.batch_decode(output.sequences, skip_special_tokens=True)[0]


receipt['asr_control'] = transcribe(jfk)
assert 'ask not what your country can do for you' in ' '.join(words(receipt['asr_control']))
for label, pcm in outputs.items():
    signal = torchaudio.functional.resample(torch.from_numpy(pcm), orig_freq=24000, new_freq=16000).numpy()
    np.save(OUT / (label + '-asr-input.npy'), signal)
    actual = transcribe(signal)
    receipt['arms'][label].update(transcript=actual, wer=wer(actual),
        meets_existing_gate=wer(actual) <= .2,
        asr_pcm_sha256=hashlib.sha256(signal.tobytes()).hexdigest())
    save()
receipt.update(diagnostic_complete=True, native_waveform_parity_tested=False,
    production_accepted=False, original_generation_control_only=True)
save()
print(json.dumps(receipt, indent=2), flush=True)
