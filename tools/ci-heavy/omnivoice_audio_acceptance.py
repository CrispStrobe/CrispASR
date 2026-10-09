#!/usr/bin/env python3
"""Pinned utility oracle, real CLI/session synthesis, and TTS->ASR acceptance."""
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import wave
import numpy as np
from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(os.environ['HEAVY_OUT'])
TEMP = Path(os.environ['HEAVY_SCRATCH']) / 'omnivoice-518'
OUT.mkdir(parents=True, exist_ok=True)
TEMP.mkdir(parents=True, exist_ok=True)
os.environ['TMPDIR'] = str(TEMP)
MODEL_PIN = '92266739c4496b04a497e164cf8c2801a792551e'
FIXTURE_PIN = 'd0a7d7a8be318a5841dfdbe6ad37d3acf75523e3'
TEXT = 'The quick brown fox jumps over the lazy dog.'


def run(command, tag, env=None):
    with (OUT / (tag + '.log')).open('w') as log:
        result = subprocess.run(list(map(str,command)), cwd=ROOT, env=env, stdout=log,
                                stderr=subprocess.STDOUT, timeout=3600)
    print(tag, result.returncode, (OUT/(tag+'.log')).read_text()[-2000:], flush=True)
    assert result.returncode == 0, tag


run([sys.executable, ROOT/'tools/ci-heavy/omnivoice_audio_parity.py'], 'utility-oracle')
build = TEMP/'build'
run(['cmake','-S',ROOT,'-B',build,'-G','Ninja','-DCMAKE_BUILD_TYPE=Release',
     '-DBUILD_SHARED_LIBS=ON','-DGGML_NATIVE=OFF','-DGGML_CUDA=OFF',
     '-DCRISPASR_BUILD_TESTS=OFF','-DCRISPASR_BUILD_SERVER=OFF'], 'configure')
run(['cmake','--build',build,'--target','crispasr-cli','crispasr-lib','-j4'], 'build')
model_dir = TEMP/'models'
paths = {name: hf_hub_download('cstr/omnivoice-GGUF',name,revision=MODEL_PIN,local_dir=model_dir)
         for name in ('omnivoice-q8_0.gguf','omnivoice-tokenizer-f16.gguf')}
manifest_path = hf_hub_download('cstr/crispasr-regression-fixtures',
    'index-echo-9b/roundtrip-piper/roundtrip-audio.json', revision=FIXTURE_PIN)
manifest = json.loads(Path(manifest_path).read_text())
reference = manifest['cases']['fox']
ref_wav = hf_hub_download('cstr/crispasr-regression-fixtures',
    'index-echo-9b/roundtrip-piper/'+reference['audio'],revision=FIXTURE_PIN)
assert hashlib.sha256(Path(ref_wav).read_bytes()).hexdigest() == reference['sha256']
sys.path.insert(0,str(ROOT/'python'))
from crispasr import Session
lib = next(build.rglob('libcrispasr.so'))
outputs = {}
# A no-reference raw/full pair holds generated codes fixed, directly checking
# the actual decode funnel against the independent upstream utilities.
for legacy in (True,False):
    label = 'session-raw' if legacy else 'session-full'
    os.environ['CRISPASR_OMNIVOICE_AUDIO_LEGACY'] = '1' if legacy else '0'
    os.environ['CRISPASR_OMNIVOICE_DUMP_CODES'] = str(OUT/(label+'.codes'))
    with Session(paths['omnivoice-q8_0.gguf'],lib_path=str(lib),backend='omnivoice',n_threads=4) as s:
        s.set_codec_path(paths['omnivoice-tokenizer-f16.gguf'])
        s.set_tts_seed(42)
        assert s._lib.crispasr_session_set_tts_seed(s._handle,42) == 0
        s.set_tts_steps(32)
        assert s._lib.crispasr_session_set_tts_steps(s._handle,32) == 0
        # Compare native DSP before the default API adds its watermark.
        s.accept_marking_responsibility('Independent upstream waveform acceptance fixture')
        pcm = s.synthesize_raw(TEXT)
        assert s.output_sample_rate() == 24000
        assert np.isfinite(pcm).all() and len(pcm) > 24000
        outputs[label] = np.asarray(pcm,dtype=np.float32)
        np.save(OUT/(label+'.npy'),pcm)
        if not legacy:
            s.set_tts_seed(42)
            outputs['session-marked'] = np.asarray(s.synthesize(TEXT),dtype=np.float32)
            np.save(OUT/'session-marked.npy',outputs['session-marked'])
assert (OUT/'session-raw.codes').read_bytes() == (OUT/'session-full.codes').read_bytes(), 'Generation changed in decode-only A/B'
# Load actual upstream utility AST functions through the independent module.
import runpy
oracle = runpy.run_path(str(ROOT/'tools/ci-heavy/omnivoice_audio_parity.py'))['namespace']
expected = oracle['remove_silence'](outputs['session-raw'][None,:],24000,500,100,100)
peak = np.max(np.abs(expected)) if expected.size else 0
if peak > 1e-6:
    expected *= .5/peak
expected = oracle['fade_and_pad_audio'](expected).ravel()
assert expected.shape == outputs['session-full'].shape
error = float(np.max(np.abs(expected-outputs['session-full'])))
(OUT/'waveform-oracle.json').write_text(json.dumps(dict(max_abs=error,
    compared_surface='attested raw C ABI; upstream DSP before automatic watermark',
    marked_output_differs=bool(not np.array_equal(outputs['session-marked'],outputs['session-full']))),indent=2)+'\n')
assert error <= 2e-6, error
assert not np.array_equal(outputs['session-marked'],outputs['session-full']), 'Default watermark not exercised'
# Reference preprocessing must run through both real user surfaces.
os.environ['CRISPASR_OMNIVOICE_AUDIO_LEGACY'] = '0'
os.environ.pop('CRISPASR_OMNIVOICE_DUMP_CODES',None)
with Session(paths['omnivoice-q8_0.gguf'],lib_path=str(lib),backend='omnivoice',n_threads=4) as s:
    s.set_codec_path(paths['omnivoice-tokenizer-f16.gguf'])
    s.set_tts_seed(42)
    s.set_tts_steps(32)
    s.set_voice(ref_wav,ref_text=reference['text'].rstrip('.'))
    s.accept_marking_responsibility('Reference DSP acceptance fixture')
    outputs['session-clone'] = np.asarray(s.synthesize_raw(TEXT),dtype=np.float32)
    s.set_tts_seed(42)
    outputs['session-marked-clone'] = np.asarray(s.synthesize(TEXT),dtype=np.float32)
    np.save(OUT/'session-clone.npy',outputs['session-clone'])
cli_wav = OUT/'cli-clone.wav'
run([build/'bin/crispasr','--backend','omnivoice','-m',paths['omnivoice-q8_0.gguf'],
     '--codec-model',paths['omnivoice-tokenizer-f16.gguf'],'--no-gpu','-t','4',
     '--seed','42','--tts-steps','32','--voice',ref_wav,'--ref-text',reference['text'].rstrip('.'),
     '--tts',TEXT,'--tts-output',cli_wav,'--no-watermark','--no-spoken-disclaimer',
     '--accept-marking-responsibility'], 'cli-clone')
with wave.open(str(cli_wav)) as wav:
    assert wav.getframerate() == 24000 and wav.getnchannels() == 1 and wav.getsampwidth() == 2
    outputs['cli-clone'] = np.frombuffer(wav.readframes(wav.getnframes()),dtype='<i2').astype(np.float32)/32768
asr = hf_hub_download('cstr/nemotron-3.5-asr-streaming-GGUF',
    'nemotron-3.5-asr-streaming-0.6b-q4_k.gguf',revision='bbd95a9ca5fa0dfca3312a122dfc45a2b578b9c2')
results = {}
with Session(asr,lib_path=str(lib),backend='nemotron',n_threads=4) as s:
    for label,pcm in outputs.items():
        assert np.isfinite(pcm).all() and len(pcm) > 24000, label
        if label not in ('session-raw','session-marked','session-marked-clone'):
            assert np.max(np.abs(pcm[:2400])) <= 1/32768 and np.max(np.abs(pcm[-2400:])) <= 1/32768, label
        actual = ' '.join(segment.text for segment in s.transcribe(pcm,sample_rate=24000,language='en'))
        words = lambda text: re.findall('[a-z]+',re.sub(r'<[^>]*>','',text).lower())
        ref,hyp = words(TEXT),words(actual)
        row = list(range(len(hyp)+1))
        for i,w in enumerate(ref,1):
            new = [i]
            for j,v in enumerate(hyp,1):
                new.append(min(new[-1]+1,row[j]+1,row[j-1]+(w!=v)))
            row = new
        results[label] = dict(transcript=actual,wer=row[-1]/len(ref),samples=len(pcm),
                              sha256=hashlib.sha256(pcm.tobytes()).hexdigest())
        (OUT/'roundtrips.json').write_text(json.dumps(results,indent=2)+'\n')
assert len(results) == 6 and all(case['wer'] <= .2 for case in results.values()), results
receipt = dict(passed=True,model_revision=MODEL_PIN,fixture_revision=FIXTURE_PIN,roundtrips=results,
    decode_waveform_max_abs=error,decode_codes_exact=True,
    source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip())
(OUT/'omnivoice-518-acceptance.json').write_text(json.dumps(receipt,indent=2)+'\n')
(OUT/'summary.md').write_text('OmniVoice independent utility/decode oracle, CLI/session cloning and four ASR roundtrips PASS.\n')
