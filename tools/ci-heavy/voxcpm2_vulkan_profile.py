#!/usr/bin/env python3
"""Ten-step Vulkan cold/warm profile with repeated PCM and actual ASR gates."""
import ctypes as C
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import zipfile
import numpy as np
import soundfile as sf
from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(os.environ['HEAVY_OUT'])
TEMP = Path(os.environ['HEAVY_SCRATCH'])/'voxcpm2-profile'
TEMP.mkdir(parents=True,exist_ok=True)
BUILD = Path(os.environ['VOX_PROFILE_BUILD'])
MODEL_PIN = '25b5cf03fdbf20011dad9a77def6112023cc0fe3'
TEXTS = {'short':'Hello, this is a short test sentence.',
         'long':'The sun is shining today. Hello, this is a short test sentence.'}
PERF_ONLY = '--perf-only' in sys.argv
if PERF_ONLY:
    os.environ['GGML_VK_PERF_LOGGER'] = '1'
    TEXTS = {'short': TEXTS['short']}
os.environ.update(CRISPASR_VOXCPM2_BENCH='1',CRISPASR_VOXCPM2_INFERENCE_STEPS='10')
sys.path.insert(0,str(ROOT/'python'))
from crispasr import Session


class Params(C.Structure):
    _fields_ = [(name,C.c_int) for name in ('abi_version','n_threads','use_gpu','verbosity','flash_attn','n_gpu_layers')] + [('reserved',C.c_int*6)]


def open_session(model,backend,device):
    s = Session.__new__(Session)
    s._lib = C.CDLL(str(next(BUILD.rglob('libcrispasr.so'))))
    s._handle,s._progress_cb_holder = None,None
    s._setup_session_signatures()
    s._lib.crispasr_set_gpu_backend.argtypes = [C.c_char_p]
    s._lib.crispasr_set_gpu_backend(device.encode())
    params = Params(2,4,1,1,1,-1)
    s._handle = s._lib.crispasr_session_open_with_params(os.fsencode(model),backend.encode(),C.byref(params))
    assert s._handle,(model,device)
    s.backend,s._n_threads = backend,4
    return s


receipt = dict(source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
               model_revision=MODEL_PIN,steps=10,seed=2,scope=__doc__,passed=False,
               hardware_scope='NVIDIA Vulkan; cannot certify Intel B390',models={},calls=[],roundtrips={})


def save():
    (OUT/'voxcpm2-vulkan-profile.json').write_text(json.dumps(receipt,indent=2)+'\n')


save()
waveforms = {}
for cohort,name in [('q8','voxcpm2-q8_0.gguf'),('mixed','voxcpm2-q8_0-locdit-f16.gguf')]:
    model = hf_hub_download('cstr/voxcpm2-GGUF',name,revision=MODEL_PIN,local_dir=TEMP/'models')
    with open(model,'rb') as f:
        digest = hashlib.file_digest(f,'sha256').hexdigest()
    receipt['models'][cohort] = dict(file=name,sha256=digest,bytes=Path(model).stat().st_size)
    canonical = {}
    with open_session(model,'voxcpm2','vulkan') as session:
        session.set_tts_steps(10)
        assert session._lib.crispasr_session_set_tts_steps(session._handle,10) == 0
        for rep in range(1 if PERF_ONLY else 8):
            for key,text in TEXTS.items():
                session.set_tts_seed(2)
                assert session._lib.crispasr_session_set_tts_seed(session._handle,2) == 0
                print('VOX_CALL_BEGIN',cohort,key,rep,flush=True)
                started = time.perf_counter()
                pcm = np.asarray(session.synthesize(text),dtype=np.float32)
                elapsed = time.perf_counter()-started
                print('VOX_CALL_END',cohort,key,rep,elapsed,flush=True)
                assert session.output_sample_rate() == 48000
                assert len(pcm)>24000 and np.isfinite(pcm).all() and np.max(np.abs(pcm))>.01
                digest = hashlib.sha256(pcm.tobytes()).hexdigest()
                canonical.setdefault(key,digest)
                assert digest == canonical[key], (cohort,key,rep,'Repeated seeded PCM drift')
                receipt['calls'].append(dict(cohort=cohort,text=key,repetition=rep,cold=rep==0 and key=='short', first_shape=rep==0,
                    measured=rep>=2,seconds=elapsed,samples=len(pcm),audio_seconds=len(pcm)/48000,
                    rtf=elapsed/(len(pcm)/48000),sha256=digest))
                if key not in waveforms.get(cohort,{}):
                    waveforms.setdefault(cohort,{})[key] = pcm
                    np.save(OUT/(cohort+'-'+key+'.npy'),pcm)
                save()
# Actual CLI default: remove the environment override, so the ten-step native
# default and the explicit CLI sentinel are exercised independently of session.
if not PERF_ONLY:
    for cohort,name in [('q8','voxcpm2-q8_0.gguf'),('mixed','voxcpm2-q8_0-locdit-f16.gguf')]:
        wav = OUT/(cohort+'-cli-short.wav')
        env = dict(os.environ)
        env.pop('CRISPASR_VOXCPM2_INFERENCE_STEPS',None)
        with (OUT/(cohort+'-cli.log')).open('w') as log:
            subprocess.run([str(BUILD/'bin/crispasr'),'--backend','voxcpm2-tts','-m',str(TEMP/'models'/name),
                '--tts',TEXTS['short'],'--tts-output',str(wav),'--seed','2','-t','4',
                '--gpu-backend','vulkan',
                '--no-watermark','--no-spoken-disclaimer','--accept-marking-responsibility'],
                env=env,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=1200)
        trace=(OUT/(cohort+'-cli.log')).read_text()
        assert set(re.findall(r'voxcpm2\[bench\]: cfm.steps=(\d+)',trace)) == {'10'}
        assert 'voxcpm2: backend = Vulkan' in trace, 'CLI did not use Vulkan'
        pcm,sr = sf.read(wav,dtype='float32'); assert sr==48000 and pcm.ndim==1
        waveforms[cohort]['cli-short'] = pcm
if PERF_ONLY:
    receipt['performance_only'] = True
    save()
    raise SystemExit(0)
# Release all TTS state before opening actual CUDA ASR. This checks GPU-generated
# speech without spending a Kaggle session on CPU-only acceptance.
asr = hf_hub_download('cstr/nemotron-3.5-asr-streaming-GGUF','nemotron-3.5-asr-streaming-0.6b-q4_k.gguf',
                      revision='bbd95a9ca5fa0dfca3312a122dfc45a2b578b9c2')
# Replay the actual failed v2 session PCM at both rates. The historical getter
# claimed 48 kHz for audio that had already been decimated to 24 kHz.
failed = hf_hub_download('cstr/crispasr-regression-fixtures',
    'voxcpm2/vulkan-profile-20261008-v2-failed/proof.zip',
    revision='f5a04b061d6ceb5c090c97a3f3fadfd2066e6398')
assert hashlib.sha256(Path(failed).read_bytes()).hexdigest() == '774d7f1a791b807a50b43898a3f674bdebc772ab2be998f35cc4a927823c8521'
with zipfile.ZipFile(failed) as archive:
    with archive.open('voxcpm2-profile/q8-short.npy') as file:
        historical = np.load(file)
with open_session(asr,'nemotron','cuda') as session:
    receipt['historical_rate_control'] = {}
    for rate in [48000,24000]:
        text = ' '.join(seg.text for seg in session.transcribe(historical,sample_rate=rate,language='en'))
        receipt['historical_rate_control'][str(rate)] = text
        save()
    normalize_control = lambda text: re.findall('[a-z]+',text.lower())
    # Record the historical gate without aborting before current-output probes.
    # v3 exposed repeated words at the corrected historical rate; do not relax
    # exact acceptance or hide that defect behind a sample-rate diagnosis.
    receipt['historical_rate_control_pass'] = (
        normalize_control(receipt['historical_rate_control']['24000']) == normalize_control(TEXTS['short'])
        and normalize_control(receipt['historical_rate_control']['48000']) != normalize_control(TEXTS['short']))
    save()
    for cohort,cases in waveforms.items():
        for key,pcm in cases.items():
            actual = ' '.join(seg.text for seg in session.transcribe(pcm,sample_rate=48000,language='en'))
            normalize = lambda text: re.findall('[a-z]+',text.lower())
            receipt['roundtrips'][cohort+'-'+key] = dict(transcript=actual,expected=TEXTS['short' if key=='cli-short' else key],
                                                       exact=normalize(actual)==normalize(TEXTS['short' if key=='cli-short' else key]))
            save()
assert len(receipt['roundtrips']) == 6 and all(case['exact'] for case in receipt['roundtrips'].values()), receipt['roundtrips']
assert receipt['historical_rate_control_pass'], receipt['historical_rate_control']
receipt['passed'] = True
save()
