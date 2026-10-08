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
import numpy as np
from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(os.environ['HEAVY_OUT'])
TEMP = Path(os.environ['HEAVY_SCRATCH'])/'voxcpm2-profile'
TEMP.mkdir(parents=True,exist_ok=True)
BUILD = Path(os.environ['VOX_PROFILE_BUILD'])
MODEL_PIN = '25b5cf03fdbf20011dad9a77def6112023cc0fe3'
TEXTS = {'short':'Hello, this is a short test sentence.',
         'long':'The sun is shining today. Hello, this is a short test sentence.'}
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
        for rep in range(8):
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
                receipt['calls'].append(dict(cohort=cohort,text=key,repetition=rep,cold=rep==0,
                    measured=rep>=2,seconds=elapsed,samples=len(pcm),audio_seconds=len(pcm)/48000,
                    rtf=elapsed/(len(pcm)/48000),sha256=digest))
                if key not in waveforms.get(cohort,{}):
                    waveforms.setdefault(cohort,{})[key] = pcm
                    np.save(OUT/(cohort+'-'+key+'.npy'),pcm)
                save()
# Release all TTS state before opening actual CUDA ASR. This checks GPU-generated
# speech without spending a Kaggle session on CPU-only acceptance.
asr = hf_hub_download('cstr/nemotron-3.5-asr-streaming-GGUF','nemotron-3.5-asr-streaming-0.6b-q4_k.gguf',
                      revision='bbd95a9ca5fa0dfca3312a122dfc45a2b578b9c2')
with open_session(asr,'nemotron','cuda') as session:
    for cohort,cases in waveforms.items():
        for key,pcm in cases.items():
            actual = ' '.join(seg.text for seg in session.transcribe(pcm,sample_rate=48000,language='en'))
            normalize = lambda text: re.findall('[a-z]+',text.lower())
            receipt['roundtrips'][cohort+'-'+key] = dict(transcript=actual,expected=TEXTS[key],
                                                       exact=normalize(actual)==normalize(TEXTS[key]))
            save()
assert all(case['exact'] for case in receipt['roundtrips'].values()), receipt['roundtrips']
receipt['passed'] = True
save()
