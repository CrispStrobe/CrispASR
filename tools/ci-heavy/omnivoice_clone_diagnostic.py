#!/usr/bin/env python3
"""Isolate OmniVoice reference and decode cleanup; diagnostic only, not acceptance."""
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
TEMP = Path(os.environ['HEAVY_SCRATCH']) / 'omnivoice-518-clone-diagnostic'
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
# Punctuation-only holds the legacy audio reference fixed. The next arm adds
# reference silence cleanup; the final arm changes only output DSP. Every arm
# uses the original rejected seed/steps/model and the same actual voice clip.
ARMS = {
    'legacy': ('1', '0', reference['text'].rstrip('.')),
    'punctuation': ('1', '0', reference['text']),
    'reference-cleanup': ('1', '1', reference['text']),
    'output-cleanup': ('0', '1', reference['text']),
}
receipt = dict(passed=False, diagnostic_complete=False, seed=42, steps=32,
    scope=__doc__, model_revision=MODEL_PIN, fixture_revision=FIXTURE_PIN,
    source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
    arms={}, roundtrips={})

def save():
    (OUT/'clone-diagnostic.json').write_text(json.dumps(receipt,indent=2)+'\n')

save()
outputs = {}
for label,(legacy,preprocess,ref_text) in ARMS.items():
    os.environ.update(CRISPASR_OMNIVOICE_AUDIO_LEGACY=legacy,
                      CRISPASR_OMNIVOICE_PREPROCESS_PROMPT=preprocess,
                      CRISPASR_OMNIVOICE_DUMP_CODES=str(OUT/(label+'.codes')))
    # Clear unrelated overrides so the captured recipe is reproducible.
    for control in ('POSTPROCESS_OUTPUT','PAD_DURATION','FADE_DURATION'):
        os.environ.pop('CRISPASR_OMNIVOICE_'+control,None)
    with Session(paths['omnivoice-q8_0.gguf'],lib_path=str(lib),backend='omnivoice',n_threads=4) as s:
        s.set_codec_path(paths['omnivoice-tokenizer-f16.gguf'])
        s.set_tts_seed(42)
        s.set_tts_steps(32)
        s.set_voice(ref_wav,ref_text=ref_text)
        s.accept_marking_responsibility('Fixed-seed reference/decode first-divergence diagnostic')
        pcm=np.asarray(s.synthesize_raw(TEXT),dtype=np.float32)
        assert s.output_sample_rate()==24000 and len(pcm)>24000
        assert np.isfinite(pcm).all()
        outputs[label]=pcm
        np.save(OUT/(label+'.npy'),pcm)
        receipt['arms'][label]=dict(legacy=legacy,preprocess=preprocess,ref_text=ref_text,
            samples=len(pcm),pcm_sha256=hashlib.sha256(pcm.tobytes()).hexdigest(),
            codes_sha256=hashlib.sha256((OUT/(label+'.codes')).read_bytes()).hexdigest())
        save()
assert (OUT/'reference-cleanup.codes').read_bytes()==(OUT/'output-cleanup.codes').read_bytes(), 'Output-only toggle changed generated codes'
receipt['output_only_codes_exact']=True
asr=hf_hub_download('cstr/nemotron-3.5-asr-streaming-GGUF',
    'nemotron-3.5-asr-streaming-0.6b-q4_k.gguf',revision='bbd95a9ca5fa0dfca3312a122dfc45a2b578b9c2')
with Session(asr,lib_path=str(lib),backend='nemotron',n_threads=4) as s:
    for label,pcm in outputs.items():
        actual=' '.join(seg.text for seg in s.transcribe(pcm,sample_rate=24000,language='en'))
        words=lambda text: re.findall('[a-z]+',re.sub(r'<[^>]*>','',text).lower())
        ref,hyp=words(TEXT),words(actual)
        row=list(range(len(hyp)+1))
        for i,w in enumerate(ref,1):
            new=[i]
            for j,v in enumerate(hyp,1):
                new.append(min(new[-1]+1,row[j]+1,row[j-1]+(w!=v)))
            row=new
        receipt['roundtrips'][label]=dict(transcript=actual,wer=row[-1]/len(ref),
                                         meets_existing_gate=row[-1]/len(ref)<=.2)
        save()
assert len(receipt['roundtrips'])==4
receipt['diagnostic_complete']=True
save()
print('OMNIVOICE_CLONE_DIAGNOSTIC_COMPLETE',receipt['roundtrips'],flush=True)
