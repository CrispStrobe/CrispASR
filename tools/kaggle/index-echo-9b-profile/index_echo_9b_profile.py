#!/usr/bin/env python3
"""Resident original-default BF16 versus accepted F16 on the same actual GPUs.

Both execution orders, isolated processes, first call plus three warm calls.
Every timed output must match its independent direct reference. This compares
actual implementations at different activation precision and placement; it
cannot isolate an algorithmic speedup or substitute for acceptance.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import wave

SCRIPT_VERSION = '2026-10-02.2'
SOURCE_COMMIT = 'f91a31da8d586157043639d0e6039c778d2571dc'
SOURCE_REVISION = 'b8ac6fb7d3dc17cee48a52201bd3d93dc86b0dba'
MODEL_REPO = 'cstr/index-echo-9b-GGUF'
ACCEPTANCE_REVISION = '2648560bb4b9104e1b0a2068dffb6846f1792553'
MODEL_REVISION = 'dca128e0da2c86819347b79f63da610c0b8bd472'
REFERENCE_REVISION = 'cb678dfd4806778aa55c39fe7b7a710a54cdc153'
BUNDLE_REVISION = '9ffaeaab43fece5ec5c0baeb4884e2db87625f6f'
BUNDLE_SHA256 = '166e8741738827fd0b997551bf27bfa93b5dcf04c898056fdedade87ba5c9f3e'
BUILD_COMMIT = 'f91a31da8d586157043639d0e6039c778d2571dc'
ROOT = Path('/kaggle/temp/index-echo-profile-repo')
TEMP = Path('/kaggle/temp/index-echo-profile')
OUT = Path('/kaggle/working')
TEMP.mkdir(parents=True,exist_ok=True);OUT.mkdir(parents=True,exist_ok=True)
os.environ['TMPDIR']=str(TEMP)
os.environ['OMP_NUM_THREADS']='4'
p=argparse.ArgumentParser();p.add_argument('--worker',choices=['python','f16']);p.add_argument('--order');a=p.parse_args()

def run(*args,**kw):subprocess.run(list(map(str,args)),check=True,**kw)

if not a.worker:
    hardware=subprocess.check_output(['nvidia-smi','--query-gpu=name,compute_cap,memory.total','--format=csv,noheader'],text=True).strip()
    if any(line.split(',')[1].strip()!='7.5' for line in hardware.splitlines()):
        raise RuntimeError('Inconclusive: validation runtime targets SM75')
    run('git','init',ROOT);run('git','-C',ROOT,'remote','add','origin','https://github.com/CrispStrobe/CrispASR.git')
    run('git','-C',ROOT,'fetch','--depth=1','origin',SOURCE_COMMIT);run('git','-C',ROOT,'checkout','FETCH_HEAD')
    sys.path.insert(0,str(ROOT/'tools/kaggle'));import kaggle_harness as kh
    kh.init_progress();kh.provenance(SCRIPT_VERSION,ROOT)
    run(sys.executable,'-m','pip','install','-q','transformers==5.6.0','accelerate','gguf','huggingface_hub','librosa>=0.10','soundfile>=0.12','safetensors>=0.5','silero-vad>=5.1','--no-deps')
    run(sys.executable,'-m','pip','install','-q','tokenizers','huggingface_hub','psutil','packaging','pyyaml','regex','safetensors','numpy','tqdm','onnxruntime','filelock','requests')
    from huggingface_hub import hf_hub_download,snapshot_download
    os.environ['HF_TOKEN']=kh.resolve_hf_token(require=True)
    # The acceptance receipt is uploaded only after all source/file gates pass.
    gate=hf_hub_download(MODEL_REPO,'acceptance.json',revision=ACCEPTANCE_REVISION,local_dir=TEMP/'gate')
    accepted=json.loads(Path(gate).read_text())
    if accepted.get('model_revision') != MODEL_REVISION:raise RuntimeError('Acceptance model pin mismatch')
    if accepted.get('validated') is not True or accepted.get('accepted_cohort')!='f16':
        raise RuntimeError('Performance follows correctness acceptance')
    for key in ['cpu','cuda','pipeline','roundtrip']:
        if accepted.get(key,{}).get('passed') is not True:raise RuntimeError('Missing acceptance: '+key)
    snapshot_download('IndexTeam/Index-Echo-S2TT-9B',revision=SOURCE_REVISION,local_dir=TEMP/'source')
    snapshot_download(MODEL_REPO,revision=MODEL_REVISION,local_dir=TEMP/'models',allow_patterns=['index-echo-9b-f16.gguf','index-echo-9b-decoder-f16.gguf'])
    for clip,path in [('jfk','jfk_11s'),('zh','zh')]:
        f=hf_hub_download('cstr/crispasr-regression-fixtures',f'index-echo-9b-f32-generation/{path}/ref.gguf',revision=REFERENCE_REVISION,local_dir=TEMP/'refs')
        (TEMP/f'{clip}-ref.gguf').symlink_to(f)
    f=hf_hub_download('cstr/crispasr-index-echo-cuda-validation','index-echo-cuda-validation.tar.gz',repo_type='dataset',revision=BUNDLE_REVISION,local_dir=TEMP/'artifact')
    h=hashlib.sha256()
    with open(f,'rb') as stream:
        for block in iter(lambda:stream.read(8*1024**2),b''):h.update(block)
    if h.hexdigest()!=BUNDLE_SHA256:raise RuntimeError('Runtime hash mismatch')
    import tarfile
    with tarfile.open(f) as tar:tar.extractall(TEMP,filter='data')
    if json.loads((TEMP/'bundle/provenance.json').read_text())['sha']!=BUILD_COMMIT:raise RuntimeError('Runtime build mismatch')
    os.environ['LD_LIBRARY_PATH']=str(TEMP/'bundle')+':'+os.environ.get('LD_LIBRARY_PATH','')
    results={}
    for label,arms in [('AB',['python','f16']),('BA',['f16','python'])]:
        results[label]={}
        for arm in arms:
            with (OUT/f'profile-{label}-{arm}.log').open('w') as log,kh.build_heartbeat('profile.'+label+'.'+arm,interval_s=30):
                run(sys.executable,__file__,'--worker',arm,'--order',label,stdout=log,stderr=subprocess.STDOUT)
            results[label][arm]=json.loads((OUT/f'profile-{label}-{arm}.json').read_text())
    receipt=dict(script_version=SCRIPT_VERSION,hardware=hardware,source_revision=SOURCE_REVISION,model_repo=MODEL_REPO,model_revision=MODEL_REVISION,acceptance_revision=ACCEPTANCE_REVISION,reference_revision=REFERENCE_REVISION,bundle_revision=BUNDLE_REVISION,build_commit=BUILD_COMMIT,threads=4,orders=results,validated=True,
        scope='Original resident BF16 versus native F16; different activation precision and recorded default layer placement. First call separate; three warm calls per clip and both execution orders. Exact independent output required.')
    (OUT/'profile.json').write_text(json.dumps(receipt,indent=2,ensure_ascii=False)+'\n')
    print('Resident profile completed; all timed outputs exact',flush=True)
    sys.exit(0)

import numpy as np
from gguf import GGUFReader
sys.path.insert(0,str(ROOT/'tools'));from reference_backends.index_echo import load_blueprint,precision_audit
sys.path.insert(0,str(ROOT/'python'));from crispasr import Session
# Reuse the existing exact-source comparison without accepting any variants.
sys.path.insert(0,str(ROOT/'tools'));from index_echo_acceptance import compare_case
import re

def cues(text):
    lines=[line.strip() for line in text.splitlines() if line.strip()]
    assert len(lines)%3==0
    out=[]
    for i in range(0,len(lines),3):
        m=re.fullmatch(r'\[(\d+):(\d+(?:\.\d+)?)-(\d+):(\d+(?:\.\d+)?)\]',lines[i]);assert m
        out.append(dict(start=60*int(m[1])+float(m[2]),end=60*int(m[3])+float(m[4]),text=lines[i+1]+'\n'+lines[i+2]))
    return out

started=time.perf_counter();result=dict(arm=a.worker,order=a.order,clips={})
if a.worker=='python':
    import torch
    torch.set_num_threads(4);torch.set_grad_enabled(False)
    memory={str(i):f'{int(torch.cuda.mem_get_info(i)[0]/2**30)-(2 if i==0 else 1)}GiB' for i in range(torch.cuda.device_count())};memory['cpu']='0GiB'
    os.environ.update(INDEX_ECHO_REF_DEVICE='cuda:0',INDEX_ECHO_REF_DTYPE='bfloat16',INDEX_ECHO_REF_DEVICE_MAP='auto',INDEX_ECHO_REF_MAX_MEMORY=json.dumps(memory),INDEX_ECHO_REF_OFFLOAD_DIR=str(TEMP/'offload'))
    _,model=load_blueprint(TEMP/'source');result['parameter_dtypes']=precision_audit(model);result['placement']=dict(model.llm.hf_device_map)
    assert not any(v in ['cpu','disk'] for v in result['placement'].values())
    for name in ['tower','connector','llm']:assert list(result['parameter_dtypes'][name]['parameter_elements'])==['torch.bfloat16']
    result['torch']=torch.__version__
else:
    os.environ['INDEX_ECHO_BENCH']='1'
    model=Session(str(TEMP/'models/index-echo-9b-f16.gguf'),lib_path=str(next((TEMP/'bundle').glob('libcrispasr.so*'))),n_threads=4)
result['load_seconds']=time.perf_counter()-started
try:
    for clip,audio in [('jfk',ROOT/'samples/jfk.wav'),('zh',ROOT/'samples/paraformer_zh.wav')]:
        reader=GGUFReader(TEMP/f'{clip}-ref.gguf');gold=cues(reader.fields['crispasr.ref.generated_text'].contents());del reader
        with wave.open(str(audio),'rb') as w:rate=w.getframerate();assert rate==16000;pcm=np.frombuffer(w.readframes(w.getnframes()),dtype=np.int16).astype(np.float32)/32768
        iterations=[]
        for i in range(4):
            if a.worker=='python':
                for d in range(torch.cuda.device_count()):torch.cuda.synchronize(d)
            started=time.perf_counter()
            if a.worker=='python':
                raw,_=model.translate_window(str(audio),[],lang='en',max_new_tokens=2000);actual=cues(raw)
                for d in range(torch.cuda.device_count()):torch.cuda.synchronize(d)
            else:actual=[dict(start=s.start,end=s.end,text=s.text) for s in model.transcribe(pcm)]
            elapsed=time.perf_counter()-started;compare_case(actual,{'independent-source':gold},'f16')
            iterations.append(dict(iteration=i,seconds=elapsed,segments=actual));print(a.worker,clip,i,elapsed,flush=True)
        result['clips'][clip]=dict(audio_seconds=len(pcm)/rate,first_seconds=iterations[0]['seconds'],warm_median_seconds=statistics.median(x['seconds'] for x in iterations[1:]),iterations=iterations)
finally:
    if a.worker=='f16':model.close()
    (OUT/f'profile-{a.order}-{a.worker}.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
