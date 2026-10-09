#!/usr/bin/env python3
"""Run pinned CPU-built Nemotron artifacts on real CUDA; no compilation/quantization."""
import ctypes
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tarfile
import wave
import zipfile

VERSION = '2026-10-09.1'
SOURCE = '1977bd047d73b5e35cf5d267801fe21e859cfc1c'
BUILD_RUN = 37943881572
BUNDLE_REV = ''  # Fill only after successful exact-source CI build/publication.
BUNDLE_SHA = ''
ORIGINAL_REV = '10e579dedf52a16a159cbc5d35e17f8bffa77190'
ORIGINAL_SHA = '7762f2a73d8ae7dad05d667fbf0bca393a0b3b7ad622b1b1f09473fa8f5d0463'
MODEL_REV = 'bbd95a9ca5fa0dfca3312a122dfc45a2b578b9c2'
TEMP = Path('/kaggle/temp/nemotron-frontend')
OUT = Path('/kaggle/working')
TEMP.mkdir(parents=True, exist_ok=True)
os.environ.update(TMPDIR=str(TEMP), HF_HOME=str(TEMP/'hf'), HF_XET_CACHE=str(TEMP/'xet'),
    OMP_NUM_THREADS='4', CRISPASR_NEMOTRON_CONTEXT_PRESET='0',
    CRISPASR_NEMOTRON_STREAM_CHUNKS_PER_STEP='1',
    CRISPASR_NEMOTRON_STREAM_INCREMENTAL_FRONTEND='0')
assert len(BUNDLE_REV) == 40 and len(BUNDLE_SHA) == 64, 'Immutable successful CI bundle required'
hardware = subprocess.check_output(['nvidia-smi', '--query-gpu=name,compute_cap,memory.total',
    '--format=csv,noheader'], text=True).strip()
if not hardware or any(row.split(',')[1].strip() != '7.5' for row in hardware.splitlines()):
    (OUT/'inconclusive.json').write_text(json.dumps(dict(hardware=hardware, conclusive=False,
        reason='SM75 bundle requires actual T4 hardware; no models downloaded')))
    raise SystemExit(0)
sdk = TEMP/'sdk'
for command in [['git','init',sdk], ['git','-C',sdk,'remote','add','origin','https://github.com/CrispStrobe/CrispASR.git'],
                ['git','-C',sdk,'fetch','--depth=1','origin',SOURCE], ['git','-C',sdk,'checkout','FETCH_HEAD']]:
    subprocess.run(list(map(str,command)),check=True)
sys.path.insert(0,str(sdk/'tools/kaggle'))
import kaggle_harness as kh
kh.init_progress()
kh.provenance(VERSION,sdk)
os.environ['HF_TOKEN'] = kh.resolve_hf_token(require=True)
subprocess.run([sys.executable,'-m','pip','install','-q','numpy','gguf','huggingface_hub'],check=True)
import numpy as np
from gguf import GGUFReader
from huggingface_hub import hf_hub_download


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024**2),b''): h.update(block)
    return h.hexdigest()


def fetch(repo,path,rev,sha=None,kind='model'):
    result=Path(hf_hub_download(repo,path,revision=rev,repo_type=kind))
    if sha: assert digest(result)==sha
    return result


archive=fetch('cstr/crispasr-index-echo-cuda-validation','index-echo-cuda-validation.tar.gz',BUNDLE_REV,BUNDLE_SHA,'dataset')
with tarfile.open(archive) as tar: tar.extractall(TEMP,filter='data')
bundle=TEMP/'bundle'
assert json.loads((bundle/'provenance.json').read_text())['sha']==SOURCE
original=fetch('cstr/crispasr-regression-fixtures','tts-asr/nemotron-reference-20261009/proof.zip',ORIGINAL_REV,ORIGINAL_SHA)
with zipfile.ZipFile(original) as z: z.extractall(TEMP/'original')
original_path=next((TEMP/'original').rglob('nemotron-tts-reference.json'))
prior=json.loads(original_path.read_text())
model=fetch('cstr/nemotron-3.5-asr-streaming-GGUF','nemotron-3.5-asr-streaming-0.6b-f16.gguf',MODEL_REV,prior['native_models']['f16']['sha256'])
manifest=json.loads((sdk/'tests/regression/manifest.json').read_text())
entry=next(x for x in manifest['backends'] if x['backend_id']=='nemotron')
ref_path=fetch(manifest['fixtures']['repo'],entry['fixture_ref_path'],manifest['fixtures']['revision'])
reference=GGUFReader(str(ref_path))
lib=ctypes.CDLL(str(bundle/'libcrispasr.so'))
fp=ctypes.POINTER(ctypes.c_float)
ip=ctypes.POINTER(ctypes.c_int)
class Params(ctypes.Structure):
    _fields_=[('n_threads',ctypes.c_int),('use_flash',ctypes.c_bool),('verbosity',ctypes.c_int),('use_gpu',ctypes.c_bool)]
lib.nemotron_context_default_params.argtypes=[]
lib.nemotron_context_default_params.restype=Params
lib.nemotron_init_from_file.argtypes=[ctypes.c_char_p,Params]
lib.nemotron_init_from_file.restype=ctypes.c_void_p
lib.nemotron_free.argtypes=[ctypes.c_void_p]
lib.nemotron_free.restype=None
lib.nemotron_compute_mel.argtypes=[ctypes.c_void_p,fp,ctypes.c_int,ip,ip]
lib.nemotron_compute_mel.restype=fp
for name in ['nemotron_run_preencode_ext','nemotron_run_encoder_ext']:
    fn=getattr(lib,name);fn.argtypes=[ctypes.c_void_p,fp,ctypes.c_int,ctypes.c_int,ip,ip];fn.restype=fp
allocator=ctypes.CDLL(None);allocator.free.argtypes=[ctypes.c_void_p];allocator.free.restype=None
params=lib.nemotron_context_default_params();params.n_threads=4;params.use_gpu=True;params.verbosity=1
receipt=dict(version=VERSION,source=SOURCE,build_run=BUILD_RUN,bundle_revision=BUNDLE_REV,bundle_sha256=BUNDLE_SHA,
    hardware=hardware,model_revision=MODEL_REV,original_revision=ORIGINAL_REV,original_sha256=ORIGINAL_SHA,
    reference_revision=manifest['fixtures']['revision'],validated=False,stages={},fresh={},reused={},streams={},failed=[])
def save(): (OUT/'validation.json').write_text(json.dumps(receipt,indent=2)+'\n')
save()
ctx=lib.nemotron_init_from_file(str(model).encode(),params);assert ctx
allocation=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_gpu_memory','--format=csv,noheader,nounits'],text=True)
receipt['gpu_process_allocations']=allocation.strip()
assert any(int(row.split(',')[0].strip())==os.getpid() and int(row.split(',')[1].strip())>=256 for row in allocation.splitlines()), 'Native context must allocate model buffers on actual CUDA'
save()
with wave.open(str(sdk/entry['sample'])) as wav:
    assert wav.getframerate()==16000 and wav.getnchannels()==1 and wav.getsampwidth()==2
    pcm=np.frombuffer(wav.readframes(wav.getnframes()),dtype='<i2').astype(np.float32)/32768
arrays={};nm=ctypes.c_int();tm=ctypes.c_int()
mel=lib.nemotron_compute_mel(ctx,pcm.ctypes.data_as(fp),len(pcm),ctypes.byref(nm),ctypes.byref(tm));assert mel
try:
    arrays['mel_spectrogram']=np.ctypeslib.as_array(mel,shape=(nm.value*tm.value,)).copy().reshape(tm.value,nm.value).T.copy()
    for stage,api in [('pre_encode_output','nemotron_run_preencode_ext'),('encoder_output','nemotron_run_encoder_ext')]:
        t=ctypes.c_int();d=ctypes.c_int();data=getattr(lib,api)(ctx,mel,nm.value,tm.value,ctypes.byref(t),ctypes.byref(d));assert data
        try: arrays[stage]=np.ctypeslib.as_array(data,shape=(t.value*d.value,)).copy().reshape(t.value,d.value)
        finally: allocator.free(data)
finally: allocator.free(mel)
# Retain complete arrays and the existing COS_LAST_DIM row convention; also report scale.
for name,array in list(arrays.items()):
    tensor=next(t for t in reference.tensors if t.name==name)
    expected=np.asarray(tensor.data,dtype=np.float32).copy();assert array.shape==expected.shape,(name,array.shape,expected.shape)
    arrays['reference_'+name]=expected
    a=array.astype(np.float64).reshape(-1,int(tensor.shape[-1]));b=expected.astype(np.float64).reshape(a.shape)
    na=np.linalg.norm(a,axis=1);nb=np.linalg.norm(b,axis=1);power=na*nb
    cosine=np.divide(np.sum(a*b,axis=1),power,out=np.ones_like(power),where=power>1e-12)
    cosine[(na<=1e-12)!=(nb<=1e-12)]=0
    cosine=cosine.astype(np.float32)  # Ref.compare stores each cosine in float32.
    report=dict(cosine_min=float(cosine.min()),native_norm=float(np.linalg.norm(a)),reference_norm=float(np.linalg.norm(b)),
        relative_l2=float(np.linalg.norm(a-b)/max(np.linalg.norm(b),1e-30)),max_abs=float(np.max(np.abs(a-b))))
    report['passed']=bool(np.isfinite(array).all() and report['cosine_min']>=0.999)
    receipt['stages'][name]=report
    if not report['passed']: receipt['failed'].append(name)
np.savez_compressed(OUT/'stage-arrays.npz',**arrays);save()
assert not receipt['failed'],'F16 GPU stage control failed; no downstream acceptance'
# Actual CUDA streaming invariants, full frontend retained. No upstream streaming parity claim.
CB=ctypes.CFUNCTYPE(None,ctypes.c_int,ctypes.c_float,ctypes.c_void_p)
lib.nemotron_stream_create.argtypes=[ctypes.c_void_p];lib.nemotron_stream_create.restype=ctypes.c_void_p
lib.nemotron_stream_free.argtypes=[ctypes.c_void_p];lib.nemotron_stream_free.restype=None
lib.nemotron_stream_reset.argtypes=[ctypes.c_void_p];lib.nemotron_stream_reset.restype=None
lib.nemotron_stream_append.argtypes=[ctypes.c_void_p,fp,ctypes.c_int,ctypes.c_bool,CB,ctypes.c_void_p];lib.nemotron_stream_append.restype=ctypes.c_bool
lib.nemotron_stream_processed_frames.argtypes=[ctypes.c_void_p];lib.nemotron_stream_processed_frames.restype=ctypes.c_int
lib.nemotron_set_context_preset.argtypes=[ctypes.c_void_p,ctypes.c_int];lib.nemotron_set_context_preset.restype=None
lib.nemotron_token_to_str.argtypes=[ctypes.c_void_p,ctypes.c_int];lib.nemotron_token_to_str.restype=ctypes.c_char_p
words=lambda text:re.findall('[a-z]+',re.sub(r'<[^>]*>','',text).lower())
long_pcm=np.tile(pcm,3)
for preset in [0,2,3]:
    lib.nemotron_set_context_preset(ctx,preset);stream=lib.nemotron_stream_create(ctx);assert stream
    outputs=[]
    try:
        for packet in [1600,1600,3200]:
            lib.nemotron_stream_reset(stream);tokens=[];frames=[]
            @CB
            def callback(token,prob,user): tokens.append([token,float(prob)])
            for start in range(0,len(long_pcm),packet):
                chunk=np.ascontiguousarray(long_pcm[start:start+packet]);assert lib.nemotron_stream_append(stream,chunk.ctypes.data_as(fp),len(chunk),False,callback,None)
                frames.append(lib.nemotron_stream_processed_frames(stream))
            before_flush=len(tokens);assert before_flush>0
            assert lib.nemotron_stream_append(stream,None,0,True,callback,None)
            first_flush=list(tokens);assert lib.nemotron_stream_append(stream,None,0,True,callback,None);assert tokens==first_flush,'duplicate final flush'
            assert all(x<=y for x,y in zip(frames,frames[1:]))
            assert all(np.isfinite(p) and 0<=p<=1 for _,p in tokens)
            text=''.join(lib.nemotron_token_to_str(ctx,t).decode('utf-8') for t,_ in tokens).replace('▁',' ')
            outputs.append(dict(packet_samples=packet,tokens=tokens,text=text,tokens_before_flush=before_flush,final_frames=lib.nemotron_stream_processed_frames(stream)))
        if preset==0:
            assert words(outputs[0]['text'])==words(entry['expected_transcript'])*3,'streamed JFK content mismatch'
        assert outputs[0]['tokens']==outputs[1]['tokens'],'reset/repeat token confidence mismatch'
        assert [x[0] for x in outputs[0]['tokens']]==[x[0] for x in outputs[2]['tokens']],'packet-size token mismatch'
        receipt['streams'][str(preset)]=outputs;save();kh.step('stream.complete',preset=preset)
    finally: lib.nemotron_stream_free(stream)
lib.nemotron_free(ctx)
sys.path.insert(0,str(sdk/'python'))
from crispasr import Session
cases={}
for label,case in prior['cases'].items():
    data=np.load(original_path.parent/case['file'],allow_pickle=False);assert hashlib.sha256(data.tobytes()).hexdigest()==case['sha256'];cases[label]=data
assert len(cases)==26
words=lambda text:re.findall('[a-z]+',re.sub(r'<[^>]*>','',text).lower())
for label,data in cases.items():
    with Session(str(model),lib_path=str(bundle/'libcrispasr.so'),backend='nemotron',n_threads=4) as session:
        receipt['fresh'][label]=' '.join(s.text for s in session.transcribe(data,sample_rate=16000,language='en'))
    if words(receipt['fresh'][label])!=words(prior['original'][label]['transcript']): receipt['failed'].append('original:'+label)
    save()
with Session(str(model),lib_path=str(bundle/'libcrispasr.so'),backend='nemotron',n_threads=4) as session:
    for label,data in reversed(list(cases.items())):
        receipt['reused'][label]=' '.join(s.text for s in session.transcribe(data,sample_rate=16000,language='en'))
        if receipt['reused'][label]!=receipt['fresh'][label]: receipt['failed'].append('state:'+label)
        save()
receipt['validated']=not receipt['failed'];save()
if not receipt['validated']: raise RuntimeError('F16 GPU validation failed: '+str(receipt['failed']))
kh.step('all.complete',validated=True)
