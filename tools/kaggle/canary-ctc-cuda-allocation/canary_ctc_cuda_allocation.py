#!/usr/bin/env python3
"""Issue 532: GPU allocation/reuse diagnosis; no private-clip/original-parity claim."""
import argparse
import ctypes as C
import hashlib
import json
import os
from pathlib import Path
import site
import subprocess
import sys
import tarfile
import wave
import zipfile

VERSION = '2026-10-12.canary-ctc-cuda.1'
SOURCE = 'eb633d908120cc5f44b8af8cd4aeb1191f16bfd8'
PACKAGE_REV = 'c00064122d9327d3b72be260e21ddc529b1ca054'
PACKAGE_PATH = 'release/38199002944/crispasr-linux-x86_64-cuda.tar.gz'
PACKAGE_SHA = 'abeb01ea8c5183685ef05d5f167ee2b51dfcf83a67aedfbdd2e0802683f0a178'
MODEL_REV = '2b22dc9aff585bc8368a5228173c8afe22c155b7'
MODEL_SHA = 'd16dbf18f9a66f59c0ceb61b204caca5dda21742d6e9dc304c9d0518c81ee38c'
CPU_REV = '2cfe9f89b0d64a3450e03023b87f5178fbdc949a'
CPU_SHA = 'fcc552aa6f5623d64a32a639db18857980ba9f9be7b0786d831a667bcd5ef0bd'
SCRATCH = Path('/kaggle/temp/canary-ctc-cuda')
OUT = Path('/kaggle/working')
SDK = SCRATCH/'sdk'
SCRATCH.mkdir(parents=True, exist_ok=True)
os.environ.update(TMPDIR=str(SCRATCH),HF_HOME=str(SCRATCH/'hf'),HF_XET_CACHE=str(SCRATCH/'xet'),OMP_NUM_THREADS='4')
args=argparse.ArgumentParser();args.add_argument('--bucket',type=int,choices=(0,128));arg=args.parse_args()

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
    return h.hexdigest()

if arg.bucket is None:
    hardware=subprocess.check_output(['nvidia-smi','--query-gpu=name,compute_cap,memory.total,driver_version','--format=csv,noheader'],text=True).strip()
    assert hardware and all(int(float(r.split(',')[1])*10) in (60,61,70,75,80,86,89,90,120) for r in hardware.splitlines())
    for command in [['git','init',SDK],['git','-C',SDK,'remote','add','origin','https://github.com/CrispStrobe/CrispASR.git'],['git','-C',SDK,'fetch','--depth=1','origin',SOURCE],['git','-C',SDK,'checkout','FETCH_HEAD']]:subprocess.run(list(map(str,command)),check=True)
    sys.path.insert(0,str(SDK/'tools/kaggle'))
    import kaggle_harness as kh
    kh.init_progress();kh.provenance(VERSION,SDK)
    os.environ['HF_TOKEN']=kh.resolve_hf_token(require=True)
    subprocess.run([sys.executable,'-m','pip','install','-q','numpy','huggingface_hub'],check=True)
    from huggingface_hub import hf_hub_download
    def fetch(repo,name,rev,digest,kind):
        p=Path(hf_hub_download(repo,name,revision=rev,repo_type=kind));assert sha(p)==digest;return p
    archive=fetch('cstr/crispasr-regression-fixtures',PACKAGE_PATH,PACKAGE_REV,PACKAGE_SHA,'dataset')
    with tarfile.open(archive) as t:t.extractall(SCRATCH,filter='data')
    model=fetch('cstr/canary-ctc-aligner-GGUF','canary-ctc-aligner-q4_k.gguf',MODEL_REV,MODEL_SHA,'model')
    assert model.stat().st_size==392167040
    archive=fetch('cstr/crispasr-regression-fixtures','canary/ctc-allocation-20261011/proof.zip',CPU_REV,CPU_SHA,'dataset')
    with zipfile.ZipFile(archive) as z:z.extractall(SCRATCH/'cpu-proof')
    receipt=dict(version=VERSION,source=SOURCE,package_sha256=PACKAGE_SHA,model_sha256=MODEL_SHA,
                 model_revision=MODEL_REV,cpu_proof_revision=CPU_REV,hardware=hardware,entry_sha256=sha(__file__),
                 original_parity=False,reporter_recording_reproduced=False,allocation_crash_reproduced=False,
                 complete=False,passed=False,buckets=[])
    for bucket in (0,128):
        env=dict(os.environ,CRISPASR_FC_BUCKET=str(bucket),CRISPASR_CANARY_MODEL=str(model))
        with (OUT/f'bucket-{bucket}.log').open('w') as log:
            rc=subprocess.run([sys.executable,str(Path(__file__).resolve()),'--bucket',str(bucket)],env=env,stdout=log,stderr=subprocess.STDOUT).returncode
        row=dict(bucket=bucket,returncode=rc)
        result=OUT/f'bucket-{bucket}/validation.json'
        if result.exists():row['result']=json.loads(result.read_text())
        receipt['buckets'].append(row)
        receipt['allocation_crash_reproduced']=any(r['returncode']==-6 for r in receipt['buckets'])
        (OUT/'validation.json').write_text(json.dumps(receipt,indent=2)+'\n')
    receipt.update(complete=True,passed=all(r['returncode']==0 and r['result']['passed'] for r in receipt['buckets']))
    (OUT/'validation.json').write_text(json.dumps(receipt,indent=2)+'\n')
    assert receipt['passed'],receipt['buckets']
    kh.step('all.complete',passed=True)
    raise SystemExit(0)

# The bucket value is cached by native code; each arm must have its own process.
import numpy as np
bucket=arg.bucket
os.environ['CRISPASR_FC_BUCKET']=str(bucket)
DEST=OUT/f'bucket-{bucket}';DEST.mkdir(exist_ok=True)
bundle=SCRATCH/'crispasr-linux-x86_64-cuda'
for name in ('libcuda.so.1','libcudart.so.12','libcublasLt.so.12','libcublas.so.12'):
    candidates=[Path('/usr/local/cuda/lib64')/name]
    for prefix in site.getsitepackages():candidates.extend(Path(prefix).glob('nvidia/*/lib/'+name))
    path=next((p for p in candidates if p.is_file()),None)
    C.CDLL(str(path) if path else name,mode=C.RTLD_GLOBAL)
lib=C.CDLL(str(bundle/'libcrispasr.so.1'))
lib.ggml_backend_load_all_from_path.argtypes=[C.c_char_p];lib.ggml_backend_load_all_from_path(str(bundle).encode())
class Params(C.Structure):_fields_=[('n_threads',C.c_int),('verbosity',C.c_int)]
fp=C.POINTER(C.c_float);ip=C.POINTER(C.c_int)
lib.canary_ctc_context_default_params.restype=Params
lib.canary_ctc_init_from_file.argtypes=[C.c_char_p,Params];lib.canary_ctc_init_from_file.restype=C.c_void_p
lib.canary_ctc_free.argtypes=[C.c_void_p]
lib.canary_ctc_compute_logits.argtypes=[C.c_void_p,fp,C.c_int,C.POINTER(fp),ip,ip];lib.canary_ctc_compute_logits.restype=C.c_int
allocator=C.CDLL(None);allocator.free.argtypes=[C.c_void_p]
params=lib.canary_ctc_context_default_params();params.n_threads=4;params.verbosity=1
model=os.environ['CRISPASR_CANARY_MODEL'].encode()
ctx=lib.canary_ctc_init_from_file(model,params);assert ctx
maps=Path('/proc/self/maps').read_text();assert str(bundle/'libggml-cuda.so') in maps
allocation=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_gpu_memory','--format=csv,noheader,nounits'],text=True)
assert any(int(r.split(',')[0])==os.getpid() and int(r.split(',')[1])>=256 for r in allocation.splitlines())
receipt=dict(bucket=bucket,passed=False,complete=False,gpu_allocations=allocation.strip(),cuda_plugin_sha256=sha(bundle/'libggml-cuda.so'),cases=[])
def save():(DEST/'validation.json').write_text(json.dumps(receipt,indent=2)+'\n')
save()
with wave.open(str(SDK/'samples/jfk.wav')) as w:
    assert (w.getframerate(),w.getnchannels(),w.getsampwidth())==(16000,1,2)
    base=np.frombuffer(w.readframes(w.getnframes()),dtype='<i2').astype(np.float32)/32768
assert hashlib.sha256(base.tobytes()).hexdigest()=='ebd52851100536db02d12c49fddd010372dcdc70243562e057553d476b706ae0'
def compute(context,pcm):
    values=fp();frames=C.c_int();vocab=C.c_int()
    rc=lib.canary_ctc_compute_logits(context,pcm.ctypes.data_as(fp),len(pcm),C.byref(values),C.byref(frames),C.byref(vocab));assert rc==0 and values
    try:return np.ctypeslib.as_array(values,shape=(frames.value*vocab.value,)).copy().reshape(frames.value,vocab.value)
    finally:allocator.free(values)
try:
    for seq,seconds in enumerate((11,4,33,90,11)):
        receipt['current']=dict(seq=seq,seconds=seconds,phase='native computation');save()
        pcm=np.resize(base,seconds*16000);a=compute(ctx,pcm);assert np.isfinite(a).all()
        path=DEST/f'case-{seq}.npy';np.save(path,a)
        row=dict(seq=seq,seconds=seconds,shape=list(a.shape),sha256=sha(path),finite=True)
        if seq in (0,4):
            fresh=lib.canary_ctc_init_from_file(model,params);assert fresh
            try:row['fresh_reused_exact']=bool(np.array_equal(a,compute(fresh,pcm)))
            finally:lib.canary_ctc_free(fresh)
            assert row['fresh_reused_exact'],row
        cpu_path=next((SCRATCH/'cpu-proof').rglob(f'bucket-{bucket}/case-{seq}.f32'))
        b=np.fromfile(cpu_path,dtype=np.float32).reshape(a.shape)
        aa=a.astype(np.float64);bb=b.astype(np.float64);na=np.linalg.norm(aa,axis=1);nb=np.linalg.norm(bb,axis=1)
        cos=np.divide((aa*bb).sum(axis=1),na*nb,out=np.ones_like(na),where=na*nb>0)
        row['cpu_gpu_diagnostic']=dict(row_cosine_min=float(cos.min()),norm_ratio=float(np.linalg.norm(aa)/np.linalg.norm(bb)),relative_l2=float(np.linalg.norm(aa-bb)/np.linalg.norm(bb)),argmax_exact=bool(np.array_equal(a.argmax(-1),b.argmax(-1))))
        receipt['cases'].append(row);save();print('case complete',bucket,seq,seconds,flush=True)
    receipt.update(complete=True,passed=True);save()
finally:lib.canary_ctc_free(ctx)
