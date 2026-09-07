#!/usr/bin/env python3
"""Q4 Parakeet TDT A/B: current ggml v0.17 fork versus v0.23 consolidation."""
import io, json, os, shutil, statistics, subprocess, sys, time
from pathlib import Path

WORK = Path('/kaggle/working')
TEMP = Path('/kaggle/temp') if Path('/kaggle/temp').is_dir() else Path('/tmp')
REPO_CURRENT = TEMP / 'CrispASR-current'
REPO_V023 = TEMP / 'CrispASR-v023'
BRANCH = 'perf/81-round2'
GGML_V023 = '069a517d6755ea489d86596013bc5e5d95d5465b'
SCRIPT_VERSION = 'ggml-v023-q4-ab-v1'

subprocess.check_call(['git','clone','--depth','1','-b',BRANCH,'https://github.com/CrispStrobe/CrispASR',str(REPO_CURRENT)])
subprocess.check_call(['git','submodule','update','--init','ggml','third_party/c2pa-audio'],cwd=REPO_CURRENT)
shutil.copytree(REPO_CURRENT, REPO_V023, symlinks=True)
subprocess.check_call(['git','fetch','origin','sync/upstream-v0.23','--depth','1'],cwd=REPO_V023/'ggml')
subprocess.check_call(['git','checkout','--detach',GGML_V023],cwd=REPO_V023/'ggml')

sys.path.insert(0,str(REPO_CURRENT/'tools'/'kaggle'))
import kaggle_harness as kh
kh.init_progress()
token = kh.resolve_hf_token('HF_TOKEN')
arch = kh.detect_cuda_arch()
commit = subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO_CURRENT,text=True).strip()
kh.step('provenance', commit=commit, branch=BRANCH, ggml_v023=GGML_V023, script_version=SCRIPT_VERSION, cuda_arch=arch)
print(f'SCRIPT_VERSION={SCRIPT_VERSION} COMMIT={commit} GGML_V023={GGML_V023} CUDA_ARCH={arch}',flush=True)
subprocess.check_call([sys.executable,'-m','pip','install','--quiet','huggingface_hub','soundfile','datasets'])
os.environ['HF_HUB_ENABLE_HF_TRANSFER']='0'
kh.install_build_toolchain()

flags = kh.cuda_build_flags(arch) + kh.cache_and_link_flags()
libs = {}
for name, repo in [('current',REPO_CURRENT),('v023',REPO_V023)]:
    build = TEMP / f'build-{name}'
    subprocess.check_call(['cmake','-G','Ninja','-B',str(build),'-S',str(repo),'-DCMAKE_BUILD_TYPE=Release','-DCRISPASR_NO_C2PA_NATIVE=ON',*flags])
    with kh.build_heartbeat(f'build.{name}'):
        kh.sh_with_progress(f'cmake --build {build} -j {kh.safe_build_jobs(True)} --target crispasr-lib')
    libs[name] = build/'src'/'libcrispasr.so'
    if not libs[name].is_file(): raise RuntimeError(f'missing {libs[name]}')
    kh.step(f'build.{name}.done')

from huggingface_hub import hf_hub_download
import numpy as np
import soundfile as sf
from datasets import Audio, load_dataset
model = hf_hub_download('cstr/parakeet-tdt-0.6b-v3-GGUF','parakeet-tdt-0.6b-v3-q4_k.gguf',local_dir=str(TEMP/'models'),token=token)
jfk,sr=sf.read(str(REPO_CURRENT/'samples'/'jfk.wav'),dtype='float32')
if sr != 16000: raise RuntimeError(sr)
ds=load_dataset('hf-internal-testing/librispeech_asr_dummy','clean',split='validation').cast_column('audio',Audio(decode=False))
gap=np.zeros(int(.3*sr),dtype='float32'); parts=[]; samples=0
for row in ds:
    source=row['audio']; audio,r=sf.read(io.BytesIO(source['bytes']) if source.get('bytes') is not None else source['path'],dtype='float32')
    if r != sr: raise RuntimeError(r)
    audio=np.asarray(audio,dtype='float32'); parts.extend((audio,gap)); samples += len(audio)+len(gap)
    if samples/sr >= 120: break
varied=np.concatenate(parts)
clips=TEMP/'clips'; clips.mkdir(exist_ok=True)
paths={'jfk':clips/'jfk.wav','varied':clips/'varied.wav'}
sf.write(paths['jfk'],jfk,sr); sf.write(paths['varied'],varied,sr)
durations={'jfk':len(jfk)/sr,'varied':len(varied)/sr}

CHILD=r'''\nimport json,os,sys,time,statistics\nimport soundfile as sf\nsys.path.insert(0,os.path.join(sys.argv[1],'python'))\nos.environ['CRISPASR_LIB_PATH']=sys.argv[2]\nfrom crispasr import Session\ns=Session(sys.argv[3],n_threads=4); paths=json.loads(sys.argv[4]); out={}\nfor name,path in paths.items():\n pcm,sr=sf.read(path,dtype='float32'); times=[]; texts=[]\n for _ in range(4 if name == 'jfk' else 3):\n  t=time.perf_counter(); segs=s.transcribe(pcm.copy(),language='en'); times.append(time.perf_counter()-t); texts.append(' '.join(x.text for x in segs).strip())\n out[name]={'times_s':times,'texts':texts}\nprint('RESULT::'+json.dumps(out),flush=True)\n'''

def run(name):
    env=dict(os.environ); env['CRISPASR_PARAKEET_BENCH']='1'; env['CRISPASR_PARAKEET_DECODE_TIMING']='1'
    p=subprocess.run([sys.executable,'-c',CHILD,str(REPO_CURRENT),str(libs[name]),model,json.dumps({k:str(v) for k,v in paths.items()})],env=env,text=True,capture_output=True,timeout=3600)
    if p.returncode: raise RuntimeError(f'{name} failed ({p.returncode}):\n{p.stderr[-6000:]}')
    payload=next((x[8:] for x in p.stdout.splitlines() if x.startswith('RESULT::')),None)
    if payload is None: raise RuntimeError(p.stdout[-2000:])
    out=json.loads(payload)
    for clip,row in out.items():
        row['transcript_stable']=len(set(row['texts']))==1
        row['median_s']=statistics.median(row['times_s'][1:] or row['times_s'])
        row['rtf_x']=durations[clip]/row['median_s']
    return out

results={'script_version':SCRIPT_VERSION,'commit':commit,'ggml_v023':GGML_V023,'cuda_arch':arch,'durations_s':durations,'arms':{}}
for name in ['current','v023']:
    kh.step(f'bench.{name}'); results['arms'][name]=run(name)
base=results['arms']['current']; new=results['arms']['v023']
stable=all(r['transcript_stable'] for arm in results['arms'].values() for r in arm.values())
equal=all(new[c]['texts'][-1]==base[c]['texts'][-1] for c in paths)
results['validation']={'passed':stable and equal,'all_arms_stable':stable,'implementation_equal':equal}
(WORK/'results.json').write_text(json.dumps(results,indent=2)+'\n')
kh.export_ccache_tar()
kh.step('done',validation=results['validation'])
print(json.dumps(results,indent=2),flush=True)
if not results['validation']['passed']: raise RuntimeError('transcript parity failed')
