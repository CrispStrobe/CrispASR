#!/usr/bin/env python3
"""Offline CPU-only #589 prefill capture/replay. Generated bundle is self-contained.
No submission, downloader, GPU, or audio task is contained in this script.
"""
import base64
import fcntl
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tarfile
import time

BUNDLE_B64 = '__BUNDLE_B64__'
BUNDLE_SHA256 = '__BUNDLE_SHA256__'
SOURCE_COMMIT = '__SOURCE_COMMIT__'
GGML_COMMIT = '1b972bb754aa5f83e2d872ccf37a335717f6658d'
MODEL_SHA256 = 'ff68a234c5a1069e379e4d3fdbf29bb07ec4981d0ad8841bb19bc442991413f3'
MODEL_BYTES = 7225646272
ORACLE_IDS = [15226,279,1467,3897,553,5257,21326,1119,8806,2550,11,34888,279,12460,7743,315,1817,19511,18601,624,2918,1946,510,29073,220,15,25,1096,7699,572,7907,553,20443,11229,624,38741,2550,510,151652]


def sha(path,deadline=None):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(8<<20),b''):
            if deadline is not None and time.monotonic()>deadline:raise TimeoutError('streamed hash timeout')
            h.update(block)
    return h.hexdigest()


def unpack(payload,destination):
    with tarfile.open(fileobj=io.BytesIO(payload),mode='r:gz') as archive:
        members=archive.getmembers()
        if sum(m.size for m in members)>256<<20:raise RuntimeError('embedded source/runtime exceeds256MiB')
        for member in members:
            p=Path(member.name)
            if p.is_absolute() or '..' in p.parts or not (member.isfile() or member.isdir()):
                raise RuntimeError('unsafe source member')
        archive.extractall(destination,members=members,filter='data')


def main():
    work=Path('/kaggle/temp/issue589-prefill-cpu')
    work.mkdir(parents=True,exist_ok=False)
    output=Path('/kaggle/working/issue589-prefill-cpu-results');output.mkdir(exist_ok=False)
    lock=open('/kaggle/working/issue589-prefill-cpu.lock','a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    os.set_inheritable(lock.fileno(),True)
    os.environ.update(CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
        HF_DATASETS_OFFLINE='1',PIP_NO_INDEX='1',PYTHONNOUSERSITE='1',OMP_NUM_THREADS='1',
        OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',USE_TF='0',USE_FLAX='0')
    payload=base64.b64decode(BUNDLE_B64)
    if hashlib.sha256(payload).hexdigest()!=BUNDLE_SHA256:raise RuntimeError('source bundle hash mismatch')
    unpack(payload,work)
    manifest=json.loads((work/'bundle-manifest.json').read_text())
    if manifest['source_commit']!=SOURCE_COMMIT or manifest['ggml_commit']!=GGML_COMMIT:
        raise RuntimeError('source identity mismatch')
    for name,digest in manifest['files'].items():
        if sha(work/name)!=digest:raise RuntimeError('embedded file hash mismatch: '+name)
    root=work/'source';vendor=work/'vendor'
    sys.path.insert(0,str(vendor));sys.path.insert(0,str(root/'tools/kaggle'))
    import kaggle_harness as kh
    kh.init_progress(output/'progress.jsonl',hf_progress_repo='')
    import torch
    import transformers
    from transformers.models.qwen2 import modeling_qwen2
    torch.set_num_threads(1)
    if torch.cuda.is_available():raise RuntimeError('CPU-only execution required')
    # Only the actual model dataset is mounted. Both Kaggle mount layouts work.
    candidates=list({p.resolve() for p in Path('/kaggle/input').rglob('vibevoice-7b-q4_k.gguf') if p.is_file()})
    if len(candidates)!=1:raise RuntimeError('require one exact mounted GGUF')
    model=candidates[0]
    if model.stat().st_size!=MODEL_BYTES:raise RuntimeError('wrong mounted pack size')
    results=dict(scope='direct CPU base-backend39-token prefill only; no CUDA/Metal equivalence or quality claim',
        source_commit=SOURCE_COMMIT,ggml_commit=GGML_COMMIT,bundle_sha256=BUNDLE_SHA256,
        model_sha256=MODEL_SHA256,model_bytes=MODEL_BYTES,torch_version=torch.__version__,
        transformers_version=transformers.__version__,upstream_qwen2_sha256=sha(modeling_qwen2.__file__),
        config_provenance=manifest['config_provenance'],stages=[])
    def record():
        (output/'result.json').write_text(json.dumps(results,indent=2)+'\n')
    def guard():
        mem=dict(line.split(':',1) for line in Path('/proc/meminfo').read_text().splitlines())
        state=dict(free_bytes=shutil.disk_usage(work).free,available_kib=int(mem['MemAvailable'].split()[0]),
            load1=os.getloadavg()[0],load_max=max(8.,2*os.cpu_count()))
        if state['free_bytes']<=12*2**30 or state['available_kib']<=22*2**20 or state['load1']>state['load_max']:
            raise RuntimeError('resource guard failed: '+json.dumps(state))
        if sum(p.stat().st_size for p in output.rglob('*') if p.is_file())>256<<20:
            raise RuntimeError('retained output exceeds256MiB')
        return state
    def rss_tree(pid):
        total=0
        try:
            status=dict(line.split(':',1) for line in Path(f'/proc/{pid}/status').read_text().splitlines() if ':' in line)
            total=int(status.get('VmRSS','0 kB').split()[0])
            for child in Path(f'/proc/{pid}/task/{pid}/children').read_text().split():total+=rss_tree(int(child))
        except (FileNotFoundError,ProcessLookupError):pass
        return total
    def run(name,command,timeout):
        stage=dict(name=name,command=[str(x) for x in command],guards=guard(),timeout_seconds=timeout)
        peak=output/(name+'.peak-kib');started=time.monotonic()
        env=dict(os.environ,PYTHONPATH=str(vendor),TMPDIR=str(work))
        process=None
        try:
            with (output/(name+'.log')).open('wb') as log:
                process=subprocess.Popen(['/usr/bin/time','-f','%M','-o',str(peak),*map(str,command)],
                    env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,pass_fds=(lock.fileno(),))
                stage['observed_process_tree_peak_kib']=0
                while process.poll() is None:
                    observed=rss_tree(process.pid)
                    stage['observed_process_tree_peak_kib']=max(stage['observed_process_tree_peak_kib'],observed)
                    if observed>18*2**20:
                        os.killpg(process.pid,signal.SIGKILL);process.wait()
                        stage['memory_watchdog']=True;raise RuntimeError(name+' exceeded18GiB RSS watchdog')
                    if time.monotonic()-started>timeout:raise subprocess.TimeoutExpired(command,timeout)
                    time.sleep(.25)
                stage['exit_code']=process.returncode
                if stage['exit_code']:raise RuntimeError(name+' failed')
        except subprocess.TimeoutExpired:
            stage['timeout']=True
            os.killpg(process.pid,signal.SIGKILL);process.wait();raise
        finally:
            stage['elapsed_seconds']=time.monotonic()-started
            if peak.exists():stage['peak_rss_kib']=peak.read_text().strip()
            results['stages'].append(stage);record()
    record();guard()
    # Hash before any model inference; streaming verification never copies GGUF.
    if sha(model,deadline=time.monotonic()+600)!=MODEL_SHA256:raise RuntimeError('wrong mounted pack SHA256')
    required=['cmake','ninja','c++','/usr/bin/time']
    if any(shutil.which(name) is None for name in required):raise RuntimeError('offline build tool missing; no install attempted')
    build=work/'build'
    flags=['-DBUILD_SHARED_LIBS=OFF','-DGGML_NATIVE=OFF','-DGGML_BLAS=OFF','-DGGML_CUDA=OFF',
        '-DGGML_VULKAN=OFF','-DGGML_METAL=OFF','-DGGML_SYCL=OFF','-DGGML_OPENCL=OFF',
        '-DGGML_BUILD_TESTS=OFF','-DGGML_BUILD_EXAMPLES=OFF','-DFETCHCONTENT_FULLY_DISCONNECTED=ON',
        '-DCRISPASR_BUILD_TESTS=OFF','-DCRISPASR_BUILD_SERVER=OFF']
    run('configure',['cmake','-G','Ninja','-S',root/'ggml','-B',build,'-DCMAKE_BUILD_TYPE=Release',*flags],60)
    with kh.build_heartbeat('cpu.ggml.build'):
        run('build',['cmake','--build',build,'--target','ggml','ggml-cpu','ggml-base','-j1'],600)
    libs=build/'src';binary=work/'prefill-only'
    run('compile',['c++','-std=c++17','-O2','-ffunction-sections','-fdata-sections','-Wl,--gc-sections',
        '-I',root/'src','-I',root/'ggml/include','-I',root/'ggml/src',
        root/'tools/kaggle/issue589-prefill-cpu/capture.cpp',root/'src/core/gguf_loader.cpp',
        libs/'libggml.a',libs/'libggml-cpu.a',libs/'libggml-base.a','-fopenmp','-ldl','-lpthread','-lm','-o',binary],300)
    results['binary_sha256']=sha(binary);results['backend']='direct vibevoice_synthesize, use_gpu=false, notCLI'
    native=output/'native';native.mkdir()
    # Exactly one native inference process; the driver hard-codes exact disclosure,
    # seed42/steps20/CFG1.3 and original flags, including native flash attention ON.
    run('native-prefill-only',[binary,model,native],1800)
    spec=importlib.util.spec_from_file_location('capture_contract',root/'tools/vibevoice_first_frame_capture.py')
    capture=importlib.util.module_from_spec(spec);spec.loader.exec_module(capture)
    results['captured_layers']=capture.validate_prefill_only(native)
    import struct
    ids=list(struct.unpack('<39i',(native/'tts_prompt_ids.bin').read_bytes()))
    if ids!=ORACLE_IDS:raise RuntimeError('39-token prompt differs from retained exact disclosure oracle')
    if sum(p.stat().st_size for p in native.iterdir())>64<<20:raise RuntimeError('native capture exceeds64MiB')
    for precision in ('f32','f16'):
        run('replay-'+precision,[sys.executable,root/'tools/vibevoice_prefill_replay.py','--gguf',model,
            '--expected-sha256',MODEL_SHA256,'--config',work/'blueprint-config.json','--capture',native,
            '--output',output/('upstream-'+precision),'--kv-precision',precision],1800)
    results['status']='complete_measurements_only'
    results['limitations']=['blueprint config is not proven original conversion config',
        'CPU native flash/eager arithmetic differs; no originalCUDA/Metal equivalence',
        'no diffusion/audio/quality result; no comparison threshold changed']
    results['payload_sha256']={str(p.relative_to(output)):sha(p) for p in output.rglob('*.bin')}
    record()
    print(json.dumps(results,indent=2))


if __name__=='__main__':
    try:main()
    except Exception as error:
        Path('/kaggle/working/issue589-prefill-failure.json').write_text(json.dumps(dict(status='failed_or_blocked',error=repr(error)),indent=2)+'\n')
        raise
