#!/usr/bin/env python3
"""A fixed independent ASR architecture on all sealed Dia clips; no gate promotion."""
import argparse
import hashlib
import importlib.metadata
import importlib.util
import io
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tarfile
import time
import zipfile

import numpy as np

SOURCE='31243bad24cc746f07d4c8bfdd2d974872cb1803'
MODEL='25a8566e1d0c1e2231d1c762132cd20e0f96a85d16145c3a00adf5d1ac670ead'
TEXTS={'legacy':'[S1] We are committed to making it secure.',
       'jfk':'And so my fellow Americans ask not what your country can do for you ask what you can do for your country.',
       **{f'hello-{s}':'[S1] Hello there, how are you doing today?' for s in [42,123]},
       **{f'fox-{s}':'[S1] The quick brown fox jumps over the lazy dog.' for s in [42,123]}}
OPTIONS=dict(language='en',task='transcribe',temperature=0.,condition_on_previous_text=False,
             initial_prompt=None,fp16=False,verbose=None)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def score(text,expected):
    a=re.findall('[a-z]+',re.sub(r'\[S[12]\]','',expected).lower());b=re.findall('[a-z]+',text.lower())
    row=list(range(len(b)+1))
    for i,x in enumerate(a,1):
        new=[i]
        for j,y in enumerate(b,1):
            new.append(min(new[-1]+1,row[j]+1,row[j-1]+(x!=y)))
        row=new
    return row[-1]/len(a)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--storage',type=Path,default=Path('/mnt/storage/crispasr'))
    p.add_argument('--selftest',action='store_true');args=p.parse_args()
    assert score('We are committed to making it secure.',TEXTS['legacy'])==0
    assert score('',TEXTS['fox-42'])==1
    assert score('The quick brown fix jumps over the lazy dig.',TEXTS['fox-123'])==2/9
    if args.selftest:
        print('FIXED_WHISPER_SCORING_CONTROLS_PASS');return
    base=args.storage/'dia-whisper-crosscheck-20261016'
    out=base/'run-v1';out.mkdir(exist_ok=False)
    cache=base/'models';cache.mkdir(exist_ok=True)
    scratch=base/'scratch';scratch.mkdir(exist_ok=True)
    os.environ.update(TMPDIR=str(scratch),NUMBA_CACHE_DIR=str(scratch/'numba'),TIKTOKEN_CACHE_DIR=str(scratch/'tiktoken'))
    repo=Path(__file__).resolve().parents[2]
    coordinator_source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=repo)
    method=out/'method';method.mkdir();shutil.copyfile(__file__,method/Path(__file__).name)
    assert importlib.metadata.version('openai-whisper')=='20250625'
    package=Path(importlib.util.find_spec('whisper').origin).parent
    archive=base/'source-preparation-v1/original-source.tar.gz'
    source_meta=json.loads((base/'source-preparation-v1/source-download.json').read_text())
    assert source_meta['source']==SOURCE and source_meta['sha256']==sha(archive)
    shutil.copyfile(archive,method/'original-source.tar.gz')
    (method/'original-source-download.json').write_text(json.dumps(source_meta,indent=2)+'\n')
    prefix='whisper-'+SOURCE+'/'
    verified=[]
    with tarfile.open(archive) as tar:
        for member in tar.getmembers():
            if not member.isfile() or not member.name.startswith(prefix+'whisper/'):
                continue
            relative=member.name.removeprefix(prefix+'whisper/')
            if not (relative.endswith('.py') or relative.startswith('assets/')):
                continue
            source=tar.extractfile(member).read();installed=package/relative
            assert installed.is_file() and installed.read_bytes()==source,relative
            verified.append(relative)
        assert {f.relative_to(package).as_posix() for f in package.rglob('*.py')}=={n for n in verified if n.endswith('.py')}
        (method/'Whisper-LICENSE').write_bytes(tar.extractfile(prefix+'LICENSE').read())
    import torch
    import soundfile as sf
    import whisper
    assert Path(whisper.__file__).resolve().parent==package.resolve()
    torch.set_num_threads(4);torch.manual_seed(0)
    assert whisper._MODELS['base.en'].split('/')[-2]==MODEL
    report=dict(source=coordinator_source,recognizer_source=SOURCE,recognizer='original-openai-whisper-base.en',
        weights_sha256=MODEL,options=OPTIONS,threads=4,torch=torch.__version__,device='cpu',
        package_version=importlib.metadata.version('openai-whisper'),source_files_verified=len(verified),
        source_archive=source_meta,input_artifacts={},results={},complete=False,
        speech_acceptance=False,production_defaults_changed=False,gpu_acceptance=False,q4_acceptance=False)
    def checkpoint():
        report['payloads']=[dict(path=f.relative_to(out).as_posix(),bytes=f.stat().st_size,sha256=sha(f))
            for f in sorted(out.rglob('*')) if f.is_file() and f!=out/'receipt.json']
        (out/'receipt.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    def audio(name,data,expected_sha):
        assert hashlib.sha256(data).hexdigest()==expected_sha,name
        original=np.load(io.BytesIO(data),allow_pickle=False)
        assert original.dtype==np.float32 and original.ndim==1 and original.size and np.isfinite(original).all()
        folder=out/name;folder.mkdir()
        (folder/'sealed-input.npy').write_bytes(data)
        sf.write(folder/'input.wav',original,16000 if name=='jfk' else 44100,subtype='FLOAT')
        return folder
    inputs={}
    crossed=args.storage/'dia-asr-crosscheck-20261016/hosted-38930760374-v1'
    broader=args.storage/'dia-broader-speech-20261016/hosted-38913941927-v1'
    try:
        with zipfile.ZipFile(crossed/'raw.zip') as z:
            r=json.loads(z.read('receipt.json'));assert r['source']=='30e75dcdd8df573ed1f73e9c97b17f192b82d7af'
            index={v['path']:v for v in r['payloads']}
            for name in ['legacy','jfk']:
                n='audio/'+name+'.npy';inputs[name]=(audio(name,z.read(n),index[n]['sha256']),TEXTS[name])
            report['input_artifacts']['crossed']=dict(run=38930760374,sha256='fdaa4da6e39a86ca52ddca48de50bc8c93de77f8d7bcfd906aab765517fee416')
        with zipfile.ZipFile(broader/'raw.zip') as z:
            r=json.loads(z.read('receipt.json'));assert r['source']=='17a7aedc508be885cce0024199bd48525d3fc41b'
            index={v['path']:v for v in r['payloads']}
            for key in ['hello-42','hello-123','fox-42','fox-123']:
                for label,suffix in [('original','original/canonical-pcm.npy'),('native','serial-f32-portable/native-pcm.npy'),
                    ('original-dac','serial-f32-portable/original-dac-pcm.npy'),('cli','serial-f32-portable/cli-pcm.npy')]:
                    name=key+'-'+label;n='cases/'+key+'/arms/'+suffix
                    inputs[name]=(audio(name,z.read(n),index[n]['sha256']),TEXTS[key])
            report['input_artifacts']['broader']=dict(run=38913941927,sha256='75d9f3ec9d4fa96f1e284d2cf602afcf8b288d5cbdaa5bea5bf88532eda26c90')
        assert len(inputs)==18
        report['input_validation_scope']='Each selected payload SHA verified against retained previously SHA-verified official parent receipt; multiGB parent archives not rehashed in this run'
        (out/'ffmpeg-version.log').write_bytes(subprocess.check_output(['ffmpeg','-version']))
        checkpoint()
        print('ORIGINAL_SOURCE_AND_ALL18_INPUTS_VERIFIED',len(verified),flush=True)
        model=whisper.load_model('base.en',device='cpu',download_root=str(cache)).eval()
        assert sha(cache/'base.en.pt')==MODEL and not next(model.parameters()).is_cuda
        for name,(folder,text) in inputs.items():
            sample=whisper.load_audio(str(folder/'input.wav'))
            assert sample.dtype==np.float32 and sample.ndim==1 and sample.size and np.isfinite(sample).all()
            np.save(folder/'actual-16khz.npy',sample,allow_pickle=False)
            start=time.monotonic()
            result=model.transcribe(sample,**OPTIONS)
            elapsed=time.monotonic()-start
            (folder/'raw-whisper.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
            row=dict(transcript=result['text'].strip(),wer=score(result['text'],text),expected_text=text,
                sealed_input_sha256=sha(folder/'sealed-input.npy'),actual_16khz_sha256=sha(folder/'actual-16khz.npy'),
                seconds=sample.size/16000,elapsed_seconds=elapsed)
            report['results'][name]=row;checkpoint()
            print('WHISPER_CELL',name,json.dumps(row),flush=True)
        report['complete']=set(report['results'])==set(inputs)
        report['controls_wer_zero']=all(report['results'][k]['wer']==0 for k in ['legacy','jfk'])
        report['broader_all_wer_zero']=all(v['wer']==0 for k,v in report['results'].items() if k not in ['legacy','jfk'])
    finally:
        checkpoint()
    assert report['complete']
    print('WHISPER_CROSS_RECOGNITION_COMPLETE',report['controls_wer_zero'],report['broader_all_wer_zero'],flush=True)


if __name__=='__main__':main()
