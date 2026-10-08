#!/usr/bin/env python3
"""Actual CUDA graph-phase profile; no arithmetic/default/model changes."""
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
import soundfile as sf
from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parents[2]
STEP_STUDY = os.environ.get('MIMO_GPU_STEP_STUDY') == '1'


def save(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False)+'\n')


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def worker(build, models, audio, out, enabled):
    build, models, audio, out = map(Path,(build,models,audio,out))
    out.mkdir(parents=True,exist_ok=True)
    os.environ.update(CUDA_VISIBLE_DEVICES='0',CRISPASR_MIMO_ASR_GRAPH_PROFILE='1' if STEP_STUDY else enabled,
                      CRISPASR_MIMO_ASR_GPU_STEP_GRAPH=enabled if STEP_STUDY else '0',
                      CRISPASR_MIMO_ASR_BENCH='1',OMP_NUM_THREADS='4')
    sys.path.insert(0,str(ROOT/'python'))
    from crispasr import Session
    class Params(C.Structure):
        _fields_ = [(name,C.c_int) for name in ('abi_version','n_threads','use_gpu','verbosity','flash_attn','n_gpu_layers')] + [('reserved',C.c_int*6)]
    s = Session.__new__(Session)
    s._lib = C.CDLL(str(next(build.rglob('libcrispasr.so'))))
    s._handle, s._progress_cb_holder = None,None
    s._setup_session_signatures()
    params = Params(2,4,1,1,1,-1)
    s._handle = s._lib.crispasr_session_open_with_params(os.fsencode(models/'mimo-asr-q4_k.gguf'),b'mimo-asr',C.byref(params))
    assert s._handle
    s.backend,s._n_threads = 'mimo-asr',4
    s.set_codec_path(str(models/'mimo-tokenizer-q4_k.gguf'))
    s.set_max_new_tokens(128)
    result = dict(enabled=enabled,calls=[],passed=False)
    with s:
        for repeat in range(8):
            for lang in ('en','zh') if repeat % 2 == 0 else ('zh','en'):
                pcm,rate = sf.read(audio/(lang+'.wav'),dtype='float32')
                assert rate == 16000 and pcm.ndim == 1
                s.set_source_language(lang)
                if STEP_STUDY and repeat == 0:
                    dump = out/'logits'/lang
                    dump.mkdir(parents=True,exist_ok=True)
                    os.environ['CRISPASR_MIMO_ASR_DUMP_STEP_LOGITS'] = str(dump)
                else:
                    os.environ.pop('CRISPASR_MIMO_ASR_DUMP_STEP_LOGITS',None)
                started = time.perf_counter()
                text = ' '.join(seg.text for seg in s.transcribe(pcm)).strip()
                elapsed = time.perf_counter()-started
                assert text and (all(w in text.lower() for w in ('americans','country','ask')) if lang == 'en' else len(re.findall('[\u4e00-\u9fff]',text)) >= 5), (lang,text)
                device = subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_memory',
                                                  '--format=csv,noheader,nounits'],text=True)
                own = [line for line in device.splitlines() if line.split(',')[0].strip() == str(os.getpid())]
                assert own, 'Worker not resident on actual GPU'
                result['calls'].append(dict(language=lang,repeat=repeat,measured=repeat>=2,
                                            seconds=elapsed,text=text,device=own))
                save(out/'calls.json',result)
    os.environ.pop('CRISPASR_MIMO_ASR_DUMP_STEP_LOGITS',None)
    for lang in ('en','zh'):
        prefix = out/('cli-'+lang)
        with (out/('cli-'+lang+'.log')).open('w') as log:
            subprocess.run([str(build/'bin/crispasr'),'--backend','mimo-asr','-m',str(models/'mimo-asr-q4_k.gguf'),
                '--codec-model',str(models/'mimo-tokenizer-q4_k.gguf'),'-t','4','-l',lang,'-fa',
                '--max-new-tokens','128','-f',str(audio/(lang+'.wav')),'-otxt','-of',str(prefix)],
                stdout=log,stderr=subprocess.STDOUT,check=True,timeout=1200)
        expected = next(c['text'] for c in result['calls'] if c['language']==lang)
        assert prefix.with_suffix('.txt').read_text().strip() == expected, 'CLI/session mismatch'
    result['passed'] = True
    save(out/'calls.json',result)


def main():
    out = Path(os.environ['HEAVY_OUT'])
    temp = Path(os.environ['HEAVY_SCRATCH'])/'graph-profile'
    build = Path(os.environ['MIMO_PROFILE_BUILD'])
    audio,models = temp/'audio',temp/'models'
    audio.mkdir(parents=True,exist_ok=True)
    models.mkdir(parents=True,exist_ok=True)
    pins = [
        ('cstr/mimo-asr-GGUF','e2d7dfebf0afd8076771903e92958039c5074eab','mimo-asr-q4_k.gguf','12dbc7cc7a20c7add6ff00bf8b12bca1c46304e0100a5c5a6e74bdecfc57a306'),
        ('cstr/mimo-tokenizer-GGUF','fa380f4c49a8e8c62c02c00d0da5e263fc5b0dcf','mimo-tokenizer-q4_k.gguf','3f3a903b10294ead4ef6a4afec035639fd2113b1d307d42f649a97cc85670e3f')]
    for repo,revision,name,digest in pins:
        path = hf_hub_download(repo,name,revision=revision,local_dir=models)
        assert sha(path) == digest,name
    zh = hf_hub_download('FunAudioLLM/SenseVoiceSmall','example/zh.mp3',revision='3847d57b6bdf2dd8875cb1508d2af43d80a16bf7')
    for lang,source in (('en',ROOT/'samples/jfk.mp3'),('zh',zh)):
        subprocess.run(['ffmpeg','-v','error','-y','-i',str(source),'-ar','16000','-ac','1',str(audio/(lang+'.wav'))],check=True)
    receipt = dict(source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),pins=pins,
                   scope=__doc__,passed=False,arms=[],phase_rows=[],default_changed=False)
    canonical = {}
    for index,enabled in enumerate(('0','1','1','0')):
        arm = out/('arm-'+str(index))
        logpath = out/('arm-'+str(index)+'.log')
        with logpath.open('w') as log:
            subprocess.run([sys.executable,__file__,'--worker',str(build),str(models),str(audio),str(arm),enabled],
                            stdout=log,stderr=subprocess.STDOUT,check=True,timeout=2400)
        result = json.loads((arm/'calls.json').read_text())
        assert result['passed']
        for call in result['calls']:
            lang = call['language']
            canonical.setdefault(lang,call['text'])
            assert canonical[lang] == call['text'], 'Profile toggle changed decoded output'
        text = logpath.read_text()
        assert 'mimo_asr: GPU backend active' in text and 'mimo_tokenizer: RVQ backend=CUDA' in text
        rows = re.findall(r'mimo_asr_graph: path=(\w+) past=(\d+) phase=(\w+) ms=([\d.]+)',text)
        assert bool(rows) == (STEP_STUDY or enabled=='1'), 'Profiling toggle not observed'
        if STEP_STUDY:
            expected_path = 'gpu_cached_step' if enabled=='1' else 'gpu_prefill_step'
            assert any(row[0]==expected_path for row in rows), expected_path
            assert 'step logit dump' not in text, 'Diagnostic dump failed'
        elif enabled == '1':
            assert any(row[0]=='gpu_prefill_step' and row[2]=='graph_build' for row in rows)
        receipt['arms'].append(result)
        receipt['phase_rows'].append(rows)
        save(out/'mimo-graph-profile.json',receipt)
    if STEP_STUDY:
        comparisons = {}
        for lang in ('en','zh'):
            baseline = out/'arm-0'/'logits'/lang
            files = sorted(p.name for p in baseline.glob('*.f32'))
            assert len(files)>5
            comparisons[lang] = {}
            for index in [1,2,3]:
                other = out/('arm-'+str(index))/'logits'/lang
                assert sorted(p.name for p in other.glob('*.f32')) == files
                rows = []
                for name in files:
                    a = np.fromfile(other/name,dtype=np.float32).astype(np.float64)
                    b = np.fromfile(baseline/name,dtype=np.float32).astype(np.float64)
                    assert a.shape == b.shape and len(a)>150000 and np.isfinite(a).all() and np.isfinite(b).all()
                    an,bn = float(np.linalg.norm(a)),float(np.linalg.norm(b))
                    cosine = float(np.dot(a,b)/(an*bn))
                    rel = float(np.linalg.norm(a-b)/bn)
                    row = dict(step=name,cosine=cosine,relative_l2=rel,mine_norm=an,ref_norm=bn,
                               max_abs=float(np.max(np.abs(a-b))),argmax_exact=int(a.argmax())==int(b.argmax()))
                    rows.append(row)
                    comparisons[lang][str(index)] = rows
                    receipt['step_logits'] = comparisons
                    save(out/'mimo-graph-profile.json',receipt)
                    assert cosine>.999999 and rel<.001 and row['argmax_exact'], (lang,index,row)
        receipt['scope'] = 'Opt-in cached GPU T=1 graph versus unchanged legacy prefill decode; exact speech and step logits, same binary/files/GPU.'
    receipt['passed'] = True
    receipt['interpretation'] = 'Host wall phases, including synchronization. Profile toggle measures instrumentation overhead; no speedup is claimed.'
    if STEP_STUDY:
        receipt['interpretation'] = 'Same-binary ABBA on legacy/cached GPU decode. Two warmups, six measured calls/clip/process; logit dumps only during first excluded warmup. Default stays OFF.'
    save(out/'mimo-graph-profile.json',receipt)


if __name__ == '__main__':
    if len(sys.argv)>1 and sys.argv[1]=='--worker':
        worker(*sys.argv[2:])
    else:
        main()
