#!/usr/bin/env python3
"""Pinned native Breeze CLI/C ABI/server speech diagnostics on actual CUDA.

Retain all PCM, ASR, repeatability and closed-loop speaker evidence. This does
not waive the separately failed exact-code gate or replace an original oracle.
"""
import ctypes as C
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
import wave
import tarfile
import zipfile

VERSION = '2026-10-12.breeze-cuda-speech.6'
SOURCE = 'e6b4ce362d69d7965f57896466228af71b2ece1c'
# Exact CI archive: all member SHA/size checks and anonymous mirror verified.
BUNDLE_REV = '5792dad83d36d5633ea64b623fa6f714ca6488fe'
BUNDLE_PATH = 'breeze-guided-20261012/cuda-speech-e6b4ce36/breeze-cuda-speech-validation.tar.gz'
BUNDLE_SHA = 'b3aa2842ed04752ab9f04800eb2bdcac586fabf238f8a3fa5aac4a2148febb6f'
OUT = Path('/kaggle/working')
SCRATCH = Path('/kaggle/temp/breeze-cuda-speech')
ROOT = SCRATCH / 'sdk'
SCRATCH.mkdir(parents=True, exist_ok=True)
os.environ.update(TMPDIR=str(SCRATCH), HF_HOME=str(SCRATCH/'hf'), HF_XET_CACHE=str(SCRATCH/'xet'),
                  OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='4')
assert len(BUNDLE_SHA) == 64 and len(BUNDLE_REV) == 40, 'Build/mirror pins are pending; do not launch'
hardware = subprocess.check_output(['nvidia-smi', '--query-gpu=name,compute_cap,memory.total,driver_version',
                                   '--format=csv,noheader'], text=True).strip()
assert hardware and all(float(row.split(',')[1]) in (6.0,7.5) for row in hardware.splitlines()), 'Requires actual sm60/sm75 GPU'
for command in ([] if (ROOT/'.git').exists() else [['git','init',ROOT],['git','-C',ROOT,'remote','add','origin','https://github.com/CrispStrobe/CrispASR.git'],
                ['git','-C',ROOT,'fetch','--depth=1','origin',SOURCE],['git','-C',ROOT,'checkout','FETCH_HEAD']]):
    subprocess.run(list(map(str,command)),check=True)
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip() == SOURCE
sys.path.insert(0,str(ROOT/'tools/kaggle'))
import kaggle_harness as kh
kh.init_progress()
kh.provenance(VERSION,ROOT)
os.environ['HF_TOKEN'] = kh.resolve_hf_token(require=True)
subprocess.run([sys.executable,'-m','pip','install','-q','numpy','huggingface_hub','gguf',
                'resemblyzer==0.1.4','setuptools==80.9.0'],check=True)
# The speaker encoder uses preinstalled Torch in a separate GPU process,
# keeping its host CUDA runtime out of native package validation.
from huggingface_hub import hf_hub_download
PACKAGE = Path(hf_hub_download('cstr/crispasr-regression-fixtures',BUNDLE_PATH,revision=BUNDLE_REV,repo_type='dataset'))

TEXT = 'I found the file you asked about and opened it.'
REFERENCE_TEXT = 'The quick brown fox jumps over the lazy dog.'
INSTRUCTION = 'Speak clearly and calmly.'
RATE, SEED, LIMIT = 24000, 7, 128
ATTESTATION = 'Synthetic automated test fixture; marking handled by test harness.'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(data)
    return h.hexdigest()


GPU_PROCESSES = {}


def gpu_memory(pid):
    rows = subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_gpu_memory',
                                    '--format=csv,noheader,nounits'],text=True)
    return max([int(row.split(',')[1].strip()) for row in rows.splitlines()
                if row.strip() and int(row.split(',')[0].strip()) == pid] or [0])


def run(command, label, env=None, min_gpu_mib=256):
    with (OUT / (label + '.log')).open('w') as log:
        process = subprocess.Popen(list(map(str,command)),cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,env=env)
        deadline=time.monotonic()+7200
        peak=0
        while process.poll() is None:
            peak=max(peak,gpu_memory(process.pid))
            if time.monotonic() >= deadline:
                process.kill();process.wait();raise TimeoutError(label)
            time.sleep(.5)
    GPU_PROCESSES[label]=dict(pid=process.pid,peak_mib=peak,returncode=process.returncode)
    if label in ('gpu-stage-dump', 'gpu-guided-dump'):
        # The codec also allocates on CUDA. Its presence alone cannot prove
        # that Breeze selected CUDA before dynamic plugins were registered.
        actual_backend = [line.split(' = ', 1)[1].strip()
                          for line in (OUT / (label + '.log')).read_text().splitlines()
                          if line.startswith('breeze_tts_2: compute backend = ')]
        GPU_PROCESSES[label]['breeze_compute_backends'] = actual_backend
        (OUT/'gpu-subprocesses.json').write_text(json.dumps(GPU_PROCESSES,indent=2)+'\n')
        assert actual_backend and all(name.startswith('CUDA') for name in actual_backend), (label, actual_backend)
        assert peak >= 4096, 'Pinned F16 Breeze weights must reside on GPU, beyond codec allocation'
    (OUT/'gpu-subprocesses.json').write_text(json.dumps(GPU_PROCESSES,indent=2)+'\n')
    print(label,'rc',process.returncode,'GPU MiB',peak,flush=True)
    assert process.returncode == 0 and peak>=min_gpu_mib,(label,process.returncode,peak)


def write_wav(path, pcm):
    import numpy as np
    with wave.open(str(path), 'wb') as wav:
        wav.setparams((1, 2, RATE, 0, 'NONE', 'not compressed'))
        wav.writeframes((np.clip(pcm, -1, 1) * 32767).astype('<i2').tobytes())


def read_wav(path):
    import numpy as np
    with wave.open(str(path)) as wav:
        assert (wav.getframerate(), wav.getnchannels(), wav.getsampwidth()) == (RATE, 1, 2)
        return np.frombuffer(wav.readframes(wav.getnframes()), dtype='<i2').astype(np.float32) / 32768


def wer(expected, actual):
    words = lambda value: re.findall('[a-z]+', value.lower())
    target, heard = words(expected), words(actual)
    previous = list(range(len(heard) + 1))
    for index, word in enumerate(target, 1):
        row = [index]
        for j, other in enumerate(heard, 1):
            row.append(min(row[-1]+1, previous[j]+1, previous[j-1]+(word != other)))
        previous = row
    return previous[-1] / len(target)


def main():
    import numpy as np
    from huggingface_hub import hf_hub_download
    OUT.mkdir(parents=True, exist_ok=True)
    SCRATCH.mkdir(parents=True, exist_ok=True)
    os.environ.update(TMPDIR=str(SCRATCH), OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='4')
    assert sha(PACKAGE) == BUNDLE_SHA
    with tarfile.open(PACKAGE) as archive:
        archive.extractall(SCRATCH, filter='data')
    bundle = SCRATCH / 'bundle'
    # ELF dependency search is fixed at process startup. Restart after the
    # verified extraction so host CUDA cannot replace the bundled runtime.
    if os.environ.get('CRISPASR_BREEZE_BUNDLED_RUNTIME') != '1':
        native_env = dict(os.environ, CRISPASR_BREEZE_BUNDLED_RUNTIME='1',
                         CRISPASR_HOST_LIBRARY_PATH=os.environ.get('LD_LIBRARY_PATH',''))
        native_env['LD_LIBRARY_PATH'] = str(bundle)+':'+native_env['CRISPASR_HOST_LIBRARY_PATH']
        os.execve(sys.executable, [sys.executable, str(Path(__file__).resolve())], native_env)

    provenance = json.loads((bundle/'provenance.json').read_text())
    assert provenance['source'] == SOURCE and provenance['build_complete'] and not provenance['gpu_executed']
    for name,row in provenance['files'].items():
        assert sha(bundle/name) == row['sha256'] and (bundle/name).stat().st_size == row['bytes'],name
    assert not list(bundle.glob('libcuda.so*')), 'Driver/stub must not be bundled'
    sys.path.insert(0, str(bundle / 'python'))
    from crispasr import Session
    receipt = dict(source=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                   device=hardware, worker_version=VERSION, entry_sha256=sha(Path(__file__)),
                   archive_revision=BUNDLE_REV, archive_sha256=BUNDLE_SHA, gpu_executed=False, complete=False, speech_pass=False,
                   original_generated_code_gate_waived=False, expected=TEXT,
                   instruction=INSTRUCTION, seed=SEED, max_frames=LIMIT, cases={}, failures=[])
    def save():
        (OUT / 'validation.json').write_text(json.dumps(receipt, indent=2)+'\n')
    def retain(name, pcm, expected=TEXT):
        assert pcm.ndim == 1 and pcm.size and np.isfinite(pcm).all(), name
        duration = len(pcm) / RATE
        rms = float(np.sqrt(np.mean(pcm.astype(np.float64)**2)))
        assert duration > 1 and rms > .001, (name, duration, rms)
        np.save(OUT / (name+'.npy'), pcm)
        receipt['cases'][name] = dict(samples=len(pcm), rate=RATE, duration=duration, rms=rms,
                                     expected=expected, pcm_sha256=sha(OUT / (name+'.npy')))
        save()
    try:
        binary, library = bundle/'crispasr', bundle/'libcrispasr.so'
        receipt.update(cli_sha256=sha(binary), library_sha256=sha(library),
                       cuda_plugin_sha256=sha(bundle/'libggml-cuda.so'))
        models = SCRATCH / 'models'
        models.mkdir(exist_ok=True)
        pins = [
            ('breeze','cstr/breeze-tts-2-GGUF','5314fe9ae3ec6eac1a9238c4f6888fdc9fda0e83',
             'breeze-tts-2-f16.gguf','08ea3fbffe9701a3d04a4a63554876841d923fdd21e20add36d28f3ac3447da7'),
            ('codec','cstr/qwen3-tts-tokenizer-12hz-GGUF','344e279551177654762e5c7d39e217834325f222',
             'qwen3-tts-tokenizer-12hz.gguf','70dc95dbfdd9aa5d9d406236ff771d061bf17b0cda02a72513953355606e719b')]
        paths = {}
        for label, repo, revision, filename, digest in pins:
            path = Path(hf_hub_download(repo,filename,revision=revision,local_dir=models))
            assert sha(path) == digest
            paths[label] = path
            receipt.setdefault('models',{})[label] = dict(repo=repo,revision=revision,filename=filename,
                                                         sha256=digest,bytes=path.stat().st_size)
        def session():
            s = Session(str(paths['breeze']),lib_path=str(library),backend='bt2-tts',n_threads=4)
            memory=gpu_memory(os.getpid())
            maps=Path('/proc/self/maps').read_text()
            receipt['loaded_cuda_runtime_paths'] = sorted({line.split()[-1] for line in maps.splitlines() if any(name in line for name in ['libcudart','libcublas'])})
            save()
            assert str(bundle/'libggml-cuda.so') in maps and memory>=256, 'C ABI must load actual bundled CUDA plugin and allocate on GPU'
            receipt.setdefault('cabi_cuda_allocations_mib',[]).append(memory)
            for name in ['libcudart.so.12','libcublas.so.12','libcublasLt.so.12']:
                assert str(bundle/name) in maps, 'C ABI must use the packaged CUDA user runtime: '+name
            receipt['gpu_executed']=True;save()
            s.accept_marking_responsibility(ATTESTATION)
            assert s.output_sample_rate() == RATE and s.input_sample_rate() == RATE
            s.set_temperature(.9,SEED)
            assert s._lib.crispasr_session_set_temperature(s._handle,.9,SEED) == 0
            s.set_max_new_tokens(LIMIT)
            return s
        def reseed(s, seed=SEED):
            s.set_tts_seed(seed)
            assert s._lib.crispasr_session_set_tts_seed(s._handle,seed) == 0
        def cfg(s, scale):
            f=s._lib.crispasr_session_set_tts_cfg_scale
            f.argtypes=[C.c_void_p,C.c_float];f.restype=C.c_int
            assert f(s._handle,scale) == 0
        # A generated reference, a different target, and its no-reference baseline.
        with session() as s:
            reseed(s,11)
            reference=s.synthesize_raw(REFERENCE_TEXT)
            retain('reference',reference,REFERENCE_TEXT)
        voice_dir=SCRATCH/'voices';voice_dir.mkdir()
        voice=voice_dir/'synthetic.wav';write_wav(voice,reference)
        (voice_dir/'synthetic.txt').write_text(REFERENCE_TEXT+'\n')
        with session() as s:
            sequence=[('abi-plain','',1),('abi-design',INSTRUCTION,4),
                      ('abi-plain-repeat','',1),('abi-design-repeat',INSTRUCTION,4),
                      ('abi-negative',INSTRUCTION,0)]
            for label,instruction,scale in sequence:
                s.set_instruct(instruction);cfg(s,scale);reseed(s)
                retain(label,s.synthesize_raw(TEXT))
            for a,b in [('abi-plain','abi-plain-repeat'),('abi-design','abi-design-repeat'),
                        ('abi-plain','abi-negative')]:
                assert np.array_equal(np.load(OUT/(a+'.npy')),np.load(OUT/(b+'.npy'))),(a,b)
            s.set_voice(str(voice),ref_text=REFERENCE_TEXT)
            for label,instruction,scale in [('abi-clone','',1),('abi-direction',INSTRUCTION,4),
                                            ('abi-clone-repeat','',1)]:
                s.set_instruct(instruction);cfg(s,scale);reseed(s)
                retain(label,s.synthesize_raw(TEXT))
            assert np.array_equal(np.load(OUT/'abi-clone.npy'),np.load(OUT/'abi-clone-repeat.npy'))
        for label,ref,ins in [('plain',False,False),('clone',True,False),('design',False,True),
                              ('direction',True,True)]:
            path=OUT/('cli-'+label+'.wav')
            command=[binary,'--backend','bt2-tts','-m',paths['breeze'],'--codec-model',paths['codec'],
                     '-t','4','--seed',str(SEED),'--max-new-tokens',str(LIMIT),
                     '--accept-marking-responsibility','--no-spoken-disclaimer','--no-watermark',
                     '--tts',TEXT,'--tts-output',path]
            if ref:command += ['--voice',voice,'--ref-text',REFERENCE_TEXT,'--i-have-rights']
            if ins:command += ['--tts-instruct',INSTRUCTION,'--tts-cfg-scale','4']
            run(command,'cli-'+label)
            retain('cli-'+label,read_wav(path))
        # Reuse one actual server backend across plain/reference/instruction modes.
        log=(OUT/'server.log').open('w')
        proc=subprocess.Popen([str(x) for x in [binary,'--server','--port','11449','--backend','bt2-tts',
            '-m',paths['breeze'],'--codec-model',paths['codec'],'-t','4',
            '--max-new-tokens',LIMIT,'--voice-dir',voice_dir,'--accept-marking-responsibility']],
            cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        url='http://127.0.0.1:11449'
        def request(fields):
            body=dict(input=TEXT,response_format='f32',seed=SEED,spoken_disclaimer=False,
                      marking_attestation=ATTESTATION,consent_attestation=ATTESTATION)
            body.update(fields)
            req=urllib.request.Request(url+'/v1/audio/speech',data=json.dumps(body).encode(),
                                       headers={'Content-Type':'application/json'})
            try:
                with urllib.request.urlopen(req,timeout=1800) as response:return response.status,response.read()
            except urllib.error.HTTPError as error:return error.code,error.read()
        try:
            deadline=time.monotonic()+600
            while True:
                assert proc.poll() is None,'Server exited before ready'
                try:
                    with urllib.request.urlopen(url+'/health',timeout=2):break
                except (urllib.error.URLError,OSError):
                    assert time.monotonic()<deadline,'Server startup timeout'
                    time.sleep(1)
            server_memory=gpu_memory(proc.pid)
            receipt['server_cuda_allocation_mib']=server_memory;save()
            assert server_memory>=256, 'Server must allocate on actual CUDA'
            sequence=[('plain',{}),('clone',{'voice':'synthetic'}),
                      ('design',{'instructions':INSTRUCTION,'cfg_scale':4}),
                      ('direction',{'voice':'synthetic','instructions':INSTRUCTION,'cfg_scale':4}),
                      ('plain-repeat',{}),('clone-repeat',{'voice':'synthetic'}),
                      ('negative',{'instructions':INSTRUCTION,'cfg_scale':0})]
            for label,fields in sequence:
                status,body=request(fields);assert status==200,(label,status,body[:200])
                assert len(body)%4==0
                retain('server-'+label,np.frombuffer(body,dtype='<f4').copy())
            status,body=request({'voice':'no-such-voice-xyz'})
            receipt['unknown_voice_status']=status
            assert status>=400,'Unknown voice silently served default speech'
            for a,b in [('server-plain','server-plain-repeat'),('server-clone','server-clone-repeat'),
                        ('server-plain','server-negative')]:
                assert np.array_equal(np.load(OUT/(a+'.npy')),np.load(OUT/(b+'.npy'))),(a,b)
            status,body=request({});assert status==200
            assert np.array_equal(np.frombuffer(body,dtype='<f4'),np.load(OUT/'server-plain.npy'))
        finally:
            proc.terminate()
            try:proc.wait(timeout=30)
            except subprocess.TimeoutExpired:proc.kill();proc.wait()
            log.close()
        receipt['surface_contracts_pass']=True;save()
        # Run recognition after every TTS context has been released.
        asr=Path(hf_hub_download('cstr/nemotron-3.5-asr-streaming-GGUF',
             'nemotron-3.5-asr-streaming-0.6b-f16.gguf',revision='bbd95a9ca5fa0dfca3312a122dfc45a2b578b9c2'))
        assert sha(asr)=='14cbc98df495d8ab452c6a4ac367136d94ab5da55e73afaa96ba4f749dd95d09'
        receipt['asr_model']=dict(precision='F16; independent stage/transcript controls passed',revision='bbd95a9ca5fa0dfca3312a122dfc45a2b578b9c2',sha256=sha(asr))
        with Session(str(asr),lib_path=str(library),backend='nemotron',n_threads=4) as s:
            receipt['asr_cuda_allocation_mib']=gpu_memory(os.getpid());save()
            assert receipt['asr_cuda_allocation_mib']>=256, 'Recognizer must allocate on actual CUDA'
            with wave.open(str(ROOT/'samples/jfk.wav')) as wav:
                assert (wav.getframerate(),wav.getnchannels(),wav.getsampwidth())==(16000,1,2)
                control=np.frombuffer(wav.readframes(wav.getnframes()),dtype='<i2').astype(np.float32)/32768
            expected='And so my fellow Americans ask not what your country can do for you ask what you can do for your country'
            actual=' '.join(segment.text for segment in s.transcribe(control,sample_rate=16000,language='en'))
            receipt['asr_control']=dict(expected=expected,transcript=actual,wer=wer(expected,actual),wer_max=.2,
                                       audio_sha256=sha(ROOT/'samples/jfk.wav'))
            save();assert receipt['asr_control']['wer']<=.2,'Independent recognizer JFK control failed'
            for name,row in receipt['cases'].items():
                pcm=np.load(OUT/(name+'.npy'),allow_pickle=False)
                actual=' '.join(segment.text for segment in s.transcribe(pcm,sample_rate=RATE,language='en'))
                row.update(transcript=actual,wer=wer(row['expected'],actual),wer_max=.2)
                if row['wer']>.2:receipt['failures'].append(name+' ASR WER exceeds .2')
                print(name,actual,'WER',row['wer'],flush=True);save()
        speaker_code = r"""
import torch,numpy as np,json,hashlib,pathlib,sys,importlib.metadata,resemblyzer
from resemblyzer import VoiceEncoder,preprocess_wav
assert torch.cuda.is_available()
p=pathlib.Path(sys.argv[1]);encoder=VoiceEncoder(device='cuda');vectors={}
assert all(parameter.is_cuda for parameter in encoder.parameters())
weight_bytes=sum(parameter.numel()*parameter.element_size() for parameter in encoder.parameters())
torch.cuda.reset_peak_memory_stats()
for name in ['reference','abi-plain','abi-clone']:
 vectors[name]=encoder.embed_utterance(preprocess_wav(np.load(p/(name+'.npy')),source_sr=24000))
 np.save(p/('speaker-'+name+'.npy'),vectors[name])
torch.cuda.synchronize()
peak_bytes=torch.cuda.max_memory_allocated()
assert peak_bytes > weight_bytes, 'Actual CUDA inference activations must exceed resident model weights'
r=vectors['reference'];result=dict(cuda_weights_bytes=weight_bytes,peak_cuda_allocated_bytes=peak_bytes,all_parameters_cuda=True,package_version=importlib.metadata.version('resemblyzer'),
 model_sha256=hashlib.sha256((pathlib.Path(resemblyzer.__file__).parent/'pretrained.pt').read_bytes()).hexdigest(),
 torch=torch.__version__,device=torch.cuda.get_device_name(),
 clone_reference=float(np.dot(vectors['abi-clone'],r)),baseline_reference=float(np.dot(vectors['abi-plain'],r)))
(p/'speaker.json').write_text(json.dumps(result,indent=2)+'\n')
"""
        speaker_env = dict(os.environ, LD_LIBRARY_PATH=os.environ['CRISPASR_HOST_LIBRARY_PATH'])
        run([sys.executable,'-c',speaker_code,OUT],'gpu-speaker',env=speaker_env,min_gpu_mib=1)
        speaker=json.loads((OUT/'speaker.json').read_text())
        receipt['speaker_encoder']=speaker
        similarities={key:speaker[key] for key in ['clone_reference','baseline_reference']}
        receipt['speaker']=similarities
        if similarities['clone_reference']<=similarities['baseline_reference']:
            receipt['failures'].append('Closed-loop clone speaker similarity did not improve')
        receipt['speech_pass']=not receipt['failures'];save()
        stage_controls(bundle,paths,receipt,save)
        receipt['complete']=True;save()
        (OUT/'summary.md').write_text('CUDA Breeze surface and strict stage diagnostics completed. See validation.json for '
            'actual contracts, every ASR transcript and closed-loop speaker result. '
            'This does not waive the independently failed exact-code gate.\n')
        assert receipt['speech_pass'] and receipt['stages_pass'] and receipt['guided_exact_pass'],receipt['failures']
    except BaseException as error:
        receipt['error']=f'{type(error).__name__}: {error}';save();raise


def stage_controls(bundle, paths, receipt, save):
    import numpy as np
    from huggingface_hub import hf_hub_download
    archive=Path(hf_hub_download('cstr/crispasr-regression-fixtures',
        'breeze-guided-20261011/matched-budget-failed-parity-proof.zip',repo_type='dataset',
        revision='27896a1fab69d45c8b246eb659823f92a5e0e458'))
    assert sha(archive)=='13d5eed11955e07f2e5fc2f681df123c1b2b35d2f11998416571dbf962d54f33'
    proof=SCRATCH/'oracle';proof.mkdir()
    with zipfile.ZipFile(archive) as z:z.extractall(proof)
    reference=proof/'reference';oracle=json.loads((reference/'validation.json').read_text())
    assert oracle['complete'] and oracle['max_frames']==24
    assert oracle['upstream']=='58ec70ce5fa4cc361bdebf77ec40d1365da00ab2'
    assert oracle['model_revision']=='3e28c5151381a722f1d8661b4118c298caa77aa4'
    fixture=reference/'breeze-dumps'
    for name,row in oracle['arrays'].items():
        path=fixture/name;a=np.load(path,allow_pickle=False)
        assert sha(path)==row['sha256'] and list(a.shape)==row['shape'] and str(a.dtype)==row['dtype'] and np.isfinite(a).all()
    os.environ.update(BREEZE_CODEC=str(paths['codec']),BREEZE_FIXTURE_DIR=str(fixture))
    stages=OUT/'stages';guided=OUT/'guided-dump';stages.mkdir();guided.mkdir()
    run([bundle/'crispasr-diff','bt2-tts',paths['breeze'],fixture,stages],'gpu-stage-dump')
    sys.path.insert(0,str(ROOT))
    from tools.reference_backends.breeze_tts_2 import compare_stage
    required={Path(name).stem for name in oracle['arrays'] if '/' not in name and
              (name.startswith(('te_','backbone_','prompt_','dd_logits_')) or name=='dd_codes_frame0_stepwise.npy')}
    results={}
    for name in sorted(required):
        native=np.load(stages/(name+'.npy'),allow_pickle=False)
        original=np.load(fixture/(name+'.npy'),allow_pickle=False)
        assert native.size and np.isfinite(native).all(),name
        results[name]=compare_stage(name,original,native)
    receipt['stage_results']=results
    receipt['stages_pass']=all(row['pass'] for row in results.values())
    receipt['stage_arrays']={str(p.relative_to(stages)):dict(sha256=sha(p),shape=list(np.load(p,allow_pickle=False).shape))
                             for p in stages.rglob('*.npy')}
    if not receipt['stages_pass']:receipt['failures'].append('Original stage gate failed')
    save()
    run([bundle/'crispasr-diff','bt2-tts-guided',paths['breeze'],fixture,guided],'gpu-guided-dump')
    command=[sys.executable,ROOT/'tools/reference_backends/breeze_tts_2.py','--guided','--no-download','--cpp-dump',guided]
    with (OUT/'guided-comparison.log').open('w') as log:
        result=subprocess.run(list(map(str,command)),cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,timeout=600)
    receipt['guided_comparison_returncode']=result.returncode
    receipt['guided_exact_pass']=result.returncode==0
    receipt['guided_arrays']={str(p.relative_to(guided)):dict(sha256=sha(p),shape=list(np.load(p,allow_pickle=False).shape))
                              for p in guided.rglob('*.npy')}
    if not receipt['guided_exact_pass']:receipt['failures'].append('Strict original guided generation gate failed; no waiver')
    save()


if __name__=='__main__':
    main()
