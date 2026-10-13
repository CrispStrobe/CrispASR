#!/usr/bin/env python3
"""Pinned native Breeze CLI/C ABI/server speech diagnostics on GH CPU.

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

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(os.environ['HEAVY_OUT'])
SCRATCH = Path(os.environ['HEAVY_SCRATCH']) / 'breeze-surfaces'
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


def run(command, label):
    with (OUT / (label + '.log')).open('w') as log:
        result = subprocess.run(list(map(str, command)), cwd=ROOT, stdout=log,
                                stderr=subprocess.STDOUT, timeout=7200)
    print(label, 'rc', result.returncode, flush=True)
    assert result.returncode == 0, label


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
    sys.path.insert(0, str(ROOT / 'python'))
    from crispasr import Session
    receipt = dict(source=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                   device='CPU; GPU backends compiled OFF', complete=False, speech_pass=False,
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
        run(['sudo','bash','tools/ci-apt.sh','update'],'apt-update')
        run(['sudo','bash','tools/ci-apt.sh','install','-y','cmake','ninja-build','g++','libopenblas-dev'],'apt-install')
        build = SCRATCH / 'build'
        run(['cmake','-S',ROOT,'-B',build,'-G','Ninja','-DCMAKE_BUILD_TYPE=Release',
             '-DBUILD_SHARED_LIBS=ON','-DGGML_NATIVE=OFF','-DGGML_OPENBLAS=ON',
             '-DGGML_CUDA=OFF','-DGGML_VULKAN=OFF','-DGGML_METAL=OFF',
             '-DCRISPASR_BUILD_TESTS=OFF','-DCRISPASR_BUILD_SERVER=ON'],'configure')
        run(['cmake','--build',build,'--target','crispasr-cli','crispasr-lib','-j4'],'build')
        binary, library = build / 'bin/crispasr', build / 'src/libcrispasr.so'
        receipt.update(cli_sha256=sha(binary), library_sha256=sha(library))
        models = SCRATCH / 'models'
        models.mkdir()
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
                     '--no-gpu','-t','4','--seed',str(SEED),'--max-new-tokens',str(LIMIT),
                     '--accept-marking-responsibility','--no-spoken-disclaimer','--no-watermark',
                     '--tts',TEXT,'--tts-output',path]
            if ref:command += ['--voice',voice,'--ref-text',REFERENCE_TEXT,'--i-have-rights']
            if ins:command += ['--tts-instruct',INSTRUCTION,'--tts-cfg-scale','4']
            run(command,'cli-'+label)
            retain('cli-'+label,read_wav(path))
        # Reuse one actual server backend across plain/reference/instruction modes.
        log=(OUT/'server.log').open('w')
        proc=subprocess.Popen([str(x) for x in [binary,'--server','--port','11449','--backend','bt2-tts',
            '-m',paths['breeze'],'--codec-model',paths['codec'],'--no-gpu','-t','4',
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
        import resemblyzer,importlib.metadata
        from resemblyzer import VoiceEncoder,preprocess_wav
        receipt['speaker_encoder']=dict(package_version=importlib.metadata.version('resemblyzer'),
            model_sha256=sha(Path(resemblyzer.__file__).parent/'pretrained.pt'))
        encoder=VoiceEncoder(device='cpu')
        embed=lambda name:encoder.embed_utterance(preprocess_wav(np.load(OUT/(name+'.npy')),source_sr=RATE))
        r=embed('reference');baseline=embed('abi-plain');clone=embed('abi-clone')
        similarities=dict(clone_reference=float(np.dot(clone,r)),baseline_reference=float(np.dot(baseline,r)))
        receipt['speaker']=similarities
        if similarities['clone_reference']<=similarities['baseline_reference']:
            receipt['failures'].append('Closed-loop clone speaker similarity did not improve')
        receipt['complete']=True;receipt['speech_pass']=not receipt['failures'];save()
        (OUT/'summary.md').write_text('Native Breeze surface diagnostic completed. See validation.json for '
            'actual contracts, every ASR transcript and closed-loop speaker result. '
            'This does not waive the independently failed exact-code gate.\n')
        assert receipt['speech_pass'],receipt['failures']
    except BaseException as error:
        receipt['error']=f'{type(error).__name__}: {error}';save();raise


if __name__=='__main__':
    main()
