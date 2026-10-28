#!/usr/bin/env python3
"""Real GPU voice conversion against pinned original modules.

Reuse the independently audited CUDA runtime; compile only changed model/driver
translation units against its exact ggml headers. No ggml/CUDA/full SDK build,
no CPU-only Kaggle job, no hardware fishing or cross-account retry.
"""
import ctypes as ct
from contextlib import nullcontext
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import threading
import time
import traceback
import urllib.request

CONDITIONING = os.environ.get('CRISPASR_OV2_CONDITIONING') == '1'
PUBLISH = os.environ.get('CRISPASR_OV2_PUBLISH') == '1'
REPLAY = os.environ.get('CRISPASR_OV2_REPLAY') == '1'
VERSION = ('2026-10-28.openvoice-fixed-replay.3' if REPLAY else
           '2026-10-28.openvoice-conditioning.6' if CONDITIONING else '2026-10-28.openvoice-bake.8')
SOURCE = '6be9fda3629d9314da97101b8574fb25db06150f'
MODEL_DELTA_SOURCE = "1b874a104c70bc77e1a64dc1d5f4e924f0a3cfeb"
MODEL_DELTA_SHA256 = "f658c40176fe1c67584cd9ab80ca9d954433cfe7973f6ba84b43a448d59b8c13"
REPLAY_PINS = {'model': '483fd233035bb2c7cde69519d4d460742b6a70c96c5293e878fe25e72d944255',
               'reference': '259993b526944e8e36ccca436e005cbc3935a46636b42c618cfd5c3d19429c8f'}
UPSTREAM = '74a1d147b17a8c3092dd5430504bd83ef6c7eb23'
BASE_SOURCE = '191815511b85a526aace111b1bda2e28ecb7faad'
BASE_REV = '425c555beb63de4665cba604c47ea38232f5d080'
BASE_SHA = '63474b5f034ebac850b7b87ef76eecec12d009092d4c72370aeeb6be341365c5'
OUT = Path('/kaggle/working')
WORK = Path('/kaggle/temp/openvoice-bake')
ROOT = WORK / 'sdk'
BUNDLE = WORK / 'bundle'
RECEIPT = {'schema': 1, 'version': VERSION, 'source': SOURCE, 'upstream': UPSTREAM,
           'complete': False, 'model_acceptance': False, 'gpu_executed': False,
           'publication_requested': PUBLISH, 'publication_complete': False,
           'native_base_source': SOURCE, 'native_model_delta_source': MODEL_DELTA_SOURCE,
           'native_model_delta_sha256': MODEL_DELTA_SHA256,
           'cases': [], 'limitations': ['native reference encoder and flows are CPU; vocoder uses CUDA',
                                      'native-rate replay isolates resampling; raw reference without VAD',
                                      'tau0.3 uses original shared noise; free-running RNG parity is separate' if CONDITIONING else 'deterministic tau=0; upstream tau=0.3 quality separate',
                                      'closed-loop synthetic speakers, not human speaker verification']}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def save():
    (OUT / 'validation.json').write_text(json.dumps(RECEIPT, indent=2) + '\n')


def host_environment():
    env=dict(os.environ)
    env['LD_LIBRARY_PATH']=env.get('CRISPASR_OV2_HOST_LD_LIBRARY_PATH',env.get('LD_LIBRARY_PATH',''))
    return env


def resource_snapshot():
    memory = dict(line.split(":", 1) for line in Path('/proc/meminfo').read_text().splitlines())
    gpu_free = subprocess.check_output(
        ['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits'],
        text=True, timeout=30, env=host_environment())
    return {'available_bytes': int(memory['MemAvailable'].split()[0]) * 1024,
            'load': list(os.getloadavg()), 'cpus': os.cpu_count() or 1,
            'scratch_free_bytes': shutil.disk_usage(WORK).free,
            'gpu_free_mib': [int(line.strip()) for line in gpu_free.splitlines() if line.strip()]}


def guard_resources(label):
    """Retain host headroom before every compile, reference or native phase."""
    state = resource_snapshot()
    RECEIPT.setdefault('resource_checks', []).append({'phase': label, **state})
    save()
    if (state['available_bytes'] < 3 * 1024**3 or state['scratch_free_bytes'] < 2 * 1024**3
            or state['load'][0] > 2 * state['cpus'] or not state['gpu_free_mib']
            or min(state['gpu_free_mib']) < 1024):
        raise RuntimeError('Insufficient worker headroom before ' + label)


def checkpoint_model(hub, converted, model_rev, license_file):
    """Retain producer bytes even when validation has no publishing rights."""
    record = {'local_file': converted.name, 'bytes': converted.stat().st_size,
              'sha256': sha(converted)}
    if not record['bytes']:
        raise ValueError('Empty converted model checkpoint')
    RECEIPT['model_checkpoint'] = record
    save()
    if not PUBLISH:
        return
    repo = 'cstr/openvoice2-tcc-GGUF'
    hub.create_repo(repo, repo_type='model', private=False, exist_ok=True)
    if not hub.file_exists(repo, 'README.md'):
        pending = ('---\nlicense: mit\npipeline_tag: audio-to-audio\n---\n\n'
                   '# OpenVoice V2 GGUF experimental artifacts\n\n'
                   'Files under staging/ are producer checkpoints, not accepted releases. '
                   'Actual parity, GPU and speech-quality validation is pending.\n\n'
                   'Original MIT model: https://huggingface.co/myshell-ai/OpenVoiceV2/tree/' + model_rev + '\n')
        hub.upload_file(path_or_fileobj=pending.encode(), repo_id=repo, path_in_repo='README.md',
                        commit_message='Mark producer staging artifacts unvalidated')
    path = 'staging/' + VERSION + '/' + converted.name
    staged = hub.upload_file(path_or_fileobj=str(converted), repo_id=repo, path_in_repo=path,
                             commit_message='Checkpoint unvalidated conversion before hardware validation')
    hub.upload_file(path_or_fileobj=str(license_file), repo_id=repo, path_in_repo='LICENSE',
                    commit_message='Retain original MIT notice')
    record.update(repo=repo, revision=staged.oid, path=path)
    save()


def checkpoint_reference(hub, ref, case, prefix, notice, license_file):
    record = {'case': case, 'local_file': ref.name, 'bytes': ref.stat().st_size, 'sha256': sha(ref)}
    if not record['bytes']:
        raise ValueError('Empty original reference checkpoint')
    (OUT / 'reference-ATTRIBUTION.txt').write_text(notice)
    shutil.copy2(license_file, OUT / 'reference-OpenVoice-LICENSE.txt')
    RECEIPT.setdefault('reference_checkpoints', []).append(record)
    save()
    if not PUBLISH:
        return record
    try:
        published = hub.upload_file(path_or_fileobj=str(ref), repo_id='cstr/crispasr-regression-fixtures',
            repo_type='dataset', path_in_repo=f'{prefix}/reference-{case}.gguf',
            commit_message=f'OpenVoice original-module reference case {case}; quality gate separate')
        record['published_revision'] = published.oid
        hub.upload_file(path_or_fileobj=notice.encode(), repo_id='cstr/crispasr-regression-fixtures',
            repo_type='dataset', path_in_repo=prefix+'/ATTRIBUTION.txt',
            commit_message='Document OpenVoice reference provenance')
        hub.upload_file(path_or_fileobj=str(license_file), repo_id='cstr/crispasr-regression-fixtures',
            repo_type='dataset', path_in_repo=prefix+'/OpenVoice-LICENSE.txt',
            commit_message='Retain OpenVoice MIT notice')
    except Exception as exc:
        record['publication_error'] = type(exc).__name__
        save()
        raise RuntimeError('Requested reference checkpoint publication failed') from exc
    save()
    return record


def load_replay_inputs():
    candidates = sorted(p for p in Path('/kaggle/input').rglob('inputs.json')
                        if 'crispasr-openvoice-replay-20261028' in p.parts)
    if not candidates:
        raise RuntimeError('Frozen model/reference dataset is not mounted')
    manifest = candidates[0]
    metadata = json.loads(manifest.read_text())
    inputs = {}
    for role, filename in [('model', metadata['model']['name']),
                           ('reference', metadata['reference']['local_file'])]:
        if Path(filename).name != filename:
            raise ValueError('Replay input filename must be a basename')
        path = manifest.parent / filename
        if sha(path) != REPLAY_PINS[role]:
            raise ValueError('Frozen replay checksum mismatch: ' + role)
        inputs[role] = path
    shutil.copy2(manifest, OUT / 'replay-inputs.json')
    for name in ['reference-ATTRIBUTION.txt', 'reference-OpenVoice-LICENSE.txt']:
        shutil.copy2(manifest.parent / name, OUT / name)
    RECEIPT['replay_input_sha256'] = dict(REPLAY_PINS)
    save()
    return inputs


def replay(inputs):
    """Same rejected original/model bytes, CPU and genuine CUDA; no new teacher."""
    import re
    results = []
    for mode in ['cuda', 'cpu']:
        env = dict(os.environ)
        env.pop('CRISPASR_DIFF_USE_GPU', None)
        if mode == 'cuda':
            env['CRISPASR_DIFF_USE_GPU'] = '1'
        label = 'replay-' + mode
        returncode = 0
        try:
            run([WORK / 'openvoice-bake', inputs['model'], inputs['reference']], label, env=env)
        except subprocess.CalledProcessError as exc:
            returncode = exc.returncode
        log = (OUT / (label + '.log')).read_text()
        cuda_work = bool(re.search(r'^## SPLIT #\d+: CUDA', log, re.M)
                         and re.search(r'HiFi-GAN output [1-9]\d* samples', log))
        device_ok = cuda_work if mode == 'cuda' else 'vocoder backend CPU' in log
        row = {'mode': mode, 'returncode': returncode, 'passes': log.count('[PASS]'),
               'failures': log.count('[FAIL]'), 'cuda_work_observed': cuda_work}
        row['accepted'] = bool(returncode == 0 and row['passes'] == 27 and row['failures'] == 0 and device_ok)
        results.append(row)
        RECEIPT.update(replay_results=results, gpu_executed=any(r['cuda_work_observed'] for r in results))
        save()
    RECEIPT.update(complete=True, replay_acceptance=all(r['accepted'] for r in results),
                   model_acceptance=False,
                   replay_scope='Frozen tau=0 structural model replay on the original audited runtime; not speech, conditioning, Session/CLI or whole-current-ggml acceptance')
    save()
    if not RECEIPT['replay_acceptance']:
        raise RuntimeError('Frozen CPU/CUDA replay failed unchanged original-reference gates')


def run(command, name, **kwargs):
    guard_resources(name)
    if str(command[0])==sys.executable:
        kwargs.setdefault('env',host_environment())
        kwargs.setdefault('cwd',WORK)
    print(name, flush=True)
    harness=sys.modules.get('kaggle_harness')
    if harness:
        harness.step(name)
    heartbeat = harness.build_heartbeat(name) if harness and name.startswith('compile-') else nullcontext()
    with heartbeat, (OUT / (name + '.log')).open('w') as log:
        subprocess.run(list(map(str, command)), stdout=log, stderr=subprocess.STDOUT,
                       check=True, timeout=kwargs.pop('timeout', 1200), **kwargs)


def download(url, path, expected=None):
    with urllib.request.urlopen(url, timeout=120) as stream, Path(path).open('wb') as output:
        shutil.copyfileobj(stream, output, 4*1024*1024)
    if expected and sha(path) != expected:
        raise ValueError('Checksum mismatch: ' + Path(path).name)


def bootstrap():
    WORK.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    RECEIPT['entry_sha256']=sha(Path(__file__))
    os.environ.update(TMPDIR=str(WORK), HF_HOME=str(WORK/'hf'), OMP_NUM_THREADS='2',
                      OPENBLAS_NUM_THREADS='1', CRISPASR_OV2_GPU='1', GGML_SCHED_DEBUG='1')
    RECEIPT['hardware'] = subprocess.check_output(
        ['nvidia-smi','--query-gpu=name,compute_cap,memory.total,driver_version','--format=csv,noheader'],text=True).strip()
    caps = [float(row.split(',')[1]) for row in RECEIPT['hardware'].splitlines()]
    if not caps or any(cap not in (6.0,7.5) for cap in caps):
        raise RuntimeError('Frozen CUDA runtime supports sm60/sm75 only')
    guard_resources('bootstrap')
    for command in (['git','init',ROOT],['git','-C',ROOT,'remote','add','origin','https://github.com/CrispStrobe/CrispASR.git'],
                    ['git','-C',ROOT,'fetch','--depth=1','origin',SOURCE],['git','-C',ROOT,'checkout','FETCH_HEAD'],
                    ['git','-C',ROOT,'submodule','update','--init','--depth=1','ggml']):
        subprocess.run(list(map(str,command)),check=True)
    sys.path.insert(0,str(ROOT/'tools/kaggle'))
    import kaggle_harness as kh
    kh.init_progress()
    kh.provenance(VERSION,ROOT)
    os.environ['HF_TOKEN'] = kh.resolve_hf_token(require=True)
    runtime=WORK/'runtime.tar.gz'
    download(f'https://huggingface.co/datasets/cstr/crispasr-regression-fixtures/resolve/{BASE_REV}/alignment-long-cuda-20261019/v2/runtime.tar.gz',runtime,BASE_SHA)
    with tarfile.open(runtime) as archive:
        archive.extractall(WORK,filter='data')
    provenance=json.loads((BUNDLE/'provenance.json').read_text())
    assert provenance['source']==BASE_SOURCE and provenance['build_complete']
    for name,row in provenance['files'].items():
        path=BUNDLE/name
        assert path.stat().st_size==row['bytes'] and sha(path)==row['sha256'],name
    ggml=subprocess.check_output(['git','-C',ROOT/'ggml','rev-parse','HEAD'],text=True).strip()
    assert ggml=='0a862076567afd8468822ccc512c97ac6a69b340'
    RECEIPT.update(base_runtime_source=BASE_SOURCE,base_runtime_sha256=BASE_SHA,ggml=ggml)
    download('https://raw.githubusercontent.com/CrispStrobe/CrispASR/' + MODEL_DELTA_SOURCE +
             '/src/openvoice2.cpp', ROOT/'src/openvoice2.cpp', MODEL_DELTA_SHA256)
    if os.environ.get('CRISPASR_OV2_BUNDLE_READY')!='1':
        host_path=os.environ.get('LD_LIBRARY_PATH','')
        env=dict(os.environ,CRISPASR_OV2_BUNDLE_READY='1',CRISPASR_OV2_HOST_LD_LIBRARY_PATH=host_path,
                 LD_LIBRARY_PATH=str(BUNDLE)+os.pathsep+host_path)
        os.execve(sys.executable,[sys.executable,str(Path(__file__).resolve())],env)
    return kh


def main():
    if REPLAY and PUBLISH:
        raise ValueError('Frozen replay cannot request publication')
    # Re-exec only after immutable inputs are verified, without repeat cloning.
    if os.environ.get('CRISPASR_OV2_BUNDLE_READY')=='1':
        sys.path.insert(0,str(ROOT/'tools/kaggle'))
        import kaggle_harness as kh
        kh.init_progress(); kh.provenance(VERSION,ROOT)
        RECEIPT['entry_sha256']=sha(Path(__file__))
        RECEIPT.update(base_runtime_source=BASE_SOURCE,base_runtime_sha256=BASE_SHA,
                       ggml=subprocess.check_output(['git','-C',ROOT/'ggml','rev-parse','HEAD'],text=True).strip(),
                       hardware=subprocess.check_output(['nvidia-smi','--query-gpu=name,compute_cap,memory.total,driver_version','--format=csv,noheader'],text=True).strip())
    else:
        kh=bootstrap()
    if sha(ROOT/'src/openvoice2.cpp') != MODEL_DELTA_SHA256:
        raise ValueError('Native model overlay checksum mismatch')
    kh.step('native-model-overlay', source=MODEL_DELTA_SOURCE, sha256=MODEL_DELTA_SHA256)
    original_stderr=os.dup(2)
    native_log=(OUT/'native.log').open('w');os.dup2(native_log.fileno(),2)
    try:
        replay_inputs = None
        if REPLAY:
            guard_resources('frozen-replay-inputs')
            replay_inputs = load_replay_inputs()
        else:
            run([sys.executable,'-m','pip','install','-q','gguf','librosa>=0.10','soundfile','huggingface_hub','resemblyzer','webrtcvad-wheels','--no-deps'],'small-dependencies')
            run([sys.executable,'-m','pip','install','-q','faster-whisper'],'asr-dependencies')
        # Use the prebuilt CUDA ggml. These are ordinary model C++ files only;
        # link our own definitions locally to avoid old SDK symbol interposition.
        # The audited base uses dynamic backend modules. Match that ABI and
        # let registry discovery find CPU/CUDA next to the current directory.
        os.chdir(BUNDLE)
        common=['c++','-std=c++17','-O2','-fPIC','-fopenmp','-DGGML_BACKEND_DL','-I'+str(ROOT/'src'),'-I'+str(ROOT/'ggml/include'),
                '-I'+str(ROOT/'examples/cli'),'-L'+str(BUNDLE),'-Wl,-rpath,'+str(BUNDLE)]
        run(common+[ROOT/'src/openvoice2.cpp',ROOT/'src/melotts.cpp','-shared','-Wl,-Bsymbolic-functions',
                    '-lcrispasr','-lggml','-lggml-base','-ldl','-pthread','-o',WORK/'libopenvoice-probe.so'],'compile-model-delta')
        run(common+[ROOT/'tools/ci-heavy/openvoice-bake/main.cpp',ROOT/'examples/cli/crispasr_diff.cpp',
                    '-L'+str(WORK),'-Wl,-rpath,'+str(WORK),'-lopenvoice-probe','-lcrispasr',
                    '-lggml','-lggml-base','-ldl','-pthread','-o',WORK/'openvoice-bake'],'compile-diff-driver')
        RECEIPT['native_delta']={name:sha(WORK/name) for name in ['libopenvoice-probe.so','openvoice-bake']}
        if REPLAY:
            replay(replay_inputs)
            return
        import numpy as np
        import soundfile as sf
        cap=float(RECEIPT['hardware'].splitlines()[0].split(',')[1])
        RECEIPT['torch_reference_device']='cuda' if cap>=7 else 'cpu'
        from huggingface_hub import HfApi,hf_hub_download,snapshot_download
        hub=HfApi()
        model_rev=hub.model_info('myshell-ai/OpenVoiceV2').sha
        model=Path(snapshot_download('myshell-ai/OpenVoiceV2',revision=model_rev,
                    allow_patterns=['converter/*','base_speakers/ses/*.pth','README.md']))
        RECEIPT['original_model']={'repo':'myshell-ai/OpenVoiceV2','revision':model_rev,
            'files':{str(p.relative_to(model)):sha(p) for p in model.rglob('*') if p.is_file()}}
        upstream_tar=WORK/'upstream.tar.gz'
        download(f'https://codeload.github.com/myshell-ai/OpenVoice/tar.gz/{UPSTREAM}',upstream_tar)
        with tarfile.open(upstream_tar) as archive:
            archive.extractall(WORK,filter='data')
        upstream=WORK/('OpenVoice-'+UPSTREAM)
        shutil.copy2(upstream/'LICENSE',OUT/'OpenVoice-LICENSE.txt')
        run([sys.executable,ROOT/'models/convert-openvoice2-to-gguf.py','--ckpt',model/'converter/checkpoint.pth',
             '--config',model/'converter/config.json','--output',WORK/'openvoice-f32sens.gguf'],'convert-f32-sensitive')
        converted=OUT/'openvoice2-tcc-f32sens.gguf'
        shutil.copy2(WORK/'openvoice-f32sens.gguf',converted)
        RECEIPT['converted_model']={'bytes':converted.stat().st_size,'sha256':sha(converted),'name':converted.name}
        save()
        # Producer checkpoints precede crash-prone native execution. Staging
        # artifacts are explicitly unvalidated and are not registry defaults.
        checkpoint_model(hub, converted, model_rev, upstream/'LICENSE')
        melo_rev=hub.model_info('cstr/melotts-en-v2-GGUF').sha
        melo=hf_hub_download('cstr/melotts-en-v2-GGUF','melotts-en-v2-f16.gguf',revision=melo_rev)
        bert=hf_hub_download('cstr/melotts-en-v2-GGUF','bert-base-uncased-q4k.gguf',revision=melo_rev)
        RECEIPT['melo']={'repo':'cstr/melotts-en-v2-GGUF','revision':melo_rev,'model_sha256':sha(melo),'bert_sha256':sha(bert)}
        class MeloParams(ct.Structure):
            _fields_=[('n_threads',ct.c_int),('verbosity',ct.c_int),('use_gpu',ct.c_bool),
                      ('noise_scale',ct.c_float),('length_scale',ct.c_float),('noise_w',ct.c_float),
                      ('sdp_ratio',ct.c_float),('speaker_id',ct.c_int),('seed',ct.c_uint32)]
        lib=ct.CDLL(str(WORK/'libopenvoice-probe.so'))
        lib.melotts_default_params.restype=MeloParams
        lib.melotts_init_from_file.argtypes=[ct.c_char_p,MeloParams];lib.melotts_init_from_file.restype=ct.c_void_p
        lib.melotts_load_bert.argtypes=[ct.c_void_p,ct.c_char_p];lib.melotts_load_bert.restype=ct.c_bool
        lib.melotts_set_speaker_id.argtypes=[ct.c_void_p,ct.c_int]
        lib.melotts_source_speaker.argtypes=[ct.c_void_p];lib.melotts_source_speaker.restype=ct.c_char_p
        lib.melotts_synthesize.argtypes=[ct.c_void_p,ct.c_char_p,ct.POINTER(ct.POINTER(ct.c_float)),ct.POINTER(ct.c_int)]
        lib.melotts_synthesize.restype=ct.c_int
        lib.melotts_pcm_free.argtypes=[ct.POINTER(ct.c_float)]
        lib.melotts_free.argtypes=[ct.c_void_p]
        params=lib.melotts_default_params();params.n_threads=2;params.use_gpu=False;params.seed=42
        guard_resources('load-melo-model')
        ctx=lib.melotts_init_from_file(os.fsencode(melo),params)
        if not ctx or not lib.melotts_load_bert(ctx,os.fsencode(bert)):raise RuntimeError('Melo/BERT load failed')
        def synth(text,speaker):
            guard_resources('melo-synthesis')
            lib.melotts_set_speaker_id(ctx,speaker)
            speaker_key=lib.melotts_source_speaker(ctx).decode()
            ptr=ct.POINTER(ct.c_float)();rate=ct.c_int()
            count=lib.melotts_synthesize(ctx,text.encode(),ct.byref(ptr),ct.byref(rate))
            if count<=0 or not ptr:raise RuntimeError('Empty Melo output')
            try:return np.ctypeslib.as_array(ptr,shape=(count,)).copy(),rate.value,speaker_key
            finally:lib.melotts_pcm_free(ptr)
        texts=['The quick brown fox jumps over the lazy dog.',
               'We enjoy reading books and listening to music in the evening.']
        reference,ref_sr,ref_key=synth('This is a reference voice for a simple speech test.',1)
        sf.write(OUT/'reference.wav',reference,ref_sr,subtype='FLOAT')
        RECEIPT['reference']={'speaker':ref_key,'rate':ref_sr,'sha256':sha(OUT/'reference.wav')}
        os.environ.update(CRISPASR_OV2_UPSTREAM=str(upstream),CRISPASR_OV2_REFERENCE_WAV=str(OUT/'reference.wav'),
                          CRISPASR_OV2_DEVICE=RECEIPT['torch_reference_device'],
                          CRISPASR_OV2_TAU='0')
        class OpenParams(ct.Structure):
            _fields_=[('n_threads',ct.c_int),('verbosity',ct.c_int),('use_gpu',ct.c_bool),('tau',ct.c_float)]
        lib.openvoice2_context_default_params.restype=OpenParams
        lib.openvoice2_init_from_file.argtypes=[ct.c_char_p,OpenParams];lib.openvoice2_init_from_file.restype=ct.c_void_p
        lib.openvoice2_set_source_speaker.argtypes=[ct.c_void_p,ct.c_char_p];lib.openvoice2_set_source_speaker.restype=ct.c_bool
        floatp=ct.POINTER(ct.c_float)
        lib.openvoice2_convert.argtypes=[ct.c_void_p,floatp,ct.c_int,ct.c_int,floatp,ct.c_int,ct.c_int,ct.POINTER(floatp),ct.POINTER(ct.c_int)]
        lib.openvoice2_convert.restype=ct.c_bool
        lib.openvoice2_extract_speaker_embedding.argtypes=[ct.c_void_p,floatp,ct.c_int,ct.c_int,floatp]
        lib.openvoice2_extract_speaker_embedding.restype=ct.c_bool
        lib.openvoice2_free.argtypes=[ct.c_void_p]
        libc=ct.CDLL(None);libc.free.argtypes=[ct.c_void_p]
        op=lib.openvoice2_context_default_params();op.n_threads=2;op.use_gpu=True;op.tau=0
        guard_resources('load-openvoice-model')
        ov=lib.openvoice2_init_from_file(os.fsencode(WORK/'openvoice-f32sens.gguf'),op)
        if not ov:raise RuntimeError('Native conversion context load failed')
        cases=[(text,0.0) for text in texts]+([(texts[0],.3)] if CONDITIONING else [])
        for i,(text,tau) in enumerate(cases):
            baseline,sr,key=synth(text,0)
            sf.write(OUT/f'baseline-{i}.wav',baseline,sr,subtype='FLOAT')
            import librosa
            source16=librosa.resample(baseline,orig_sr=sr,target_sr=16000)
            os.environ['CRISPASR_OV2_SOURCE_SPEAKER']=key
            os.environ['CRISPASR_OV2_TAU']=str(tau)
            if tau:
                lib.openvoice2_free(ov);op.tau=tau
                guard_resources('load-noisy-openvoice-model')
                ov=lib.openvoice2_init_from_file(os.fsencode(WORK/'openvoice-f32sens.gguf'),op)
                if not ov:raise RuntimeError('Noisy context load failed')
            # Original Torch runs in a separate process with the host CUDA
            # stack. The native process keeps the audited CUDA12.4 libraries.
            np.save(WORK/'source16.npy',source16)
            reference_script=WORK/'reference.py'
            reference_script.write_text("""import sys,os,json
from pathlib import Path
import numpy as np
import soundfile as sf
sys.path.insert(0,sys.argv[1])
from dump_reference import write_gguf_archive
from reference_backends.openvoice2 import dump,DEFAULT_STAGES
model=Path(sys.argv[2]);source=np.load(sys.argv[3],allow_pickle=False)
captures=dump(model,source,set(DEFAULT_STAGES))
meta={name:captures.pop(name) for name in list(captures) if isinstance(captures[name],str)}
import torch
meta.update(backend='openvoice2',source_commit=sys.argv[6],upstream_commit=sys.argv[7],model_revision=sys.argv[8],torch_version=torch.__version__)
Path(sys.argv[5]+'.versions.json').write_text(json.dumps({'torch':torch.__version__,'numpy':np.__version__}))
if os.environ.get('CRISPASR_OV2_CONDITIONING')=='1' and float(meta['ov2_tau'])==0:
 import torch
 from openvoice.models import SynthesizerTrn
 from openvoice.mel_processing import spectrogram_torch
 cfg=json.loads((model/'converter/config.json').read_text());data=cfg['data']
 m=SynthesizerTrn(len(cfg.get('symbols',[])),data['filter_length']//2+1,n_speakers=data['n_speakers'],**cfg['model']).eval()
 m.load_state_dict(torch.load(model/'converter/checkpoint.pth',map_location='cpu',weights_only=False)['model'],strict=True)
 device=os.environ.get('CRISPASR_OV2_DEVICE','cpu');m=m.to(device)
 pcm=captures['ref_pcm'].flatten();n=len(pcm);cut=n//3
 def spec(x):
  y=torch.from_numpy(np.ascontiguousarray(x)).unsqueeze(0).to(device)
  return spectrogram_torch(y,data['filter_length'],data['sampling_rate'],data['hop_length'],data['win_length'],center=False)
 with torch.inference_mode():
  embeddings=[m.ref_enc(spec(pcm[a:b]).transpose(1,2)) for a,b in [(0,cut),(cut,n)]]
  mean=torch.stack(embeddings).mean(0).unsqueeze(-1)
  source_spec=torch.from_numpy(captures['src_spec'].T.copy()).unsqueeze(0).to(device)
  source_se=torch.from_numpy(captures['src_se'].reshape(1,-1,1).copy()).to(device)
  lengths=torch.tensor([source_spec.shape[-1]],device=device)
  wave=m.voice_conversion(source_spec,lengths,source_se,mean,tau=0)[0][0,0].cpu().numpy().copy()
  peak=np.max(np.abs(wave))
  if peak>.01:wave*=np.float32(.95)/peak
 captures['segment_mean']=mean.cpu().numpy().copy()
 captures['segment_spans']=np.array([0,cut,cut,n],dtype=np.int32)
 captures['segment_output_pcm']=wave
write_gguf_archive(captures,meta,Path(sys.argv[4]))
np.savez(sys.argv[5],**captures)
""")
            ref=OUT/f'reference-{i}.gguf'
            run([sys.executable,reference_script,ROOT/'tools',model,WORK/'source16.npy',ref,
                 WORK/'captures.npz',SOURCE,UPSTREAM,model_rev],f'original-reference-{i}')
            RECEIPT['producer_versions']=json.loads((WORK/'captures.npz.versions.json').read_text())
            with np.load(WORK/'captures.npz',allow_pickle=False) as archive:
                captures={name:archive[name].copy() for name in archive.files}
            sf.write(OUT/f'original-{i}.wav',captures['output_pcm'].flatten(),22050,subtype='FLOAT')
            # Checkpoint immediately after production, before native execution.
            fixture_prefix='openvoice2/bake-20261028/'+VERSION
            notice=("OpenVoice2 reference from original MIT modules at "+UPSTREAM+".\n"
                    "Inputs are synthetic MeloTTS audio from cstr/melotts-en-v2-GGUF revision "+melo_rev+".\n"
                    "No external speaker recording. Model/source hashes and recipe are inside the archive.\n"
                    "Structural parity and speech quality are separate gates; consult the validation receipt.\n")
            reference_checkpoint=checkpoint_reference(hub,ref,i,fixture_prefix,notice,upstream/'LICENSE')
            kh.step('native-gpu-parity',case=i)
            env=dict(os.environ,CRISPASR_DIFF_USE_GPU='1')
            run([WORK/'openvoice-bake',WORK/'openvoice-f32sens.gguf',ref],f'diff-gpu-{i}',env=env)
            log=(OUT/f'diff-gpu-{i}.log').read_text()
            import re
            expected_stages=28 if tau else 27
            if log.count('[PASS]')!=expected_stages or '[FAIL]' in log or 'vocoder backend CUDA' not in log or not re.search(r'## SPLIT #\d+: CUDA',log):
                raise RuntimeError('GPU stages missing or not actually CUDA')
            # Same public model C ABI without instrumentation, including
            # reference-cache reuse. Original tensors feed the native input.
            source_pcm=np.ascontiguousarray(captures['src_pcm'].flatten(),dtype=np.float32)
            target_pcm=np.ascontiguousarray(captures['ref_pcm'].flatten(),dtype=np.float32)
            if not lib.openvoice2_set_source_speaker(ov,key.encode()):raise RuntimeError('Source identity rejected')
            if tau:
                lib.openvoice2_set_posterior_noise.argtypes=[ct.c_void_p,floatp,ct.c_int]
                lib.openvoice2_set_posterior_noise.restype=ct.c_bool
                noise=np.ascontiguousarray(captures['enc_q_noise'].flatten(),dtype=np.float32)
                if not lib.openvoice2_set_posterior_noise(ov,noise.ctypes.data_as(floatp),len(noise)):
                    raise RuntimeError('Original posterior noise was rejected')
            native=[]
            for repeat in range(2):
                guard_resources('native-conversion')
                output=floatp();count=ct.c_int()
                ok=lib.openvoice2_convert(ov,source_pcm.ctypes.data_as(floatp),len(source_pcm),22050,
                    target_pcm.ctypes.data_as(floatp),len(target_pcm),22050,ct.byref(output),ct.byref(count))
                if not ok or not output or count.value<=0:raise RuntimeError('Public C conversion failed')
                try:native.append(np.ctypeslib.as_array(output,shape=(count.value,)).copy())
                finally:libc.free(output)
            if not np.array_equal(native[0],native[1]):raise RuntimeError('Cache/reuse changed deterministic output')
            truth=captures['output_pcm'].flatten()
            cosine=float(np.dot(native[0].astype(np.float64),truth)/(np.linalg.norm(native[0].astype(np.float64))*np.linalg.norm(truth)))
            if cosine<.99:raise RuntimeError('Uninstrumented waveform parity failed')
            embedding=np.empty(256,dtype=np.float32)
            guard_resources('native-speaker-embedding')
            if not lib.openvoice2_extract_speaker_embedding(ov,target_pcm.ctypes.data_as(floatp),len(target_pcm),22050,
                                                         embedding.ctypes.data_as(floatp)):raise RuntimeError('Embedding C ABI failed')
            original_embedding=captures['target_se'].flatten()
            emb_cos=float(np.dot(embedding,original_embedding)/(np.linalg.norm(embedding)*np.linalg.norm(original_embedding)))
            emb_ratio=float(np.linalg.norm(embedding)/np.linalg.norm(original_embedding))
            if emb_cos<.999 or not .98<emb_ratio<1.02:raise RuntimeError('Embedding C ABI parity failed')
            if CONDITIONING and not tau:
                lib.openvoice2_extract_speaker_embedding_segments.argtypes=[ct.c_void_p,floatp,ct.c_int,ct.c_int,
                                                                           ct.POINTER(ct.c_int),ct.POINTER(ct.c_int),ct.c_int,floatp]
                lib.openvoice2_extract_speaker_embedding_segments.restype=ct.c_bool
                spans=captures['segment_spans'].astype(np.int32)
                starts=np.ascontiguousarray(spans[[0,2]]);ends=np.ascontiguousarray(spans[[1,3]])
                average=np.empty(256,dtype=np.float32)
                guard_resources('native-clip-mean')
                ok=lib.openvoice2_extract_speaker_embedding_segments(ov,target_pcm.ctypes.data_as(floatp),len(target_pcm),22050,
                    starts.ctypes.data_as(ct.POINTER(ct.c_int)),ends.ctypes.data_as(ct.POINTER(ct.c_int)),2,average.ctypes.data_as(floatp))
                if not ok:raise RuntimeError('Native segment averaging failed')
                truth_mean=captures['segment_mean'].flatten()
                average_cos=float(np.dot(average,truth_mean)/(np.linalg.norm(average)*np.linalg.norm(truth_mean)))
                if average_cos<.999 or not .98<float(np.linalg.norm(average)/np.linalg.norm(truth_mean))<1.02:
                    raise RuntimeError('Original clip-mean embedding parity failed')
                lib.openvoice2_set_target_embedding.argtypes=[ct.c_void_p,floatp];lib.openvoice2_set_target_embedding.restype=ct.c_bool
                if not lib.openvoice2_set_target_embedding(ov,average.ctypes.data_as(floatp)):raise RuntimeError('Mean conditioning rejected')
                ptr=floatp();size=ct.c_int()
                guard_resources('native-mean-conditioned-conversion')
                if not lib.openvoice2_convert(ov,source_pcm.ctypes.data_as(floatp),len(source_pcm),22050,
                    target_pcm.ctypes.data_as(floatp),len(target_pcm),22050,ct.byref(ptr),ct.byref(size)):
                    raise RuntimeError('Mean-conditioned conversion failed')
                try:mean_wave=np.ctypeslib.as_array(ptr,shape=(size.value,)).copy()
                finally:libc.free(ptr)
                original_mean_wave=captures['segment_output_pcm'].flatten()
                mean_cos=float(np.dot(mean_wave.astype(np.float64),original_mean_wave)/(np.linalg.norm(mean_wave.astype(np.float64))*np.linalg.norm(original_mean_wave)))
                mean_ratio=float(np.linalg.norm(mean_wave)/np.linalg.norm(original_mean_wave))
                if mean_cos<.99 or not .98<mean_ratio<1.02:raise RuntimeError('Mean-conditioned waveform parity failed')
                if not lib.openvoice2_set_target_embedding(ov,None):raise RuntimeError('Clearing conditioning failed')
                RECEIPT.setdefault('clip_means',[]).append({'case':i,'embedding_cosine':average_cos,'waveform_cosine':mean_cos,'waveform_norm_ratio':mean_ratio})
            sf.write(OUT/f'clone-{i}.wav',native[0],22050,subtype='FLOAT')
            RECEIPT['cases'].append({'text':text,'speaker':key,'tau':tau,'ref_sha256':sha(ref),'gpu_parity':True,'published_revision':reference_checkpoint.get('published_revision'),
                                     'uninstrumented_waveform_cosine':cosine,'embedding_cosine':emb_cos,'embedding_norm_ratio':emb_ratio,'repeat_exact':True})
            save()
        lib.melotts_free(ctx)
        lib.openvoice2_free(ov)
        RECEIPT['gpu_executed']=True
        # Independent ASR content and speaker encoder, not OpenVoice's own
        # embedding space, judge actual converted waveforms.
        quality_script=WORK/'quality.py'
        quality_script.write_text("""import sys,json,re
from pathlib import Path
from huggingface_hub import HfApi,snapshot_download
from faster_whisper import WhisperModel
from resemblyzer import VoiceEncoder,preprocess_wav
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
from math import gcd
out=Path(sys.argv[1]);receipt=json.loads((out/'validation.json').read_text())
hub=HfApi();revision=hub.model_info('Systran/faster-whisper-tiny.en').sha
model=snapshot_download('Systran/faster-whisper-tiny.en',revision=revision)
asr=WhisperModel(model,device='cpu',compute_type='int8',cpu_threads=2)
encoder=VoiceEncoder(device=receipt['torch_reference_device'])
reference=encoder.embed_utterance(preprocess_wav(out/'reference.wav'))
def asr_audio(path):
 # Decode our PCM WAVs directly: PyAV versions differ on metadata_errors.
 # Whisper's array input must already be mono float32 at 16 kHz.
 pcm,rate=sf.read(path,dtype='float32',always_2d=True)
 pcm=pcm.mean(axis=1)
 if rate!=16000:
  divisor=gcd(rate,16000);pcm=resample_poly(pcm,16000//divisor,rate//divisor)
 return np.ascontiguousarray(pcm,dtype=np.float32)
def wer(expected,actual):
 a=re.findall(r'[a-z]+',expected.lower());b=re.findall(r'[a-z]+',actual.lower());row=list(range(len(b)+1))
 for i,x in enumerate(a,1):
  nxt=[i]
  for j,y in enumerate(b,1):nxt.append(min(row[j]+1,nxt[-1]+1,row[j-1]+(x!=y)))
  row=nxt
 return row[-1]/max(1,len(a))
for i,case in enumerate(receipt['cases']):
 quality={}
 for name in ['baseline','original','clone']:
  path=out/f'{name}-{i}.wav';segments,_=asr.transcribe(asr_audio(path),language='en',beam_size=5)
  text=' '.join(segment.text.strip() for segment in segments)
  vector=encoder.embed_utterance(preprocess_wav(path))
  quality[name]={'transcript':text,'wer':wer(case['text'],text),'speaker_cosine':float(np.dot(vector,reference))}
 case['quality']=quality
 case['quality_pass']=all(q['wer']<=.2 for q in quality.values()) and quality['clone']['speaker_cosine']>quality['baseline']['speaker_cosine']+.01
receipt['asr']={'repo':'Systran/faster-whisper-tiny.en','revision':revision}
(out/'quality.json').write_text(json.dumps(receipt,indent=2))
""")
        save()
        run([sys.executable,quality_script,OUT],'independent-quality')
        quality=json.loads((OUT/'quality.json').read_text())
        RECEIPT['cases']=quality['cases'];RECEIPT['asr']=quality['asr']
        RECEIPT['references_published']=bool(RECEIPT['cases']) and all(case.get('published_revision') for case in RECEIPT['cases'])
        RECEIPT['model_acceptance']=all(case['quality_pass'] for case in RECEIPT['cases'])
        RECEIPT['complete']=True
        if RECEIPT['model_acceptance'] and PUBLISH:
            try:
                repo='cstr/openvoice2-tcc-GGUF'
                hub.create_repo(repo,repo_type='model',private=False,exist_ok=True)
                published=hub.upload_file(path_or_fileobj=str(converted),repo_id=repo,
                    path_in_repo=converted.name,commit_message='OpenVoice V2 F32-sensitive conversion with recorded parity')
                RECEIPT['published_model']={'repo':repo,'revision':published.oid,**RECEIPT['converted_model']}
                readme=('---\nlicense: mit\npipeline_tag: audio-to-audio\n---\n\n'
                        '# OpenVoice V2 tone-color converter\n\n'
                        'GGUF for CrispASR. Reference encoder, posterior encoder and flows retain F32; decoder weights use F16.\n\n'
                        'Original MIT model: https://huggingface.co/myshell-ai/OpenVoiceV2/tree/'+model_rev+'\n\n'
                        'This is a voice converter, not a standalone text-to-speech model. Use the exact source speaker embedding.\n'
                        'The recorded validation is bounded synthetic English speech; broader languages, natural references and devices remain unqualified.\n'
                        'Reference recipe: https://github.com/CrispStrobe/CrispASR/blob/'+SOURCE+'/docs/openvoice2-baking.md\n')
                hub.upload_file(path_or_fileobj=readme.encode(),repo_id=repo,path_in_repo='README.md',commit_message='Document model scope and upstream license')
                hub.upload_file(path_or_fileobj=str(upstream/'LICENSE'),repo_id=repo,path_in_repo='LICENSE',commit_message='Retain original MIT notice')
            except Exception as exc:
                RECEIPT['model_publication_error']=type(exc).__name__
        RECEIPT['publication_complete']=bool(PUBLISH and RECEIPT.get('published_model')
            and all(r.get('published_revision') for r in RECEIPT.get('reference_checkpoints',[]))
            and not RECEIPT.get('model_publication_error'))
        RECEIPT['artifacts']={p.name:{'bytes':p.stat().st_size,'sha256':sha(p)} for p in OUT.iterdir() if p.is_file() and p.name!='validation.json'}
        save()
        kh.step('parity-complete',cases=len(RECEIPT['cases']),quality=RECEIPT['model_acceptance'])
        if not RECEIPT['model_acceptance']:
            raise RuntimeError('Content or independent speaker-similarity gate failed; preserve original/native controls')
        if PUBLISH and not RECEIPT['publication_complete']:
            raise RuntimeError('Requested publication incomplete; see retained publication errors')
    finally:
        os.dup2(original_stderr,2);os.close(original_stderr);native_log.close()

if __name__=='__main__':
    try:main()
    except Exception:
        RECEIPT['error']=traceback.format_exc();save();raise
