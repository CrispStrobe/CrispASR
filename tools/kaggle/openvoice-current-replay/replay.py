#!/usr/bin/env python3
"""Frozen OpenVoice inputs on the shipped e2 ggml ABI; no full SDK build."""
import ctypes as ct
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import traceback
import urllib.request

WORKER_SOURCE = 'd84007612167105cf24c7af80014d1f0ea1bd8a3'
WORKER_SHA256 = 'b1fed1e4518ceeb517c749be67ce3eeee9d3243153b5f50043f6edf9c4584f20'
SOURCE = '5f87a56265e4bbd7e2adc7052254e576d5774abb'
GGML = 'e2ae679a563014f46c546ce6c6448e8601e4dea6'
SDK_NAME = 'libcrispasr-linux-x86_64-cuda'
SDK_SHA256 = '8fbd68e1822cd16e58055753fc67120c20784451e286d94177f7f7fb78b76cbd'
SDK_BYTES = 196467235
SESSION = os.environ.get('CRISPASR_OV2_SESSION') == '1'
VERSION = ('2026-10-28.openvoice-session.2' if SESSION else
           '2026-10-28.openvoice-current-replay.2')
WORK = Path('/kaggle/temp/openvoice-current-replay')
OUT = Path('/kaggle/working')


def checked_download(url, path, digest):
    with urllib.request.urlopen(url, timeout=120) as response, path.open('wb') as stream:
        shutil.copyfileobj(response, stream, 4 * 1024 * 1024)
    if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
        raise ValueError('Checksum mismatch: ' + path.name)


def load_worker():
    WORK.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    path = WORK / 'shared-worker.py'
    checked_download('https://raw.githubusercontent.com/CrispStrobe/CrispASR/' + WORKER_SOURCE +
                     '/tools/kaggle/openvoice-bake/openvoice_bake.py', path, WORKER_SHA256)
    spec = importlib.util.spec_from_file_location('openvoice_shared', path)
    worker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(worker)
    worker.WORK = WORK
    worker.ROOT = WORK / 'source'
    worker.BUNDLE = WORK / SDK_NAME / 'lib'
    worker.OUT = OUT
    worker.RECEIPT = dict(schema=1, version=VERSION, source=SOURCE, ggml=GGML,
        native_model_delta_source=worker.MODEL_DELTA_SOURCE,
        native_model_delta_sha256=worker.MODEL_DELTA_SHA256,
        worker_source=WORKER_SOURCE, worker_sha256=WORKER_SHA256,
        entry_sha256=worker.sha(Path(__file__)), complete=False, model_acceptance=False,
        gpu_executed=False, publication_requested=False, publication_complete=False,
        sdk=dict(tag='v0.8.47', asset=SDK_NAME + '.tar.gz', sha256=SDK_SHA256, bytes=SDK_BYTES),
        limitations=['Frozen tau=0 structural replay only; speech quality is a separate gate',
                     'Only OpenVoice/Melo model objects rebuilt; shipped SDK supplies core and ggml',
                     'High-level Session and CLI are not qualified by this replay'])
    return worker


def check_symbol_owners(library):
    """The shipped SDK contains older model symbols: prove our overlay owns them."""
    class DlInfo(ct.Structure):
        _fields_ = [('filename', ct.c_char_p), ('base', ct.c_void_p),
                    ('symbol', ct.c_char_p), ('address', ct.c_void_p)]
    dl = ct.CDLL('libdl.so.2')
    dl.dladdr.argtypes = [ct.c_void_p, ct.POINTER(DlInfo)]
    dl.dladdr.restype = ct.c_int
    handle = ct.CDLL(str(library), mode=os.RTLD_LOCAL | os.RTLD_NOW)
    owners = {}
    for name in ['openvoice2_context_default_params', 'openvoice2_init_from_file',
                 'openvoice2_convert', 'openvoice2_set_stage_callback']:
        info = DlInfo()
        if not dl.dladdr(ct.cast(getattr(handle, name), ct.c_void_p), ct.byref(info)):
            raise RuntimeError('Cannot resolve overlay owner: ' + name)
        owner = Path(os.fsdecode(info.filename)).resolve()
        if owner != library.resolve():
            raise RuntimeError('Model symbol interposed by shipped SDK: ' + name)
        owners[name] = owner.name
    return owners


def bootstrap(worker):
    worker.RECEIPT['hardware'] = subprocess.check_output(
        ['nvidia-smi', '--query-gpu=name,compute_cap,memory.total,driver_version',
         '--format=csv,noheader'], text=True, env=worker.host_environment()).strip()
    worker.guard_resources('bootstrap-current-sdk')
    root = worker.ROOT
    for command in (['git', 'init', root], ['git', '-C', root, 'remote', 'add', 'origin',
                    'https://github.com/CrispStrobe/CrispASR.git'],
                    ['git', '-C', root, 'fetch', '--depth=1', 'origin', SOURCE],
                    ['git', '-C', root, 'checkout', 'FETCH_HEAD'],
                    ['git', '-C', root, 'submodule', 'update', '--init', '--depth=1', 'ggml']):
        subprocess.run(list(map(str, command)), check=True)
    actual = subprocess.check_output(['git', '-C', root / 'ggml', 'rev-parse', 'HEAD'], text=True).strip()
    if actual != GGML:
        raise ValueError('SDK/header ggml mismatch')
    worker.download('https://raw.githubusercontent.com/CrispStrobe/CrispASR/' +
                    worker.MODEL_DELTA_SOURCE + '/src/openvoice2.cpp',
                    root / 'src/openvoice2.cpp', worker.MODEL_DELTA_SHA256)
    archive_path = WORK / (SDK_NAME + '.tar.gz')
    worker.download('https://github.com/CrispStrobe/CrispASR/releases/download/v0.8.47/' +
                    archive_path.name, archive_path, SDK_SHA256)
    if archive_path.stat().st_size != SDK_BYTES:
        raise ValueError('SDK byte count mismatch')
    with tarfile.open(archive_path) as archive:
        archive.extractall(WORK, filter='data')
    sdk = WORK / SDK_NAME
    for notice in ['LICENSE', 'THIRD_PARTY_NOTICES.txt']:
        shutil.copy2(sdk / notice, OUT / ('SDK-' + notice))
    # No vendor SDK installs. Record and use only the host's existing providers.
    providers = sorted({str(p) for base in [Path('/usr/local/lib'), Path('/opt/conda/lib')]
                        for p in base.glob('python*/**/nvidia/*/lib') if p.is_dir()})
    providers += [str(p) for p in [Path('/usr/local/cuda/lib64'), Path('/usr/local/nvidia/lib64')]
                  if p.is_dir()]
    host_path = os.environ.get('LD_LIBRARY_PATH', '')
    worker.RECEIPT['host_provider_directories'] = [Path(p).parent.name + '/' + Path(p).name
                                                 for p in providers]
    worker.save()
    (WORK / 'bootstrap.json').write_text(json.dumps(worker.RECEIPT))
    env = dict(os.environ, CRISPASR_OV2_CURRENT_READY='1',
               CRISPASR_OV2_HOST_LD_LIBRARY_PATH=host_path,
               LD_LIBRARY_PATH=os.pathsep.join([str(worker.BUNDLE), *providers, host_path]))
    os.execve(sys.executable, [sys.executable, str(Path(__file__).resolve())], env)


def check_dependencies(worker):
    for name in ['libcrispasr.so', 'libggml-cuda.so']:
        label = 'dependencies-' + name
        worker.run(['ldd', '-r', worker.BUNDLE / name], label)
        log = (worker.OUT / (label + '.log')).read_text()
        if 'not found' in log or 'undefined symbol:' in log:
            raise RuntimeError('Unresolved host runtime dependency: ' + name)


def main(worker):
    os.environ.update(TMPDIR=str(WORK), OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1',
                      CRISPASR_OV2_GPU='1', GGML_SCHED_DEBUG='1')
    if os.environ.get('CRISPASR_OV2_CURRENT_READY') != '1':
        bootstrap(worker)
    worker.RECEIPT = json.loads((WORK / 'bootstrap.json').read_text())
    root, bundle = worker.ROOT, worker.BUNDLE
    if worker.sha(root / 'src/openvoice2.cpp') != worker.MODEL_DELTA_SHA256:
        raise ValueError('Model overlay changed after bootstrap')
    os.chdir(bundle)
    check_dependencies(worker)
    # This SDK uses linked backends (unlike the older frozen DL runtime).
    common = ['c++', '-std=c++17', '-O2', '-fPIC', '-fopenmp',
              '-I' + str(root / 'src'), '-I' + str(root / 'ggml/include'),
              '-I' + str(root / 'examples/cli'), '-L' + str(bundle), '-Wl,-rpath,' + str(bundle)]
    library = WORK / 'libopenvoice-probe.so'
    worker.run(common + [root / 'src/openvoice2.cpp', root / 'src/melotts.cpp', '-shared',
               '-Wl,-Bsymbolic-functions', '-lcrispasr', '-lggml', '-lggml-base', '-lggml-cpu', '-ldl',
               '-pthread', '-o', library], 'compile-current-model-delta')
    worker.run(common + [root / 'tools/ci-heavy/openvoice-bake/main.cpp',
               root / 'examples/cli/crispasr_diff.cpp', '-L' + str(WORK), '-Wl,-rpath,' + str(WORK),
               '-lopenvoice-probe', '-lcrispasr', '-lggml', '-lggml-base', '-lggml-cpu', '-ldl', '-pthread',
               '-o', WORK / 'openvoice-bake'], 'compile-current-diff-driver')
    worker.RECEIPT['symbol_owners'] = check_symbol_owners(library)
    worker.RECEIPT['native_delta'] = {p.name: worker.sha(p) for p in [library, WORK / 'openvoice-bake']}
    worker.save()
    worker.replay(worker.load_replay_inputs())
    worker.RECEIPT['replay_scope'] = ('Frozen tau=0 CPU/CUDA replay with v0.8.47 SDK, matching e2 '
        'ggml headers and verified model overlay; no speech quality or Session/CLI acceptance')
    worker.save()
    if SESSION:
        session_validation(worker)


def session_validation(worker):
    """Actual Session callers over the qualified SDK/model overlay composition."""
    import ast
    worker.RECEIPT.update(complete=False, session_acceptance=False)
    worker.save()
    candidates = sorted(p for p in Path('/kaggle/input').rglob('validation.json')
                        if 'crispasr-openvoice-conditioning' in p.parts)
    if not candidates:
        raise RuntimeError('Accepted conditioning output is not mounted')
    if worker.sha(candidates[0]) != '46badec89720409414dad5c61f3e5cc1c34928a3e0b8391a1cf4f92e4d90dec8':
        raise ValueError('Accepted conditioning receipt checksum mismatch')
    evidence = json.loads(candidates[0].read_text())
    if not (evidence['version'] == '2026-10-28.openvoice-conditioning.6'
            and evidence['complete'] and evidence['model_acceptance'] and len(evidence['cases']) == 3):
        raise ValueError('Conditioning input is not the accepted version 4')
    for name in ['reference.wav', 'original-0.wav', 'original-1.wav',
                 'reference-ATTRIBUTION.txt', 'reference-OpenVoice-LICENSE.txt']:
        path = candidates[0].parent / name
        if worker.sha(path) != evidence['artifacts'][name]['sha256']:
            raise ValueError('Conditioning artifact checksum mismatch: ' + name)
        shutil.copy2(path, OUT / name)
    revision = evidence['melo']['revision']
    model, bert = WORK / 'opaque-engine.gguf', WORK / 'bert.gguf'
    for path, filename, digest in [(model, 'melotts-en-v2-f16.gguf', evidence['melo']['model_sha256']),
                                   (bert, 'bert-base-uncased-q4k.gguf', evidence['melo']['bert_sha256'])]:
        worker.download('https://huggingface.co/cstr/melotts-en-v2-GGUF/resolve/' + revision + '/' + filename,
                        path, digest)
    inputs = worker.load_replay_inputs()
    worker.run([sys.executable, '-m', 'pip', 'install', '-q', 'soundfile', 'scipy',
                'faster-whisper', 'resemblyzer', 'webrtcvad-wheels'], 'session-quality-dependencies')
    script = WORK / 'session-check.py'
    script.write_text(SESSION_SCRIPT)
    env = dict(os.environ, LD_PRELOAD=str(WORK / 'libopenvoice-probe.so'),
               CRISPASR_MELOTTS_BERT=str(bert), CRISPASR_OV2_GPU='1')
    worker.run([sys.executable, script, worker.ROOT, worker.BUNDLE, model, inputs['model'], OUT, WORK],
               'session-native', env=env)
    session = json.loads((OUT / 'session.json').read_text())
    log = (OUT / 'session-native.log').read_text()
    cuda = ('openvoice2: vocoder backend CUDA' in log and
            'HiFi-GAN output' in log and '## SPLIT #' in log and ': CUDA' in log)
    if not (session['native_complete'] and cuda and len(session['cases']) == 2):
        raise RuntimeError('Session did not prove native conversion and actual CUDA')
    worker.RECEIPT.update(cases=session['cases'], torch_reference_device='cuda',
                          session_native=session, session_cuda_observed=cuda)
    worker.save()
    # Use the accepted quality decoder and unchanged gates, adding raw output
    # to the required content controls alongside normal marked synthesis.
    source = WORK / 'quality-worker.py'
    worker.download('https://raw.githubusercontent.com/CrispStrobe/CrispASR/' + QUALITY_SOURCE +
                    '/tools/kaggle/openvoice-bake/openvoice_bake.py', source, QUALITY_SHA256)
    phases = [node.value for node in ast.walk(ast.parse(source.read_text()))
              if isinstance(node, ast.Constant) and isinstance(node.value, str)
              and node.value.startswith('import sys,json,re')]
    if len(phases) != 1:
        raise ValueError('Ambiguous pinned quality phase')
    quality = WORK / 'session-quality.py'
    quality.write_text(phases[0].replace("['baseline','original','clone']", "['baseline','original','raw','clone']"))
    worker.run([sys.executable, quality, OUT], 'session-independent-quality')
    result = json.loads((OUT / 'quality.json').read_text())
    accepted = len(result['cases']) == 2 and all(case['quality_pass'] for case in result['cases'])
    worker.RECEIPT.update(cases=result['cases'], asr=result['asr'], complete=True,
                          session_acceptance=accepted, model_acceptance=False,
                          session_scope='Python Session/C ABI on shipped v0.8.47 SDK with verified model overlay; not a rebuilt-current-library or CLI qualification')
    worker.save()
    if not accepted:
        raise RuntimeError('Session content or speaker quality gate failed')


QUALITY_SOURCE = 'c5287611ef8416704a216d1a91c1a0fea5127a66'
QUALITY_SHA256 = '61435c8198ae6de1443c73c36905e4e8b9293a1f54bb17761ead4472011c9179'
SESSION_SCRIPT = '''import importlib.util,json,os,sys
from pathlib import Path
import numpy as np
import soundfile as sf
# The loader has already installed the overlay. Do not inject it into resource
# probes, whose clean vendor environment intentionally excludes SDK paths.
os.environ.pop('LD_PRELOAD',None)
root,bundle,model,converter,out,work=map(Path,sys.argv[1:])
sys.path.insert(0,str(root/'python'))
from crispasr import Session
spec=importlib.util.spec_from_file_location('resources',work/'shared-worker.py')
resources=importlib.util.module_from_spec(spec);spec.loader.exec_module(resources)
resources.WORK=work;resources.OUT=out
record=dict(cases=[],native_complete=False,speaker_identity='unknown',raw_artifacts='Private model-generated reference validation; normal marked output tested separately')
resources.RECEIPT=record
resources.save=lambda:(out/'session.json').write_text(json.dumps(record,indent=2))
texts=['The quick brown fox jumps over the lazy dog.',
       'We enjoy reading books and listening to music in the evening.']
for i,text in enumerate(texts):
 resources.guard_resources('session-open')
 with Session(str(model),lib_path=str(bundle/'libcrispasr.so'),n_threads=2,use_gpu=True,verbosity=1) as s:
  if s.backend!='melotts':raise RuntimeError('C ABI architecture detection failed')
  # Generated reference PCM does not establish the identity of a preset donor.
  s.set_speaker_id(0);s.set_speaker_identity('unknown')
  s.accept_marking_responsibility('Private synthetic validation: raw clips stay private and labelled; normal marked synthesis is also checked')
  resources.guard_resources('session-baseline')
  s.set_tts_seed(42);baseline=s.synthesize_raw(text);rate=s.output_sample_rate()
  if rate!=44100 or not np.isfinite(baseline).all():raise RuntimeError('Baseline PCM/rate failed')
  sf.write(out/f'baseline-{i}.wav',baseline,rate,subtype='FLOAT')
  resources.guard_resources('session-codec-and-voice')
  s.set_codec_path(str(converter));s.set_voice(str(out/'reference.wav'))
  resources.guard_resources('session-raw-clone')
  s.set_tts_seed(42);raw=s.synthesize_raw(text)
  if s.output_sample_rate()!=22050 or not len(raw) or not np.isfinite(raw).all():raise RuntimeError('Clone PCM/rate failed')
  resources.guard_resources('session-repeat')
  s.set_tts_seed(42);repeat=s.synthesize_raw(text)
  if not np.array_equal(raw,repeat):raise RuntimeError('Reseeded Session clone differs')
  resources.guard_resources('session-marked-clone')
  s.set_tts_seed(42);marked=s.synthesize(text)
  if marked.shape!=raw.shape or not np.isfinite(marked).all() or np.array_equal(raw,marked):raise RuntimeError('Marked synthesis control failed')
  sf.write(out/f'raw-{i}.wav',raw,22050,subtype='FLOAT')
  sf.write(out/f'clone-{i}.wav',marked,22050,subtype='FLOAT')
  record['cases'].append(dict(text=text,backend=s.backend,baseline_rate=rate,clone_rate=22050,
                            repeat_exact=True,marked_differs=True,samples=len(raw)))
  resources.save()
record['native_complete']=True;resources.save()
'''
if __name__ == '__main__':
    shared = load_worker()
    try:
        main(shared)
    except Exception:
        shared.RECEIPT['error'] = traceback.format_exc()
        shared.save()
        raise
