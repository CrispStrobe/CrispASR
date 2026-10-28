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
VERSION = '2026-10-28.openvoice-current-replay.1'
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


if __name__ == '__main__':
    shared = load_worker()
    try:
        main(shared)
    except Exception:
        shared.RECEIPT['error'] = traceback.format_exc()
        shared.save()
        raise
