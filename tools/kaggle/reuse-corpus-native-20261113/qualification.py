"""Offline, bounded original/native RE-USE corpus intermediate diagnostic.

prepare.py appends the verified source/native-tool payload to the submitted copy.
The tracked runner remains readable. No network or package installation occurs.
"""
import base64
import gc
import gzip
import lzma
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tarfile
import io
import time
import signal

VERSION = 'reuse-corpus-native-20261113-v3'
SOURCE = '8d3b6aa393046b048643709555efc76719ee0bcc'
EXPECTED_STAGES = (['frontend_magnitude', 'frontend_phase'] +
                   [f'encoder_layer_{i}' for i in range(6)] +
                   [name for i in range(30) for name in (f'block_time_{i}', f'block_{i}')] +
                   ['amplitude', 'output_phase', 'waveform'])
CASES = {
    'speaker-61-white-10db': {
        'wav': '063c76b076997232acdb1d9b614c8a9b2eb51afd568e1e29d925fcc34fd73843',
        'raw': '94fa6fafd589bc8ab50cde86623467e2f8871881881daa9061b29cee3b0d2f9c'},
    'speaker-121-white-0db': {
        'wav': 'ea9a324d9e4abc3cad89d884f118114b77aff492cef4fc4eea76499445f219bb',
        'raw': '824d9a5921b2fa4925bb9660d551f46f7b3153d4873762a103e80236136b756c'},
}
PAYLOAD = None  # replaced in the generated single-file submission
ACTIVE_CHILD = None


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def save(path, value):
    temporary = path.with_suffix('.partial')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def unpack(payload, destination):
    data = lzma.decompress(base64.b85decode(payload['data']))
    if hashlib.sha256(data).hexdigest() != payload['sha256']:
        raise ValueError('payload hash mismatch')
    with tarfile.open(fileobj=io.BytesIO(data)) as archive:
        members = archive.getmembers()
        names = [item.name for item in members]
        if len(set(names)) != len(names) or set(names) != set(payload['manifest']):
            raise ValueError('payload manifest has duplicate, missing or unlisted members')
        for member in members:
            path = Path(member.name)
            if not member.isfile() or path.is_absolute() or '..' in path.parts:
                raise ValueError('unsafe payload member')
            if hashlib.sha256(archive.extractfile(member).read()).hexdigest() != payload['manifest'][member.name]:
                raise ValueError('payload member hash mismatch')
        archive.extractall(destination, members=members)
    return payload['manifest']


def unique_file(root, name, digest=None):
    paths = {str(p.resolve()): p for p in root.rglob(name)}
    if digest:
        paths = {key: p for key, p in paths.items() if sha(p) == digest}
    if len(paths) != 1:
        raise ValueError('expected exactly one verified input: ' + name)
    return next(iter(paths.values()))


def inherit_vendor(root):
    os.environ['PYTHONPATH'] = os.pathsep.join([str(root/'vendor'), os.environ.get('PYTHONPATH', '')])


def parse_rows(log):
    groups = {'unconditioned': {}, 'conditioned': {}}
    pattern = re.compile(r'^mkl-avx2(-conditioned)?/(\S+)\s+(PASS|FAIL)\s+rel_rms=')
    for line in log.splitlines():
        match = pattern.match(line)
        if match:
            arm = 'conditioned' if match[1] else 'unconditioned'
            if match[2] in groups[arm]:
                raise ValueError('duplicate stage: ' + match[2])
            groups[arm][match[2]] = match[3]
    for arm, rows in groups.items():
        if set(rows) != set(EXPECTED_STAGES) or any(x != 'PASS' for x in rows.values()):
            raise ValueError(arm + ': unchanged 71-stage native gate failed')
    return {arm: len(rows) for arm, rows in groups.items()}


def run(args, log, env=None, timeout=None):
    global ACTIVE_CHILD
    with log.open('w') as stream:
        proc = subprocess.Popen(list(map(str, args)), stdout=stream, stderr=subprocess.STDOUT,
                                env=env, start_new_session=True)
        ACTIVE_CHILD = proc.pid
        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
            raise TimeoutError(f'{log.name}: child deadline reached; log retained')
        finally:
            ACTIVE_CHILD = None
    if proc.returncode:
        raise RuntimeError(f'{log.name}: child exit {proc.returncode}; complete log retained')


def main():
    if PAYLOAD is None:
        raise ValueError('run prepare.py to freeze the single-file submission first')
    os.environ.update(OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1',
                      MKL_ENABLE_INSTRUCTIONS='AVX2', CRISPASR_REUSE_FRONTEND='mkl-avx2',
                      HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    os.environ.pop('MKL_CBWR', None)
    root = Path(os.environ.get('REUSE_CORPUS_TEMP', '/kaggle/temp/reuse-corpus-native'))
    out = Path(os.environ.get('REUSE_CORPUS_OUT', '/kaggle/working'))
    inputs = Path(os.environ.get('REUSE_CORPUS_INPUT', '/kaggle/input'))
    root.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    manifest = unpack(PAYLOAD, root)
    for name, digest in manifest.items():
        if sha(root/name) != digest:
            raise ValueError('extracted source/tool hash mismatch: ' + name)
    source = root/'source'
    sys.path[:0] = [str(source/'tools'), str(source/'tools/kaggle'), str(root/'vendor'),
                   str(source/'tools/kaggle/reuse-quality-cpu-20261112'),
                   str(source/'tools/ci-heavy/issue579_reuse')]
    inherit_vendor(root)
    import numpy as np
    import torch
    import soundfile as sf
    import kaggle_harness as kh
    from dump_reference import write_gguf_archive
    from reference_backends import reuse
    from retain_reference import retain_reference
    kh.init_progress(progress_path=out/'progress.jsonl', hf_progress_repo='')
    torch.set_num_threads(2)
    torch.set_grad_enabled(False)
    torch_lib = Path(torch.__file__).parent/'lib'
    runtime = torch_lib/'libtorch_cpu.so'
    if not runtime.is_file():
        raise ValueError('preinstalled Torch CPU runtime unavailable')
    os.environ['LD_LIBRARY_PATH'] = ':'.join([str(root/'native'), str(torch_lib), os.environ.get('LD_LIBRARY_PATH', '')])
    provenance = dict(version=VERSION, source=SOURCE, script_sha256=sha(__file__), payload=manifest,
                      torch=torch.__version__, torch_config=torch.__config__.show(),
                      torch_cpu_sha256=sha(runtime), python=sys.version,
                      selection='Data-informed contrasting diagnostics from the retained four-speaker 16-case matrix; no corpus-wide claim.',
                      scope='Original upstream model with audited CPU non-fused adapter versus scalar native research comparator; no product backend or GPU qualification.')
    save(out/'provenance.json', provenance)
    # Resolve and hash exact retained noise bytes before build/model work.
    selected = []
    for case_id, expected in CASES.items():
        path = unique_file(inputs, case_id+'-input.wav', expected['wav'])
        audio, rate = sf.read(path, dtype='float32')
        if rate != 16000 or audio.shape != (80000,) or not np.isfinite(audio).all():
            raise ValueError('expected exactly five seconds of finite mono 16 kHz float audio')
        if hashlib.sha256(audio.tobytes()).hexdigest() != expected['raw']:
            raise ValueError('retained noise sample bytes differ')
        shutil.copyfile(path, out/path.name)
        old = json.loads(unique_file(inputs, case_id+'.json').read_text())
        old_output = unique_file(inputs, case_id+'-output.wav', old['output_sha256'])
        previous_wave, previous_rate = sf.read(old_output, dtype='float32')
        if previous_rate != 16000 or previous_wave.shape != audio.shape:
            raise ValueError('invalid retained original output waveform')
        selected.append((case_id, audio, old, previous_wave))
    model = unique_file(inputs, 'model.safetensors', '06badaf4029b297f97f7fe48d62990b48d5482608210df577354e0d4e0ed70e1').parent
    from quality_cpu import MODEL_HASHES, verify_pinned_files
    provenance['model_files'] = verify_pinned_files(model, MODEL_HASHES)
    save(out/'provenance.json', provenance)
    # Public DFTI/VML symbols only. These executables were built with official
    # SDK headers; headers and proprietary runtime are not redistributed.
    for path in (root/'native').iterdir():
        path.chmod(0o755)
    env = os.environ.copy()
    env.update(REUSE_MKL_FRONTEND_PROBE=str(root/'native/reuse-frontend-mkl-probe'),
               REUSE_MKL_PROFILE_CONTROL=str(root/'native/reuse-profile-mkl'))
    run([sys.executable, source/'tests/test_reuse_mkl_probe.py'], out/'frontend-controls.log', env)
    # Compile the native model/diff code on the worker, avoiding host ELF/glibc
    # assumptions. Imported frontend supplies the existing optional target ABI.
    imported = root/'import-frontend.cmake'
    imported.write_text('add_library(crispasr-reuse-mkl SHARED IMPORTED GLOBAL)\n'
                        f'set_target_properties(crispasr-reuse-mkl PROPERTIES IMPORTED_LOCATION "{root}/native/libreuse-frontend.so" INTERFACE_COMPILE_DEFINITIONS CRISPASR_REUSE_HAVE_MKL INTERFACE_LINK_LIBRARIES "{runtime}")\n')
    # install_build_toolchain invokes apt/pip, so its internet bootstrap is
    # deliberately replaced with fail-fast discovery for this offline worker.
    for tool in ('cmake', 'make', 'g++'):
        if not shutil.which(tool):
            raise ValueError('offline worker lacks required build tool: ' + tool)
    kh._HAS_CCACHE = shutil.which('ccache') is not None
    kh._HAS_MOLD = shutil.which('mold') is not None
    os.environ.update(CCACHE_DIR=str(root/'ccache'), CCACHE_MAXSIZE='256M')
    build = root/'build'
    run(['cmake', '-S', source/'tools/ci-heavy/issue579_reuse', '-B', build,
         '-DCMAKE_BUILD_TYPE=Release', '-DGGML_CUDA=OFF', '-DGGML_METAL=OFF', '-DGGML_BLAS=OFF',
         '-DCRISPASR_REUSE_MKL_FRONTEND=OFF',
         f'-DCMAKE_PROJECT_reuse_research_tools_INCLUDE={imported}',
         *kh.cache_and_link_flags()], out/'configure.log')
    with kh.build_heartbeat('corpus.native.build', interval_s=30):
        # safe_build_jobs(False) returns shell syntax; this runner uses argv.
        jobs = kh.safe_build_jobs(gpu=False)
        jobs = str(min(2, os.cpu_count() or 1)) if jobs == '$(nproc)' else jobs
        run(['cmake', '--build', build, '--target', 'reuse-long-diff', 'reuse-diff-controls',
             '-j'+jobs], out/'build.log')
    run(['ctest', '--test-dir', build, '-R', '^reuse-diff-controls$', '--no-tests=error', '--output-on-failure'], out/'diff-controls.log')
    model_gguf = root/'model.gguf'
    run([sys.executable, source/'models/convert-reuse-to-gguf.py', '--input', model, '--output', model_gguf], out/'convert.log')
    provenance.update(native_model_sha256=sha(model_gguf), comparator_sha256=sha(build/'reuse-long-diff'))
    save(out/'provenance.json', provenance)
    results = []
    for case_id, audio, old, previous_wave in selected:
        kh.step('corpus.reference.start', case_id=case_id)
        with kh.build_heartbeat(case_id+'.reference', interval_s=30):
            captures = reuse.dump(model, audio, set(reuse.DEFAULT_STAGES), max_samples=80000)
        reference_wave = captures['waveform'].copy()
        old_comparison = {}
        for name, item in old['stage_fingerprints'].items():
            value = np.ascontiguousarray(captures[name], dtype='float32')
            norm = float(np.linalg.norm(value.astype('float64')))
            old_comparison[name] = dict(shape_matches=list(value.shape)==item['shape'],
                                       sha256_matches=hashlib.sha256(value.tobytes()).hexdigest()==item['sha256'],
                                       fresh_norm=norm, old_norm=item['norm'])
        save(out/(case_id+'-fingerprint-diagnostic.json'), old_comparison)
        # A new full archive is the numerical oracle. Old fingerprints only
        # diagnose environment changes and never count as numerical parity.
        reference = root/(case_id+'-reference.gguf')
        write_gguf_archive(captures, {'backend': 'reuse'}, reference)
        del captures
        gc.collect()
        artifact = retain_reference(reference, out/reference.name)
        reference.unlink()
        result = dict(case_id=case_id, reference=artifact, input=CASES[case_id], status='reference-retained',
                      fresh_vs_retained_original_rel_rms=float(np.linalg.norm(reference_wave-previous_wave)/np.linalg.norm(previous_wave)),
                      fresh_vs_retained_original_max_abs=float(np.max(abs(reference_wave-previous_wave))))
        results.append(result)
        save(out/'case-results.json', results)
        sf.write(out/(case_id+'-reference.wav'), reference_wave, 16000, subtype='FLOAT')
        native = out/(case_id+'-native.f32')
        log = out/(case_id+'-diff.log')
        with kh.build_heartbeat(case_id+'.native', interval_s=30):
            run([build/'reuse-long-diff', model_gguf, out/reference.name, native], log)
        result.update(parse_rows(log.read_text()))
        wave = np.fromfile(native, dtype='<f4')
        if wave.shape != audio.shape or not np.isfinite(wave).all():
            raise ValueError('invalid native waveform')
        sf.write(out/(case_id+'-native.wav'), wave, 16000, subtype='FLOAT')
        result.update(status='pass', native_reference_rel_rms=float(np.linalg.norm(wave-reference_wave)/np.linalg.norm(reference_wave)))
        save(out/'case-results.json', results)
        kh.step('corpus.native.done', case_id=case_id, unconditioned=71, conditioned=71)
    save(out/'result.json', dict(status='pass', cases=results, elapsed_seconds=time.monotonic()-start))


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        out = Path(os.environ.get('REUSE_CORPUS_OUT', '/kaggle/working'))
        out.mkdir(parents=True, exist_ok=True)
        save(out/'failure.json', dict(status='failed', error_type=type(error).__name__, error=str(error)))
        raise
