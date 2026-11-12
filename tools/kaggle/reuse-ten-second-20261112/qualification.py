"""Kaggle-only ten-second RE-USE candidate with immutable runtime source."""
import gc
import hashlib
import inspect
import json
import os
from pathlib import Path
import re
import resource
import shutil
import subprocess
import sys
import threading

# Freeze runtime source separately from this uploaded script, after integrating
# chunked(max_window_samples=...). Never replace this with a moving branch.
SOURCE_REVISION = 'a5a46b6c5b616b7556d3bfa27335d33a95c74794'
SCRIPT_VERSION = 'reuse-ten-second-20261112-v1'
BASE_REVISION = '7b4f0935ab287f128315a0898e72fb97e9c2a977'
MODEL_HASH = '06badaf4029b297f97f7fe48d62990b48d5482608210df577354e0d4e0ed70e1'
MODEL_REVISION = '31f82f2efad1095034492f236e8dac9239ba713b'
CONFIG_HASH = 'e69986483774f01893d2727503addc49ef201d19e6e1a266b8d4e971e28c478d'
EXPECTED_STAGES = (['frontend_magnitude', 'frontend_phase'] +
                   [f'encoder_layer_{i}' for i in range(6)] +
                   [name for i in range(30) for name in (f'block_time_{i}', f'block_{i}')] +
                   ['amplitude', 'output_phase', 'waveform'])


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def save_json(path, value):
    temporary = path.with_suffix(path.suffix + '.partial')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def parse_rows(log):
    """Require every unique stage in each arm; diagnostic rows cannot count."""
    groups = {'unconditioned': {}, 'conditioned': {}}
    pattern = re.compile(r'^mkl-avx2(-conditioned)?/(\S+)\s+(PASS|FAIL)\s+rel_rms=')
    for line in log.splitlines():
        match = pattern.match(line)
        if not match:
            continue
        arm = 'conditioned' if match[1] else 'unconditioned'
        if match[2] in groups[arm]:
            raise ValueError(f'duplicate {arm} stage: {match[2]}')
        groups[arm][match[2]] = match[3]
    for arm, rows in groups.items():
        if set(rows) != set(EXPECTED_STAGES):
            raise ValueError(f'{arm}: expected all 71 unique stages, got {len(rows)}')
        if any(status != 'PASS' for status in rows.values()):
            raise ValueError(f'{arm}: native intermediate gate failed')
    return {arm: len(rows) for arm, rows in groups.items()}


def live_tree_rss_kib(proc_root=Path('/proc'), root_pid=None):
    """Sum current Linux RSS for this process and discovered descendants.

    Status files can disappear while scanning. Shared pages are deliberately
    counted in each process, so this is a conservative resident-memory estimate.
    """
    root_pid = os.getpid() if root_pid is None else root_pid
    processes = {}
    for entry in proc_root.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            fields = dict(line.split(':', 1) for line in (entry/'status').read_text().splitlines() if ':' in line)
            processes[int(entry.name)] = (int(fields['PPid'].strip()), int(fields.get('VmRSS', '0 kB').split()[0]))
        except (OSError, KeyError, ValueError):
            continue  # process exited during this snapshot
    if root_pid not in processes:
        raise RuntimeError('unable to inspect resident-memory process tree')
    selected = {root_pid}
    while True:
        expanded = selected | {pid for pid, (parent, _) in processes.items() if parent in selected}
        if expanded == selected:
            break
        selected = expanded
    return sum(processes[pid][1] for pid in selected)


def check_budget(root, out):
    # Linux getrusage is KiB; include completed-child peaks and live tree RSS.
    peak = max(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
               resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss, live_tree_rss_kib())
    if peak > 10 * 1024 * 1024:
        raise MemoryError('research live tree or recorded process peak exceeded 10 GiB RSS budget')
    for directory in (root, out):
        if shutil.disk_usage(directory).free < 2 * 1024**3:
            raise OSError('less than 2 GiB scratch/output reserve remains')
    total = sum(path.stat().st_size for path in out.rglob('*') if path.is_file())
    if total > 8 * 1024**3:
        raise OSError('research output exceeded 8 GiB budget')


def run(args, **kwargs):
    print('+', ' '.join(map(str, args)), flush=True)
    return subprocess.run([str(arg) for arg in args], check=True, **kwargs)


def main():
    if not re.fullmatch(r'[0-9a-f]{40}', SOURCE_REVISION):
        raise ValueError('Freeze SOURCE_REVISION to the integrated immutable source before submission')
    root = Path('/kaggle/temp/reuse-ten-second-20261112')
    out = Path('/kaggle/working')
    root.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    os.environ.update(OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1',
                      MKL_ENABLE_INSTRUCTIONS='AVX2', CRISPASR_REUSE_FRONTEND='mkl-avx2',
                      PYTHONUNBUFFERED='1', TMPDIR=str(root), PIP_NO_CACHE_DIR='1',
                      HF_HOME=str(root/'hf-cache'))
    os.environ.pop('MKL_CBWR', None)
    # Fail before clone/build/model load on unsupported Torch GPU draws; no retries.
    import torch
    gpu = {'available': torch.cuda.is_available(), 'torch': torch.__version__}
    if gpu['available']:
        gpu.update(name=torch.cuda.get_device_name(0), capability=list(torch.cuda.get_device_capability(0)))
    save_json(out/'gpu-preflight.json', gpu)
    if not gpu['available'] or gpu['capability'][0] < 7:
        save_json(out/'ten-second-result.json', {'conclusive': False, 'reason': 'unsupported Torch CUDA device', 'gpu': gpu})
        return
    torch.ones(1, device='cuda').sum().item()  # verify actual kernel execution
    source = root/'source'
    run(['git', 'init', source])
    run(['git', '-C', source, 'remote', 'add', 'origin', 'https://github.com/CrispStrobe/CrispASR.git'])
    run(['git', '-C', source, 'fetch', '--depth', '1', 'origin', SOURCE_REVISION])
    run(['git', '-C', source, 'checkout', '--detach', 'FETCH_HEAD'])
    run(['git', '-C', source, 'submodule', 'update', '--init', '--recursive', '--depth', '1'])
    revision = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
    if revision != SOURCE_REVISION:
        raise ValueError('runtime source revision mismatch')
    sys.path[:0] = [str(source/'tools/kaggle'), str(source/'tools'), str(source/'tools/ci-heavy/issue579_reuse')]
    import kaggle_harness as kh
    kh.init_progress(progress_path=out/'progress.jsonl')
    token = kh.resolve_hf_token()
    if token:
        os.environ['HF_TOKEN'] = token
    kh.step('reuse.ten.start', source=revision, script_version=SCRIPT_VERSION)
    provenance = {'script_version': SCRIPT_VERSION, 'script_sha256': sha(__file__),
                  'base_revision': BASE_REVISION, 'source_revision': revision, 'gpu': gpu,
                  'submodules': subprocess.check_output(['git', '-C', str(source), 'submodule', 'status', '--recursive'], text=True),
                  'scope': 'One ten-second speech proxy plus synthetic noise; research interface only; no production, perceptual-quality, fused CUDA or native GPU-backend claim.'}
    save_json(out/'provenance.json', provenance)
    # Continuous disk/RSS guard preserves prior references if later work fails.
    stopped = threading.Event()
    def watch_budget():
        while not stopped.wait(5):
            try:
                check_budget(root, out)
            except Exception as error:
                save_json(out/'budget-failure.json', {'error': str(error)})
                os._exit(2)
    threading.Thread(target=watch_budget, daemon=True).start()
    check_budget(root, out)
    run([sys.executable, '-m', 'pip', 'install', '--quiet', '--no-deps', 'safetensors==0.5.3', 'einops==0.8.1',
         'gguf==0.17.1', 'soundfile==0.13.1', 'scipy==1.15.3', 'pystoi==0.4.1'])
    sdk = root/'sdk'
    run([sys.executable, '-m', 'pip', 'install', '--quiet', '--prefix', sdk, 'mkl==2025.0.1', 'mkl-include==2025.0.1'])
    header = next(sdk.rglob('mkl.h'))
    lib = next(sdk.rglob('libmkl_rt.so.2'))
    os.environ['LD_LIBRARY_PATH'] = ':'.join([str(lib.parent), os.environ.get('LD_LIBRARY_PATH', '')])
    import numpy as np
    import soundfile as sf
    from safetensors.torch import load_file
    from dump_reference import write_gguf_archive
    from reference_backends import reuse
    from retain_reference import retain_reference
    from chunked import chunked
    from quality_metrics import quality
    from pystoi import stoi
    if 'max_window_samples' not in inspect.signature(chunked).parameters:
        raise ValueError('frozen source lacks the explicit five-second chunking contract')
    torch.set_num_threads(2)
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    models = {}
    for directory, _, files in os.walk('/kaggle/input', followlinks=True):
        if {'config.json', 'model.safetensors'} <= set(files):
            path = Path(directory)
            if sha(path/'config.json') == CONFIG_HASH and sha(path/'model.safetensors') == MODEL_HASH:
                models[str(path.resolve())] = path
    if len(models) != 1:
        raise ValueError(f'expected exactly one hash-pinned RE-USE model, got {len(models)}')
    mounted_model = next(iter(models.values()))
    # Execute Python sources from the immutable upstream revision, rather than
    # trusting arbitrary Python files beside the mounted, hash-pinned weights.
    import huggingface_hub
    from huggingface_hub import snapshot_download
    provenance['huggingface_hub'] = huggingface_hub.__version__
    model_dir = root/'upstream'
    snapshot_download('nvidia/NVIDIA-NemotronLabs-RE-USE', revision=MODEL_REVISION,
                      allow_patterns=['*.py', 'config.json'], local_dir=model_dir)
    if sha(model_dir/'config.json') != CONFIG_HASH:
        raise ValueError('pinned upstream configuration mismatch')
    (model_dir/'model.safetensors').symlink_to(mounted_model/'model.safetensors')
    provenance['model_revision'] = MODEL_REVISION
    provenance['upstream_source_sha256'] = {str(path.relative_to(model_dir)): sha(path)
                                           for path in sorted(model_dir.rglob('*.py'))}
    provenance.update(model_sha256=MODEL_HASH, config_sha256=CONFIG_HASH,
                      numpy=np.__version__, torch=torch.__version__, torch_config=torch.__config__.show(),
                      source_audio_sha256=sha(source/'samples/jfk.wav'), mkl_rt_sha256=sha(lib),
                      python=sys.version, dependencies=subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True))
    save_json(out/'provenance.json', provenance)
    kh.install_build_toolchain()
    os.environ['CCACHE_MAXSIZE'] = '512M'
    run(['ccache', '-M', '512M']) if shutil.which('ccache') else None
    build = root/'build'
    # GPU computation is the original unfused Python model below. This standalone
    # scalar native comparator deliberately builds CPU-only; no nvcc/fat binary.
    run(['cmake', '-S', source/'tools/ci-heavy/issue579_reuse', '-B', build, '-G', 'Ninja',
         '-DCMAKE_BUILD_TYPE=Release', '-DGGML_CUDA=OFF', '-DCRISPASR_REUSE_MKL_FRONTEND=ON',
         f'-DCRISPASR_REUSE_MKL_INCLUDE_DIR={header.parent}', f'-DCRISPASR_REUSE_MKL_LIBRARY={lib}',
         *kh.cache_and_link_flags()])
    with kh.build_heartbeat('reuse.ten.build', interval_s=30):
        run(['cmake', '--build', build, '--target', 'reuse-long-diff', 'reuse-diff-controls',
             'reuse-frontend-mkl-probe', 'reuse-profile-mkl', '-j'+kh.safe_build_jobs(gpu=True)])
    run(['ctest', '--test-dir', build, '-R', '^reuse-(diff-controls|mkl-probe-controls)$', '--no-tests=error', '--output-on-failure'])
    for control in ('reference_dumper', 'reference_retention', 'chunked', 'quality_metrics'):
        run([sys.executable, source/f'tests/test_reuse_{control}.py', '-q'])
    model_gguf = root/'model.gguf'
    run([sys.executable, source/'models/convert-reuse-to-gguf.py', '--input', model_dir, '--output', model_gguf])
    provenance['model_gguf_sha256'] = sha(model_gguf)
    save_json(out/'provenance.json', provenance)
    clean, rate = sf.read(source/'samples/jfk.wav', dtype='float32', always_2d=True)
    if rate != 16000 or clean.shape[1] != 1 or len(clean) < 160000:
        raise ValueError('expected ten seconds of 16 kHz mono JFK speech proxy')
    clean = np.ascontiguousarray(clean[:160000, 0])
    noise = np.random.default_rng(579).standard_normal(len(clean)).astype(np.float32)
    noise *= np.sqrt(np.mean(clean.astype(np.float64)**2)/(np.mean(noise.astype(np.float64)**2)*10**0.5))
    noisy = clean + noise
    sf.write(out/'input-proxy.wav', clean, rate, subtype='FLOAT')
    sf.write(out/'input-noisy.wav', noisy, rate, subtype='FLOAT')
    provenance.update(noise_seed=579, noise_target_snr_db=5, samples=len(clean), sample_rate=rate,
                      input_proxy_raw_f32_sha256=hashlib.sha256(clean.tobytes()).hexdigest(),
                      input_noisy_raw_f32_sha256=hashlib.sha256(noisy.tobytes()).hexdigest(),
                      input_proxy_wav_sha256=sha(out/'input-proxy.wav'),
                      input_noisy_wav_sha256=sha(out/'input-noisy.wav'))
    save_json(out/'provenance.json', provenance)
    windows, native_windows, gpu_windows = [], [], []
    manifest = {'source_revision': revision, 'script_sha256': sha(__file__), 'references': []}
    sys.path.insert(1, str(model_dir))
    from models.generator_SEMamba_time_d4 import SEMamba
    from models.stfts import mag_phase_stft, mag_phase_istft
    def gpu_reference(block):
        # Adapter is imported from the frozen research tools; .cuda() really
        # executes the original model and non-fused Mamba equations on this GPU.
        net = SEMamba(json.loads((model_dir/'config.json').read_text())).cuda().eval()
        net.load_state_dict(load_file(str(model_dir/'model.safetensors')), strict=True)
        mag, phase, _ = mag_phase_stft(torch.from_numpy(block[None]).cuda(), 640, 80, 640, 'relu_log1p', addeps=False)
        amp, phase, _ = net(mag, phase)
        zeros = (torch.expm1(torch.relu(amp)) == 0).sum(1)/amp.shape[1]
        amp[:, :, (zeros > 0.5)[0]] = 0
        waveform = mag_phase_istft(amp, phase, 640, 80, 640, 'relu_log1p').cpu().numpy()[0].copy()
        torch.cuda.synchronize()
        del net, mag, phase, amp, zeros
        torch.cuda.empty_cache()
        gc.collect()
        if waveform.shape != block.shape or not np.isfinite(waveform).all():
            raise ValueError('invalid GPU reference waveform')
        return waveform
    def enhance(block, index, start):
        check_budget(root, out)
        kh.step('reuse.ten.window.start', index=index, start_sample=start)
        with kh.build_heartbeat(f'reuse.ten.window.{index}.gpu', interval_s=30):
            gpu_wave = gpu_reference(block)
        gpu_windows.append(gpu_wave)
        with kh.build_heartbeat(f'reuse.ten.window.{index}.reference', interval_s=30):
            capture = reuse.dump(model_dir, block, set(reuse.DEFAULT_STAGES), max_samples=80000)
        ref = root/f'window-{index}-reference.gguf'
        wave = capture['waveform'].copy()
        tensors = {name: {'shape': list(value.shape), 'dtype': str(value.dtype),
                          'sha256': hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()}
                   for name, value in sorted(capture.items())}
        write_gguf_archive(capture, {'backend': 'reuse'}, ref)
        del capture
        gc.collect()
        artifact = retain_reference(ref, out/ref.name)
        ref.unlink()
        artifact.update(index=index, start_sample=start, samples=len(block), tensors=tensors,
                        raw_audio_sha256=hashlib.sha256(block.tobytes()).hexdigest())
        manifest['references'].append(artifact)
        # Atomic durable checkpoint precedes native comparison. Never delete it.
        save_json(out/'reference-manifest.json', manifest)
        kh.step('reuse.ten.reference.retained', index=index, sha256=artifact['sha256'], bytes=artifact['bytes'])
        native = root/f'window-{index}-native.f32'
        with kh.build_heartbeat(f'reuse.ten.window.{index}.diff', interval_s=30):
            proc = subprocess.run([str(build/'reuse-long-diff'), str(model_gguf), str(out/artifact['name']), str(native)],
                                  text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        (out/f'window-{index}-diff.log').write_text(proc.stdout)
        result = {'index': index, 'start_sample': start, 'exit_code': proc.returncode,
                  'reference_sha256': artifact['sha256']}
        windows.append(result)
        save_json(out/'window-results.json', windows)
        if proc.returncode:
            raise RuntimeError(f'window {index}: native comparison failed; retained reference and log available')
        result.update(parse_rows(proc.stdout))
        native_wave = np.fromfile(native, dtype='<f4')
        if native_wave.shape != block.shape or not np.isfinite(native_wave).all():
            raise ValueError('invalid native waveform')
        native_windows.append(native_wave)
        native.unlink()
        result['gpu_vs_cpu_reference_rel_rms'] = float(np.linalg.norm(gpu_wave-wave)/max(np.linalg.norm(wave), 1e-30))
        save_json(out/'window-results.json', windows)
        kh.step('reuse.ten.window.done', **result)
        return wave
    policy = dict(window_samples=80000, hop_samples=40000, max_window_samples=80000)
    reference_wave, starts = chunked(noisy, enhance, **policy)
    if starts != [0, 40000, 80000] or len(windows) != 3:
        raise ValueError('expected exactly three overlapping five-second windows')
    native_wave, native_starts = chunked(noisy, lambda block, index, start: native_windows[index], **policy)
    gpu_wave, gpu_starts = chunked(noisy, lambda block, index, start: gpu_windows[index], **policy)
    if native_starts != starts or gpu_starts != starts:
        raise ValueError('stitch plans disagree')
    stats = {}
    for name, waveform in (('input', noisy), ('reference-chunked', reference_wave),
                           ('native-chunked', native_wave), ('reference-gpu-unfused-chunked', gpu_wave)):
        stats[name] = quality(clean, waveform)
        stats[name]['stoi'] = float(stoi(clean, waveform, rate, extended=False))
        if not np.isfinite(stats[name]['stoi']):
            raise ValueError('nonfinite STOI')
        if name != 'input':
            sf.write(out/f'output-{name}.wav', waveform, rate, subtype='FLOAT')
    report = {'conclusive': True, 'source_revision': revision, 'script_version': SCRIPT_VERSION,
              'samples': len(clean), 'sample_rate': rate, 'noise_seed': 579, 'noise_target_snr_db': 5,
              'window_samples': 80000, 'hop_samples': 40000, 'starts': starts, 'windows': windows,
              'quality': stats, 'native_vs_reference_rel_rms': float(np.linalg.norm(native_wave-reference_wave)/max(np.linalg.norm(reference_wave), 1e-30)),
              'seams': [{'sample': start, 'native_reference_max_abs_error_10ms': float(np.max(np.abs(native_wave[start-80:start+80]-reference_wave[start-80:start+80])))} for start in starts[1:]],
              'scope': provenance['scope']}
    save_json(out/'ten-second-result.json', report)
    save_json(out/'artifact-manifest.json', {path.name: {'sha256': sha(path), 'bytes': path.stat().st_size}
                                           for path in out.iterdir() if path.is_file() and path.name not in ('artifact-manifest.json', 'progress.jsonl')})
    check_budget(root, out)
    stopped.set()
    kh.step('reuse.ten.done', windows=len(windows), pass_rows=426)
    print('REUSE_TEN_SECOND_RESULT', json.dumps(report, allow_nan=False), flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        failure_path = Path('/kaggle/working/ten-second-failure.json')
        if failure_path.parent.is_dir():
            save_json(failure_path, {'conclusive': False, 'error_type': type(error).__name__,
                                     'error': str(error), 'source_revision': SOURCE_REVISION,
                                     'script_version': SCRIPT_VERSION})
        raise
