"""Prepare-only candidate: exact failing five-second context, original fused CUDA A/B.

No compilation: reuse the source-audited extensions from the accepted fused run.
Separate subprocesses keep the research adapter out of the fused import graph.
"""
import hashlib
import json
import os
import re
import shutil
from pathlib import Path
import subprocess
import sys
import urllib.request

SOURCE_REVISION = '529e073f8c9c99b688be5823e1be78443d50763e'
MODEL_REVISION = '31f82f2efad1095034492f236e8dac9239ba713b'
WHEEL_REVISION = '5d53593a148d2ae5e99667a3f04a04d4fddc2107'
SCRIPT_VERSION = 'reuse-quality-followup-20261112-v1'
ROOT = Path('/kaggle/temp/reuse-quality-followup-20261112')
OUT = Path('/kaggle/working')
ARCHIVE_HASH = '25da6651db2ea3ceb7910daec5ff7968db83aa29b56732f1f0f7114b2763cb02'
WINDOW_HASH = '4143c5cff09bc5b07437d027fcadc6c3b7cf1366f6856a86b4be99b990308469'
STAGES = (['raw_audio', 'frontend_magnitude', 'frontend_phase'] +
          [f'encoder_layer_{i}' for i in range(6)] +
          [name for i in range(30) for name in (f'block_time_{i}', f'block_{i}')] +
          ['amplitude', 'output_phase', 'waveform'])
EXPECTED_WHEELS = {'causal_conv1d-1.5.0.post8-cp313-cp313-linux_x86_64.whl': '96ba07e982904fc13f6ef345d10b4f29d4a5f4350fe8ed1bcf2b2bd81d93eef6', 'mamba_ssm-2.2.4-cp313-cp313-linux_x86_64.whl': '91dd8763ba0139acb544942310d01d5fbf6b27da07b6697c746476496b95a67c'}
DEPENDENCIES = [{'package': 'transformers', 'version': '4.46.3', 'filename': 'transformers-4.46.3-py3-none-any.whl', 'sha256': 'a12ef6f52841fd190a3e5602145b542d03507222f2c64ebb7ee92e8788093aef', 'url': 'https://files.pythonhosted.org/packages/51/51/b87caa939fedf307496e4dbf412f4b909af3d9ca8b189fc3b65c1faa456f/transformers-4.46.3-py3-none-any.whl'}, {'package': 'tokenizers', 'version': '0.20.3', 'filename': 'tokenizers-0.20.3-cp313-cp313-manylinux_2_17_x86_64.manylinux2014_x86_64.whl', 'sha256': '07d7851a72717321022f3774e84aa9d595a041d643fafa2e87fbc9b18711dac0', 'url': 'https://files.pythonhosted.org/packages/66/52/7a171bd4929e3ffe61a29b4340fe5b73484709f92a8162a18946e124c34c/tokenizers-0.20.3-cp313-cp313-manylinux_2_17_x86_64.manylinux2014_x86_64.whl'}, {'package': 'huggingface-hub', 'version': '0.26.2', 'filename': 'huggingface_hub-0.26.2-py3-none-any.whl', 'sha256': '98c2a5a8e786c7b2cb6fdeb2740893cba4d53e312572ed3d8afafda65b128c46', 'url': 'https://files.pythonhosted.org/packages/60/bf/cea0b9720c32fa01b0c4ec4b16b9f4ae34ca106b202ebbae9f03ab98cd8f/huggingface_hub-0.26.2-py3-none-any.whl'}]
MODEL_HASH = '06badaf4029b297f97f7fe48d62990b48d5482608210df577354e0d4e0ed70e1'
CONFIG_HASH = 'e69986483774f01893d2727503addc49ef201d19e6e1a266b8d4e971e28c478d'
PINS = {'mamba': {'repo': 'state-spaces/mamba', 'sha': '95d8aba8a8c75aedcaa6143713b11e745e7cd0d9', 'roots': ['csrc', 'mamba_ssm'], 'files': 49, 'manifest': '6da7433597dcb69d7f4f81d4716c3dbabbe00c3cd5d75c31d25007500b14ca08'}, 'causal': {'repo': 'Dao-AILab/causal-conv1d', 'sha': '82867a9d2e6907cc0f637ac6aff318f696838548', 'roots': ['csrc', 'causal_conv1d'], 'files': 10, 'manifest': 'cc1276bcad2be267885f53111ddeb155d83c878f126e22027cdef96123a41cea'}, 'cccl': {'repo': 'NVIDIA/cccl', 'sha': '207d20fb6a4165fcd2551bcf2f04b8a5ea40829f'}}


def capture(site, arm):
    sys.path.insert(0, str(site))
    if arm not in ('fused', 'adapter'):
        raise ValueError('unknown capture arm')
    import importlib
    for item in DEPENDENCIES:
        module = importlib.import_module(item['package'].replace('-', '_'))
        if module.__version__ != item['version'] or not Path(module.__file__).resolve().is_relative_to(site.resolve()):
            raise RuntimeError('isolated dependency version/path mismatch')
    import numpy as np
    import torch
    source = ROOT / 'source'
    if arm == 'adapter':
        sys.path.insert(0, str(source/'tools/ci-heavy/issue579_reuse'))
        import mamba_ssm
        expected_module = source/'tools/ci-heavy/issue579_reuse/mamba_ssm.py'
        if Path(mamba_ssm.__file__).resolve() != expected_module.resolve():
            raise RuntimeError('adapter import origin mismatch')
    else:
        import mamba_ssm
        import selective_scan_cuda
        import causal_conv1d_cuda
    from safetensors.torch import load_file
    for module in (() if arm == 'adapter' else (mamba_ssm, selective_scan_cuda, causal_conv1d_cuda)):
        if not Path(module.__file__).resolve().is_relative_to(site.resolve()):
            raise RuntimeError('unexpected module or adapter: ' + str(module.__file__))
    if arm == 'fused' and mamba_ssm.__version__ != '2.2.4':
        raise RuntimeError('wrong original Mamba version')
    modeldir = ROOT/'upstream'
    if digest(modeldir/'model.safetensors') != MODEL_HASH or digest(modeldir/'config.json') != CONFIG_HASH:
        raise RuntimeError('pinned model/config mismatch')
    sys.path.insert(1, str(modeldir))
    from models.generator_SEMamba_time_d4 import SEMamba
    from models.stfts import mag_phase_stft, mag_phase_istft
    audio = np.load(ROOT/'input.npy', allow_pickle=False)
    if hashlib.sha256(audio.tobytes()).hexdigest() != WINDOW_HASH:
        raise RuntimeError('input hash mismatch')
    torch.set_grad_enabled(False)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = SEMamba(json.loads((modeldir / 'config.json').read_text()))
    model.load_state_dict(load_file(str(modeldir / 'model.safetensors')), strict=True)
    model = model.cuda().eval()
    blocks = [module for module in model.modules() if isinstance(module, mamba_ssm.Mamba)]
    if len(blocks) != 120 or (arm == 'fused' and not all(module.use_fast_path for module in blocks)):
        raise RuntimeError('the original model fast path was disabled or replaced')
    calls = {'selective_scan_cuda.fwd': 0, 'causal_conv1d_cuda.causal_conv1d_fwd': 0}
    def count(module, name, label):
        original = getattr(module, name)
        def counted(*args, **kwargs):
            calls[label] += 1
            return original(*args, **kwargs)
        setattr(module, name, counted)
    if arm == 'fused':
        count(selective_scan_cuda, 'fwd', 'selective_scan_cuda.fwd')
        count(causal_conv1d_cuda, 'causal_conv1d_fwd', 'causal_conv1d_cuda.causal_conv1d_fwd')
    stages = {'raw_audio': audio}
    def hook(name):
        def save(_module, _args, value):
            stages[name] = value[0].detach().float().cpu().contiguous().numpy().copy()
        return save
    def time_hook(index):
        def save(_module, args):
            stages[f'block_time_{index}'] = args[0].permute(2, 0, 1).detach().float().cpu().contiguous().numpy().copy()
        return save
    layers = [model.dense_encoder.dense_conv_1, *model.dense_encoder.dense_block.dense_block,
              model.dense_encoder.dense_conv_2]
    for index, layer in enumerate(layers):
        layer.register_forward_hook(hook(f'encoder_layer_{index}'))
    for index, block in enumerate(model.TSMamba):
        block.freq_mamba.register_forward_pre_hook(time_hook(index))
        block.register_forward_hook(hook(f'block_{index}'))
    x = torch.from_numpy(audio[None]).cuda()
    magnitude, phase, _ = mag_phase_stft(x, 640, 80, 640, 'relu_log1p', addeps=False)
    stages['frontend_magnitude'] = magnitude[0].T.float().cpu().contiguous().numpy().copy()
    stages['frontend_phase'] = phase[0].T.float().cpu().contiguous().numpy().copy()
    amplitude, output_phase, _ = model(magnitude, phase)
    stages['amplitude'] = amplitude[0].T.float().cpu().contiguous().numpy().copy()
    stages['output_phase'] = output_phase[0].T.float().cpu().contiguous().numpy().copy()
    zeros = (torch.expm1(torch.relu(amplitude)) == 0).sum(1) / amplitude.shape[1]
    amplitude[:, :, (zeros > .5)[0]] = 0
    waveform = mag_phase_istft(amplitude, output_phase, 640, 80, 640, 'relu_log1p')
    stages['waveform'] = waveform[0].float().cpu().numpy().copy()
    torch.cuda.synchronize()
    if len(stages) != 72 or any(not np.isfinite(value).all() for value in stages.values()):
        raise RuntimeError('missing/nonfinite original stages')
    if (arm == 'fused' and any(value != 120 for value in calls.values())) or stages['waveform'].shape != audio.shape:
        raise RuntimeError('original fused extensions did not execute the expected 120 calls')
    if float(np.linalg.norm(stages['waveform'])) <= 1e-6:
        raise RuntimeError('original fused output is silent')
    np.savez_compressed(OUT / f'{arm}-stages.npz', **stages)
    report = {'script_version': SCRIPT_VERSION, 'torch': str(torch.__version__), 'cuda': torch.version.cuda,
              'device': torch.cuda.get_device_name(), 'mamba': getattr(mamba_ssm, '__version__', 'audited-adapter'), 'fused_kernel': arm == 'fused', 'arm': arm, 'mamba_module': str(mamba_ssm.__file__),
              'extension_calls': calls, 'input_sha256': WINDOW_HASH, 'model_sha256': MODEL_HASH,
              'stage_shapes': {name: list(value.shape) for name, value in stages.items()},
              'stage_metadata': {name: {'shape': list(value.shape), 'dtype': str(value.dtype),
                  'sha256': hashlib.sha256(value.tobytes()).hexdigest()} for name, value in stages.items()},
              'stages_sha256': digest(OUT / f'{arm}-stages.npz'),
              'note': 'Original source capture. Native parity acceptance remains a separate unchanged gate.'}
    write_json(f'{arm}-capture-result.json', report)
    print('REUSE_ORIGINAL_FUSED_CAPTURE', json.dumps(report), flush=True)


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            value.update(block)
    return value.hexdigest()


def write_json(name, value):
    target = OUT/name
    temporary = target.with_suffix(target.suffix+'.partial')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(target)


def run(args, **kwargs):
    return subprocess.run(list(map(str, args)), check=True, **kwargs)


def exact_inputs(source):
    """Noise normalization happens on all ten seconds BEFORE selecting context."""
    import numpy as np
    import soundfile as sf
    clean, rate = sf.read(source/'samples/jfk.wav', dtype='float32', always_2d=True)
    if rate != 16000 or clean.shape[1] != 1 or len(clean) < 160000:
        raise ValueError('expected original ten-second 16kHz mono JFK')
    clean = np.ascontiguousarray(clean[:160000, 0])
    noise = np.random.default_rng(579).standard_normal(len(clean)).astype(np.float32)
    noise *= np.sqrt(np.mean(clean.astype(np.float64)**2)/
                     (np.mean(noise.astype(np.float64)**2)*10**0.5))
    noisy = clean+noise
    block = np.ascontiguousarray(noisy[80000:160000])
    if hashlib.sha256(block.tobytes()).hexdigest() != WINDOW_HASH:
        raise ValueError('exact final-window reconstruction mismatch')
    return clean, noisy, block


def compare_arrays(reference, actual):
    import numpy as np
    a = np.asarray(reference, dtype=np.float64)
    b = np.asarray(actual, dtype=np.float64)
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('stage shape or finite-value mismatch')
    norm_a, norm_b = np.linalg.norm(a), np.linalg.norm(b)
    return {'shape': list(a.shape), 'reference_norm': float(norm_a), 'actual_norm': float(norm_b),
            'rel_rms': float(np.linalg.norm(b-a)/max(norm_a, 1e-30)),
            'cosine': float(np.sum(a*b)/max(norm_a*norm_b, 1e-30)),
            'max_abs': float(np.max(np.abs(b-a)))}


def compare_results(archive, clean, noisy):
    import numpy as np
    import soundfile as sf
    import gguf
    sys.path.insert(0, str(ROOT/'source/tools/ci-heavy/issue579_reuse'))
    from quality_metrics import quality
    reader = gguf.GGUFReader(str(archive))
    cpu = {tensor.name: tensor.data for tensor in reader.tensors}
    # No model reload or scalar native replay: the original CPU archive already
    # passed 142 native stage rows. This job diagnoses the reference itself.
    with np.load(OUT/'fused-stages.npz', allow_pickle=False) as fused, np.load(OUT/'adapter-stages.npz', allow_pickle=False) as adapter:
        if set(fused.files) != set(STAGES) or set(adapter.files) != set(STAGES) or not set(STAGES) <= cpu.keys():
            raise ValueError('missing/extra capture stages')
        # Same-device frontend identity gates structural alignment. Model-stage
        # errors are observations: no new tolerance is invented by this job.
        for name in ('raw_audio', 'frontend_magnitude', 'frontend_phase'):
            if not np.array_equal(fused[name], adapter[name]):
                raise ValueError('same-GPU input/frontend structural gate failed: '+name)
        if not np.array_equal(cpu['raw_audio'].reshape(-1), noisy[80000:160000]):
            raise ValueError('retained CPU archive belongs to a different input')
        rows = {}
        for name in STAGES:
            rows[name] = {'fused_vs_gpu_adapter': compare_arrays(adapter[name], fused[name]),
                          'fused_vs_cpu_adapter': compare_arrays(cpu[name], fused[name]),
                          'gpu_adapter_vs_cpu_adapter': compare_arrays(cpu[name], adapter[name])}
        write_json('stage-comparisons.json', rows)
        stats = {'input': quality(clean[80000:160000], noisy[80000:160000])}
        for arm, wave in [('fused', fused['waveform']), ('adapter', adapter['waveform']), ('cpu-adapter', cpu['waveform'].reshape(-1))]:
            stats[arm] = {'full_5s': quality(clean[80000:160000], wave),
                          'troubled_7_to_8s_in_context': quality(clean[112000:128000], wave[32000:48000])}
            sf.write(OUT/f'output-{arm}-5s.wav', wave, 16000, subtype='FLOAT')
            sf.write(OUT/f'output-{arm}-7-to-8s.wav', wave[32000:48000], 16000, subtype='FLOAT')
        report = {'conclusive': True, 'script_version': SCRIPT_VERSION, 'quality': stats,
                  'scope': 'Exact final five-second model context; 7–8s is cropped AFTER inference. No stitching re-execution, product support, perceptual quality or native CUDA claim.',
                  'comparisons': 'stage-comparisons.json', 'archive_sha256': ARCHIVE_HASH,
                  'native_evidence': 'Existing window-2-diff.log, 71 unconditioned plus 71 conditioned rows; no new native execution.'}
        write_json('quality-followup-result.json', report)


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    os.environ.update(OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1',
                      USE_TF='0', USE_FLAX='0', PYTHONNOUSERSITE='1',
                      PYTHONDONTWRITEBYTECODE='1', TMPDIR=str(ROOT), HF_HOME=str(ROOT/'hf-cache'))
    import torch
    gpu = {'torch': str(torch.__version__), 'cuda': torch.version.cuda,
           'python': sys.version, 'cxx11_abi': bool(torch._C._GLIBCXX_USE_CXX11_ABI),
           'available': torch.cuda.is_available()}
    if gpu['available']:
        gpu.update(name=torch.cuda.get_device_name(), capability=list(torch.cuda.get_device_capability()))
    write_json('gpu-preflight.json', gpu)
    if (gpu['torch'] != '2.11.0+cu130' or gpu['cuda'] != '13.0' or not gpu['cxx11_abi'] or
            sys.version_info[:2] != (3, 13) or gpu.get('capability') != [7, 5]):
        write_json('quality-followup-result.json', {'conclusive': False, 'reason': 'pinned wheels require original CPython3.13/Torch2.11/CUDA13/CXX11 ABI and sm75', 'gpu': gpu})
        return  # No rebuild, Torch replacement or automatic GPU re-draw.
    torch.ones(1, device='cuda').sum().item()
    source = ROOT/'source'
    run(['git', 'init', source])
    run(['git', '-C', source, 'remote', 'add', 'origin', 'https://github.com/CrispStrobe/CrispASR.git'])
    run(['git', '-C', source, 'fetch', '--depth', '1', 'origin', SOURCE_REVISION])
    run(['git', '-C', source, 'checkout', '--detach', 'FETCH_HEAD'])
    if subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip() != SOURCE_REVISION:
        raise ValueError('source revision mismatch')
    sys.path.insert(0, str(source/'tools/kaggle'))
    import kaggle_harness as kh
    kh.init_progress(progress_path=OUT/'progress.jsonl')
    kh.step('reuse.quality.start', source=SOURCE_REVISION, script_version=SCRIPT_VERSION)
    token = kh.resolve_hf_token(require=True)
    from huggingface_hub import hf_hub_download, snapshot_download
    matches = [p for p in Path('/kaggle/input').rglob('window-2-reference.gguf') if p.is_file()]
    unique = {str(p.resolve()): p for p in matches}
    if len(unique) != 1:
        raise ValueError('expected one retained window-2 archive via kernel_sources')
    archive = next(iter(unique.values()))
    if digest(archive) != ARCHIVE_HASH:
        raise ValueError('retained original CPU archive checksum mismatch')
    native_log = archive.parent/'window-2-diff.log'
    if not native_log.is_file():
        raise ValueError('retained native stage log is missing')
    observed = {'unconditioned': {}, 'conditioned': {}}
    for line in native_log.read_text().splitlines():
        match = re.match(r'^mkl-avx2(-conditioned)?/(\S+)\s+(PASS|FAIL)\s+rel_rms=', line)
        if match:
            arm = 'conditioned' if match[1] else 'unconditioned'
            if match[2] in observed[arm]:
                raise ValueError('duplicate original native stage')
            observed[arm][match[2]] = match[3]
    for rows in observed.values():
        if set(rows) != set(STAGES)-{'raw_audio'} or set(rows.values()) != {'PASS'}:
            raise ValueError('original native log lacks all 71 PASS stages per arm')
    shutil.copyfile(native_log, OUT/'original-window-2-diff.log')
    write_json('original-native-evidence.json', {'archive_sha256': ARCHIVE_HASH,
               'log_sha256': digest(native_log), 'rows': observed, 'reexecuted': False})
    # Model source always comes from the exact original upstream revision.
    modeldir = ROOT/'upstream'
    snapshot_download('nvidia/NVIDIA-NemotronLabs-RE-USE', revision=MODEL_REVISION,
                      allow_patterns=['*.py', 'config.json'], local_dir=modeldir, token=token)
    candidates = {str(p.resolve()): p for p in Path('/kaggle/input').rglob('model.safetensors')}
    valid = [p for p in candidates.values() if digest(p) == MODEL_HASH]
    if len(valid) != 1:
        raise ValueError('expected one hash-pinned checkpoint')
    (modeldir/'model.safetensors').symlink_to(valid[0])
    if digest(modeldir/'config.json') != CONFIG_HASH:
        raise ValueError('upstream configuration mismatch')
    # Small released packages only; preserve preinstalled Torch and NumPy.
    run([sys.executable, '-m', 'pip', 'install', '--no-deps', '--only-binary=:all:',
         'safetensors==0.5.3', 'einops==0.8.1', 'gguf==0.17.1', 'soundfile==0.13.1'])
    import numpy as np
    if np.__version__ != '2.1.3':
        raise ValueError('exact producer noise construction requires NumPy2.1.3')
    clean, noisy, block = exact_inputs(source)
    import numpy as np
    np.save(ROOT/'input.npy', block, allow_pickle=False)
    dependencies = ROOT/'wheels'
    dependencies.mkdir()
    wheels = []
    for filename, expected in EXPECTED_WHEELS.items():
        path = Path(hf_hub_download('cstr/crispasr-ccache', repo_type='dataset', revision=WHEEL_REVISION,
                                  filename='reuse-fused-20261107/'+filename, local_dir=dependencies, token=token))
        if digest(path) != expected:
            raise ValueError('fused wheel digest mismatch')
        wheels.append(path)
    proof = {}
    for filename in ['mamba-wheel-checkpoint.json', 'source-provenance.json']:
        path = Path(hf_hub_download('cstr/crispasr-ccache', repo_type='dataset', revision=WHEEL_REVISION,
                                  filename='reuse-fused-20261107/'+filename, local_dir=dependencies, token=token))
        proof[filename] = json.loads(path.read_text())
    checkpoint = proof['mamba-wheel-checkpoint.json']
    if checkpoint.get('wheels') != EXPECTED_WHEELS or checkpoint.get('package_kernel_source_unchanged') is not True:
        raise ValueError('original successful build checkpoint mismatch')
    provenance = proof['source-provenance.json']
    if not all(provenance.get(k, {}).get('commit') == PINS[k]['sha'] and
               provenance.get(k, {}).get('source_manifest_sha256') == PINS[k]['manifest'] for k in ('mamba','causal')):
        raise ValueError('original extension source manifest mismatch')
    write_json('reused-wheel-provenance.json', proof)
    for item in DEPENDENCIES:
        path = dependencies/item['filename']
        with urllib.request.urlopen(item['url'], timeout=120) as stream:
            path.write_bytes(stream.read())
        if digest(path) != item['sha256']:
            raise ValueError('dependency wheel checksum mismatch')
        wheels.append(path)
    site = ROOT/'site-packages'
    run([sys.executable, '-m', 'pip', 'install', '--no-deps', '--no-index', '--no-compile', '--target', site, *wheels], timeout=180)
    if (site/'torch').exists() or list(site.glob('torch-*.dist-info')):
        raise ValueError('unexpected Torch install')
    write_json('provenance.json', {'source_revision': SOURCE_REVISION, 'script_version': SCRIPT_VERSION,
               'script_sha256': digest(__file__), 'model_revision': MODEL_REVISION,
               'model_sha256': MODEL_HASH, 'config_sha256': CONFIG_HASH, 'archive_sha256': ARCHIVE_HASH,
               'window_raw_sha256': WINDOW_HASH, 'global_samples': 160000, 'start_sample': 80000,
               'samples': 80000, 'noise_seed': 579, 'noise_snr_db': 5,
               'source_audio_sha256': digest(source/'samples/jfk.wav'),
               'upstream_sources': {str(p.relative_to(modeldir)): digest(p) for p in sorted(modeldir.rglob('*.py'))},
               'adapter_sha256': digest(source/'tools/ci-heavy/issue579_reuse/mamba_ssm.py'),
               'dependency_wheels': DEPENDENCIES, 'extension_wheels': EXPECTED_WHEELS, 'gpu': gpu,
               'dependencies': subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True)})
    # No native build: existing archive and its 142 native rows are preserved.
    # Heartbeats still cover both GPU captures so slow stages remain observable.
    for arm in ('fused', 'adapter'):
        with kh.build_heartbeat(f'reuse.quality.{arm}', interval_s=30):
            run([sys.executable, __file__, '--capture', site, arm], timeout=1800,
                env={**os.environ, 'PYTHONPATH': str(site)})
        kh.step('reuse.quality.capture.retained', arm=arm, sha256=digest(OUT/f'{arm}-stages.npz'))
    compare_results(archive, clean, noisy)
    write_json('artifact-manifest.json', {p.name: {'sha256': digest(p), 'bytes': p.stat().st_size}
               for p in OUT.iterdir() if p.is_file() and p.name not in ('artifact-manifest.json','progress.jsonl')})
    kh.step('reuse.quality.done')


if __name__ == '__main__':
    try:
        if len(sys.argv) == 4 and sys.argv[1] == '--capture':
            capture(Path(sys.argv[2]), sys.argv[3])
        else:
            main()
    except Exception as error:
        if OUT.is_dir():
            write_json('quality-followup-failure.json', {'conclusive': False, 'error': str(error), 'type': type(error).__name__})
        raise
