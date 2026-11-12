"""Compare every RE-USE stage on one explicit five-second original-model input."""
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

os.environ.update(OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1',
                  MKL_ENABLE_INSTRUCTIONS='AVX2', CRISPASR_REUSE_FRONTEND='mkl-avx2')
os.environ.pop('MKL_CBWR', None)
root = Path('/kaggle/temp/reuse-full-window-20261112')
out = Path('/kaggle/working')
root.mkdir(parents=True, exist_ok=True)
source = root / 'source'


def run(args, **kwargs):
    print('+', ' '.join(map(str, args)), flush=True)
    return subprocess.run([str(x) for x in args], check=True, **kwargs)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


run(['git', 'clone', '--branch', 'work/reuse-long-audio-20261112', '--recurse-submodules',
     'https://github.com/CrispStrobe/CrispASR.git', source])
revision = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
sys.path.insert(0, str(source/'tools/kaggle'))
import kaggle_harness as kh
kh.init_progress(progress_path=out/'progress.jsonl')
token = kh.resolve_hf_token()
if token:
    os.environ['HF_TOKEN'] = token
kh.step('reuse.full.start', source=revision)
run([sys.executable, '-m', 'pip', 'install', '--quiet', 'safetensors', 'einops', 'gguf', 'soundfile', 'scipy'])
sdk = root/'sdk'
run([sys.executable, '-m', 'pip', 'install', '--quiet', '--prefix', sdk, 'mkl==2025.0.1', 'mkl-include==2025.0.1'])
header = next(sdk.rglob('mkl.h'))
lib = next(sdk.rglob('libmkl_rt.so.2'))
os.environ['LD_LIBRARY_PATH'] = ':'.join([str(lib.parent), os.environ.get('LD_LIBRARY_PATH', '')])
sys.path[:0] = [str(source/'tools'), str(source/'tools/ci-heavy/issue579_reuse')]
import numpy as np
import soundfile as sf
import torch
from safetensors.torch import load_file
from dump_reference import write_gguf_archive
from reference_backends import reuse

torch.set_num_threads(2)
torch.set_grad_enabled(False)
models = {}
for directory, _, files in os.walk('/kaggle/input', followlinks=True):
    if {'config.json', 'model.safetensors'} <= set(files):
        p = Path(directory); models[str(p.resolve())] = p
if len(models) != 1:
    raise RuntimeError(f'expected one RE-USE model, found {len(models)}')
model = next(iter(models.values()))
assert sha(model/'model.safetensors') == '06badaf4029b297f97f7fe48d62990b48d5482608210df577354e0d4e0ed70e1'
assert sha(model/'config.json') == 'e69986483774f01893d2727503addc49ef201d19e6e1a266b8d4e971e28c478d'
model_gguf = root/'model.gguf'
run([sys.executable, source/'models/convert-reuse-to-gguf.py', '--input', model, '--output', model_gguf])
build = root/'build'
run(['cmake', '-S', source/'tools/ci-heavy/issue579_reuse', '-B', build, '-G', 'Ninja',
     '-DCMAKE_BUILD_TYPE=Release', '-DCRISPASR_REUSE_MKL_FRONTEND=ON',
     f'-DCRISPASR_REUSE_MKL_INCLUDE_DIR={header.parent}', f'-DCRISPASR_REUSE_MKL_LIBRARY={lib}'])
with kh.build_heartbeat('reuse.full.build', interval_s=30):
    run(['cmake', '--build', build, '--target', 'reuse-long-diff', 'reuse-diff-controls', '-j2'])
run(['ctest', '--test-dir', build, '-R', '^reuse-diff-controls$', '--output-on-failure'])
run([sys.executable, source/'tests/test_reuse_reference_dumper.py', '-q'])

# Identical seed and input construction to the independent chunk experiment.
clean, rate = sf.read(source/'samples/jfk.wav', dtype='float32', always_2d=True)
if rate != 16000 or clean.shape[1] != 1 or len(clean) < 80000:
    raise ValueError('expected five seconds of 16 kHz mono speech-proxy audio')
clean = np.ascontiguousarray(clean[:80000, 0])
noise = np.random.default_rng(579).standard_normal(len(clean)).astype(np.float32)
noise *= np.sqrt(np.mean(clean.astype(np.float64)**2)/(np.mean(noise.astype(np.float64)**2)*10**0.5))
noisy = clean + noise
sf.write(out/'input-proxy.wav', clean, 16000, subtype='FLOAT')
sf.write(out/'input-noisy.wav', noisy, 16000, subtype='FLOAT')
# Independent CUDA execution of the audited non-fused Python adapter. This
# exercises a real GPU and records cross-device reference differences; it is
# not a fused-kernel or native GPU-backend parity claim.
if not torch.cuda.is_available():
    raise RuntimeError('CUDA device required for the independent GPU reference')
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
sys.path.insert(1, str(model))
from models.generator_SEMamba_time_d4 import SEMamba
from models.stfts import mag_phase_stft, mag_phase_istft
gpu_name = torch.cuda.get_device_name(0)
gpu_model = SEMamba(json.loads((model/'config.json').read_text())).cuda().eval()
gpu_model.load_state_dict(load_file(str(model/'model.safetensors')), strict=True)
gpu_mag, gpu_phase_tensor, _ = mag_phase_stft(torch.from_numpy(noisy[None]).cuda(), 640, 80, 640, 'relu_log1p', addeps=False)
gpu_phase = gpu_phase_tensor[0].T.contiguous().cpu().numpy().copy()
kh.step('reuse.full.gpu_reference.start', samples=len(noisy), gpu=gpu_name)
with kh.build_heartbeat('reuse.full.gpu_reference', interval_s=30):
    gpu_amp, gpu_phase_tensor, _ = gpu_model(gpu_mag, gpu_phase_tensor)
    gpu_zeros = (torch.expm1(torch.relu(gpu_amp)) == 0).sum(1) / gpu_amp.shape[1]
    gpu_amp[:, :, (gpu_zeros > 0.5)[0]] = 0
    gpu_wave = mag_phase_istft(gpu_amp, gpu_phase_tensor, 640, 80, 640, 'relu_log1p').cpu().numpy()[0].copy()
    torch.cuda.synchronize()
if gpu_wave.shape != noisy.shape or not np.isfinite(gpu_wave).all():
    raise ValueError('GPU reference output is non-finite or has the wrong length')
sf.write(out/'output-reference-gpu-unfused.wav', gpu_wave, 16000, subtype='FLOAT')
del gpu_model, gpu_mag, gpu_phase_tensor, gpu_amp, gpu_zeros
torch.cuda.empty_cache()
kh.step('reuse.full.gpu_reference.done')
kh.step('reuse.full.reference.start', samples=len(noisy))
with kh.build_heartbeat('reuse.full.reference', interval_s=30):
    captures = reuse.dump(model, noisy, set(reuse.DEFAULT_STAGES), max_samples=80000)
reference_wave = captures['waveform'].copy()
cpu_phase = captures['frontend_phase'].copy()
reference = root/'reference.gguf'
write_gguf_archive(captures, {'backend':'reuse'}, reference)
del captures
gc.collect()
sf.write(out/'output-reference-whole.wav', reference_wave, 16000, subtype='FLOAT')
kh.step('reuse.full.reference.done', reference_sha256=sha(reference))
native = root/'native.f32'
with kh.build_heartbeat('reuse.full.diff', interval_s=30):
    proc = subprocess.run([str(build/'reuse-long-diff'), str(model_gguf), str(reference), str(native)],
                          text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
(out/'full-diff.log').write_text(proc.stdout)
lines = proc.stdout.splitlines()
passed = sum(' PASS ' in line for line in lines)
failed = sum(' FAIL ' in line for line in lines)
report = {'source_commit':revision, 'sample_rate':rate, 'samples':len(noisy),
          'source_audio_sha256':sha(source/'samples/jfk.wav'), 'model_sha256':sha(model/'model.safetensors'),
          'noise_seed':579, 'reference_sha256':sha(reference), 'pass_rows':passed, 'fail_rows':failed,
          'exit_code':proc.returncode, 'gpu_name':gpu_name,
          'gpu_reference_vs_cpu_wave_rel_rms':float(np.linalg.norm(gpu_wave-reference_wave)/max(np.linalg.norm(reference_wave),1e-30)),
          'gpu_reference_vs_cpu_phase_rel_rms':float(np.linalg.norm(gpu_phase-cpu_phase)/max(np.linalg.norm(cpu_phase),1e-30)),
          'scope':'Five-second full-window diagnostic, scalar native model and AVX2 oneMKL frontend; CUDA comparison uses the original model with the audited non-fused adapter. Not fused GPU parity, streaming or a product backend.'}
if native.is_file():
    waveform = np.fromfile(native, dtype='<f4')
    if waveform.shape != noisy.shape or not np.isfinite(waveform).all():
        raise ValueError('native full-window waveform invalid')
    sf.write(out/'output-native-whole.wav', waveform, 16000, subtype='FLOAT')
    report['native_vs_reference_rel_rms'] = float(np.linalg.norm(waveform-reference_wave)/max(np.linalg.norm(reference_wave),1e-30))
(out/'full-window-result.json').write_text(json.dumps(report, indent=2)+'\n')
print('REUSE_FULL_WINDOW_RESULT', json.dumps(report), flush=True)
kh.step('reuse.full.done', pass_rows=passed, fail_rows=failed, exit_code=proc.returncode)
if proc.returncode or passed != 142 or failed:
    raise SystemExit('five-second intermediate gates did not all pass')
