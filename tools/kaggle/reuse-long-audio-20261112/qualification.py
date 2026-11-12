"""Five-second RE-USE diagnostic: per-window stages and synthetic-noise quality."""
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
root = Path('/kaggle/temp/reuse-long-audio-20261112')
out = Path('/kaggle/working')
root.mkdir(parents=True, exist_ok=True)
source = root / 'source'
branch = 'work/reuse-long-audio-20261112'


def run(args, **kwargs):
    print('+', ' '.join(map(str, args)), flush=True)
    return subprocess.run([str(x) for x in args], check=True, **kwargs)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


run(['git', 'clone', '--branch', branch, '--recurse-submodules',
     'https://github.com/CrispStrobe/CrispASR.git', source])
sys.path.insert(0, str(source/'tools/kaggle'))
import kaggle_harness as kh
kh.init_progress(progress_path=out/'progress.jsonl')
token = kh.resolve_hf_token()
if token:
    os.environ['HF_TOKEN'] = token
kh.step('reuse.long.start')
revision = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
run([sys.executable, '-m', 'pip', 'install', '--quiet', 'safetensors', 'einops', 'gguf', 'soundfile', 'scipy'])
sdk = root / 'sdk'
run([sys.executable, '-m', 'pip', 'install', '--quiet', '--prefix', sdk, 'mkl==2025.0.1', 'mkl-include==2025.0.1'])
header = next(sdk.rglob('mkl.h'))
lib = next(sdk.rglob('libmkl_rt.so.2'))
os.environ['LD_LIBRARY_PATH'] = ':'.join([str(lib.parent), os.environ.get('LD_LIBRARY_PATH', '')])
sys.path[:0] = [str(source / 'tools'), str(source / 'tools/ci-heavy/issue579_reuse')]
import numpy as np
import soundfile as sf
import torch
from safetensors.torch import load_file
from dump_reference import write_gguf_archive
from reference_backends import reuse
from chunked import chunked

torch.set_num_threads(2)
torch.set_grad_enabled(False)
models = {}
for directory, _, files in os.walk('/kaggle/input', followlinks=True):
    if {'config.json', 'model.safetensors'} <= set(files):
        p = Path(directory); models[str(p.resolve())] = p
if len(models) != 1:
    raise RuntimeError(f'expected one RE-USE model, found {len(models)}')
model_dir = next(iter(models.values()))
assert sha(model_dir / 'model.safetensors') == '06badaf4029b297f97f7fe48d62990b48d5482608210df577354e0d4e0ed70e1'
assert sha(model_dir / 'config.json') == 'e69986483774f01893d2727503addc49ef201d19e6e1a266b8d4e971e28c478d'
model_gguf = root / 'model.gguf'
run([sys.executable, source/'models/convert-reuse-to-gguf.py', '--input', model_dir, '--output', model_gguf])
build = root / 'build'
run(['cmake', '-S', source/'tools/ci-heavy/issue579_reuse', '-B', build, '-G', 'Ninja',
     '-DCMAKE_BUILD_TYPE=Release', '-DCRISPASR_REUSE_MKL_FRONTEND=ON',
     f'-DCRISPASR_REUSE_MKL_INCLUDE_DIR={header.parent}', f'-DCRISPASR_REUSE_MKL_LIBRARY={lib}'])
with kh.build_heartbeat('reuse.long.build', interval_s=30):
    run(['cmake', '--build', build, '--target', 'reuse-research-diff', 'reuse-diff-controls', '-j2'])
run(['ctest', '--test-dir', build, '-R', '^reuse-diff-controls$', '--output-on-failure'])
run([sys.executable, source/'tests/test_reuse_reference_dumper.py', '-q'])
run([sys.executable, source/'tests/test_reuse_chunked.py', '-q'])

# The repository sample is a clean-speech proxy, not a certified clean recording.
# Add deterministic synthetic noise at 5 dB to measure a bounded sanity case.
clean, sr = sf.read(source/'samples/jfk.wav', dtype='float32', always_2d=True)
if sr != 16000 or clean.shape[1] != 1 or len(clean) < 80000:
    raise ValueError('expected at least five seconds of 16 kHz mono JFK audio')
clean = np.ascontiguousarray(clean[:80000, 0])
noise = np.random.default_rng(579).standard_normal(len(clean)).astype(np.float32)
noise *= np.sqrt(np.mean(clean.astype(np.float64)**2) / (np.mean(noise.astype(np.float64)**2) * 10**0.5))
noisy = clean + noise
sf.write(out/'input-noisy.wav', noisy, 16000, subtype='FLOAT')
sf.write(out/'input-proxy.wav', clean, 16000, subtype='FLOAT')
results = []
native_windows = []


def enhance(block, index, start):
    print(f'WINDOW_START {index} offset={start}', flush=True)
    kh.step('reuse.long.window.start', index=index, start=start)
    with kh.build_heartbeat(f'reuse.long.window.{index}.reference', interval_s=30):
        capture = reuse.dump(model_dir, block, set(reuse.DEFAULT_STAGES))
    ref = root / f'window-{index}.gguf'
    write_gguf_archive(capture, {'backend':'reuse'}, ref)
    native_path = root / f'window-{index}-native.f32'
    with kh.build_heartbeat(f'reuse.long.window.{index}.diff', interval_s=30):
        proc = subprocess.run([str(build/'reuse-research-diff'), str(model_gguf), str(ref), str(native_path)],
                              text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (out/f'window-{index}-diff.log').write_text(proc.stdout)
    lines = proc.stdout.splitlines()
    passed = sum(' PASS ' in line for line in lines)
    failed = sum(' FAIL ' in line for line in lines)
    results.append({'index':index,'start_sample':start,'pass_rows':passed,'fail_rows':failed,'exit_code':proc.returncode})
    print('WINDOW_RESULT', json.dumps(results[-1]), flush=True)
    kh.step('reuse.long.window.done', **results[-1])
    native_windows.append(np.fromfile(native_path, dtype='<f4') if native_path.is_file() else None)
    if native_path.is_file(): native_path.unlink()
    wave = capture['waveform'].copy()
    del capture
    ref.unlink()
    gc.collect()
    return wave


chunked_wave, starts = chunked(noisy, enhance)
sf.write(out/'output-reference-chunked.wav', chunked_wave, 16000, subtype='FLOAT')
native_wave = None
if all(w is not None and len(w) == 16000 for w in native_windows):
    native_wave, native_starts = chunked(noisy, lambda block, i, start: native_windows[i])
    assert native_starts == starts
    sf.write(out/'output-native-chunked.wav', native_wave, 16000, subtype='FLOAT')

# Run the audited Python reference model on the entire five-second waveform.
# This bypasses the bounded dump interface only for the waveform-only baseline.
sys.path.insert(1, str(model_dir))
from models.generator_SEMamba_time_d4 import SEMamba
from models.stfts import mag_phase_stft, mag_phase_istft
model = SEMamba(json.loads((model_dir/'config.json').read_text())).eval()
model.load_state_dict(load_file(str(model_dir/'model.safetensors')), strict=True)
mag, phase, _ = mag_phase_stft(torch.from_numpy(noisy[None]), 640, 80, 640, 'relu_log1p', addeps=False)
with kh.build_heartbeat('reuse.long.whole.reference', interval_s=30):
    amp, phase, _ = model(mag, phase)
zeros = (torch.expm1(torch.relu(amp)) == 0).sum(1) / amp.shape[1]
amp[:, :, (zeros > 0.5)[0]] = 0
whole = mag_phase_istft(amp, phase, 640, 80, 640, 'relu_log1p').numpy()[0]
if whole.shape != clean.shape or not np.isfinite(whole).all():
    raise ValueError('full reference output is non-finite or has the wrong length')
sf.write(out/'output-reference-whole.wav', whole, 16000, subtype='FLOAT')


def snr(reference, estimate):
    return float(10*np.log10(np.sum(reference.astype(np.float64)**2) /
                             max(np.sum((reference.astype(np.float64)-estimate)**2), 1e-30)))


seams = [s for s in starts[1:] if s < len(clean)]
report = {'source_commit':revision, 'sample_rate':sr, 'samples':len(clean), 'input_sha256':sha(source/'samples/jfk.wav'),
          'model_sha256':sha(model_dir/'model.safetensors'), 'noise_seed':579, 'input_snr_db':snr(clean,noisy),
          'whole_reference_snr_db':snr(clean,whole), 'chunked_reference_snr_db':snr(clean,chunked_wave),
          'native_chunked_snr_db':snr(clean,native_wave) if native_wave is not None else None,
          'native_vs_reference_chunk_rel_rms':float(np.linalg.norm(native_wave-chunked_wave)/max(np.linalg.norm(chunked_wave),1e-30)) if native_wave is not None else None,
          'whole_vs_chunk_rel_rms':float(np.linalg.norm(whole-chunked_wave)/max(np.linalg.norm(whole),1e-30)),
          'seam_max_abs_error_vs_whole':float(max(np.max(np.abs(chunked_wave[max(0,s-80):s+80]-whole[max(0,s-80):s+80])) for s in seams)),
          'windows':results, 'scope':'One 5 s speech-proxy recording with deterministic synthetic noise; reference and stage-gated native quality. Not a production chunking policy or a broad quality benchmark.'}
(out/'long-audio-result.json').write_text(json.dumps(report, indent=2)+'\n')
print('REUSE_LONG_AUDIO_RESULT', json.dumps(report), flush=True)
kh.step('reuse.long.done', windows=len(results))
if native_wave is None or any(r['exit_code'] or r['pass_rows'] != 142 or r['fail_rows'] for r in results):
    raise SystemExit('one or more per-window 71-stage comparisons failed')
