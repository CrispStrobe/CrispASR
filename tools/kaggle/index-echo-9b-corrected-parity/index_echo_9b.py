#!/usr/bin/env python3
"""9B: released F32 GPU/offload oracle, then native CUDA F16/Q8 parity.

Run only after conversion/quantization; never derive the oracle from GGUF.
Large files live on Kaggle's ephemeral temp disk, output contains receipts.
"""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

SCRIPT_VERSION = '2026-10-02.2'
SOURCE_COMMIT = 'b1e4443ddf2871e8db1b24d344691b2fab6f2924'
SOURCE_REVISION = 'b8ac6fb7d3dc17cee48a52201bd3d93dc86b0dba'
MODEL_REPO = 'cstr/index-echo-9b-GGUF'
FIXTURE_REPO = 'cstr/crispasr-regression-fixtures'
ROOT = Path('/kaggle/temp/index-echo-9b-repo')
TEMP = Path('/kaggle/temp/index-echo-9b')
OUT = Path('/kaggle/working')
TEMP.mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)
os.environ['TMPDIR'] = str(TEMP)


def run(*command, **kwargs):
    subprocess.run(list(map(str, command)), check=True, **kwargs)


import torch
hardware = subprocess.check_output(['nvidia-smi', '--query-gpu=name,compute_cap,memory.total',
                                    '--format=csv,noheader'], text=True).strip()
print('actual GPU:', hardware, 'torch:', torch.__version__, flush=True)
if not torch.cuda.is_available() or any(torch.cuda.get_device_capability(i)[0] < 7
                                       for i in range(torch.cuda.device_count())):
    raise RuntimeError('Inconclusive: actual GPU unsupported by installed PyTorch; no weight pull')
if sum(torch.cuda.get_device_properties(i).total_memory for i in range(torch.cuda.device_count())) < 24 * 2**30:
    raise RuntimeError('Inconclusive: F16 native validation requires at least 24 GiB aggregate VRAM; no weight pull')
run('git', 'init', ROOT)
run('git', '-C', ROOT, 'remote', 'add', 'origin', 'https://github.com/CrispStrobe/CrispASR.git')
run('git', '-C', ROOT, 'fetch', '--depth=1', 'origin', SOURCE_COMMIT)
run('git', '-C', ROOT, 'checkout', 'FETCH_HEAD')
run('git', '-C', ROOT, 'submodule', 'update', '--init', '--recursive', '--depth=1')
sys.path.insert(0, str(ROOT / 'tools/kaggle'))
import kaggle_harness as kh
kh.init_progress()
kh.provenance(SCRIPT_VERSION, ROOT)
# Preserve the preinstalled CUDA PyTorch. No pip torch or torchaudio install.
run(sys.executable, '-m', 'pip', 'install', '-q', 'transformers==5.6.0', 'accelerate',
    'gguf', 'huggingface_hub', 'librosa>=0.10', 'soundfile>=0.12', 'safetensors>=0.5',
    'silero-vad>=5.1', '--no-deps')
# Other Transformers/Accelerate dependencies are small and do not replace torch.
run(sys.executable, '-m', 'pip', 'install', '-q', 'tokenizers', 'huggingface_hub',
    'psutil', 'packaging', 'pyyaml', 'regex', 'safetensors', 'numpy', 'tqdm',
    'onnxruntime', 'filelock', 'requests')
from huggingface_hub import HfApi, snapshot_download, hf_hub_download
import psutil
api = HfApi(token=kh.resolve_hf_token(require=True))
os.environ['HF_TOKEN'] = api.token
model_revision = 'd1cc752e82bb97842052f2a5cd335512da984f96'
files = api.list_repo_files(MODEL_REPO, revision=model_revision)
for cohort in ['f16', 'q8_0']:
    for suffix in [f'{cohort}.gguf', f'decoder-{cohort}.gguf']:
        if 'index-echo-9b-' + suffix not in files:
            raise RuntimeError('Convert and quantize must finish before dumping references')
receipt = dict(script_version=SCRIPT_VERSION, source_commit=SOURCE_COMMIT, source_revision=SOURCE_REVISION,
               model_repo=MODEL_REPO, model_revision=model_revision, hardware=hardware,
               torch=torch.__version__, reference_dtype='float32', cases=[])


def save():
    (OUT / 'index-echo-9b.json').write_text(json.dumps(receipt, indent=2, ensure_ascii=False) + '\n')


memory = {str(i): f'{int(torch.cuda.mem_get_info(i)[0] / 2**30) - (3 if i == 0 else 1)}GiB'
          for i in range(torch.cuda.device_count())}
memory['cpu'] = f'{max(1, int(psutil.virtual_memory().available / 2**30) - 5)}GiB'
receipt['reference_max_memory'] = memory
save()
source = Path(snapshot_download('IndexTeam/Index-Echo-S2TT-9B', revision=SOURCE_REVISION,
                               local_dir=TEMP / 'source'))
reference_env = dict(os.environ, INDEX_ECHO_REF_DEVICE='cuda:0', INDEX_ECHO_REF_DTYPE='float32',
                     INDEX_ECHO_REF_DEVICE_MAP='auto', INDEX_ECHO_REF_CAPTURE_GENERATION='1', INDEX_ECHO_REF_MAX_MEMORY=json.dumps(memory),
                     INDEX_ECHO_REF_OFFLOAD_DIR=str(TEMP / 'offload'), INDEX_ECHO_REF_THREADS='4')
refs = TEMP / 'references'
refs.mkdir()
# Independent short-tail audio derived from the repository sample.
import wave
with wave.open(str(ROOT / 'samples/jfk.wav'), 'rb') as wav:
    params = wav.getparams()
    wav.setpos(8 * 16000)
    tail = wav.readframes(3 * 16000)
with wave.open(str(refs / 'jfk-tail.wav'), 'wb') as wav:
    wav.setparams(params)
    wav.writeframes(tail)
clips = [('jfk', 'jfk_11s', ROOT / 'samples/jfk.wav'),
         ('zh', 'zh', ROOT / 'samples/paraformer_zh.wav'),
         ('jfk-tail', 'jfk_tail', refs / 'jfk-tail.wav')]
for clip, remote, audio in clips:
    ref = refs / f'{clip}-ref.gguf'
    with (OUT / f'reference-{clip}.log').open('w') as log, kh.build_heartbeat('reference.' + clip, interval_s=30):
        run(sys.executable, ROOT / 'tools/dump_reference.py', '--backend', 'index-echo',
            '--model-dir', source, '--audio', audio, '--output', ref,
            env=reference_env, stdout=log, stderr=subprocess.STDOUT, timeout=7200)
    from gguf import GGUFReader
    reader = GGUFReader(ref)
    golden = reader.fields['crispasr.ref.generated_text'].contents()
    aligned = reader.fields['crispasr.ref.raw_greedy_alignment'].contents()
    import importlib.util
    spec = importlib.util.spec_from_file_location('released_parse', source / 'infer.py')
    released = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(released)
    cues, warnings = released.parse_hyp(golden)
    if not cues or warnings or any(not c['en'] for c in cues) or aligned != 'True':
        kh.step('reference.' + clip + '.rejected', parse_warnings=warnings, raw_greedy_alignment=aligned)
        raise RuntimeError('Fresh released source output/cache is not a valid oracle: ' + clip)
    del reader
    api.upload_file(path_or_fileobj=ref, path_in_repo=f'index-echo-9b-f32-generation/{remote}/ref.gguf', repo_id=FIXTURE_REPO)
    kh.step('reference.' + clip + '.uploaded')
api.upload_file(path_or_fileobj=refs / 'jfk-tail.wav', path_in_repo='index-echo-9b-f32-generation/jfk_tail/audio.wav', repo_id=FIXTURE_REPO)
with (OUT / 'reference-pipeline.log').open('w') as log, kh.build_heartbeat('reference.pipeline', interval_s=30):
    code = ('import sys; sys.path.insert(0, sys.argv[1]); '
            'from reference_backends.index_echo import dump_pipeline; '
            'dump_pipeline(sys.argv[2], sys.argv[3], sys.argv[4])')
    run(sys.executable, '-c', code, ROOT / 'tools', source, refs, ROOT / 'samples',
        env=reference_env, stdout=log, stderr=subprocess.STDOUT, timeout=14400)
api.upload_file(path_or_fileobj=refs / 'pipeline.json', path_in_repo='index-echo-9b-f32-generation/pipeline/reference.json', repo_id=FIXTURE_REPO)
api.upload_file(path_or_fileobj=refs / 'pipeline-multi.wav', path_in_repo='index-echo-9b/pipeline/audio.wav', repo_id=FIXTURE_REPO)
receipt['fixture_repo'] = FIXTURE_REPO
receipt['fixture_revision'] = api.model_info(FIXTURE_REPO).sha
shutil.copy2(refs / 'pipeline.json', OUT / 'reference-pipeline.json')
save()
kh.step('reference.complete', fixture_revision=receipt['fixture_revision'])
shutil.rmtree(source)
shutil.rmtree(TEMP / 'offload', ignore_errors=True)
kh.install_build_toolchain()
arch = kh.detect_cuda_arch()
build = TEMP / 'build'
flags = ['-DCMAKE_BUILD_TYPE=Release', '-DCRISPASR_BUILD_TESTS=OFF', '-DCRISPASR_BUILD_SERVER=OFF', '-DBUILD_SHARED_LIBS=ON']
flags += kh.cuda_build_flags(arch) + kh.cache_and_link_flags()
run('cmake', '-S', ROOT, '-B', build, '-G', 'Ninja', *flags)
with kh.build_heartbeat('build.cuda', interval_s=30):
    kh.sh_with_progress(f'cmake --build {build} --target crispasr-cli crispasr-lib crispasr-diff -j{kh.safe_build_jobs(gpu=True)}', cwd=str(ROOT))
receipt['cuda_arch'] = arch
failed = []
models = TEMP / 'models'
from gguf import GGUFReader
sys.path.insert(0, str(ROOT / 'python'))
sys.path.insert(0, str(ROOT / 'tools/ci-heavy'))
from crispasr import Session
from index_echo_pipeline_check import check_pipeline
library = next(build.rglob('libcrispasr.so'))
# The native checker reads immutable independent references through these links.
(models / 'reference-f32').mkdir(parents=True, exist_ok=True)
for filename in ['pipeline.json', 'pipeline-multi.wav']:
    (models / 'reference-f32' / filename).symlink_to(refs / filename)
for cohort in ['f16', 'q8_0']:
    snapshot_download(MODEL_REPO, revision=model_revision, local_dir=models,
                      allow_patterns=[f'index-echo-9b-{cohort}.gguf', f'index-echo-9b-decoder-{cohort}.gguf'])
    primary = models / f'index-echo-9b-{cohort}.gguf'
    for clip, _, audio in clips:
        tag = f'{cohort}-{clip}'
        with (OUT / f'{tag}-diff.log').open('w') as log, kh.build_heartbeat(tag + '.diff', interval_s=30):
            diff = subprocess.run([str(build / 'bin/crispasr-diff'), 'index-echo', str(primary),
                                   str(refs / f'{clip}-ref.gguf'), str(audio)],
                                  stdout=log, stderr=subprocess.STDOUT, timeout=7200)
        reader = GGUFReader(refs / f'{clip}-ref.gguf')
        golden = reader.fields['crispasr.ref.generated_text'].contents()
        del reader
        expected = [line.strip() for line in golden.splitlines() if line.strip() and not line.strip().startswith('[')]
        prefix = OUT / tag
        started = time.perf_counter()
        with (OUT / f'{tag}-cli.log').open('w') as log, kh.build_heartbeat(tag + '.decode', interval_s=30):
            decode = subprocess.run([str(build / 'bin/crispasr'), '-m', str(primary), '-f', str(audio),
                                     '-l', 'auto', '-t', '4', '-osrt', '-of', str(prefix)],
                                    stdout=log, stderr=subprocess.STDOUT, timeout=7200)
        srt = prefix.with_suffix('.srt')
        actual = [line.strip() for line in srt.read_text().splitlines() if line.strip()
                  and not line.strip().isdigit() and '-->' not in line] if srt.exists() else []
        passed = diff.returncode == 0 and decode.returncode == 0 and actual == expected
        receipt['cases'].append(dict(tag=tag, stage_rc=diff.returncode, decode_rc=decode.returncode,
                                     text_match=actual == expected, actual=actual, expected=expected,
                                     cold_seconds=time.perf_counter() - started, passed=passed))
        if not passed:
            failed.append(tag)
        save()
        kh.step(tag, passed=passed)
    anonymous = models / f'opaque-{cohort}.gguf'
    anonymous.symlink_to(primary)
    import numpy as np
    with Session(str(anonymous), lib_path=str(library), n_threads=4) as session:
        assert session.backend == 'index-echo', session.backend
        for clip, _, audio in clips:
            with wave.open(str(audio), 'rb') as wav:
                pcm = np.frombuffer(wav.readframes(wav.getnframes()), dtype=np.int16).astype(np.float32) / 32768
            segments = session.transcribe(pcm)
            reader = GGUFReader(refs / f'{clip}-ref.gguf')
            golden = reader.fields['crispasr.ref.generated_text'].contents()
            cues, warnings = released.parse_hyp(golden)
            actual = [dict(start=s.start, end=s.end, text=s.text) for s in segments]
            expected = [dict(start=c['st'], end=c['et'], text=c['zh']+'\n'+c['en']) for c in cues]
            matched = not warnings and len(actual) == len(expected) and all(
                a['text'] == e['text'] and abs(a['start']-e['start']) <= .0051 and abs(a['end']-e['end']) <= .0051
                for a, e in zip(actual, expected))
            receipt.setdefault('c_abi', []).append(dict(cohort=cohort, clip=clip, anonymous_metadata=True,
                                                       matched=matched, actual=actual, expected=expected))
            if not matched:
                failed.append(cohort + '-' + clip + '-c-abi')
            del reader
    anonymous.unlink()
    with kh.build_heartbeat(cohort+'.full-pipeline', interval_s=30):
        pipeline_failed = check_pipeline(ROOT, OUT, build, library, models, cohort, 'reference-f32',
                                         model_prefix='index-echo-9b')
    failed.extend(cohort + ':' + error for error in pipeline_failed)
    receipt.setdefault('full_pipeline', {})[cohort] = pipeline_failed
    save()
    primary.unlink()
    (models / f'index-echo-9b-decoder-{cohort}.gguf').unlink()
receipt['failed'] = failed
receipt['direct_cuda_passed'] = not failed
receipt['full_pipeline_native_pending'] = False
receipt['validated'] = not failed
save()
if failed:
    raise RuntimeError('Stage or decoded parity failed: ' + ', '.join(failed))
