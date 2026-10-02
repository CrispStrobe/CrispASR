#!/usr/bin/env python3
"""Independent CUDA stage/cache, exact decoding, full-file and roundtrip acceptance.

CPU synthesis already ran on GitHub. Consume its pinned WAVs; never use a GPU
session for CPU synthesis or pretend these tests prove the full-file pipeline.
"""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile

SCRIPT_VERSION = '2026-10-02.4'
SOURCE_COMMIT = '4584fc035c3655e3826d5796b3756e47f68c25e4'
BUILD_COMMIT = 'f91a31da8d586157043639d0e6039c778d2571dc'
MODEL_REVISION = 'dffbadf0f173446fee0364a0807803d2b2fb6f49'
REFERENCE_REVISION = 'cb678dfd4806778aa55c39fe7b7a710a54cdc153'
AUDIO_REVISION = 'd0a7d7a8be318a5841dfdbe6ad37d3acf75523e3'
BUILD_RUN = 36997586995
BUNDLE_REVISION = '9ffaeaab43fece5ec5c0baeb4884e2db87625f6f'
BUNDLE_SHA256 = '166e8741738827fd0b997551bf27bfa93b5dcf04c898056fdedade87ba5c9f3e'
# An experiment may reuse these exact acceptance gates with a different pinned
# CI runtime. Defaults retain the published proof; configuration is recorded.
config_path = os.environ.get('INDEX_ECHO_VALIDATION_CONFIG')
config = json.loads(Path(config_path).read_text()) if config_path else {}
SOURCE_COMMIT = config.get('source_commit', SOURCE_COMMIT)
BUILD_COMMIT = config.get('build_commit', BUILD_COMMIT)
BUILD_RUN = config.get('build_run', BUILD_RUN)
BUNDLE_REVISION = config.get('bundle_revision', BUNDLE_REVISION)
BUNDLE_SHA256 = config.get('bundle_sha256', BUNDLE_SHA256)
if len(SOURCE_COMMIT) != 40 or len(BUILD_COMMIT) != 40:
    raise RuntimeError('Source and build pins must be immutable commits')
if len(REFERENCE_REVISION) != 40 or len(BUNDLE_REVISION) != 40 or len(BUNDLE_SHA256) != 64:
    raise RuntimeError('Independent reference and CI bundle pins must be set before launch')
ROOT = Path('/kaggle/temp/index-echo-validation-repo')
TEMP = Path('/kaggle/temp/index-echo-validation')
OUT = Path('/kaggle/working')
TEMP.mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)
os.environ['TMPDIR'] = str(TEMP)


def run(*command, **kw):
    subprocess.run(list(map(str, command)), check=True, **kw)


hardware = subprocess.check_output(['nvidia-smi', '--query-gpu=name,compute_cap,memory.total',
                                    '--format=csv,noheader'], text=True).strip()
print('actual GPU:', hardware, flush=True)
rows = [line.split(',') for line in hardware.splitlines()]
if any(row[1].strip() != '7.5' for row in rows) or sum(int(row[-1].strip().split()[0]) for row in rows) < 24 * 1024:
    raise RuntimeError('Inconclusive: CI bundle targets SM75 and F16 requires 24 GiB aggregate VRAM; no model pull')
run('git', 'init', ROOT)
run('git', '-C', ROOT, 'remote', 'add', 'origin', 'https://github.com/CrispStrobe/CrispASR.git')
run('git', '-C', ROOT, 'fetch', '--depth=1', 'origin', SOURCE_COMMIT)
run('git', '-C', ROOT, 'checkout', 'FETCH_HEAD')
run('git', '-C', ROOT, 'submodule', 'update', '--init', '--recursive', '--depth=1')
sys.path.insert(0, str(ROOT / 'tools/kaggle'))
import kaggle_harness as kh
kh.init_progress()
kh.provenance(SCRIPT_VERSION, ROOT)
run(sys.executable, '-m', 'pip', 'install', '-q', 'huggingface_hub', 'gguf', 'numpy')
from huggingface_hub import HfApi, hf_hub_download, snapshot_download
os.environ['HF_TOKEN'] = kh.resolve_hf_token(require=True)
fixture_repo = 'cstr/crispasr-regression-fixtures'
manifest_path = hf_hub_download(fixture_repo, 'index-echo-9b/roundtrip-piper/roundtrip-audio.json',
                                revision=AUDIO_REVISION, local_dir=TEMP / 'fixtures')
manifest = json.loads(Path(manifest_path).read_text())
for item in manifest['cases'].values():
    path = hf_hub_download(fixture_repo, 'index-echo-9b/roundtrip-piper/' + item['audio'],
                           revision=AUDIO_REVISION, local_dir=TEMP / 'fixtures')
    import hashlib
    if hashlib.sha256(Path(path).read_bytes()).hexdigest() != item['sha256']:
        raise RuntimeError('Synthetic fixture checksum mismatch')
    shutil.copy2(path, OUT / item['audio'])
# CPU compilation happened on GitHub; Kaggle executes the pinned build on GPUs.
archive_path = hf_hub_download('cstr/crispasr-index-echo-cuda-validation',
    'index-echo-cuda-validation.tar.gz', repo_type='dataset',
    revision=BUNDLE_REVISION, local_dir=TEMP / 'artifact')
import hashlib
digest = hashlib.sha256()
with open(archive_path, 'rb') as stream:
    for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''): digest.update(chunk)
if digest.hexdigest() != BUNDLE_SHA256:
    raise RuntimeError('CI bundle checksum mismatch')
with tarfile.open(archive_path) as archive:
    archive.extractall(TEMP, filter='data')
bundle = TEMP / 'bundle'
provenance = json.loads((bundle / 'provenance.json').read_text())
if provenance['sha'] != BUILD_COMMIT or provenance['architectures'] != [75]:
    raise RuntimeError('CI bundle build provenance mismatch')
os.environ['LD_LIBRARY_PATH'] = str(bundle)+':'+os.environ.get('LD_LIBRARY_PATH','')
arch = kh.detect_cuda_arch()
build = TEMP / 'build'
(build / 'bin').mkdir(parents=True)
for executable in ['crispasr', 'crispasr-diff']:
    (build / 'bin' / executable).symlink_to(bundle / executable)
kh.step('bundle.ready', build_run=BUILD_RUN, build_commit=BUILD_COMMIT,
        revision=BUNDLE_REVISION, sha256=BUNDLE_SHA256)
sys.path.insert(0, str(ROOT / 'tools/ci-heavy'))
from index_echo_roundtrip import check_roundtrips
library = next(bundle.glob('libcrispasr.so*'))
from gguf import GGUFReader
import numpy as np
import re
import wave
sys.path.insert(0, str(ROOT / 'python'))
from crispasr import Session
from index_echo_pipeline_check import check_pipeline
refs = TEMP / 'references'
refs.mkdir(exist_ok=True)
for local, remote in {
    'jfk-ref.gguf': 'jfk_11s/ref.gguf',
    'zh-ref.gguf': 'zh/ref.gguf',
    'jfk-tail-ref.gguf': 'jfk_tail/ref.gguf',
    'jfk-tail.wav': 'jfk_tail/audio.wav',
}.items():
    source = hf_hub_download(fixture_repo, 'index-echo-9b-f32-generation/' + remote,
                             revision=REFERENCE_REVISION, local_dir=TEMP / 'fixtures')
    (refs / local).symlink_to(source)
source = hf_hub_download(fixture_repo, 'index-echo-9b-f32-zh-context/pipeline/reference.json',
                         revision=REFERENCE_REVISION, local_dir=TEMP / 'fixtures')
(refs / 'pipeline.json').symlink_to(source)
oracle = json.loads(Path(source).read_text())
if oracle.get('complete') is not True or set(oracle['cases']) != {'jfk-en','zh-en','zh-ja','zh-es','multi-en'}:
    raise RuntimeError('Incomplete independent original-source file oracle')
source = hf_hub_download(fixture_repo, 'index-echo-9b/pipeline-zh-context/audio.wav',
                         revision=REFERENCE_REVISION, local_dir=TEMP / 'fixtures')
(refs / 'pipeline-multi.wav').symlink_to(source)
clips = [('jfk', ROOT / 'samples/jfk.wav'), ('zh', ROOT / 'samples/paraformer_zh.wav'),
         ('jfk-tail', refs / 'jfk-tail.wav')]
receipt = dict(script_version=SCRIPT_VERSION, source_commit=SOURCE_COMMIT,
               model_revision=MODEL_REVISION, reference_revision=REFERENCE_REVISION,
               audio_revision=AUDIO_REVISION, pipeline_reference_dtype='independently forced all-float32', direct_reference_dtype='float32', build_commit=BUILD_COMMIT, build_run=BUILD_RUN, bundle_revision=BUNDLE_REVISION, bundle_sha256=BUNDLE_SHA256, hardware=hardware, cuda_arch=arch,
               full_pipeline_checked=False, cohorts={}, validated=False, runtime_config=config,
               pipeline_disable=os.environ.get('CRISPASR_LLAMA_PIPELINE_DISABLE', '0'))
failed = []


def save():
    receipt['failed'] = failed
    (OUT / 'cuda-validation.json').write_text(json.dumps(receipt, indent=2, ensure_ascii=False) + '\n')


def reference_cues(text):
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) % 3:
        raise RuntimeError('Malformed independent source subtitles')
    cues = []
    for i in range(0, len(lines), 3):
        m = re.fullmatch(r'\[(\d+):(\d+(?:\.\d+)?)-(\d+):(\d+(?:\.\d+)?)\]', lines[i])
        if not m:
            raise RuntimeError('Malformed source timestamp')
        cues.append(dict(start=60*int(m[1])+float(m[2]), end=60*int(m[3])+float(m[4]),
                         text=lines[i+1]+'\n'+lines[i+2]))
    return cues


def read_srt(path):
    if not path.exists(): return []
    cues = []
    for block in re.split(r'\n\s*\n', path.read_text().strip()):
        lines = block.splitlines()
        if len(lines) < 3 or not lines[0].strip().isdigit():
            raise RuntimeError('Malformed native CLI subtitles')
        match = re.fullmatch(r'(\d+):(\d+):(\d+),(\d+) --> (\d+):(\d+):(\d+),(\d+)', lines[1].strip())
        if not match: raise RuntimeError('Malformed native CLI timing')
        times = [3600*int(match[i])+60*int(match[i+1])+int(match[i+2])+int(match[i+3])/1000 for i in [1,5]]
        cues.append(dict(start=times[0], end=times[1], text='\n'.join(lines[2:])))
    return cues


def cues_match(actual, expected):
    return len(actual)==len(expected) and all(a['text']==e['text'] and abs(a['start']-e['start'])<=.0051 and abs(a['end']-e['end'])<=.0051 for a,e in zip(actual,expected))


for cohort in ['f16']:
    models = Path(snapshot_download('cstr/index-echo-9b-GGUF', revision=MODEL_REVISION,
                  local_dir=TEMP / 'models', allow_patterns=[f'index-echo-9b-{cohort}.gguf', f'index-echo-9b-decoder-{cohort}.gguf']))
    (models / 'reference-f32').mkdir(exist_ok=True)
    for name in ['pipeline.json', 'pipeline-multi.wav']:
        path = models / 'reference-f32' / name
        if not path.exists(): path.symlink_to(refs / name)
    primary = models / f'index-echo-9b-{cohort}.gguf'
    result = dict(stages={}, c_abi={}, cli={})
    receipt['cohorts'][cohort] = result
    for clip, audio in clips:
        with (OUT / f'{cohort}-{clip}-diff.log').open('w') as log, kh.build_heartbeat(cohort+'.'+clip+'.diff', interval_s=30):
            diff = subprocess.run([str(build / 'bin/crispasr-diff'), 'index-echo', str(primary),
                                   str(refs / f'{clip}-ref.gguf'), str(audio)],
                                  stdout=log, stderr=subprocess.STDOUT, timeout=3600)
        cuda_used = 'load_tensors: layer' in (OUT / f'{cohort}-{clip}-diff.log').read_text() and 'assigned to device CUDA' in (OUT / f'{cohort}-{clip}-diff.log').read_text()
        result['stages'][clip] = dict(rc=diff.returncode, cuda_used=cuda_used)
        if diff.returncode or not cuda_used: failed.append(cohort+':'+clip+':stage')
        save()
    anonymous = models / f'opaque-{cohort}.gguf'
    anonymous.symlink_to(primary)
    with kh.build_heartbeat(cohort+'.c-abi', interval_s=30), Session(str(anonymous), lib_path=str(library), n_threads=4) as session:
        assert session.backend == 'index-echo', session.backend
        for clip, audio in clips:
            with wave.open(str(audio), 'rb') as wav:
                assert wav.getframerate()==16000 and wav.getnchannels()==1 and wav.getsampwidth()==2
                pcm = np.frombuffer(wav.readframes(wav.getnframes()), dtype=np.int16).astype(np.float32)/32768
            segments = session.transcribe(pcm)
            reader = GGUFReader(refs / f'{clip}-ref.gguf')
            expected = reference_cues(reader.fields['crispasr.ref.generated_text'].contents())
            del reader
            actual = [dict(start=s.start, end=s.end, text=s.text) for s in segments]
            match = cues_match(actual, expected)
            result['c_abi'][clip] = dict(actual=actual, expected=expected, passed=match, anonymous_metadata=True)
            if not match: failed.append(cohort+':'+clip+':c-abi')
            save()
    anonymous.unlink()
    # Release ABI weights before loading the pair in a CLI process.
    for clip, audio in clips:
        prefix = OUT / f'{cohort}-{clip}-cli'
        with (OUT / f'{cohort}-{clip}-cli.log').open('w') as log, kh.build_heartbeat(cohort+'.'+clip+'.cli', interval_s=30):
            decoded = subprocess.run([str(build / 'bin/crispasr'), '-m', str(primary), '-f', str(audio), '-l', 'auto', '-t', '4', '-osrt', '-of', str(prefix)], stdout=log, stderr=subprocess.STDOUT, timeout=3600)
        srt = prefix.with_suffix('.srt')
        actual = read_srt(srt)
        expected = result['c_abi'][clip]['expected']
        match = decoded.returncode==0 and cues_match(actual, expected)
        result['cli'][clip] = dict(rc=decoded.returncode, actual=actual, expected=expected, passed=match)
        if not match: failed.append(cohort+':'+clip+':cli')
        save()
    with kh.build_heartbeat(cohort+'.pipeline', interval_s=30):
        pipeline_failures = check_pipeline(ROOT, OUT, build, library, models, cohort, 'reference-f32', model_prefix='index-echo-9b')
    pipeline_cli = read_srt(OUT / f'pipeline-{cohort}-cli.srt')
    pipeline_expected = json.loads((refs / 'pipeline.json').read_text())['cases']['jfk-en']['segments']
    if not cues_match(pipeline_cli, pipeline_expected): pipeline_failures.append('real CLI exact subtitle/timestamp mismatch')
    result['pipeline_cli'] = dict(actual=pipeline_cli, expected=pipeline_expected)
    result['pipeline_failures'] = pipeline_failures
    receipt['full_pipeline_checked'] = all('pipeline_failures' in item for item in receipt['cohorts'].values())
    failed.extend(cohort+':pipeline:'+item for item in pipeline_failures)
    with kh.build_heartbeat(cohort+'.roundtrip', interval_s=30):
        roundtrip_failures = check_roundtrips(ROOT, OUT, build / 'bin/crispasr', library, primary, manifest, use_gpu=True)
    result['roundtrip_failures'] = roundtrip_failures
    failed.extend(cohort+':roundtrip:'+item for item in roundtrip_failures)
    save()
    if not config.get('keep_models', False):
        primary.unlink()
        (models / f'index-echo-9b-decoder-{cohort}.gguf').unlink()
receipt['validated'] = not failed
save()
if failed:
    raise RuntimeError('CUDA acceptance failed: ' + ', '.join(failed))
