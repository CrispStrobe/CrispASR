#!/usr/bin/env python3
"""Build Index-Echo's shared ABI, CLI and stage diff on a hosted CPU runner."""
import argparse
import json
import os
import re
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(os.environ['HEAVY_OUT'])
BUILD = Path(os.environ['HEAVY_SCRATCH']) / 'index-echo-build'
OUT.mkdir(parents=True, exist_ok=True)


def run(*args):
    print('run:', *args, flush=True)
    subprocess.run(list(map(str, args)), cwd=ROOT, check=True)


parser = argparse.ArgumentParser()
parser.add_argument('--build-only', action='store_true')
parser.add_argument('--stage-diagnostic', action='store_true', help='Numerical investigation only; never model/decoded acceptance')
parser.add_argument('--fixture-prefix', help='Explicit independent capture recipe namespace')
parser.add_argument('--size', choices=['2b', '9b'], default='2b')
parser.add_argument('--model-revision', help='Immutable staging pin (required for unregistered 9B)')
parser.add_argument('--fixture-revision', help='Immutable independent reference pin')
parser.add_argument('--regression', action='store_true', help='Run the actual pinned nightly driver after native validation')
parser.add_argument('--reference-subdir', choices=['reference', 'reference-f32'], default='reference')
parser.add_argument('--pipeline', action='store_true', help='Validate released file/VAD/target/context oracle')
parser.add_argument('--pipeline-fixture-prefix', help='Independent file oracle namespace, when different from stage captures')
parser.add_argument('--pipeline-audio-path', help='Exact companion audio path in the pinned fixture repository')
parser.add_argument('--cohorts', nargs='+', choices=['f16', 'q8_0', 'q8_0_selective', 'q8_0_ffn', 'q4_k', 'q4_k_selective'], default=['f16'])
parser.add_argument('--clips', nargs='+', choices=['jfk', 'zh', 'jfk-tail'], default=['jfk', 'zh', 'jfk-tail'])
args = parser.parse_args()
prefix = 'index-echo-' + args.size
if args.size == '9b' and not args.build_only and not (args.model_revision and args.fixture_revision):
    parser.error('9B validation requires immutable model and fixture revisions')
run('cmake', '-S', ROOT, '-B', BUILD, '-G', 'Ninja', '-DCMAKE_BUILD_TYPE=Release',
    '-DBUILD_SHARED_LIBS=ON', '-DCRISPASR_BUILD_SERVER=OFF', '-DGGML_NATIVE=OFF')
run('cmake', '--build', BUILD, '--target', 'crispasr-cli', 'crispasr-lib', 'crispasr-diff',
    'test-index-echo-windows', 'test-index-echo-batch', 'test-index-echo-connector', 'test-session-autochunk', 'test-arch-backend-map',
    'test-crispasr-diff-compare', 'test-registry', 'test-index-echo-live', '-j', '4')
for test in ['test-index-echo-windows', 'test-index-echo-batch', 'test-index-echo-connector', 'test-session-autochunk', 'test-arch-backend-map',
             'test-crispasr-diff-compare', 'test-registry']:
    run(BUILD / 'bin' / test)
run(sys.executable, ROOT / 'tools/gen-feature-matrix.py', '--crispasr', BUILD / 'bin/crispasr')
run(sys.executable, ROOT / 'tools/gen-backend-caps-table.py', '--crispasr', BUILD / 'bin/crispasr')
for relative in ['docs/feature-matrix.md', 'docs/feature-matrix.html', 'src/core/backend_caps_table.h']:
    source = ROOT / relative
    if source.exists(): shutil.copy2(source, OUT / source.name)
run('cmake', '--build', BUILD, '--target', 'crispasr-lib', '-j', '4')
library = next(BUILD.rglob('libcrispasr.so'))
run(sys.executable, ROOT / 'tools/check-backend-wiring.py', '--crispasr', BUILD / 'bin/crispasr',
    '--lib', library, '--require-lib')
if args.build_only:
    (OUT / 'summary.md').write_text('Index-Echo shared library, CLI, diff and integration unit tests built. '
                                   'Model parity remains pending.\n')
    sys.exit(0)
from huggingface_hub import hf_hub_download, snapshot_download
manifest = json.loads((ROOT / 'tests/regression/manifest.json').read_text())
entry = next((e for e in manifest['backends'] if e['name'] == prefix), None)
destination = entry['gguf']['repo'] if entry else 'cstr/' + prefix + '-GGUF'
model_revision = args.model_revision or entry['gguf']['revision']
fixtures = dict(manifest['fixtures'])
if args.fixture_revision: fixtures['revision'] = args.fixture_revision
fixture_prefix = args.fixture_prefix or prefix + ('-f32' if args.reference_subdir == 'reference-f32' else '')


def download_references(models):
    # Model weights and independent reference captures have different canonical
    # repositories and immutable pins. Never depend on temporary transfer HEAD.
    names = {'jfk': 'jfk_11s', 'zh': 'zh', 'jfk-tail': 'jfk_tail'}
    paths = {f'{clip}-ref.gguf': f'{fixture_prefix}/{names[clip]}/ref.gguf' for clip in args.clips}
    if 'jfk-tail' in args.clips:
        paths['jfk-tail.wav'] = f'{fixture_prefix}/jfk_tail/audio.wav'
    if args.pipeline:
        paths['pipeline.json'] = f'{args.pipeline_fixture_prefix or fixture_prefix}/pipeline/reference.json'
        paths['pipeline-multi.wav'] = args.pipeline_audio_path or f'{prefix}/pipeline/audio.wav'
    folder = models / args.reference_subdir
    folder.mkdir(parents=True, exist_ok=True)
    for name, remote in paths.items():
        source = Path(hf_hub_download(fixtures['repo'], remote, revision=fixtures['revision'],
                       local_dir=Path(os.environ['HEAVY_SCRATCH']) / 'index-echo-fixtures'))
        target = folder / name
        if not target.exists():
            target.symlink_to(source)

(OUT / 'validation-provenance.json').write_text(json.dumps(dict(
    model_repo=destination, model_revision=model_revision,
    fixture_repo=fixtures["repo"], fixture_revision=fixtures["revision"],
    stage_fixture_prefix=fixture_prefix,
    pipeline_fixture_prefix=args.pipeline_fixture_prefix or fixture_prefix,
    pipeline_audio_path=args.pipeline_audio_path or f'{prefix}/pipeline/audio.wav',
    source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
    ggml_commit=subprocess.check_output(['git', '-C', 'ggml', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
    threads=4, cpu=subprocess.check_output(['uname', '-m'], text=True).strip()), indent=2) + '\n')

def validate_cohort(cohort):
    models = Path(snapshot_download(destination, revision=model_revision, local_dir=Path(os.environ['HEAVY_SCRATCH']) / 'index-echo-models',
        allow_patterns=[f'{prefix}-{cohort}.gguf', f'{prefix}-decoder-{cohort}.gguf']))
    download_references(models)
    os.environ['TMPDIR'] = os.environ['HEAVY_SCRATCH']
    failures = []
    for clip in args.clips:
        audio = models / args.reference_subdir / 'jfk-tail.wav' if clip == 'jfk-tail' else ROOT / 'samples' / ('paraformer_zh.wav' if clip == 'zh' else 'jfk.wav')
        log_path = OUT / f'{cohort}-{clip}-diff.log'
        command = [str(BUILD / 'bin/crispasr-diff'), 'index-echo', str(models / f'{prefix}-{cohort}.gguf'),
                   str(models / f'{args.reference_subdir}/{clip}-ref.gguf'), str(audio)]
        with log_path.open('w') as log:
            result = subprocess.run(command, env=dict(os.environ, CRISPASR_DIFF_NO_GPU='1'),
                                    stdout=log, stderr=subprocess.STDOUT, timeout=3600)
        print(clip, 'stage diff rc:', result.returncode, log_path.read_text()[-16000:], flush=True)
        if result.returncode: failures.append(clip)
    (OUT / f'stage-results-{cohort}.json').write_text(json.dumps(dict(failed=list(failures)), indent=2))

    if args.stage_diagnostic:
        return failures

    # This live test embeds 2B source transcripts. 9B uses its own independent
    # transcript oracle through the real C ABI below, never a 2B expectation.
    if args.size == '2b':
        subprocess.run([str(BUILD / 'bin/test-index-echo-live')], cwd=ROOT, check=True,
                       env=dict(os.environ, CRISPASR_MODEL_INDEX_ECHO=str(models / f'{prefix}-{cohort}.gguf')))

    # Open a model with an arbitrary filename through the actual Python Session,
    # which tests shared metadata detection and the shipped C ABI, not CLI heuristics.
    sys.path.insert(0, str(ROOT / 'python'))
    import numpy as np
    import wave
    from gguf import GGUFReader
    from crispasr import Session
    assert 'index-echo' in Session.available_backends(lib_path=str(library))
    renamed = models / f'model-without-backend-hint-{cohort}.gguf'
    renamed.symlink_to(models / f'{prefix}-{cohort}.gguf')
    decoded = {}


    def reference_cues(text):
        # Released timestamp / transcript / translation format, parsed independently
        # of the native implementation. These fixtures have no optional context.
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        expected = []
        for i in range(0, len(lines), 3):
            match = re.fullmatch(r'\[(\d+):(\d+(?:\.\d+)?)-(\d+):(\d+(?:\.\d+)?)\]', lines[i])
            if not match or i + 2 >= len(lines):
                raise RuntimeError('Malformed independent decoded reference')
            expected.append(dict(start=60 * int(match[1]) + float(match[2]),
                                 end=60 * int(match[3]) + float(match[4]),
                                 text=lines[i + 1] + '\n' + lines[i + 2]))
        return expected


    with Session(str(renamed), lib_path=str(library), n_threads=4) as session:
        assert session.backend == 'index-echo', session.backend
        for clip in args.clips:
            audio = models / args.reference_subdir / 'jfk-tail.wav' if clip == 'jfk-tail' else ROOT / 'samples' / ('paraformer_zh.wav' if clip == 'zh' else 'jfk.wav')
            with wave.open(str(audio), 'rb') as wav:
                assert wav.getframerate() == 16000 and wav.getnchannels() == 1 and wav.getsampwidth() == 2
                pcm = np.frombuffer(wav.readframes(wav.getnframes()), dtype=np.int16).astype(np.float32) / 32768
            segments = session.transcribe(pcm)
            reader = GGUFReader(models / f'{args.reference_subdir}/{clip}-ref.gguf')
            reference = reader.fields['crispasr.ref.generated_text'].contents()
            decoded[clip] = dict(reference=reference, segments=[dict(start=s.start, end=s.end, text=s.text) for s in segments])
            if not segments or any(not s.text or s.end < s.start for s in segments):
                failures.append(clip + ': missing/invalid decoded cues')
            expected = reference_cues(reference)
            actual = decoded[clip]['segments']
            if len(actual) != len(expected) or any(
                    a['text'] != e['text'] or abs(a['start'] - e['start']) > 0.0051 or
                    abs(a['end'] - e['end']) > 0.0051 for a, e in zip(actual, expected)):
                failures.append(clip + ': decoded text/timestamp mismatch')
            # Preserve complete text and timing for review rather than hiding a
            # numerically correct but behaviorally wrong output behind cosine.
            print('decoded', clip, json.dumps(decoded[clip], ensure_ascii=False), flush=True)
            del reader
    (OUT / f'decoded-{cohort}.json').write_text(json.dumps(decoded, indent=2, ensure_ascii=False))
    if args.pipeline:
        from index_echo_pipeline_check import check_pipeline
        failures.extend(check_pipeline(ROOT, OUT, BUILD, library, models, cohort, args.reference_subdir, model_prefix=prefix))
    return failures


results = {}
for cohort in args.cohorts:
    try:
        results[cohort] = validate_cohort(cohort)
    finally:
        # Only one paired cohort coexists on the runner's temporary disk.
        for name in (f'{prefix}-{cohort}.gguf', f'{prefix}-decoder-{cohort}.gguf'):
            (Path(os.environ['HEAVY_SCRATCH']) / 'index-echo-models' / name).unlink(missing_ok=True)
if args.regression:
    sys.path.insert(0, str(ROOT / 'tests/regression'))
    import run_one
    failed = run_one.regression_for(prefix, manifest,
        Path(os.environ['HEAVY_SCRATCH']) / 'index-echo-nightly', BUILD / 'bin/crispasr', BUILD / 'bin/crispasr-diff')
    results['nightly_regression'] = ['Pinned nightly regression failed'] if failed else []

if args.stage_diagnostic:
    (OUT / 'diagnostic-status.json').write_text(json.dumps(dict(validated=False, decoded_output_checked=False,
        reason='Numerical diagnostic; source capture behavior is under investigation'), indent=2)+'\n')
(OUT / 'cohort-results.json').write_text(json.dumps(results, indent=2))
if any(results.values()):
    raise RuntimeError('Cohort validation failed: ' + json.dumps(results))
if args.stage_diagnostic:
    (OUT / 'summary.md').write_text('Numerical diagnostic passed; source/decoded acceptance remains pending.\n')
    sys.exit(0)
(OUT / 'summary.md').write_text(', '.join(args.cohorts) + ': stage/magnitude/prompt/cache parity and '
    'Python Session metadata autodetection / exact decoded text/timestamp parity passed.\n')
