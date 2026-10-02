#!/usr/bin/env python3
"""Actual Piper synthesis -> Index-Echo CLI and shared-ABI recognition.

CPU work runs through heavy-cpu.yml. WAVs, synthesis logs, SRTs, immutable
model pins and independent text error rates remain in the result artifact.
This behavioral test supplements source stage/cache parity; it cannot replace it.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import wave

PIPER_REPO = 'cstr/piper-en_US-lessac-medium-GGUF'
PIPER_REVISION = '6b3d385695cd2e91c1af9d37ca5684ee649e2e3d'
MODEL_REVISION = 'd1cc752e82bb97842052f2a5cd335512da984f96'
TEXTS = {
    'fox': 'The quick brown fox jumps over the lazy dog.',
    'window': 'Please close the window before you leave the room.',
    'station': 'We will meet at the train station tomorrow morning.',
}


def words(text):
    return re.findall(r"[a-z]+(?:'[a-z]+)?", text.lower())


def word_error_rate(expected, actual):
    wanted, heard = words(expected), words(actual)
    previous = list(range(len(heard) + 1))
    for i, word in enumerate(wanted, 1):
        current = [i]
        for j, other in enumerate(heard, 1):
            current.append(min(current[-1] + 1, previous[j] + 1,
                               previous[j - 1] + (word != other)))
        previous = current
    return previous[-1] / max(1, len(wanted))


def check_roundtrips(root, out, cli, library, primary, audio_manifest, use_gpu=False):
    """Reusable recognition gate for CPU CI and a subsequent CUDA proof."""
    import numpy as np
    sys.path.insert(0, str(root / 'python'))
    from crispasr import Session
    failures, results = [], {}
    # The C ABI must detect the architecture independently of the filename.
    anonymous = primary.parent / 'roundtrip-opaque.gguf'
    anonymous.symlink_to(primary)
    try:
        # Session opens with the library's default device policy. The hosted
        # CPU job builds without CUDA; a CUDA caller supplies its CUDA library.
        with Session(str(anonymous), lib_path=str(library), n_threads=4) as session:
            assert session.backend == 'index-echo', session.backend
            session.set_target_language('en')
            for name, item in audio_manifest['cases'].items():
                audio = out / item['audio']
                with wave.open(str(audio), 'rb') as wav:
                    assert (wav.getframerate(), wav.getnchannels(), wav.getsampwidth()) == (16000, 1, 2)
                    pcm = np.frombuffer(wav.readframes(wav.getnframes()), dtype=np.int16).astype(np.float32) / 32768
                started = time.perf_counter()
                segments = session.transcribe(pcm)
                elapsed = time.perf_counter() - started
                # Index-Echo emits source transcript and target translation on
                # separate lines. Score the requested English target only.
                heard = ' '.join(s.text.strip().splitlines()[-1] for s in segments if s.text.strip())
                wer = word_error_rate(item['text'], heard)
                valid_segments = bool(segments) and all(0 <= s.start < s.end <= len(pcm) / 16000 + .1 for s in segments)
                results[name] = dict(expected=item['text'], english=heard, wer=wer, wer_max=.10,
                                     segments=[dict(start=s.start, end=s.end, text=s.text) for s in segments],
                                     valid_segments=valid_segments,
                                     audio_sha256=hashlib.sha256(audio.read_bytes()).hexdigest(),
                                     seconds=len(pcm) / 16000, elapsed_seconds=elapsed)
                print('roundtrip ABI', name, heard, 'WER', wer, flush=True)
    finally:
        anonymous.unlink()
    # Release the resident ABI model before the CLI loads another copy. This
    # matters for hosted RAM and is essential for the F16 CUDA VRAM budget.
    for name, item in audio_manifest['cases'].items():
        prefix = out / (primary.stem + '-' + name)
        command = [str(cli), '-m', str(primary), '-f', str(out / item['audio']), '-l', 'auto',
                   '-t', '4', '-osrt', '-of', str(prefix)]
        if not use_gpu:
            command.append('-ng')
        with prefix.with_suffix('.log').open('w') as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=3600)
        srt = prefix.with_suffix('.srt')
        cli_lines = [line.strip() for line in srt.read_text().splitlines()
                     if line.strip() and not line.strip().isdigit() and '-->' not in line] if srt.exists() else []
        case = results[name]
        abi_lines = [line.strip() for s in case['segments'] for line in s['text'].splitlines() if line.strip()]
        passed = case['valid_segments'] and case['wer'] <= .10 and result.returncode == 0 and cli_lines == abi_lines
        case.update(cli_rc=result.returncode, cli_abi_text_match=cli_lines == abi_lines, passed=passed)
        print('roundtrip', primary.stem, name, json.dumps(case, ensure_ascii=False), flush=True)
        if not passed:
            failures.append(name)
    receipt = dict(model=primary.name, gpu=use_gpu, cases=results, failed=failures,
                   source_stage_parity_checked=False, roundtrip_passed=not failures)
    (out / (primary.stem + '-roundtrip.json')).write_text(json.dumps(receipt, indent=2, ensure_ascii=False) + '\n')
    return failures


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--synthesize-only', action='store_true')
    args = parser.parse_args()
    from huggingface_hub import hf_hub_download, snapshot_download
    root = Path(__file__).resolve().parents[2]
    out = Path(os.environ['HEAVY_OUT'])
    scratch = Path(os.environ['HEAVY_SCRATCH']) / 'index-echo-roundtrip'
    out.mkdir(parents=True, exist_ok=True)
    scratch.mkdir(parents=True, exist_ok=True)
    os.environ['TMPDIR'] = str(scratch)
    build = scratch / 'build'

    def run(*command):
        print('run:', *map(str, command), flush=True)
        subprocess.run(list(map(str, command)), cwd=root, check=True)

    run('sudo', 'apt-get', 'update', '-qq')
    run('sudo', 'apt-get', 'install', '-y', '-qq', 'espeak-ng', 'ffmpeg')
    run('cmake', '-S', root, '-B', build, '-G', 'Ninja', '-DCMAKE_BUILD_TYPE=Release',
        '-DBUILD_SHARED_LIBS=ON', '-DCRISPASR_BUILD_TESTS=OFF', '-DCRISPASR_BUILD_SERVER=OFF', '-DGGML_NATIVE=OFF')
    run('cmake', '--build', build, '--target', 'crispasr-cli', 'crispasr-lib', '-j', '4')
    cli = build / 'bin/crispasr'
    piper = hf_hub_download(PIPER_REPO, 'piper-en_US-lessac-medium-f16.gguf',
                           revision=PIPER_REVISION, local_dir=scratch / 'tts')
    manifest = dict(tts_repo=PIPER_REPO, tts_revision=PIPER_REVISION, seed=1234,
                    source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(),
                    model_repo='cstr/index-echo-9b-GGUF', model_revision=MODEL_REVISION, cases={})
    for name, text in TEXTS.items():
        original, audio = out / (name + '-piper.wav'), out / (name + '.wav')
        with (out / (name + '-synthesis.log')).open('w') as log:
            subprocess.run([str(cli), '--backend', 'piper', '-m', piper, '-l', 'en',
                            '--tts', text, '--tts-output', str(original), '--seed', '1234', '-t', '4',
                            '--no-spoken-disclaimer', '--accept-marking-responsibility'],
                           stdout=log, stderr=subprocess.STDOUT, check=True, timeout=600)
        run('ffmpeg', '-v', 'error', '-y', '-i', original, '-ar', '16000', '-ac', '1', '-c:a', 'pcm_s16le', audio)
        import numpy as np
        with wave.open(str(audio), 'rb') as wav:
            seconds = wav.getnframes() / wav.getframerate()
            pcm = np.frombuffer(wav.readframes(wav.getnframes()), dtype=np.int16).astype(np.float32) / 32768
        assert .5 < seconds < 30 and np.isfinite(pcm).all() and float(np.sqrt(np.mean(pcm * pcm))) > .001
        manifest['cases'][name] = dict(text=text, audio=audio.name, seconds=seconds,
                                       sha256=hashlib.sha256(audio.read_bytes()).hexdigest())
    (out / 'roundtrip-audio.json').write_text(json.dumps(manifest, indent=2) + '\n')
    if args.synthesize_only:
        return
    models = Path(snapshot_download(manifest['model_repo'], revision=MODEL_REVISION,
                                   local_dir=scratch / 'models', allow_patterns=['*q8_0.gguf']))
    failed = check_roundtrips(root, out, cli, next(build.rglob('libcrispasr.so')),
                             models / 'index-echo-9b-q8_0.gguf', manifest)
    (out / 'summary.md').write_text(f'Index-Echo 9B Q8 CPU: {3-len(failed)}/3 Piper round-trips passed. '
                                   'Source stage/cache parity is a separate gate.\n')
    if failed:
        raise RuntimeError('TTS -> Index-Echo recognition failed: ' + ', '.join(failed))


if __name__ == '__main__':
    main()
