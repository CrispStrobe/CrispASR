#!/usr/bin/env python3
"""Diagnostic CUDA MiMo full ASR: actual CLI/session readback, then warm A/B.

Original pinned Q4 LM/codec, one model lifetime at a time, same GPU and binary.
The temporary precision override applies only to encoder.* tokenizer weights;
LM arithmetic is unchanged between default and precise (both flash enabled).
The eager control disables flash in both components. No production acceptance
or recovered original-checkpoint precision is implied by this study.
"""
import argparse
import ctypes as C
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

import numpy as np
import gguf
import soundfile as sf
from huggingface_hub import hf_hub_download

from pr492_acceptance import ROOT, PINS, TOK_STAGES, digest, metrics

MODES = {'default': (False, True), 'precise': (True, True), 'eager': (True, False)}
CUBLAS = 'CRISPASR_DIAG_CUDA_Q4_CUBLAS'
TILE = 'CRISPASR_DIAG_MIMO_TILE_F32'
CODEC_PREFIXES = ('encoder.', 'enc.blk.')
# Independent official encoder + unmodified official quantizer, GH 37821379986,
# upstream 691ce54144a6844cc641fd96046a6ba20776c8b0, original pinned Q4 codec.
# Hash all 276x8 code IDs in frame-major little-endian I32 order; no tolerance.
OFFICIAL_CODES = '65da2d8bbf0338e80d3322ad88325f71c66db069767a97add8095aa891a7867b'
JFK_PCM = '0c03e80f9c348c2319823d6505d6d875a61b90fc7c817a1cc762b64981842142'


def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def session_open(build, model, codec, flash):
    sys.path.insert(0, str(ROOT / 'python'))
    from crispasr import Session

    class OpenParams(C.Structure):
        _fields_ = [(x, C.c_int) for x in ['abi_version', 'n_threads', 'use_gpu', 'verbosity',
                                          'flash_attn', 'n_gpu_layers']] + [('reserved', C.c_int * 6)]

    session = Session.__new__(Session)
    session._lib = C.CDLL(str(next(build.rglob('libcrispasr.so')).resolve(strict=True)))
    session._handle, session._progress_cb_holder = None, None
    session._setup_session_signatures()
    opts = OpenParams(2, 4, 1, 1, int(flash), -1)
    session._handle = session._lib.crispasr_session_open_with_params(
        os.fsencode(model), b'mimo-asr', C.byref(opts))
    assert session._handle, 'GPU session failed to open'
    session.backend, session._n_threads = 'mimo-asr', 4
    session.set_codec_path(str(codec))
    session.set_max_new_tokens(128)
    return session


def speech_guard(text, language):
    assert text, language
    if language == 'en':
        assert all(w in re.findall('[a-z]+', text.lower()) for w in ['americans', 'country', 'ask']), text
    else:
        assert len(re.findall('[\u4e00-\u9fff]', text)) >= 5, text


def worker(args):
    build, out, model, codec, audio = map(Path, args[:5])
    mode, phase = args[5:7]
    precise, flash = MODES[mode]
    for name in [CUBLAS, TILE, 'CRISPASR_MIMO_FORCE_CPU', 'CRISPASR_MIMO_TOK_CPU',
                 'CRISPASR_CORE_ATTN_EAGER_F32', 'CRISPASR_N_GPU_LAYERS']:
        os.environ.pop(name, None)
    if precise:
        os.environ[CUBLAS] = '1'
        if flash:
            os.environ[TILE] = '1'
    os.environ.update(CUDA_VISIBLE_DEVICES='0', NV_TF32_OVERRIDE='0', OMP_NUM_THREADS='4',
                      CRISPASR_GGUF_MMAP='1')
    out.mkdir(parents=True, exist_ok=True)
    reference = json.loads((audio / 'accepted-speech.json').read_text()) if phase == 'profile' else None
    clips = {lang: sf.read(audio / (lang + '.wav'), dtype='float32') for lang in ['en', 'zh']}
    for pcm, sr in clips.values():
        assert sr == 16000 and pcm.ndim == 1 and np.isfinite(pcm).all()
    result = dict(mode=mode, phase=phase, flash=flash, encoder_only_precision=precise,
                  speech={}, calls=[], passed=False)
    save = lambda: write(out / 'speech.json', result)
    save()
    started = time.perf_counter()
    session = session_open(build, model, codec, flash)
    result['session_open_seconds'] = time.perf_counter() - started
    # Loading the codec is lazy: warm passes below cover that cold path.
    with session:
        repetitions = 8 if phase == 'profile' else 1
        for rep in range(repetitions):
            for lang in (['en', 'zh'] if rep % 2 == 0 else ['zh', 'en']):
                session.set_source_language(lang)
                started = time.perf_counter()
                text = ' '.join(s.text for s in session.transcribe(clips[lang][0])).strip()
                elapsed = time.perf_counter() - started
                speech_guard(text, lang)
                if reference is not None:
                    assert text == reference[lang], (mode, lang, text, reference[lang])
                if lang in result['speech']:
                    assert text == result['speech'][lang]['abi'], 'Repeated decode changed'
                result['speech'][lang] = dict(abi=text, audio_sha256=digest(audio / (lang + '.wav')))
                result['calls'].append(dict(language=lang, repetition=rep, seconds=elapsed,
                                            measured=phase == 'profile' and rep >= 2, text=text))
                save()
    # Model/KV released before a separate CLI process loads the same files.
    if phase == 'accept':
        for lang in ['en', 'zh']:
            prefix = out / ('cli-' + lang)
            with (out / ('cli-' + lang + '.log')).open('w') as log:
                subprocess.run([str(build / 'bin/crispasr'), '--backend', 'mimo-asr', '-m', str(model),
                                '--codec-model', str(codec), '-t', '4', '-l', lang,
                                '-fa' if flash else '-nfa', '--max-new-tokens', '128',
                                '-f', str(audio / (lang + '.wav')), '-otxt', '-of', str(prefix)],
                               stdout=log, stderr=subprocess.STDOUT, check=True, timeout=1200)
            text = prefix.with_suffix('.txt').read_text().strip()
            result['speech'][lang]['cli'] = text
            save()
            assert text == result['speech'][lang]['abi'], (mode, lang, text, result['speech'][lang]['abi'])
    result['passed'] = True
    save()


def launch(command, logfile):
    """Real inference, with sampled device process memory (lower bound only)."""
    peak, samples = 0, 0
    started = time.monotonic()
    gpu = subprocess.check_output(['nvidia-smi', '-i', '0', '--query-gpu=uuid',
                                   '--format=csv,noheader'], text=True).strip()
    with logfile.open('w') as log:
        proc = subprocess.Popen(list(map(str, command)), stdout=log, stderr=subprocess.STDOUT)
        try:
            while proc.poll() is None:
                if time.monotonic() - started > 2400:
                    raise TimeoutError(str(logfile))
                sample = subprocess.run(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,used_memory',
                                         '--format=csv,noheader,nounits'], capture_output=True,
                                        text=True, timeout=15, check=True)
                # CUDA_VISIBLE_DEVICES=0 selects GPU0. Parent worker and its CLI
                # child never overlap model lifetimes; no other GPU job runs.
                rows = [line.split(',') for line in sample.stdout.splitlines() if line.strip()]
                memory = sum(int(row[2].strip()) for row in rows
                             if row[0].strip() == gpu and row[2].strip().isdigit())
                peak, samples = max(peak, memory), samples + 1
                time.sleep(.2)
        finally:
            if proc.poll() is None:
                proc.kill()
            proc.wait()
    assert proc.returncode == 0, f'{logfile}: rc={proc.returncode}'
    assert peak > 0 and samples > 0, 'No actual device allocation observed'
    return dict(sampled_peak_device_process_mib=peak, samples=samples,
                gpu_uuid=gpu,
                scope='GPU0 process memory sampled every >=200ms; lower bound, includes load')


def check_dispatch(logs, mode):
    text = '\n'.join(p.read_text() for p in logs)
    assert 'mimo_asr: GPU backend active' in text, 'LM silently fell back to CPU'
    assert 'mimo_tokenizer: RVQ backend=CUDA' in text, 'Codec silently fell back to CPU'
    traces = [line for line in text.splitlines() if 'MIMO_DIAG_CUDA_CUBLAS_F32' in line]
    tiles = [line for line in text.splitlines() if 'MIMO_DIAG_TILE_F32' in line]
    assert bool(traces) == (mode != 'default'), (mode, 'cuBLAS trace')
    assert bool(tiles) == (mode == 'precise'), (mode, 'TILE trace')
    assert all(re.search(r' weight=(encoder\.|enc\.blk\.)', line) for line in traces), 'LM arithmetic override detected'
    return dict(cublas=traces, tile=tiles)


def weight_inventory(path, codec):
    reader = gguf.GGUFReader(str(path))
    quantized = [t.name for t in reader.tensors if gguf.GGML_QUANT_SIZES[t.tensor_type][0] > 1]
    assert quantized, 'Missing actual quantized matrices'
    selected = [n for n in quantized if n.startswith(CODEC_PREFIXES)]
    assert (len(selected) == len(quantized)) if codec else not selected, (codec, selected)
    return dict(quantized=quantized, selected=selected)


def tokenizer_gate(build, codec, audio, out):
    """Recheck actual scoped dispatch and exact official codes in THIS full binary."""
    pcm, sr = sf.read(audio / 'en.wav', dtype='float32')
    assert sr == 16000 and hashlib.sha256(pcm.astype('<f4').tobytes()).hexdigest() == JFK_PCM
    library = next(build.rglob('libcrispasr.so')).resolve(strict=True)
    result = dict(official_codes_sha256=OFFICIAL_CODES, pcm_sha256=JFK_PCM, arms={}, passed=False)
    os.environ.update(CUDA_VISIBLE_DEVICES='0', NV_TF32_OVERRIDE='0', MIMO_TOKENIZER_DIAG_GPU='1')
    os.environ.pop('CRISPASR_MIMO_TOK_CPU', None)
    os.environ.pop('CRISPASR_MIMO_FORCE_CPU', None)
    os.environ.pop('CRISPASR_CORE_ATTN_EAGER_F32', None)
    for flash in [1, 0]:
        mode = 'precise' if flash else 'eager'
        directory = out / ('tokenizer-' + mode)
        os.environ[CUBLAS] = '1'
        if flash:
            os.environ[TILE] = '1'
        else:
            os.environ.pop(TILE, None)
        logfile = out / ('tokenizer-' + mode + '.log')
        memory = launch([sys.executable, ROOT / 'tools/ci-heavy/pr492_tokenizer_diagnose.py',
                         '--native', library, codec, audio / 'en.wav', directory, flash], logfile)
        log = logfile.read_text()
        assert 'mimo_tokenizer: RVQ backend=CUDA' in log
        traces = [line for line in log.splitlines() if 'MIMO_DIAG_CUDA_CUBLAS_F32' in line]
        assert traces and all(re.search(r' weight=(encoder\.|enc\.blk\.)', line) for line in traces)
        assert ('MIMO_DIAG_TILE_F32' in log) == bool(flash)
        codes = np.load(directory / 'tok_codes.npy')
        assert codes.size == 2208 and np.array_equal(codes, codes.astype(np.int32))
        sha = hashlib.sha256(codes.astype('<i4').tobytes()).hexdigest()
        assert sha == OFFICIAL_CODES, 'Full-library tokenizer disagrees with official encoder/RVQ'
        result['arms'][mode] = dict(codes_sha256=sha, exact_codes=2208, memory=memory, dispatch=traces)
    for name in [CUBLAS, TILE]:
        os.environ.pop(name, None)
    result['attention_ab'] = {}
    for stage in TOK_STAGES[:-1]:
        a, b = [np.load(out / ('tokenizer-' + mode) / (stage + '.npy')) for mode in ['precise', 'eager']]
        m = metrics(a, b)
        assert m['cosine'] >= .9999 and m['relative_l2'] <= .005, stage
        result['attention_ab'][stage] = m
    result['passed'] = True
    write(out / 'full-library-tokenizer.json', result)
    return result


def main():
    out = Path(os.environ['HEAVY_OUT'])
    scratch = Path(os.environ['HEAVY_SCRATCH'])
    build = Path(os.environ['MIMO_ASR_DIAG_BUILD'])
    out.mkdir(parents=True, exist_ok=True)
    audio = scratch / 'audio'
    audio.mkdir(parents=True, exist_ok=True)
    models = {}
    for key in ['q4_k', 'codec']:
        repo, rev, filename, sha = PINS[key]
        models[key] = Path(hf_hub_download(repo, filename, revision=rev, local_dir=scratch / 'models'))
        assert digest(models[key]) == sha, key
    zh = hf_hub_download('FunAudioLLM/SenseVoiceSmall', 'example/zh.mp3',
                         revision='3847d57b6bdf2dd8875cb1508d2af43d80a16bf7', local_dir=scratch / 'clips')
    for lang, source in [('en', ROOT / 'samples/jfk.mp3'), ('zh', zh)]:
        subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-i', str(source),
                        '-ar', '16000', '-ac', '1', str(audio / (lang + '.wav'))], check=True)
        (out / (lang + '.wav')).write_bytes((audio / (lang + '.wav')).read_bytes())
    receipt = dict(scope=__doc__, source=subprocess.check_output(['git', 'rev-parse', 'HEAD'],
                   cwd=ROOT, text=True).strip(), pins={k: PINS[k] for k in models},
                   bytes={k: p.stat().st_size for k, p in models.items()},
                   speech_passed=False, profile_passed=False, full_pr_acceptance=False,
                   acceptance={}, profile={})
    save = lambda: write(out / 'asr-cuda.json', receipt)
    save()
    # Positive/negative controls use actual GGUF names, including enc.blk.*;
    # production codec stem/norm names alone do not select quantized matrices.
    receipt['weight_scope'] = {key: weight_inventory(path, key == 'codec') for key, path in models.items()}
    save()
    receipt['tokenizer_gate'] = tokenizer_gate(build, models['codec'], audio, out)
    save()
    command = [sys.executable, Path(__file__).resolve(), '--worker', build, None,
               models['q4_k'], models['codec'], audio]
    # Actual full decoded-output acceptance precedes every profile request.
    for mode in MODES:
        directory = out / ('accept-' + mode)
        command[4] = directory
        memory = launch([*command, mode, 'accept'], out / ('accept-' + mode + '.log'))
        result = json.loads((directory / 'speech.json').read_text())
        assert result['passed']
        logs = [out / ('accept-' + mode + '.log'), *directory.glob('cli-*.log')]
        # Require actual device evidence and precision dispatch on EVERY surface.
        dispatch = {p.name: check_dispatch([p], mode) for p in logs}
        receipt['acceptance'][mode] = dict(result=result, memory=memory, dispatch=dispatch)
        save()
    accepted = {lang: receipt['acceptance']['default']['result']['speech'][lang]['abi'] for lang in ['en', 'zh']}
    for arm in receipt['acceptance'].values():
        for lang in ['en', 'zh']:
            assert arm['result']['speech'][lang]['abi'] == accepted[lang], 'Precision changed decoded output'
            assert arm['result']['speech'][lang]['cli'] == accepted[lang], 'CLI changed decoded output'
    receipt['speech_passed'] = True
    write(audio / 'accepted-speech.json', accepted)
    save()
    # Same binary/device/files, ABBA process order, 2 warm + 6 measured per clip.
    for index, mode in enumerate(['default', 'precise', 'precise', 'default']):
        name = f'profile-{index}-{mode}'
        directory = out / name
        command[4] = directory
        memory = launch([*command, mode, 'profile'], out / (name + '.log'))
        result = json.loads((directory / 'speech.json').read_text())
        assert result['passed']
        receipt['profile'][name] = dict(result=result, memory=memory,
                                       dispatch=check_dispatch([out / (name + '.log')], mode))
        save()
    receipt['warm_median_seconds'] = {mode: {lang: float(np.median([
        call['seconds'] for item in receipt['profile'].values() if item['result']['mode'] == mode
        for call in item['result']['calls'] if call['language'] == lang and call['measured']]))
        for lang in ['en', 'zh']} for mode in ['default', 'precise']}
    receipt['profile_passed'] = True
    save()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', nargs=7)
    args = parser.parse_args()
    if args.worker:
        worker(args.worker)
    else:
        main()
