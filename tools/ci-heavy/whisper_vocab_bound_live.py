#!/usr/bin/env python3
"""Hosted CPU smoke for the real Whisper model loader's special-token bound."""

import hashlib
import json
import os
from pathlib import Path
import re
import resource
import shutil
import struct
import subprocess
import time
import urllib.request


SOURCE = 'c2e86fbce7c371cb70e88b0653119ff15b3edd63'
MODEL_REV = '5359861c739e955e79d9a303bcbc70fb988958b1'
MODEL_SHA256 = '921e4cf8686fdd993dcd081a5da5b6c365bfde1162e72b08d75ac75289920b1f'
JFK_SHA256 = '59dfb9a4acb36fe2a2affc14bacbee2920ff435cb13cc314a08c13f66ba7860e'
EXPECTED_VOCAB = 51864
INVALID_VOCAB = 50363  # tiny.en timestamp-begin ID: the new bound must reject equality.
REFERENCE = ('and so my fellow americans ask not what your country can do for you '
             'ask what you can do for your country')


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def headroom(path):
    memory = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
    state = dict(available_gib=int(memory['MemAvailable'].split()[0]) / 1024**2,
                 scratch_free_gib=shutil.disk_usage(path).free / 1024**3,
                 load_1m=os.getloadavg()[0], cpus=os.cpu_count() or 1)
    if state['available_gib'] < 3 or state['scratch_free_gib'] < 5 or state['load_1m'] > 2 * state['cpus']:
        raise RuntimeError('Insufficient hosted resources for Whisper loader smoke')
    return state


def wer(text):
    ref = re.findall(r'[a-z0-9]+', REFERENCE.lower())
    hyp = re.findall(r'[a-z0-9]+', text.lower())
    row = list(range(len(hyp) + 1))
    for i, word in enumerate(ref, 1):
        nxt = [i]
        for j, got in enumerate(hyp, 1):
            nxt.append(min(nxt[-1] + 1, row[j] + 1, row[j - 1] + (word != got)))
        row = nxt
    return row[-1] / len(ref)


def main():
    if os.environ.get('GITHUB_ACTIONS') != 'true':
        raise RuntimeError('Hosted-only: no local builds, model downloads, or inference')
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    out = Path(os.environ['HEAVY_OUT'])
    scratch = Path(os.environ['HEAVY_SCRATCH']) / 'whisper-vocab-bound-live'
    out.mkdir(parents=True, exist_ok=True)
    scratch.mkdir(parents=True, exist_ok=True)
    receipt = dict(complete=False, acceptance=False, candidate_source=SOURCE,
                   worker_sha256=sha256(__file__),
                   model_revision=MODEL_REV, model_sha256=MODEL_SHA256,
                   audio_sha256=JFK_SHA256, invalid_vocab=INVALID_VOCAB,
                   scope='Real CPU tiny.en/JFK loader and inference; one isolated header mutation; no GPU or parity claim',
                   phases={})

    def save():
        (out / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')

    def run(command, label, timeout=300, cwd=None, env=None, expect_success=True):
        receipt['phases'][label] = dict(started=True)
        save()
        print('WHISPER_VOCAB_BOUND_PHASE', label, 'start', flush=True)
        started = time.monotonic()
        try:
            result = subprocess.run(list(map(str, command)), cwd=cwd or scratch, env=env,
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                    text=True, errors='replace', timeout=timeout)
        except subprocess.TimeoutExpired as error:
            partial = error.stdout or b''
            if isinstance(partial, bytes):
                partial = partial.decode('utf-8', 'replace')
            for old, new in ((str(scratch), '<scratch>'), (str(out), '<out>')):
                partial = partial.replace(old, new)
            (out / (label + '.log')).write_text(partial)
            receipt['phases'][label] = dict(timeout_seconds=timeout,
                                            log_sha256=sha256(out / (label + '.log')))
            save()
            raise RuntimeError(f'{label} timed out') from error
        log = result.stdout
        for old, new in ((str(scratch), '<scratch>'), (str(out), '<out>')):
            log = log.replace(old, new)
        (out / (label + '.log')).write_text(log)
        receipt['phases'][label] = dict(returncode=result.returncode,
                                        seconds=round(time.monotonic() - started, 3),
                                        log_sha256=sha256(out / (label + '.log')))
        save()
        print('WHISPER_VOCAB_BOUND_PHASE', label, 'rc', result.returncode, flush=True)
        if expect_success and result.returncode:
            raise RuntimeError(f'{label} failed; see retained sanitized log')
        return result.returncode, log

    save()
    receipt['resource_checks'] = [dict(stage='admission', **headroom(scratch))]
    save()
    source = scratch / 'source'
    run(['git', 'init', source], 'git-init')
    run(['git', '-C', source, 'remote', 'add', 'origin',
         'https://github.com/CrispStrobe/CrispASR.git'], 'git-remote')
    run(['git', '-C', source, 'fetch', '--depth=1', 'origin', SOURCE], 'git-fetch', timeout=300)
    run(['git', '-C', source, 'checkout', 'FETCH_HEAD'], 'git-checkout')
    actual = subprocess.check_output(['git', '-C', source, 'rev-parse', 'HEAD'], text=True).strip()
    if actual != SOURCE:
        raise RuntimeError('Hosted source checkout differs from exact candidate pin')
    run(['git', '-C', source, 'submodule', 'update', '--init', '--recursive'],
        'git-submodules', timeout=600)
    receipt['resource_checks'].append(dict(stage='before-build', **headroom(scratch)))
    save()
    build = scratch / 'build'
    run(['cmake', '-S', source, '-B', build, '-DCMAKE_BUILD_TYPE=Release',
         '-DGGML_NATIVE=OFF', '-DGGML_CUDA=OFF', '-DGGML_METAL=OFF',
         '-DGGML_VULKAN=OFF', '-DGGML_BLAS=OFF',
         '-DCRISPASR_BUILD_TESTS=OFF', '-DCRISPASR_BUILD_SERVER=OFF'],
        'configure', timeout=600)
    receipt['resource_checks'].append(dict(stage='immediately-before-build', **headroom(scratch)))
    save()
    run(['cmake', '--build', build, '--target', 'crispasr-cli', '-j2'],
        'build', timeout=5400)
    binary = build / 'bin' / 'crispasr'
    if not binary.is_file():
        raise RuntimeError('Built CLI missing')
    receipt['cli_sha256'] = sha256(binary)
    receipt['resource_checks'].append(dict(stage='before-inference', **headroom(scratch)))
    save()
    model = scratch / 'ggml-tiny.en.bin'
    model_url = (f'https://huggingface.co/ggerganov/whisper.cpp/resolve/{MODEL_REV}/'
                 'ggml-tiny.en.bin')
    with urllib.request.urlopen(model_url, timeout=180) as stream, model.open('wb') as target:
        shutil.copyfileobj(stream, target, 1024 * 1024)
    if sha256(model) != MODEL_SHA256:
        raise RuntimeError('Pinned tiny.en model checksum mismatch')
    audio = source / 'samples/jfk.wav'
    if sha256(audio) != JFK_SHA256:
        raise RuntimeError('Pinned JFK fixture checksum mismatch')
    with model.open('rb') as stream:
        magic, vocab = struct.unpack('<Ii', stream.read(8))
    if magic != 0x67676D6C or vocab != EXPECTED_VOCAB:
        raise RuntimeError('Unexpected GGML header; mutation position not qualified')
    receipt['valid_header_vocab'] = vocab
    save()
    env = dict(os.environ, GGML_BACKEND_PATH=str(binary.parent),
               LD_LIBRARY_PATH=str(binary.parent) + os.pathsep + os.environ.get('LD_LIBRARY_PATH', ''))

    def command(path, label):
        return [binary, '--backend', 'whisper', '-m', path, '-f', audio,
                '-l', 'en', '-t', '2', '-bs', '1', '-otxt', '-of', scratch / label]

    receipt['resource_checks'].append(dict(stage='immediately-before-valid-inference', **headroom(scratch)))
    save()
    run(command(model, 'valid'), 'valid-inference', timeout=300, env=env)
    transcript = scratch / 'valid.txt'
    if not transcript.is_file() or not transcript.read_text().strip():
        raise RuntimeError('Valid model produced no transcript')
    receipt['valid_transcript_sha256'] = sha256(transcript)
    receipt['valid_wer'] = wer(transcript.read_text())
    if receipt['valid_wer'] > 0.2:
        raise RuntimeError('Valid model JFK quality control failed')
    save()
    bad = scratch / 'invalid-vocab.bin'
    shutil.copyfile(model, bad)
    with bad.open('r+b') as stream:
        stream.seek(4)
        stream.write(struct.pack('<i', INVALID_VOCAB))
    with bad.open('rb') as stream:
        assert struct.unpack('<Ii', stream.read(8)) == (magic, INVALID_VOCAB)
    if bad.stat().st_size != model.stat().st_size:
        raise RuntimeError('Header mutation changed model size')
    receipt['mutated_model_sha256'] = sha256(bad)
    receipt['resource_checks'].append(dict(stage='immediately-before-invalid-load', **headroom(scratch)))
    save()
    rc, log = run(command(bad, 'invalid'), 'invalid-load', timeout=120,
                  env=env, expect_success=False)
    if rc != 3 or not re.search(
        r'invalid model \(special token outside vocabulary of 50363\)', log
    ) or 'error: failed to initialize whisper context' not in log or (scratch / 'invalid.txt').exists():
        raise RuntimeError('Malformed model was not cleanly rejected at the special-token bound')
    receipt['invalid_model_returncode'] = rc
    receipt['complete'] = receipt['acceptance'] = True
    save()


if __name__ == '__main__':
    main()
