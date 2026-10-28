#!/usr/bin/env python3
"""Diagnose disclosure decoding with the retained full-source CUDA application.

This is a diagnostic, never a replacement for the original failed quality gate.
No model, seed, numerical code, recognition threshold or default is changed.
"""
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tarfile
import urllib.request

APPLICATION_SOURCE = '91e1e97b399d105924915c0e8d5ea0176359b147'
CHECKPOINT_SHA = '16c7b7f10141cdb8f023bbf0dcd4c274a8f9f2215a501767545666df06906d05'
CONDITIONING_SHA = '46badec89720409414dad5c61f3e5cc1c34928a3e0b8391a1cf4f92e4d90dec8'
HELPER_SOURCE = 'd84007612167105cf24c7af80014d1f0ea1bd8a3'
HELPER_SHA = 'b1fed1e4518ceeb517c749be67ce3eeee9d3243153b5f50043f6edf9c4584f20'
RECIPIENT = 'af2f46629cfa40da8c9fffdb9923877d1c4eb312d4ce3b4e36a42aa673f8ba47'
ASR = [('Systran/faster-whisper-tiny.en', '0d3d19a32d3338f10357c0889762bd8d64bbdeba', 'cpu', 'int8'),
       ('Systran/faster-whisper-small.en', 'd1d751a5f8271d482d14ca55d9e2deeebbae577f', 'cuda', 'float16')]
WORK = Path('/kaggle/temp/openvoice-disclosure')
OUT = Path('/kaggle/working')
ROOT = WORK / 'source'
PRIVATE = WORK / 'private'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def fetch(url, path, expected=None):
    with urllib.request.urlopen(url, timeout=120) as response, Path(path).open('wb') as dest:
        shutil.copyfileobj(response, dest)
    if expected and sha(path) != expected:
        raise ValueError('Immutable input checksum mismatch: ' + Path(path).name)


def main():
    WORK.mkdir(parents=True, exist_ok=True); OUT.mkdir(parents=True, exist_ok=True)
    PRIVATE.mkdir(exist_ok=True)
    helper_path = WORK / 'resources.py'
    fetch('https://raw.githubusercontent.com/CrispStrobe/CrispASR/' + HELPER_SOURCE +
          '/tools/kaggle/openvoice-bake/openvoice_bake.py', helper_path, HELPER_SHA)
    spec = importlib.util.spec_from_file_location('resources', helper_path)
    helper = importlib.util.module_from_spec(spec); spec.loader.exec_module(helper)
    helper.WORK = WORK; helper.OUT = OUT
    receipt = dict(schema=1, complete=False, application_acceptance=False,
        scope='Diagnostic only: unchanged failed full-disclosure WER gate retained',
        application_source=APPLICATION_SOURCE, checkpoint_sha256=CHECKPOINT_SHA,
        entry_sha256=sha(__file__), observations=[], resource_checks=[])
    def save():
        (OUT / 'disclosure-diagnostic.json').write_text(json.dumps(receipt, indent=2))
    helper.RECEIPT = receipt; helper.save = save
    helper.guard_resources('disclosure-bootstrap')
    if os.environ.get('CRISPASR_DISCLOSURE_READY') != '1':
        for cmd in [['git', 'init', ROOT], ['git', '-C', ROOT, 'remote', 'add', 'origin',
                     'https://github.com/CrispStrobe/CrispASR.git'],
                    ['git', '-C', ROOT, 'fetch', '--depth=1', 'origin', APPLICATION_SOURCE],
                    ['git', '-C', ROOT, 'checkout', 'FETCH_HEAD']]:
            subprocess.run(list(map(str, cmd)), check=True)
        checkpoints = list(Path('/kaggle/input').rglob('current-application-checkpoint.tar.gz'))
        if not checkpoints:
            # The operator obtains the same per-file expiring URL used by
            # Kaggle's output client. No account API token enters this worker.
            url = os.environ.pop('CRISPASR_CHECKPOINT_URL', '')
            if not url.startswith('https://'):
                raise ValueError('A scoped expiring checkpoint download URL is required')
            checkpoint = WORK/'retained-application.tar.gz'
            helper.guard_resources('retained-checkpoint-input')
            fetch(url, checkpoint, CHECKPOINT_SHA)
            checkpoints = [checkpoint]
        if len(checkpoints) != 1 or sha(checkpoints[0]) != CHECKPOINT_SHA:
            raise ValueError('Exactly the retained version-1 application is required')
        with tarfile.open(checkpoints[0]) as archive:
            archive.extractall(WORK, filter='data')
        providers = sorted({str(p) for base in [Path('/usr/local/lib'), Path('/opt/conda/lib')]
                            for p in base.glob('python*/**/nvidia/*/lib') if p.is_dir()})
        providers += [str(p) for p in [Path('/usr/local/cuda/lib64'), Path('/usr/local/nvidia/lib64')] if p.is_dir()]
        host = os.environ.get('LD_LIBRARY_PATH', '')
        env = dict(os.environ, CRISPASR_DISCLOSURE_READY='1',
                   CRISPASR_OV2_HOST_LD_LIBRARY_PATH=host,
                   LD_LIBRARY_PATH=os.pathsep.join([str(WORK/'candidate'), *providers, host]),
                   OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1', GGML_SCHED_DEBUG='1')
        os.execve(sys.executable, [sys.executable, str(Path(__file__).resolve())], env)
    sys.path.insert(0, str(ROOT / 'tools/ci-heavy'))
    from openvoice_interfaces import TEXTS, DISCLOSURE, cuda_vocoder_segments, wer
    helper.run([sys.executable, '-m', 'pip', 'install', '-q',
                'faster-whisper', 'soundfile', 'scipy', 'cryptography'], 'dependencies')
    sys.path.insert(0, str(ROOT / 'tools/validation'))
    from private_evidence import seal_directory
    try:
        candidate = WORK / 'candidate'; cli = candidate / 'crispasr'
        receipt['binary_sha256'] = sha(cli)
        helper.run(['ldd', '-r', cli], 'runtime-dependencies')
        log = (OUT / 'runtime-dependencies.log').read_text()
        if 'not found' in log or 'undefined symbol:' in log:
            raise RuntimeError('Retained application cannot load unchanged')
        loaded = re.findall(r'^\s*(lib(?:ggml\S*|crispasr\S*)\.so\S*) => (\S+)', log, re.M)
        if not loaded or any(Path(p).resolve().parent != candidate.resolve() for _, p in loaded):
            raise RuntimeError('Retained application providers were displaced')
        metadata = [p for p in Path('/kaggle/input').rglob('validation.json') if sha(p) == CONDITIONING_SHA]
        if len(metadata) != 1:
            raise ValueError('Accepted real model/reference data required')
        original = json.loads(metadata[0].read_text()); data = WORK / 'data'; data.mkdir(exist_ok=True)
        for name in ['reference.wav', 'openvoice2-tcc-f32sens.gguf',
                     'reference-ATTRIBUTION.txt', 'reference-OpenVoice-LICENSE.txt']:
            path = metadata[0].parent / name
            if sha(path) != original['artifacts'][name]['sha256']:
                raise ValueError('Accepted input checksum mismatch')
            shutil.copy2(path, data / name)
        for name in ['reference-ATTRIBUTION.txt', 'reference-OpenVoice-LICENSE.txt']:
            shutil.copy2(data / name, OUT / name)
        melo = data / 'opaque-engine.gguf'; bert = data / 'bert.gguf'
        for path, name, digest in [(melo, 'melotts-en-v2-f16.gguf', original['melo']['model_sha256']),
                                   (bert, 'bert-base-uncased-q4k.gguf', original['melo']['bert_sha256'])]:
            fetch('https://huggingface.co/cstr/melotts-en-v2-GGUF/resolve/' +
                  original['melo']['revision'] + '/' + name, path, digest)
        common = [cli, '--backend', 'melotts', '-m', melo, '--codec-model', bert,
                  '-t', '2', '--seed', '42', '--speaker-identity', 'unknown', '--gpu-backend', 'cuda']
        env = dict(os.environ, CRISPASR_OV2_GPU='1', CRISPASR_MELOTTS_BERT=str(bert))
        def native(command, label):
            helper.guard_resources(label)
            with (PRIVATE / (label + '.log')).open('w') as output:
                subprocess.run(list(map(str, command)), env=env, stdout=output,
                               stderr=subprocess.STDOUT, check=True, timeout=900)
        native(common + ['--tts', DISCLOSURE, '--tts-output', PRIVATE/'neutral.wav'], 'neutral')
        import numpy as np
        import soundfile as sf
        from scipy.signal import resample_poly
        probes = [('neutral', PRIVATE/'neutral.wav', DISCLOSURE)]
        neutral, neutral_rate = sf.read(PRIVATE/'neutral.wav', dtype='float32')
        if neutral_rate != 44100:
            raise RuntimeError('Neutral baseline rate changed')
        neutral_half = resample_poly(neutral, 1, 2)
        for i, text in enumerate(TEXTS):
            label = 'disclosed-' + str(i); path = PRIVATE/(label+'.wav')
            native(common + ['--voice', data/'reference.wav', '--i-have-rights',
                             '--tts', text, '--tts-output', path], label)
            log = (PRIVATE/(label+'.log')).read_text()
            if cuda_vocoder_segments(log) != 1 or len(re.findall(r'openvoice2: HiFi-GAN output [1-9]\d* samples', log)) != 1:
                raise RuntimeError('Actual converter CUDA or neutral bypass changed')
            sizes = re.findall(r'disclaimer synthesized \((\d+) samples @ 22050 Hz', log)
            pcm, rate = sf.read(path, dtype='float32')
            if rate != 22050 or len(sizes) != 1:
                raise RuntimeError('Disclosure boundaries/rate unavailable')
            count = int(sizes[0]); prefix = pcm[:count]; suffix = pcm[count + 6615:]
            if len(prefix) != len(neutral_half) or not len(suffix):
                raise RuntimeError('Neutral disclosure dimensions changed')
            receipt.setdefault('prefix_comparisons', []).append(dict(case=i,
                cosine=float(np.dot(prefix, neutral_half)/(np.linalg.norm(prefix)*np.linalg.norm(neutral_half))),
                norm_ratio=float(np.linalg.norm(prefix)/np.linalg.norm(neutral_half)), samples=count))
            for part, audio, expected in [('prefix', prefix, DISCLOSURE), ('content', suffix, text)]:
                dest = PRIVATE/(part+'-'+str(i)+'.wav'); sf.write(dest, audio, rate, subtype='FLOAT')
                probes.append((part+'-'+str(i), dest, expected))
            probes.append((label, path, DISCLOSURE+' '+text))
        from faster_whisper import WhisperModel
        from huggingface_hub import snapshot_download
        private_results = []
        for model, revision, device, precision in ASR:
            helper.guard_resources('load-' + model.rsplit('/', 1)[1])
            model_path = snapshot_download(model, revision=revision)
            decoder = WhisperModel(model_path, device=device, compute_type=precision, cpu_threads=2)
            for label, path, expected in probes:
                helper.guard_resources('decode-' + label)
                pcm, rate = sf.read(path, dtype='float32'); divisor = math.gcd(rate, 16000)
                audio = np.ascontiguousarray(resample_poly(pcm, 16000//divisor, rate//divisor), dtype=np.float32)
                segments, _ = decoder.transcribe(audio, language='en', beam_size=5)
                transcript = ' '.join(segment.text.strip() for segment in segments)
                row = dict(model=model, revision=revision, device=device, precision=precision,
                           probe=label, wer=wer(expected, transcript), audio_sha256=sha(path))
                receipt['observations'].append(row)
                private_results.append(dict(**row, transcript=transcript, expected=expected))
                (PRIVATE/'decoded.json').write_text(json.dumps(private_results, indent=2)); save()
            del decoder
        receipt['complete'] = True
    except Exception:
        receipt['error'] = 'Diagnostic failed; inspect retained logs and private evidence'
        raise
    finally:
        receipt['private_evidence'] = seal_directory(PRIVATE, OUT/'private-evidence.bin', RECIPIENT)
        save()


if __name__ == '__main__':
    main()
