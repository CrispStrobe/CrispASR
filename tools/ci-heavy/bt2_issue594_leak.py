#!/usr/bin/env python3
"""#594: bt2-tts speaks its style instruction on some inputs.

Replays the reporter's exact server requests (q4_k model, Piedmont voice,
instruction, seed 87, cfg 1.0, server sentence chunking) through the native
server built from --before-ref and from HEAD, transcribes every output with an
independent recognizer and counts instruction n-grams that reach the audio.

Args: --before-ref SHA  [--indices 46,66,...|all]  [--arms before,after]
The reporter's attachments are downloaded and hash-pinned at run time.
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
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(os.environ['HEAVY_OUT'])
SCRATCH = Path(os.environ['HEAVY_SCRATCH']) / 'bt2-594'
RATE = 24000
ATTEST = 'I have the speaker\'s consent'
ATTACH = {
    'report': ('https://github.com/user-attachments/files/34869659/tts_junk_report.json',
               '61f71b2d4f45c66770c51f16c31fdb9a42a4478af03f4d072afcc9f4ba4c3b58'),
    'voice': ('https://github.com/user-attachments/files/34868690/Piedmont.wav',
              'fbd0edcdd3d2cc93493ca1c85c37e35b4598744ea82ee3e4e0b6d0ecd5d1fed6'),
}
PINS = [
    ('breeze', 'cstr/breeze-tts-2-GGUF', '5314fe9ae3ec6eac1a9238c4f6888fdc9fda0e83', 'breeze-tts-2-q4_k.gguf',
     '133eceae5030edab9a8569524181d92ba406fc14ea8c3861a61a3ee177c387d7'),
    ('codec', 'cstr/qwen3-tts-tokenizer-12hz-GGUF', '344e279551177654762e5c7d39e217834325f222',
     'qwen3-tts-tokenizer-12hz.gguf', '70dc95dbfdd9aa5d9d406236ff771d061bf17b0cda02a72513953355606e719b'),
]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def run(cmd, label, cwd=ROOT):
    with (OUT / (label + '.log')).open('w') as log:
        r = subprocess.run(list(map(str, cmd)), cwd=cwd, stdout=log, stderr=subprocess.STDOUT, timeout=7200)
    print(label, 'rc', r.returncode, flush=True)
    assert r.returncode == 0, label


def words(s):
    return re.findall("[a-z0-9]+", s.lower())


def wer(ref, hyp):
    t, h = words(ref), words(hyp)
    prev = list(range(len(h) + 1))
    for i, w in enumerate(t, 1):
        row = [i]
        for j, o in enumerate(h, 1):
            row.append(min(row[-1] + 1, prev[j] + 1, prev[j - 1] + (w != o)))
        prev = row
    return prev[-1] / max(1, len(t))


def leak(instruction, reference, hyp, n=3):
    """Instruction word n-grams heard that are not in the reference text."""
    grams = lambda ws: {tuple(ws[i:i + n]) for i in range(len(ws) - n + 1)}
    ins = grams(words(instruction)) - grams(words(reference))
    return sorted(' '.join(g) for g in grams(words(hyp)) & ins)


def build(src, name):
    b = SCRATCH / ('build-' + name)
    run(['cmake', '-S', src, '-B', b, '-G', 'Ninja', '-DCMAKE_BUILD_TYPE=Release', '-DBUILD_SHARED_LIBS=ON',
         '-DGGML_NATIVE=OFF', '-DGGML_OPENBLAS=ON', '-DGGML_CUDA=OFF', '-DGGML_VULKAN=OFF', '-DGGML_METAL=OFF',
         '-DCRISPASR_BUILD_TESTS=OFF', '-DCRISPASR_BUILD_SERVER=ON'], 'configure-' + name)
    run(['cmake', '--build', b, '--target', 'crispasr-cli', 'crispasr-lib', '-j4'], 'build-' + name)
    return b / 'bin/crispasr', b / 'src/libcrispasr.so'


def main():
    import numpy as np
    from huggingface_hub import hf_hub_download
    ap = argparse.ArgumentParser()
    ap.add_argument('--before-ref', required=True)
    ap.add_argument('--indices', default='46,66,206,233,384,163')
    ap.add_argument('--arms', default='before,after')
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    SCRATCH.mkdir(parents=True, exist_ok=True)
    os.environ.update(TMPDIR=str(SCRATCH), OPENBLAS_NUM_THREADS='1')
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    rec = dict(source=head, before_ref=a.before_ref, arms={}, complete=False)
    save = lambda: (OUT / 'validation.json').write_text(json.dumps(rec, indent=2) + '\n')

    inputs = SCRATCH / 'inputs'
    inputs.mkdir(exist_ok=True)
    files = {}
    for k, (url, digest) in ATTACH.items():
        p = inputs / url.rsplit('/', 1)[1]
        urllib.request.urlretrieve(url, p)
        assert sha(p) == digest, k
        files[k] = p
    report = json.loads(files['report'].read_text())
    voices = SCRATCH / 'voices'
    voices.mkdir(exist_ok=True)
    (voices / 'Piedmont.wav').write_bytes(files['voice'].read_bytes())
    # The reporter's voice transcript, so the clone prompt does not depend on
    # which recognizer the server would pick for auto-transcription.
    (voices / 'Piedmont.txt').write_text(report['transcript'] + '\n')
    rows = report['inputs'] if a.indices == 'all' else \
        [r for r in report['inputs'] if r['index'] in {int(x) for x in a.indices.split(',')}]
    rec['indices'] = [r['index'] for r in rows]

    run(['sudo', 'bash', 'tools/ci-apt.sh', 'update'], 'apt-update')
    run(['sudo', 'bash', 'tools/ci-apt.sh', 'install', '-y', 'cmake', 'ninja-build', 'g++', 'libopenblas-dev'],
        'apt-install')
    srcs = {'after': ROOT}
    if 'before' in a.arms:
        bsrc = SCRATCH / 'before-src'
        run(['git', 'fetch', '--depth=50', 'origin', a.before_ref], 'fetch-before')
        run(['git', 'worktree', 'add', '--detach', bsrc, a.before_ref], 'worktree-before')
        run(['git', 'submodule', 'update', '--init', '--recursive'], 'submodule-before', cwd=bsrc)
        srcs['before'] = bsrc
    models = SCRATCH / 'models'
    paths = {}
    for label, repo, rev, fn, digest in PINS:
        p = Path(hf_hub_download(repo, fn, revision=rev, local_dir=models))
        assert sha(p) == digest, label
        paths[label] = p
    rec['models'] = {l: dict(repo=r, revision=v, file=f, sha256=d) for l, r, v, f, d in PINS}
    save()

    port = 11594
    for arm in a.arms.split(','):
        binary, lib = build(srcs[arm], arm)
        rec['arms'][arm] = dict(cases={})
        log = (OUT / ('server-' + arm + '.log')).open('w')
        proc = subprocess.Popen([str(x) for x in [
            binary, '--server', '--backend', 'bt2-tts', '-m', paths['breeze'], '--codec-model', paths['codec'],
            '--port', port, '--voice-dir', voices, '--accept-license', 'other', '--i-have-rights',
            '--speaker-identity', 'synthetic', '--no-watermark', '--accept-marking-responsibility',
            '--no-spoken-disclaimer', '--no-gpu', '-t', '4']], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
            env=dict(os.environ, CRISPASR_ACCEPT_LICENSE='1'))
        url = f'http://127.0.0.1:{port}'
        try:
            dl = time.monotonic() + 900
            while True:
                assert proc.poll() is None, 'server exited'
                try:
                    with urllib.request.urlopen(url + '/health', timeout=2):
                        break
                except (urllib.error.URLError, OSError):
                    assert time.monotonic() < dl, 'server start timeout'
                    time.sleep(2)
            for r in rows:
                body = dict(cfg_scale=1.0, consent_attestation=ATTEST, input=r['reference'],
                            instructions=report['instructions'], model='bt2-tts', response_format='f32',
                            seed=report['seed'], speaker_identity='synthetic', spoken_disclaimer=False,
                            stream=False, voice=report['voice'])
                req = urllib.request.Request(url + '/v1/audio/speech', data=json.dumps(body).encode(),
                                             headers={'Content-Type': 'application/json'})
                t0 = time.monotonic()
                with urllib.request.urlopen(req, timeout=5400) as resp:
                    pcm = np.frombuffer(resp.read(), dtype='<f4').copy()
                name = f'{arm}-{r["index"]:06d}'
                np.save(OUT / (name + '.npy'), pcm)
                rec['arms'][arm]['cases'][str(r['index'])] = dict(
                    seconds=len(pcm) / RATE, wall=time.monotonic() - t0, chars=len(r['reference']),
                    pcm_sha256=sha(OUT / (name + '.npy')))
                print(name, len(pcm) / RATE, 's', flush=True)
                save()
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
            log.close()
        rec['arms'][arm]['library'] = str(lib)

    sys.path.insert(0, str(ROOT / 'python'))
    from crispasr import Session
    asr = Path(hf_hub_download('cstr/nemotron-3.5-asr-streaming-GGUF', 'nemotron-3.5-asr-streaming-0.6b-f16.gguf',
                               revision='bbd95a9ca5fa0dfca3312a122dfc45a2b578b9c2'))
    assert sha(asr) == '14cbc98df495d8ab452c6a4ac367136d94ab5da55e73afaa96ba4f749dd95d09'
    lib = rec['arms'][a.arms.split(',')[-1]]['library']
    ref = {str(r['index']): r['reference'] for r in rows}
    with Session(str(asr), lib_path=lib, backend='nemotron', n_threads=4) as s:
        for arm, d in rec['arms'].items():
            for idx, c in d['cases'].items():
                pcm = np.load(OUT / f'{arm}-{int(idx):06d}.npy')
                hyp = ' '.join(seg.text for seg in s.transcribe(pcm, sample_rate=RATE, language='en'))
                c.update(transcript=hyp, wer=wer(ref[idx], hyp),
                         leaked=leak(report['instructions'], ref[idx], hyp))
                print(arm, idx, 'WER %.3f' % c['wer'], 'leak', len(c['leaked']), flush=True)
                save()
    for arm, d in rec['arms'].items():
        cs = d['cases'].values()
        d['n_leaking'] = sum(1 for c in cs if len(c['leaked']) >= 2)
        d['mean_wer'] = sum(c['wer'] for c in cs) / max(1, len(cs))
    rec['complete'] = True
    save()
    lines = ['| arm | cases | leaking (>=2 instruction 3-grams) | mean WER |', '|---|---|---|---|']
    lines += [f'| {k} | {len(d["cases"])} | {d["n_leaking"]} | {d["mean_wer"]:.3f} |' for k, d in rec['arms'].items()]
    (OUT / 'summary.md').write_text('#594 bt2-tts instruction leak replay\n\n' + '\n'.join(lines) + '\n')
    if 'after' in rec['arms']:
        assert rec['arms']['after']['n_leaking'] == 0, 'instruction still leaks after the fix'


if __name__ == '__main__':
    main()
