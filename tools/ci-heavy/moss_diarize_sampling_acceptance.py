#!/usr/bin/env python3
"""#564: MOSS-Transcribe-Diarize session beam size and optional sampling, hosted CPU.

Real model, real CLI and real Python Session over the C ABI. Checks:
  greedy  — default and explicit temperature 0 give identical text, which still
            contains the JFK sentence; do_sample=False overrides a temperature.
  sample  — temperature 0.7 with a fixed seed reproduces itself (CLI and Session)
            and keeps the transcript recognizable.
  beam    — beam_size 2 reaches the decoder: with beam active, two different
            sampling seeds give identical output (sampling is ignored only when
            beam search really runs), and the "sampling settings ignored" notice
            appears.
No GPU, speed or broad accuracy claim.
"""
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import wave

import numpy as np
from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(os.environ['HEAVY_OUT'])
SCRATCH = Path(os.environ['HEAVY_SCRATCH']) / 'moss564'
OUT.mkdir(parents=True, exist_ok=True)
SCRATCH.mkdir(parents=True, exist_ok=True)
os.environ['HF_HOME'] = str(SCRATCH / 'hf')
results = {'commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
           'scope': '#564 CPU acceptance of session/CLI beam + sampling forwarding', 'cases': []}
JFK = 'and so my fellow americans ask not what your country can do for you ask what you can do for your country'


def save():
    (OUT / 'acceptance.json').write_text(json.dumps(results, indent=2, ensure_ascii=False) + '\n')


def record(name, ok, **detail):
    results['cases'].append(dict(name=name, ok=bool(ok), **detail))
    save()
    print(('PASS ' if ok else 'FAIL ') + name, detail, flush=True)
    assert ok, name


def run(cmd, tag, timeout=3600):
    log = OUT / (tag + '.log')
    with log.open('w') as f:
        r = subprocess.run(list(map(str, cmd)), cwd=ROOT, stdin=subprocess.DEVNULL, stdout=f,
                           stderr=subprocess.STDOUT, timeout=timeout)
    text = log.read_text(errors='replace')
    assert r.returncode == 0, (tag, text[-1500:])
    return text


def words(text):
    # Drop MOSS speaker/timestamp tags, keep lowercase words.
    text = re.sub(r'\[[^\]]*\]', ' ', text)
    return re.findall(r"[a-z']+", text.lower())


def overlap(text):
    ref = JFK.split()
    got = set(words(text))
    return sum(w in got for w in ref) / len(ref)


build = SCRATCH / 'build'
run(['cmake', '-S', ROOT, '-B', build, '-G', 'Ninja', '-DCMAKE_BUILD_TYPE=Release', '-DBUILD_SHARED_LIBS=ON',
     '-DGGML_NATIVE=OFF', '-DCRISPASR_BUILD_SERVER=OFF', '-DCRISPASR_BUILD_TESTS=ON'], 'configure')
run(['cmake', '--build', build, '--target', 'crispasr-cli', 'crispasr-lib', 'test-core-logit-sampler', '-j4'],
    'build')
run([build / 'bin/test-core-logit-sampler'], 'unit-sampler')

model = hf_hub_download('cstr/MOSS-Transcribe-Diarize-GGUF', 'moss-transcribe-diarize-0.9b-q4_k.gguf',
                        local_dir=SCRATCH / 'model')
results['model'] = dict(repo='cstr/MOSS-Transcribe-Diarize-GGUF', file=Path(model).name,
                        bytes=Path(model).stat().st_size)
wav = ROOT / 'samples/jfk.wav'
cli = build / 'bin/crispasr'


def cli_run(tag, *extra):
    # Text comes from the -otxt file; stderr (kept in the log) carries notices.
    log = run([cli, '--backend', 'moss-diarize', '-m', model, '-f', wav, '-t', '4', '--no-gpu',
               '-otxt', '-of', OUT / tag, *extra], tag)
    return (OUT / (tag + '.txt')).read_text(encoding='utf-8').strip(), log


g0, _ = cli_run('cli-greedy')
g1, _ = cli_run('cli-greedy-t0', '--temperature', '0')
record('cli greedy default == explicit temperature 0', g0 == g1, text=g0)
record('cli greedy transcript recognizable', overlap(g0) >= 0.9, overlap=overlap(g0))
s1, _ = cli_run('cli-sample-a', '--temperature', '0.7', '--seed', '7')
s2, _ = cli_run('cli-sample-b', '--temperature', '0.7', '--seed', '7')
record('cli sampling reproducible with fixed seed', s1 == s2, text=s1)
record('cli sampled transcript recognizable', overlap(s1) >= 0.8, overlap=overlap(s1))
b1, blog = cli_run('cli-beam-a', '--beam-size', '2', '--temperature', '0.7', '--seed', '7')
b2, _ = cli_run('cli-beam-b', '--beam-size', '2', '--temperature', '0.7', '--seed', '8')
record('cli beam active ignores sampling seed', b1 == b2, text=b1)
record('cli beam notice printed', 'sampling settings ignored' in blog)

sys.path.insert(0, str(ROOT / 'python'))
os.environ['CRISPASR_LIB_PATH'] = str(next((build / 'src').glob('libcrispasr.so')))
import crispasr  # noqa: E402

with wave.open(str(wav)) as f:
    pcm = np.frombuffer(f.readframes(f.getnframes()), '<i2').astype(np.float32) / 32768.0


def sess_text(configure):
    s = crispasr.Session(model, n_threads=4, use_gpu=False, verbosity=1)
    try:
        configure(s)
        return ' '.join(seg.text for seg in s.transcribe(pcm)).strip()
    finally:
        s.close()


sg = sess_text(lambda s: None)
record('session greedy recognizable', overlap(sg) >= 0.9, text=sg)
sd = sess_text(lambda s: (s.set_temperature(0.7, seed=7), s.set_do_sample(False)))
record('session do_sample=False forces greedy', sd == sg, text=sd)
sa = sess_text(lambda s: (s.set_temperature(0.7, seed=7), s.set_top_k(50), s.set_top_p(0.95)))
sb = sess_text(lambda s: (s.set_temperature(0.7, seed=7), s.set_top_k(50), s.set_top_p(0.95)))
record('session sampling reproducible with fixed seed', sa == sb, text=sa)
record('session sampled transcript recognizable', overlap(sa) >= 0.8, overlap=overlap(sa))
ba = sess_text(lambda s: (s.set_beam_size(2), s.set_temperature(0.7, seed=7)))
bb = sess_text(lambda s: (s.set_beam_size(2), s.set_temperature(0.7, seed=8)))
record('session beam_size reaches decoder (sampling seed ignored)', ba == bb, text=ba)
record('session beam transcript recognizable', overlap(ba) >= 0.9, overlap=overlap(ba))

results['complete'] = True
save()
(OUT / 'summary.md').write_text('#564 MOSS-Diarize beam/sampling acceptance: all %d checks PASS\n'
                                % len(results['cases']))
