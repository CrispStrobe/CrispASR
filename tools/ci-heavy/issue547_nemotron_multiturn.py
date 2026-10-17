#!/usr/bin/env python3
"""Issue #547 — every utterance of a `--stream --vad` session must transcribe.

Reporter (0.8.43, `--no-gpu`, nemotron-3.5-asr-streaming, f16 *and* q4_k):
in `--stream --stream-json --vad` mode the first utterance finalizes correctly,
then the next 1-2 utterances emit `{"type":"final","text":""}` with **no
preceding `partial` at all**, even though the same clips transcribe fine in file
mode and a later utterance in the same session recovers. Reordering the clips
moved which utterance failed.

Root cause (CLI-only): the partial-decode cadence throttle
(`--stream-partial-decode-ms`, default = one `--stream-step`) skipped the ASR
decode on any step shorter than its interval. The utterance state machine does
not consult that throttle, so a new utterance could still OPEN on such a step —
typically the final, partial read at EOF, or a short pipe read. The utterance
then had no text; and when its VAD span was below the redecode floor
(`kStreamRedecodeMinSamples`, 2 s) `finalize_utterance` could not re-decode it
either, so it emitted an empty `final` and silently dropped the speech.

This job builds the CLI on a hosted CPU, downloads the q4_k model + the Silero
VAD, and drives the reported recipe over four jfk.wav phrases arranged exactly so
that the LAST utterance opens on the short step. Before the fix the run is red
(one empty final, no partial, matching the report); after it every turn carries
text. It also runs the in-repo `[nemotron]` live streaming tests.

Evidence: $HEAVY_OUT/evidence.json, summary.md, and the per-step logs.
"""
import json
import os
import re
import subprocess
import sys
import wave
from pathlib import Path

import numpy as np
from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(os.environ['HEAVY_OUT'])
SCRATCH = Path(os.environ['HEAVY_SCRATCH']) / 'issue547'
OUT.mkdir(parents=True, exist_ok=True)
SCRATCH.mkdir(parents=True, exist_ok=True)
os.environ['TMPDIR'] = str(SCRATCH)
os.environ['HF_HOME'] = str(SCRATCH / 'hf')
os.environ.pop('CRISPASR_MODELS_DIR', None)
SR = 16000
results = {
    'commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
    'issue': 547,
    'cases': [],
}
(OUT / 'summary.md').write_text('# Issue #547 — nemotron-multiturn\n\n')


def save():
    (OUT / 'evidence.json').write_text(json.dumps(results, indent=2, ensure_ascii=False) + '\n')


def run(command, tag, timeout=2400, stdin=None, env=None, capture_stdout=False, check=True):
    with (OUT / (tag + '.log')).open('wb') as log:
        r = subprocess.run([str(c) for c in command], cwd=ROOT, stdin=stdin,
                           stdout=subprocess.PIPE if capture_stdout else log,
                           stderr=log, timeout=timeout, env=env)
    text = (OUT / (tag + '.log')).read_text(errors='replace')
    print(f'== {tag} rc={r.returncode} ==\n{text[-1200:]}', flush=True)
    if check:
        assert r.returncode == 0, f'{tag} exited {r.returncode}'
    if capture_stdout:
        (OUT / (tag + '.stdout')).write_bytes(r.stdout)
        return r.stdout.decode(errors='replace')
    return r.returncode if not check else text


def wav_read_mono(path):
    with wave.open(str(path), 'rb') as w:
        assert w.getframerate() == SR and w.getnchannels() == 1 and w.getsampwidth() == 2, path
        return np.frombuffer(w.readframes(w.getnframes()), dtype='<i2').astype(np.float32) / 32768.0


def wav_write_mono(path, pcm):
    with wave.open(str(path), 'wb') as w:
        w.setparams((1, 2, SR, 0, 'NONE', 'not compressed'))
        w.writeframes((np.clip(pcm, -1, 1) * 32767.0).astype('<i2').tobytes())


def words(text):
    return re.findall(r"[a-z0-9']+", text.lower())


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------
build = SCRATCH / 'build'
run(['cmake', '-S', ROOT, '-B', build, '-G', 'Ninja', '-DCMAKE_BUILD_TYPE=Release', '-DGGML_NATIVE=OFF',
     '-DGGML_BLAS=OFF', '-DCRISPASR_BUILD_SERVER=OFF', '-DCRISPASR_BUILD_TESTS=ON'], 'configure')
run(['cmake', '--build', build, '--target', 'crispasr-cli', 'test-nemotron', 'test-stream-partial-decode', '-j4'],
    'build')
cli = build / 'bin/crispasr'
test_nemotron = build / 'bin/test-nemotron'
test_partial = build / 'bin/test-stream-partial-decode'
assert cli.is_file() and test_nemotron.is_file() and test_partial.is_file()

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
model = hf_hub_download('cstr/nemotron-3.5-asr-streaming-GGUF',
                        'nemotron-3.5-asr-streaming-0.6b-q4_k.gguf', local_dir=SCRATCH / 'models')
vad = hf_hub_download('ggml-org/whisper-vad', 'ggml-silero-v6.2.0.bin', local_dir=SCRATCH / 'models')
results['model'] = {'repo': 'cstr/nemotron-3.5-asr-streaming-GGUF', 'file': Path(model).name,
                    'bytes': Path(model).stat().st_size}
save()

# ---------------------------------------------------------------------------
# Fixture: the four silence-separated jfk.wav phrases, arranged A C D B.
#
# The arrangement is the reproducer, not decoration: it makes the FINAL
# utterance the short one, so it opens on the final partial step (13.64 s total
# against a 3 s step) and trips the throttle. ABCD does not fail; the reporter's
# "reordering moves which utterance fails" is the same sensitivity.
# ---------------------------------------------------------------------------
jfk = wav_read_mono(ROOT / 'samples/jfk.wav')
PHRASES = {
    'A': (0.30, 2.28),   # "And so my fellow Americans"
    'B': (3.25, 4.43),   # "ask not"
    'C': (5.00, 7.88),   # "what your country can do for you"
    'D': (8.05, 10.98),  # "ask what you can do for your country"
}
ORDER = 'ACDB'
PAD = 0.15               # small lead/tail on each clip, like the reporter's files
GAP_S = 1.2
# Every utterance's final must carry these words (a decoded-output check, not a
# byte-match: the 0.6b model truncates the clipped phrase onsets differently
# between f16 and q4_k, so an exact string or a strict word-overlap gate would be
# flaky. These sets are satisfied by both quants and are empty never.)
EXPECT_WORDS = {
    1: ['fellow', 'americans'],
    2: ['country', 'can', 'do'],
    3: ['what', 'your', 'country'],
    4: ['ask', 'not'],
}


def phrase_pcm(key):
    a, b = PHRASES[key]
    a, b = max(0.0, a - PAD), min(len(jfk) / SR, b + PAD)
    return jfk[int(a * SR):int(b * SR)].copy()


gap = np.zeros(int(GAP_S * SR), dtype=np.float32)
utts = [phrase_pcm(k) for k in ORDER]
pieces = [u for i, u in enumerate(utts) for u in ([u, gap] if i < len(utts) - 1 else [u])]
multi = np.concatenate(pieces)
multi_pcm = SCRATCH / 'multi_turn.pcm'
multi_pcm.write_bytes((np.clip(multi, -1, 1) * 32767.0).astype('<i2').tobytes())
for i, u in enumerate(utts):
    wav_write_mono(SCRATCH / f'utt{i}.wav', u)
results['audio'] = {'order': ORDER, 'total_s': len(multi) / SR, 'gap_s': GAP_S,
                    'utterance_s': [len(u) / SR for u in utts]}
save()

# ---------------------------------------------------------------------------
# File-mode reference per clip (recorded evidence; also proves the clips decode
# alone in this exact build).
# ---------------------------------------------------------------------------
references = []
for i in range(len(utts)):
    text = run([cli, '--no-gpu', '--backend', 'nemotron', '-m', model, '-l', 'en-US', '-t', '4',
                '-f', SCRATCH / f'utt{i}.wav'], f'file-{i}', capture_stdout=True)
    lines = [ln.strip() for ln in text.splitlines() if words(ln)]
    references.append(max(lines, key=lambda ln: len(words(ln))) if lines else '')
results['file_mode'] = references
save()

# ---------------------------------------------------------------------------
# The reported recipe
# ---------------------------------------------------------------------------
with multi_pcm.open('rb') as stdin:
    stream_log = run([cli, '--no-gpu', '--backend', 'nemotron', '-m', model, '-l', 'en-US', '-t', '4',
                      '--stream', '--stream-json', '--vad', '--vad-model', vad,
                      '--stream-final-on-silence-ms', '400'],
                     'stream-cli', stdin=stdin)

events = []
for line in stream_log.splitlines():
    line = line.strip()
    if line.startswith('{'):
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            pass
finals = [e for e in events if e.get('type') == 'final']
partials = [e for e in events if e.get('type') == 'partial']
results['events'] = events
results['n_finals'] = len(finals)
results['n_partials'] = len(partials)
save()

# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------
problems = []
if len(finals) != len(utts):
    problems.append(f'expected {len(utts)} final events, got {len(finals)}')
for f in finals:
    uid = f.get('utterance_id')
    text = (f.get('text') or '').strip()
    ref = references[uid - 1] if isinstance(uid, int) and 1 <= uid <= len(references) else ''
    got = set(words(text))
    need = EXPECT_WORDS.get(uid, [])
    missing = [w for w in need if w not in got]
    results['cases'].append({'utterance_id': uid, 't0': f.get('t0'), 't1': f.get('t1'), 'text': text,
                             'reference': ref, 'need_words': need, 'missing_words': missing})
    if not text:
        # The report's exact signature: an empty final clobbers the turn.
        problems.append(f'utterance {uid} final is EMPTY (file-mode reference: {ref!r})')
        continue
    if missing:
        problems.append(f'utterance {uid} final {text!r} is missing {missing} (reference: {ref!r})')
save()

missing_events = [uid for uid in range(1, len(utts) + 1)
                  if uid not in {f.get('utterance_id') for f in finals}]
if missing_events:
    problems.append(f'no final event for utterance(s) {missing_events}')

# ---------------------------------------------------------------------------
# In-repo guards. The cadence-policy unit tests are pure and must pass; the
# live backend multi-turn test pins the encoder-cache invariant the report
# suspected. Only the named test is run — test-nemotron also carries a GPU
# stream-capture case that needs CRISPASR_TEST_STREAM_CAPTURE and a GPU, and a
# bare `[nemotron]` spec would abort on it before the CLI findings above are
# reported.
# ---------------------------------------------------------------------------
run([test_partial, '[stream-partial-decode]'], 'test-stream-partial-decode')

test_env = dict(os.environ, CRISPASR_MODEL_NEMOTRON=str(model), CRISPASR_TEST_CPU_ONLY='1',
                CRISPASR_TEST_N_THREADS='4')
rc = run([test_nemotron, 'nemotron: every turn of a multi-utterance session transcribes'],
         'test-nemotron-live', env=test_env, timeout=3000, check=False)
live_log = (OUT / 'test-nemotron-live.log').read_text(errors='replace')
if rc != 0 and 'No tests ran' not in live_log:
    problems.append(f'nemotron multi-turn live test exited {rc}')

(OUT / 'summary.md').write_text(
    '# Issue #547 — nemotron `--stream --vad` multi-turn\n\n'
    f'* fixture: jfk phrases in `{ORDER}` order, {GAP_S} s gaps, {len(multi) / SR:.2f} s\n'
    f'* finals: {len(finals)} / {len(utts)}   partials: {len(partials)}\n\n'
    + '\n'.join(f'* turn {c["utterance_id"]}: {c["text"]!r} '
                f'(ref {c["reference"]!r}, missing {c["missing_words"] or "none"})'
                for c in results['cases']) + '\n'
    + ('\n**FAILED**: ' + '; '.join(problems) + '\n' if problems else '\nAll turns transcribed.\n'))

save()
if problems:
    raise SystemExit('ISSUE 547 REPRODUCED:\n  ' + '\n  '.join(problems))
print(f'issue 547: all {len(utts)} utterances transcribed', flush=True)
