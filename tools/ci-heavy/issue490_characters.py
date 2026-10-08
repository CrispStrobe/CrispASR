#!/usr/bin/env python3
"""#490: real Arabic Q4 CTC label spans through CLI, C ABI and Python.

This tests post-logit Viterbi/output behavior. It does not claim new model-stage
parity or human-annotated phonetic boundary accuracy.
"""
import base64
import ctypes as C
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import urllib.request

import gguf
from huggingface_hub import hf_hub_download
import numpy as np
import pyarrow.parquet as pq
import soundfile as sf

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(os.environ['HEAVY_OUT'])
SCRATCH = Path(os.environ['HEAVY_SCRATCH']) / 'issue490'
OUT.mkdir(parents=True, exist_ok=True)
SCRATCH.mkdir(parents=True, exist_ok=True)
os.environ['TMPDIR'] = str(SCRATCH)
os.environ['HF_HOME'] = str(SCRATCH / 'hf')
os.environ.pop('CRISPASR_ALIGN_NO_ROMANIZE', None)
os.environ.pop('CRISPASR_ALIGN_SENTINEL_REDISTRIBUTE', None)
receipt = {'commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
           'passed': False, 'scope': __doc__}


def save():
    (OUT / 'acceptance.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n')


def run(command, tag):
    with (OUT / (tag + '.log')).open('w') as log:
        result = subprocess.run(list(map(str, command)), cwd=ROOT, stdin=subprocess.DEVNULL,
                                stdout=log, stderr=subprocess.STDOUT, timeout=2400)
    text = (OUT / (tag + '.log')).read_text()
    print(tag, result.returncode, text[-1200:], flush=True)
    assert result.returncode == 0, tag
    return text


def pinned(repo, name, revision, digest, repo_type='model'):
    path = Path(hf_hub_download(repo, name, revision=revision, repo_type=repo_type, local_dir=SCRATCH / repo))
    with path.open('rb') as f:
        assert hashlib.file_digest(f, 'sha256').hexdigest() == digest, name
    return path


def reference_spans(logits, words, vocab, blank, frame_seconds):
    """Independent full-sequence Viterbi on raw scores (row shifts cancel).

    Keep each label occurrence distinct, including adjacent identical letters.
    The data are the actual captured model logits, not the native traceback.
    """
    reverse = {token: i for i, token in enumerate(vocab)}
    labels, occurrences = [], []
    for wi, word in enumerate(words):
        if wi and '|' in reverse:
            labels.append(reverse['|'])
            occurrences.append(None)
        for cp in word:
            key = cp.lower() if cp.isascii() else cp
            token = reverse.get(key, reverse.get(cp))
            if token is not None and token != blank:
                labels.append(token)
                occurrences.append((wi, cp))
    sequence = np.full(2 * len(labels) + 1, blank, dtype=np.int32)
    sequence[1::2] = labels
    score = np.full(len(sequence), -np.inf)
    score[0], score[1] = logits[0, blank], logits[0, sequence[1]]
    back = np.zeros((len(logits), len(sequence)), dtype=np.int8)
    for t in range(1, len(logits)):
        candidates = np.stack((score, np.r_[-np.inf, score[:-1]], np.r_[[-np.inf, -np.inf], score[:-2]]))
        allowed = np.zeros(len(sequence), dtype=bool)
        allowed[2:] = (sequence[2:] != blank) & (sequence[2:] != sequence[:-2])
        candidates[2, ~allowed] = -np.inf
        back[t] = np.argmax(candidates, axis=0)
        score = candidates.max(axis=0) + logits[t, sequence]
    state = len(sequence) - 1 if score[-1] >= score[-2] else len(sequence) - 2
    assert np.isfinite(score[state]), 'Reference has no complete path'
    path = np.zeros(len(logits), dtype=np.int32)
    for t in range(len(logits) - 1, -1, -1):
        path[t] = state
        if t:
            state -= int(back[t, state])
    output = [[] for _ in words]
    for li, occurrence in enumerate(occurrences):
        if occurrence is None:
            continue
        frames = np.flatnonzero(path == 2 * li + 1)
        assert len(frames)
        wi, cp = occurrence
        output[wi].append({'char': cp, 'start': round(frames[0] * frame_seconds, 2),
                          'end': round((frames[-1] + 1) * frame_seconds, 2)})
    return output


save()
build = SCRATCH / 'build'
run(['cmake', '-S', ROOT, '-B', build, '-G', 'Ninja', '-DCMAKE_BUILD_TYPE=Release',
     '-DBUILD_SHARED_LIBS=ON', '-DGGML_NATIVE=OFF', '-DGGML_BLAS=OFF',
     '-DCRISPASR_BUILD_SERVER=OFF', '-DCRISPASR_BUILD_TESTS=ON'], 'configure')
run(['cmake', '--build', build, '--target', 'crispasr-cli', 'crispasr-lib', 'test-align-characters',
     'test-align-only', '-j4'], 'build')
run([build / 'bin/test-align-characters'], 'known-paths')
run([build / 'bin/test-align-only', '[unit]'], 'existing-align-units')

model = pinned('cstr/wav2vec2-large-xlsr-53-arabic-GGUF', 'wav2vec2-large-xlsr-53-arabic-q4_k.gguf',
               '869ad5c1519d683b534400996836611d35719af9',
               'd1b186a975a1e3506ff61ca90c301caeadfc78c1f483f4b3cd2d899409ce1be7')
parquet = pinned('google/fleurs', 'parquet-data/ar_eg/test-00000-of-00001.parquet',
                 '70bb2e84b976b7e960aa89f1c648e09c59f894dd',
                 '3386d58ff9619c2b1395dcaed0a476363fae15420bc68b7ca4870cffdff9936d', 'dataset')
row = next(pq.ParquetFile(parquet).iter_batches(batch_size=1)).to_pylist()[0]
text = row['transcription']
assert row['id'] == 1993 and len(text) > 50, row
pcm, sr = sf.read(io.BytesIO(row['audio']['bytes']), dtype='float32')
assert sr == 16000 and pcm.ndim == 1
wav = OUT / 'arabic.wav'
sf.write(wav, pcm, sr, subtype='PCM_16')
pcm, _ = sf.read(wav, dtype='float32')
(OUT / 'transcript.txt').write_text(text + '\n')
receipt.update(audio_sha256=hashlib.sha256(wav.read_bytes()).hexdigest(), transcript=text,
               model_sha256='d1b186a975a1e3506ff61ca90c301caeadfc78c1f483f4b3cd2d899409ce1be7',
               dataset_revision='70bb2e84b976b7e960aa89f1c648e09c59f894dd', dataset_row_id=1993)
save()
cli = build / 'bin/crispasr'
common = [cli, '--align-only', '-am', model, '-f', wav, '--text-file', OUT / 'transcript.txt', '-t', '4', '-ng']
run([*common, '--align-format', 'json', '--align-output', OUT / 'words.json'], 'cli-words')
run([*common, '--align-format', 'json', '--align-granularity', 'segment',
     '--align-output', OUT / 'segments.json'], 'cli-segments')
words = json.loads((OUT / 'words.json').read_text())
segments = json.loads((OUT / 'segments.json').read_text())
assert segments[0]['words'] == words
assert all(w.get('characters') for w in words)
assert len(words) == len(text.split())
assert sum(len(w['characters']) for w in words) > 50

sys.path.insert(0, str(ROOT / 'python'))
from crispasr import Session, align_words
library = next(build.rglob('libcrispasr.so'))
with Session(str(model), lib_path=str(library), n_threads=4) as session:
    recognized, logits = session.transcribe_with_logits(pcm, language='ar')
assert logits is not None and logits.shape[1] > 20
recognized_text = ' '.join(s.text for s in recognized)
assert len(set(recognized_text.split()) & set(text.split())) >= 3, recognized_text
receipt['recognized_text'] = recognized_text
reader = gguf.GGUFReader(str(model))
vocab = reader.fields['tokenizer.ggml.tokens'].contents()
blank = int(reader.fields['wav2vec2.pad_token_id'].contents())
strides = [int(reader.fields[f'wav2vec2.conv_stride_{i}'].contents()) for i in range(7)]
ref = reference_spans(logits, text.split(), vocab, blank, np.prod(strides) / 16000.0)
for actual, expected in zip(words, ref):
    assert actual['characters'] == expected, (actual, expected)
(OUT / 'independent-reference.json').write_text(json.dumps(ref, ensure_ascii=False, indent=2) + '\n')
np.save(OUT / 'logits.npy', logits)

aligned = align_words(str(model), text, pcm, t_offset=3.25, n_threads=4, lib_path=str(library))
assert len(aligned) == len(words)
for actual, expected in zip(aligned, words):
    assert actual.text == expected['word']
    assert abs(actual.start - expected['start'] - 3.25) < 1e-8
    assert abs(actual.end - expected['end'] - 3.25) < 1e-8
    assert len(actual.characters) == len(expected['characters'])
    for char, target in zip(actual.characters, expected['characters']):
        assert char.text == target['char']
        assert abs(char.start - target['start'] - 3.25) < 1e-8
        assert abs(char.end - target['end'] - 3.25) < 1e-8

lib = C.CDLL(str(library))
for suffix, restype, nargs in [('n_characters', C.c_int, 2), ('character_text', C.c_char_p, 3),
                             ('character_t0', C.c_int64, 3), ('character_t1', C.c_int64, 3)]:
    fn = getattr(lib, 'crispasr_align_result_' + suffix)
    fn.restype, fn.argtypes = restype, [C.c_void_p] + [C.c_int] * (nargs - 1)
    assert fn(None, -1, *([-1] if nargs == 3 else [])) == (b'' if restype == C.c_char_p else 0)
# Exercise the Java/JNA wrapper against the same native library and real audio.
# Pin the dependency already declared by bindings/java/build.gradle.
jna = SCRATCH / 'jna-5.13.0.jar'
with urllib.request.urlopen('https://repo.maven.apache.org/maven2/net/java/dev/jna/jna/5.13.0/jna-5.13.0.jar') as response:
    jna.write_bytes(response.read())
assert hashlib.sha256(jna.read_bytes()).hexdigest() == '66d4f819a062a51a1d5627bffc23fac55d1677f0e0a1feba144aabdd670a64bb'
classes = SCRATCH / 'java-classes'
classes.mkdir(exist_ok=True)
run(['javac', '-encoding', 'UTF-8', '-cp', jna, '-d', classes,
     ROOT / 'bindings/java/src/main/java/io/github/ggerganov/whispercpp/CrispasrSession.java',
     ROOT / 'tools/ci-heavy/Issue490Characters.java'], 'java-compile')
java_output = OUT / 'java-spans.tsv'
run(['java', '-Dfile.encoding=UTF-8', '-Djna.library.path=' + str(library.parent),
     '-cp', str(classes) + os.pathsep + str(jna), 'Issue490Characters',
     model, wav, OUT / 'transcript.txt', java_output], 'java-alignment')
expected_rows = []
for word in words:
    expected_rows.append(('W', word['word'], round(word['start'] * 100) + 325, round(word['end'] * 100) + 325))
    expected_rows.extend(('C', cp['char'], round(cp['start'] * 100) + 325, round(cp['end'] * 100) + 325)
                         for cp in word['characters'])
actual_rows = []
for line in java_output.read_text().splitlines():
    kind, encoded, start, end = line.split('\t')
    actual_rows.append((kind, base64.b64decode(encoded).decode('utf-8'), int(start), int(end)))
assert actual_rows == expected_rows, (actual_rows, expected_rows)
receipt['java_jna'] = 'exact word/character text and centisecond times with offset 325'

receipt.update(passed=True, words=len(words), characters=sum(map(len, ref)),
               independent_viterbi='exact character text/start/end', cli_word_segment_json_equal=True,
               python_cabi_offset_seconds=3.25, invalid_cabi_accessors='passed')
save()
(OUT / 'summary.md').write_text('Arabic Q4 character alignment PASS: exact independent post-logit Viterbi, CLI word/segment JSON, C ABI/Python offsets and invalid handles.\n')
print('ARABIC_CTC_CHARACTER_ACCEPTANCE_PASS', flush=True)
