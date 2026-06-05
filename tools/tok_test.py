#!/usr/bin/env python
# Quick comparison of expected GPT-2 BPE tokens vs C++ tokenize_text_bpe logic
import json
from pathlib import Path

vocab_path = '/Volumes/backups/ai/huggingface-hub/models--ResembleAI--chatterbox-turbo/vocab.json'
merges_path = '/Volumes/backups/ai/huggingface-hub/models--ResembleAI--chatterbox-turbo/merges.txt'

vocab = json.loads(Path(vocab_path).read_text())
merges = [line for line in Path(merges_path).read_text().splitlines() if line and not line.startswith('#')]
merge_rank = {tuple(line.split()): i for i, line in enumerate(merges)}

# GPT-2 byte-level encoding: bytes that aren't printable get mapped to special unicode
def gpt2_bytes_to_unicode():
    bs = list(range(ord("!"), ord("~")+1)) + list(range(ord("\xa1"), ord("\xac")+1)) + list(range(ord("\xae"), ord("\xff")+1))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b); cs.append(2**8 + n); n += 1
    return dict(zip(bs, [chr(c) for c in cs]))

be = gpt2_bytes_to_unicode()

def bytes_to_unicode(s):
    return ''.join(be[b] for b in s.encode('utf-8'))

def bpe(piece):
    word = list(piece)
    if not word: return []
    pairs = lambda w: set(zip(w[:-1], w[1:]))
    while True:
        ps = pairs(word)
        if not ps: break
        bigram = min(ps, key=lambda p: merge_rank.get(p, 10**9))
        if bigram not in merge_rank: break
        a, b = bigram
        new = []
        i = 0
        while i < len(word):
            if i+1 < len(word) and word[i]==a and word[i+1]==b:
                new.append(a+b); i += 2
            else:
                new.append(word[i]); i += 1
        word = new
    return word

def tokenize_correct(text):
    """GPT-2 pre-tokenizer: ' ?\\w+' chunks."""
    chunks = []
    i = 0
    while i < len(text):
        start = i
        if text[i] == ' ':
            i += 1
        while i < len(text) and text[i] != ' ':
            i += 1
        chunks.append(text[start:i])
    return [(c, bytes_to_unicode(c), bpe(bytes_to_unicode(c))) for c in chunks if c]

def tokenize_cpp_bug(text):
    """C++ tokenize_text_bpe — trailing-space-on-prev-word bug."""
    chunks = []
    buf = ''
    for i, ch in enumerate(text):
        if i > 0 and text[i] != ' ' and text[i-1] == ' ':
            if buf:
                chunks.append(buf)
                buf = ''
        if text[i] != ' ' or i == 0 or text[i-1] != ' ':
            buf += text[i]
    if buf:
        chunks.append(buf)
    return [(c, bytes_to_unicode(c), bpe(bytes_to_unicode(c))) for c in chunks]

for text in ['hello world test', 'hello chatterbox turbo']:
    print(f'\n=== {text!r} ===')
    print('CORRECT (Python-style):')
    for c, u, pieces in tokenize_correct(text):
        ids = [vocab.get(p, '?') for p in pieces]
        print(f'  chunk={c!r:25s} bytes={u!r:25s} -> {pieces}  ids={ids}')
    print('C++ BUG (trailing-space):')
    for c, u, pieces in tokenize_cpp_bug(text):
        ids = [vocab.get(p, '?') for p in pieces]
        print(f'  chunk={c!r:25s} bytes={u!r:25s} -> {pieces}  ids={ids}')

def tokenize_cpp_FIXED(text):
    """My new C++ tokenize_text_bpe — leading-space-on-next-word."""
    chunks = []
    i = 0
    while i < len(text):
        start = i
        if i < len(text) and text[i] == ' ':
            if i + 1 < len(text) and text[i+1] != ' ':
                i += 1
            else:
                while i < len(text) and text[i] == ' ':
                    i += 1
                chunks.append(text[start:i])
                continue
        while i < len(text) and text[i] != ' ':
            i += 1
        chunks.append(text[start:i])
    return [(c, bytes_to_unicode(c), bpe(bytes_to_unicode(c))) for c in chunks]

for text in ['hello world test', 'hello chatterbox turbo']:
    print(f'\n=== FIXED {text!r} ===')
    for c, u, pieces in tokenize_cpp_FIXED(text):
        ids = [vocab.get(p, '?') for p in pieces]
        print(f'  chunk={c!r:25s} bytes={u!r:25s} -> {pieces}  ids={ids}')
