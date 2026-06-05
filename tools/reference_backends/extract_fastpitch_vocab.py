#!/usr/bin/env python3
"""
Reconstruct the NeMo EnglishPhonemesTokenizer vocabulary for FastPitch.

The vocab is deterministic from the tokenizer config:
  - space, consonants, vowels×stresses, lowercase chars, apostrophe, punct
  - Then pad, blank (if add_blank_at), oov

Output: JSON mapping symbol -> id, suitable for embedding in GGUF.
"""

import itertools
import json
import string
import sys


# NeMo ARPABET phoneme sets
CONSONANTS = (
    'B', 'CH', 'D', 'DH', 'F', 'G',
    'HH', 'JH', 'K', 'L', 'M', 'N',
    'NG', 'P', 'R', 'S', 'SH', 'T',
    'TH', 'V', 'W', 'Y', 'Z', 'ZH',
)

VOWELS = (
    'AA', 'AE', 'AH', 'AO', 'AW',
    'AY', 'EH', 'ER', 'EY', 'IH',
    'IY', 'OW', 'OY', 'UH', 'UW',
)

PUNCT_LIST = (
    ',', '.', '!', '?', '-',
    ':', ';', '/', '"', '(',
    ')', '[', ']', '{', '}',
)


def build_vocab(stresses=True, chars=True, apostrophe=True, punct=True,
                add_blank_at=True):
    """Build the vocabulary in the same order as NeMo's EnglishPhonemesTokenizer."""
    tokens = []

    # Space
    tokens.append(' ')

    # Consonants
    tokens.extend(CONSONANTS)

    # Vowels (optionally with stress markers)
    vowels = list(VOWELS)
    if stresses:
        vowels = [f'{p}{s}' for p, s in itertools.product(vowels, (0, 1, 2))]
    tokens.extend(vowels)

    # Lowercase characters
    if chars:
        tokens.extend(string.ascii_lowercase)

    # Apostrophe
    if apostrophe:
        tokens.append("'")

    # Punctuation
    if punct:
        tokens.extend(PUNCT_LIST)

    # Pad
    tokens.append('<pad>')

    # Blank (if add_blank_at is not None)
    if add_blank_at:
        tokens.append('<blank>')

    # OOV
    tokens.append('<oov>')

    return tokens


def main():
    tokens = build_vocab()

    print(f"Vocab size: {len(tokens)}")
    print()

    vocab = {}
    for i, t in enumerate(tokens):
        vocab[t] = i
        print(f"  {i:3d}: '{t}'")

    # Save as JSON
    if len(sys.argv) > 1:
        with open(sys.argv[1], 'w') as f:
            json.dump(vocab, f, indent=2, ensure_ascii=False)
        print(f"\nSaved to {sys.argv[1]}")


if __name__ == "__main__":
    main()
