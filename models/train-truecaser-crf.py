#!/usr/bin/env python3
"""Train a CRF truecaser from German Wikipedia and export to binary format.

The CRF predicts per-word casing labels (lc/u1/uc) using context features:
  - word identity (lowercased)
  - 3-char suffix
  - previous/next word identity
  - is preceded by article (der/die/das/ein/eine/...)
  - word shape (has digits, has hyphen, all-alpha)
  - position in sentence (first word)

Output: compact binary format loadable by src/truecaser_crf.cpp

Usage:
    python models/train-truecaser-crf.py --output /mnt/storage/models/truecaser-crf-de.bin
"""

import argparse
import os
import struct
import sys
import time
from collections import defaultdict

import pycrfsuite
from datasets import load_dataset

LABELS = ["lc", "u1", "uc"]
ARTICLES = {"der", "die", "das", "ein", "eine", "einem", "einen", "einer", "eines",
            "dem", "den", "des", "kein", "keine", "keinem", "keinen", "keiner", "keines"}


def word_label(w):
    """Determine the casing label of a word."""
    if not any(c.isalpha() for c in w):
        return None
    if w == w.lower():
        return "lc"
    if w[0].isupper() and (len(w) == 1 or w[1:] == w[1:].lower()):
        return "u1"
    if w == w.upper() and len(w) > 1:
        return "uc"
    return "lc"  # mixed case → treat as lc


def word_features(words_lower, i, is_first):
    """Extract features for word at position i."""
    w = words_lower[i]
    feats = [
        f"w={w}",
        f"w[-3:]={w[-3:]}" if len(w) >= 3 else f"w[-3:]={w}",
        f"w[-2:]={w[-2:]}" if len(w) >= 2 else f"w[-2:]={w}",
        f"len={min(len(w), 15)}",
    ]

    # Suffix patterns (German noun indicators)
    for suffix in ["ung", "heit", "keit", "schaft", "tion", "ment", "nis", "tät", "ismus"]:
        if w.endswith(suffix):
            feats.append(f"noun_suffix={suffix}")

    # Word shape
    if any(c.isdigit() for c in w):
        feats.append("has_digit")
    if "-" in w:
        feats.append("has_hyphen")
    if w.isalpha():
        feats.append("all_alpha")

    # Context
    if i > 0:
        prev = words_lower[i - 1]
        feats.append(f"w-1={prev}")
        feats.append(f"w-1[-3:]={prev[-3:]}" if len(prev) >= 3 else f"w-1[-3:]={prev}")
        if prev in ARTICLES:
            feats.append("after_article")
    else:
        feats.append("BOS")

    if i < len(words_lower) - 1:
        nxt = words_lower[i + 1]
        feats.append(f"w+1={nxt}")
    else:
        feats.append("EOS")

    if is_first:
        feats.append("sent_start")

    return feats


def extract_sentences(text):
    """Split text into sentences and extract (features, labels) pairs."""
    # Simple sentence splitting on . ? !
    import re
    sents = re.split(r'(?<=[.?!])\s+', text)
    result = []
    for sent in sents:
        words = sent.split()
        if len(words) < 3:
            continue
        words_lower = [w.lower() for w in words]
        labels = [word_label(w) for w in words]
        # Skip sentences with unlabelable words
        if None in labels:
            # Filter out unlabelable
            filtered = [(wl, lab, i) for i, (wl, lab) in enumerate(zip(words_lower, labels)) if lab is not None]
            if len(filtered) < 2:
                continue
            words_lower = [f[0] for f in filtered]
            labels = [f[1] for f in filtered]
            # Recalculate features with new indices
            feats = [word_features(words_lower, i, i == 0) for i in range(len(words_lower))]
        else:
            feats = [word_features(words_lower, i, i == 0) for i in range(len(words_lower))]
        # Skip first word (sentence-initial capitalization is ambiguous)
        feats = feats[1:]
        labels = labels[1:]
        if feats:
            result.append((feats, labels))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, help="Output binary file")
    parser.add_argument("--lines", type=int, default=1_000_000, help="Wikipedia lines to process")
    args = parser.parse_args()

    print("Loading German Wikipedia...")
    ds = load_dataset("wikimedia/wikipedia", "20231101.de", split="train", streaming=True)

    X_train = []
    y_train = []
    n_lines = 0
    t0 = time.time()

    for item in ds:
        text = item["text"]
        for sent_feats, sent_labels in extract_sentences(text):
            X_train.append(sent_feats)
            y_train.append(sent_labels)
        n_lines += 1
        if n_lines >= args.lines:
            break
        if n_lines % 100000 == 0:
            print(f"  {n_lines} lines, {len(X_train)} sentences, {time.time()-t0:.0f}s")

    print(f"Training data: {len(X_train)} sentences from {n_lines} lines ({time.time()-t0:.0f}s)")

    # Train CRF
    print("Training CRF...")
    trainer = pycrfsuite.Trainer(verbose=True)
    for xseq, yseq in zip(X_train, y_train):
        trainer.append(xseq, yseq)

    trainer.set_params({
        'c1': 0.1,           # L1 regularization
        'c2': 0.01,          # L2 regularization
        'max_iterations': 100,
        'feature.possible_transitions': True,
    })

    crf_model_path = args.output + ".crfsuite"
    trainer.train(crf_model_path)
    print(f"CRF model saved to {crf_model_path} ({os.path.getsize(crf_model_path) / 1e6:.1f} MB)")

    # Load and inspect
    tagger = pycrfsuite.Tagger()
    tagger.open(crf_model_path)
    print(f"Labels: {tagger.labels()}")

    # Test on our problem sentences
    tests = [
        "die schnelle braune katze springt über den faulen hund",
        "guten morgen wie geht es ihnen heute",
        "die technologie verändert unsere welt rasant",
    ]
    for text in tests:
        words = text.split()
        feats = [word_features(words, i, i == 0) for i in range(len(words))]
        preds = tagger.tag(feats)
        result = []
        for w, p in zip(words, preds):
            if p == "u1":
                result.append(w[0].upper() + w[1:])
            elif p == "uc":
                result.append(w.upper())
            else:
                result.append(w)
        print(f"  In:  {text}")
        print(f"  Out: {' '.join(result)}")

    # Export to compact binary format
    # Format:
    #   magic: "CRF1" (4 bytes)
    #   n_labels: uint16
    #   label strings (uint16 len + bytes, repeated n_labels times)
    #   transition matrix: n_labels * n_labels * float32
    #   n_features: uint32
    #   for each feature:
    #     uint16 key_len, key_bytes (feature string)
    #     n_labels * float32 (weights for each label)
    info = tagger.info()

    # Get transitions
    transitions = {}
    for (l1, l2), w in info.transitions.items():
        transitions[(l1, l2)] = w

    labels = tagger.labels()
    n_labels = len(labels)
    label_to_idx = {l: i for i, l in enumerate(labels)}

    # Get state features (unigram features)
    # info.state_features is {(attr, label): weight}
    feat_weights = defaultdict(lambda: [0.0] * n_labels)
    for (attr, label), weight in info.state_features.items():
        idx = label_to_idx[label]
        feat_weights[attr][idx] = weight

    print(f"\nExporting: {n_labels} labels, {len(feat_weights)} features")

    with open(args.output, "wb") as f:
        # Magic
        f.write(b"CRF1")

        # Labels
        f.write(struct.pack("<H", n_labels))
        for label in labels:
            lb = label.encode("utf-8")
            f.write(struct.pack("<H", len(lb)))
            f.write(lb)

        # Transition matrix [n_labels x n_labels] row-major
        for l1 in labels:
            for l2 in labels:
                w = transitions.get((l1, l2), 0.0)
                f.write(struct.pack("<f", w))

        # Features
        f.write(struct.pack("<I", len(feat_weights)))
        for attr, weights in sorted(feat_weights.items()):
            ab = attr.encode("utf-8")
            f.write(struct.pack("<H", len(ab)))
            f.write(ab)
            for w in weights:
                f.write(struct.pack("<f", w))

    sz = os.path.getsize(args.output)
    print(f"Binary model: {args.output} ({sz / 1e6:.1f} MB)")

    # Cleanup crfsuite file
    os.remove(crf_model_path)
    print("Done")


if __name__ == "__main__":
    main()
