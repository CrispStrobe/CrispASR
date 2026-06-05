"""GlotLID (cis-lmu/glotlid V3) reference dump backend.

Captures the fastText classifier forward pass — tokenization → hashed
char-n-gram bucket lookup → mean-pool → linear → softmax — at every
stage the C++ port (``src/lid_fasttext.{h,cpp}``) needs to reproduce.

Stages dumped (matches ``DEFAULT_STAGES`` below):

  ``token_words``        ``str`` (metadata) — whitespace-tokenized words
  ``input_ids``          ``i32 [n_input]``  — concatenated word_ids and
                                              ``n_words + bucket_idx`` entries
                                              that the classifier mean-pools
  ``embedding_bag_out``  ``f32 [dim]``      — mean of the ``input_ids`` rows
                                              of the input matrix
  ``logits``             ``f32 [n_labels]`` — ``output_matrix @ embedding_bag_out``
  ``softmax``            ``f32 [n_labels]`` — post-softmax probabilities
  ``top1_score``         ``f32 [1]``        — softmax score of the argmax
  ``top1_label``         ``str`` (metadata) — argmax label (no ``__label__`` prefix)
  ``input_text``         ``str`` (metadata) — verbatim input string

The "audio" arg is unused (required by the dispatcher signature). The
input text comes from the ``GLOTLID_TEXT`` env var (default: a multilingual
sample picked to exercise non-Latin scripts so the diff harness catches
UTF-8 codepoint-vs-byte bugs in the hash port).

Cross-checks performed during dump (raise on mismatch):

  * our mean-pool of the input rows ≈ ``model.get_sentence_vector(text)``
  * our top-1 label ≈ ``model.predict(text)[0][0]``

These guard against subtle drift between our manual reproduction and
fastText's compiled forward.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import numpy as np

DEFAULT_STAGES = [
    "input_ids",
    "embedding_bag_out",
    "logits",
    "softmax",
    "top1_score",
]

DEFAULT_TEXT = (
    "Hello world. Bonjour le monde. Hallo Welt. "
    "你好世界。 مرحبا بالعالم. नमस्ते दुनिया।"
)


def _whitespace_tokenize(text: str) -> List[str]:
    """fastText's tokenizer is whitespace-split with newline normalization.

    For LID input we replicate the supervised-classifier path: split on
    ASCII whitespace, drop empty tokens. fastText's compiled
    ``Dictionary::readWord`` also treats ``\\n`` as ``</s>`` but for
    classification that EOS row is never used (the classifier never sees
    EOS in supervised mode). Match the behavior of
    ``fasttext.tokenize(text)`` for parity.
    """
    import fasttext
    return list(fasttext.tokenize(text))


EOS_ROW_ID = 0  # fastText always reserves row 0 for the </s> pseudo-token


def _gather_input_ids(model, words: List[str]) -> Tuple[List[int], List[int], List[List[int]]]:
    """For each word, gather the input-matrix row indices fastText would
    mean-pool over, plus the trailing ``</s>`` row.

    Returns:
        all_ids:      flat list of row indices (matches ``computeHidden`` input)
        word_ids:     per-word ``word_id`` (``-1`` for OOV)
        per_word_ids: per-word list of row indices (word_id + subword bucket IDs)

    Mirrors ``Dictionary::getLine`` + ``addSubwords`` from the upstream
    fastText source:

      - in-vocab word: word_id + all subword bucket IDs (offset by ``n_words``)
      - OOV word:      only subword bucket IDs (no ``-1`` placeholder)
      - Trailing ``</s>`` (row 0) — fastText's ``readWord`` injects EOS at
        end-of-stream in supervised mode, and ``initNgrams`` skips subword
        expansion for ``</s>`` so its precomputed row list is just ``[0]``.

    The trailing ``</s>`` row is what bites every fastText port that
    skips it: cosine drops to ~0.97 even though every other index is
    correct. The C++ port MUST append it too. (Verified end-to-end:
    appending ``[0]`` brings ``cos`` to 1.0 within float32 epsilon
    against ``model.get_sentence_vector(text)``.)
    """
    all_ids: List[int] = []
    word_ids: List[int] = []
    per_word_ids: List[List[int]] = []
    for w in words:
        wid = model.get_word_id(w)
        word_ids.append(int(wid))
        # get_subwords returns (subword_strs, row_ids_with_offset).
        # For an in-vocab word the first entry is the word itself with
        # row_id == wid; remaining entries are subwords with row_id ==
        # n_words + bucket_hash.
        # For an OOV word the first entry is the word with row_id == -1
        # (which fastText skips); the rest are subword buckets.
        _subs, ids = model.get_subwords(w)
        ids = list(int(i) for i in ids)
        if wid < 0:
            # OOV: drop the leading -1 (or any -1 sentinel) and keep
            # only positive subword bucket rows.
            ids = [i for i in ids if i >= 0]
        # Either way, ids is now the list of row indices to add.
        all_ids.extend(ids)
        per_word_ids.append(ids)

    # End-of-line </s> pseudo-token. fastText injects this in
    # supervised mode regardless of input punctuation.
    all_ids.append(EOS_ROW_ID)
    word_ids.append(EOS_ROW_ID)
    per_word_ids.append([EOS_ROW_ID])

    return all_ids, word_ids, per_word_ids


def _softmax(x: np.ndarray) -> np.ndarray:
    m = float(x.max())
    e = np.exp(x - m)
    return (e / e.sum()).astype(np.float32)


def _log_sigmoid(x: float) -> float:
    """Numerically-stable log(sigmoid(x))."""
    import math
    if x >= 0:
        return -math.log1p(math.exp(-x))
    return x - math.log1p(math.exp(x))


def _parse_fasttext_label_counts(bin_path: Path) -> List[Tuple[str, int]]:
    """Return (label, count) for every label in the .bin in dictionary
    order. Companion to the converter's parser; see that file for the
    binary layout."""
    import struct
    data = bin_path.read_bytes()
    off = 0
    magic, _version = struct.unpack_from('<II', data, off); off += 8
    if magic != 0x2f4f16ba:
        raise SystemExit(f"unexpected fastText magic {magic:#x} in {bin_path}")
    off += 12 * 4 + 8
    size_, _nwords, _nlabels = struct.unpack_from('<iii', data, off); off += 12
    off += 16
    out: List[Tuple[str, int]] = []
    for _ in range(size_):
        end = data.index(b'\x00', off); word = data[off:end].decode('utf-8'); off = end + 1
        count = struct.unpack_from('<q', data, off)[0]; off += 8
        etype = data[off]; off += 1
        if etype == 1:
            out.append((word, count))
    return out


def _build_hs_tree_paths(counts: List[int]) -> Tuple[List[List[int]], List[List[int]]]:
    """Port of fastText src/model.cc::buildTree; returns (paths, codes)."""
    osz = len(counts)
    tree = [{'parent': -1, 'left': -1, 'right': -1, 'count': int(1e15), 'binary': False}
            for _ in range(2 * osz - 1)]
    for i in range(osz):
        tree[i]['count'] = counts[i]
    leaf, node = osz - 1, osz
    for i in range(osz, 2 * osz - 1):
        mini = [0, 0]
        for j in range(2):
            if leaf >= 0 and tree[leaf]['count'] < tree[node]['count']:
                mini[j] = leaf; leaf -= 1
            else:
                mini[j] = node; node += 1
        tree[i]['left'], tree[i]['right'] = mini[0], mini[1]
        tree[i]['count'] = tree[mini[0]]['count'] + tree[mini[1]]['count']
        tree[mini[0]]['parent'] = i
        tree[mini[1]]['parent'] = i
        tree[mini[1]]['binary'] = True
    paths = [[] for _ in range(osz)]
    codes = [[] for _ in range(osz)]
    for i in range(osz):
        j = i
        while tree[j]['parent'] != -1:
            paths[i].append(tree[j]['parent'] - osz)
            codes[i].append(1 if tree[j]['binary'] else 0)
            j = tree[j]['parent']
    return paths, codes


def dump(*, model_dir: Path, audio: np.ndarray, stages: Set[str],
         max_new_tokens: int) -> Dict[str, Any]:
    """Run the GlotLID classifier forward and return captured stage tensors."""
    del audio, max_new_tokens  # unused — text-only backend

    try:
        import fasttext
    except ImportError as e:
        raise SystemExit(
            "fasttext Python package not found. "
            "pip install fasttext  (the official one, not fasttext-langdetect).") from e

    # Resolve the model.bin path. ``model_dir`` may be the HF snapshot dir
    # (containing ``model.bin``) or the .bin file itself.
    p = Path(model_dir)
    if p.is_dir():
        cand = p / "model.bin"
        if not cand.exists():
            # Fallback: look for any single .bin under the snapshot dir.
            bins = list(p.glob("*.bin"))
            if len(bins) != 1:
                raise SystemExit(
                    f"GlotLID dump: cannot locate model.bin under {p}. "
                    f"Found {len(bins)} .bin files.")
            cand = bins[0]
        bin_path = cand
    else:
        bin_path = p

    print(f"  loading fastText model: {bin_path}")
    model = fasttext.load_model(str(bin_path))

    # Single env var across variants; GLOTLID_TEXT kept as backward-compat fallback.
    text = (os.environ.get("LID_TEXT")
            or os.environ.get("GLOTLID_TEXT")
            or DEFAULT_TEXT)
    words = _whitespace_tokenize(text)
    if not words:
        raise SystemExit("GlotLID dump: empty token list (GLOTLID_TEXT was empty)")
    print(f"  text          : {text!r}")
    print(f"  tokenized     : {words}")

    n_words = len(model.get_words())
    n_labels = len(model.get_labels())
    in_mat = model.get_input_matrix()    # (n_words + bucket, dim) float32
    out_mat = model.get_output_matrix()  # (n_labels, dim) float32
    dim = in_mat.shape[1]

    # ── Stage: input_ids ───────────────────────────────────────────────
    all_ids, _word_ids, _per_word = _gather_input_ids(model, words)
    if not all_ids:
        raise SystemExit(
            "GlotLID dump: no row IDs gathered. Every input word was OOV with "
            "zero hashable subwords — input is too short or all-empty.")
    input_ids = np.asarray(all_ids, dtype=np.int32)

    # ── Stage: embedding_bag_out ───────────────────────────────────────
    rows = np.asarray(in_mat[input_ids], dtype=np.float32)  # (n_input, dim)
    embedding_bag = rows.mean(axis=0).astype(np.float32)    # (dim,)

    # Cross-check: model.get_sentence_vector should match within a tight
    # epsilon (it's the same compiled C++ forward).
    sv = np.asarray(model.get_sentence_vector(text), dtype=np.float32)
    if sv.shape != embedding_bag.shape:
        raise SystemExit(
            f"GlotLID dump: sentence-vector shape {sv.shape} != reproduced "
            f"shape {embedding_bag.shape}")
    cos = float(np.dot(sv, embedding_bag) /
                (np.linalg.norm(sv) * np.linalg.norm(embedding_bag) + 1e-12))
    max_abs = float(np.max(np.abs(sv - embedding_bag)))
    print(f"  embedding-bag cross-check vs model.get_sentence_vector(): "
          f"cos={cos:.6f} max_abs={max_abs:.4e}")
    if cos < 0.9999 or max_abs > 1e-3:
        raise SystemExit(
            f"GlotLID dump: reproduced embedding-bag drifted from "
            f"fastText's own (cos={cos:.6f} max_abs={max_abs:.4e}). "
            f"This means our row-id reproduction is buggy — fix before "
            f"trusting the C++ port.")

    # Detect loss type — flat softmax (GlotLID) vs hierarchical (LID-176).
    args_obj = model.f.getArgs()
    loss_enum = int(args_obj.loss.value) if hasattr(args_obj.loss, 'value') else int(args_obj.loss)
    loss_str = "softmax" if loss_enum == 0 else "hs" if loss_enum == 1 else f"unknown_{loss_enum}"
    print(f"  loss          : {loss_str}")

    full_labels = list(model.get_labels())   # with __label__ prefix
    label_prefix = "__label__"
    short_labels = [
        lab[len(label_prefix):] if lab.startswith(label_prefix) else lab
        for lab in full_labels
    ]

    if loss_str == "softmax":
        # ── Stage: logits = output @ embedding_bag ─────────────────
        logits = (np.asarray(out_mat, dtype=np.float32) @ embedding_bag).astype(np.float32)
        # ── Stage: softmax ────────────────────────────────────────
        probs = _softmax(logits)
    elif loss_str == "hs":
        # Reconstruct the Huffman tree (.bin parse) and walk it per label.
        bin_labels = _parse_fasttext_label_counts(bin_path)
        if [w for w, _ in bin_labels] != full_labels:
            raise SystemExit("HS dump: label order mismatch between .bin and "
                             "get_labels(); tree would be miscomputed.")
        counts = [c for _, c in bin_labels]
        paths, codes = _build_hs_tree_paths(counts)
        # Per-label log P(label | hidden) via the tree
        out_f32 = np.asarray(out_mat, dtype=np.float32)
        logits = np.zeros(n_labels, dtype=np.float32)  # = log-prob — keep "logits" name for stage parity
        for i in range(n_labels):
            s = 0.0
            for node_idx, code in zip(paths[i], codes[i]):
                f = float(out_f32[node_idx] @ embedding_bag)
                s += _log_sigmoid((2 * code - 1) * f)
            logits[i] = s
        probs = np.exp(logits).astype(np.float32)  # already normalised — sum ≈ 1
    else:
        raise SystemExit(f"GlotLID dump: unsupported loss '{loss_str}'")

    # Cross-check: our argmax label should match model.predict(text).
    top1_idx = int(np.argmax(probs))
    our_top1 = short_labels[top1_idx]
    fa_labels, fa_scores = model.predict(text, k=1)
    fa_top1 = fa_labels[0][len(label_prefix):] if fa_labels[0].startswith(label_prefix) else fa_labels[0]
    fa_score = float(fa_scores[0])
    print(f"  top-1: ours={our_top1!r} score={float(probs[top1_idx]):.6f}  "
          f"|  fastText={fa_top1!r} score={fa_score:.6f}")
    if our_top1 != fa_top1:
        raise SystemExit(
            f"GlotLID dump: argmax mismatch (ours={our_top1!r} vs "
            f"fastText={fa_top1!r}). The reference forward is broken — "
            f"refusing to dump.")

    # ── Pack outputs ───────────────────────────────────────────────────
    out: Dict[str, Any] = {}
    if "input_ids" in stages:
        out["input_ids"] = input_ids.astype(np.int32, copy=False)
    if "embedding_bag_out" in stages:
        out["embedding_bag_out"] = embedding_bag.astype(np.float32, copy=False)
    if "logits" in stages:
        out["logits"] = logits.astype(np.float32, copy=False)
    if "softmax" in stages:
        out["softmax"] = probs.astype(np.float32, copy=False)
    if "top1_score" in stages:
        out["top1_score"] = np.asarray([probs[top1_idx]], dtype=np.float32)

    # String-typed captures auto-route into GGUF metadata via
    # write_gguf_archive (see tools/dump_reference.py:365–367).
    out["input_text"] = text
    out["top1_label"] = our_top1
    out["token_words"] = " ".join(words)

    print(f"  dumped {sum(1 for v in out.values() if isinstance(v, np.ndarray))} "
          f"tensors + 3 metadata strings")
    print(f"  dim={dim} n_labels={n_labels} n_words={n_words} "
          f"input_ids_len={len(input_ids)}")
    return out
