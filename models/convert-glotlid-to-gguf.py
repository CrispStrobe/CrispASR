#!/usr/bin/env python3
"""
Convert a fastText supervised LID model (``model.bin``) to GGUF.

Handles **GlotLID** (cis-lmu/glotlid, V3, ~2102 ISO 639-3 + script labels,
Apache-2.0) and **LID-176** (Facebook's lid.176, CC-BY-SA-3.0, 176 langs).
Architecture is identical: hashed character n-gram embedding bag → mean
pool → linear → softmax. Only weights, vocab, and metadata differ.

The two tensors written:

  ``lid_fasttext.embedding``  F32  ``[input_rows, dim]``
  ``lid_fasttext.output``     F32  ``[n_labels, dim]``

where ``input_rows = n_words + bucket`` for fastText's standard hashed
representation (the first ``n_words`` rows are vocabulary word
embeddings, the next ``bucket`` rows are subword hash buckets).

Metadata keys:

  ``lid_fasttext.variant``         str   "glotlid-v3" | "fasttext-lid176"
  ``lid_fasttext.dim``             u32   embedding dimension (256)
  ``lid_fasttext.bucket``          u32   subword hash bucket count
  ``lid_fasttext.n_words``         u32   word vocab size
  ``lid_fasttext.n_labels``        u32   number of language labels
  ``lid_fasttext.minn``            u32   minimum char n-gram length
  ``lid_fasttext.maxn``            u32   maximum char n-gram length
  ``lid_fasttext.word_ngrams``     u32   word n-gram order (typically 1)
  ``lid_fasttext.label_prefix``    str   "__label__" by convention
  ``lid_fasttext.labels``          str-array  language labels (in label-id order)
  ``lid_fasttext.words``           str-array  word vocab (in word-id order)

Usage:

    HF_HOME=/Volumes/backups/ai/huggingface-hub \\
    HUGGINGFACE_HUB_CACHE=/Volumes/backups/ai/huggingface-hub \\
    python models/convert-glotlid-to-gguf.py \\
        --variant glotlid-v3 \\
        --input  /Volumes/backups/ai/huggingface-hub/models--cis-lmu--glotlid/snapshots/<rev>/model.bin \\
        --output /Volumes/backups/ai/crispasr-models/lid-glotlid/lid-glotlid-f16.gguf \\
        --dtype f16

For ``--variant fasttext-lid176`` the input is the upstream
``lid.176.bin`` from Facebook (CC-BY-SA-3.0 — distributors must keep
the SA notice).

This converter writes F32 or F16. Use ``crispasr-quantize`` for Q8_0
and below; per docs/quantize.md, embeddings should be diff-harness
verified before locking in K-quants on a 2M-row table.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

try:
    from gguf import GGUFWriter, GGMLQuantizationType
except ImportError:
    sys.exit("pip install gguf")

try:
    import fasttext
except ImportError:
    sys.exit("pip install fasttext")


KNOWN_VARIANTS = {"glotlid-v3", "fasttext-lid176"}


def _strip_label_prefix(labels: list[str], prefix: str) -> list[str]:
    """fastText labels come back as ``__label__eng_Latn``; strip the prefix."""
    out = []
    for lab in labels:
        if lab.startswith(prefix):
            out.append(lab[len(prefix):])
        else:
            out.append(lab)
    return out


_DTYPE_MAP = {
    "f32":  GGMLQuantizationType.F32,
    "f16":  GGMLQuantizationType.F16,
    "q8_0": GGMLQuantizationType.Q8_0,
    "q5_k": GGMLQuantizationType.Q5_K,
    "q4_k": GGMLQuantizationType.Q4_K,
}


def _parse_fasttext_label_counts(bin_path: Path) -> list[tuple[str, int, int]]:
    """Parse fastText's binary model.bin to extract (label, count, n_labels).

    fastText doesn't expose label frequencies via the Python API, but
    they're needed to deterministically rebuild the hierarchical-softmax
    Huffman tree at conversion time. Format from src/dictionary.cc::save:

      magic(u32) version(u32)
      args: 12*i32 + 1*f64
      dictionary header: size(i32) nwords(i32) nlabels(i32) ntokens(i64) pruneidx_size(i64)
      for size entries: cstring word; i64 count; i8 type   (type 0=word, 1=label)

    Returns list of (label_with_prefix, count, label_index_in_label_space).
    """
    import struct
    data = bin_path.read_bytes()
    off = 0
    magic, version = struct.unpack_from('<II', data, off); off += 8
    if magic != 0x2f4f16ba:
        raise SystemExit(f"unexpected fastText magic {magic:#x} in {bin_path}")
    off += 12 * 4 + 8  # skip args (12 i32 + 1 f64)
    size_, _nwords, _nlabels = struct.unpack_from('<iii', data, off); off += 12
    off += 16          # skip ntokens + pruneidx_size
    labels: list[tuple[str, int, int]] = []
    label_idx = 0
    for _ in range(size_):
        end = data.index(b'\x00', off); word = data[off:end].decode('utf-8'); off = end + 1
        count = struct.unpack_from('<q', data, off)[0]; off += 8
        etype = data[off]; off += 1
        if etype == 1:
            labels.append((word, count, label_idx))
            label_idx += 1
    return labels


def _build_hs_tree(label_counts: list[int]) -> tuple[list[list[int]], list[list[int]]]:
    """Port of fastText src/model.cc::buildTree.

    Given label counts in dictionary order (= label-id order), build the
    Huffman tree exactly the way fastText does at load time and return
    per-label (path, code) pairs:

      path[i][k] = internal-node index in [0, osz-1) for step k of label i's
                   walk from leaf to root
      code[i][k] = 1 if the step came from the right child, 0 if left

    The Huffman build uses a two-pointer approach over (sorted) leaves
    and the freshly-built internal nodes — see fastText's `Model::buildTree`.
    """
    osz = len(label_counts)
    tree = [{'parent': -1, 'left': -1, 'right': -1, 'count': int(1e15), 'binary': False}
            for _ in range(2 * osz - 1)]
    for i in range(osz):
        tree[i]['count'] = label_counts[i]
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
    paths: list[list[int]] = [[] for _ in range(osz)]
    codes: list[list[int]] = [[] for _ in range(osz)]
    for i in range(osz):
        j = i
        while tree[j]['parent'] != -1:
            paths[i].append(tree[j]['parent'] - osz)
            codes[i].append(1 if tree[j]['binary'] else 0)
            j = tree[j]['parent']
    return paths, codes


def convert(in_path: Path, out_path: Path, variant: str, dtype: str) -> None:
    if variant not in KNOWN_VARIANTS:
        sys.exit(f"unknown variant '{variant}'. Known: {sorted(KNOWN_VARIANTS)}")
    if dtype not in _DTYPE_MAP:
        sys.exit(f"unsupported --dtype '{dtype}'. Choose one of: {sorted(_DTYPE_MAP)}")

    print(f"loading fastText model: {in_path}")
    m = fasttext.load_model(str(in_path))

    # ── shape & metadata extraction ────────────────────────────────────
    dim = int(m.get_dimension())
    labels_raw = list(m.get_labels())  # full ['__label__eng_Latn', ...]
    label_prefix = "__label__"
    labels = _strip_label_prefix(labels_raw, label_prefix)
    n_labels = len(labels)

    words = list(m.get_words())
    n_words = len(words)

    in_mat = m.get_input_matrix()    # numpy ndarray shape (n_words+bucket, dim)
    out_mat = m.get_output_matrix()  # numpy ndarray shape (n_labels, dim)

    if in_mat.shape[1] != dim:
        sys.exit(f"input matrix dim {in_mat.shape[1]} != reported dim {dim}")
    if out_mat.shape != (n_labels, dim):
        sys.exit(f"output matrix shape {out_mat.shape} != ({n_labels}, {dim})")

    bucket = int(in_mat.shape[0]) - n_words
    if bucket < 0:
        sys.exit(f"input matrix has {in_mat.shape[0]} rows but n_words={n_words}; "
                 f"refusing to invent a negative bucket count")

    # ── n-gram args — fastText doesn't expose them on the Python side, but
    # we can ask the C++-via-Python helper through a probe. minn/maxn are
    # captured by examining .get_subwords() output for a known token: the
    # number of subwords for a length-L word is sum_{n=minn..maxn} max(0, L-n+1).
    # In practice, GlotLID-V3 uses minn=2, maxn=5 (per its training recipe);
    # LID-176 uses minn=2, maxn=5 also. We hardcode and verify.
    minn, maxn = _probe_ngram_range(m)
    word_ngrams = 1  # both GlotLID and LID-176 use word_ngrams=1

    # Detect HS vs flat softmax. fastText's loss enum: softmax=0, hs=1, ns=2, ova=3.
    args_obj = m.f.getArgs()
    loss_enum = int(args_obj.loss.value) if hasattr(args_obj.loss, 'value') else int(args_obj.loss)
    if loss_enum == 1:  # hs
        loss_str = "hs"
    elif loss_enum == 0:
        loss_str = "softmax"
    else:
        loss_str = f"loss_enum_{loss_enum}"

    # For HS, parse counts from the .bin (the only reliable source) and
    # rebuild the Huffman tree deterministically. The fastText label
    # order from the .bin must match get_labels() — we double-check.
    hs_paths_data: list[int] = []
    hs_codes_data: list[int] = []
    hs_path_offsets: list[int] = [0]
    if loss_str == "hs":
        bin_labels = _parse_fasttext_label_counts(in_path)
        if [w for w, _, _ in bin_labels] != labels_raw:
            sys.exit("convert-glotlid: label order mismatch between .bin parse "
                     "and m.get_labels(). HS tree would be miscomputed; aborting.")
        counts = [c for _, c, _ in bin_labels]
        paths, codes = _build_hs_tree(counts)
        for path, code in zip(paths, codes):
            hs_paths_data.extend(path)
            hs_codes_data.extend(code)
            hs_path_offsets.append(len(hs_paths_data))

    print(f"  variant       : {variant}")
    print(f"  loss          : {loss_str}")
    print(f"  dim           : {dim}")
    print(f"  bucket        : {bucket}")
    print(f"  n_words       : {n_words}")
    print(f"  n_labels      : {n_labels}")
    print(f"  ngram range   : [{minn}, {maxn}]")
    print(f"  input matrix  : {in_mat.shape}  {in_mat.dtype}")
    print(f"  output matrix : {out_mat.shape}  {out_mat.dtype}")
    if loss_str == "hs":
        avg_path = len(hs_paths_data) / n_labels if n_labels else 0
        print(f"  HS tree       : {len(hs_paths_data)} total path steps, "
              f"avg path len = {avg_path:.2f}")
    sample_labels = ", ".join(labels[:6])
    if n_labels > 6:
        sample_labels += f", ... (+{n_labels - 6} more)"
    print(f"  labels sample : {sample_labels}")

    qtype = _DTYPE_MAP[dtype]
    in_f32 = np.ascontiguousarray(in_mat, dtype=np.float32)
    out_f32 = np.ascontiguousarray(out_mat, dtype=np.float32)
    if qtype in (GGMLQuantizationType.F32, GGMLQuantizationType.F16):
        in_arr = in_f32 if qtype == GGMLQuantizationType.F32 else in_f32.astype(np.float16)
        out_arr = out_f32 if qtype == GGMLQuantizationType.F32 else out_f32.astype(np.float16)
        gtype_in = qtype
        gtype_out = qtype
    else:
        # K-quants need rows that are a multiple of 256 elements.
        # GlotLID + LID-176 both use dim=256 (GlotLID) or dim=16 (LID-176).
        # For dim=16, K-quants would error; fall back to Q8_0 for those.
        from gguf import quantize
        gtype_in = qtype
        gtype_out = qtype
        if dim < 256 and qtype.name.endswith("_K"):
            print(f"  NOTE: dim={dim} < 256, falling back to Q8_0 for K-quant variant")
            gtype_in = GGMLQuantizationType.Q8_0
            gtype_out = GGMLQuantizationType.Q8_0
        in_arr = quantize(in_f32, gtype_in)
        out_arr = quantize(out_f32, gtype_out)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"writing GGUF: {out_path}")
    w = GGUFWriter(str(out_path), arch="lid-fasttext", use_temp_file=False)
    w.add_name(f"lid-fasttext-{variant}")
    w.add_description(
        "fastText supervised language identifier — hashed char-n-gram "
        "embedding bag + linear + softmax."
    )

    # Scalar metadata
    w.add_string("lid_fasttext.variant", variant)
    w.add_uint32("lid_fasttext.dim", dim)
    w.add_uint32("lid_fasttext.bucket", bucket)
    w.add_uint32("lid_fasttext.n_words", n_words)
    w.add_uint32("lid_fasttext.n_labels", n_labels)
    w.add_uint32("lid_fasttext.minn", minn)
    w.add_uint32("lid_fasttext.maxn", maxn)
    w.add_uint32("lid_fasttext.word_ngrams", word_ngrams)
    w.add_string("lid_fasttext.label_prefix", label_prefix)
    w.add_string("lid_fasttext.loss", loss_str)
    if loss_str == "hs":
        # CSR-style: paths_data[offsets[i]:offsets[i+1]] is label i's
        # path (internal-node row indices into output_matrix);
        # codes_data is parallel.
        # Stored as int32/int8 tensors so the C++ side can consume
        # them without bit-unpacking metadata arrays.
        paths_arr = np.asarray(hs_paths_data, dtype=np.int32)
        codes_arr = np.asarray(hs_codes_data, dtype=np.int8)
        offsets_arr = np.asarray(hs_path_offsets, dtype=np.int32)
        w.add_tensor("lid_fasttext.hs_path_offsets", offsets_arr,
                     raw_dtype=GGMLQuantizationType.I32)
        w.add_tensor("lid_fasttext.hs_paths", paths_arr,
                     raw_dtype=GGMLQuantizationType.I32)
        w.add_tensor("lid_fasttext.hs_codes", codes_arr,
                     raw_dtype=GGMLQuantizationType.I8)

    # Vocab tables (kokoro tokenizer pattern: GGUF string-array metadata)
    w.add_array("lid_fasttext.labels", labels)
    w.add_array("lid_fasttext.words", words)

    # Tensors. Names include ".weight" so crispasr-quantize's
    # is_weight gate (sname.find("weight") != npos) picks them up
    # for K-quant re-quantization.
    w.add_tensor("lid_fasttext.embedding.weight", in_arr, raw_dtype=gtype_in)
    w.add_tensor("lid_fasttext.output.weight", out_arr, raw_dtype=gtype_out)

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()

    sz_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"  wrote {out_path}  ({sz_mb:.1f} MiB)")


def _probe_ngram_range(m) -> tuple[int, int]:
    """fastText doesn't surface ``minn``/``maxn`` directly. Reverse-engineer
    them from get_subwords() on a long ASCII probe string.

    For a length-L UTF-8-codepoint word ``<word>`` (after wrapping with
    boundaries), the number of subwords of length n is exactly
    ``max(0, (L+2) - n + 1)``. Solving for the smallest and largest n that
    fastText returned gives ``minn`` and ``maxn``.

    We use a 12-char probe to give plenty of dynamic range, and we filter
    out the word itself (the leading entry is the literal word, not a
    subword) by length.
    """
    probe = "abcdefghijkl"  # 12 chars → wrapped <abcdefghijkl> = 14 codepoints
    subs, _ids = m.get_subwords(probe)
    # First entry is the word itself; rest are subwords. Bracket characters
    # appear in the subword set only at the boundaries.
    sw_lens = [len(s) for s in subs[1:]]
    if not sw_lens:
        # Edge case — fallback to fastText's compiled-in default for
        # supervised classifiers.
        return 0, 0
    minn = min(sw_lens)
    maxn = max(sw_lens)
    return minn, maxn


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Convert fastText LID (.bin) → GGUF for CrispASR",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--variant", required=True, choices=sorted(KNOWN_VARIANTS),
                    help="which fastText LID family this is")
    ap.add_argument("--input", required=True, type=Path,
                    help="path to fastText model.bin (or lid.176.bin)")
    ap.add_argument("--output", required=True, type=Path,
                    help="output .gguf path")
    ap.add_argument("--dtype", default="f16",
                    choices=sorted(_DTYPE_MAP.keys()),
                    help="storage dtype for the matrices (default f16). "
                         "K-quants fall back to Q8_0 when row width < 256 "
                         "(e.g. LID-176 has dim=16).")
    args = ap.parse_args()
    convert(args.input, args.output, args.variant, args.dtype)


if __name__ == "__main__":
    main()
