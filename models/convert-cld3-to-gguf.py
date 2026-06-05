#!/usr/bin/env python3
"""Convert Google CLD3's embedded weights to GGUF.

Source: https://github.com/google/cld3 (Apache-2.0). The full model lives
as plain-text C++ array literals in ``src/lang_id_nn_params.cc`` (1.76 MB);
no binary side-car. We regex-parse the literals, dequantize the six
``uint8 + per-row bfloat16-style scale`` embedding tables to F32, transpose
the hidden / softmax FCs from CLD3's ``[in_dim, out_dim]`` storage to GGUF's
``[out_dim, in_dim]`` (so ``ggml_mul_mat(W, x) = y`` works naturally), and
write an ``arch=lid-cld3`` GGUF.

Architecture (verified against ``embedding_network.cc`` and
``lang_id_nn_params.h``):

    [text]
      ├─ feature 0 (cbog id_dim=1000 size=2 incl_terms=true) → bigrams      → 16-d
      ├─ feature 1 (cbog id_dim=5000 size=4 incl_terms=true) → quadgrams    → 16-d
      ├─ feature 2 (continuous-bag-of-relevant-scripts)      → rel-scripts  →  8-d
      ├─ feature 3 (script)                                  → text-script  →  8-d
      ├─ feature 4 (cbog id_dim=5000 size=3 incl_terms=true) → trigrams     → 16-d
      └─ feature 5 (cbog id_dim=100  size=1 incl_terms=true) → unigrams     → 16-d
    concat[80] → FC + ReLU → hidden[208] → FC → logits[109] → softmax → 109 labels

Critical notes that bit during the port (see LEARNINGS.md / the CLD3 brief):

  * CLD3's ``float16`` is **bfloat16-style** (top 16 bits of fp32),
    NOT IEEE 754 binary16. Decode via ``(uint32 << 16).view(np.float32)``;
    using numpy ``<f2`` would silently produce garbage.
  * Embeddings dequantize as ``(uint8_row - 128) * scale[row]`` —
    symmetric quant with bias 128 (see embedding_network.cc:122-123).
  * Hash function is **MurmurHash2-32** with seed ``0xBEEF``, NOT CityHash
    (see utils.cc:137-183 — constants ``m=0x5bd1e995, r=24`` are textbook
    MurmurHash2). The C++ port and reference dumper both have to match.
  * Hidden FC width = 208 (not the concat dim of 80). The 109-language
    list lives in ``task_context_params.cc``, not ``lang_id_nn_params.cc``.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np

try:
    import gguf
except ImportError:
    sys.exit("pip install gguf")


# ---------------------------------------------------------------------------
# Constants — verified against upstream src/lang_id_nn_params.{cc,h} and
# src/task_context_params.cc. If CLD3 ever updates the model, regen these.
# ---------------------------------------------------------------------------

# Six embedding tables: (rows, cols). Order matches the feature spec in
# task_context_params.cc kLanguageIdentifierFeatures.
EMBEDDING_SHAPES = [
    (1000, 16),  # 0: bigrams        (cbog id_dim=1000 size=2)
    (5000, 16),  # 1: quadgrams      (cbog id_dim=5000 size=4)
    (12, 8),     # 2: relevant-scripts
    (103, 8),    # 3: text-script
    (5000, 16),  # 4: trigrams       (cbog id_dim=5000 size=3)
    (100, 16),   # 5: unigrams       (cbog id_dim=100  size=1)
]
CONCAT_OFFSETS = [0, 16, 32, 40, 48, 64]  # cumulative; sum = 80
CONCAT_DIM = 80
HIDDEN_DIM = 208
N_LABELS = 109

# Names mirror the cbog/RelevantScript/Script taxonomy in upstream so the
# C++ port and reference dumper can switch on them when extracting features.
FEATURE_NAMES = [
    "bigrams",
    "quadgrams",
    "relevant-scripts",
    "text-script",
    "trigrams",
    "unigrams",
]
# Per-feature parameters needed at inference. cbog params are
# (kind, id_dim, ngram_size, include_terminators, include_spaces, use_equal_weight).
# The non-cbog features (relevant-scripts, text-script) have kind != "cbog".
FEATURE_KINDS = ["cbog", "cbog", "relevant-scripts", "script", "cbog", "cbog"]
FEATURE_NGRAM_SIZES = [2, 4, 0, 0, 3, 1]


# ---------------------------------------------------------------------------
# bfloat16-style decode (CLD3's float16 = upper 16 bits of binary32).
# ---------------------------------------------------------------------------

def _bf16u_to_f32(u: np.ndarray) -> np.ndarray:
    """Decode CLD3 ``float16`` (uint16 = upper 16 bits of fp32) to float32."""
    return (u.astype(np.uint32) << 16).view(np.float32)


# ---------------------------------------------------------------------------
# Regex parsing of `lang_id_nn_params.cc`. The literal-list syntax is
# uniform: `const <TYPE> LangIdNNParams::<NAME>[] = { … };` with `u`-suffixed
# uint integers, `f`-suffixed floats. Parser is brace-aware (handles a
# possible trailing comma) and ignores whitespace and `//` line comments.
# ---------------------------------------------------------------------------

_DECL_RE = re.compile(
    r"const\s+(?P<type>uint8|float16|float|int|int32)\s+"
    r"LangIdNNParams::(?P<name>k[A-Za-z0-9_]+)\s*\[\s*\]\s*=\s*\{",
)


def _parse_array_block(src: str, start: int, end: int) -> list[str]:
    """Return the comma-separated literal tokens from ``src[start:end]``."""
    body = src[start:end]
    # Drop line comments and replace whitespace with single spaces.
    body = re.sub(r"//[^\n]*", "", body)
    body = body.replace("\n", " ")
    return [t.strip() for t in body.split(",") if t.strip()]


def _decode_token(tok: str, type_name: str) -> int | float:
    """Decode one literal token. uint8/float16 → int (with optional 'u'),
    float → float (with optional 'f'). Unsigned ints survive numpy roundtrip.
    """
    if type_name in ("uint8", "float16", "int", "int32"):
        return int(tok.rstrip("uU"))
    if type_name == "float":
        return float(tok.rstrip("fF"))
    raise ValueError(f"unknown literal type: {type_name}")


def parse_arrays(cc_path: Path) -> dict[str, np.ndarray]:
    """Parse every ``const TYPE LangIdNNParams::kName[] = {…};`` array
    out of ``cc_path``. Returns a name→ndarray dict; uint8 arrays come back
    as ``np.uint8``, float16 as ``np.uint16`` (raw — caller decodes via
    ``_bf16u_to_f32``), float as ``np.float32``, int as ``np.int32``.
    """
    print(f"Reading: {cc_path}")
    src = cc_path.read_text()
    arrays: dict[str, np.ndarray] = {}

    for m in _DECL_RE.finditer(src):
        type_name = m.group("type")
        name = m.group("name")
        # Find the matching closing brace. The literals contain no nested
        # braces so a depth counter works; we still write one defensively.
        depth = 1
        i = m.end()
        while i < len(src) and depth > 0:
            c = src[i]
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    break
            i += 1
        if depth != 0:
            raise SystemExit(f"unterminated array literal: {name}")
        toks = _parse_array_block(src, m.end(), i)
        try:
            values = [_decode_token(t, type_name) for t in toks]
        except ValueError as exc:
            raise SystemExit(f"failed to parse {name}: {exc}") from exc

        if type_name == "uint8":
            arr = np.asarray(values, dtype=np.uint8)
        elif type_name == "float16":
            arr = np.asarray(values, dtype=np.uint16)
        elif type_name == "float":
            arr = np.asarray(values, dtype=np.float32)
        else:  # int / int32
            arr = np.asarray(values, dtype=np.int32)
        arrays[name] = arr
        print(f"  {name:<32} type={type_name:<8} n={arr.size}")

    return arrays


# ---------------------------------------------------------------------------
# Language-name list — single string array in task_context_params.cc.
# Format: `const char *const TaskContextParams::kLanguageNames[] = { "eo",
# "co", … nullptr, };`. Trailing nullptr terminates.
# ---------------------------------------------------------------------------

def parse_language_names(cc_path: Path) -> list[str]:
    print(f"Reading: {cc_path}")
    src = cc_path.read_text()
    m = re.search(
        r"kLanguageNames\s*\[\s*\]\s*=\s*\{(?P<body>.*?)\};",
        src,
        re.DOTALL,
    )
    if not m:
        raise SystemExit("kLanguageNames not found in task_context_params.cc")
    body = m.group("body")
    body = re.sub(r"//[^\n]*", "", body)
    names = re.findall(r'"([^"]*)"', body)
    return names


# ---------------------------------------------------------------------------
# Conversion main.
# ---------------------------------------------------------------------------

def convert(upstream_src: Path, out_path: Path, write_f16: bool) -> None:
    cc = upstream_src / "lang_id_nn_params.cc"
    tc = upstream_src / "task_context_params.cc"
    if not cc.exists():
        raise SystemExit(f"{cc}: not found")
    if not tc.exists():
        raise SystemExit(f"{tc}: not found")

    arrays = parse_arrays(cc)
    languages = parse_language_names(tc)
    print(f"Languages: {len(languages)} (first={languages[:5]} last={languages[-3:]})")
    if len(languages) != N_LABELS:
        raise SystemExit(
            f"expected {N_LABELS} languages, got {len(languages)} — upstream "
            f"format may have changed; re-verify N_LABELS / EMBEDDING_SHAPES",
        )

    # Sanity-check shape arrays.
    for k, expected in [
        ("kEmbeddingsNumRows", [r for r, _ in EMBEDDING_SHAPES]),
        ("kEmbeddingsNumCols", [c for _, c in EMBEDDING_SHAPES]),
        ("kConcatOffsetValues", CONCAT_OFFSETS),
        ("kHiddenNumRows", [CONCAT_DIM]),
        ("kHiddenNumCols", [HIDDEN_DIM]),
        ("kSoftmaxNumRows", [HIDDEN_DIM]),
        ("kSoftmaxNumCols", [N_LABELS]),
        ("kHiddenBiasNumRows", [HIDDEN_DIM]),
        ("kSoftmaxBiasNumRows", [N_LABELS]),
    ]:
        if k not in arrays:
            raise SystemExit(f"{k} missing from upstream")
        got = arrays[k].tolist()
        if got != expected:
            raise SystemExit(
                f"upstream {k}={got} but converter expects {expected}; "
                f"refusing to dump (constants drifted, re-verify the port)",
            )

    # Dequantize embeddings: (uint8 - 128) * scale[row] per
    # embedding_network.cc:122-123.
    embeddings_f32: list[np.ndarray] = []
    for i, (rows, cols) in enumerate(EMBEDDING_SHAPES):
        u8 = arrays[f"kEmbeddingsWeights{i}"]
        sc = arrays[f"kEmbeddingsQuantScales{i}"]
        if u8.size != rows * cols:
            raise SystemExit(
                f"kEmbeddingsWeights{i}: have {u8.size} expected {rows*cols}",
            )
        if sc.size != rows:
            raise SystemExit(
                f"kEmbeddingsQuantScales{i}: have {sc.size} expected {rows}",
            )
        u8_2d = u8.reshape(rows, cols)
        sc_f32 = _bf16u_to_f32(sc).reshape(rows, 1)
        emb = (u8_2d.astype(np.int32) - 128).astype(np.float32) * sc_f32
        embeddings_f32.append(emb.astype(np.float32, copy=False))
        print(
            f"  embedding[{i}] {FEATURE_NAMES[i]:<18} "
            f"shape={emb.shape} "
            f"min={emb.min():+.4f} max={emb.max():+.4f} mean={emb.mean():+.4f}"
        )

    # Hidden FC: stored [in=80, out=208]. GGUF expects ggml_mul_mat(W, x)=y
    # where W is [out, in], so transpose.
    hw_in_out = arrays["kHiddenWeights0"].reshape(CONCAT_DIM, HIDDEN_DIM)
    hidden_w = hw_in_out.T.copy()  # [208, 80]
    hidden_b = arrays["kHiddenBiasWeights0"].reshape(HIDDEN_DIM)

    sw_in_out = arrays["kSoftmaxWeights0"].reshape(HIDDEN_DIM, N_LABELS)
    softmax_w = sw_in_out.T.copy()  # [109, 208]
    softmax_b = arrays["kSoftmaxBiasWeights0"].reshape(N_LABELS)

    print(
        f"  hidden_w  shape={hidden_w.shape} "
        f"min={hidden_w.min():+.4f} max={hidden_w.max():+.4f}"
    )
    print(
        f"  softmax_w shape={softmax_w.shape} "
        f"min={softmax_w.min():+.4f} max={softmax_w.max():+.4f}"
    )

    # Cast quantizable tensors to F16 if requested. Biases stay F32 — they're
    # tiny (1.3 KB combined) and noise-sensitive in shallow classifiers.
    weight_dtype = np.float16 if write_f16 else np.float32
    embeddings_out = [emb.astype(weight_dtype, copy=False) for emb in embeddings_f32]
    hidden_w_out = hidden_w.astype(weight_dtype, copy=False)
    softmax_w_out = softmax_w.astype(weight_dtype, copy=False)

    # ---------------------------------------------------------------------
    # GGUF write.
    # ---------------------------------------------------------------------
    print(f"Writing: {out_path}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = gguf.GGUFWriter(str(out_path), arch="lid-cld3")

    # `arch=` already writes general.architecture — used by the auto-routing
    # text-LID dispatcher to pick lid_cld3_* vs lid_fasttext_* loaders.
    writer.add_string("general.name", "Google CLD3")
    writer.add_string("general.license", "Apache-2.0")
    writer.add_string("general.source.url", "https://github.com/google/cld3")

    writer.add_uint32("lid_cld3.dim_total", CONCAT_DIM)
    writer.add_uint32("lid_cld3.hidden_dim", HIDDEN_DIM)
    writer.add_uint32("lid_cld3.n_features", len(EMBEDDING_SHAPES))
    writer.add_uint32("lid_cld3.n_labels", N_LABELS)
    writer.add_array(
        "lid_cld3.feature_rows",
        [r for r, _ in EMBEDDING_SHAPES],
    )
    writer.add_array(
        "lid_cld3.feature_cols",
        [c for _, c in EMBEDDING_SHAPES],
    )
    writer.add_array("lid_cld3.feature_offsets", CONCAT_OFFSETS)
    writer.add_array("lid_cld3.feature_kinds", FEATURE_KINDS)
    writer.add_array("lid_cld3.feature_names", FEATURE_NAMES)
    writer.add_array("lid_cld3.feature_ngram_sizes", FEATURE_NGRAM_SIZES)
    writer.add_array("lid_cld3.labels", languages)

    # Tensors. Names use the `.weight` suffix on every quantizable tensor so
    # crispasr-quantize's is_weight gate matches them — see LEARNINGS.md
    # "crispasr-quantize is_weight gate" for the silent-no-op trap.
    for i, emb in enumerate(embeddings_out):
        writer.add_tensor(f"lid_cld3.embedding.{i}.weight", emb)
    writer.add_tensor("lid_cld3.hidden.weight", hidden_w_out)
    writer.add_tensor("lid_cld3.hidden.bias", hidden_b.astype(np.float32))
    writer.add_tensor("lid_cld3.output.weight", softmax_w_out)
    writer.add_tensor("lid_cld3.output.bias", softmax_b.astype(np.float32))

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    sz = out_path.stat().st_size
    print(f"Wrote {out_path} ({sz/1e6:.2f} MB)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--upstream-src",
        type=Path,
        default=Path("/Volumes/backups/ai/upstream/cld3/src"),
        help="Path to google/cld3's src/ directory.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("/Volumes/backups/ai/crispasr-models/lid-cld3/cld3-f16.gguf"),
        help="Output GGUF path.",
    )
    parser.add_argument(
        "--f32",
        action="store_true",
        help="Write F32 weights (default: F16). Biases are always F32.",
    )
    args = parser.parse_args()
    convert(args.upstream_src, args.out, write_f16=not args.f32)


if __name__ == "__main__":
    main()
