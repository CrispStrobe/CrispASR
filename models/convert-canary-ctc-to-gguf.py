#!/usr/bin/env python3
"""
Convert the auxiliary CTC timestamp model that ships inside Canary 1B v2's
.nemo tarball to a GGUF that can be loaded by `parakeet-main` (or by a
forced-alignment helper) to recover word-level timestamps for canary's
output via Viterbi forced alignment.

What's in canary-1b-v2.nemo:
  - model_weights.ckpt                  ← the main encoder-decoder canary model
  - timestamps_asr_model_weights.ckpt   ← THIS file: a separate Parakeet-style
                                          24-layer FastConformer encoder + a
                                          Conv1d CTC head over canary's vocab
  - timestamps_asr_model_config.yaml    ← config for the aux model

The aux model is a fully-trained 600M-parameter CTC ASR model in its own
right. NeMo Forced Aligner (NFA) uses it to compute frame-aligned word
timestamps for canary's transcript output. Per the Canary paper section 5
this is the official path for getting reliable word/segment timestamps.

Architecture (from inspection):
  Encoder: 24-layer FastConformer (no biases on q/k/v/out/ff — same as
           parakeet-tdt-0.6b-v3, NOT canary which has biases on everything)
  CTC head: Conv1d(1024, 16385, kernel=1)  (16384 vocab + 1 blank)
  Vocab:    16384 SentencePiece tokens, same family as canary's main
            tokenizer (separate file inside the .nemo)
  Audio:    same NeMo mel preprocessor as canary/parakeet

GGUF schema: identical to parakeet's encoder, plus two extra tensors
for the CTC head:
  encoder.* (29 tensors per layer × 24 = 696, plus pre-encode + preprocessor)
  ctc.weight   (16385, 1024)   F16
  ctc.bias     (16385,)         F32

The runtime can then load this GGUF through parakeet's encoder loader
(it's the same architecture) and run a CTC matmul on the encoder output
to get [T_enc × 16385] per-frame logits, ready for Viterbi alignment.

Usage:
    python models/convert-canary-ctc-to-gguf.py \\
        --nemo  path/to/canary-1b-v2.nemo \\
        --output canary-ctc-aligner.gguf
"""

from __future__ import annotations

import argparse
import sys
import tarfile
import tempfile
from pathlib import Path

import numpy as np

try:
    import gguf
except ImportError:
    sys.exit("pip install gguf")
try:
    import torch
except ImportError:
    sys.exit("pip install torch")
try:
    import sentencepiece as spm
except ImportError:
    sys.exit("pip install sentencepiece")


# ---------------------------------------------------------------------------
# .nemo unpacking — extract ONLY the timestamps_asr_model files
# ---------------------------------------------------------------------------


def unpack_nemo_aux(nemo_path: Path, out_dir: Path) -> dict:
    """Extract the aux CTC model files from a canary .nemo tarball."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    with tarfile.open(nemo_path, "r") as tf:
        for m in tf.getmembers():
            n = m.name.lstrip("./")
            if n == "timestamps_asr_model_weights.ckpt":
                tf.extract(m, out_dir, filter="data")
                paths["weights"] = out_dir / m.name.lstrip("./")
            elif n == "timestamps_asr_model_config.yaml":
                tf.extract(m, out_dir, filter="data")
                paths["config"] = out_dir / m.name.lstrip("./")
            elif n.endswith("_tokenizer.model") and "spe_tokenizer" not in n:
                # Pick the tokenizer that matches the aux model's vocab.
                # In canary-1b-v2.nemo there are multiple tokenizer files;
                # we want the one referenced by the aux model's config.
                # Easiest heuristic: take all and filter later.
                tf.extract(m, out_dir, filter="data")
                paths.setdefault("spm_candidates", []).append(
                    out_dir / m.name.lstrip("./")
                )
    if "weights" not in paths:
        sys.exit(f"could not find timestamps_asr_model_weights.ckpt in {nemo_path}")
    return paths


# ---------------------------------------------------------------------------
# Tensor name remapping (mostly identical to convert-parakeet-to-gguf.py)
# ---------------------------------------------------------------------------


def remap_name(nemo_name: str) -> str | None:
    n = nemo_name
    if n.endswith("num_batches_tracked"):
        return None

    # ---- preprocessor ----
    if n == "preprocessor.featurizer.fb":
        return "preprocessor.fb"
    if n == "preprocessor.featurizer.window":
        return "preprocessor.window"

    # ---- pre-encoder ----
    if n.startswith("encoder.pre_encode."):
        return n.replace("encoder.pre_encode.", "encoder.pre.")

    # ---- encoder layers ----
    if n.startswith("encoder.layers."):
        rest = n[len("encoder.layers.") :]
        layer_id, sub = rest.split(".", 1)
        sub = (
            sub.replace("feed_forward1", "ff1")
            .replace("feed_forward2", "ff2")
            .replace("norm_feed_forward1", "norm_ff1")
            .replace("norm_feed_forward2", "norm_ff2")
            .replace("norm_self_att", "norm_attn")
            .replace("self_attn.linear_q", "attn.q")
            .replace("self_attn.linear_k", "attn.k")
            .replace("self_attn.linear_v", "attn.v")
            .replace("self_attn.linear_out", "attn.out")
            .replace("self_attn.linear_pos", "attn.pos")
            .replace("self_attn.pos_bias_u", "attn.pos_bias_u")
            .replace("self_attn.pos_bias_v", "attn.pos_bias_v")
            .replace("conv.pointwise_conv1", "conv.pw1")
            .replace("conv.depthwise_conv", "conv.dw")
            .replace("conv.pointwise_conv2", "conv.pw2")
            .replace("conv.batch_norm", "conv.bn")
        )
        return f"encoder.layers.{layer_id}.{sub}"

    # ---- CTC head ----
    # NeMo's CTC head is `decoder.decoder_layers.0.{weight,bias}`
    # weight has shape (vocab+1, hidden, kernel=1), so we squeeze the last dim
    if n == "decoder.decoder_layers.0.weight":
        return "ctc.weight"
    if n == "decoder.decoder_layers.0.bias":
        return "ctc.bias"

    print(f"  [warn] unmapped tensor: {n}", file=sys.stderr)
    return None


def is_f32_tensor(gguf_name: str, shape: tuple[int, ...]) -> bool:
    if gguf_name.startswith("preprocessor."):
        return True
    if gguf_name.endswith(".bias"):
        return True
    if "norm" in gguf_name:
        return True
    if "bn" in gguf_name:
        return True
    if "pos_bias_u" in gguf_name or "pos_bias_v" in gguf_name:
        return True
    if len(shape) <= 1:
        return True
    return False


# ---------------------------------------------------------------------------
# Main conversion
# ---------------------------------------------------------------------------


def convert(nemo_path: Path, out_path: Path) -> None:
    print(f"Loading: {nemo_path}")
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        paths = unpack_nemo_aux(nemo_path, td_path)
        print(f"  weights: {paths['weights']}")

        sd = torch.load(str(paths["weights"]), map_location="cpu", weights_only=True)
        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]

        # Pick the SentencePiece tokenizer that matches the CTC head vocab size.
        # CTC head shape is (vocab+1, 1024, 1), so we want vocab_size = head[0]-1.
        target_vocab = int(sd["decoder.decoder_layers.0.weight"].shape[0]) - 1
        spm_path = None
        for cand in paths.get("spm_candidates", []):
            try:
                sp = spm.SentencePieceProcessor(model_file=str(cand))
                if sp.get_piece_size() == target_vocab:
                    spm_path = cand
                    break
            except Exception:
                continue
        if spm_path is None:
            print(f"  [warn] no SentencePiece tokenizer matches vocab={target_vocab}")
            print("         CTC head will work but the embedded vocab may be wrong")
            # Pick the first as fallback
            spm_path = paths.get("spm_candidates", [None])[0]

        if spm_path is not None:
            sp = spm.SentencePieceProcessor(model_file=str(spm_path))
            vocab = [sp.id_to_piece(i) for i in range(sp.get_piece_size())]
            print(f"  spm:     {spm_path.name}  ({len(vocab)} pieces)")
        else:
            vocab = [f"<{i}>" for i in range(target_vocab)]
            print("  spm:     none — using placeholder vocab")

    # ----- write GGUF -----
    print(f"\nWriting: {out_path}")
    writer = gguf.GGUFWriter(str(out_path), arch="canary-ctc")

    # Hyperparameters (parakeet-style encoder + CTC head)
    writer.add_uint32("canary_ctc.sample_rate", 16000)
    writer.add_uint32("canary_ctc.n_mels", 128)
    writer.add_uint32("canary_ctc.n_fft", 512)
    writer.add_uint32("canary_ctc.win_length", 400)
    writer.add_uint32("canary_ctc.hop_length", 160)
    writer.add_uint32("canary_ctc.d_model", 1024)
    writer.add_uint32("canary_ctc.n_layers", 24)
    writer.add_uint32("canary_ctc.n_heads", 8)
    writer.add_uint32("canary_ctc.head_dim", 128)
    writer.add_uint32("canary_ctc.ff_dim", 4096)
    writer.add_uint32("canary_ctc.subsampling_factor", 8)
    writer.add_uint32("canary_ctc.subsampling_channels", 256)
    writer.add_uint32("canary_ctc.conv_kernel", 9)
    writer.add_uint32("canary_ctc.vocab_size", len(vocab))
    writer.add_uint32("canary_ctc.blank_id", len(vocab))  # blank is the last index
    writer.add_uint32("canary_ctc.frame_dur_cs", 8)  # 80 ms / encoder frame

    writer.add_array("tokenizer.ggml.tokens", vocab)

    n_written = 0
    n_f16 = 0
    n_f32 = 0
    layers_seen = set()

    for name in sorted(sd.keys()):
        gguf_name = remap_name(name)
        if gguf_name is None:
            continue
        t = sd[name].cpu().numpy()
        if t.dtype == np.float64:
            t = t.astype(np.float32)

        # Squeeze the trailing kernel-1 dimension on the CTC weight so it
        # becomes a plain 2D linear (vocab+1, hidden) instead of 3D Conv1d.
        if gguf_name == "ctc.weight" and t.ndim == 3 and t.shape[-1] == 1:
            t = t.squeeze(-1)

        if is_f32_tensor(gguf_name, t.shape):
            t = t.astype(np.float32)
            n_f32 += 1
        else:
            t = t.astype(np.float16)
            n_f16 += 1

        writer.add_tensor(gguf_name, t)
        n_written += 1
        if n_written <= 25 or n_written % 100 == 0:
            print(f"  {gguf_name:60s}  {str(t.shape):28s}  {t.dtype}")

        if gguf_name.startswith("encoder.layers.") and ".conv.dw.weight" in gguf_name:
            li = int(gguf_name.split(".")[2])
            layers_seen.add(li)

    # Inject zero conv.dw.bias per encoder layer (BN-fold target). Same as
    # convert-parakeet-to-gguf.py — the runtime folds BN into the depthwise
    # conv at load time and writes the absorbed bias here.
    for li in sorted(layers_seen):
        bias = np.zeros(1024, dtype=np.float32)
        writer.add_tensor(f"encoder.layers.{li}.conv.dw.bias", bias)
        n_written += 1
        n_f32 += 1

    print(
        f"\n  total tensors: {n_written}  (F16: {n_f16}, F32: {n_f32})  "
        f"(+{len(layers_seen)} synthetic conv.dw.bias)"
    )

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"\nDone: {out_path}  ({out_path.stat().st_size / 1e6:.1f} MB)")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Extract Canary 1B v2's auxiliary CTC alignment model and convert to GGUF F16"
    )
    p.add_argument("--nemo", required=True, type=Path, help="path to canary-1b-v2.nemo")
    p.add_argument("--output", required=True, type=Path, help="output GGUF path")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    convert(args.nemo, args.output)
