#!/usr/bin/env python3
"""Convert NeMo FastConformer CTC checkpoints to GGUF.

Handles the NeMo ``stt_*_fastconformer_ctc_*`` family of standalone ASR
models — currently smoke-tested with ``nvidia/stt_en_fastconformer_ctc_large``
but the same architecture applies to every FastConformer-CTC release:

  - input        = 16 kHz mono PCM, 80 log-mel features
  - pre-encoder  = dw_striding ×8 subsampling (5 × Conv2d + Linear)
  - encoder body = 18 FastConformer blocks (Macaron FFN + rel-pos MHSA
                    with untied Transformer-XL biases + Conformer conv
                    + Macaron FFN), d_model=512, n_heads=8, head_dim=64,
                    ff_dim=2048, conv_kernel=9
  - head         = Conv1d(d_model → vocab, k=1) CTC classifier
  - vocab        = SentencePiece unigram 1024 tokens

The model architecture is identical to our existing canary_ctc aligner
encoder *except* for the presence of biases on every Q/K/V/out/FFN linear
and conv.pw1/pw2 layer (canary_ctc is bias-less; FastConformer-CTC has
biases). src/canary_ctc.cpp now loads those biases optionally so the
same runtime hosts both bias-less canary_ctc aligners and bias-carrying
FastConformer-CTC standalones — the GGUF arch tag stays "canary-ctc" and
the C++ loader dispatches automatically based on what's actually in the
tensor table.

Usage:
  python models/convert-stt-fastconformer-ctc-to-gguf.py \\
      --nemo  /path/to/stt_en_fastconformer_ctc_large.nemo \\
      --output models/stt-en-fastconformer-ctc-large-f16.gguf
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


def unpack_nemo(nemo_path: Path, out_dir: Path) -> dict:
    """Extract model_weights.ckpt + model_config.yaml + tokenizer.model."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    with tarfile.open(nemo_path, "r") as tf:
        for m in tf.getmembers():
            n = m.name.lstrip("./")
            if n == "model_weights.ckpt":
                tf.extract(m, out_dir, filter="data")
                paths["weights"] = out_dir / m.name.lstrip("./")
            elif n == "model_config.yaml":
                tf.extract(m, out_dir, filter="data")
                paths["config"] = out_dir / m.name.lstrip("./")
            elif n.endswith("_tokenizer.model"):
                tf.extract(m, out_dir, filter="data")
                paths.setdefault("spm_candidates", []).append(
                    out_dir / m.name.lstrip("./")
                )
    if "weights" not in paths:
        sys.exit(f"could not find model_weights.ckpt in {nemo_path}")
    return paths


def remap_name(nemo_name: str) -> str | None:
    """NeMo state-dict name → canary-ctc GGUF tensor name."""
    n = nemo_name
    if n.endswith("num_batches_tracked"):
        return None
    if n == "preprocessor.featurizer.fb":
        return "preprocessor.fb"
    if n == "preprocessor.featurizer.window":
        return "preprocessor.window"
    if n.startswith("encoder.pre_encode."):
        return n.replace("encoder.pre_encode.", "encoder.pre.")
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
    # NeMo CTC head: `decoder.decoder_layers.0.{weight,bias}` with
    # weight shape (vocab+1, hidden, kernel=1). We squeeze the last dim
    # during tensor write.
    if n == "decoder.decoder_layers.0.weight":
        return "ctc.weight"
    if n == "decoder.decoder_layers.0.bias":
        return "ctc.bias"
    print(f"  [warn] unmapped tensor: {n}", file=sys.stderr)
    return None


def is_f32_tensor(name: str, shape: tuple[int, ...]) -> bool:
    if name.startswith("preprocessor."):
        return True
    if name.endswith(".bias"):
        return True
    if "norm" in name:
        return True
    if "bn" in name:
        return True
    if "pos_bias_u" in name or "pos_bias_v" in name:
        return True
    if len(shape) <= 1:
        return True
    return False


def parse_yaml_hparams(yaml_path: Path) -> dict:
    """Extract the handful of hparams we actually need from model_config.yaml.

    We don't want a PyYAML dependency for a toy parser — the NeMo configs
    are line-based and very regular for the keys we care about.
    """
    hp = {}
    with open(yaml_path, encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if ":" not in s:
                continue
            k, _, v = s.partition(":")
            k = k.strip()
            v = v.strip().split("#", 1)[0].strip()
            if not v:
                continue
            if v.lstrip("-").isdigit():
                hp[k] = int(v)
            elif v.replace(".", "", 1).replace("e-", "", 1).lstrip("-").isdigit():
                try:
                    hp[k] = float(v)
                except ValueError:
                    pass
    return hp


def convert(nemo_path: Path, out_path: Path) -> None:
    print(f"Loading: {nemo_path}")
    with tempfile.TemporaryDirectory() as td:
        paths = unpack_nemo(nemo_path, Path(td))
        print(f"  weights: {paths['weights']}")

        sd = torch.load(str(paths["weights"]), map_location="cpu", weights_only=True)
        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]

        hp = parse_yaml_hparams(paths["config"]) if "config" in paths else {}

        # Infer vocab size from the CTC head (most authoritative source).
        ctc_w = sd["decoder.decoder_layers.0.weight"]  # (vocab+1, hidden, 1)
        vocab_plus_blank = int(ctc_w.shape[0])
        vocab_size = vocab_plus_blank - 1
        hidden = int(ctc_w.shape[1])

        # Load the SentencePiece vocab.
        spm_path = None
        for cand in paths.get("spm_candidates", []):
            try:
                sp = spm.SentencePieceProcessor(model_file=str(cand))
                if sp.get_piece_size() == vocab_size:
                    spm_path = cand
                    break
            except Exception:
                continue
        if spm_path is None:
            sys.exit(
                f"could not find a SentencePiece tokenizer with {vocab_size} pieces"
            )
        sp = spm.SentencePieceProcessor(model_file=str(spm_path))
        vocab = [sp.id_to_piece(i) for i in range(sp.get_piece_size())]
        print(f"  spm:     {spm_path.name} ({len(vocab)} pieces)")

        # Encoder hparams from the YAML (with sane defaults for the Large variant).
        n_layers = hp.get("n_layers", 18)
        d_model = hp.get("d_model", hidden)
        n_heads = hp.get("n_heads", 8)
        head_dim = d_model // n_heads
        ff_dim = d_model * hp.get("ff_expansion_factor", 4)
        sub_factor = hp.get("subsampling_factor", 8)
        sub_channels = hp.get("subsampling_conv_channels", 256)
        conv_kernel = hp.get("conv_kernel_size", 9)
        n_mels = hp.get("features", 80)
        n_fft = hp.get("n_fft", 512)
        win_length = int(hp.get("window_size", 0.025) * hp.get("sample_rate", 16000))
        hop_length = int(hp.get("window_stride", 0.01) * hp.get("sample_rate", 16000))

        print(
            f"  encoder: n_layers={n_layers} d_model={d_model} heads={n_heads}"
            f" head_dim={head_dim} ff_dim={ff_dim} mels={n_mels}"
        )
        print(f"  ctc head: vocab={vocab_size} (+1 blank)")

        out_path.parent.mkdir(parents=True, exist_ok=True)
        # arch tag stays "canary-ctc" so src/canary_ctc.cpp hosts the
        # result unchanged — the bias-carrying branch is selected at
        # load time based on tensor presence, not on the arch tag.
        writer = gguf.GGUFWriter(str(out_path), arch="canary-ctc", use_temp_file=True)

        writer.add_uint32("canary_ctc.sample_rate", 16000)
        writer.add_uint32("canary_ctc.n_mels", n_mels)
        writer.add_uint32("canary_ctc.n_fft", n_fft)
        writer.add_uint32("canary_ctc.win_length", win_length)
        writer.add_uint32("canary_ctc.hop_length", hop_length)
        writer.add_uint32("canary_ctc.d_model", d_model)
        writer.add_uint32("canary_ctc.n_layers", n_layers)
        writer.add_uint32("canary_ctc.n_heads", n_heads)
        writer.add_uint32("canary_ctc.head_dim", head_dim)
        writer.add_uint32("canary_ctc.ff_dim", ff_dim)
        writer.add_uint32("canary_ctc.subsampling_factor", sub_factor)
        writer.add_uint32("canary_ctc.subsampling_channels", sub_channels)
        writer.add_uint32("canary_ctc.conv_kernel", conv_kernel)
        writer.add_uint32("canary_ctc.vocab_size", vocab_size)
        writer.add_uint32("canary_ctc.blank_id", vocab_size)
        writer.add_uint32("canary_ctc.frame_dur_cs", 8)
        # NeMo FastConformer-CTC standalones are trained with
        # xscaling=true — the runtime multiplies the pre-encoder output
        # by sqrt(d_model) before the first block. The canary_ctc
        # aligner is the only other consumer of this loader and was
        # trained with xscaling=false; its GGUF omits the key and the
        # C++ default of 0 keeps that path unchanged.
        writer.add_uint32("canary_ctc.xscaling", 1)

        writer.add_array("tokenizer.ggml.tokens", vocab)

        n_written = 0
        n_f16 = n_f32 = 0
        for name in sorted(sd.keys()):
            gguf_name = remap_name(name)
            if gguf_name is None:
                continue
            t = sd[name].cpu().numpy()
            if t.dtype == np.float64:
                t = t.astype(np.float32)
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

        # NOTE: stt_en_fc_ctc already ships `conv.depthwise_conv.bias`
        # (renamed to `encoder.layers.*.conv.dw.bias` by remap_name), so
        # we don't need to inject zero bias tensors here the way the
        # canary_ctc aligner converter does. The runtime's BN folding
        # writes the absorbed bias back into this slot at load time.
        print(f"\n  total tensors: {n_written}  (F16: {n_f16}, F32: {n_f32})")

        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()
    print(f"\nDone: {out_path}  ({out_path.stat().st_size / 1e6:.1f} MB)")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Convert a NeMo FastConformer-CTC .nemo checkpoint to GGUF"
    )
    p.add_argument("--nemo", required=True, type=Path, help="path to the .nemo tarball")
    p.add_argument("--output", required=True, type=Path)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    convert(args.nemo, args.output)
