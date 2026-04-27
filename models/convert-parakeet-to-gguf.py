#!/usr/bin/env python3
"""
Convert nvidia/parakeet-tdt-0.6b-v3 (a NeMo .nemo checkpoint) → GGUF F16.

Architecture (from model_config.yaml + tensor inspection):

  preprocessor (mel):     128 mel bins, n_fft=512, win=25ms, stride=10ms (Hann)
  encoder (24× FastConformer):
    pre_encode:           3-stage dw_striding Conv2d (8× time downsample, 128→16 freq)
                          out: linear(4096 → 1024)
    layer i:              FFN1(½) → MHA(rel_pos, untied bias) → conv(dw, k=9, BN) → FFN2(½) → LN
    d_model = 1024  n_heads = 8  ff = 4096  v = 8192
  decoder.prediction:     embed(8193, 640) + 2-layer LSTM(640, 640)
  joint:                  enc(1024→640) + pred(640→640) → tanh → linear(640 → 8198)
                          8198 = 8192 vocab + 1 blank + 5 TDT durations {0,1,2,3,4}

GGUF tensor naming (mirrors what the C++ loader will expect):

  preprocessor.fb                                    F32  (128, 257)
  preprocessor.window                                F32  (400,)

  encoder.pre.conv.{0,2,3,5,6}.{weight,bias}         F16/F32
  encoder.pre.out.{weight,bias}                      F16/F32

  encoder.layers.{i}.norm_ff1.{weight,bias}          F32
  encoder.layers.{i}.ff1.linear1.weight              F16
  encoder.layers.{i}.ff1.linear2.weight              F16
  encoder.layers.{i}.ff1.linear1.bias                F32   (zero in this model — bias-less FF)
  ... (analogous for ff2)

  encoder.layers.{i}.norm_attn.{weight,bias}         F32
  encoder.layers.{i}.attn.{q,k,v,out,pos}.weight     F16
  encoder.layers.{i}.attn.pos_bias_u                 F32  (8, 128)
  encoder.layers.{i}.attn.pos_bias_v                 F32  (8, 128)

  encoder.layers.{i}.norm_conv.{weight,bias}         F32
  encoder.layers.{i}.conv.pw1.weight                 F16  (2048, 1024, 1)
  encoder.layers.{i}.conv.dw.weight                  F16  (1024, 1, 9)
  encoder.layers.{i}.conv.bn.{weight,bias,running_mean,running_var}  F32
  encoder.layers.{i}.conv.pw2.weight                 F16  (1024, 1024, 1)

  encoder.layers.{i}.norm_ff2.{weight,bias}          F32
  encoder.layers.{i}.norm_out.{weight,bias}          F32

  decoder.embed.weight                               F16  (8193, 640)
  decoder.lstm.{0,1}.{w_ih,w_hh,b_ih,b_hh}           F16/F32

  joint.enc.{weight,bias}                            F16/F32
  joint.pred.{weight,bias}                           F16/F32
  joint.out.{weight,bias}                            F16/F32

GGUF metadata keys (under `parakeet.*`):
  parakeet.sample_rate          = 16000
  parakeet.n_mels               = 128
  parakeet.n_fft                = 512
  parakeet.win_length           = 400
  parakeet.hop_length           = 160
  parakeet.frame_dur_ms         = 80   (10 ms × 8× subsampling)
  parakeet.d_model              = 1024
  parakeet.n_layers             = 24
  parakeet.n_heads              = 8
  parakeet.head_dim             = 128
  parakeet.ff_dim               = 4096
  parakeet.subsampling_factor   = 8
  parakeet.subsampling_channels = 256
  parakeet.conv_kernel          = 9
  parakeet.pred_hidden          = 640
  parakeet.pred_layers          = 2
  parakeet.joint_hidden         = 640
  parakeet.vocab_size           = 8192
  parakeet.blank_id             = 8192
  parakeet.n_tdt_durations      = 5
  parakeet.tdt_durations        = [0, 1, 2, 3, 4]

  tokenizer.ggml.tokens         = [<8192 strings from SentencePiece>]
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
# .nemo unpacking
# ---------------------------------------------------------------------------


def unpack_nemo(nemo_path: Path, out_dir: Path) -> dict:
    """Extract .nemo tarball, return paths to weights / config / tokenizer."""
    out_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(nemo_path, "r") as tf:
        tf.extractall(out_dir)
    paths = {}
    for f in out_dir.iterdir():
        n = f.name
        if n.endswith("model_weights.ckpt"):
            paths["weights"] = f
        elif n.endswith("model_config.yaml"):
            paths["config"] = f
        elif n.endswith("_tokenizer.model"):
            paths["spm"] = f
        elif n.endswith("_vocab.txt"):
            paths["vocab"] = f
    if "weights" not in paths or "spm" not in paths:
        sys.exit(f"could not find weights / tokenizer in {nemo_path}")
    return paths


# ---------------------------------------------------------------------------
# Tensor name remapping
# ---------------------------------------------------------------------------


def remap_name(nemo_name: str) -> str | None:
    """
    Map NeMo state-dict keys to GGUF-friendly names.
    Returns None for tensors we deliberately drop (e.g. num_batches_tracked).
    """
    n = nemo_name

    # Skip BatchNorm stats counter
    if n.endswith("num_batches_tracked"):
        return None

    # ---- preprocessor (mel filterbank + Hann window) ----
    if n == "preprocessor.featurizer.fb":
        return "preprocessor.fb"
    if n == "preprocessor.featurizer.window":
        return "preprocessor.window"

    # ---- pre-encoder (subsampling Conv2d stack) ----
    if n.startswith("encoder.pre_encode."):
        # encoder.pre_encode.conv.{0,2,3,5,6}.{weight,bias}
        # encoder.pre_encode.out.{weight,bias}
        return n.replace("encoder.pre_encode.", "encoder.pre.")

    # ---- conformer layers ----
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

    # ---- decoder (predictor) ----
    if n == "decoder.prediction.embed.weight":
        return "decoder.embed.weight"
    if n.startswith("decoder.prediction.dec_rnn.lstm."):
        suf = n[len("decoder.prediction.dec_rnn.lstm.") :]
        # weight_ih_l0 / weight_hh_l0 / bias_ih_l0 / bias_hh_l0
        for key, gguf_key in [
            ("weight_ih_l0", "lstm.0.w_ih"),
            ("weight_hh_l0", "lstm.0.w_hh"),
            ("bias_ih_l0", "lstm.0.b_ih"),
            ("bias_hh_l0", "lstm.0.b_hh"),
            ("weight_ih_l1", "lstm.1.w_ih"),
            ("weight_hh_l1", "lstm.1.w_hh"),
            ("bias_ih_l1", "lstm.1.b_ih"),
            ("bias_hh_l1", "lstm.1.b_hh"),
        ]:
            if suf == key:
                return f"decoder.{gguf_key}"

    # ---- joint ----
    if n == "joint.enc.weight":
        return "joint.enc.weight"
    if n == "joint.enc.bias":
        return "joint.enc.bias"
    if n == "joint.pred.weight":
        return "joint.pred.weight"
    if n == "joint.pred.bias":
        return "joint.pred.bias"
    if n == "joint.joint_net.2.weight":
        return "joint.out.weight"
    if n == "joint.joint_net.2.bias":
        return "joint.out.bias"

    print(f"  [warn] unmapped tensor: {n}", file=sys.stderr)
    return None


# Tensors that should stay F32 even when --quant-linear is set:
# layer norms, batch-norm stats, biases, the mel filterbank, the rel-pos biases.
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


def model_d_for_layer(sd, layer_id: int) -> int:
    """Look up d_model for the given encoder layer (1024 for parakeet-tdt-0.6b-v3)."""
    key = f"encoder.layers.{layer_id}.conv.depthwise_conv.weight"
    return int(sd[key].shape[0])


def convert(nemo_path: Path, out_path: Path) -> None:
    print(f"Loading: {nemo_path}")
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        paths = unpack_nemo(nemo_path, td_path)
        print(f"  config: {paths['config']}")
        print(f"  spm:    {paths['spm']}")

        sd = torch.load(str(paths["weights"]), map_location="cpu", weights_only=True)
        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]

        # Load tokenizer
        sp = spm.SentencePieceProcessor(model_file=str(paths["spm"]))
        vocab = [sp.id_to_piece(i) for i in range(sp.get_piece_size())]
        print(f"  vocab:  {len(vocab)} pieces")

    # ----- write GGUF -----
    print(f"Writing: {out_path}")
    writer = gguf.GGUFWriter(str(out_path), arch="parakeet")

    # Hyper-parameters (hard-coded for parakeet-tdt-0.6b-v3 based on config inspection)
    writer.add_uint32("parakeet.sample_rate", 16000)
    writer.add_uint32("parakeet.n_mels", 128)
    writer.add_uint32("parakeet.n_fft", 512)
    writer.add_uint32("parakeet.win_length", 400)
    writer.add_uint32("parakeet.hop_length", 160)
    writer.add_uint32("parakeet.d_model", 1024)
    writer.add_uint32("parakeet.n_layers", 24)
    writer.add_uint32("parakeet.n_heads", 8)
    writer.add_uint32("parakeet.head_dim", 128)
    writer.add_uint32("parakeet.ff_dim", 4096)
    writer.add_uint32("parakeet.subsampling_factor", 8)
    writer.add_uint32("parakeet.subsampling_channels", 256)
    writer.add_uint32("parakeet.conv_kernel", 9)
    writer.add_uint32("parakeet.pred_hidden", 640)
    writer.add_uint32("parakeet.pred_layers", 2)
    writer.add_uint32("parakeet.joint_hidden", 640)
    writer.add_uint32("parakeet.vocab_size", len(vocab))
    writer.add_uint32("parakeet.blank_id", len(vocab))  # blank is vocab_size
    writer.add_uint32("parakeet.n_tdt_durations", 5)
    writer.add_array("parakeet.tdt_durations", [0, 1, 2, 3, 4])
    writer.add_uint32("parakeet.frame_dur_cs", 8)  # 80 ms = 8 cs

    writer.add_array("tokenizer.ggml.tokens", vocab)

    # ----- tensors -----
    n_written = 0
    n_f16 = 0
    n_f32 = 0
    layers_seen = set()
    layers_with_dw_bias = set()
    for name in sorted(sd.keys()):
        gguf_name = remap_name(name)
        if gguf_name is None:
            continue
        t = sd[name].cpu().numpy()
        if t.dtype == np.float64:
            t = t.astype(np.float32)

        if is_f32_tensor(gguf_name, t.shape):
            t = t.astype(np.float32)
            n_f32 += 1
        else:
            t = t.astype(np.float16)
            n_f16 += 1

        writer.add_tensor(gguf_name, t)
        n_written += 1
        if n_written <= 30 or n_written % 50 == 0:
            print(f"  {gguf_name:60s}  {str(t.shape):28s}  {t.dtype}")

        # Track encoder layers so we can add a zero conv.dw.bias for each.
        if gguf_name.startswith("encoder.layers.") and ".conv.dw.weight" in gguf_name:
            li = int(gguf_name.split(".")[2])
            layers_seen.add(li)
        # Track layers that already have a dw.bias from the checkpoint.
        if gguf_name.startswith("encoder.layers.") and ".conv.dw.bias" in gguf_name:
            li = int(gguf_name.split(".")[2])
            layers_with_dw_bias.add(li)

    # Inject a zero-valued conv.dw.bias per encoder layer that doesn't already
    # have one. Older NeMo models have bias-less depthwise conv (BN provides the
    # bias term), but newer ones (parakeet-ja) have an explicit bias.
    for li in sorted(layers_seen):
        if li in layers_with_dw_bias:
            continue  # already has a real bias from the checkpoint
        bias = np.zeros(int(model_d_for_layer(sd, li)), dtype=np.float32)
        gguf_name = f"encoder.layers.{li}.conv.dw.bias"
        writer.add_tensor(gguf_name, bias)
        n_written += 1
        n_f32 += 1

    print(
        f"\n  total tensors: {n_written}  (F16: {n_f16}, F32: {n_f32})  "
        f"(+{len(layers_seen) - len(layers_with_dw_bias)} synthetic conv.dw.bias)"
    )

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"\nDone: {out_path}  ({out_path.stat().st_size / 1e6:.1f} MB)")


# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Convert Parakeet TDT .nemo → GGUF F16")
    p.add_argument("--nemo", required=True, type=Path, help="path to .nemo file")
    p.add_argument("--output", required=True, type=Path, help="output GGUF path")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    convert(args.nemo, args.output)
