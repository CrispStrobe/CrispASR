#!/usr/bin/env python3
"""
Convert nvidia/canary-1b-v2 (a NeMo .nemo checkpoint) → GGUF F16.

Architecture:

  preprocessor (mel):     128 mel bins, n_fft=512, win=25 ms, stride=10 ms (Hann)
  encoder (32× FastConformer):
    pre_encode:           3-stage dw_striding Conv2d (8× time downsample)
                          out: linear(4096 → 1024)
    layer i:              FFN1(½) → MHA(rel_pos, untied bias) → conv → FFN2(½) → LN
    d_model = 1024  n_heads = 8  ff = 4096  conv_k = 9  use_bias = True
  transf_decoder (8 layers):
    embedding:            token (16384, 1024) + learned pos_enc (1024, 1024) + LN
    layer i:              SA(causal, with bias) → CrossAttn → FFN → 3× LN
  log_softmax (output head):
    linear (1024 → 16384)

Decoder prompt format (mirrors Cohere — same vocab convention):
  <|startoftranscript|> <|src_lang|> <|target_lang|> <|nopnc|>|<|pnc|> <|notimestamp|> <|nodiarize|> ...

If src == target → ASR. If src != target → speech translation.

GGUF tensor naming (mirrors what the C++ loader will expect):

  preprocessor.fb                                      F32  (128, 257)
  preprocessor.window                                  F32  (400,)

  encoder.pre.conv.{0,2,3,5,6}.{weight,bias}           F32 / F16
  encoder.pre.out.{weight,bias}                        F16 / F32

  encoder.layers.{i}.norm_ff1.{weight,bias}            F32
  encoder.layers.{i}.ff1.linear1.{weight,bias}         F16 / F32
  encoder.layers.{i}.ff1.linear2.{weight,bias}         F16 / F32
  encoder.layers.{i}.norm_attn.{weight,bias}           F32
  encoder.layers.{i}.attn.{q,k,v,out}.{weight,bias}    F16 / F32
  encoder.layers.{i}.attn.pos.weight                   F16
  encoder.layers.{i}.attn.pos_bias_{u,v}               F32
  encoder.layers.{i}.norm_conv.{weight,bias}           F32
  encoder.layers.{i}.conv.pw1.{weight,bias}            F16 / F32
  encoder.layers.{i}.conv.dw.{weight,bias}             F16 / F32  (BN folded at load)
  encoder.layers.{i}.conv.pw2.{weight,bias}            F16 / F32
  encoder.layers.{i}.norm_ff2.{weight,bias}            F32
  encoder.layers.{i}.ff2.linear1.{weight,bias}         F16 / F32
  encoder.layers.{i}.ff2.linear2.{weight,bias}         F16 / F32
  encoder.layers.{i}.norm_out.{weight,bias}            F32

  decoder.embed.weight                                 F16  (16384, 1024)
  decoder.pos_enc                                      F32  (1024, 1024)
  decoder.embed_ln.{weight,bias}                       F32
  decoder.layers.{i}.norm_sa.{weight,bias}             F32
  decoder.layers.{i}.sa_{q,k,v,out}.{weight,bias}      F16 / F32
  decoder.layers.{i}.norm_ca.{weight,bias}             F32
  decoder.layers.{i}.ca_{q,k,v,out}.{weight,bias}      F16 / F32
  decoder.layers.{i}.norm_ff.{weight,bias}             F32
  decoder.layers.{i}.ff_in.{weight,bias}               F16 / F32
  decoder.layers.{i}.ff_out.{weight,bias}              F16 / F32
  decoder.final_norm.{weight,bias}                     F32

  decoder.head.{weight,bias}                           F16 / F32  (1024 → 16384)

GGUF metadata keys (under `canary.*`):
  canary.sample_rate          = 16000
  canary.n_mels               = 128
  canary.n_fft                = 512
  canary.win_length           = 400
  canary.hop_length           = 160
  canary.d_model              = 1024
  canary.enc_n_layers         = 32
  canary.dec_n_layers         = 8
  canary.n_heads              = 8
  canary.head_dim             = 128
  canary.ff_dim               = 4096
  canary.subsampling_factor   = 8
  canary.subsampling_channels = 256
  canary.conv_kernel          = 9
  canary.vocab_size           = 16384
  canary.max_dec_ctx          = 1024
  canary.frame_dur_cs         = 8

  tokenizer.ggml.tokens       = [<16384 strings from SentencePiece>]
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
    out_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(nemo_path, "r") as tf:
        tf.extractall(out_dir, filter="data")
    paths = {}
    for f in out_dir.iterdir():
        n = f.name
        # Skip the auxiliary timestamps_asr_model_weights.ckpt
        if n == "model_weights.ckpt":
            paths["weights"] = f
        elif n == "model_config.yaml":
            paths["config"] = f
        elif n.endswith("_tokenizer.model"):
            paths.setdefault("spm", f)  # multiple, take first
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

    # ---- decoder embedding ----
    if n == "transf_decoder._embedding.token_embedding.weight":
        return "decoder.embed.weight"
    if n == "transf_decoder._embedding.position_embedding.pos_enc":
        return "decoder.pos_enc"
    if n == "transf_decoder._embedding.layer_norm.weight":
        return "decoder.embed_ln.weight"
    if n == "transf_decoder._embedding.layer_norm.bias":
        return "decoder.embed_ln.bias"

    # ---- decoder layers ----
    if n.startswith("transf_decoder._decoder.layers."):
        rest = n[len("transf_decoder._decoder.layers.") :]
        layer_id, sub = rest.split(".", 1)

        # NeMo TransformerDecoderBlock has three sub-layers and three LNs.
        # The actual order in pre-LN is:
        #   x = x + self_attn (after layer_norm_1)
        #   x = x + cross_attn (after layer_norm_2)
        #   x = x + ffn (after layer_norm_3)
        # first_sub_layer = self-attention
        # second_sub_layer = cross-attention
        # third_sub_layer = FFN
        sub = (
            sub.replace("layer_norm_1", "norm_sa")
            .replace("layer_norm_2", "norm_ca")
            .replace("layer_norm_3", "norm_ff")
            .replace("first_sub_layer.query_net", "sa_q")
            .replace("first_sub_layer.key_net", "sa_k")
            .replace("first_sub_layer.value_net", "sa_v")
            .replace("first_sub_layer.out_projection", "sa_out")
            .replace("second_sub_layer.query_net", "ca_q")
            .replace("second_sub_layer.key_net", "ca_k")
            .replace("second_sub_layer.value_net", "ca_v")
            .replace("second_sub_layer.out_projection", "ca_out")
            .replace("third_sub_layer.dense_in", "ff_in")
            .replace("third_sub_layer.dense_out", "ff_out")
        )
        return f"decoder.layers.{layer_id}.{sub}"

    if n == "transf_decoder._decoder.final_layer_norm.weight":
        return "decoder.final_norm.weight"
    if n == "transf_decoder._decoder.final_layer_norm.bias":
        return "decoder.final_norm.bias"

    # ---- output head ----
    if n == "log_softmax.mlp.layer0.weight":
        return "decoder.head.weight"
    if n == "log_softmax.mlp.layer0.bias":
        return "decoder.head.bias"

    print(f"  [warn] unmapped tensor: {n}", file=sys.stderr)
    return None


# Tensors that should stay F32 even when --quant-linear is set:
def is_f32_tensor(gguf_name: str, shape: tuple[int, ...]) -> bool:
    if gguf_name.startswith("preprocessor."):
        return True
    if gguf_name == "decoder.pos_enc":
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
        paths = unpack_nemo(nemo_path, td_path)
        print(f"  config:  {paths['config']}")
        print(f"  weights: {paths['weights']}")
        print(f"  spm:     {paths['spm']}")

        sd = torch.load(str(paths["weights"]), map_location="cpu", weights_only=True)
        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]

        sp = spm.SentencePieceProcessor(model_file=str(paths["spm"]))
        vocab = [sp.id_to_piece(i) for i in range(sp.get_piece_size())]
        print(f"  vocab:   {len(vocab)} pieces")

    # ----- write GGUF -----
    print(f"\nWriting: {out_path}")
    writer = gguf.GGUFWriter(str(out_path), arch="canary")

    writer.add_uint32("canary.sample_rate", 16000)
    writer.add_uint32("canary.n_mels", 128)
    writer.add_uint32("canary.n_fft", 512)
    writer.add_uint32("canary.win_length", 400)
    writer.add_uint32("canary.hop_length", 160)
    writer.add_uint32("canary.d_model", 1024)
    writer.add_uint32("canary.enc_n_layers", 32)
    writer.add_uint32("canary.dec_n_layers", 8)
    writer.add_uint32("canary.n_heads", 8)
    writer.add_uint32("canary.head_dim", 128)
    writer.add_uint32("canary.ff_dim", 4096)
    writer.add_uint32("canary.subsampling_factor", 8)
    writer.add_uint32("canary.subsampling_channels", 256)
    writer.add_uint32("canary.conv_kernel", 9)
    writer.add_uint32("canary.vocab_size", len(vocab))
    writer.add_uint32("canary.max_dec_ctx", 1024)
    writer.add_uint32("canary.frame_dur_cs", 8)

    writer.add_array("tokenizer.ggml.tokens", vocab)

    n_written = 0
    n_f16 = 0
    n_f32 = 0
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
        if n_written <= 25 or n_written % 100 == 0:
            print(f"  {gguf_name:60s}  {str(t.shape):30s}  {t.dtype}")

        # Track encoder layers for synthetic conv.dw bias zeros
        # NOTE: Canary's depthwise_conv ALREADY has a bias tensor (use_bias=True),
        # so this is conditional — only inject if missing.

    # Inject zero conv.dw.bias only if absent (Canary's already has it)
    needed_bias_layers = set()
    for name in sd.keys():
        if "encoder.layers." in name and ".conv.depthwise_conv.weight" in name:
            li = int(name.split(".")[2])
            needed_bias_layers.add(li)
    have_bias = set()
    for name in sd.keys():
        if "encoder.layers." in name and ".conv.depthwise_conv.bias" in name:
            li = int(name.split(".")[2])
            have_bias.add(li)
    missing = needed_bias_layers - have_bias
    for li in sorted(missing):
        bias = np.zeros(1024, dtype=np.float32)
        writer.add_tensor(f"encoder.layers.{li}.conv.dw.bias", bias)
        n_written += 1
        n_f32 += 1
    if missing:
        print(f"  injected {len(missing)} synthetic conv.dw.bias zeros")
    else:
        print(
            f"  conv.dw.bias already present in all {len(needed_bias_layers)} layers (Canary has use_bias=True)"
        )

    print(f"\n  total tensors: {n_written}  (F16: {n_f16}, F32: {n_f32})")

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"\nDone: {out_path}  ({out_path.stat().st_size / 1e6:.1f} MB)")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Convert Canary 1B v2 .nemo → GGUF F16")
    p.add_argument("--nemo", required=True, type=Path, help="path to .nemo file")
    p.add_argument("--output", required=True, type=Path, help="output GGUF path")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    convert(args.nemo, args.output)
