#!/usr/bin/env python3
"""
Convert IndexTTS-1.5 checkpoints to GGUF for the CrispASR ``indextts`` TTS backend.

Produces two GGUF files from three source checkpoints in the IndexTTS-1.5 model dir:

  indextts-gpt.gguf     — GPT-2 (24L/1280d/20h) + ConformerEncoder (6-block, conv2d2)
                          + PerceiverResampler (2-layer) + embeddings + positional embs
                          Source: gpt.pth (no weight_norm — tensors are written as-is)

  indextts-bigvgan.gguf — BigVGAN vocoder: ECAPA-TDNN speaker encoder + 6× ConvTranspose1d
                          upsamplers + AMPBlock1 resblocks (SnakeBeta) + conditioning layers
                          Source: bigvgan_generator.pth['generator']
                          Weight-norm layers (conv_pre, ups, resblocks, conv_post) are fused:
                            weight = weight_v * (weight_g / ||weight_v||)

  dvae.pth is NOT used at inference — skip it.

Architecture (from config.yaml, confirmed against the state dicts):

  GPT:     model_dim=1280, heads=20, layers=24, max_mel_tokens=1815, max_text_tokens=600
           number_text_tokens=12000, number_mel_codes=8194 (8192+start+stop)
           mel_length_compression=1024, condition_num_latent=32, condition_type=conformer_perceiver
           Conformer:  6 blocks, input=100-mel, output_size=512, conv2d2 subsampling
           Perceiver:  2 layers, latents=32, model_dim=1280
  BigVGAN: gpt_dim=1280, upsample_initial_channel=1536
           upsample_rates=[4,4,4,4,2,2], upsample_kernel_sizes=[8,8,4,4,4,4]
           resblock_kernel_sizes=[3,7,11], resblock_dilation_sizes=[[1,3,5]]*3
           activation=snakebeta, cond_d_vector_in_each_upsampling_layer=True
           speaker_embedding_dim=512, num_mels=100, sampling_rate=24000, hop_size=256

Usage:

    python models/convert-indextts-to-gguf.py \\
        --model-dir /path/to/IndexTTS-1.5 \\
        --output-dir /Volumes/backups/ai/crispasr-models/indextts \\
        [--outtype f16|f32]
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
    import torch
except ImportError:
    sys.exit("pip install torch")


# ---------------------------------------------------------------------------
# Weight-norm fusion
# ---------------------------------------------------------------------------

def fuse_weight_norm(sd: dict) -> dict:
    """Fuse weight_norm reparametrisations: weight = weight_v * (weight_g / ||weight_v||).

    ||·|| is the L2 norm over all dims except dim-0 (PyTorch's norm_except_dim).
    Returns a new dict; does not mutate the input.
    """
    out: dict = {}
    pairs: dict[str, dict] = {}
    for k, v in sd.items():
        if k.endswith(".weight_g"):
            pairs.setdefault(k[:-len(".weight_g")], {})["g"] = v
        elif k.endswith(".weight_v"):
            pairs.setdefault(k[:-len(".weight_v")], {})["v"] = v
        else:
            out[k] = v
    fused = 0
    for stem, parts in pairs.items():
        if "g" not in parts or "v" not in parts:
            for suf, t in parts.items():
                out[f"{stem}.weight_{suf}"] = t
            continue
        g = parts["g"].to(torch.float32)
        v = parts["v"].to(torch.float32)
        flat = v.reshape(v.shape[0], -1)
        norm = flat.norm(p=2, dim=1).reshape(g.shape)
        out[f"{stem}.weight"] = v * (g / norm.clamp_min(1e-12))
        fused += 1
    print(f"  weight_norm fused: {fused} pairs", file=sys.stderr)
    return out


# ---------------------------------------------------------------------------
# GGUF writer helpers
# ---------------------------------------------------------------------------

def write_tensor(writer: GGUFWriter, name: str, t: torch.Tensor,
                 out_dtype: np.dtype, qt: GGMLQuantizationType) -> None:
    arr = t.to(torch.float32).clamp_(-65504.0, 65504.0).detach().numpy().astype(out_dtype)
    writer.add_tensor(name, arr, raw_dtype=qt)


# GPT-2 Conv1D stores weights as [in, out] instead of nn.Linear's [out, in].
# ggml_mul_mat(w, x) needs ne[0]=in_dim (after GGUF's NumPy→ne reversal,
# that means the NumPy array should be [out, in]). So we transpose Conv1D
# weights to match the convention.
_GPT2_CONV1D_SUFFIXES = (
    ".attn.c_attn.weight",
    ".attn.c_proj.weight",
    ".mlp.c_fc.weight",
    ".mlp.c_proj.weight",
)


def write_tensor_gpt(writer: GGUFWriter, name: str, t: torch.Tensor,
                     out_dtype: np.dtype, qt: GGMLQuantizationType) -> None:
    """Write GPT tensor, transposing Conv1D weights to nn.Linear convention.
    Biases, norms, and 1D tensors are kept as F32 for ggml type compatibility."""
    if any(name.endswith(s) for s in _GPT2_CONV1D_SUFFIXES):
        t = t.t().contiguous()
    # Keep biases, norms, and small tensors as F32
    if t.ndim <= 1 or name.endswith(".bias") or "norm" in name or "gamma" in name:
        write_tensor(writer, name, t, np.float32, GGMLQuantizationType.F32)
    else:
        write_tensor(writer, name, t, out_dtype, qt)


# ---------------------------------------------------------------------------
# GPT GGUF
# ---------------------------------------------------------------------------

# Keys that exist only during training / are unused at inference.
_GPT_SKIP = {
    "text_head.weight",   # text LM head — not used in mel-autoregressive inference
    "text_head.bias",
}


def _shorten_gpt(name: str) -> str:
    """Shorten tensor names to fit the 64-char GGUF limit."""
    name = name.replace("conditioning_encoder.", "cond_enc.")
    name = name.replace("encoders.", "enc.")
    name = name.replace("conv_module.", "conv.")
    name = name.replace("depthwise_conv.", "dw.")
    name = name.replace("pointwise_conv1.", "pw1.")
    name = name.replace("pointwise_conv2.", "pw2.")
    name = name.replace("self_attn.", "sa.")
    name = name.replace("feed_forward.", "ff.")
    name = name.replace("perceiver_encoder.", "perc.")
    name = name.replace("text_pos_embedding.emb.", "text_pos.")
    name = name.replace("mel_pos_embedding.emb.", "mel_pos.")
    return name


def convert_gpt(gpt_pth: Path, out_path: Path, outtype: str,
                bpe_model_path: Path | None = None) -> None:
    print(f"\n=== GPT: {gpt_pth.name} → {out_path.name} ===", file=sys.stderr)
    sd = torch.load(str(gpt_pth), map_location="cpu", weights_only=False)
    # gpt.pth may be a raw state dict or wrapped
    if isinstance(sd, dict) and "model" in sd and not any(
        k.startswith("gpt.") for k in sd
    ):
        sd = sd["model"]

    out_dtype = np.float16 if outtype == "f16" else np.float32
    qt = GGMLQuantizationType.F16 if outtype == "f16" else GGMLQuantizationType.F32

    w = GGUFWriter(str(out_path), arch="indextts.gpt", use_temp_file=True)
    w.add_name("indextts-gpt")

    # Embed SentencePiece vocabulary for runtime tokenization
    if bpe_model_path and bpe_model_path.is_file():
        import sentencepiece as spm
        sp = spm.SentencePieceProcessor()
        sp.Load(str(bpe_model_path))
        vocab = [sp.IdToPiece(i) for i in range(sp.GetPieceSize())]
        scores = [sp.GetScore(i) for i in range(sp.GetPieceSize())]
        w.add_array("tokenizer.ggml.tokens", vocab)
        w.add_array("tokenizer.ggml.scores", scores)
        w.add_string("tokenizer.ggml.model", "bpe")
        print(f"  embedded {len(vocab)} BPE tokens from {bpe_model_path.name}",
              file=sys.stderr)

    # Hyperparameters (from config.yaml, confirmed against tensor shapes)
    def u32(k, v): w.add_uint32(k, int(v))

    u32("indextts.gpt.model_dim",              1280)
    u32("indextts.gpt.layers",                 24)
    u32("indextts.gpt.heads",                  20)
    u32("indextts.gpt.number_text_tokens",     12000)
    u32("indextts.gpt.number_mel_codes",       8194)
    u32("indextts.gpt.start_mel_token",        8192)
    u32("indextts.gpt.stop_mel_token",         8193)
    u32("indextts.gpt.max_mel_tokens",         1815)
    u32("indextts.gpt.max_text_tokens",        600)
    u32("indextts.gpt.mel_length_compression", 1024)
    u32("indextts.gpt.condition_num_latent",   32)
    u32("indextts.conformer.num_blocks",       6)
    u32("indextts.conformer.output_size",      512)
    u32("indextts.conformer.linear_units",     2048)
    u32("indextts.conformer.attention_heads",  8)
    u32("indextts.perceiver.num_layers",       2)
    u32("indextts.sampling_rate",              24000)

    written = 0
    skipped = 0
    for name, tensor in sd.items():
        if name in _GPT_SKIP:
            skipped += 1
            continue
        short = _shorten_gpt(name)
        write_tensor_gpt(w, short, tensor, out_dtype, qt)
        written += 1

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"  wrote {written} tensors, skipped {skipped} → {out_path}", file=sys.stderr)


# ---------------------------------------------------------------------------
# BigVGAN GGUF
# ---------------------------------------------------------------------------

# Tensors that are training-only or unused at inference.
_BVG_SKIP_PREFIXES = (
    "logit_scale",          # contrastive loss scale — training only
)

# Suffixes that are training-only BatchNorm bookkeeping — not used at inference.
_SKIP_SUFFIXES = (
    ".num_batches_tracked",
)


def _shorten_bvg(name: str) -> str:
    """Shorten BigVGAN tensor names to fit the 64-char GGUF limit."""
    name = name.replace("speaker_encoder.", "se.")
    name = name.replace("res2net_block.", "r2n.")
    name = name.replace("resblocks.", "resb.")
    name = name.replace("blocks.", "b.")
    name = name.replace("conv.conv.", "c.")
    name = name.replace("norm.norm.", "n.")
    name = name.replace("activations.", "act.")
    name = name.replace("downsample.lowpass.", "ds.")
    name = name.replace("upsample.", "us.")
    name = name.replace("running_mean", "rm")
    name = name.replace("running_var", "rv")
    return name


def convert_bigvgan(bvg_pth: Path, out_path: Path, outtype: str) -> None:
    print(f"\n=== BigVGAN: {bvg_pth.name} → {out_path.name} ===", file=sys.stderr)
    raw = torch.load(str(bvg_pth), map_location="cpu", weights_only=False)
    if isinstance(raw, dict) and "generator" in raw:
        sd = raw["generator"]
    else:
        sd = raw

    # Fuse weight_norm (conv_pre, ups, resblocks, conv_post)
    sd = fuse_weight_norm(sd)

    out_dtype = np.float16 if outtype == "f16" else np.float32
    qt = GGMLQuantizationType.F16 if outtype == "f16" else GGMLQuantizationType.F32

    w = GGUFWriter(str(out_path), arch="indextts.bigvgan", use_temp_file=True)
    w.add_name("indextts-bigvgan")

    # Hyperparameters
    def u32(k, v): w.add_uint32(k, int(v))

    u32("indextts.bigvgan.gpt_dim",             1280)
    u32("indextts.bigvgan.upsample_initial_channel", 1536)
    u32("indextts.bigvgan.num_upsamples",        6)
    u32("indextts.bigvgan.num_kernels",          3)
    u32("indextts.bigvgan.speaker_embedding_dim", 512)
    u32("indextts.bigvgan.num_mels",             100)
    u32("indextts.sampling_rate",                24000)
    u32("indextts.bigvgan.hop_size",             256)
    w.add_array("indextts.bigvgan.upsample_rates",        [4, 4, 4, 4, 2, 2])
    w.add_array("indextts.bigvgan.upsample_kernel_sizes", [8, 8, 4, 4, 4, 4])
    w.add_array("indextts.bigvgan.resblock_kernel_sizes", [3, 7, 11])
    # 3 resblocks × dilations [1,3,5] — flattened
    w.add_array("indextts.bigvgan.resblock_dilation_sizes", [1, 3, 5, 1, 3, 5, 1, 3, 5])
    w.add_string("indextts.bigvgan.activation", "snakebeta")
    w.add_bool("indextts.bigvgan.cond_in_each_up_layer", True)

    written = 0
    skipped = 0
    for name, tensor in sd.items():
        if any(name.startswith(p) for p in _BVG_SKIP_PREFIXES):
            skipped += 1
            continue
        if any(name.endswith(s) for s in _SKIP_SUFFIXES):
            skipped += 1
            continue
        short = _shorten_bvg(name)
        # Keep biases, norms, alphas, betas, and 1D tensors as F32
        if tensor.ndim <= 1 or name.endswith(".bias") or "norm" in name \
                or "alpha" in name or "beta" in name or "filter" in name \
                or "running_mean" in name or "running_var" in name:
            write_tensor(w, short, tensor, np.float32, GGMLQuantizationType.F32)
        else:
            write_tensor(w, short, tensor, out_dtype, qt)
        written += 1

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"  wrote {written} tensors, skipped {skipped} → {out_path}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Convert IndexTTS-1.5 checkpoints to two GGUF files"
    )
    ap.add_argument(
        "--model-dir", required=True,
        help="Directory containing gpt.pth, bigvgan_generator.pth, config.yaml",
    )
    ap.add_argument(
        "--output-dir", required=True,
        help="Directory to write indextts-gpt.gguf and indextts-bigvgan.gguf",
    )
    ap.add_argument(
        "--outtype", default="f16", choices=["f16", "f32"],
        help="Output tensor dtype (default: f16)",
    )
    ap.add_argument(
        "--gpt-only", action="store_true",
        help="Convert only the GPT checkpoint",
    )
    ap.add_argument(
        "--bigvgan-only", action="store_true",
        help="Convert only the BigVGAN checkpoint",
    )
    args = ap.parse_args()

    model_dir = Path(args.model_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    gpt_pth = model_dir / "gpt.pth"
    bvg_pth = model_dir / "bigvgan_generator.pth"

    for p in ([gpt_pth] if not args.bigvgan_only else []) + \
             ([bvg_pth] if not args.gpt_only else []):
        if not p.is_file():
            sys.exit(f"not found: {p}")

    bpe_path = model_dir / "bpe.model"

    if not args.bigvgan_only:
        convert_gpt(gpt_pth, out_dir / "indextts-gpt.gguf", args.outtype,
                    bpe_model_path=bpe_path if bpe_path.is_file() else None)

    if not args.gpt_only:
        convert_bigvgan(bvg_pth, out_dir / "indextts-bigvgan.gguf", args.outtype)

    print("\nDone.", file=sys.stderr)


if __name__ == "__main__":
    main()
