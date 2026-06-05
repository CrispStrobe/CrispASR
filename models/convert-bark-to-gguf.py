#!/usr/bin/env python3
"""
Convert Suno Bark (suno/bark-small or suno/bark) HuggingFace .pt checkpoints
-> GGUF F16/F32 for the CrispASR `bark` TTS backend.

Bark is a 3-stage hierarchical TTS system:
  Stage 1 — Semantic model (GPT-2):  text -> semantic tokens (~10K vocab)
  Stage 2 — Coarse model (GPT-2):    semantic -> coarse EnCodec (2 codebooks, 1024 each)
  Stage 3 — Fine model (non-causal GPT): coarse -> fine EnCodec (8 codebooks)
  + EnCodec 24 kHz decoder: fine codes -> PCM

Architecture (bark-small / USE_SMALL_MODELS=True):
  text:   12L, 768d, 12h, block_size=1024,  input_vocab=10048, output_vocab=10048
  coarse: 12L, 768d, 12h, block_size=1024,  input_vocab=12096, output_vocab=12096
  fine:   12L, 768d, 12h, block_size=1024,  input_vocab=1056,  output_vocab=1056
          n_codes_total=8, n_codes_given=1

EnCodec 24 kHz decoder (SEANet):
  ratios=[8,5,4,2], n_filters=32, dimension=128, channels=1
  2x LSTM layers, ELU activation, weight_norm
  Decoder: Conv1d(128, 512, k=7) -> LSTM(512, 2L) ->
    4x [ELU + ConvTranspose1d + 1x ResBlock(dilations=[1,1])] ->
    ELU + Conv1d(32, 1, k=7)
  Quantizer: 8 codebooks (for 6kbps), each 1024 entries, dim=128

Usage:
    python models/convert-bark-to-gguf.py \
        --output /mnt/storage/bark-tts/bark-small-f16.gguf

    # Large model:
    python models/convert-bark-to-gguf.py \
        --use-large \
        --output /mnt/storage/bark-tts/bark-large-f16.gguf
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
# Constants from bark/generation.py
# ---------------------------------------------------------------------------

SEMANTIC_VOCAB_SIZE = 10_000
CODEBOOK_SIZE = 1024
N_COARSE_CODEBOOKS = 2
N_FINE_CODEBOOKS = 8
SAMPLE_RATE = 24_000

TEXT_ENCODING_OFFSET = 10_048
SEMANTIC_PAD_TOKEN = 10_000
TEXT_PAD_TOKEN = 129_595
SEMANTIC_INFER_TOKEN = 129_599

COARSE_SEMANTIC_PAD_TOKEN = 12_048
COARSE_INFER_TOKEN = 12_050


# ---------------------------------------------------------------------------
# Load Bark model checkpoints
# ---------------------------------------------------------------------------

def download_bark_models(use_small: bool) -> dict:
    """Download and load all 3 Bark sub-model checkpoints + EnCodec."""
    from huggingface_hub import hf_hub_download
    import os

    cache_dir = os.path.join(
        os.getenv("XDG_CACHE_HOME", os.path.join(os.path.expanduser("~"), ".cache")),
        "suno", "bark_v0",
    )
    os.makedirs(cache_dir, exist_ok=True)

    if use_small:
        files = {"text": "text.pt", "coarse": "coarse.pt", "fine": "fine.pt"}
    else:
        files = {"text": "text_2.pt", "coarse": "coarse_2.pt", "fine": "fine_2.pt"}

    models = {}
    for model_type, fname in files.items():
        local = os.path.join(cache_dir, fname)
        if not os.path.exists(local):
            print(f"Downloading {fname}...", file=sys.stderr)
            hf_hub_download(repo_id="suno/bark", filename=fname, local_dir=cache_dir)
        print(f"Loading {fname}...", file=sys.stderr)
        models[model_type] = torch.load(local, map_location="cpu", weights_only=False)

    return models


def load_encodec_weights() -> dict:
    """Load EnCodec 24 kHz model weights.

    Tries the encodec package first, falls back to downloading the
    checkpoint directly (avoids torchaudio dependency issues).
    """
    try:
        from encodec import EncodecModel
        model = EncodecModel.encodec_model_24khz()
        model.set_target_bandwidth(6.0)  # 8 codebooks
        model.eval()
        return model.state_dict()
    except (ImportError, OSError) as e:
        print(f"  encodec package unavailable ({e}), downloading checkpoint directly...",
              file=sys.stderr)

    # Fallback: download the checkpoint directly
    import urllib.request
    import os

    url = "https://dl.fbaipublicfiles.com/encodec/v0/encodec_24khz-d7cc33bc.th"
    cache_dir = os.path.join(
        os.getenv("XDG_CACHE_HOME", os.path.join(os.path.expanduser("~"), ".cache")),
        "encodec",
    )
    os.makedirs(cache_dir, exist_ok=True)
    local = os.path.join(cache_dir, "encodec_24khz-d7cc33bc.th")

    if not os.path.exists(local):
        print(f"  Downloading EnCodec checkpoint to {local}...", file=sys.stderr)
        urllib.request.urlretrieve(url, local)

    sd = torch.load(local, map_location="cpu", weights_only=False)
    return sd


# ---------------------------------------------------------------------------
# Tensor name remapping
# ---------------------------------------------------------------------------

def map_text_tensor(name: str) -> str | None:
    """Map Bark text (semantic) model tensor names to GGUF."""
    # Strip _orig_mod. prefix
    if name.startswith("_orig_mod."):
        name = name[len("_orig_mod."):]

    if name == "transformer.wte.weight":
        return "text.token_embd.weight"
    if name == "transformer.wpe.weight":
        return "text.pos_embd.weight"
    if name == "transformer.ln_f.weight":
        return "text.output_norm.weight"
    if name == "transformer.ln_f.bias":
        return "text.output_norm.bias"
    if name == "lm_head.weight":
        return "text.output.weight"

    # Layers: transformer.h.{i}.{attn/mlp/ln}
    if name.startswith("transformer.h."):
        rest = name[len("transformer.h."):]
        parts = rest.split(".", 1)
        layer = parts[0]
        field = parts[1]

        field = field.replace("ln_1.weight", "attn_norm.weight")
        field = field.replace("ln_1.bias", "attn_norm.bias")
        field = field.replace("ln_2.weight", "ffn_norm.weight")
        field = field.replace("ln_2.bias", "ffn_norm.bias")
        field = field.replace("attn.c_attn.weight", "attn_qkv.weight")
        field = field.replace("attn.c_attn.bias", "attn_qkv.bias")
        field = field.replace("attn.c_proj.weight", "attn_output.weight")
        field = field.replace("attn.c_proj.bias", "attn_output.bias")
        field = field.replace("mlp.c_fc.weight", "ffn_up.weight")
        field = field.replace("mlp.c_fc.bias", "ffn_up.bias")
        field = field.replace("mlp.c_proj.weight", "ffn_down.weight")
        field = field.replace("mlp.c_proj.bias", "ffn_down.bias")

        return f"text.blk.{layer}.{field}"

    return None


def map_coarse_tensor(name: str) -> str | None:
    """Map Bark coarse model tensor names to GGUF."""
    if name.startswith("_orig_mod."):
        name = name[len("_orig_mod."):]

    if name == "transformer.wte.weight":
        return "coarse.token_embd.weight"
    if name == "transformer.wpe.weight":
        return "coarse.pos_embd.weight"
    if name == "transformer.ln_f.weight":
        return "coarse.output_norm.weight"
    if name == "transformer.ln_f.bias":
        return "coarse.output_norm.bias"
    if name == "lm_head.weight":
        return "coarse.output.weight"

    if name.startswith("transformer.h."):
        rest = name[len("transformer.h."):]
        parts = rest.split(".", 1)
        layer = parts[0]
        field = parts[1]

        field = field.replace("ln_1.weight", "attn_norm.weight")
        field = field.replace("ln_1.bias", "attn_norm.bias")
        field = field.replace("ln_2.weight", "ffn_norm.weight")
        field = field.replace("ln_2.bias", "ffn_norm.bias")
        field = field.replace("attn.c_attn.weight", "attn_qkv.weight")
        field = field.replace("attn.c_attn.bias", "attn_qkv.bias")
        field = field.replace("attn.c_proj.weight", "attn_output.weight")
        field = field.replace("attn.c_proj.bias", "attn_output.bias")
        field = field.replace("mlp.c_fc.weight", "ffn_up.weight")
        field = field.replace("mlp.c_fc.bias", "ffn_up.bias")
        field = field.replace("mlp.c_proj.weight", "ffn_down.weight")
        field = field.replace("mlp.c_proj.bias", "ffn_down.bias")

        return f"coarse.blk.{layer}.{field}"

    return None


def map_fine_tensor(name: str) -> str | None:
    """Map Bark fine model tensor names to GGUF.

    Fine model has multiple token embeddings (wtes) and lm_heads instead
    of a single wte + lm_head. Non-causal attention (same QKV/proj layout).
    """
    if name.startswith("_orig_mod."):
        name = name[len("_orig_mod."):]

    # Multiple token embeddings: transformer.wtes.{i}.weight
    if name.startswith("transformer.wtes."):
        rest = name[len("transformer.wtes."):]
        # "0.weight" -> idx=0
        parts = rest.split(".", 1)
        idx = parts[0]
        return f"fine.token_embd.{idx}.weight"

    if name == "transformer.wpe.weight":
        return "fine.pos_embd.weight"
    if name == "transformer.ln_f.weight":
        return "fine.output_norm.weight"
    if name == "transformer.ln_f.bias":
        return "fine.output_norm.bias"

    # Multiple lm heads: lm_heads.{i}.weight
    if name.startswith("lm_heads."):
        rest = name[len("lm_heads."):]
        parts = rest.split(".", 1)
        idx = parts[0]
        return f"fine.output.{idx}.weight"

    # Layers: transformer.h.{i}.{attn/mlp/ln}
    if name.startswith("transformer.h."):
        rest = name[len("transformer.h."):]
        parts = rest.split(".", 1)
        layer = parts[0]
        field = parts[1]

        field = field.replace("ln_1.weight", "attn_norm.weight")
        field = field.replace("ln_1.bias", "attn_norm.bias")
        field = field.replace("ln_2.weight", "ffn_norm.weight")
        field = field.replace("ln_2.bias", "ffn_norm.bias")
        field = field.replace("attn.c_attn.weight", "attn_qkv.weight")
        field = field.replace("attn.c_attn.bias", "attn_qkv.bias")
        field = field.replace("attn.c_proj.weight", "attn_output.weight")
        field = field.replace("attn.c_proj.bias", "attn_output.bias")
        field = field.replace("mlp.c_fc.weight", "ffn_up.weight")
        field = field.replace("mlp.c_fc.bias", "ffn_up.bias")
        field = field.replace("mlp.c_proj.weight", "ffn_down.weight")
        field = field.replace("mlp.c_proj.bias", "ffn_down.bias")

        return f"fine.blk.{layer}.{field}"

    return None


def map_encodec_decoder_tensor(name: str) -> str | None:
    """Map EnCodec decoder + quantizer tensor names to GGUF.

    EnCodec decoder (SEANet) architecture at 24 kHz / 6 kbps:
      Quantizer: 8 codebooks, 1024 entries, dim=128
        quantizer.vq.layers.{i}._codebook.embed  (1024, 128)
      Decoder: model is a Sequential:
        [0] Conv1d(128, 512, k=7)
        [1] LSTM(512, 512, 2 layers)
        [2] ELU + ConvTranspose1d(512, 256, k=16, s=8)
        [3] ResBlock(256)
        [4] ELU + ConvTranspose1d(256, 128, k=10, s=5)
        [5] ResBlock(128)
        [6] ELU + ConvTranspose1d(128, 64, k=8, s=4)
        [7] ResBlock(64)
        [8] ELU + ConvTranspose1d(64, 32, k=4, s=2)
        [9] ResBlock(32)
        [10] ELU + Conv1d(32, 1, k=7)
    """
    # Quantizer codebooks -- only first 8 for 6 kbps (Bark uses 8 codebooks)
    if name.startswith("quantizer.vq.layers."):
        rest = name[len("quantizer.vq.layers."):]
        # e.g. "0._codebook.embed" -> cb index 0
        parts = rest.split(".", 1)
        idx = int(parts[0])
        field = parts[1]
        if idx >= 8:
            return None  # Only need first 8 codebooks
        if field == "_codebook.embed":
            return f"encodec.quantizer.{idx}.embed"
        # Skip other quantizer weights (not needed for decode-only)
        return None

    # Decoder
    if name.startswith("decoder.model."):
        rest = name[len("decoder.model."):]
        return f"encodec.decoder.{_map_seanet_decoder(rest)}"

    # Skip encoder weights entirely (we only decode)
    if name.startswith("encoder."):
        return None

    return None


def _map_seanet_decoder(rest: str) -> str:
    """Map SEANet decoder sub-module names.

    The decoder.model is a Sequential with indices:
    [0] SConv1d (input conv: 128 -> n_filters*2^4 = 512)
    [1] SLSTM (2 layers)
    [2-9] 4 upsample blocks, each: [act, SConvTranspose1d, ResBlock]
           The actual Sequential layout depends on the implementation.
           From seanet.py decoder init: for each ratio we have:
             act + ConvTranspose1d, then n_residual_layers ResBlocks.
           With n_residual_layers=1 and 4 ratios, it's:
           [2] ELU, [3] ConvTranspose1d (512->256, s=8)
           [4] ResBlock(256)
           [5] ELU, [6] ConvTranspose1d (256->128, s=5)
           [7] ResBlock(128)
           [8] ELU, [9] ConvTranspose1d (128->64, s=4)
           [10] ResBlock(64)
           [11] ELU, [12] ConvTranspose1d (64->32, s=2)
           [13] ResBlock(32)
           [14] ELU
           [15] SConv1d (32->1, k=7) -- output conv

    Weight norm means conv weights are stored as weight_g + weight_v
    (or as pre-applied weight after remove_weight_norm).
    """
    # Just pass through -- the C++ loader will map by full path.
    # We use the dotted index form as-is.
    return rest


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Convert Bark TTS to GGUF")
    ap.add_argument("--output", required=True, help="Output GGUF path")
    ap.add_argument("--outtype", default="f16", choices=["f32", "f16"])
    ap.add_argument("--use-large", action="store_true",
                    help="Use large Bark model instead of small")
    ap.add_argument("--skip-encodec", action="store_true",
                    help="Skip EnCodec decoder weights (for separate codec GGUF)")
    args = ap.parse_args()

    use_small = not args.use_large

    # Load checkpoints
    models = download_bark_models(use_small)
    print("\nBark model checkpoints loaded.", file=sys.stderr)

    # Extract model configs from checkpoints
    text_args = models["text"]["model_args"]
    coarse_args = models["coarse"]["model_args"]
    fine_args = models["fine"]["model_args"]

    # Normalize vocab key names
    for d in [text_args, coarse_args, fine_args]:
        if "vocab_size" in d and "input_vocab_size" not in d:
            d["input_vocab_size"] = d["vocab_size"]
            d["output_vocab_size"] = d["vocab_size"]

    print(f"\nText model:   {text_args}", file=sys.stderr)
    print(f"Coarse model: {coarse_args}", file=sys.stderr)
    print(f"Fine model:   {fine_args}", file=sys.stderr)

    out_dtype = np.float16 if args.outtype == "f16" else np.float32
    out_qt = GGMLQuantizationType.F16 if args.outtype == "f16" else GGMLQuantizationType.F32

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    w = GGUFWriter(str(out_path), arch="bark", use_temp_file=False)

    # ----- metadata -----------------------------------------------------
    variant = "small" if use_small else "large"
    w.add_name(f"bark-{variant}")

    def u32(k, v):
        w.add_uint32(k, int(v))

    def f32(k, v):
        w.add_float32(k, float(v))

    # Text (semantic) model hparams
    u32("bark.text.n_layer", text_args["n_layer"])
    u32("bark.text.n_head", text_args["n_head"])
    u32("bark.text.n_embd", text_args["n_embd"])
    u32("bark.text.block_size", text_args["block_size"])
    u32("bark.text.input_vocab_size", text_args["input_vocab_size"])
    u32("bark.text.output_vocab_size", text_args["output_vocab_size"])

    # Coarse model hparams
    u32("bark.coarse.n_layer", coarse_args["n_layer"])
    u32("bark.coarse.n_head", coarse_args["n_head"])
    u32("bark.coarse.n_embd", coarse_args["n_embd"])
    u32("bark.coarse.block_size", coarse_args["block_size"])
    u32("bark.coarse.input_vocab_size", coarse_args["input_vocab_size"])
    u32("bark.coarse.output_vocab_size", coarse_args["output_vocab_size"])

    # Fine model hparams
    u32("bark.fine.n_layer", fine_args["n_layer"])
    u32("bark.fine.n_head", fine_args["n_head"])
    u32("bark.fine.n_embd", fine_args["n_embd"])
    u32("bark.fine.block_size", fine_args["block_size"])
    u32("bark.fine.input_vocab_size", fine_args["input_vocab_size"])
    u32("bark.fine.output_vocab_size", fine_args["output_vocab_size"])
    u32("bark.fine.n_codes_total", fine_args.get("n_codes_total", 8))
    u32("bark.fine.n_codes_given", fine_args.get("n_codes_given", 1))

    # Pipeline constants
    u32("bark.sample_rate", SAMPLE_RATE)
    u32("bark.semantic_vocab_size", SEMANTIC_VOCAB_SIZE)
    u32("bark.codebook_size", CODEBOOK_SIZE)
    u32("bark.n_coarse_codebooks", N_COARSE_CODEBOOKS)
    u32("bark.n_fine_codebooks", N_FINE_CODEBOOKS)
    u32("bark.text_encoding_offset", TEXT_ENCODING_OFFSET)
    u32("bark.semantic_pad_token", SEMANTIC_PAD_TOKEN)
    u32("bark.text_pad_token", TEXT_PAD_TOKEN)
    u32("bark.semantic_infer_token", SEMANTIC_INFER_TOKEN)
    u32("bark.coarse_semantic_pad_token", COARSE_SEMANTIC_PAD_TOKEN)
    u32("bark.coarse_infer_token", COARSE_INFER_TOKEN)
    w.add_string("bark.variant", variant)

    # EnCodec hparams
    u32("bark.encodec.sample_rate", SAMPLE_RATE)
    u32("bark.encodec.n_codebooks", 8)
    u32("bark.encodec.codebook_size", 1024)
    u32("bark.encodec.codebook_dim", 128)
    u32("bark.encodec.n_filters", 32)
    w.add_array("bark.encodec.ratios", [8, 5, 4, 2])
    u32("bark.encodec.lstm_layers", 2)
    u32("bark.encodec.kernel_size", 7)

    # Embed BERT tokenizer vocab for the text stage
    try:
        from transformers import BertTokenizer
        tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")
        vocab = tokenizer.get_vocab()
        max_id = max(vocab.values())
        toks = [""] * (max_id + 1)
        for tok, idx in vocab.items():
            toks[idx] = tok
        w.add_token_list(toks)
        print(f"  BERT tokenizer: {len(toks)} tokens", file=sys.stderr)
    except Exception as e:
        print(f"  WARNING: could not load BERT tokenizer: {e}", file=sys.stderr)

    # ----- tensors: text model ------------------------------------------
    n_total = 0

    text_sd = models["text"]["model"]
    for name in sorted(text_sd.keys()):
        # Strip _orig_mod. prefix
        clean = name
        if clean.startswith("_orig_mod."):
            clean = clean[len("_orig_mod."):]
        if clean.endswith(".attn.bias"):
            continue  # causal mask, reconstructed at runtime
        gn = map_text_tensor(name)
        if gn is None:
            print(f"  [skip text] {name}", file=sys.stderr)
            continue
        t = text_sd[name].to(torch.float32).numpy()
        if t.ndim <= 1:
            t = np.ascontiguousarray(t.astype(np.float32))
            w.add_tensor(gn, t, raw_dtype=GGMLQuantizationType.F32)
        else:
            t = np.ascontiguousarray(t.astype(out_dtype))
            w.add_tensor(gn, t, raw_dtype=out_qt)
        n_total += 1
        if n_total <= 10 or n_total % 50 == 0:
            print(f"  [{n_total}] {gn:55s} {t.shape}  {t.dtype}", file=sys.stderr)

    print(f"Text model: {n_total} tensors", file=sys.stderr)

    # ----- tensors: coarse model ----------------------------------------
    n_coarse_start = n_total
    coarse_sd = models["coarse"]["model"]
    for name in sorted(coarse_sd.keys()):
        clean = name
        if clean.startswith("_orig_mod."):
            clean = clean[len("_orig_mod."):]
        if clean.endswith(".attn.bias"):
            continue
        gn = map_coarse_tensor(name)
        if gn is None:
            print(f"  [skip coarse] {name}", file=sys.stderr)
            continue
        t = coarse_sd[name].to(torch.float32).numpy()
        if t.ndim <= 1:
            t = np.ascontiguousarray(t.astype(np.float32))
            w.add_tensor(gn, t, raw_dtype=GGMLQuantizationType.F32)
        else:
            t = np.ascontiguousarray(t.astype(out_dtype))
            w.add_tensor(gn, t, raw_dtype=out_qt)
        n_total += 1
        if (n_total - n_coarse_start) <= 5 or n_total % 50 == 0:
            print(f"  [{n_total}] {gn:55s} {t.shape}  {t.dtype}", file=sys.stderr)

    print(f"Coarse model: {n_total - n_coarse_start} tensors", file=sys.stderr)

    # ----- tensors: fine model ------------------------------------------
    n_fine_start = n_total
    fine_sd = models["fine"]["model"]
    for name in sorted(fine_sd.keys()):
        clean = name
        if clean.startswith("_orig_mod."):
            clean = clean[len("_orig_mod."):]
        if clean.endswith(".attn.bias"):
            continue
        gn = map_fine_tensor(name)
        if gn is None:
            print(f"  [skip fine] {name}", file=sys.stderr)
            continue
        t = fine_sd[name].to(torch.float32).numpy()
        if t.ndim <= 1:
            t = np.ascontiguousarray(t.astype(np.float32))
            w.add_tensor(gn, t, raw_dtype=GGMLQuantizationType.F32)
        else:
            t = np.ascontiguousarray(t.astype(out_dtype))
            w.add_tensor(gn, t, raw_dtype=out_qt)
        n_total += 1
        if (n_total - n_fine_start) <= 5 or n_total % 50 == 0:
            print(f"  [{n_total}] {gn:55s} {t.shape}  {t.dtype}", file=sys.stderr)

    print(f"Fine model: {n_total - n_fine_start} tensors", file=sys.stderr)

    # ----- tensors: EnCodec decoder + quantizer -------------------------
    if not args.skip_encodec:
        n_enc_start = n_total
        print("\nLoading EnCodec 24 kHz decoder...", file=sys.stderr)
        enc_sd = load_encodec_weights()

        for name in sorted(enc_sd.keys()):
            gn = map_encodec_decoder_tensor(name)
            if gn is None:
                continue
            t = enc_sd[name].to(torch.float32).numpy()
            if t.ndim <= 1:
                t = np.ascontiguousarray(t.astype(np.float32))
                w.add_tensor(gn, t, raw_dtype=GGMLQuantizationType.F32)
            else:
                t = np.ascontiguousarray(t.astype(out_dtype))
                w.add_tensor(gn, t, raw_dtype=out_qt)
            n_total += 1
            if (n_total - n_enc_start) <= 5 or n_total % 50 == 0:
                print(f"  [{n_total}] {gn:55s} {t.shape}  {t.dtype}", file=sys.stderr)

        print(f"EnCodec decoder: {n_total - n_enc_start} tensors", file=sys.stderr)

    # ----- write --------------------------------------------------------
    print(f"\nTotal: {n_total} tensors", file=sys.stderr)
    print(f"Writing {out_path}...", file=sys.stderr)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()

    sz = out_path.stat().st_size / 1e6
    print(f"Done: {out_path}  ({sz:.1f} MB, {n_total} tensors)", file=sys.stderr)


if __name__ == "__main__":
    main()
