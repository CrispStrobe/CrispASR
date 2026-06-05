#!/usr/bin/env python3
"""
Convert Microsoft SpeechT5 TTS + HiFi-GAN vocoder to GGUF for the CrispASR
``speecht5`` backend.

Architecture (microsoft/speecht5_tts):
  Text encoder:
    - Embedding(81, 768) + ScaledPositionalEncoding(768)
    - 12 encoder layers: pre-LN transformer with relative position bias
      (SpeechT5RelativePositionalEncoding: Embedding(2*160, 64))
      768-d, 12 heads, 3072 FFN, GELU, post-LN
  Speech decoder:
    - Prenet: 2x Linear(80->256) with ReLU + consistent dropout
      + Linear(256->768) + ScaledPositionalEncoding(768)
      + speaker projection: Linear(512+768, 768) + ReLU
    - 6 decoder layers: self-attn + cross-attn + FFN, post-LN
    - Postnet output: Linear(768->160) feat_out, Linear(768->2) prob_out
    - 5-layer conv postnet: Conv1d+BN+Tanh (80->256->256->256->256->80), k=5

  HiFi-GAN vocoder (microsoft/speecht5_hifigan):
    - conv_pre: Conv1d(80, 512, k=7)
    - 4 upsample stages: ConvTranspose1d with rates [4,4,4,4], kernels [8,8,8,8]
      channels: 512 -> 256 -> 128 -> 64 -> 32
    - 12 resblocks (3 per upsample): kernels [3,7,11], dilations [(1,3,5),(1,3,5),(1,3,5)]
      Each resblock: 3 pairs of (LeakyReLU + Conv1d_dilated + LeakyReLU + Conv1d_d1)
    - conv_post: Conv1d(32, 1, k=7)
    - mean/scale normalization buffers

Produces ONE GGUF containing both TTS model and vocoder weights.

Usage:
    python models/convert-speecht5-to-gguf.py \\
        --tts microsoft/speecht5_tts \\
        --vocoder microsoft/speecht5_hifigan \\
        --output /mnt/storage/speecht5/speecht5-tts-f16.gguf

    # German fine-tune:
    python models/convert-speecht5-to-gguf.py \\
        --tts sjdata/speecht5_finetuned_common_voice_11_de \\
        --vocoder microsoft/speecht5_hifigan \\
        --output /mnt/storage/speecht5/speecht5-tts-de-f16.gguf
"""

from __future__ import annotations

import argparse
import json
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

try:
    from safetensors.torch import load_file as safe_load
except ImportError:
    safe_load = None

try:
    from huggingface_hub import snapshot_download
except ImportError:
    snapshot_download = None


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def resolve_dir(model_id: str) -> Path:
    p = Path(model_id)
    if p.is_dir():
        return p
    if snapshot_download is None:
        sys.exit(f"Directory {model_id} not found and huggingface_hub not installed")
    print(f"Downloading {model_id}...", file=sys.stderr)
    return Path(snapshot_download(model_id))


def load_state_dict(model_dir: Path, prefix: str = "") -> dict[str, torch.Tensor]:
    """Load state dict from safetensors or pytorch_model.bin."""
    safetensor_files = sorted(model_dir.glob("*.safetensors"))
    if safetensor_files and safe_load is not None:
        sd = {}
        for f in safetensor_files:
            sd.update(safe_load(str(f)))
        return sd

    bin_files = sorted(model_dir.glob("pytorch_model*.bin"))
    if bin_files:
        sd = {}
        for f in bin_files:
            sd.update(torch.load(f, map_location="cpu", weights_only=True))
        return sd

    sys.exit(f"No model files found in {model_dir}")


def load_config(model_dir: Path) -> dict:
    cfg_path = model_dir / "config.json"
    if cfg_path.exists():
        with open(cfg_path, encoding="utf-8") as f:
            return json.load(f)
    return {}


# ---------------------------------------------------------------------------
# Weight-norm fusion
# ---------------------------------------------------------------------------


def fuse_weight_norm(sd: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Fuse weight_g + weight_v into a single weight tensor."""
    fused = {}
    keys_done = set()
    for key in sorted(sd.keys()):
        if key.endswith(".weight_g"):
            base = key[:-len(".weight_g")]
            v_key = base + ".weight_v"
            if v_key in sd:
                g = sd[key]
                v = sd[v_key]
                # weight = g * v / ||v||
                norm = torch.linalg.norm(v.view(v.shape[0], -1), dim=1)
                weight = g.view(-1, *([1] * (v.dim() - 1))) * v / norm.view(-1, *([1] * (v.dim() - 1)))
                fused[base + ".weight"] = weight
                keys_done.add(key)
                keys_done.add(v_key)
    result = {}
    for key, val in sd.items():
        if key not in keys_done:
            result[key] = val
    result.update(fused)
    return result


# ---------------------------------------------------------------------------
# Vocab loading
# ---------------------------------------------------------------------------


def load_vocab(model_dir: Path) -> list[str]:
    """Load SpeechT5 tokenizer vocabulary."""
    # Try spm_char.model or tokenizer_config.json
    vocab_path = model_dir / "spm_char.model"
    if vocab_path.exists():
        try:
            import sentencepiece as spm
            sp = spm.SentencePieceProcessor()
            sp.Load(str(vocab_path))
            return [sp.IdToPiece(i) for i in range(sp.GetPieceSize())]
        except ImportError:
            pass

    # Fallback: build basic vocab from config vocab_size
    # SpeechT5 uses a small char-level vocab (81 tokens)
    # The standard SpeechT5 tokenizer is character-based
    tokenizer_path = model_dir / "tokenizer_config.json"
    if tokenizer_path.exists():
        with open(tokenizer_path, encoding="utf-8") as f:
            tok_cfg = json.load(f)

    # Try loading from the tokenizer files
    vocab_file = model_dir / "spm_char.vocab"
    if vocab_file.exists():
        vocab = []
        with open(vocab_file, encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if parts:
                    vocab.append(parts[0])
        return vocab

    return []


# ---------------------------------------------------------------------------
# Main conversion
# ---------------------------------------------------------------------------


def convert(args):
    tts_dir = resolve_dir(args.tts)
    voc_dir = resolve_dir(args.vocoder)

    tts_cfg = load_config(tts_dir)
    voc_cfg = load_config(voc_dir)

    # Load state dicts
    tts_sd = load_state_dict(tts_dir)
    voc_sd = load_state_dict(voc_dir)

    # Fuse weight-norm in vocoder
    voc_sd = fuse_weight_norm(voc_sd)

    # Load vocab
    vocab = load_vocab(tts_dir)

    # Config values
    hidden_size = tts_cfg.get("hidden_size", 768)
    num_mel_bins = tts_cfg.get("num_mel_bins", 80)
    encoder_layers = tts_cfg.get("encoder_layers", 12)
    decoder_layers = tts_cfg.get("decoder_layers", 6)
    encoder_attention_heads = tts_cfg.get("encoder_attention_heads", 12)
    decoder_attention_heads = tts_cfg.get("decoder_attention_heads", 12)
    encoder_ffn_dim = tts_cfg.get("encoder_ffn_dim", 3072)
    decoder_ffn_dim = tts_cfg.get("decoder_ffn_dim", 3072)
    vocab_size = tts_cfg.get("vocab_size", 81)
    reduction_factor = tts_cfg.get("reduction_factor", 2)
    speech_decoder_prenet_layers = tts_cfg.get("speech_decoder_prenet_layers", 2)
    speech_decoder_prenet_units = tts_cfg.get("speech_decoder_prenet_units", 256)
    speech_decoder_postnet_layers = tts_cfg.get("speech_decoder_postnet_layers", 5)
    speech_decoder_postnet_units = tts_cfg.get("speech_decoder_postnet_units", 256)
    speech_decoder_postnet_kernel = tts_cfg.get("speech_decoder_postnet_kernel", 5)
    speaker_embedding_dim = tts_cfg.get("speaker_embedding_dim", 512)
    max_text_positions = tts_cfg.get("max_text_positions", 450)
    max_speech_positions = tts_cfg.get("max_speech_positions", 4000)
    encoder_max_relative_position = tts_cfg.get("encoder_max_relative_position", 160)
    layer_norm_eps = tts_cfg.get("layer_norm_eps", 1e-5)

    # Vocoder config
    voc_model_in_dim = voc_cfg.get("model_in_dim", 80)
    voc_upsample_initial_channel = voc_cfg.get("upsample_initial_channel", 512)
    voc_upsample_rates = voc_cfg.get("upsample_rates", [4, 4, 4, 4])
    voc_upsample_kernel_sizes = voc_cfg.get("upsample_kernel_sizes", [8, 8, 8, 8])
    voc_resblock_kernel_sizes = voc_cfg.get("resblock_kernel_sizes", [3, 7, 11])
    voc_resblock_dilation_sizes = voc_cfg.get("resblock_dilation_sizes",
                                               [[1, 3, 5], [1, 3, 5], [1, 3, 5]])
    voc_leaky_relu_slope = voc_cfg.get("leaky_relu_slope", 0.1)
    voc_normalize_before = voc_cfg.get("normalize_before", True)
    voc_sampling_rate = voc_cfg.get("sampling_rate", 16000)

    print(f"TTS config: hidden={hidden_size} mel={num_mel_bins} enc_layers={encoder_layers} "
          f"dec_layers={decoder_layers} vocab={vocab_size} reduction={reduction_factor}",
          file=sys.stderr)
    print(f"Vocoder config: in_dim={voc_model_in_dim} init_ch={voc_upsample_initial_channel} "
          f"rates={voc_upsample_rates} kernels={voc_upsample_kernel_sizes}",
          file=sys.stderr)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = GGUFWriter(str(out_path), "speecht5-tts")

    # ── KV metadata ──────────────────────────────────────────────────

    writer.add_uint32("speecht5.hidden_size", hidden_size)
    writer.add_uint32("speecht5.num_mel_bins", num_mel_bins)
    writer.add_uint32("speecht5.encoder_layers", encoder_layers)
    writer.add_uint32("speecht5.decoder_layers", decoder_layers)
    writer.add_uint32("speecht5.encoder_attention_heads", encoder_attention_heads)
    writer.add_uint32("speecht5.decoder_attention_heads", decoder_attention_heads)
    writer.add_uint32("speecht5.encoder_ffn_dim", encoder_ffn_dim)
    writer.add_uint32("speecht5.decoder_ffn_dim", decoder_ffn_dim)
    writer.add_uint32("speecht5.vocab_size", vocab_size)
    writer.add_uint32("speecht5.reduction_factor", reduction_factor)
    writer.add_uint32("speecht5.speech_decoder_prenet_layers", speech_decoder_prenet_layers)
    writer.add_uint32("speecht5.speech_decoder_prenet_units", speech_decoder_prenet_units)
    writer.add_uint32("speecht5.speech_decoder_postnet_layers", speech_decoder_postnet_layers)
    writer.add_uint32("speecht5.speech_decoder_postnet_units", speech_decoder_postnet_units)
    writer.add_uint32("speecht5.speech_decoder_postnet_kernel", speech_decoder_postnet_kernel)
    writer.add_uint32("speecht5.speaker_embedding_dim", speaker_embedding_dim)
    writer.add_uint32("speecht5.max_text_positions", max_text_positions)
    writer.add_uint32("speecht5.max_speech_positions", max_speech_positions)
    writer.add_uint32("speecht5.encoder_max_relative_position", encoder_max_relative_position)
    writer.add_float32("speecht5.layer_norm_eps", layer_norm_eps)

    # Vocoder KV
    writer.add_uint32("speecht5.vocoder.model_in_dim", voc_model_in_dim)
    writer.add_uint32("speecht5.vocoder.upsample_initial_channel", voc_upsample_initial_channel)
    writer.add_array("speecht5.vocoder.upsample_rates",
                     [int(x) for x in voc_upsample_rates])
    writer.add_array("speecht5.vocoder.upsample_kernel_sizes",
                     [int(x) for x in voc_upsample_kernel_sizes])
    writer.add_array("speecht5.vocoder.resblock_kernel_sizes",
                     [int(x) for x in voc_resblock_kernel_sizes])
    # Flatten dilation sizes for GGUF (store as flat array)
    flat_dilations = []
    for d in voc_resblock_dilation_sizes:
        flat_dilations.extend(d)
    writer.add_array("speecht5.vocoder.resblock_dilation_sizes",
                     [int(x) for x in flat_dilations])
    writer.add_uint32("speecht5.vocoder.num_dilations_per_resblock",
                      len(voc_resblock_dilation_sizes[0]) if voc_resblock_dilation_sizes else 3)
    writer.add_float32("speecht5.vocoder.leaky_relu_slope", voc_leaky_relu_slope)
    writer.add_bool("speecht5.vocoder.normalize_before", voc_normalize_before)
    writer.add_uint32("speecht5.vocoder.sampling_rate", voc_sampling_rate)

    # Vocab
    if vocab:
        writer.add_array("speecht5.vocab", vocab)
        print(f"Vocab: {len(vocab)} tokens", file=sys.stderr)

    # ── Tensor renaming: HF TTS model → flat GGUF names ─────────────

    def add_tensor(name: str, data: torch.Tensor, dtype=GGMLQuantizationType.F16):
        arr = data.detach().float().numpy()
        # Force 1D tensors (biases, norms, scalars) to F32 for ggml compatibility
        if arr.ndim <= 1 or dtype == GGMLQuantizationType.F32:
            arr = arr.astype(np.float32)
        else:
            arr = arr.astype(np.float16)
        writer.add_tensor(name, arr)

    n_tensors = 0

    # --- Text encoder prenet ---
    prefix_map = {
        "speecht5.encoder.prenet.embed_tokens.weight": "enc.embed.weight",
        "speecht5.encoder.prenet.encode_positions.alpha": "enc.pos_alpha",
    }
    for hf_name, gguf_name in prefix_map.items():
        if hf_name in tts_sd:
            # Embeddings stay F32 for accuracy
            dt = GGMLQuantizationType.F32 if "embed" in gguf_name else GGMLQuantizationType.F16
            add_tensor(gguf_name, tts_sd[hf_name], dt)
            n_tensors += 1

    # --- Encoder layers ---
    # Encoder layer norm (before layers) + relative position encoding
    enc_ln_keys = {
        "speecht5.encoder.wrapped_encoder.layer_norm.weight": "enc.ln.weight",
        "speecht5.encoder.wrapped_encoder.layer_norm.bias": "enc.ln.bias",
        "speecht5.encoder.wrapped_encoder.embed_positions.pe_k.weight": "enc.rel_pos.weight",
    }
    for hf_name, gguf_name in enc_ln_keys.items():
        if hf_name in tts_sd:
            dt = GGMLQuantizationType.F32 if "ln" in gguf_name else GGMLQuantizationType.F16
            add_tensor(gguf_name, tts_sd[hf_name], dt)
            n_tensors += 1

    for i in range(encoder_layers):
        hf_prefix = f"speecht5.encoder.wrapped_encoder.layers.{i}"
        gg_prefix = f"enc.layer.{i}"

        layer_keys = {
            # Self attention
            f"{hf_prefix}.attention.q_proj.weight": f"{gg_prefix}.attn.q.weight",
            f"{hf_prefix}.attention.q_proj.bias": f"{gg_prefix}.attn.q.bias",
            f"{hf_prefix}.attention.k_proj.weight": f"{gg_prefix}.attn.k.weight",
            f"{hf_prefix}.attention.k_proj.bias": f"{gg_prefix}.attn.k.bias",
            f"{hf_prefix}.attention.v_proj.weight": f"{gg_prefix}.attn.v.weight",
            f"{hf_prefix}.attention.v_proj.bias": f"{gg_prefix}.attn.v.bias",
            f"{hf_prefix}.attention.out_proj.weight": f"{gg_prefix}.attn.o.weight",
            f"{hf_prefix}.attention.out_proj.bias": f"{gg_prefix}.attn.o.bias",
            # LayerNorm after self-attn
            f"{hf_prefix}.layer_norm.weight": f"{gg_prefix}.ln1.weight",
            f"{hf_prefix}.layer_norm.bias": f"{gg_prefix}.ln1.bias",
            # FFN
            f"{hf_prefix}.feed_forward.intermediate_dense.weight": f"{gg_prefix}.ffn.up.weight",
            f"{hf_prefix}.feed_forward.intermediate_dense.bias": f"{gg_prefix}.ffn.up.bias",
            f"{hf_prefix}.feed_forward.output_dense.weight": f"{gg_prefix}.ffn.down.weight",
            f"{hf_prefix}.feed_forward.output_dense.bias": f"{gg_prefix}.ffn.down.bias",
            # Final layer norm
            f"{hf_prefix}.final_layer_norm.weight": f"{gg_prefix}.ln2.weight",
            f"{hf_prefix}.final_layer_norm.bias": f"{gg_prefix}.ln2.bias",
        }

        for hf_name, gguf_name in layer_keys.items():
            if hf_name in tts_sd:
                dt = GGMLQuantizationType.F32 if ".ln" in gguf_name else GGMLQuantizationType.F16
                add_tensor(gguf_name, tts_sd[hf_name], dt)
                n_tensors += 1

    # --- Speech decoder prenet ---
    for j in range(speech_decoder_prenet_layers):
        hf_w = f"speecht5.decoder.prenet.layers.{j}.weight"
        hf_b = f"speecht5.decoder.prenet.layers.{j}.bias"
        if hf_w in tts_sd:
            add_tensor(f"dec.prenet.{j}.weight", tts_sd[hf_w])
            n_tensors += 1
        if hf_b in tts_sd:
            add_tensor(f"dec.prenet.{j}.bias", tts_sd[hf_b])
            n_tensors += 1

    dec_prenet_keys = {
        "speecht5.decoder.prenet.final_layer.weight": "dec.prenet.final.weight",
        "speecht5.decoder.prenet.final_layer.bias": "dec.prenet.final.bias",
        "speecht5.decoder.prenet.encode_positions.alpha": "dec.pos_alpha",
        "speecht5.decoder.prenet.speaker_embeds_layer.weight": "dec.spk_proj.weight",
        "speecht5.decoder.prenet.speaker_embeds_layer.bias": "dec.spk_proj.bias",
    }
    for hf_name, gguf_name in dec_prenet_keys.items():
        if hf_name in tts_sd:
            add_tensor(gguf_name, tts_sd[hf_name])
            n_tensors += 1

    # --- Decoder layers ---
    for i in range(decoder_layers):
        hf_prefix = f"speecht5.decoder.wrapped_decoder.layers.{i}"
        gg_prefix = f"dec.layer.{i}"

        layer_keys = {
            # Self attention
            f"{hf_prefix}.self_attn.q_proj.weight": f"{gg_prefix}.self_attn.q.weight",
            f"{hf_prefix}.self_attn.q_proj.bias": f"{gg_prefix}.self_attn.q.bias",
            f"{hf_prefix}.self_attn.k_proj.weight": f"{gg_prefix}.self_attn.k.weight",
            f"{hf_prefix}.self_attn.k_proj.bias": f"{gg_prefix}.self_attn.k.bias",
            f"{hf_prefix}.self_attn.v_proj.weight": f"{gg_prefix}.self_attn.v.weight",
            f"{hf_prefix}.self_attn.v_proj.bias": f"{gg_prefix}.self_attn.v.bias",
            f"{hf_prefix}.self_attn.out_proj.weight": f"{gg_prefix}.self_attn.o.weight",
            f"{hf_prefix}.self_attn.out_proj.bias": f"{gg_prefix}.self_attn.o.bias",
            # Self-attn layer norm
            f"{hf_prefix}.self_attn_layer_norm.weight": f"{gg_prefix}.ln_self.weight",
            f"{hf_prefix}.self_attn_layer_norm.bias": f"{gg_prefix}.ln_self.bias",
            # Cross attention
            f"{hf_prefix}.encoder_attn.q_proj.weight": f"{gg_prefix}.cross_attn.q.weight",
            f"{hf_prefix}.encoder_attn.q_proj.bias": f"{gg_prefix}.cross_attn.q.bias",
            f"{hf_prefix}.encoder_attn.k_proj.weight": f"{gg_prefix}.cross_attn.k.weight",
            f"{hf_prefix}.encoder_attn.k_proj.bias": f"{gg_prefix}.cross_attn.k.bias",
            f"{hf_prefix}.encoder_attn.v_proj.weight": f"{gg_prefix}.cross_attn.v.weight",
            f"{hf_prefix}.encoder_attn.v_proj.bias": f"{gg_prefix}.cross_attn.v.bias",
            f"{hf_prefix}.encoder_attn.out_proj.weight": f"{gg_prefix}.cross_attn.o.weight",
            f"{hf_prefix}.encoder_attn.out_proj.bias": f"{gg_prefix}.cross_attn.o.bias",
            # Cross-attn layer norm
            f"{hf_prefix}.encoder_attn_layer_norm.weight": f"{gg_prefix}.ln_cross.weight",
            f"{hf_prefix}.encoder_attn_layer_norm.bias": f"{gg_prefix}.ln_cross.bias",
            # FFN
            f"{hf_prefix}.feed_forward.intermediate_dense.weight": f"{gg_prefix}.ffn.up.weight",
            f"{hf_prefix}.feed_forward.intermediate_dense.bias": f"{gg_prefix}.ffn.up.bias",
            f"{hf_prefix}.feed_forward.output_dense.weight": f"{gg_prefix}.ffn.down.weight",
            f"{hf_prefix}.feed_forward.output_dense.bias": f"{gg_prefix}.ffn.down.bias",
            # Final layer norm
            f"{hf_prefix}.final_layer_norm.weight": f"{gg_prefix}.ln_final.weight",
            f"{hf_prefix}.final_layer_norm.bias": f"{gg_prefix}.ln_final.bias",
        }

        for hf_name, gguf_name in layer_keys.items():
            if hf_name in tts_sd:
                dt = GGMLQuantizationType.F32 if ".ln" in gguf_name else GGMLQuantizationType.F16
                add_tensor(gguf_name, tts_sd[hf_name], dt)
                n_tensors += 1

    # --- Decoder postnet ---
    postnet_keys = {
        "speech_decoder_postnet.feat_out.weight": "dec.postnet.feat_out.weight",
        "speech_decoder_postnet.feat_out.bias": "dec.postnet.feat_out.bias",
        "speech_decoder_postnet.prob_out.weight": "dec.postnet.prob_out.weight",
        "speech_decoder_postnet.prob_out.bias": "dec.postnet.prob_out.bias",
    }
    for hf_name, gguf_name in postnet_keys.items():
        if hf_name in tts_sd:
            add_tensor(gguf_name, tts_sd[hf_name])
            n_tensors += 1

    # Postnet conv layers (5 layers, each with conv + batch_norm)
    for i in range(speech_decoder_postnet_layers):
        hf_prefix = f"speech_decoder_postnet.layers.{i}"
        gg_prefix = f"dec.postnet.conv.{i}"

        conv_keys = {
            f"{hf_prefix}.conv.weight": f"{gg_prefix}.weight",
            # Conv has no bias when using batch norm
            f"{hf_prefix}.batch_norm.weight": f"{gg_prefix}.bn.weight",
            f"{hf_prefix}.batch_norm.bias": f"{gg_prefix}.bn.bias",
            f"{hf_prefix}.batch_norm.running_mean": f"{gg_prefix}.bn.mean",
            f"{hf_prefix}.batch_norm.running_var": f"{gg_prefix}.bn.var",
        }
        for hf_name, gguf_name in conv_keys.items():
            if hf_name in tts_sd:
                # BN stats as F32 for numerical stability
                dt = GGMLQuantizationType.F32 if ".bn." in gguf_name else GGMLQuantizationType.F16
                add_tensor(gguf_name, tts_sd[hf_name], dt)
                n_tensors += 1

    # --- HiFi-GAN vocoder ---
    voc_prefix_map = {
        "conv_pre.weight": "voc.conv_pre.weight",
        "conv_pre.bias": "voc.conv_pre.bias",
        "conv_post.weight": "voc.conv_post.weight",
        "conv_post.bias": "voc.conv_post.bias",
        "mean": "voc.mean",
        "scale": "voc.scale",
    }
    for hf_name, gguf_name in voc_prefix_map.items():
        if hf_name in voc_sd:
            dt = GGMLQuantizationType.F32 if hf_name in ("mean", "scale") else GGMLQuantizationType.F16
            add_tensor(gguf_name, voc_sd[hf_name], dt)
            n_tensors += 1

    # Upsampler layers
    num_upsamples = len(voc_upsample_rates)
    for i in range(num_upsamples):
        for suffix in ["weight", "bias"]:
            hf_name = f"upsampler.{i}.{suffix}"
            if hf_name in voc_sd:
                add_tensor(f"voc.ups.{i}.{suffix}", voc_sd[hf_name])
                n_tensors += 1

    # Resblocks
    num_kernels = len(voc_resblock_kernel_sizes)
    for i in range(num_upsamples):
        for j in range(num_kernels):
            rb_idx = i * num_kernels + j
            num_dilations = len(voc_resblock_dilation_sizes[j]) if j < len(voc_resblock_dilation_sizes) else 3

            for d in range(num_dilations):
                for conv_set in ["convs1", "convs2"]:
                    for suffix in ["weight", "bias"]:
                        hf_name = f"resblocks.{rb_idx}.{conv_set}.{d}.{suffix}"
                        if hf_name in voc_sd:
                            add_tensor(f"voc.resblocks.{rb_idx}.{conv_set}.{d}.{suffix}",
                                       voc_sd[hf_name])
                            n_tensors += 1

    print(f"Wrote {n_tensors} tensors to {out_path}", file=sys.stderr)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    # Report size
    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"Output: {out_path} ({size_mb:.1f} MB)", file=sys.stderr)


def main():
    ap = argparse.ArgumentParser(description="Convert SpeechT5 TTS + HiFi-GAN to GGUF")
    ap.add_argument("--tts", required=True,
                    help="SpeechT5 TTS model directory or HF repo ID (e.g. microsoft/speecht5_tts)")
    ap.add_argument("--vocoder", required=True,
                    help="HiFi-GAN vocoder directory or HF repo ID (e.g. microsoft/speecht5_hifigan)")
    ap.add_argument("--output", "-o", required=True,
                    help="Output GGUF file path")
    args = ap.parse_args()
    convert(args)


if __name__ == "__main__":
    main()
