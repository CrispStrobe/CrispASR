#!/usr/bin/env python3
"""
Convert NVIDIA FastPitch + HiFi-GAN (.nemo archives or HuggingFace) -> single GGUF
for the CrispASR `fastpitch` backend.

FastPitch architecture (non-autoregressive parallel TTS):
  - Text encoder: N-layer bidirectional Transformer (FFTransformerEncoder)
    with ConditionalLayerNorm, PositionwiseConvFF, MultiHeadAttn
  - Duration predictor: TemporalPredictor (Conv1d stack + Linear)
  - Pitch predictor: TemporalPredictor (Conv1d stack + Linear)
  - Pitch embedding: Conv1d (pitch values -> embedding space)
  - Length regulator: repeat_interleave by predicted durations
  - Mel decoder: N-layer Transformer (FFTransformerDecoder)
  - Output projection: Linear -> n_mel_channels
  - Speaker embedding: Embedding lookup (multi-speaker)
  - HiFi-GAN vocoder: conv_pre + upsample stages with MRF resblocks + conv_post

Produces ONE GGUF with all weights + hparams as KV metadata.

Usage:
    # From NeMo .nemo files:
    python models/convert-fastpitch-to-gguf.py \
        --nemo /mnt/storage/fastpitch/German_multispeaker_FastPitch_nemo.nemo \
        --vocoder-nemo /mnt/storage/fastpitch/tts_de_hui_hifigan_ft_fastpitch_multispeaker_5.nemo \
        --output /mnt/storage/fastpitch/fastpitch-de-multi-f16.gguf

    # From HuggingFace:
    python models/convert-fastpitch-to-gguf.py \
        --hf-model inOXcrm/German_multispeaker_FastPitch_nemo \
        --hf-vocoder nvidia/tts_de_hui_hifigan_ft_fastpitch_multispeaker_5 \
        --output /mnt/storage/fastpitch/fastpitch-de-multi-f16.gguf
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
import tarfile
import tempfile
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
# NeMo archive loading
# ---------------------------------------------------------------------------


def load_nemo_archive(nemo_path: str):
    """Load a .nemo archive, returning (config_dict, state_dict).

    NeMo archives can be gzip-compressed or plain tar.
    """
    import yaml

    # Try gzip first, fall back to plain tar
    for mode in ("r:gz", "r"):
        try:
            with tarfile.open(nemo_path, mode) as tar:
                config = None
                weights = None

                for member in tar.getmembers():
                    if member.name.endswith("model_config.yaml"):
                        f = tar.extractfile(member)
                        config = yaml.safe_load(f)
                    elif member.name.endswith("model_weights.ckpt"):
                        f = tar.extractfile(member)
                        buf = io.BytesIO(f.read())
                        weights = torch.load(buf, map_location="cpu",
                                             weights_only=False)

                if config is None:
                    raise ValueError(f"No model_config.yaml in {nemo_path}")
                if weights is None:
                    raise ValueError(f"No model_weights.ckpt in {nemo_path}")

                return config, weights
        except tarfile.ReadError:
            continue

    raise ValueError(f"Cannot open {nemo_path} as tar archive")


def load_from_hf(model_name: str, model_class_name: str):
    """Load model from HuggingFace, return (config, state_dict, model)."""
    if model_class_name == "FastPitchModel":
        from nemo.collections.tts.models import FastPitchModel
        model = FastPitchModel.from_pretrained(model_name)
    elif model_class_name == "HifiGanModel":
        from nemo.collections.tts.models import HifiGanModel
        model = HifiGanModel.from_pretrained(model_name)
    else:
        raise ValueError(f"Unknown model class: {model_class_name}")

    model = model.eval().cpu()
    return model.cfg, model.state_dict(), model


# ---------------------------------------------------------------------------
# Weight renaming: NeMo state_dict keys -> GGUF tensor names
# ---------------------------------------------------------------------------


def generate_sinusoidal_pe(inv_freq: torch.Tensor, max_len: int = 2048) -> np.ndarray:
    """Generate sinusoidal positional encodings from inv_freq tensor.

    NeMo PositionalEmbedding computes:
      sinusoid_inp = pos_seq.unsqueeze(-1) @ inv_freq.unsqueeze(0)  # (T, d/2)
      pos_emb = cat([sin(sinusoid_inp), cos(sinusoid_inp)], dim=1)  # (T, d)

    So the layout is [sin_0..sin_{d/2-1}, cos_0..cos_{d/2-1}] (concatenated,
    NOT interleaved). The output shape is (1, T, d) but we store as (T, d).
    """
    d_half = inv_freq.shape[0]
    d_model = d_half * 2
    positions = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)  # (max_len, 1)
    inv_freq_row = inv_freq.float().unsqueeze(0)  # (1, d_half)
    sinusoid = positions * inv_freq_row  # (max_len, d_half)
    # NeMo: cat([sin, cos], dim=1) — sin first half, cos second half
    pe = torch.cat([sinusoid.sin(), sinusoid.cos()], dim=1)  # (max_len, d_model)
    return pe.numpy()  # (max_len, d_model)


def rename_fastpitch_weights(state_dict: dict) -> dict:
    """
    Rename FastPitch state_dict keys to clean GGUF tensor names.

    Handles both NeMo naming conventions:
      Convention A (nvidia/tts_en_fastpitch, newer NeMo):
        .dec_attn.layer_norm. -> .attn_norm.
        .pos_ff.layer_norm.   -> .ffn_norm.
        .pos_ff.CoreNet.0.    -> .ffn.conv1.
        .pos_ff.CoreNet.2.    -> .ffn.conv2.
        .pos_emb.inv_freq     -> (generate sinusoidal PE)
      Convention B (German multispeaker, older NeMo):
        .norm1.               -> .attn_norm.
        .norm2.               -> .ffn_norm.
        .pos_ff.conv_1.       -> .ffn.conv1.
        .pos_ff.conv_2.       -> .ffn.conv2.
        .pos_emb.pe           -> pos_emb (stored directly)

    We produce:
      enc.emb.weight
      enc.pos_emb                        (d_model, max_len) sinusoidal PE
      enc.layer.{i}.attn.qkv.weight / .bias
      enc.layer.{i}.attn.out.weight
      enc.layer.{i}.attn_norm.weight / .bias
      enc.layer.{i}.ffn.conv1.weight / .bias
      enc.layer.{i}.ffn.conv2.weight / .bias
      enc.layer.{i}.ffn_norm.weight / .bias
      dur_pred.conv.{i}.weight / .bias
      dur_pred.norm.{i}.weight / .bias
      dur_pred.fc.weight / .bias
      pitch_pred.conv.{i}.weight / .bias
      pitch_pred.norm.{i}.weight / .bias
      pitch_pred.fc.weight / .bias
      pitch_emb.weight / .bias
      speaker_emb.weight
      dec.layer.{i}.... (same structure as enc)
      dec.pos_emb
      proj.weight / .bias
    """
    import re
    renamed = {}

    # Collect inv_freq tensors for sinusoidal PE generation
    enc_inv_freq = None
    dec_inv_freq = None

    for key, tensor in state_dict.items():
        new_key = key

        # Strip "fastpitch." prefix if present
        if new_key.startswith("fastpitch."):
            new_key = new_key[len("fastpitch."):]

        # Skip training-only aligner weights
        if new_key.startswith("aligner."):
            continue

        # Skip pitch_mean/pitch_std (stored as metadata, not tensors)
        if new_key in ("pitch_mean", "pitch_std"):
            continue

        # Handle inv_freq: collect and generate PE later
        if new_key == "encoder.pos_emb.inv_freq":
            enc_inv_freq = tensor
            continue
        if new_key == "decoder.pos_emb.inv_freq":
            dec_inv_freq = tensor
            continue

        # Encoder
        new_key = new_key.replace("encoder.word_emb.weight", "enc.emb.weight")
        new_key = new_key.replace("encoder.pos_emb.pe", "enc.pos_emb")

        # Encoder/decoder layer renames (handle both naming conventions)
        def rename_layer(k, src_prefix, dst_prefix):
            k = k.replace(src_prefix, dst_prefix)
            # Attention
            k = k.replace(".dec_attn.qkv_net.", ".attn.qkv.")
            k = k.replace(".dec_attn.o_net.", ".attn.out.")
            # Convention A: LayerNorm inside submodule
            k = k.replace(".dec_attn.layer_norm.", ".attn_norm.")
            k = k.replace(".pos_ff.layer_norm.", ".ffn_norm.")
            # Convention B: LayerNorm at layer level
            k = k.replace(".norm1.", ".attn_norm.")
            k = k.replace(".norm2.", ".ffn_norm.")
            # Convention A: CoreNet.{0,2} for FFN convolutions
            k = k.replace(".pos_ff.CoreNet.0.", ".ffn.conv1.")
            k = k.replace(".pos_ff.CoreNet.2.", ".ffn.conv2.")
            # Convention B: conv_1/conv_2
            k = k.replace(".pos_ff.conv_1.", ".ffn.conv1.")
            k = k.replace(".pos_ff.conv_2.", ".ffn.conv2.")
            return k

        if "encoder.layers." in new_key:
            new_key = rename_layer(new_key, "encoder.layers.", "enc.layer.")
        elif "encoder.cond_input." in new_key:
            new_key = new_key.replace("encoder.cond_input.", "enc.cond_input.")

        if "decoder.layers." in new_key:
            new_key = rename_layer(new_key, "decoder.layers.", "dec.layer.")
        elif "decoder.pos_emb.pe" in new_key:
            new_key = new_key.replace("decoder.pos_emb.pe", "dec.pos_emb")
        elif "decoder.cond_input." in new_key:
            new_key = new_key.replace("decoder.cond_input.", "dec.cond_input.")

        # Duration predictor
        if "duration_predictor." in new_key:
            new_key = new_key.replace("duration_predictor.", "dur_pred.")
            new_key = new_key.replace(".layers.", ".block.")
            m = re.match(r"dur_pred\.block\.(\d+)\.(conv|norm)\.(weight|bias)", new_key)
            if m:
                idx, sub, wb = m.groups()
                new_key = f"dur_pred.{sub}.{idx}.{wb}"

        # Pitch predictor
        if "pitch_predictor." in new_key:
            new_key = new_key.replace("pitch_predictor.", "pitch_pred.")
            new_key = new_key.replace(".layers.", ".block.")
            m = re.match(r"pitch_pred\.block\.(\d+)\.(conv|norm)\.(weight|bias)", new_key)
            if m:
                idx, sub, wb = m.groups()
                new_key = f"pitch_pred.{sub}.{idx}.{wb}"

        # ConditionalLayerNorm projections
        new_key = new_key.replace(".cond_weight.", ".cond_w.")
        new_key = new_key.replace(".cond_bias.", ".cond_b.")

        renamed[new_key] = tensor

    # Generate sinusoidal positional encodings from inv_freq
    if enc_inv_freq is not None:
        pe = generate_sinusoidal_pe(enc_inv_freq, max_len=2048)
        renamed["enc.pos_emb"] = pe
        print(f"  generated enc.pos_emb from inv_freq: {pe.shape}")
    if dec_inv_freq is not None:
        pe = generate_sinusoidal_pe(dec_inv_freq, max_len=2048)
        renamed["dec.pos_emb"] = pe
        print(f"  generated dec.pos_emb from inv_freq: {pe.shape}")

    return renamed


def rename_vocoder_weights(state_dict: dict) -> dict:
    """Rename HiFi-GAN vocoder state_dict keys to voc.* namespace.

    Skips discriminator (mpd.*, msd.*) and mel-spec processor weights
    which are training-only and not needed for inference.
    """
    renamed = {}

    for key, tensor in state_dict.items():
        # Skip training-only weights
        if any(key.startswith(p) for p in [
            "mpd.", "msd.", "audio_to_melspec", "trg_melspec"
        ]):
            continue

        new_key = key

        # Strip common NeMo prefixes
        for prefix in ["generator.", "vocoder.", "hifi_gan."]:
            if new_key.startswith(prefix):
                new_key = new_key[len(prefix):]
                break

        # Add voc. prefix
        new_key = "voc." + new_key

        renamed[new_key] = tensor

    return renamed


def fuse_weight_norm(state_dict: dict) -> dict:
    """Fuse weight_norm (weight_g, weight_v) pairs into a single weight tensor."""
    fused = {}
    processed = set()

    for key in list(state_dict.keys()):
        if key.endswith(".weight_g"):
            base = key[:-len(".weight_g")]
            v_key = base + ".weight_v"
            if v_key in state_dict:
                g = state_dict[key]
                v = state_dict[v_key]
                # weight_norm: w = g * v / ||v||
                # g shape: (out_ch, 1, 1) for conv, v shape: (out_ch, in_ch, kernel)
                norm = torch.norm(v.reshape(v.shape[0], -1), dim=1, keepdim=True)
                for _ in range(v.ndim - 2):
                    norm = norm.unsqueeze(-1)
                g_expanded = g
                while g_expanded.ndim < v.ndim:
                    g_expanded = g_expanded.unsqueeze(-1)
                fused_w = g_expanded * v / (norm + 1e-12)
                fused[base + ".weight"] = fused_w
                processed.add(key)
                processed.add(v_key)

    # Add non-weight_norm tensors
    for key, tensor in state_dict.items():
        if key not in processed:
            fused[key] = tensor

    return fused


# ---------------------------------------------------------------------------
# Conv weight transposition: PyTorch (Cout, Cin, K) -> ggml (K, Cin, Cout)
# For ConvTranspose1d: PyTorch (Cin, Cout, K) -> ggml (K, Cout, Cin)
# ---------------------------------------------------------------------------


def transpose_conv_weight(arr: np.ndarray) -> np.ndarray:
    """Transpose 3D conv weight from PyTorch to ggml layout.

    PyTorch Conv1d weight: (Cout, Cin, K)
    Numpy row-major -> ggml column-major mapping: last axis -> ne[0]
    So (Cout, Cin, K) numpy -> ggml ne[0]=K, ne[1]=Cin, ne[2]=Cout
    This is ALREADY what ggml_conv_1d expects!
    No transpose needed — just return as-is.
    """
    return arr


def is_conv_weight(name: str, shape: tuple) -> bool:
    """Check if a tensor is a Conv1d weight (3D, not embedding)."""
    if len(shape) != 3:
        return False
    # Conv1d weights are always 3D: (Cout, Cin, K) or (Cin, Cout, K) for transpose
    # But NOT embedding tables which can also be 3D
    if "emb" in name and "conv" not in name.lower():
        return False
    return True


def is_conv_transpose_weight(name: str) -> bool:
    """Check if a tensor is a ConvTranspose1d weight."""
    return "ups." in name and name.endswith(".weight")


# ---------------------------------------------------------------------------
# Hparams extraction
# ---------------------------------------------------------------------------


def extract_hparams_from_config(cfg: dict) -> dict:
    """Extract FastPitch hparams from NeMo config dict."""
    # NeMo config can be nested under different keys
    fp_cfg = cfg
    if "model" in cfg:
        fp_cfg = cfg["model"]

    # Encoder hparams
    enc = fp_cfg.get("input_fft", fp_cfg.get("encoder", {}))
    dec = fp_cfg.get("output_fft", fp_cfg.get("decoder", {}))
    dur = fp_cfg.get("duration_predictor", {})
    pitch = fp_cfg.get("pitch_predictor", {})

    hparams = {
        # General
        "n_mel_channels": fp_cfg.get("n_mel_channels", 80),
        "n_speakers": fp_cfg.get("n_speakers", 5),
        "symbols_embedding_dim": fp_cfg.get("symbols_embedding_dim",
                                             fp_cfg.get("d_model", 384)),
        "max_token_duration": fp_cfg.get("max_token_duration", 75),

        # Encoder
        "enc_n_layers": enc.get("n_layer", enc.get("n_layers", 6)),
        "enc_n_heads": enc.get("n_head", enc.get("n_heads", 1)),
        "enc_d_head": enc.get("d_head", 64),
        "enc_d_inner": enc.get("d_inner", enc.get("filter_size", 1024)),
        "enc_kernel_sizes": enc.get("kernel_sizes",
                                     enc.get("kernel_size", [3, 3])),
        "enc_dropout": enc.get("dropout", enc.get("dropatt", 0.1)),

        # Decoder
        "dec_n_layers": dec.get("n_layer", dec.get("n_layers", 6)),
        "dec_n_heads": dec.get("n_head", dec.get("n_heads", 1)),
        "dec_d_head": dec.get("d_head", 64),
        "dec_d_inner": dec.get("d_inner", dec.get("filter_size", 1024)),
        "dec_kernel_sizes": dec.get("kernel_sizes",
                                     dec.get("kernel_size", [3, 3])),
        "dec_dropout": dec.get("dropout", dec.get("dropatt", 0.1)),

        # Duration predictor
        "dur_n_layers": dur.get("n_layers", 2),
        "dur_filter_size": dur.get("filter_size", 256),
        "dur_kernel_size": dur.get("kernel_size", 3),

        # Pitch predictor
        "pitch_n_layers": pitch.get("n_layers", 2),
        "pitch_filter_size": pitch.get("filter_size", 256),
        "pitch_kernel_size": pitch.get("kernel_size", 3),

        # Pitch embedding
        "pitch_embedding_kernel_size": fp_cfg.get("pitch_embedding_kernel_size", 3),

        # Audio
        "sample_rate": fp_cfg.get("sample_rate",
                                   cfg.get("sample_rate", 22050)),
    }

    return hparams


def extract_hparams_from_weights(state_dict: dict) -> dict:
    """Infer hparams from weight tensor shapes when config is unavailable."""
    hparams = {}

    # Embedding dim from enc.emb.weight
    emb = state_dict.get("enc.emb.weight")
    if emb is not None:
        t = emb if isinstance(emb, np.ndarray) else emb.numpy()
        hparams["n_symbols"] = t.shape[0]
        hparams["symbols_embedding_dim"] = t.shape[1]

    # Count encoder layers
    n_enc = 0
    while f"enc.layer.{n_enc}.attn.qkv.weight" in state_dict:
        n_enc += 1
    hparams["enc_n_layers"] = n_enc

    # Count decoder layers
    n_dec = 0
    while f"dec.layer.{n_dec}.attn.qkv.weight" in state_dict:
        n_dec += 1
    hparams["dec_n_layers"] = n_dec

    # Attention heads from QKV weight shape
    qkv = state_dict.get("enc.layer.0.attn.qkv.weight")
    if qkv is not None:
        t = qkv if isinstance(qkv, np.ndarray) else qkv.numpy()
        d_model = t.shape[1] if t.ndim == 2 else t.shape[0]
        d_qkv = t.shape[0] if t.ndim == 2 else t.shape[1]
        # d_qkv = 3 * n_heads * d_head
        hparams["enc_d_model"] = d_model

    # Duration predictor layers
    n_dur = 0
    while f"dur_pred.conv.{n_dur}.weight" in state_dict:
        n_dur += 1
    hparams["dur_n_layers"] = n_dur

    # Pitch predictor layers
    n_pitch = 0
    while f"pitch_pred.conv.{n_pitch}.weight" in state_dict:
        n_pitch += 1
    hparams["pitch_n_layers"] = n_pitch

    # Speaker embedding
    spk = state_dict.get("speaker_emb.weight")
    if spk is not None:
        t = spk if isinstance(spk, np.ndarray) else spk.numpy()
        hparams["n_speakers"] = t.shape[0]

    # Output projection -> mel channels
    proj = state_dict.get("proj.weight")
    if proj is not None:
        t = proj if isinstance(proj, np.ndarray) else proj.numpy()
        hparams["n_mel_channels"] = t.shape[0]

    return hparams


def extract_vocoder_hparams(state_dict: dict) -> dict:
    """Extract HiFi-GAN vocoder hparams from weights."""
    hparams = {}

    # conv_pre input channels = mel_dim
    conv_pre = state_dict.get("voc.conv_pre.weight")
    if conv_pre is not None:
        t = conv_pre if isinstance(conv_pre, np.ndarray) else conv_pre.numpy()
        hparams["voc_model_in_dim"] = t.shape[1] if t.ndim == 3 else 80
        hparams["voc_upsample_initial_ch"] = t.shape[0] if t.ndim == 3 else 512

    # Count upsample stages and get rates/kernels
    upsample_rates = []
    upsample_kernels = []
    i = 0
    while f"voc.ups.{i}.weight" in state_dict:
        w = state_dict[f"voc.ups.{i}.weight"]
        t = w if isinstance(w, np.ndarray) else w.numpy()
        kernel = t.shape[2] if t.ndim == 3 else 0
        # For ConvTranspose1d, stride typically = kernel // 2
        # But we can also infer from channel ratios
        # NeMo convention: upsample_kernel_size = stride * 2
        stride = kernel // 2
        upsample_rates.append(stride)
        upsample_kernels.append(kernel)
        i += 1
    hparams["voc_upsample_rates"] = upsample_rates
    hparams["voc_upsample_kernel_sizes"] = upsample_kernels
    hparams["voc_n_upsamples"] = len(upsample_rates)

    # Count resblocks and get kernel sizes
    resblock_kernels = set()
    n_resblocks = 0
    while f"voc.resblocks.{n_resblocks}.convs1.0.weight" in state_dict:
        w = state_dict[f"voc.resblocks.{n_resblocks}.convs1.0.weight"]
        t = w if isinstance(w, np.ndarray) else w.numpy()
        k = t.shape[2] if t.ndim == 3 else 3
        resblock_kernels.add(k)
        n_resblocks += 1
    hparams["voc_n_resblocks"] = n_resblocks
    hparams["voc_resblock_kernel_sizes"] = sorted(resblock_kernels)

    # Dilations from resblock convs1 weights
    # Count dilations per resblock (number of convs1.{d} entries)
    n_dilations = 0
    while f"voc.resblocks.0.convs1.{n_dilations}.weight" in state_dict:
        n_dilations += 1
    hparams["voc_n_dilations"] = n_dilations

    return hparams


# ---------------------------------------------------------------------------
# Main converter
# ---------------------------------------------------------------------------


def build_tokenizer_vocab(cfg: dict) -> list:
    """Build the NeMo EnglishPhonemesTokenizer vocabulary from config.

    Returns the ordered list of token strings matching the model's embedding
    table, suitable for storing in GGUF metadata.
    """
    import itertools as it
    import string as strmod

    if cfg is None:
        return []

    tt = cfg.get("text_tokenizer", {})
    target = tt.get("_target_", "")
    if "EnglishPhonemes" not in target and "IPA" not in target:
        return []  # Unknown tokenizer type

    stresses = tt.get("stresses", True)
    chars = tt.get("chars", True)
    apostrophe = tt.get("apostrophe", True)
    punct = tt.get("punct", True)
    add_blank = tt.get("add_blank_at", None) is not None or tt.get("add_blank_at") is True

    CONSONANTS = ('B','CH','D','DH','F','G','HH','JH','K','L','M','N',
                  'NG','P','R','S','SH','T','TH','V','W','Y','Z','ZH')
    VOWELS = ('AA','AE','AH','AO','AW','AY','EH','ER','EY','IH',
              'IY','OW','OY','UH','UW')
    PUNCT = (',','.',  '!','?','-',':',';','/','\"','(',')',
             '[',']','{','}')

    tokens = [' ']  # space
    tokens.extend(CONSONANTS)
    vowels = list(VOWELS)
    if stresses:
        vowels = [f'{p}{s}' for p, s in it.product(vowels, (0, 1, 2))]
    tokens.extend(vowels)
    if chars:
        tokens.extend(strmod.ascii_lowercase)
    if apostrophe:
        tokens.append("'")
    if punct:
        tokens.extend(PUNCT)
    tokens.append('<pad>')
    if add_blank:
        tokens.append('<blank>')
    tokens.append('<oov>')
    return tokens


def download_nemo_from_hf(repo_id: str, filename: str = None) -> str:
    """Download a .nemo file from HuggingFace, return local path."""
    from huggingface_hub import hf_hub_download, list_repo_files

    if filename is None:
        # Find the .nemo file in the repo
        files = list_repo_files(repo_id)
        nemo_files = [f for f in files if f.endswith(".nemo")]
        if not nemo_files:
            raise ValueError(f"No .nemo file found in {repo_id}")
        filename = nemo_files[0]

    path = hf_hub_download(repo_id, filename)
    return path


def extract_pitch_stats(state_dict: dict) -> dict:
    """Extract pitch_mean and pitch_std from state dict (stored as scalar tensors)."""
    stats = {}
    for key in ("pitch_mean", "pitch_std"):
        # Try with and without fastpitch. prefix
        for prefix in ("fastpitch.", ""):
            full_key = prefix + key
            if full_key in state_dict:
                t = state_dict[full_key]
                val = t.item() if hasattr(t, "item") else float(t)
                stats[key] = val
                break
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Convert FastPitch + HiFi-GAN to GGUF")
    parser.add_argument("--nemo", help="Path to FastPitch .nemo")
    parser.add_argument("--vocoder-nemo", help="Path to HiFi-GAN vocoder .nemo")
    parser.add_argument("--hf-model",
                        default="nvidia/tts_en_fastpitch",
                        help="HuggingFace FastPitch model name")
    parser.add_argument("--hf-vocoder",
                        default="nvidia/tts_hifigan",
                        help="HuggingFace HiFi-GAN model name")
    parser.add_argument("--output", required=True, help="Output .gguf path")
    parser.add_argument("--ftype", default="f16",
                        choices=["f16", "f32", "q8_0"],
                        help="Weight storage type")
    parser.add_argument("--use-nemo-api", action="store_true",
                        help="Use NeMo model API instead of direct .nemo loading")
    args = parser.parse_args()

    # ── Load FastPitch ──
    if args.nemo:
        print(f"Loading FastPitch from .nemo: {args.nemo}")
        fp_cfg, fp_sd = load_nemo_archive(args.nemo)
    elif args.use_nemo_api:
        print(f"Loading FastPitch from HF via NeMo: {args.hf_model}")
        fp_cfg, fp_sd, _ = load_from_hf(args.hf_model, "FastPitchModel")
    else:
        # Download .nemo from HF and load directly (avoids NeMo import issues)
        print(f"Downloading FastPitch .nemo from HF: {args.hf_model}")
        nemo_path = download_nemo_from_hf(args.hf_model)
        print(f"  -> {nemo_path}")
        fp_cfg, fp_sd = load_nemo_archive(nemo_path)

    # Extract pitch stats before renaming (may be lost)
    pitch_stats = extract_pitch_stats(fp_sd)
    if pitch_stats:
        print(f"  pitch stats: {pitch_stats}")

    # ── Load Vocoder ──
    if args.vocoder_nemo:
        print(f"Loading HiFi-GAN from .nemo: {args.vocoder_nemo}")
        voc_cfg, voc_sd = load_nemo_archive(args.vocoder_nemo)
    elif args.use_nemo_api:
        print(f"Loading HiFi-GAN from HF via NeMo: {args.hf_vocoder}")
        voc_cfg, voc_sd, _ = load_from_hf(args.hf_vocoder, "HifiGanModel")
    else:
        print(f"Downloading HiFi-GAN .nemo from HF: {args.hf_vocoder}")
        nemo_path = download_nemo_from_hf(args.hf_vocoder)
        print(f"  -> {nemo_path}")
        voc_cfg, voc_sd = load_nemo_archive(nemo_path)

    # ── Fuse weight_norm ──
    print("Fusing weight_norm parameters...")
    fp_sd = fuse_weight_norm(fp_sd)
    voc_sd = fuse_weight_norm(voc_sd)

    # ── Rename weights ──
    print("Renaming FastPitch weights...")
    fp_tensors = rename_fastpitch_weights(fp_sd)

    print("Renaming vocoder weights...")
    voc_tensors = rename_vocoder_weights(voc_sd)

    # Merge
    all_tensors = {}
    all_tensors.update(fp_tensors)
    all_tensors.update(voc_tensors)

    # ── Extract hparams ──
    hp_config = extract_hparams_from_config(fp_cfg) if fp_cfg else {}
    hp_weights = extract_hparams_from_weights(fp_tensors)
    hp_voc = extract_vocoder_hparams(voc_tensors)

    # Merge: weight-derived takes precedence when config is missing
    hparams = {**hp_config, **hp_weights, **hp_voc}

    print(f"\nHyperparameters:")
    for k, v in sorted(hparams.items()):
        print(f"  {k}: {v}")

    # ── Determine ftype ──
    ftype_map = {
        "f16": GGMLQuantizationType.F16,
        "f32": GGMLQuantizationType.F32,
        "q8_0": GGMLQuantizationType.Q8_0,
    }
    target_ftype = ftype_map[args.ftype]

    # ── Write GGUF ──
    print(f"\nWriting GGUF: {args.output}")
    writer = GGUFWriter(str(args.output), arch="fastpitch")

    # KV metadata
    writer.add_uint32("fastpitch.n_mel_channels",
                      hparams.get("n_mel_channels", 80))
    writer.add_uint32("fastpitch.n_speakers",
                      hparams.get("n_speakers", 5))
    writer.add_uint32("fastpitch.symbols_embedding_dim",
                      hparams.get("symbols_embedding_dim", 384))
    writer.add_uint32("fastpitch.max_token_duration",
                      hparams.get("max_token_duration", 75))

    # Encoder
    writer.add_uint32("fastpitch.enc_n_layers",
                      hparams.get("enc_n_layers", 6))
    writer.add_uint32("fastpitch.enc_n_heads",
                      hparams.get("enc_n_heads", 1))
    writer.add_uint32("fastpitch.enc_d_head",
                      hparams.get("enc_d_head", 64))
    writer.add_uint32("fastpitch.enc_d_inner",
                      hparams.get("enc_d_inner", 1024))

    # Decoder
    writer.add_uint32("fastpitch.dec_n_layers",
                      hparams.get("dec_n_layers", 6))
    writer.add_uint32("fastpitch.dec_n_heads",
                      hparams.get("dec_n_heads", 1))
    writer.add_uint32("fastpitch.dec_d_head",
                      hparams.get("dec_d_head", 64))
    writer.add_uint32("fastpitch.dec_d_inner",
                      hparams.get("dec_d_inner", 1024))

    # Duration predictor
    writer.add_uint32("fastpitch.dur_n_layers",
                      hparams.get("dur_n_layers", 2))
    writer.add_uint32("fastpitch.dur_filter_size",
                      hparams.get("dur_filter_size", 256))
    writer.add_uint32("fastpitch.dur_kernel_size",
                      hparams.get("dur_kernel_size", 3))

    # Pitch predictor
    writer.add_uint32("fastpitch.pitch_n_layers",
                      hparams.get("pitch_n_layers", 2))
    writer.add_uint32("fastpitch.pitch_filter_size",
                      hparams.get("pitch_filter_size", 256))
    writer.add_uint32("fastpitch.pitch_kernel_size",
                      hparams.get("pitch_kernel_size", 3))

    # Pitch embedding
    writer.add_uint32("fastpitch.pitch_embedding_kernel_size",
                      hparams.get("pitch_embedding_kernel_size", 3))

    # Audio
    writer.add_uint32("fastpitch.sample_rate",
                      hparams.get("sample_rate", 22050))

    # Vocoder hparams
    writer.add_uint32("fastpitch.voc_model_in_dim",
                      hparams.get("voc_model_in_dim", 80))
    writer.add_uint32("fastpitch.voc_upsample_initial_ch",
                      hparams.get("voc_upsample_initial_ch", 512))

    voc_rates = hparams.get("voc_upsample_rates", [8, 8, 2, 2])
    voc_kernels = hparams.get("voc_upsample_kernel_sizes", [16, 16, 4, 4])
    voc_rb_kernels = hparams.get("voc_resblock_kernel_sizes", [3, 7, 11])

    writer.add_array("fastpitch.voc_upsample_rates",
                     [int(x) for x in voc_rates])
    writer.add_array("fastpitch.voc_upsample_kernel_sizes",
                     [int(x) for x in voc_kernels])
    writer.add_array("fastpitch.voc_resblock_kernel_sizes",
                     [int(x) for x in voc_rb_kernels])

    # Dilations: standard HiFi-GAN [[1,3,5],[1,3,5],[1,3,5]]
    n_dilations = hparams.get("voc_n_dilations", 3)
    # Flatten: [n_kernels * n_dilations] with standard [1,3,5] pattern
    dilation_pattern = [1, 3, 5][:n_dilations]
    flat_dilations = dilation_pattern * len(voc_rb_kernels)
    writer.add_array("fastpitch.voc_resblock_dilations",
                     [int(x) for x in flat_dilations])
    writer.add_uint32("fastpitch.voc_n_dilations", n_dilations)

    # N symbols (vocab size)
    if "n_symbols" in hparams:
        writer.add_uint32("fastpitch.n_symbols", hparams["n_symbols"])

    # Pitch statistics (for denormalization; from training config)
    if "pitch_mean" in pitch_stats:
        writer.add_float32("fastpitch.pitch_mean", pitch_stats["pitch_mean"])
    if "pitch_std" in pitch_stats:
        writer.add_float32("fastpitch.pitch_std", pitch_stats["pitch_std"])
    # Also store pitch stats from training config if available
    train_ds = fp_cfg.get("train_ds", {}) if fp_cfg else {}
    ds = train_ds.get("dataset", {}) if isinstance(train_ds, dict) else {}
    if ds.get("pitch_mean") and ds["pitch_mean"] != 0:
        writer.add_float32("fastpitch.pitch_mean", float(ds["pitch_mean"]))
        writer.add_float32("fastpitch.pitch_std", float(ds["pitch_std"]))

    # Tokenizer vocabulary (ARPABET phonemes for English)
    # Stored as semicolon-separated string for compact metadata
    vocab = build_tokenizer_vocab(fp_cfg)
    if vocab:
        writer.add_string("fastpitch.tokenizer_vocab",
                          ";".join(vocab))
        writer.add_uint32("fastpitch.tokenizer_vocab_size", len(vocab))

    # ── Write tensors ──
    n_tensors = 0
    n_params = 0

    for name, tensor in sorted(all_tensors.items()):
        if isinstance(tensor, torch.Tensor):
            arr = tensor.detach().float().numpy()
        else:
            arr = np.array(tensor, dtype=np.float32)

        # Transpose conv weights
        if is_conv_weight(name, arr.shape):
            arr = transpose_conv_weight(arr)

        # Choose quantization
        if target_ftype == GGMLQuantizationType.F16 and arr.ndim >= 2:
            qt = GGMLQuantizationType.F16
        elif target_ftype == GGMLQuantizationType.Q8_0 and arr.size >= 64:
            qt = GGMLQuantizationType.Q8_0
        else:
            qt = GGMLQuantizationType.F32

        # 1D biases and norms always F32
        if arr.ndim <= 1 or "norm" in name or name.endswith(".bias"):
            qt = GGMLQuantizationType.F32

        writer.add_tensor(name, arr, raw_dtype=qt)
        n_tensors += 1
        n_params += arr.size

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    file_size = os.path.getsize(args.output)
    print(f"\nDone: {n_tensors} tensors, {n_params/1e6:.1f}M params, "
          f"{file_size/1024/1024:.1f} MB")


if __name__ == "__main__":
    main()
