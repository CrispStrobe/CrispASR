#!/usr/bin/env python3
"""
Convert BananaMind-TTS-V2.1-Preview (Tacotron-lite + HiFi-GAN) -> GGUF
for the CrispASR `bananamind-tts` backend.

Architecture:
  - Text encoder: Embedding -> 3x Conv1d(256, k=5) + BatchNorm + ReLU -> BiLSTM(256->128+128)
  - Decoder: Autoregressive GRU with location-sensitive attention
    - Prenet: 2x Linear+ReLU (256->128)
    - Attention GRU: GRUCell(128+256, 512)
    - Location-sensitive attention: Conv1d(2, 32, k=31) + Linear projections
    - Decoder GRU: GRUCell(512+256, 512)
    - Mel projection: Linear(768, 80*4)
    - Stop projection: Linear(768, 4)
  - Postnet: 5x Conv1d + BatchNorm + Tanh (refines mel)
  - HiFi-GAN vocoder: conv_pre -> 4 upsample stages + MRF resblocks -> conv_post

Usage:
    python models/convert-bananamind-tts-to-gguf.py \\
        --model Banaxi-Tech/BananaMind-TTS-V2.1-Preview \\
        --locale en-us \\
        --output /mnt/storage/gguf-models/bananamind-tts-en-f16.gguf

    python models/convert-bananamind-tts-to-gguf.py \\
        --model Banaxi-Tech/BananaMind-TTS-V2.1-Preview \\
        --locale de-de \\
        --output /mnt/storage/gguf-models/bananamind-tts-de-f16.gguf
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

# EU AI Act Art. 50(4): whose voice this checkpoint's preset speakers are.
# Shared with every other converter so the metadata key cannot drift — a drift
# fails OPEN (the stamp is simply never found) and nothing errors.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _speaker_identity_arg import add_speaker_identity_arg, stamp_speaker_identity


try:
    from gguf import GGUFWriter, GGMLQuantizationType
except ImportError:
    sys.exit("pip install gguf")

try:
    from safetensors.torch import load_file as load_safetensors_torch
except ImportError:
    sys.exit("pip install safetensors torch")


# ---------------------------------------------------------------------------
# Weight renaming
# ---------------------------------------------------------------------------


def rename_acoustic_weights(state_dict: dict) -> dict:
    """Rename BananaMind acoustic model weights to clean GGUF tensor names.

    PyTorch model structure:
        encoder.embedding.weight         -> enc.emb.weight
        encoder.convs.{i}.conv.weight    -> enc.conv.{i}.weight
        encoder.convs.{i}.conv.bias      -> enc.conv.{i}.bias
        encoder.convs.{i}.bn.weight      -> enc.conv_bn.{i}.weight
        encoder.convs.{i}.bn.bias        -> enc.conv_bn.{i}.bias
        encoder.convs.{i}.bn.running_mean -> enc.conv_bn.{i}.mean
        encoder.convs.{i}.bn.running_var  -> enc.conv_bn.{i}.var
        encoder.lstm.*                   -> enc.lstm.*
        prenet.net.{3*i}.weight          -> prenet.{i}.weight
        prenet.net.{3*i}.bias            -> prenet.{i}.bias
        attention_rnn.*                  -> attn_rnn.*
        attention.query_layer.*          -> attn.query.*
        attention.memory_layer.*         -> attn.memory.*
        attention.location_conv.*        -> attn.loc_conv.*
        attention.location_layer.*       -> attn.loc_fc.*
        attention.v.*                    -> attn.v.*
        decoder_rnn.*                    -> dec_rnn.*
        mel_proj.*                       -> mel_proj.*
        stop_proj.*                      -> stop_proj.*
        postnet.net.{5*i}.weight (conv)  -> postnet.conv.{i}.weight
        postnet.net.{5*i+1}.weight (bn)  -> postnet.bn.{i}.weight
        ...
    """
    renamed = {}

    for key, tensor in state_dict.items():
        new_key = key

        # ── Encoder ──
        new_key = new_key.replace("encoder.embedding.", "enc.emb.")
        # ConvBlock: encoder.convs.{i}.conv.* -> enc.conv.{i}.*
        new_key = new_key.replace("encoder.convs.", "enc.convs.")
        if "enc.convs." in new_key:
            # enc.convs.{i}.conv.weight -> enc.conv.{i}.weight
            new_key = new_key.replace(".conv.weight", ".weight")
            new_key = new_key.replace(".conv.bias", ".bias")
            # enc.convs.{i}.bn.* -> enc.conv_bn.{i}.*
            new_key = new_key.replace(".bn.weight", ".bn_w")
            new_key = new_key.replace(".bn.bias", ".bn_b")
            new_key = new_key.replace(".bn.running_mean", ".bn_mean")
            new_key = new_key.replace(".bn.running_var", ".bn_var")
            new_key = new_key.replace(".bn.num_batches_tracked", ".bn_nbt")
            # enc.convs.{i}.* -> enc.conv.{i}.*
            new_key = new_key.replace("enc.convs.", "enc.conv.")

        # LSTM
        new_key = new_key.replace("encoder.lstm.", "enc.lstm.")

        # ── Prenet ──
        # prenet.net is Sequential: [Linear, ReLU, Dropout, Linear, ReLU, Dropout]
        # We only want the Linear layers (indices 0, 3)
        if "prenet.net." in new_key:
            import re
            m = re.match(r"prenet\.net\.(\d+)\.(weight|bias)", new_key)
            if m:
                idx = int(m.group(1))
                wb = m.group(2)
                # Linear layers are at positions 0, 3 (each block is Linear+ReLU+Dropout = 3)
                layer_idx = idx // 3
                new_key = f"prenet.{layer_idx}.{wb}"

        # ── Attention RNN ──
        new_key = new_key.replace("attention_rnn.", "attn_rnn.")

        # ── Location-sensitive attention ──
        new_key = new_key.replace("attention.query_layer.", "attn.query.")
        new_key = new_key.replace("attention.memory_layer.", "attn.memory.")
        new_key = new_key.replace("attention.location_conv.", "attn.loc_conv.")
        new_key = new_key.replace("attention.location_layer.", "attn.loc_fc.")
        new_key = new_key.replace("attention.v.", "attn.v.")

        # ── Decoder RNN ──
        new_key = new_key.replace("decoder_rnn.", "dec_rnn.")

        # ── Postnet ──
        # postnet.net is Sequential: [Conv, BN, Tanh, Dropout, Conv, BN, ...]
        # For the first 4 layers: [Conv, BN, Tanh, Dropout] = 4 items
        # Last layer: [Conv, BN] = 2 items
        if "postnet.net." in new_key:
            import re
            m = re.match(r"postnet\.net\.(\d+)\.(.*)", new_key)
            if m:
                idx = int(m.group(1))
                rest = m.group(2)
                # Determine which postnet layer this is:
                # Layer structure varies. Let's figure out from the rest what kind of tensor this is.
                if rest.startswith("weight") or rest.startswith("bias"):
                    # This is a Conv1d weight/bias
                    # Conv layers appear at module indices that have .weight with 3 dims
                    pass
                # Actually, let's just map based on module type from the PyTorch model.
                # The postnet.net Sequential is:
                #   [Conv1d, BatchNorm1d,   (repeated for each layer)
                #    Tanh, Dropout,          (for non-last layers)
                #    Conv1d, BatchNorm1d]    (last layer, no Tanh/Dropout)
                # But this isn't clean. Let's figure it out from shapes.
                # Conv1d weight is 3D, BN weight is 1D.
                # We'll handle this by checking dimensionality below.
                new_key = f"postnet.{idx}.{rest}"

        # Skip batch norm tracking counter
        if "num_batches_tracked" in new_key:
            continue

        renamed[new_key] = tensor

    return renamed


def reorganize_postnet(tensors: dict) -> dict:
    """Reorganize postnet sequential indices into proper conv/bn pairs.

    The PyTorch postnet Sequential has variable-length blocks:
    - Non-last layers: [Conv1d, BatchNorm1d, Tanh, Dropout] (4 items)
    - Last layer: [Conv1d, BatchNorm1d] (2 items)

    We want: postnet.conv.{i}.weight/bias, postnet.bn.{i}.weight/bias/mean/var
    """
    import re

    postnet_entries = {}
    other_entries = {}

    for key, tensor in tensors.items():
        m = re.match(r"postnet\.(\d+)\.(.*)", key)
        if m:
            postnet_entries[key] = (int(m.group(1)), m.group(2), tensor)
        else:
            other_entries[key] = tensor

    if not postnet_entries:
        return tensors

    # Sort by module index
    sorted_entries = sorted(postnet_entries.values(), key=lambda x: x[0])

    # Group consecutive modules into conv/bn pairs
    # Conv1d weight is 3D, BN weight is 1D
    result = dict(other_entries)
    layer_idx = 0
    i = 0
    indices_by_type = {}  # module_idx -> (type, layer_idx)

    # First pass: identify which module indices are conv vs bn
    for mod_idx, rest, tensor in sorted_entries:
        if mod_idx not in indices_by_type:
            if tensor.ndim == 3:
                indices_by_type[mod_idx] = ("conv", layer_idx)
            elif tensor.ndim == 1 and "weight" in rest:
                # Could be BN weight - check if previous module was a conv
                indices_by_type[mod_idx] = ("bn", layer_idx - 1)
            else:
                indices_by_type[mod_idx] = ("bn", layer_idx - 1)

        if tensor.ndim == 3 and mod_idx not in indices_by_type:
            layer_idx += 1

    # Actually, let's be more precise. We know the structure from the code:
    # postnet = [Conv, BN, Tanh, Dropout, Conv, BN, Tanh, Dropout, ..., Conv, BN]
    # Tanh and Dropout don't have weights, so the weight-bearing indices are:
    # 0=Conv, 1=BN, 4=Conv, 5=BN, 8=Conv, 9=BN, 12=Conv, 13=BN, 16=Conv, 17=BN
    # But that pattern depends on whether the model keeps Tanh/Dropout as separate modules.

    # Let's figure this out from the actual indices we see.
    seen_indices = sorted(set(mod_idx for mod_idx, _, _ in sorted_entries))

    # Pair them: each consecutive pair of indices is (conv, bn)
    conv_bn_pairs = []
    pair = []
    for idx in seen_indices:
        pair.append(idx)
        if len(pair) == 2:
            conv_bn_pairs.append(tuple(pair))
            pair = []
    if pair:
        conv_bn_pairs.append(tuple(pair))

    # Map module index -> (type, layer_index)
    idx_map = {}
    for li, pair in enumerate(conv_bn_pairs):
        idx_map[pair[0]] = ("conv", li)
        if len(pair) > 1:
            idx_map[pair[1]] = ("bn", li)

    # Second pass: rename
    for mod_idx, rest, tensor in sorted_entries:
        if mod_idx not in idx_map:
            continue
        type_name, li = idx_map[mod_idx]
        # Map BN running_mean/running_var
        rest = rest.replace("running_mean", "mean")
        rest = rest.replace("running_var", "var")
        new_key = f"postnet.{type_name}.{li}.{rest}"
        result[new_key] = tensor

    return result


def rename_vocoder_weights(state_dict: dict) -> dict:
    """Rename HiFi-GAN vocoder weights to voc.* namespace.

    Generator-only: skip discriminator weights.
    BananaMind stores vocoder as a separate safetensors with clean names:
      conv_pre.weight -> voc.conv_pre.weight
      ups.{i}.weight  -> voc.ups.{i}.weight
      resblocks.{i}.convs1.{j}.weight -> voc.resblocks.{i}.convs1.{j}.weight
      conv_post.weight -> voc.conv_post.weight
    """
    renamed = {}
    for key, tensor in state_dict.items():
        renamed["voc." + key] = tensor
    return renamed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Convert BananaMind-TTS-V2.1 to GGUF")
    parser.add_argument("--model", required=True,
                        help="HuggingFace model ID or local path")
    parser.add_argument("--locale", required=True,
                        choices=["en-us", "de-de"],
                        help="Locale to convert")
    parser.add_argument("--output", required=True,
                        help="Output .gguf path")
    parser.add_argument("--ftype", default="f16",
                        choices=["f16", "f32"],
                        help="Weight storage type")
    add_speaker_identity_arg(parser)
    args = parser.parse_args()

    # ── Resolve model files ──
    model_dir = Path(args.model)
    if not model_dir.exists():
        from huggingface_hub import snapshot_download
        print(f"Downloading {args.model} from HuggingFace...")
        cache_dir = os.environ.get("HF_HOME",
                                    "/mnt/akademie_storage/huggingface/hub")
        model_dir = Path(snapshot_download(args.model, cache_dir=cache_dir))
        print(f"  -> {model_dir}")

    locale_dir = model_dir / args.locale
    if not locale_dir.exists():
        sys.exit(f"Locale directory not found: {locale_dir}")

    # ── Load config ──
    config_path = locale_dir / "config.json"
    with open(config_path, encoding="utf-8") as f:
        config = json.load(f)

    banana_config = config.get("banana_config", {})
    model_config = banana_config.get("model", {})
    audio_config = banana_config.get("audio", {})
    tokenizer_config = config.get("tokenizer", {})
    vocoder_config = config.get("vocoder", {})
    voc_arch_config = vocoder_config.get("config", banana_config.get("vocoder", {}))

    print(f"Locale: {args.locale}")
    print(f"  hidden_size: {model_config.get('hidden_size', 256)}")
    print(f"  decoder_dim: {model_config.get('decoder_dim', 512)}")
    print(f"  n_mels: {audio_config.get('n_mels', 80)}")
    print(f"  reduction_factor: {model_config.get('reduction_factor', 4)}")
    print(f"  sample_rate: {audio_config.get('sample_rate', 22050)}")

    # Helper: load safetensors via torch, convert to numpy float32
    def load_safetensors(path):
        sd = load_safetensors_torch(str(path), device="cpu")
        return {k: v.float().numpy() for k, v in sd.items()}

    # ── Load acoustic model ──
    acoustic_path = locale_dir / "model.safetensors"
    print(f"\nLoading acoustic model: {acoustic_path}")
    acoustic_sd = load_safetensors(acoustic_path)
    print(f"  {len(acoustic_sd)} tensors loaded")

    # Print tensor names for debugging
    for name, t in sorted(acoustic_sd.items()):
        print(f"    {name}: {t.shape} {t.dtype}")

    # ── Load vocoder ──
    vocoder_path = locale_dir / "vocoder.safetensors"
    print(f"\nLoading vocoder: {vocoder_path}")
    vocoder_sd = load_safetensors(vocoder_path)
    print(f"  {len(vocoder_sd)} tensors loaded")

    for name, t in sorted(vocoder_sd.items()):
        print(f"    {name}: {t.shape} {t.dtype}")

    # ── Rename weights ──
    print("\nRenaming acoustic weights...")
    acoustic_tensors = rename_acoustic_weights(acoustic_sd)
    acoustic_tensors = reorganize_postnet(acoustic_tensors)

    print("\nRenaming vocoder weights...")
    vocoder_tensors = rename_vocoder_weights(vocoder_sd)

    # Merge
    all_tensors = {}
    all_tensors.update(acoustic_tensors)
    all_tensors.update(vocoder_tensors)

    print(f"\nFinal tensor names:")
    for name in sorted(all_tensors.keys()):
        t = all_tensors[name]
        print(f"  {name}: {t.shape} {t.dtype}")

    # ── Write GGUF ──
    ftype_map = {
        "f16": GGMLQuantizationType.F16,
        "f32": GGMLQuantizationType.F32,
    }
    target_ftype = ftype_map[args.ftype]

    print(f"\nWriting GGUF: {args.output}")
    writer = GGUFWriter(str(args.output), arch="bananamind_tts")
    stamp_speaker_identity(writer, args)

    # ── KV metadata ──

    # Model architecture
    hidden = model_config.get("hidden_size", 256)
    decoder_dim = model_config.get("decoder_dim", 512)
    attention_dim = model_config.get("attention_dim", 128)
    n_mels = audio_config.get("n_mels", 80)
    reduction_factor = model_config.get("reduction_factor", 4)

    writer.add_uint32("bananamind_tts.hidden_size", hidden)
    writer.add_uint32("bananamind_tts.decoder_dim", decoder_dim)
    writer.add_uint32("bananamind_tts.attention_dim", attention_dim)
    writer.add_uint32("bananamind_tts.n_mels", n_mels)
    writer.add_uint32("bananamind_tts.reduction_factor", reduction_factor)

    # Encoder
    writer.add_uint32("bananamind_tts.encoder_conv_layers",
                      model_config.get("encoder_conv_layers", 3))
    prenet_sizes = model_config.get("prenet_sizes", [256, 128])
    writer.add_array("bananamind_tts.prenet_sizes",
                     [int(x) for x in prenet_sizes])

    # Attention
    writer.add_uint32("bananamind_tts.location_channels",
                      model_config.get("location_channels", 32))
    writer.add_uint32("bananamind_tts.location_kernel_size",
                      model_config.get("location_kernel_size", 31))

    # Postnet
    writer.add_uint32("bananamind_tts.postnet_channels",
                      model_config.get("postnet_channels", 512))
    writer.add_uint32("bananamind_tts.postnet_layers",
                      model_config.get("postnet_layers", 5))

    # Audio / inference
    sample_rate = audio_config.get("sample_rate", 22050)
    writer.add_uint32("bananamind_tts.sample_rate", sample_rate)
    writer.add_uint32("bananamind_tts.max_decoder_steps",
                      model_config.get("max_decoder_steps", 1200))

    # Stop threshold stored as integer (x1000) to avoid float KV issues
    stop_threshold = model_config.get("stop_threshold", 0.55)
    writer.add_float32("bananamind_tts.stop_threshold", stop_threshold)
    writer.add_uint32("bananamind_tts.attention_window",
                      model_config.get("attention_window", 12))

    # Mel normalization stats
    mel_stats = audio_config.get("mel_stats", {})
    if mel_stats.get("normalized_training", False):
        writer.add_float32("bananamind_tts.mel_mean",
                           float(mel_stats["mean"]))
        writer.add_float32("bananamind_tts.mel_std",
                           float(mel_stats["std"]))
        writer.add_float32("bananamind_tts.mel_min",
                           float(mel_stats.get("min", -12.0)))
        writer.add_float32("bananamind_tts.mel_max",
                           float(mel_stats.get("max", 8.0)))

    # Tokenizer
    symbols = tokenizer_config.get("symbols", [])
    if symbols:
        writer.add_array("tokenizer.ggml.tokens", symbols)
        writer.add_uint32("bananamind_tts.vocab_size", len(symbols))
        # Store ampersand replacement
        ampersand = tokenizer_config.get("ampersand_replacement", " and ")
        writer.add_string("bananamind_tts.ampersand_replacement", ampersand)

    # Language
    writer.add_string("bananamind_tts.language", args.locale)

    # Vocoder hparams
    voc_rates = voc_arch_config.get("upsample_rates", [8, 8, 2, 2])
    voc_kernels = voc_arch_config.get("upsample_kernel_sizes", [16, 16, 4, 4])
    voc_rb_kernels = voc_arch_config.get("resblock_kernel_sizes", [3, 7, 11])
    voc_rb_dilations = voc_arch_config.get("resblock_dilation_sizes",
                                            [[1, 3, 5], [1, 3, 5], [1, 3, 5]])

    writer.add_uint32("bananamind_tts.voc_initial_channels",
                      voc_arch_config.get("initial_channels", 256))
    writer.add_array("bananamind_tts.voc_upsample_rates",
                     [int(x) for x in voc_rates])
    writer.add_array("bananamind_tts.voc_upsample_kernel_sizes",
                     [int(x) for x in voc_kernels])
    writer.add_array("bananamind_tts.voc_resblock_kernel_sizes",
                     [int(x) for x in voc_rb_kernels])

    # Flatten dilations
    n_dilations = len(voc_rb_dilations[0]) if voc_rb_dilations else 3
    flat_dilations = []
    for d in voc_rb_dilations:
        flat_dilations.extend([int(x) for x in d])
    writer.add_array("bananamind_tts.voc_resblock_dilations", flat_dilations)
    writer.add_uint32("bananamind_tts.voc_n_dilations", n_dilations)

    voc_slope = voc_arch_config.get("leaky_relu_slope", 0.1)
    writer.add_float32("bananamind_tts.voc_leaky_relu_slope", voc_slope)

    # ── Write tensors ──
    n_tensors = 0
    n_params = 0

    for name, tensor in sorted(all_tensors.items()):
        arr = np.array(tensor, dtype=np.float32)

        # Choose quantization
        if target_ftype == GGMLQuantizationType.F16 and arr.ndim >= 2:
            qt = GGMLQuantizationType.F16
        else:
            qt = GGMLQuantizationType.F32

        # 1D biases, norms, and BN params always F32
        if arr.ndim <= 1 or "norm" in name or name.endswith(".bias") \
                or "_bn" in name or ".bn_" in name or ".mean" in name \
                or ".var" in name:
            qt = GGMLQuantizationType.F32

        # add_tensor() with an explicit raw_dtype does NOT convert — it labels
        # the array's bytes with the type you name. Passing an f32 array as F16
        # writes f32 bytes tagged F16, and a reader takes the first half of them
        # and reinterprets each 4-byte float as two 2-byte ones. See
        # tests/test_gguf_dtype_cast.py; this shipped for real in
        # cstr/fastpitch-en-GGUF's f16 build (943,872 NaNs).
        #
        # Latent here rather than shipped: every published bananamind file is
        # f32 or q8_0, so --ftype f16 was never exercised.
        if qt == GGMLQuantizationType.F16:
            arr = arr.astype(np.float16)
        elif qt == GGMLQuantizationType.F32:
            arr = arr.astype(np.float32)

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
