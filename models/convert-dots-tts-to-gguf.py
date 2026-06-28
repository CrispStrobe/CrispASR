#!/usr/bin/env python3
"""
Convert rednote-hilab/dots.tts-{soar,base,mf} safetensors → GGUF for the
CrispASR `dots-tts` backend.

dots.tts is a 2B-parameter continuous AR TTS:
  - LLM backbone: Qwen2.5-1.5B (28L, 12Q/2KV heads, hidden=1536)
  - PatchEncoder: 24L causal transformer (in_dim=128 → hidden=1024)
  - DiT flow-matching head: 18L AdaLN DiT (hidden=1024)
  - Projection layers: hidden_proj, latent_proj, coordinate_proj, xvec_proj, eos_proj
  - Speaker encoder (CAM++): separate file → dots-tts-spk.gguf
  - Vocoder (BigVGAN): separate file → dots-tts-vocoder.gguf

Produces THREE GGUF files:
  1. dots-tts-soar.gguf       — LLM + DiT + PatchEncoder + projections + tokenizer
  2. dots-tts-vocoder.gguf    — BigVGAN encoder + decoder
  3. dots-tts-spk.gguf        — CAM++ speaker encoder

Usage:
    python models/convert-dots-tts-to-gguf.py \\
        --model-dir /mnt/storage/dots-tts-soar \\
        --output-dir /mnt/storage/gguf-models

    # Or from HuggingFace:
    python models/convert-dots-tts-to-gguf.py \\
        --hf-repo rednote-hilab/dots.tts-soar \\
        --output-dir /mnt/storage/gguf-models
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

try:
    from gguf import GGUFWriter, GGMLQuantizationType
except ImportError:
    sys.exit("pip install gguf")

try:
    from safetensors import safe_open
except ImportError:
    sys.exit("pip install safetensors")

try:
    import torch
except ImportError:
    sys.exit("pip install torch")


# ── Helpers ──────────────────────────────────────────────────────────

def to_f16(t: torch.Tensor) -> np.ndarray:
    return t.detach().to(torch.float16).numpy()

def to_f32(t: torch.Tensor) -> np.ndarray:
    return t.detach().to(torch.float32).numpy()

try:
    from gguf import quants as gguf_quants
    _HAS_GGUF_QUANTS = True
except ImportError:
    _HAS_GGUF_QUANTS = False

def _quantize_to(t: torch.Tensor, qtype: GGMLQuantizationType):
    data_f32 = t.detach().to(torch.float32).numpy()
    if _HAS_GGUF_QUANTS:
        try:
            return gguf_quants.quantize(data_f32, qtype), qtype
        except Exception:
            return to_f16(t), GGMLQuantizationType.F16
    else:
        return to_f16(t), GGMLQuantizationType.F16


def choose_dtype(name: str, shape: list, t: torch.Tensor, quant: str = "f16"):
    """Choose tensor precision based on name and size."""
    n = int(np.prod(shape))

    # Small tensors, norms, biases → always F32
    if t.ndim <= 1 or n < 256:
        return to_f32(t), GGMLQuantizationType.F32

    # Conditioning / critical pathway → always F32
    always_f32 = (
        'norm' in name or
        'bias' in name or
        'embed' in name.split('.')[-1] or
        'adaln' in name or
        'time_' in name or
        'eos_proj' in name or
        'xvec_proj' in name or
        'latent_stats' in name or
        'inv_freq' in name or
        'layer_scale' in name
    )
    if always_f32:
        return to_f32(t), GGMLQuantizationType.F32

    # Bulk weight matrices
    if quant == "q8_0":
        return _quantize_to(t, GGMLQuantizationType.Q8_0)
    elif quant == "q4_k":
        return _quantize_to(t, GGMLQuantizationType.Q4_K)
    else:
        return to_f16(t), GGMLQuantizationType.F16


def add_tensor(writer, name, t, quant_level):
    shape = list(t.shape)
    data, dtype = choose_dtype(name, shape, t, quant=quant_level)
    if dtype not in (GGMLQuantizationType.F32, GGMLQuantizationType.F16):
        writer.add_tensor(name, data, raw_dtype=dtype)
    else:
        writer.add_tensor(name, data)


# ── Tensor name remapping ────────────────────────────────────────────

def map_core_name(hf_name: str) -> str | None:
    """Map model.safetensors key → GGUF tensor name for the core model.

    Actual safetensors keys (NO 'core.' prefix):
      llm.model.layers.0.self_attn.q_proj.weight
      llm.model.layers.0.self_attn.q_proj.bias
      llm.model.norm.weight
      patch_encoder.encoder.layers.0.attn.q_proj.weight
      patch_encoder.encoder.layers.0.ffn.fc1.weight
      patch_encoder.in_proj.weight
      patch_encoder.ds_proj.weight
      velocity_field_predictor.blocks.0.adaLN_modulation.1.weight
      velocity_field_predictor.blocks.0.attn.q_proj.weight
      velocity_field_predictor.blocks.0.ffn.fc1.weight
      velocity_field_predictor.time_embedder.mlp.0.weight
      velocity_field_predictor.input_layer.weight
      velocity_field_predictor.output_layer.linear.weight
      velocity_field_predictor.output_layer.adaLN_modulation.1.weight
      hidden_proj.weight
      latent_proj.weight
      coordinate_proj.weight
      xvec_proj.0.weight  xvec_proj.1.weight (BatchNorm)
      eos_proj.0.weight   eos_proj.2.weight
    """
    n = hf_name

    # ── LLM (Qwen2.5-1.5B) ──
    if n.startswith("llm."):
        n = n.replace("llm.model.", "llm.")
        n = n.replace("llm.lm_head.", "llm.lm_head.")
        n = n.replace(".self_attn.", ".")
        n = n.replace(".mlp.gate_proj.", ".gate.")
        n = n.replace(".mlp.up_proj.", ".up.")
        n = n.replace(".mlp.down_proj.", ".down.")
        n = n.replace(".input_layernorm.", ".attn_norm.")
        n = n.replace(".post_attention_layernorm.", ".ffn_norm.")
        n = n.replace("llm.embed_tokens.", "llm.tok_emb.")
        return "dots." + n

    # ── PatchEncoder (VAESemanticEncoder) ──
    if n.startswith("patch_encoder."):
        n = n.replace("patch_encoder.", "penc.")
        n = n.replace("penc.encoder.layers.", "penc.layers.")
        n = n.replace("penc.encoder.norm.", "penc.final_norm.")
        n = n.replace(".attn.", ".")
        # FFN is 2-layer (fc1/fc2), NOT SwiGLU
        n = n.replace(".ffn.fc1.", ".ffn_up.")
        n = n.replace(".ffn.fc2.", ".ffn_down.")
        # ds_proj is the downsample projection
        n = n.replace("penc.ds_proj.", "penc.ds_conv.")
        n = n.replace("penc.in_proj.", "penc.in_proj.")
        n = n.replace("penc.out_proj.", "penc.out_proj.")
        return "dots." + n

    # ── DiT (velocity_field_predictor) ──
    if n.startswith("velocity_field_predictor."):
        n = n.replace("velocity_field_predictor.", "dit.")
        n = n.replace("dit.blocks.", "dit.blk.")
        # AdaLN modulation: .adaLN_modulation.1. → .adaln.
        n = n.replace(".adaLN_modulation.1.", ".adaln.")
        n = n.replace(".attn.", ".")
        # FFN is 2-layer (fc1/fc2)
        n = n.replace(".ffn.fc1.", ".ffn_up.")
        n = n.replace(".ffn.fc2.", ".ffn_down.")
        # Timestep embedder
        n = n.replace("dit.time_embedder.", "dit.time_emb.")
        # Input/output layers
        n = n.replace("dit.input_layer.", "dit.in_proj.")
        n = n.replace("dit.output_layer.linear.", "dit.final_proj.")
        n = n.replace("dit.output_layer.adaLN_modulation.1.", "dit.final_adaln.")
        n = n.replace("dit.output_layer.norm.", "dit.final_norm.")
        return "dots." + n

    # ── Projection layers (no prefix in safetensors) ──
    if n.startswith("hidden_proj."):
        return "dots." + n
    if n.startswith("latent_proj."):
        return "dots." + n
    if n.startswith("coordinate_proj."):
        return "dots." + n
    if n.startswith("xvec_proj."):
        return "dots." + n
    if n.startswith("eos_proj."):
        return "dots." + n

    # Skip unknown
    return None


def map_vocoder_name(hf_name: str) -> str | None:
    """Map vocoder.safetensors key → GGUF tensor name."""
    if not hf_name.startswith("vocoder."):
        # Also handle keys without the vocoder. prefix (standalone file)
        n = hf_name
    else:
        n = hf_name.replace("vocoder.", "")

    # Skip encoder weights (we only need decoder for inference)
    # Actually, we need the encoder for voice cloning (ref audio → latents)
    # Keep both encoder and decoder

    # Prefix everything with voc.
    return "dots.voc." + n


def map_speaker_name(hf_name: str) -> str | None:
    """Map speaker_encoder.safetensors key → GGUF tensor name."""
    if hf_name.startswith("xvector_extractor."):
        n = hf_name.replace("xvector_extractor.", "")
    else:
        n = hf_name
    return "dots.spk." + n


# ── Model resolution ──────────────────────────────────────────────────

def resolve_model_dir(args) -> Path:
    """Resolve model directory from args, downloading from HF if needed."""
    if args.model_dir:
        d = Path(args.model_dir)
        assert d.exists(), f"Model dir not found: {d}"
        return d

    if args.hf_repo:
        from huggingface_hub import snapshot_download
        cache_dir = os.environ.get("HF_HOME",
                    os.environ.get("HF_HUB_CACHE",
                    "/mnt/akademie_storage/huggingface/hub"))
        d = Path(snapshot_download(args.hf_repo, cache_dir=cache_dir))
        print(f"Downloaded to: {d}")
        return d

    sys.exit("--model-dir or --hf-repo required")


# ── Main ──────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Convert dots.tts → GGUF")
    p.add_argument("--model-dir", type=Path, default=None,
                   help="Local directory with dots.tts files")
    p.add_argument("--hf-repo", type=str, default=None,
                   help="HuggingFace repo ID (e.g. rednote-hilab/dots.tts-soar)")
    p.add_argument("--output-dir", type=Path, required=True,
                   help="Output directory for GGUF files")
    p.add_argument("--quant", choices=["f16", "q8_0", "q4_k"], default="f16",
                   help="Quantization level for bulk weights")
    p.add_argument("--name", type=str, default="dots-tts-soar",
                   help="Base name for output files")
    p.add_argument("--skip-vocoder", action="store_true",
                   help="Skip vocoder conversion")
    p.add_argument("--skip-speaker", action="store_true",
                   help="Skip speaker encoder conversion")
    args = p.parse_args()

    model_dir = resolve_model_dir(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    quant = args.quant

    # ── Load config ──
    config_path = model_dir / "config.json"
    assert config_path.exists(), f"config.json not found in {model_dir}"
    with open(config_path) as f:
        config = json.load(f)

    llm_config_path = model_dir / "llm_config.json"
    assert llm_config_path.exists(), f"llm_config.json not found in {model_dir}"
    with open(llm_config_path) as f:
        llm_config = json.load(f)

    print(f"Model: dots.tts")
    print(f"  LLM: Qwen2.5 {llm_config['num_hidden_layers']}L, "
          f"hidden={llm_config['hidden_size']}, "
          f"heads={llm_config['num_attention_heads']}Q/{llm_config['num_key_value_heads']}KV")
    print(f"  PatchEncoder: {config['PatchEncoder']['num_layers']}L, "
          f"hidden={config['PatchEncoder']['hidden_size']}")
    print(f"  DiT: {config['DiT']['num_layers']}L, "
          f"hidden={config['DiT']['hidden_size']}")
    print(f"  Vocoder: {config['vocoder']['sample_rate']} Hz, "
          f"upsample={config['vocoder']['upsample_rates']}")

    # ════════════════════════════════════════════════════════════════════
    # 1. Core model GGUF (LLM + DiT + PatchEncoder + projections + tokenizer)
    # ════════════════════════════════════════════════════════════════════
    core_path = args.output_dir / f"{args.name}-{quant}.gguf"
    print(f"\n=== Core model → {core_path} ===")

    model_st = model_dir / "model.safetensors"
    assert model_st.exists(), f"model.safetensors not found: {model_st}"

    w = GGUFWriter(str(core_path), arch="dots-tts")

    # ── Hyperparameters ──
    w.add_string("dots.arch", "dots-tts")
    # LLM
    w.add_int32("dots.llm.n_layers", llm_config["num_hidden_layers"])
    w.add_int32("dots.llm.hidden_size", llm_config["hidden_size"])
    w.add_int32("dots.llm.intermediate_size", llm_config["intermediate_size"])
    w.add_int32("dots.llm.n_heads", llm_config["num_attention_heads"])
    w.add_int32("dots.llm.n_kv_heads", llm_config["num_key_value_heads"])
    w.add_int32("dots.llm.vocab_size", llm_config["vocab_size"])
    w.add_float32("dots.llm.rope_theta", llm_config["rope_theta"])
    w.add_float32("dots.llm.rms_norm_eps", llm_config["rms_norm_eps"])
    # PatchEncoder
    w.add_int32("dots.penc.n_layers", config["PatchEncoder"]["num_layers"])
    w.add_int32("dots.penc.hidden_size", config["PatchEncoder"]["hidden_size"])
    w.add_int32("dots.penc.ffn_hidden_size", config["PatchEncoder"]["ffn_hidden_size"])
    w.add_int32("dots.penc.n_heads", config["PatchEncoder"]["num_heads"])
    w.add_int32("dots.penc.input_dim", config["PatchEncoder"]["input_dim"])
    w.add_float32("dots.penc.rope_theta", config["PatchEncoder"]["rotary_theta"])
    # DiT
    w.add_int32("dots.dit.n_layers", config["DiT"]["num_layers"])
    w.add_int32("dots.dit.hidden_size", config["DiT"]["hidden_size"])
    w.add_int32("dots.dit.ffn_hidden_size", config["DiT"]["ffn_hidden_size"])
    w.add_int32("dots.dit.n_heads", config["DiT"]["num_heads"])
    w.add_float32("dots.dit.rope_theta", config["DiT"]["rotary_theta"])
    # Model
    w.add_int32("dots.latent_dim", config["latent_dim"])
    w.add_int32("dots.patch_size", config["patch_size"])
    w.add_float32("dots.cfg_droprate", config["cfg_droprate"])
    w.add_int32("dots.spk_dim", config["campplus_embedding_size"])
    w.add_float32("dots.fm_sigma", config.get("fm_sigma", 0.0))

    # ── Tokenizer ──
    tokenizer_path = model_dir / "tokenizer.json"
    if tokenizer_path.exists():
        with open(tokenizer_path, "r", encoding="utf-8") as tf:
            tok_data = json.load(tf)
        # Extract vocab (token → id mapping)
        vocab = tok_data.get("model", {}).get("vocab", {})
        if vocab:
            # Sort by id to get token list
            sorted_tokens = sorted(vocab.items(), key=lambda x: x[1])
            token_strings = [t[0] for t in sorted_tokens]
            w.add_array("dots.tokenizer.tokens", token_strings)
            print(f"  Tokenizer: {len(token_strings)} tokens")
        # Extract merges
        merges = tok_data.get("model", {}).get("merges", [])
        if merges:
            w.add_array("dots.tokenizer.merges", merges)
            print(f"  Merges: {len(merges)}")

    # ── Special tokens ──
    special_tokens_path = model_dir / "added_tokens.json"
    if special_tokens_path.exists():
        with open(special_tokens_path) as f:
            added_tokens = json.load(f)
        # Find audio_gen_span and other special tokens
        for tok_name, tok_id in added_tokens.items():
            if "audio_gen_span" in tok_name:
                w.add_int32("dots.token.audio_gen_span", tok_id)
            elif "audio_comp_span" in tok_name:
                w.add_int32("dots.token.audio_comp_span", tok_id)
            elif "text_cond_end" in tok_name:
                w.add_int32("dots.token.text_cond_end", tok_id)
        print(f"  Special tokens: {len(added_tokens)}")

    # ── Latent stats (mean/std for denormalization) ──
    latent_stats_path = model_dir / "latent_stats.pt"
    if latent_stats_path.exists():
        stats = torch.load(str(latent_stats_path), map_location="cpu",
                          weights_only=False)
        if isinstance(stats, dict):
            for k, v in stats.items():
                t = v if isinstance(v, torch.Tensor) else torch.tensor(v)
                w.add_tensor(f"dots.latent_stats.{k}", to_f32(t))
                print(f"  latent_stats.{k}: {list(t.shape)}")

    # ── Core tensors ──
    n_core = 0
    with safe_open(str(model_st), framework='pt') as f:
        keys = sorted(f.keys())
        for k in keys:
            gguf_name = map_core_name(k)
            if gguf_name is None:
                print(f"  SKIP: {k}")
                continue
            t = f.get_tensor(k)
            add_tensor(w, gguf_name, t, quant)
            n_core += 1
    print(f"  Core tensors: {n_core}")

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    fsize = core_path.stat().st_size
    print(f"  → {core_path} ({fsize/1024/1024:.1f} MiB)")

    # ════════════════════════════════════════════════════════════════════
    # 2. Vocoder GGUF (BigVGAN encoder + decoder)
    # ════════════════════════════════════════════════════════════════════
    if not args.skip_vocoder:
        voc_st = model_dir / "vocoder.safetensors"
        if voc_st.exists():
            voc_path = args.output_dir / f"{args.name}-vocoder-{quant}.gguf"
            print(f"\n=== Vocoder → {voc_path} ===")

            wv = GGUFWriter(str(voc_path), arch="dots-tts-vocoder")
            wv.add_string("dots.arch", "dots-tts-vocoder")
            # Vocoder config
            vc = config["vocoder"]
            wv.add_int32("dots.voc.sample_rate", vc["sample_rate"])
            wv.add_int32("dots.voc.latent_dim", vc["latent_dim"])
            wv.add_int32("dots.voc.upsample_initial_channel", vc["upsample_initial_channel"])
            wv.add_int32("dots.voc.mi_num_layers", vc["mi_num_layers"])
            wv.add_array("dots.voc.upsample_rates",
                        [int(x) for x in vc["upsample_rates"]])
            wv.add_array("dots.voc.upsample_kernel_sizes",
                        [int(x) for x in vc["upsample_kernel_sizes"]])
            wv.add_array("dots.voc.downsample_rates",
                        [int(x) for x in vc["downsample_rates"]])
            wv.add_array("dots.voc.downsample_channels",
                        [int(x) for x in vc["downsample_channels"]])
            wv.add_array("dots.voc.resblock_kernel_sizes",
                        [int(x) for x in vc["resblock_kernel_sizes"]])
            # Flatten dilation sizes
            flat_dilations = []
            for d in vc["resblock_dilation_sizes"]:
                flat_dilations.extend([int(x) for x in d])
            wv.add_array("dots.voc.resblock_dilation_sizes", flat_dilations)
            wv.add_string("dots.voc.activation", vc["activation"])
            wv.add_bool("dots.voc.causal", vc.get("causal", True))

            n_voc = 0
            with safe_open(str(voc_st), framework='pt') as f:
                keys = sorted(f.keys())
                for k in keys:
                    gguf_name = map_vocoder_name(k)
                    if gguf_name is None:
                        continue
                    t = f.get_tensor(k)
                    add_tensor(wv, gguf_name, t, quant)
                    n_voc += 1
            print(f"  Vocoder tensors: {n_voc}")

            wv.write_header_to_file()
            wv.write_kv_data_to_file()
            wv.write_tensors_to_file()
            wv.close()
            fsize = voc_path.stat().st_size
            print(f"  → {voc_path} ({fsize/1024/1024:.1f} MiB)")
        else:
            print(f"\n  WARNING: vocoder.safetensors not found, skipping vocoder")

    # ════════════════════════════════════════════════════════════════════
    # 3. Speaker encoder GGUF (CAM++)
    # ════════════════════════════════════════════════════════════════════
    if not args.skip_speaker:
        spk_st = model_dir / "speaker_encoder.safetensors"
        if spk_st.exists():
            spk_path = args.output_dir / f"{args.name}-spk-{quant}.gguf"
            print(f"\n=== Speaker encoder → {spk_path} ===")

            ws = GGUFWriter(str(spk_path), arch="dots-tts-spk")
            ws.add_string("dots.arch", "dots-tts-spk")
            ws.add_int32("dots.spk.embedding_size", config["campplus_embedding_size"])

            n_spk = 0
            with safe_open(str(spk_st), framework='pt') as f:
                keys = sorted(f.keys())
                for k in keys:
                    gguf_name = map_speaker_name(k)
                    if gguf_name is None:
                        continue
                    t = f.get_tensor(k)
                    add_tensor(ws, gguf_name, t, quant)
                    n_spk += 1
            print(f"  Speaker tensors: {n_spk}")

            ws.write_header_to_file()
            ws.write_kv_data_to_file()
            ws.write_tensors_to_file()
            ws.close()
            fsize = spk_path.stat().st_size
            print(f"  → {spk_path} ({fsize/1024/1024:.1f} MiB)")
        else:
            print(f"\n  WARNING: speaker_encoder.safetensors not found, skipping speaker")

    print("\nDone!")


if __name__ == "__main__":
    main()
