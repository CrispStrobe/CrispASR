#!/usr/bin/env python3
"""VibeVoice-1.5B TTS reference dump.

Runs the full VibeVoice TTS pipeline (text -> Qwen2 LM -> DDIM diffusion ->
sigma-VAE decoder -> 24kHz audio) and dumps intermediate tensors to binary
files for comparison with the C++ ggml implementation.

Dumped stages (all little-endian):

  text_token_ids.bin          int32   [T_text]          tokenized text IDs
  lm_hidden_states.bin        float32 [T_text, d_lm]    LM hidden states (all tokens)
  lm_last_hidden.bin          float32 [d_lm]            last token's hidden state
  pred_t_emb_step0.bin        float32 [d_lm]            timestep embedding at step 0
  pred_c_step0.bin            float32 [d_lm]            combined c = cond + t_emb at step 0
  pred_eps_step0.bin          float32 [vae_dim]          prediction head output at step 0
  latent_after_diffusion.bin  float32 [T_frames, vae_dim]  final denoised latent
  scaled_latent.bin           float32 [T_frames, vae_dim]  after scaling
  audio_output.bin            float32 [N_samples]        raw audio PCM
  audio_output.wav            24kHz mono WAV

Key architecture facts (1.5B TTS model):
  - Qwen2-1.5B LM: d_lm=1536, 28 layers, vocab=151936
  - Prediction head: 4x AdaLN + SwiGLU layers, vae_dim=64
  - Timestep embedding: sinusoidal[256] -> MLP(256->1536->1536)
  - DDIM: cosine beta schedule, v-prediction, 20 steps
  - sigma-VAE decoder: transposed ConvNeXt, 3200x upsample -> 24kHz
  - Latent scaling: z_scaled = z / scaling_factor - bias_factor

Usage:
  python tools/reference_backends/vibevoice_tts.py \\
      --text "Hello world" \\
      --output-dir /tmp/vibevoice_tts_ref \\
      --model microsoft/VibeVoice-1.5B \\
      --num-steps 20 --seed 42
"""

from __future__ import annotations

import argparse
import json
import math
import os
import struct
import wave
from pathlib import Path

import numpy as np


# ── Cosine DDIM schedule (matches C++ make_ddim_schedule) ───────────────────

def make_ddim_schedule(num_inference_steps: int, num_train_steps: int = 1000):
    """Cosine beta schedule: alphas_cumprod[t] = cos^2((t/T + s)/(1+s) * pi/2)."""
    offset = 0.008

    def alpha_bar(t):
        frac = (t / num_train_steps + offset) / (1.0 + offset)
        val = math.cos(frac * math.pi * 0.5)
        return val * val

    alphas_cumprod = np.empty(num_train_steps, dtype=np.float64)
    a0 = alpha_bar(0)
    for i in range(num_train_steps):
        alphas_cumprod[i] = alpha_bar(i + 1) / a0
        alphas_cumprod[i] = np.clip(alphas_cumprod[i], 0.0001, 0.9999)

    # Linearly spaced timesteps from T-1 down to 0
    timesteps = np.round(
        np.linspace(num_train_steps - 1, 0, num_inference_steps)
    ).astype(np.int64)
    timesteps[-1] = 0

    return alphas_cumprod.astype(np.float32), timesteps


def ddim_step_v(alphas_cumprod, timesteps, step_idx, x, v_pred):
    """DDIM step with v-prediction (matches C++ ddim_step)."""
    t = int(timesteps[step_idx])
    t_prev = int(timesteps[step_idx + 1]) if step_idx + 1 < len(timesteps) else -1

    alpha_t = float(alphas_cumprod[t])
    alpha_prev = float(alphas_cumprod[t_prev]) if t_prev >= 0 else 1.0

    sqrt_alpha_t = math.sqrt(alpha_t)
    sqrt_1ma_t = math.sqrt(1.0 - alpha_t)
    sqrt_alpha_prev = math.sqrt(alpha_prev)
    sqrt_1ma_prev = math.sqrt(1.0 - alpha_prev)

    # v = sqrt(alpha_t) * eps - sqrt(1-alpha_t) * x0
    # => x0 = sqrt(alpha_t) * x_t - sqrt(1-alpha_t) * v
    # => eps = sqrt(1-alpha_t) * x_t + sqrt(alpha_t) * v
    x0_pred = sqrt_alpha_t * x - sqrt_1ma_t * v_pred
    eps_pred = sqrt_1ma_t * x + sqrt_alpha_t * v_pred
    return sqrt_alpha_prev * x0_pred + sqrt_1ma_prev * eps_pred


# ── Sinusoidal timestep embedding (matches C++ compute_sinusoidal_embed) ────

def sinusoidal_embed(t: float, dim: int = 256) -> np.ndarray:
    """Sinusoidal embedding: [cos(t*freq_0)...cos(t*freq_{d/2-1}), sin(...)].

    Matches the C++ implementation: cos first half, sin second half.
    """
    half = dim // 2
    freqs = np.exp(-np.log(10000.0) * np.arange(half, dtype=np.float32) / half)
    args = t * freqs
    out = np.empty(dim, dtype=np.float32)
    out[:half] = np.cos(args)
    out[half:] = np.sin(args)
    return out


# ── Binary dump helpers ─────────────────────────────────────────────────────

def dump_f32(path: Path, arr: np.ndarray):
    arr.astype(np.float32).tofile(str(path))
    print(f"  wrote {path.name}: {arr.shape} float32 ({arr.nbytes} bytes)")


def dump_i32(path: Path, arr: np.ndarray):
    arr.astype(np.int32).tofile(str(path))
    print(f"  wrote {path.name}: {arr.shape} int32 ({arr.nbytes} bytes)")


def dump_wav(path: Path, audio: np.ndarray, sr: int = 24000):
    audio_i16 = np.clip(audio * 32767.0, -32768, 32767).astype(np.int16)
    with wave.open(str(path), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(audio_i16.tobytes())
    print(f"  wrote {path.name}: {len(audio)} samples, {len(audio)/sr:.2f}s @ {sr}Hz")


# ── Main pipeline ───────────────────────────────────────────────────────────

def run_tts_pipeline(
    model_dir: str,
    text: str,
    output_dir: str,
    num_steps: int = 20,
    seed: int = 42,
    n_frames: int | None = None,
):
    import torch
    from safetensors.torch import load_file
    from transformers import AutoTokenizer

    model_path = Path(model_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # ── 1. Load config ──────────────────────────────────────────────────────
    cfg_path = model_path / "config.json"
    with open(cfg_path) as f:
        cfg = json.load(f)

    dec_cfg = cfg.get("decoder_config", {})
    at_cfg = cfg.get("acoustic_tokenizer_config", cfg.get("acoustic_tokenizer", {}))

    d_lm = dec_cfg["hidden_size"]           # 1536
    n_layers = dec_cfg["num_hidden_layers"]  # 28
    vae_dim = at_cfg.get("vae_dim", 64)

    print(f"config: d_lm={d_lm}, n_layers={n_layers}, vae_dim={vae_dim}")

    # ── 2. Tokenize text ────────────────────────────────────────────────────
    print(f"tokenizing: {text!r}")
    # VibeVoice's model_type isn't registered in transformers yet.
    # Fall back to Qwen2.5 tokenizer (same BPE vocab).
    tok_err = None
    tokenizer = None
    for tok_src in [str(model_path), "Qwen/Qwen2.5-1.5B"]:
        try:
            tokenizer = AutoTokenizer.from_pretrained(tok_src, trust_remote_code=True)
            print(f"  tokenizer loaded from: {tok_src}")
            break
        except Exception as e:
            tok_err = e
    if tokenizer is None:
        raise RuntimeError(f"Could not load tokenizer: {tok_err}")
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    token_ids_np = np.array(token_ids, dtype=np.int32)
    print(f"  {len(token_ids)} tokens: {token_ids[:20]}{'...' if len(token_ids) > 20 else ''}")
    dump_i32(out_path / "text_token_ids.bin", token_ids_np)

    # ── 3. Load all model shards ────────────────────────────────────────────
    index_path = model_path / "model.safetensors.index.json"
    if index_path.exists():
        with open(index_path) as f:
            weight_map = json.load(f)["weight_map"]
        shards = sorted(set(weight_map.values()))
        print(f"loading {len(shards)} safetensor shards...")
        state_dict = {}
        for shard_name in shards:
            print(f"  loading {shard_name}")
            sd = load_file(str(model_path / shard_name))
            state_dict.update(sd)
    else:
        # Single-file model
        st_path = model_path / "model.safetensors"
        print(f"loading {st_path}")
        state_dict = load_file(str(st_path))

    print(f"  {len(state_dict)} tensors loaded")

    def G(key):
        """Get tensor from state_dict, cast to float32."""
        t = state_dict.get(key)
        if t is None:
            raise KeyError(f"Missing tensor: {key}")
        return t.float()

    def Gopt(key):
        """Get tensor if it exists, else None."""
        t = state_dict.get(key)
        return t.float() if t is not None else None

    # ── 4. Run Qwen2 LM forward (text -> hidden states) ────────────────────
    print("running Qwen2 LM forward pass...")

    # Token embeddings
    embed_w = G("model.language_model.embed_tokens.weight")  # [vocab, d_lm]
    token_ids_t = torch.tensor(token_ids, dtype=torch.long)
    hidden = embed_w[token_ids_t]  # [T_text, d_lm]

    n_tokens = len(token_ids)

    # RoPE frequencies
    head_dim = dec_cfg.get("head_dim", d_lm // dec_cfg["num_attention_heads"])
    rope_theta = dec_cfg.get("rope_theta", 1000000.0)
    positions = torch.arange(n_tokens, dtype=torch.float32)
    freqs = 1.0 / (rope_theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    angles = positions.unsqueeze(1) * freqs.unsqueeze(0)  # [T, head_dim/2]
    cos_cache = torch.cos(angles)  # [T, head_dim/2]
    sin_cache = torch.sin(angles)  # [T, head_dim/2]

    # Causal mask
    causal_mask = torch.full((n_tokens, n_tokens), float("-inf"))
    causal_mask = torch.triu(causal_mask, diagonal=1)  # upper triangle = -inf

    n_heads = dec_cfg["num_attention_heads"]
    n_kv_heads = dec_cfg["num_key_value_heads"]
    gqa_groups = n_heads // n_kv_heads

    with torch.no_grad():
        for il in range(n_layers):
            if il % 7 == 0:
                print(f"  layer {il}/{n_layers}")
            prefix = f"model.language_model.layers.{il}"

            residual = hidden

            # Pre-RMSNorm (input_layernorm)
            norm_w = G(f"{prefix}.input_layernorm.weight")
            rms = hidden.pow(2).mean(dim=-1, keepdim=True).add(1e-6).sqrt()
            h = (hidden / rms) * norm_w

            # Self-attention: Q, K, V projections
            q = torch.nn.functional.linear(h, G(f"{prefix}.self_attn.q_proj.weight"),
                                           Gopt(f"{prefix}.self_attn.q_proj.bias"))
            k = torch.nn.functional.linear(h, G(f"{prefix}.self_attn.k_proj.weight"),
                                           Gopt(f"{prefix}.self_attn.k_proj.bias"))
            v = torch.nn.functional.linear(h, G(f"{prefix}.self_attn.v_proj.weight"),
                                           Gopt(f"{prefix}.self_attn.v_proj.bias"))

            # Reshape for multi-head attention
            q = q.view(n_tokens, n_heads, head_dim)      # [T, n_heads, head_dim]
            k = k.view(n_tokens, n_kv_heads, head_dim)    # [T, n_kv_heads, head_dim]
            v = v.view(n_tokens, n_kv_heads, head_dim)    # [T, n_kv_heads, head_dim]

            # Apply RoPE
            q_r = q[..., ::2]
            q_i = q[..., 1::2]
            q_rot = torch.stack([
                q_r * cos_cache.unsqueeze(1) - q_i * sin_cache.unsqueeze(1),
                q_r * sin_cache.unsqueeze(1) + q_i * cos_cache.unsqueeze(1),
            ], dim=-1).flatten(-2)
            q = q_rot

            k_r = k[..., ::2]
            k_i = k[..., 1::2]
            k_rot = torch.stack([
                k_r * cos_cache.unsqueeze(1) - k_i * sin_cache.unsqueeze(1),
                k_r * sin_cache.unsqueeze(1) + k_i * cos_cache.unsqueeze(1),
            ], dim=-1).flatten(-2)
            k = k_rot

            # GQA: repeat K, V for grouped query attention
            if gqa_groups > 1:
                k = k.unsqueeze(2).expand(-1, -1, gqa_groups, -1).reshape(n_tokens, n_heads, head_dim)
                v = v.unsqueeze(2).expand(-1, -1, gqa_groups, -1).reshape(n_tokens, n_heads, head_dim)

            # Attention: [T, n_heads, head_dim] -> transpose to [n_heads, T, head_dim]
            q = q.permute(1, 0, 2)  # [n_heads, T, head_dim]
            k = k.permute(1, 0, 2)
            v = v.permute(1, 0, 2)

            scale = 1.0 / math.sqrt(head_dim)
            attn = torch.matmul(q, k.transpose(-2, -1)) * scale  # [n_heads, T, T]
            attn = attn + causal_mask.unsqueeze(0)
            attn = torch.softmax(attn, dim=-1)
            attn_out = torch.matmul(attn, v)  # [n_heads, T, head_dim]

            # Merge heads
            attn_out = attn_out.permute(1, 0, 2).contiguous().reshape(n_tokens, d_lm)

            # Output projection
            attn_out = torch.nn.functional.linear(attn_out, G(f"{prefix}.self_attn.o_proj.weight"))

            hidden = residual + attn_out

            # Post-attention RMSNorm (post_attention_layernorm)
            residual = hidden
            norm_w2 = G(f"{prefix}.post_attention_layernorm.weight")
            rms = hidden.pow(2).mean(dim=-1, keepdim=True).add(1e-6).sqrt()
            h = (hidden / rms) * norm_w2

            # MLP: gate_proj, up_proj, down_proj (SwiGLU)
            gate = torch.nn.functional.linear(h, G(f"{prefix}.mlp.gate_proj.weight"))
            up = torch.nn.functional.linear(h, G(f"{prefix}.mlp.up_proj.weight"))
            h = torch.nn.functional.silu(gate) * up
            h = torch.nn.functional.linear(h, G(f"{prefix}.mlp.down_proj.weight"))

            hidden = residual + h

        # Final RMSNorm
        final_norm_w = G("model.language_model.norm.weight")
        rms = hidden.pow(2).mean(dim=-1, keepdim=True).add(1e-6).sqrt()
        hidden = (hidden / rms) * final_norm_w

    lm_hidden = hidden.numpy()  # [T_text, d_lm]
    dump_f32(out_path / "lm_hidden_states.bin", lm_hidden)

    last_hidden = lm_hidden[-1]  # [d_lm]
    dump_f32(out_path / "lm_last_hidden.bin", last_hidden)
    print(f"  LM hidden: shape={lm_hidden.shape}, last rms={np.sqrt(np.mean(last_hidden**2)):.4f}")

    # ── 5. Estimate frame count ─────────────────────────────────────────────
    if n_frames is None:
        # ~0.5 frames per character, capped at 300 frames (~40 sec)
        n_frames = max(1, int(len(text) * 0.5))
        n_frames = min(n_frames, 300)
    print(f"generating {n_frames} frames (~{n_frames / 7.5:.1f}s) with {num_steps} DDIM steps")

    # ── 6. DDIM diffusion ───────────────────────────────────────────────────
    alphas_cumprod, timesteps = make_ddim_schedule(num_steps)

    rng = np.random.default_rng(seed)
    latent = rng.standard_normal((n_frames, vae_dim)).astype(np.float32)

    # Condition: broadcast last hidden state to all frames
    condition = last_hidden.copy()  # [d_lm]

    with torch.no_grad():
        for step in range(num_steps):
            t = float(timesteps[step])
            if step % 5 == 0:
                print(f"  diffusion step {step + 1}/{num_steps} (t={int(t)})")

            # Sinusoidal timestep embedding [256]
            t_sin = sinusoidal_embed(t, 256)
            t_sin_t = torch.from_numpy(t_sin)

            # Time embedding MLP: sin[256] -> Linear -> SiLU -> Linear -> [d_lm]
            t_emb = torch.nn.functional.linear(t_sin_t, G("model.prediction_head.t_embedder.mlp.0.weight"),
                                                Gopt("model.prediction_head.t_embedder.mlp.0.bias"))
            t_emb = torch.nn.functional.silu(t_emb)
            t_emb = torch.nn.functional.linear(t_emb, G("model.prediction_head.t_embedder.mlp.2.weight"),
                                                Gopt("model.prediction_head.t_embedder.mlp.2.bias"))
            # t_emb: [d_lm]

            if step == 0:
                dump_f32(out_path / "pred_t_emb_step0.bin", t_emb.numpy())

            # Project noisy latent: [n_frames, vae_dim] -> [n_frames, d_lm]
            latent_t = torch.from_numpy(latent)
            x = torch.nn.functional.linear(latent_t, G("model.prediction_head.noisy_images_proj.weight"),
                                           Gopt("model.prediction_head.noisy_proj.bias"))

            # Project condition: [d_lm] -> [d_lm]
            cond_t = torch.from_numpy(condition)
            cond_proj = torch.nn.functional.linear(cond_t, G("model.prediction_head.cond_proj.weight"),
                                                    Gopt("model.prediction_head.cond_proj.bias"))
            # cond_proj: [d_lm], broadcast to all frames

            # Combined: c = cond_proj + t_emb (both [d_lm], broadcast over frames)
            c = cond_proj + t_emb  # [d_lm]

            if step == 0:
                dump_f32(out_path / "pred_c_step0.bin", c.numpy())

            # 4 AdaLN + SwiGLU layers
            for li in range(4):
                base = f"model.prediction_head.layers.{li}"

                # AdaLN: c[d_lm] -> [3*d_lm] via linear (no bias)
                adaln_out = torch.nn.functional.linear(c, G(f"{base}.adaLN_modulation.1.weight"),
                                                        Gopt(f"{base}.adaLN_modulation.1.bias"))
                shift = adaln_out[:d_lm]
                scale = adaln_out[d_lm:2*d_lm]
                gate = adaln_out[2*d_lm:]

                # RMSNorm + modulate: h = norm(x) * weight * (1 + scale) + shift
                norm_w = G(f"{base}.norm.weight")
                rms = x.pow(2).mean(dim=-1, keepdim=True).add(1e-6).sqrt()
                h = (x / rms) * norm_w
                h = h * (1.0 + scale.unsqueeze(0)) + shift.unsqueeze(0)

                # SwiGLU FFN
                gate_proj = torch.nn.functional.linear(h, G(f"{base}.ffn.gate_proj.weight"))
                up_proj = torch.nn.functional.linear(h, G(f"{base}.ffn.up_proj.weight"))
                h = torch.nn.functional.silu(gate_proj) * up_proj
                h = torch.nn.functional.linear(h, G(f"{base}.ffn.down_proj.weight"))

                # DiT-style gate (no sigmoid)
                h = h * gate.unsqueeze(0)

                x = x + h

            # Final AdaLN + project to vae_dim
            final_adaln = torch.nn.functional.linear(t_emb, G("model.prediction_head.final_layer.adaLN_modulation.1.weight"),
                                                      Gopt("model.prediction_head.final_layer.adaLN_modulation.1.bias"))
            shift = final_adaln[:d_lm]
            scale = final_adaln[d_lm:]

            norm_w_final = Gopt("model.prediction_head.final_layer.norm.weight")
            rms = x.pow(2).mean(dim=-1, keepdim=True).add(1e-6).sqrt()
            if norm_w_final is not None:
                h = (x / rms) * norm_w_final
            else:
                h = x / rms
            h = h * (1.0 + scale.unsqueeze(0)) + shift.unsqueeze(0)

            v_pred = torch.nn.functional.linear(h, G("model.prediction_head.final_layer.linear.weight"),
                                                 Gopt("model.prediction_head.final_layer.linear.bias"))
            # v_pred: [n_frames, vae_dim]

            if step == 0:
                # Dump first frame's prediction
                dump_f32(out_path / "pred_eps_step0.bin", v_pred[0].numpy())

            # DDIM step (v-prediction)
            v_pred_np = v_pred.numpy()
            latent = ddim_step_v(alphas_cumprod, timesteps, step, latent, v_pred_np).astype(np.float32)

    dump_f32(out_path / "latent_after_diffusion.bin", latent)
    print(f"  diffusion done: latent shape={latent.shape}, "
          f"rms={np.sqrt(np.mean(latent**2)):.4f}, "
          f"min={latent.min():.4f}, max={latent.max():.4f}")

    # ── 7. Latent scaling ───────────────────────────────────────────────────
    sf_t = Gopt("model.speech_scaling_factor") or Gopt("speech_scaling_factor")
    bf_t = Gopt("model.speech_bias_factor") or Gopt("speech_bias_factor")
    scaling_factor = float(sf_t.item()) if sf_t is not None else 0.196
    bias_factor = float(bf_t.item()) if bf_t is not None else -0.049

    scaled_latent = latent / scaling_factor - bias_factor
    dump_f32(out_path / "scaled_latent.bin", scaled_latent)
    print(f"  scaled latent: rms={np.sqrt(np.mean(scaled_latent**2)):.4f} "
          f"(sf={scaling_factor:.4f}, bf={bias_factor:.4f})")

    # ── 8. Decode through sigma-VAE decoder ─────────────────────────────────
    print("running sigma-VAE decoder...")

    # Decoder architecture (mirrored encoder):
    #   stem conv1d(vae_dim -> ch, K=7, stride=1)
    #   6 upsample stages with ConvTranspose1d + ConvNeXt blocks
    #   head conv1d(ch_last -> 1, K=7) -> mono audio
    # Ratios: [8, 5, 5, 4, 2, 2] (decoder order, same as config)

    encoder_ratios = at_cfg.get("encoder_ratios", [8, 5, 5, 4, 2, 2])

    # Determine tensor key prefix for decoder
    # Try common prefixes
    dec_prefix = None
    for prefix_candidate in [
        "model.acoustic_tokenizer.decoder",
        "model.acoustic_decoder",
    ]:
        test_key = f"{prefix_candidate}.upsample_layers.0.0.conv.conv.weight"
        if test_key in state_dict:
            dec_prefix = prefix_candidate
            break

    if dec_prefix is None:
        # Scan state_dict for decoder-like keys
        dec_keys = [k for k in state_dict if "decoder" in k.lower() and "acoustic" in k.lower()]
        if dec_keys:
            # Extract common prefix
            sample = dec_keys[0]
            if ".decoder." in sample:
                dec_prefix = sample[:sample.index(".decoder.") + len(".decoder")]
            print(f"  auto-detected decoder prefix: {dec_prefix}")
        else:
            print("  WARNING: no decoder tensors found, skipping audio generation")
            print(f"  sample keys: {list(state_dict.keys())[:10]}")
            return

    print(f"  decoder prefix: {dec_prefix}")

    def D(key):
        full = f"{dec_prefix}.{key}"
        t = state_dict.get(full)
        if t is None:
            raise KeyError(f"Missing decoder tensor: {full}")
        return t.float()

    def Dopt(key):
        full = f"{dec_prefix}.{key}"
        t = state_dict.get(full)
        return t.float() if t is not None else None

    with torch.no_grad():
        # Input: [n_frames, vae_dim] -> [1, vae_dim, n_frames] (channels-first)
        x = torch.from_numpy(scaled_latent).T.unsqueeze(0)  # [1, vae_dim, T]

        # Stem conv: upsample_layers.0.0 (stride 1)
        stem_w = D("upsample_layers.0.0.conv.conv.weight")
        stem_b = Dopt("upsample_layers.0.0.conv.conv.bias")
        K = stem_w.shape[2]
        x = torch.nn.functional.pad(x, (K - 1, 0), mode="constant", value=0)
        x = torch.nn.functional.conv1d(x, stem_w, stem_b, stride=1)
        print(f"  after stem: {x.shape}")

        # Process stages
        n_stages = len(encoder_ratios) + 1  # stem + 6 upsample stages
        si = 0  # stage index for ConvNeXt blocks

        # Stage 0 ConvNeXt blocks (after stem)
        bi = 0
        while True:
            base = f"stages.{si}.{bi}"
            norm_key = f"{base}.norm.weight"
            if Dopt(norm_key) is None:
                break

            # ConvNeXt Block1D
            res = x
            # Mixer: RMSNorm -> depthwise conv -> gamma -> residual
            norm_w = D(f"{base}.norm.weight")
            rms = x.pow(2).mean(dim=1, keepdim=True).add(1e-5).sqrt()
            h = (x / rms) * norm_w.unsqueeze(0).unsqueeze(-1)

            dw_w = D(f"{base}.mixer.conv.conv.conv.weight")
            dw_b = Dopt(f"{base}.mixer.conv.conv.conv.bias")
            K_dw = dw_w.shape[2]
            h = torch.nn.functional.pad(h, (K_dw - 1, 0), mode="constant", value=0)
            h = torch.nn.functional.conv1d(h, dw_w, dw_b, stride=1, groups=dw_w.shape[0])

            gamma = Dopt(f"{base}.gamma")
            if gamma is not None:
                h = h * gamma.unsqueeze(0).unsqueeze(-1)
            x = res + h

            # FFN: RMSNorm -> Linear -> SiLU -> Linear -> gamma -> residual
            res = x
            ffn_norm_w = D(f"{base}.ffn_norm.weight")
            rms = x.pow(2).mean(dim=1, keepdim=True).add(1e-5).sqrt()
            h = (x / rms) * ffn_norm_w.unsqueeze(0).unsqueeze(-1)

            h = h.permute(0, 2, 1)  # [B, T, C]
            h = torch.nn.functional.linear(h, D(f"{base}.ffn.linear1.weight"),
                                           Dopt(f"{base}.ffn.linear1.bias"))
            h = torch.nn.functional.silu(h)
            h = torch.nn.functional.linear(h, D(f"{base}.ffn.linear2.weight"),
                                           Dopt(f"{base}.ffn.linear2.bias"))
            h = h.permute(0, 2, 1)  # [B, C, T]

            ffn_gamma = Dopt(f"{base}.ffn_gamma")
            if ffn_gamma is not None:
                h = h * ffn_gamma.unsqueeze(0).unsqueeze(-1)
            x = res + h

            bi += 1

        # Upsample stages 1..6
        for ui in range(len(encoder_ratios)):
            stride = encoder_ratios[ui]
            si += 1
            ds_idx = ui + 1

            # Transposed conv (upsample)
            us_w = D(f"upsample_layers.{ds_idx}.0.convtr.convtr.weight")
            us_b = Dopt(f"upsample_layers.{ds_idx}.0.convtr.convtr.bias")

            T_in = x.shape[-1]
            # ConvTranspose1d weight: [C_in, C_out, K]
            x_up = torch.nn.functional.conv_transpose1d(
                x, us_w, us_b, stride=stride
            )
            # Trim to T_in * stride (causal: remove K - stride from end)
            T_target = T_in * stride
            K_us = us_w.shape[2]
            T_raw = x_up.shape[-1]
            if T_raw > T_target:
                x_up = x_up[..., :T_target]
            x = x_up
            print(f"  after upsample {ui} (stride={stride}): {x.shape}")

            # ConvNeXt blocks for this stage
            bi = 0
            while True:
                base = f"stages.{si}.{bi}"
                if Dopt(f"{base}.norm.weight") is None:
                    break

                res = x
                norm_w = D(f"{base}.norm.weight")
                rms = x.pow(2).mean(dim=1, keepdim=True).add(1e-5).sqrt()
                h = (x / rms) * norm_w.unsqueeze(0).unsqueeze(-1)

                dw_w = D(f"{base}.mixer.conv.conv.conv.weight")
                dw_b = Dopt(f"{base}.mixer.conv.conv.conv.bias")
                K_dw = dw_w.shape[2]
                h = torch.nn.functional.pad(h, (K_dw - 1, 0), mode="constant", value=0)
                h = torch.nn.functional.conv1d(h, dw_w, dw_b, stride=1, groups=dw_w.shape[0])

                gamma = Dopt(f"{base}.gamma")
                if gamma is not None:
                    h = h * gamma.unsqueeze(0).unsqueeze(-1)
                x = res + h

                res = x
                ffn_norm_w = D(f"{base}.ffn_norm.weight")
                rms = x.pow(2).mean(dim=1, keepdim=True).add(1e-5).sqrt()
                h = (x / rms) * ffn_norm_w.unsqueeze(0).unsqueeze(-1)

                h = h.permute(0, 2, 1)
                h = torch.nn.functional.linear(h, D(f"{base}.ffn.linear1.weight"),
                                               Dopt(f"{base}.ffn.linear1.bias"))
                h = torch.nn.functional.silu(h)
                h = torch.nn.functional.linear(h, D(f"{base}.ffn.linear2.weight"),
                                               Dopt(f"{base}.ffn.linear2.bias"))
                h = h.permute(0, 2, 1)

                ffn_gamma = Dopt(f"{base}.ffn_gamma")
                if ffn_gamma is not None:
                    h = h * ffn_gamma.unsqueeze(0).unsqueeze(-1)
                x = res + h

                bi += 1

        # Optional final norm
        final_norm = Dopt("norm.weight")
        if final_norm is not None:
            rms = x.pow(2).mean(dim=1, keepdim=True).add(1e-5).sqrt()
            x = (x / rms) * final_norm.unsqueeze(0).unsqueeze(-1)

        # Head conv -> 1 channel (mono audio)
        head_w = D("head.conv.conv.weight")
        head_b = Dopt("head.conv.conv.bias")
        K_head = head_w.shape[2]
        x = torch.nn.functional.pad(x, (K_head - 1, 0), mode="constant", value=0)
        x = torch.nn.functional.conv1d(x, head_w, head_b, stride=1)

        audio = x.squeeze().numpy()  # [N_samples]
        print(f"  decoder output: {audio.shape} samples ({len(audio)/24000:.2f}s)")

    dump_f32(out_path / "audio_output.bin", audio)
    dump_wav(out_path / "audio_output.wav", audio, sr=24000)

    print(f"\nall outputs written to {out_path}")
    print(f"  audio rms={np.sqrt(np.mean(audio**2)):.4f}, "
          f"min={audio.min():.4f}, max={audio.max():.4f}")


# ── CLI entry point ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="VibeVoice-1.5B TTS reference dump",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--text", required=True,
                        help="Text to synthesize")
    parser.add_argument("--output-dir", required=True,
                        help="Directory for output binary dumps")
    parser.add_argument("--model", default="microsoft/VibeVoice-1.5B",
                        help="HuggingFace model ID or local path (default: microsoft/VibeVoice-1.5B)")
    parser.add_argument("--num-steps", type=int, default=20,
                        help="Number of DDIM diffusion steps (default: 20)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for latent noise (default: 42)")
    parser.add_argument("--n-frames", type=int, default=None,
                        help="Number of speech frames (default: auto from text length)")
    args = parser.parse_args()

    # Resolve model directory
    model_dir = args.model
    if not os.path.isdir(model_dir):
        print(f"downloading model from HuggingFace: {model_dir}")
        from huggingface_hub import snapshot_download
        model_dir = snapshot_download(model_dir)
        print(f"  cached at: {model_dir}")

    run_tts_pipeline(
        model_dir=model_dir,
        text=args.text,
        output_dir=args.output_dir,
        num_steps=args.num_steps,
        seed=args.seed,
        n_frames=args.n_frames,
    )


if __name__ == "__main__":
    main()
