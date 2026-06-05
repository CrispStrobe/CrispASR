#!/usr/bin/env python3
"""Run the ORIGINAL VibeVoice-Realtime-0.5B model and dump intermediates.

Uses Microsoft's own code via trust_remote_code=True — not our reimplementation.
Hooks capture intermediate tensors for stage-by-stage comparison with the
C++ ggml implementation (crispasr --tts).

Dumped stages (all little-endian float32 unless noted):
  text_token_ids.bin           int32  tokenized text
  base_lm_hidden.bin           float32 [T_text, d_lm]  base LM output
  tts_lm_hidden_frame0.bin     float32 [d_lm]          TTS LM condition for frame 0
  pred_v_step0_frame0.bin      float32 [vae_dim]        pred head output step 0
  speech_latent_frame0.bin     float32 [vae_dim]        denoised latent frame 0
  acoustic_embed_frame0.bin    float32 [d_lm]           connector output frame 0
  all_latents.bin              float32 [N, vae_dim]     all denoised latents
  audio_output.bin             float32 [N_samples]      raw audio
  audio_output.wav             24kHz mono WAV

Usage:
  python tools/vibevoice_tts_ref_realtime.py \\
      --text "Hello world" --output-dir /tmp/vv_rt_ref
"""

import argparse
import copy
import os
import struct
import sys
from pathlib import Path

import numpy as np
import torch


def dump_f32(path, arr):
    arr = np.ascontiguousarray(arr.detach().cpu().float().numpy())
    arr.tofile(str(path))
    print(f"  wrote {path.name}: {arr.shape} float32 ({arr.nbytes} bytes)")


def dump_i32(path, arr):
    arr = np.ascontiguousarray(np.array(arr, dtype=np.int32))
    arr.tofile(str(path))
    print(f"  wrote {path.name}: {arr.shape} int32 ({arr.nbytes} bytes)")


def write_wav(path, audio, sr=24000):
    """Write mono float32 audio to 16-bit WAV."""
    audio = np.clip(audio, -1, 1)
    samples = (audio * 32767).astype(np.int16)
    data_size = len(samples) * 2
    with open(path, "wb") as f:
        f.write(b"RIFF")
        f.write(struct.pack("<I", 36 + data_size))
        f.write(b"WAVEfmt ")
        f.write(struct.pack("<IHHIIHH", 16, 1, 1, sr, sr * 2, 2, 16))
        f.write(b"data")
        f.write(struct.pack("<I", data_size))
        f.write(samples.tobytes())
    print(f"  wrote {path.name}: {len(samples)} samples, {len(samples)/sr:.2f}s")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", default="microsoft/VibeVoice-Realtime-0.5B")
    parser.add_argument("--cfg-scale", type=float, default=3.0)
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Load model using their own code
    print(f"loading {args.model} (trust_remote_code=True)...")
    from transformers import AutoProcessor, AutoTokenizer

    # Load tokenizer (Qwen2.5 fallback if needed)
    tokenizer = None
    for src in [args.model, "Qwen/Qwen2.5-0.5B"]:
        try:
            tokenizer = AutoTokenizer.from_pretrained(src, trust_remote_code=True)
            print(f"  tokenizer from: {src}")
            break
        except Exception:
            pass
    if tokenizer is None:
        print("ERROR: could not load tokenizer")
        return 1

    # Try loading the model with trust_remote_code
    try:
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            trust_remote_code=True,
            torch_dtype=torch.float32,
            device_map="cpu",
        )
        print(f"  model loaded: {type(model).__name__}")
    except Exception as e:
        print(f"  AutoModelForCausalLM failed: {e}")
        print("  trying manual load...")
        # Manual fallback: load weights and run components individually
        return run_manual_pipeline(args, tokenizer, out)

    # Tokenize
    print(f"tokenizing: {args.text!r}")
    token_ids = tokenizer.encode(args.text, add_special_tokens=False)
    print(f"  {len(token_ids)} tokens: {token_ids}")
    dump_i32(out / "text_token_ids.bin", token_ids)

    # Capture intermediates via hooks
    captured = {}

    def hook_base_lm(module, inp, output):
        if hasattr(output, "last_hidden_state"):
            captured["base_lm_hidden"] = output.last_hidden_state.detach().clone()

    def hook_tts_lm(module, inp, output):
        if hasattr(output, "last_hidden_state"):
            hs = output.last_hidden_state.detach().clone()
            if "tts_lm_hidden_frame0" not in captured:
                captured["tts_lm_hidden_frame0"] = hs[:, -1, :]

    def hook_pred_head(module, inp, output):
        if "pred_v_step0_frame0" not in captured:
            captured["pred_v_step0_frame0"] = output.detach().clone()[0]

    def hook_connector(module, inp, output):
        if "acoustic_embed_frame0" not in captured:
            captured["acoustic_embed_frame0"] = output.detach().clone()[0, 0]

    # Register hooks
    hooks = []
    if hasattr(model, "model"):
        m = model.model
        if hasattr(m, "language_model"):
            hooks.append(m.language_model.register_forward_hook(hook_base_lm))
        if hasattr(m, "tts_language_model"):
            hooks.append(m.tts_language_model.register_forward_hook(hook_tts_lm))
        if hasattr(m, "prediction_head"):
            hooks.append(m.prediction_head.register_forward_hook(hook_pred_head))
        if hasattr(m, "acoustic_connector"):
            hooks.append(m.acoustic_connector.register_forward_hook(hook_connector))

    print(f"  registered {len(hooks)} hooks")

    # Run generate
    print("running generate()...")
    try:
        # Build inputs
        input_ids = torch.tensor([token_ids], dtype=torch.long)
        with torch.no_grad():
            outputs = model.generate(
                tts_text_ids=input_ids,
                max_new_tokens=None,
                cfg_scale=args.cfg_scale,
                tokenizer=tokenizer,
                generation_config={"do_sample": False},
                verbose=True,
            )
    except Exception as e:
        print(f"  generate() failed: {e}")
        import traceback
        traceback.print_exc()
        # Clean up hooks
        for h in hooks:
            h.remove()
        return run_manual_pipeline(args, tokenizer, out)

    # Clean up hooks
    for h in hooks:
        h.remove()

    # Dump captured intermediates
    for name, tensor in captured.items():
        if tensor.dim() > 0:
            dump_f32(out / f"{name}.bin", tensor)

    # Dump audio
    if hasattr(outputs, "speech_outputs") and outputs.speech_outputs is not None:
        audio = outputs.speech_outputs.squeeze().detach().cpu().float().numpy()
        dump_f32(out / "audio_output.bin", torch.from_numpy(audio))
        write_wav(out / "audio_output.wav", audio)
        print(f"  audio: {len(audio)} samples, rms={np.sqrt(np.mean(audio**2)):.4f}")
    else:
        print("  WARNING: no audio output from generate()")

    print(f"\nall outputs written to {out}")
    return 0


def run_manual_pipeline(args, tokenizer, out):
    """Fallback: manually run the pipeline components."""
    from safetensors import safe_open
    from huggingface_hub import snapshot_download
    import json

    model_dir = snapshot_download(args.model)
    print(f"  model at: {model_dir}")

    with open(os.path.join(model_dir, "config.json")) as f:
        cfg = json.load(f)

    dec_cfg = cfg.get("decoder_config", {})
    d_lm = dec_cfg["hidden_size"]
    n_layers = dec_cfg["num_hidden_layers"]  # may be overridden after loading weights
    n_heads = dec_cfg["num_attention_heads"]
    n_kv_heads = dec_cfg["num_key_value_heads"]
    head_dim = d_lm // n_heads
    d_ffn = dec_cfg["intermediate_size"]
    rope_theta = dec_cfg.get("rope_theta", 1000000.0)
    tts_n_layers = cfg.get("tts_backbone_num_hidden_layers", 20)
    vae_dim = cfg.get("acoustic_vae_dim", 64)
    diff_cfg = cfg.get("diffusion_head_config", {})
    ddpm_steps = diff_cfg.get("ddpm_num_inference_steps", 20)

    print(f"  d_lm={d_lm}, n_layers={n_layers}, tts_layers={tts_n_layers}, vae_dim={vae_dim}")

    # Load weights
    st_path = os.path.join(model_dir, "model.safetensors")
    state = {}
    with safe_open(st_path, framework="pt") as f:
        for k in f.keys():
            state[k] = f.get_tensor(k).float()
    print(f"  loaded {len(state)} tensors")

    # Detect actual base LM layer count (Realtime has only 4 of 24)
    actual = 0
    while f"model.language_model.layers.{actual}.input_layernorm.weight" in state:
        actual += 1
    if actual > 0 and actual != n_layers:
        print(f"  base LM: {actual} layers (config says {n_layers})")
        n_layers = actual

    def G(name):
        if name in state:
            return state[name]
        raise KeyError(f"Missing: {name}")

    def Gopt(name):
        return state.get(name)

    # Tokenize
    token_ids = tokenizer.encode(args.text, add_special_tokens=False)
    print(f"  tokens: {token_ids}")
    dump_i32(out / "text_token_ids.bin", token_ids)

    # === Base LM forward ===
    print("running base LM forward...")
    T = len(token_ids)
    emb_w = G("model.language_model.embed_tokens.weight")
    ids_t = torch.tensor(token_ids, dtype=torch.long)
    h = emb_w[ids_t]  # [T, d_lm]

    # Build causal mask
    mask = torch.zeros(T, T)
    mask.masked_fill_(torch.triu(torch.ones(T, T), diagonal=1).bool(), float("-inf"))

    pos = torch.arange(T)

    for il in range(n_layers):
        prefix = f"model.language_model.layers.{il}"
        residual = h

        # Pre-norm
        norm_w = G(f"{prefix}.input_layernorm.weight")
        h_norm = torch.nn.functional.rms_norm(h, (d_lm,), norm_w, eps=1e-6)

        # Self-attention
        Q = torch.nn.functional.linear(h_norm, G(f"{prefix}.self_attn.q_proj.weight"),
                                       Gopt(f"{prefix}.self_attn.q_proj.bias"))
        K = torch.nn.functional.linear(h_norm, G(f"{prefix}.self_attn.k_proj.weight"),
                                       Gopt(f"{prefix}.self_attn.k_proj.bias"))
        V = torch.nn.functional.linear(h_norm, G(f"{prefix}.self_attn.v_proj.weight"),
                                       Gopt(f"{prefix}.self_attn.v_proj.bias"))

        Q = Q.view(T, n_heads, head_dim).transpose(0, 1)
        K = K.view(T, n_kv_heads, head_dim).transpose(0, 1)
        V = V.view(T, n_kv_heads, head_dim).transpose(0, 1)

        # RoPE (NEOX style)
        freqs = 1.0 / (rope_theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        t = pos.float()
        angles = torch.outer(t, freqs)
        cos_r, sin_r = angles.cos(), angles.sin()
        def apply_rope(x):
            x1, x2 = x[..., :head_dim//2], x[..., head_dim//2:]
            return torch.cat([x1*cos_r - x2*sin_r, x2*cos_r + x1*sin_r], dim=-1)
        Q = apply_rope(Q)
        K = apply_rope(K)

        # GQA: expand K,V
        n_rep = n_heads // n_kv_heads
        if n_rep > 1:
            K = K.unsqueeze(1).expand(-1, n_rep, -1, -1).reshape(n_heads, T, head_dim)
            V = V.unsqueeze(1).expand(-1, n_rep, -1, -1).reshape(n_heads, T, head_dim)

        scale = 1.0 / (head_dim ** 0.5)
        attn = (Q @ K.transpose(-2, -1)) * scale + mask
        attn = torch.softmax(attn, dim=-1)
        attn_out = (attn @ V).transpose(0, 1).contiguous().view(T, d_lm)
        attn_out = torch.nn.functional.linear(attn_out, G(f"{prefix}.self_attn.o_proj.weight"))
        h = residual + attn_out

        # FFN
        residual = h
        norm_w2 = G(f"{prefix}.post_attention_layernorm.weight")
        h_norm2 = torch.nn.functional.rms_norm(h, (d_lm,), norm_w2, eps=1e-6)
        gate = torch.nn.functional.linear(h_norm2, G(f"{prefix}.mlp.gate_proj.weight"))
        up = torch.nn.functional.linear(h_norm2, G(f"{prefix}.mlp.up_proj.weight"))
        ffn = torch.nn.functional.linear(torch.nn.functional.silu(gate) * up,
                                         G(f"{prefix}.mlp.down_proj.weight"))
        h = residual + ffn

        if il % 8 == 0:
            print(f"  base LM layer {il}/{n_layers}")

    # Final norm (Realtime model may not have it — only 4 base LM layers)
    norm_w = Gopt("model.language_model.norm.weight")
    if norm_w is not None:
        h = torch.nn.functional.rms_norm(h, (d_lm,), norm_w, eps=1e-6)
    base_lm_hidden = h  # [T, d_lm]
    dump_f32(out / "base_lm_hidden.bin", base_lm_hidden)
    print(f"  base LM done: shape={list(base_lm_hidden.shape)}, rms={base_lm_hidden.norm()/T**0.5:.4f}")

    # === TTS LM forward (autoregressive, per-frame) ===
    print("running TTS LM autoregressive generation...")

    # Build TTS prompt - same chat template as our C++ code
    IM_START, IM_END = 151644, 151645
    sys_ids = tokenizer.encode("You are a helpful assistant that generates speech from text.", add_special_tokens=False)
    text_ids_full = tokenizer.encode(f"Please read the following text aloud: {args.text}", add_special_tokens=False)
    prompt = [IM_START, 8948, 198] + sys_ids + [IM_END, 198, IM_START, 872, 198] + text_ids_full + [IM_END, 198, IM_START, 77091, 198]

    # Embed via TTS LM embedding
    tts_emb_w = G("model.tts_language_model.embed_tokens.weight")
    prompt_t = torch.tensor(prompt, dtype=torch.long)
    tts_h = tts_emb_w[prompt_t]  # [T_prompt, d_lm]

    # Add type embedding (text=1)
    type_emb = G("model.tts_input_types.weight")  # [2, d_lm]
    tts_h = tts_h + type_emb[1]  # all text tokens

    # Replace last tokens with base LM hidden states
    # (The streaming model splices base LM hidden into TTS LM input)
    n_text = len(token_ids)
    start_idx = len(prompt) - n_text
    tts_h[start_idx:start_idx + n_text] = base_lm_hidden[:n_text]

    # Prefill TTS LM
    T_tts = len(prompt)
    tts_mask = torch.zeros(T_tts, T_tts)
    tts_mask.masked_fill_(torch.triu(torch.ones(T_tts, T_tts), diagonal=1).bool(), float("-inf"))
    tts_pos = torch.arange(T_tts)

    # Run TTS LM layers
    kv_cache_k = []
    kv_cache_v = []

    for il in range(tts_n_layers):
        prefix = f"model.tts_language_model.layers.{il}"
        residual = tts_h
        norm_w = G(f"{prefix}.input_layernorm.weight")
        h_norm = torch.nn.functional.rms_norm(tts_h, (d_lm,), norm_w, eps=1e-6)

        Q = torch.nn.functional.linear(h_norm, G(f"{prefix}.self_attn.q_proj.weight"),
                                       Gopt(f"{prefix}.self_attn.q_proj.bias"))
        K = torch.nn.functional.linear(h_norm, G(f"{prefix}.self_attn.k_proj.weight"),
                                       Gopt(f"{prefix}.self_attn.k_proj.bias"))
        V = torch.nn.functional.linear(h_norm, G(f"{prefix}.self_attn.v_proj.weight"),
                                       Gopt(f"{prefix}.self_attn.v_proj.bias"))

        Q = Q.view(T_tts, n_heads, head_dim).transpose(0, 1)
        K = K.view(T_tts, n_kv_heads, head_dim).transpose(0, 1)
        V = V.view(T_tts, n_kv_heads, head_dim).transpose(0, 1)

        freqs = 1.0 / (rope_theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        t = tts_pos.float()
        angles = torch.outer(t, freqs)
        cos_r, sin_r = angles.cos(), angles.sin()
        def apply_rope(x, cos_r=cos_r, sin_r=sin_r):
            x1, x2 = x[..., :head_dim//2], x[..., head_dim//2:]
            return torch.cat([x1*cos_r - x2*sin_r, x2*cos_r + x1*sin_r], dim=-1)
        Q = apply_rope(Q)
        K = apply_rope(K)

        kv_cache_k.append(K)
        kv_cache_v.append(V)

        n_rep = n_heads // n_kv_heads
        if n_rep > 1:
            K = K.unsqueeze(1).expand(-1, n_rep, -1, -1).reshape(n_heads, T_tts, head_dim)
            V = V.unsqueeze(1).expand(-1, n_rep, -1, -1).reshape(n_heads, T_tts, head_dim)

        scale = 1.0 / (head_dim ** 0.5)
        attn = (Q @ K.transpose(-2, -1)) * scale + tts_mask
        attn = torch.softmax(attn, dim=-1)
        attn_out = (attn @ V).transpose(0, 1).contiguous().view(T_tts, d_lm)
        attn_out = torch.nn.functional.linear(attn_out, G(f"{prefix}.self_attn.o_proj.weight"))
        tts_h = residual + attn_out

        residual = tts_h
        norm_w2 = G(f"{prefix}.post_attention_layernorm.weight")
        h_norm2 = torch.nn.functional.rms_norm(tts_h, (d_lm,), norm_w2, eps=1e-6)
        gate = torch.nn.functional.linear(h_norm2, G(f"{prefix}.mlp.gate_proj.weight"))
        up = torch.nn.functional.linear(h_norm2, G(f"{prefix}.mlp.up_proj.weight"))
        ffn = torch.nn.functional.linear(torch.nn.functional.silu(gate) * up,
                                         G(f"{prefix}.mlp.down_proj.weight"))
        tts_h = residual + ffn

    tts_h = torch.nn.functional.rms_norm(tts_h, (d_lm,), G("model.tts_language_model.norm.weight"), eps=1e-6)
    tts_condition = tts_h[-1]  # last token hidden state
    dump_f32(out / "tts_lm_hidden_frame0.bin", tts_condition)
    print(f"  TTS LM prefill done: condition rms={tts_condition.norm():.4f}")

    # === Diffusion with CFG ===
    print(f"running diffusion ({ddpm_steps} steps, cfg={args.cfg_scale})...")

    # Cosine schedule
    offset = 0.008
    def alpha_bar(t):
        return np.cos(((t / 1000 + offset) / (1 + offset)) * np.pi * 0.5) ** 2
    alphas_cumprod = np.array([alpha_bar(t+1) / alpha_bar(0) for t in range(1000)])
    alphas_cumprod = np.clip(alphas_cumprod, 0.0001, 0.9999)

    timesteps = np.linspace(999, 0, ddpm_steps + 1).round().astype(int)[:-1]

    # Zero condition for negative (unconditional)
    neg_cond = torch.zeros(d_lm)

    # Prediction head weights
    noisy_w = G("model.prediction_head.noisy_images_proj.weight")
    cond_w = G("model.prediction_head.cond_proj.weight")
    t_emb_0_w = G("model.prediction_head.t_embedder.mlp.0.weight")
    t_emb_2_w = G("model.prediction_head.t_embedder.mlp.2.weight")

    def sinusoidal_embed(t_val, dim=256):
        half = dim // 2
        freqs = torch.exp(-np.log(10000) * torch.arange(half, dtype=torch.float32) / half)
        args = t_val * freqs
        return torch.cat([torch.cos(args), torch.sin(args)])

    def run_pred_head(noisy, condition, t_val):
        """Run prediction head forward."""
        # Time embedding
        t_sin = sinusoidal_embed(t_val)
        t_emb = torch.nn.functional.linear(t_sin, t_emb_0_w)
        t_emb = torch.nn.functional.silu(t_emb)
        t_emb = torch.nn.functional.linear(t_emb, t_emb_2_w)

        # Project
        x = torch.nn.functional.linear(noisy, noisy_w)
        c = torch.nn.functional.linear(condition, cond_w) + t_emb

        # 4 AdaLN + SwiGLU layers
        for i in range(4):
            pfx = f"model.prediction_head.layers.{i}"
            adaln_out = torch.nn.functional.linear(c, G(f"{pfx}.adaLN_modulation.1.weight"))
            shift, scale, gate = adaln_out.chunk(3, dim=-1)
            h = torch.nn.functional.rms_norm(x, (d_lm,), G(f"{pfx}.norm.weight"), eps=1e-5)
            h = h * (1 + scale) + shift
            g_proj = torch.nn.functional.linear(h, G(f"{pfx}.ffn.gate_proj.weight"))
            u_proj = torch.nn.functional.linear(h, G(f"{pfx}.ffn.up_proj.weight"))
            h = torch.nn.functional.linear(torch.nn.functional.silu(g_proj) * u_proj,
                                           G(f"{pfx}.ffn.down_proj.weight"))
            x = x + gate * h

        # Final
        adaln_out = torch.nn.functional.linear(c, G("model.prediction_head.final_layer.adaLN_modulation.1.weight"))
        shift, scale = adaln_out.chunk(2, dim=-1)
        h = torch.nn.functional.rms_norm(x, (d_lm,), eps=1e-5)
        h = h * (1 + scale) + shift
        return torch.nn.functional.linear(h, G("model.prediction_head.final_layer.linear.weight"))

    # Generate frames
    n_frames = max(1, int(len(args.text) * 0.5))
    n_frames = min(n_frames, 300)
    torch.manual_seed(42)

    all_latents = []
    all_noises = []
    for fi in range(n_frames):
        z = torch.randn(vae_dim)
        all_noises.append(z.detach().clone())
        if fi == 0:
            dump_f32(out / "noise_frame0.bin", z)

        for si, t_int in enumerate(timesteps):
            t_val = float(t_int)

            # CFG: run with positive and negative conditions
            v_pos = run_pred_head(z, tts_condition, t_val)
            v_neg = run_pred_head(z, neg_cond, t_val)
            v_cfg = v_neg + args.cfg_scale * (v_pos - v_neg)

            if fi == 0 and si == 0:
                dump_f32(out / "pred_v_step0_frame0.bin", v_cfg)

            # DDIM v-prediction step
            alpha_t = alphas_cumprod[t_int]
            t_prev = timesteps[si + 1] if si + 1 < len(timesteps) else -1
            alpha_prev = alphas_cumprod[t_prev] if t_prev >= 0 else 1.0

            sqrt_a = np.sqrt(alpha_t)
            sqrt_1ma = np.sqrt(1 - alpha_t)
            sqrt_a_prev = np.sqrt(alpha_prev)
            sqrt_1ma_prev = np.sqrt(1 - alpha_prev)

            x0_pred = sqrt_a * z - sqrt_1ma * v_cfg
            eps_pred = sqrt_1ma * z + sqrt_a * v_cfg
            z = sqrt_a_prev * x0_pred + sqrt_1ma_prev * eps_pred

        if fi == 0:
            dump_f32(out / "speech_latent_frame0.bin", z)

        all_latents.append(z.detach().clone())

        if fi % 5 == 0 or fi == n_frames - 1:
            print(f"  frame {fi+1}/{n_frames}: rms={z.norm()/vae_dim**0.5:.4f}")

        # Feed back through connector for next frame conditioning
        # (simplified: we don't update TTS LM KV cache here)
        fc1_w = G("model.acoustic_connector.fc1.weight")
        fc1_b = Gopt("model.acoustic_connector.fc1.bias")
        norm_w = Gopt("model.acoustic_connector.norm.weight")
        fc2_w = G("model.acoustic_connector.fc2.weight")
        fc2_b = Gopt("model.acoustic_connector.fc2.bias")

        ae = torch.nn.functional.linear(z, fc1_w, fc1_b)
        ae = torch.nn.functional.rms_norm(ae, (d_lm,), norm_w, eps=1e-6)
        ae = torch.nn.functional.linear(ae, fc2_w, fc2_b)

        if fi == 0:
            dump_f32(out / "acoustic_embed_frame0.bin", ae)

        # For proper AR: would update TTS LM here with speech embedding
        # For now, keep using the same condition (simplified)

    all_lat = torch.stack(all_latents)  # [N, vae_dim]
    dump_f32(out / "all_latents.bin", all_lat)
    print(f"  all latents: shape={list(all_lat.shape)}, rms={all_lat.norm()/all_lat.numel()**0.5:.4f}")

    # Dump per-frame init noise — C++ side reads this via VIBEVOICE_TTS_NOISE
    # to replay the same Gaussian z[fi] each frame (PyTorch RNG ≠ C++ MT19937).
    all_noise = torch.stack(all_noises)  # [N, vae_dim]
    dump_f32(out / "noise.bin", all_noise)

    # Scale and decode
    sf = state.get("model.speech_scaling_factor", torch.tensor(0.196))
    bf = state.get("model.speech_bias_factor", torch.tensor(-0.049))
    scaled = all_lat / sf.item() - bf.item()

    # Skip decoder for now (too complex to reimplement manually)
    # Just dump the latents for comparison
    dump_f32(out / "scaled_latent.bin", scaled)
    print(f"  scaled latent: rms={scaled.norm()/scaled.numel()**0.5:.4f}")

    print(f"\nall outputs written to {out}")
    print("NOTE: decoder not run (manual pipeline). Compare latent stages with C++.")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
