#!/usr/bin/env python3
"""With-voice Python reference for VibeVoice-Realtime-0.5B TTS.

Loads model weights from the original safetensors (F32), the voice prompt
from a CrispASR voice .gguf (F16 KV caches), and reproduces the same
text-window / speech-window interleaving the C++ runtime does. Dumps
frame-0 stages so they can be compared to crispasr's VIBEVOICE_TTS_DUMP
output via tools/diff_vibevoice_tts.py.

This covers the path the user actually exercises:
  crispasr --tts "..." -m vibevoice-realtime-...gguf --voice <voice.gguf>

Stages dumped:
  text_token_ids.bin         int32  tokens (with trailing \\n, matching C++)
  base_lm_hidden.bin         f32 [n_text, d_lm]  base LM with voice.lm KV
  tts_lm_hidden_frame0.bin   f32 [d_lm]         TTS-LM hidden after window 1
  noise_frame0.bin           f32 [vae_dim]      init noise for frame 0
  pred_v_step0_frame0.bin    f32 [vae_dim]      v-prediction step 0 (CFG'd)
  speech_latent_frame0.bin   f32 [vae_dim]      denoised latent frame 0
  acoustic_embed_frame0.bin  f32 [d_lm]         acoustic connector(z_0)
  noise.bin                  f32 [N, vae_dim]   per-frame init noise (only [0] is real here)

The frame-0 path is fully reference. Frames > 0 are not produced — that
needs the autoregressive loop and is a follow-up.

Usage:
  python tools/vibevoice_tts_ref_voice.py \\
      --text "Hello, how are you today?" \\
      --voice /path/to/davis.gguf \\
      --safetensors /path/to/microsoft--VibeVoice-Realtime-0.5B/.../model.safetensors \\
      --output-dir /tmp/vv_rt_ref_voice
"""

from __future__ import annotations

import argparse
import os
import struct
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def dump_f32(path: Path, arr):
    if isinstance(arr, torch.Tensor):
        arr = arr.detach().cpu().float().numpy()
    arr = np.ascontiguousarray(arr, dtype=np.float32)
    arr.tofile(str(path))
    print(f"  wrote {path.name}: {arr.shape} float32 ({arr.nbytes} bytes)")


def dump_i32(path: Path, arr):
    arr = np.ascontiguousarray(np.array(arr, dtype=np.int32))
    arr.tofile(str(path))
    print(f"  wrote {path.name}: {arr.shape} int32 ({arr.nbytes} bytes)")


def load_voice_gguf(path: str):
    """Returns dict { 'lm': {seq_len, n_layers, k[il], v[il]}, 'tts_lm': ..., 'neg_lm': ..., 'neg_tts_lm': ... }
    Each k/v is a torch.float32 tensor of shape [n_kv_heads, seq_len, head_dim].
    """
    import gguf
    r = gguf.GGUFReader(path)

    def kv_int(key):
        f = r.fields[key]
        return int(f.parts[-1].tolist()[0])

    voice = {}
    for prefix in ["lm", "tts_lm", "neg_lm", "neg_tts_lm"]:
        seq_len = kv_int(f"voice.{prefix}.seq_len")
        n_layers = kv_int(f"voice.{prefix}.n_layers")
        k_per_layer = [None] * n_layers
        v_per_layer = [None] * n_layers
        for t in r.tensors:
            if not t.name.startswith(f"voice.{prefix}."):
                continue
            # voice.{prefix}.{il}.{k|v}
            tail = t.name[len(f"voice.{prefix}.") :]
            il_str, kv_type = tail.rsplit(".", 1)
            il = int(il_str)
            arr = np.frombuffer(t.data.tobytes(), dtype=np.float16).reshape(t.data.shape)
            tensor = torch.from_numpy(arr.astype(np.float32))
            if kv_type == "k":
                k_per_layer[il] = tensor
            else:
                v_per_layer[il] = tensor
        voice[prefix] = {
            "seq_len": seq_len,
            "n_layers": n_layers,
            "k": k_per_layer,
            "v": v_per_layer,
        }
        print(f"  voice.{prefix}: {n_layers} layers, seq_len={seq_len}, head_shape={tuple(k_per_layer[0].shape)}")
    return voice


def apply_rope(x, cos_r, sin_r, head_dim):
    # x: [n_heads, T, head_dim]
    x1 = x[..., : head_dim // 2]
    x2 = x[..., head_dim // 2 :]
    return torch.cat([x1 * cos_r - x2 * sin_r, x2 * cos_r + x1 * sin_r], dim=-1)


def build_rope(head_dim, positions, theta=1000000.0):
    # NEOX style — positions [T] long
    half = head_dim // 2
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    angles = torch.outer(positions.float(), freqs)
    cos_r = angles.cos()
    sin_r = angles.sin()
    return cos_r, sin_r


def run_qwen2_layers(state, prefix_pat, n_layers, hidden, positions,
                     past_k=None, past_v=None, head_dim=64, n_heads=14, n_kv_heads=2,
                     rope_theta=1000000.0, eps=1e-6):
    """Run Qwen2 transformer layers. hidden: [T, d_lm].
    past_k, past_v: lists per layer of [n_kv_heads, past_T, head_dim] or None.
    Returns final hidden [T, d_lm], lists of new K, V per layer at current positions [n_kv_heads, T, head_dim].
    """
    T = hidden.shape[0]
    d_lm = hidden.shape[1]

    def G(name):
        if name in state:
            return state[name]
        raise KeyError(f"missing tensor: {name}")

    def Gopt(name):
        return state.get(name)

    cos_r, sin_r = build_rope(head_dim, positions, theta=rope_theta)
    new_k = []
    new_v = []
    n_rep = n_heads // n_kv_heads

    h = hidden
    for il in range(n_layers):
        prefix = prefix_pat.format(il=il)
        residual = h

        # Pre-norm
        norm_w = G(f"{prefix}.input_layernorm.weight")
        h_norm = F.rms_norm(h, (d_lm,), norm_w, eps=eps)

        # Q/K/V proj
        Q = F.linear(h_norm, G(f"{prefix}.self_attn.q_proj.weight"), Gopt(f"{prefix}.self_attn.q_proj.bias"))
        K = F.linear(h_norm, G(f"{prefix}.self_attn.k_proj.weight"), Gopt(f"{prefix}.self_attn.k_proj.bias"))
        V = F.linear(h_norm, G(f"{prefix}.self_attn.v_proj.weight"), Gopt(f"{prefix}.self_attn.v_proj.bias"))

        Q = Q.view(T, n_heads, head_dim).transpose(0, 1)        # [n_heads, T, head_dim]
        K = K.view(T, n_kv_heads, head_dim).transpose(0, 1)     # [n_kv_heads, T, head_dim]
        V = V.view(T, n_kv_heads, head_dim).transpose(0, 1)     # [n_kv_heads, T, head_dim]

        Q = apply_rope(Q, cos_r, sin_r, head_dim)
        K = apply_rope(K, cos_r, sin_r, head_dim)

        new_k.append(K.clone())
        new_v.append(V.clone())

        # Concat with past
        if past_k is not None and past_k[il] is not None:
            K_full = torch.cat([past_k[il], K], dim=1)
            V_full = torch.cat([past_v[il], V], dim=1)
        else:
            K_full = K
            V_full = V
        Lk = K_full.shape[1]
        past_T = Lk - T

        # GQA: expand K, V
        if n_rep > 1:
            K_full = K_full.unsqueeze(1).expand(-1, n_rep, -1, -1).reshape(n_heads, Lk, head_dim)
            V_full = V_full.unsqueeze(1).expand(-1, n_rep, -1, -1).reshape(n_heads, Lk, head_dim)

        # Causal mask: each query at time t (in the new tokens, indices 0..T-1) can attend
        # to all past_T keys plus its own current keys 0..t.
        scale = 1.0 / (head_dim ** 0.5)
        attn = (Q @ K_full.transpose(-2, -1)) * scale  # [n_heads, T, Lk]
        mask = torch.zeros(T, Lk)
        for q in range(T):
            mask[q, past_T + q + 1 :] = float("-inf")
        attn = attn + mask
        attn = torch.softmax(attn, dim=-1)
        attn_out = (attn @ V_full).transpose(0, 1).contiguous().view(T, d_lm)
        attn_out = F.linear(attn_out, G(f"{prefix}.self_attn.o_proj.weight"))
        h = residual + attn_out

        # FFN: SwiGLU
        residual = h
        norm_w2 = G(f"{prefix}.post_attention_layernorm.weight")
        h_norm2 = F.rms_norm(h, (d_lm,), norm_w2, eps=eps)
        gate = F.linear(h_norm2, G(f"{prefix}.mlp.gate_proj.weight"))
        up = F.linear(h_norm2, G(f"{prefix}.mlp.up_proj.weight"))
        ffn = F.linear(F.silu(gate) * up, G(f"{prefix}.mlp.down_proj.weight"))
        h = residual + ffn

    return h, new_k, new_v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", required=True)
    ap.add_argument("--voice", required=True, help="voice .gguf path")
    ap.add_argument("--safetensors", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--cfg-scale", type=float, default=3.0)
    ap.add_argument("--num-steps", type=int, default=20)
    ap.add_argument("--n-frames", type=int, default=12,
                    help="upper bound on frames to allocate noise for (we only run frame 0)")
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Tokenizer (Qwen2.5 fallback — VibeVoice has no tokenizer of its own)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B", trust_remote_code=True)
    print("loaded tokenizer Qwen/Qwen2.5-0.5B")

    # Tokenize WITH trailing \n — VibeVoice convention (LEARNINGS.md bug #17).
    text_with_nl = args.text + "\n"
    ids = tokenizer.encode(text_with_nl, add_special_tokens=False)
    print(f"  tokens (text + '\\n'): {ids}")
    dump_i32(out / "text_token_ids.bin", ids)

    # Load weights
    from safetensors.torch import load_file
    state = load_file(args.safetensors)
    state = {k: v.float() for k, v in state.items()}
    print(f"  loaded {len(state)} tensors from {args.safetensors}")

    # Detect actual n layers (Realtime base LM has 4 of 24)
    base_n_layers = 0
    while f"model.language_model.layers.{base_n_layers}.input_layernorm.weight" in state:
        base_n_layers += 1
    tts_n_layers = 0
    while f"model.tts_language_model.layers.{tts_n_layers}.input_layernorm.weight" in state:
        tts_n_layers += 1
    print(f"  base LM layers: {base_n_layers}, TTS LM layers: {tts_n_layers}")

    # Hyperparams (match config.json)
    d_lm = 896
    n_heads = 14
    n_kv_heads = 2
    head_dim = d_lm // n_heads
    rope_theta = 1000000.0
    vae_dim = 64

    # Load voice
    voice = load_voice_gguf(args.voice)
    vsl_lm = voice["lm"]["seq_len"]
    vsl_tts = voice["tts_lm"]["seq_len"]
    vsl_neg_lm = voice["neg_lm"]["seq_len"]
    vsl_neg_tts = voice["neg_tts_lm"]["seq_len"]
    print(f"  voice seq_lens: lm={vsl_lm}, tts={vsl_tts}, neg_lm={vsl_neg_lm}, neg_tts={vsl_neg_tts}")

    n_text = len(ids)

    # === Step 1: Base LM with voice.lm KV ===
    # Embed text via lm.tok_emb (base LM embedding)
    base_emb_w = state["model.language_model.embed_tokens.weight"]
    ids_t = torch.tensor(ids, dtype=torch.long)
    text_embeds = base_emb_w[ids_t]  # [n_text, d_lm]
    base_positions = torch.arange(vsl_lm, vsl_lm + n_text)
    base_hidden, _, _ = run_qwen2_layers(
        state, "model.language_model.layers.{il}", base_n_layers,
        text_embeds, base_positions,
        past_k=voice["lm"]["k"], past_v=voice["lm"]["v"],
        head_dim=head_dim, n_heads=n_heads, n_kv_heads=n_kv_heads,
        rope_theta=rope_theta,
    )
    # Realtime base LM has no final norm
    dump_f32(out / "base_lm_hidden.bin", base_hidden)
    print(f"  base LM done: shape={list(base_hidden.shape)}, rms={base_hidden.norm()/base_hidden.numel()**0.5:.4f}")

    # Also compute neg base LM hidden (on IMAGE_PAD tokens) for the negative path.
    # Uses voice.neg_lm KV (typically 1 token) — NOT voice.lm — at positions [vsl_neg_lm..].
    IMAGE_PAD = 151655
    pad_ids = torch.full((n_text,), IMAGE_PAD, dtype=torch.long)
    pad_embeds = base_emb_w[pad_ids]
    neg_base_positions = torch.arange(vsl_neg_lm, vsl_neg_lm + n_text)
    neg_base_hidden, _, _ = run_qwen2_layers(
        state, "model.language_model.layers.{il}", base_n_layers,
        pad_embeds, neg_base_positions,
        past_k=voice["neg_lm"]["k"], past_v=voice["neg_lm"]["v"],
        head_dim=head_dim, n_heads=n_heads, n_kv_heads=n_kv_heads,
        rope_theta=rope_theta,
    )
    dump_f32(out / "neg_base_lm_hidden.bin", neg_base_hidden)
    print(f"  neg base LM done (voice.neg_lm KV, pos {vsl_neg_lm}-{vsl_neg_lm+n_text-1}): "
          f"rms={neg_base_hidden.norm()/neg_base_hidden.numel()**0.5:.4f}")

    # === Step 2: TTS LM text window 1 (5 tokens, with type=1 added) ===
    type_emb = state["model.tts_input_types.weight"]   # [2, d_lm]
    type_text = type_emb[1]
    type_speech = type_emb[0]

    TEXT_WINDOW = 5
    win1 = min(n_text, TEXT_WINDOW)
    win1_input = base_hidden[:win1] + type_text  # [win1, d_lm]

    pos_positions = torch.arange(vsl_tts, vsl_tts + win1)
    tts_hidden, _, _ = run_qwen2_layers(
        state, "model.tts_language_model.layers.{il}", tts_n_layers,
        win1_input, pos_positions,
        past_k=voice["tts_lm"]["k"], past_v=voice["tts_lm"]["v"],
        head_dim=head_dim, n_heads=n_heads, n_kv_heads=n_kv_heads,
        rope_theta=rope_theta,
    )
    # TTS LM has final norm
    final_norm_w = state["model.tts_language_model.norm.weight"]
    tts_hidden = F.rms_norm(tts_hidden, (d_lm,), final_norm_w, eps=1e-6)
    tts_condition = tts_hidden[-1]   # last token = condition for first speech frame
    dump_f32(out / "tts_lm_hidden_frame0.bin", tts_condition)
    print(f"  TTS LM (win1, pos {vsl_tts}-{vsl_tts+win1-1}) condition rms={tts_condition.norm():.4f}")

    # Initial negative condition (per the official inference loop in
    # microsoft/VibeVoice modeling_vibevoice_streaming_inference.py): the prefilled
    # neg_tts_lm.last_hidden_state from running TTS LM forward on a SINGLE <|image_pad|>
    # token at pos 0, with NO past KV. The official code does NOT advance neg through
    # the text window — only positive does.
    #
    # The prefill is:
    #   1. neg base LM forward(IMAGE_PAD at pos 0, no past KV) -> neg_base_pad_hidden [1, d_lm]
    #   2. neg TTS LM forward(IMAGE_PAD at pos 0, lm_last_hidden_state=neg_base_pad_hidden,
    #      tts_text_masks=1, no past KV) -> last_hidden_state[-1] is neg_condition.
    pad_emb_for_neg = base_emb_w[torch.tensor([IMAGE_PAD], dtype=torch.long)]  # [1, d_lm]
    neg_base_pad_pos = torch.tensor([0], dtype=torch.long)
    neg_base_pad_hidden, _, _ = run_qwen2_layers(
        state, "model.language_model.layers.{il}", base_n_layers,
        pad_emb_for_neg, neg_base_pad_pos,
        past_k=None, past_v=None,
        head_dim=head_dim, n_heads=n_heads, n_kv_heads=n_kv_heads,
        rope_theta=rope_theta,
    )
    # Realtime base LM has no final norm — pass raw output as lm_last_hidden_state.
    # Now run the TTS LM prefill: input = tts_emb(IMAGE_PAD), tail-replace with
    # neg_base_pad_hidden, add type_text. The official code splices lm_last_hidden_state
    # into the tail; for a single token this overwrites the entire embed.
    tts_emb_w = state["model.tts_language_model.embed_tokens.weight"]
    pad_tts_emb = tts_emb_w[torch.tensor([IMAGE_PAD], dtype=torch.long)]  # [1, d_lm]
    # Tail-replace (entire single token) and add type_text
    neg_prefill_input = neg_base_pad_hidden + type_text  # [1, d_lm]
    neg_prefill_pos = torch.tensor([0], dtype=torch.long)
    neg_prefill_hidden, _, _ = run_qwen2_layers(
        state, "model.tts_language_model.layers.{il}", tts_n_layers,
        neg_prefill_input, neg_prefill_pos,
        past_k=None, past_v=None,
        head_dim=head_dim, n_heads=n_heads, n_kv_heads=n_kv_heads,
        rope_theta=rope_theta,
    )
    neg_prefill_hidden = F.rms_norm(neg_prefill_hidden, (d_lm,), final_norm_w, eps=1e-6)
    neg_condition = neg_prefill_hidden[-1]
    dump_f32(out / "tts_neg_condition_frame0.bin", neg_condition)
    print(f"  neg TTS LM condition (prefill IMAGE_PAD) rms={neg_condition.norm():.4f}")

    # === Step 3: Frame 0 diffusion with CFG ===
    # Cosine beta schedule (matching diffusers DPMSolverMultistepScheduler)
    offset = 0.008
    num_train = 1000
    alphas_cumprod = np.zeros(num_train)
    a_prod = 1.0
    def alpha_bar(t):
        frac = (t + offset) / (1.0 + offset)
        return np.cos(frac * np.pi * 0.5) ** 2
    for i in range(num_train):
        t1 = i / num_train
        t2 = (i + 1) / num_train
        beta = 1.0 - alpha_bar(t2) / alpha_bar(t1)
        if beta > 0.999:
            beta = 0.999
        a_prod *= 1.0 - beta
        alphas_cumprod[i] = a_prod

    # Match diffusers DPMSolverMultistepScheduler.set_timesteps with timestep_spacing="linspace":
    #   ts = linspace(0, num_train - 1, num_steps + 1).round()[::-1][:-1]
    # Last timestep is non-zero (e.g. 50 for 20 steps over 1000) — final t=0 cleanup is
    # implicit via the appended sigma=0 (final_sigmas_type="zero").
    timesteps = (
        np.linspace(0, num_train - 1, args.num_steps + 1).round()[::-1][:-1].astype(int)
    )

    # Pred-head weights
    noisy_w = state["model.prediction_head.noisy_images_proj.weight"]
    cond_w = state["model.prediction_head.cond_proj.weight"]
    t_emb_0_w = state["model.prediction_head.t_embedder.mlp.0.weight"]
    t_emb_2_w = state["model.prediction_head.t_embedder.mlp.2.weight"]
    final_adaln_w = state["model.prediction_head.final_layer.adaLN_modulation.1.weight"]
    final_lin_w = state["model.prediction_head.final_layer.linear.weight"]

    def sinusoidal_embed(t_val, dim=256):
        half = dim // 2
        freqs = torch.exp(-np.log(10000) * torch.arange(half, dtype=torch.float32) / half)
        ang = t_val * freqs
        return torch.cat([torch.cos(ang), torch.sin(ang)])

    def run_pred_head(z, condition, t_val):
        t_sin = sinusoidal_embed(t_val)
        t_emb = F.linear(t_sin, t_emb_0_w)
        t_emb = F.silu(t_emb)
        t_emb = F.linear(t_emb, t_emb_2_w)
        x = F.linear(z, noisy_w)
        c = F.linear(condition, cond_w) + t_emb
        # AdaLN expects silu(c) before the modulation projection (LEARNINGS bug #16)
        c_silu = F.silu(c)
        for i in range(4):
            pfx = f"model.prediction_head.layers.{i}"
            adaln_out = F.linear(c_silu, state[f"{pfx}.adaLN_modulation.1.weight"])
            shift, scale, gate = adaln_out.chunk(3, dim=-1)
            h = F.rms_norm(x, (d_lm,), state[f"{pfx}.norm.weight"], eps=1e-5)
            h = h * (1 + scale) + shift
            g_proj = F.linear(h, state[f"{pfx}.ffn.gate_proj.weight"])
            u_proj = F.linear(h, state[f"{pfx}.ffn.up_proj.weight"])
            h = F.linear(F.silu(g_proj) * u_proj, state[f"{pfx}.ffn.down_proj.weight"])
            x = x + gate * h
        adaln_out = F.linear(c_silu, final_adaln_w)
        shift, scale = adaln_out.chunk(2, dim=-1)
        h = F.rms_norm(x, (d_lm,), eps=1e-5)
        h = h * (1 + scale) + shift
        return F.linear(h, final_lin_w)

    # DPM-Solver++ helpers (v-prediction → x0)
    def v_to_x0(z, v, alpha_t):
        sa = np.sqrt(alpha_t)
        s1ma = np.sqrt(1.0 - alpha_t)
        return sa * z - s1ma * v

    def dpm_first_order(step_idx, z, x0):
        t = timesteps[step_idx]
        t_prev = timesteps[step_idx + 1] if step_idx + 1 < args.num_steps else -1
        sigma_t = np.sqrt(1 - alphas_cumprod[t])
        sigma_prev = np.sqrt(1 - alphas_cumprod[t_prev]) if t_prev >= 0 else 0.0
        alpha_t = np.sqrt(alphas_cumprod[t])
        alpha_prev = np.sqrt(alphas_cumprod[t_prev]) if t_prev >= 0 else 1.0
        lam_t = np.log(alpha_t / sigma_t)
        lam_prev = np.log(alpha_prev / sigma_prev) if t_prev >= 0 else 20.0
        h = lam_prev - lam_t
        return (sigma_prev / sigma_t) * z - alpha_prev * (np.exp(-h) - 1.0) * x0

    def dpm_second_order(step_idx, z, x0_cur, x0_prev, prev_step_idx):
        t = timesteps[step_idx]
        t_prev = timesteps[step_idx + 1] if step_idx + 1 < args.num_steps else -1
        s = timesteps[prev_step_idx]
        sigma_t = np.sqrt(1 - alphas_cumprod[t])
        sigma_s = np.sqrt(1 - alphas_cumprod[s])
        sigma_prev = np.sqrt(1 - alphas_cumprod[t_prev]) if t_prev >= 0 else 0.0
        alpha_t = np.sqrt(alphas_cumprod[t])
        alpha_s = np.sqrt(alphas_cumprod[s])
        alpha_prev = np.sqrt(alphas_cumprod[t_prev]) if t_prev >= 0 else 1.0
        lam_t = np.log(alpha_t / sigma_t)
        lam_s = np.log(alpha_s / sigma_s)
        lam_prev = np.log(alpha_prev / sigma_prev) if t_prev >= 0 else 20.0
        h = lam_prev - lam_t
        h0 = lam_t - lam_s
        r = h0 / h
        eh = np.exp(-h) - 1.0
        D0 = x0_cur
        D1 = (1.0 / r) * (x0_cur - x0_prev)
        return (sigma_prev / sigma_t) * z - alpha_prev * eh * D0 - 0.5 * alpha_prev * eh * D1

    # === Frame 0 ===
    torch.manual_seed(42)
    # Allocate noise table for n_frames; we only use frame 0 in reference
    all_noises = []
    for fi in range(args.n_frames):
        z0 = torch.randn(vae_dim)
        all_noises.append(z0.detach().clone())
    z = all_noises[0].clone()
    dump_f32(out / "noise_frame0.bin", z)
    dump_f32(out / "noise.bin", torch.stack(all_noises))

    prev_x0 = torch.zeros(vae_dim)
    for step in range(args.num_steps):
        t_val = float(timesteps[step])
        v_pos = run_pred_head(z, tts_condition, t_val)
        v_neg = run_pred_head(z, neg_condition, t_val)
        v_cfg = v_neg + args.cfg_scale * (v_pos - v_neg)
        if step == 0:
            dump_f32(out / "pred_v_step0_frame0.bin", v_cfg)

        # v-prediction → x0
        t_cur = timesteps[step]
        alpha_t = alphas_cumprod[t_cur]
        x0 = v_to_x0(z.detach().numpy(), v_cfg.detach().numpy(), alpha_t)
        x0_t = torch.from_numpy(x0.astype(np.float32))

        # DPM step (1st order at step 0 and last step, else 2nd order midpoint)
        is_last = step == args.num_steps - 1
        if step == 0 or is_last:
            z = torch.from_numpy(dpm_first_order(step, z.detach().numpy(), x0).astype(np.float32))
        else:
            z = torch.from_numpy(
                dpm_second_order(step, z.detach().numpy(), x0, prev_x0.detach().numpy(), step - 1).astype(np.float32)
            )
        prev_x0 = x0_t

    dump_f32(out / "speech_latent_frame0.bin", z)
    print(f"  frame 0 latent rms={z.norm()/vae_dim**0.5:.4f}")

    # === Connector → acoustic embed ===
    fc1_w = state["model.acoustic_connector.fc1.weight"]
    fc1_b = state.get("model.acoustic_connector.fc1.bias")
    norm_w = state.get("model.acoustic_connector.norm.weight")
    fc2_w = state["model.acoustic_connector.fc2.weight"]
    fc2_b = state.get("model.acoustic_connector.fc2.bias")
    ae = F.linear(z, fc1_w, fc1_b)
    ae = F.rms_norm(ae, (d_lm,), norm_w, eps=1e-6)
    ae = F.linear(ae, fc2_w, fc2_b)
    dump_f32(out / "acoustic_embed_frame0.bin", ae)
    print(f"  acoustic embed rms={ae.norm()/d_lm**0.5:.4f}")

    print(f"\nfinished. dumps in {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
