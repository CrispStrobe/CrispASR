#!/usr/bin/env python3
"""Dump Kyutai STT intermediates at every stage boundary for diff-testing.

Usage:
  python tools/dump_kyutai_stt_reference.py \
      --model kyutai/stt-1b-en_fr \
      --audio samples/jfk.wav \
      --outdir /tmp/kyutai_ref
"""

import argparse
import json
import os
import sys

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="HF model ID or local path")
    parser.add_argument("--audio", required=True, help="Audio file path")
    parser.add_argument("--outdir", required=True, help="Output directory for .npy files")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    import torch
    from safetensors import safe_open
    from huggingface_hub import hf_hub_download
    import sentencepiece as spm
    import scipy.io.wavfile as wav_io
    import scipy.signal

    # Load config
    cache_dir = os.environ.get("HF_HOME", None)
    if cache_dir:
        cache_dir += "/hub"

    if os.path.isdir(args.model):
        base = args.model
        config_path = os.path.join(base, "config.json")
    else:
        config_path = hf_hub_download(args.model, "config.json", cache_dir=cache_dir)
        base = os.path.dirname(config_path)

    with open(config_path) as f:
        config = json.load(f)

    mimi_name = config.get("mimi_name", "mimi-pytorch-e351c8d8@125.safetensors")
    tokenizer_name = config.get("tokenizer_name", "tokenizer_en_fr_audio_8000.model")

    if not os.path.isdir(args.model):
        for fname in ["model.safetensors", mimi_name, tokenizer_name]:
            hf_hub_download(args.model, fname, cache_dir=cache_dir)

    lm_path = os.path.join(base, "model.safetensors")
    mimi_path = os.path.join(base, mimi_name)
    tok_path = os.path.join(base, tokenizer_name)

    # Load weights
    print("Loading weights...")
    lm_sd = {}
    with safe_open(lm_path, framework="pt") as f:
        for key in f.keys():
            lm_sd[key] = f.get_tensor(key).float()

    mimi_sd = {}
    with safe_open(mimi_path, framework="pt") as f:
        for key in f.keys():
            mimi_sd[key] = f.get_tensor(key).float()

    # Compute codebook embeddings
    codebooks = {}
    for key in list(mimi_sd.keys()):
        if key.endswith("._codebook.embedding_sum"):
            prefix = key[:-len("embedding_sum")]
            usage = mimi_sd[prefix + "cluster_usage"]
            usage = torch.clamp(usage, min=1e-5)
            codebooks[prefix + "embedding"] = mimi_sd[key] / usage.unsqueeze(-1)

    # Load tokenizer
    sp = spm.SentencePieceProcessor()
    sp.Load(tok_path)

    # Load audio
    print(f"Loading audio: {args.audio}")
    sr, wav_data = wav_io.read(args.audio)
    if wav_data.dtype == np.int16:
        wav_data = wav_data.astype(np.float32) / 32768.0
    elif wav_data.dtype == np.int32:
        wav_data = wav_data.astype(np.float32) / 2147483648.0
    if len(wav_data.shape) > 1:
        wav_data = wav_data.mean(axis=1)
    if sr != 24000:
        n_out = int(len(wav_data) * 24000 / sr)
        wav_data = scipy.signal.resample(wav_data, n_out)
    pcm = torch.from_numpy(wav_data.astype(np.float32))
    print(f"  PCM: {pcm.shape} @ 24kHz")
    np.save(os.path.join(args.outdir, "00_pcm_24k.npy"), pcm.numpy())

    # =========================================================================
    # Stage 1: SEANet encoder
    # =========================================================================
    print("\n=== SEANet Encoder ===")

    def get_conv(prefix):
        w = mimi_sd[prefix + ".weight"]
        b = mimi_sd.get(prefix + ".bias")
        return w, b

    def streaming_conv1d_fwd(x, prefix, stride=1):
        """Causal (left) padding conv1d, matching moshi StreamingConv1d."""
        w, b = get_conv(prefix)
        kernel_size = w.shape[2]
        # Causal padding: pad LEFT with (kernel_size - stride) zeros
        pad_left = kernel_size - stride
        if pad_left > 0:
            x = torch.nn.functional.pad(x, (pad_left, 0))
        out = torch.nn.functional.conv1d(x, w, b, stride=stride, padding=0)
        return out

    def resblock_fwd(x, model_idx):
        prefix1 = f"encoder.model.{model_idx}.block.1.conv.conv"
        prefix3 = f"encoder.model.{model_idx}.block.3.conv.conv"
        h = torch.nn.functional.elu(x)
        h = streaming_conv1d_fwd(h, prefix1, stride=1)
        h = torch.nn.functional.elu(h)
        h = streaming_conv1d_fwd(h, prefix3, stride=1)
        return x + h

    # Pad to multiple of frame_size (1920) — matches official Mimi behavior
    frame_size = 1920  # SEANet total stride: 4*5*6*8 * 2(downsample) = 1920
    # Actually frame_size for SEANet only is 4*5*6*8 = 960, times downsample stride 2 = 1920
    extra = (frame_size - len(pcm) % frame_size) % frame_size
    if extra > 0:
        pcm = torch.cat([pcm, torch.zeros(extra)])
        print(f"  Padded PCM: {pcm.shape} (added {extra} zeros)")

    # Input: [1, 1, T]
    x = pcm.unsqueeze(0).unsqueeze(0)  # [1, 1, T]

    # model.0: Conv1d(1→64, k=7, s=1)
    x = streaming_conv1d_fwd(x, "encoder.model.0.conv.conv", stride=1)
    print(f"  After conv_init: {x.shape}")
    np.save(os.path.join(args.outdir, "01_seanet_conv_init.npy"), x[0].detach().numpy())

    strides = [4, 5, 6, 8]
    kernels = [8, 10, 12, 16]
    resblock_idxs = [1, 4, 7, 10]
    stride_idxs = [3, 6, 9, 12]

    for i in range(4):
        x = resblock_fwd(x, resblock_idxs[i])
        print(f"  After resblock[{i}]: {x.shape}")
        x = torch.nn.functional.elu(x)
        x = streaming_conv1d_fwd(x, f"encoder.model.{stride_idxs[i]}.conv.conv",
                                  stride=strides[i])
        print(f"  After stride_conv[{i}]: {x.shape}")

    # model.14: Conv1d(1024→512, k=3, s=1)
    x = torch.nn.functional.elu(x)
    x = streaming_conv1d_fwd(x, "encoder.model.14.conv.conv", stride=1)
    print(f"  After conv_final: {x.shape}")  # [1, 512, T_enc]
    np.save(os.path.join(args.outdir, "02_seanet_output.npy"), x[0].detach().numpy())

    # =========================================================================
    # Stage 2: Encoder transformer
    # =========================================================================
    print("\n=== Encoder Transformer ===")

    # x is [1, 512, T_enc] in PyTorch layout
    # Transformer works on [T, batch, dim] or we can use [batch, dim, T]
    # Actually, moshi's transformer operates on [batch, dim, T] layout
    # (treating the feature map like a sequence)

    # Let's work in [T, dim] layout for simplicity
    enc = x.squeeze(0).permute(1, 0)  # [T_enc, 512]
    T_enc = enc.shape[0]
    dim = enc.shape[1]  # 512
    n_heads = 8
    head_dim = dim // n_heads  # 64

    print(f"  Transformer input: [{T_enc}, {dim}]")

    for li in range(8):
        prefix = f"encoder_transformer.transformer.layers.{li}"
        residual = enc

        # LayerNorm
        norm_w = mimi_sd[f"{prefix}.norm1.weight"]
        norm_b = mimi_sd.get(f"{prefix}.norm1.bias")
        h = torch.nn.functional.layer_norm(enc, [dim], norm_w, norm_b)

        # QKV projection
        qkv_w = mimi_sd[f"{prefix}.self_attn.in_proj_weight"]  # [3*dim, dim]
        qkv = h @ qkv_w.T  # [T, 3*dim]
        Q, K, V = qkv.split(dim, dim=-1)  # each [T, dim]

        # Reshape to [T, n_heads, head_dim]
        Q = Q.view(T_enc, n_heads, head_dim)
        K = K.view(T_enc, n_heads, head_dim)
        V = V.view(T_enc, n_heads, head_dim)

        # RoPE (interleaved layout: [r0,i0,r1,i1,...])
        positions = torch.arange(T_enc)
        ds = torch.arange(head_dim // 2, dtype=torch.float32)
        freqs = torch.exp(ds * (-np.log(10000.0) * 2 / head_dim))
        angles = positions.unsqueeze(1) * freqs.unsqueeze(0)  # [T, head_dim/2]
        cos_vals = torch.cos(angles)
        sin_vals = torch.sin(angles)

        def apply_rope_interleaved(x):
            # x: [T, n_heads, head_dim] — interleaved [r0,i0,r1,i1,...]
            x = x.view(*x.shape[:-1], head_dim // 2, 2)
            xr, xi = x[..., 0].float(), x[..., 1].float()
            cos_v = cos_vals.unsqueeze(1)  # [T, 1, hd/2]
            sin_v = sin_vals.unsqueeze(1)
            or_ = xr * cos_v - xi * sin_v
            oi = xr * sin_v + xi * cos_v
            return torch.stack([or_.to(x.dtype), oi.to(x.dtype)], dim=-1).view(*x.shape[:-2], head_dim)

        Q = apply_rope_interleaved(Q)
        K = apply_rope_interleaved(K)

        # Attention with causal mask + context window
        Q_t = Q.permute(1, 0, 2).unsqueeze(0)  # [1, n_heads, T, head_dim]
        K_t = K.permute(1, 0, 2).unsqueeze(0)
        V_t = V.permute(1, 0, 2).unsqueeze(0)
        # Causal mask with context=250
        pos_q = torch.arange(T_enc).view(-1, 1)
        pos_k = torch.arange(T_enc).view(1, -1)
        delta = pos_q - pos_k
        attn_bias = (delta >= 0) & (delta < 250)  # causal + context window
        attn_bias = attn_bias.unsqueeze(0).unsqueeze(0)  # [1, 1, T, T]
        attn_out = torch.nn.functional.scaled_dot_product_attention(
            Q_t, K_t, V_t, attn_mask=attn_bias, dropout_p=0.0)
        attn_out = attn_out.squeeze(0).permute(1, 0, 2).reshape(T_enc, dim)  # [T, dim]

        # Output projection
        out_w = mimi_sd[f"{prefix}.self_attn.out_proj.weight"]  # [dim, dim]
        attn_out = attn_out @ out_w.T

        # Layer scale 1
        ls1 = mimi_sd[f"{prefix}.layer_scale_1.scale"]  # [dim]
        attn_out = attn_out * ls1

        enc = residual + attn_out

        # FFN
        residual = enc
        norm2_w = mimi_sd[f"{prefix}.norm2.weight"]
        norm2_b = mimi_sd.get(f"{prefix}.norm2.bias")
        h = torch.nn.functional.layer_norm(enc, [dim], norm2_w, norm2_b)

        ffn_up_w = mimi_sd[f"{prefix}.linear1.weight"]  # [4*dim, dim]
        ffn_down_w = mimi_sd[f"{prefix}.linear2.weight"]  # [dim, 4*dim]
        h = h @ ffn_up_w.T
        h = torch.nn.functional.gelu(h)
        h = h @ ffn_down_w.T

        ls2 = mimi_sd[f"{prefix}.layer_scale_2.scale"]
        h = h * ls2

        enc = residual + h

        if li == 0:
            np.save(os.path.join(args.outdir, "03_enc_tfm_layer0.npy"), enc.detach().numpy())

    print(f"  Transformer output: {enc.shape}")
    np.save(os.path.join(args.outdir, "04_enc_tfm_output.npy"), enc.detach().numpy())

    # =========================================================================
    # Stage 3: Downsample
    # =========================================================================
    print("\n=== Downsample ===")
    # Back to [1, dim, T] for conv1d
    x = enc.permute(1, 0).unsqueeze(0)  # [1, 512, T_enc]
    # Downsample: Conv1d(512→512, k=4, s=2, groups=512 possibly?)
    ds_w = mimi_sd["downsample.conv.conv.conv.weight"]
    ds_b = mimi_sd.get("downsample.conv.conv.conv.bias")
    print(f"  Downsample weight shape: {ds_w.shape}")
    groups = 1
    if ds_w.shape[1] == 1 and ds_w.shape[0] == 512:
        groups = 512  # depthwise
    # Causal padding: kernel=4, stride=2 → pad_left = 4-2 = 2
    x = torch.nn.functional.pad(x, (2, 0))
    x = torch.nn.functional.conv1d(x, ds_w, ds_b, stride=2, padding=0, groups=groups)
    T_frames = x.shape[2]
    print(f"  After downsample: {x.shape} → T_frames={T_frames}")
    np.save(os.path.join(args.outdir, "05_downsampled.npy"), x[0].detach().numpy())

    # =========================================================================
    # Stage 4: RVQ encode
    # =========================================================================
    print("\n=== RVQ Encode ===")

    def rvq_encode_group(x, group_prefix, n_codebooks, codebooks):
        """x: [1, dim, T], returns codes [n_codebooks, T]"""
        # Input projection
        proj_w = mimi_sd[f"{group_prefix}.input_proj.weight"]
        proj_b = mimi_sd.get(f"{group_prefix}.input_proj.bias")
        proj = torch.nn.functional.conv1d(x, proj_w, proj_b)  # [1, cdim, T]
        T = proj.shape[2]
        cdim = proj.shape[1]

        residual = proj.squeeze(0).permute(1, 0)  # [T, cdim]
        all_codes = []

        for q in range(n_codebooks):
            cb_key = f"{group_prefix}.vq.layers.{q}._codebook.embedding"
            if cb_key in codebooks:
                cb = codebooks[cb_key]
            else:
                cb = mimi_sd.get(cb_key)
            if cb is None:
                print(f"  WARNING: codebook {cb_key} not found!")
                continue

            # Nearest neighbor
            # cb: [num_codes, cdim]
            dists = torch.cdist(residual.unsqueeze(0), cb.unsqueeze(0)).squeeze(0)
            codes = dists.argmin(dim=-1)  # [T]

            # Decode and subtract
            quantized = cb[codes]  # [T, cdim]
            residual = residual - quantized

            all_codes.append(codes.numpy())

        return np.array(all_codes)  # [n_codebooks, T]

    n_q_semantic = 1
    n_q_acoustic = config["n_q"] - 1  # 31

    codes_first = rvq_encode_group(x, "quantizer.rvq_first", n_q_semantic, codebooks)
    codes_rest = rvq_encode_group(x, "quantizer.rvq_rest", n_q_acoustic, codebooks)
    all_codes = np.concatenate([codes_first, codes_rest], axis=0)
    print(f"  Codes shape: {all_codes.shape}")
    print(f"  First frame codes (first 5 codebooks): {all_codes[:5, 0]}")
    np.save(os.path.join(args.outdir, "06_rvq_codes.npy"), all_codes)

    # =========================================================================
    # Stage 5: LM decode
    # =========================================================================
    print("\n=== LM Decode ===")

    dim_lm = config["dim"]  # 2048
    n_heads_lm = config["num_heads"]  # 16
    head_dim_lm = dim_lm // n_heads_lm  # 128
    text_card = config["text_card"]  # 8000
    n_q = config["n_q"]  # 32
    hidden_scale = config["hidden_scale"]  # 4.125
    max_period = config["max_period"]  # 100000
    padding_id = config["existing_text_padding_id"]  # 3

    # Initial tokens: text_card (8000) for text, card (2048) for audio
    # These are special "start-of-sequence" tokens beyond the valid vocab/codebook range
    # Use official LMGen for ground-truth LM output
    from moshi.models import lm as lm_module
    from safetensors import safe_open as safe_open2

    lm_kwargs = {
        'dim': config['dim'], 'text_card': config['text_card'],
        'existing_text_padding_id': config['existing_text_padding_id'],
        'n_q': config['n_q'], 'dep_q': 0,
        'card': config['card'], 'num_heads': config['num_heads'],
        'num_layers': config['num_layers'], 'hidden_scale': config['hidden_scale'],
        'causal': True, 'context': config['context'],
        'max_period': config['max_period'], 'gating': 'silu',
        'norm': 'rms_norm_f32', 'positional_embedding': 'rope',
        'delays': config['delays'], 'cross_attention': False,
    }
    lm_model = lm_module.LMModel(**lm_kwargs)
    lm_state = {}
    with safe_open2(lm_path, framework='pt') as f:
        for key in f.keys():
            lm_state[key] = f.get_tensor(key)
    lm_model.load_state_dict(lm_state, strict=False)
    lm_model.eval()
    lm_model = lm_model.float()

    gen = lm_module.LMGen(lm_model, temp=0.0, temp_text=0.0, top_k=250, top_k_text=50)

    # Rebuild codes as torch tensor [1, n_q, T]
    all_codes_torch = torch.from_numpy(all_codes).unsqueeze(0).long()

    result_tokens = []
    result_text = ""

    with torch.no_grad(), gen.streaming(1):
        for t in range(T_frames):
            audio_tokens = all_codes_torch[:, :, t:t+1]  # [1, n_q, 1]
            out = gen.step(audio_tokens)
            if out is not None:
                if isinstance(out, tuple):
                    text_tok = out[0]
                else:
                    text_tok = out
                tok = text_tok.squeeze().item()
                result_tokens.append(tok)
            else:
                result_tokens.append(0)

            input_text = result_tokens[-1]

            if t < 10 or t % 20 == 0:
                tok = result_tokens[-1]
                piece = sp.IdToPiece(tok) if tok < sp.GetPieceSize() else f'<{tok}>'
                print(f"  [frame {t}] token={tok} piece=\"{piece}\"")

            tok = result_tokens[-1]
            if tok != 0 and tok != padding_id and tok < sp.GetPieceSize():
                piece = sp.IdToPiece(tok)
                piece = piece.replace("\u2581", " ")
                result_text += piece

    print(f"\n  Result: '{result_text}'")
    np.save(os.path.join(args.outdir, "08_lm_tokens.npy"), np.array(result_tokens))

    with open(os.path.join(args.outdir, "transcript.txt"), "w") as f:
        f.write(result_text)

    print(f"\nDumped all intermediates to {args.outdir}")


if __name__ == "__main__":
    main()
