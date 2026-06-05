#!/usr/bin/env python3
"""OmniASR-LLM reference activation dumper.

Manually runs the encoder forward pass using raw PyTorch ops on the state dict
(no fairseq2 dependency). Dumps intermediates as numpy files for comparison
with the C++ ggml implementation.

Usage:
  python tools/reference_backends/omniasr_llm.py \
      --checkpoint /path/to/omniASR-LLM-300M-v2.pt \
      --audio samples/jfk.wav \
      --output-dir /tmp/omniasr-ref/
"""

import argparse
import os
import struct
import math
import numpy as np

def load_wav_16k(path):
    """Load WAV file as 16kHz mono float32 PCM."""
    import subprocess
    raw = subprocess.check_output([
        'ffmpeg', '-i', path, '-f', 'f32le', '-ar', '16000', '-ac', '1', '-',
    ], stderr=subprocess.DEVNULL)
    return np.frombuffer(raw, dtype=np.float32)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--audio', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--max-enc-layers', type=int, default=24)
    args = parser.parse_args()

    import torch
    import torch.nn.functional as F

    os.makedirs(args.output_dir, exist_ok=True)

    # Load audio
    pcm = load_wav_16k(args.audio)
    print(f"Audio: {len(pcm)} samples ({len(pcm)/16000:.2f}s)")
    np.save(os.path.join(args.output_dir, 'raw_audio.npy'), pcm)

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    sd = ckpt['model']
    print(f"Loaded {len(sd)} tensors")

    pcm_t = torch.from_numpy(pcm).unsqueeze(0)  # [1, N]

    # Input normalization (wav2vec2 convention)
    mean = pcm_t.mean()
    var = pcm_t.var()
    pcm_norm = (pcm_t - mean) / torch.sqrt(var + 1e-5)
    np.save(os.path.join(args.output_dir, 'pcm_norm.npy'), pcm_norm.numpy())
    print(f"  pcm_norm: {pcm_norm.shape}, first5={pcm_norm[0,:5].tolist()}")

    # CNN Feature Extractor: 7 layers of Conv1d + LayerNorm + GELU
    n_cnn = 7
    h = pcm_norm.unsqueeze(1)  # [1, 1, N]
    for i in range(n_cnn):
        prefix = f'encoder_frontend.feature_extractor.layers.{i}'
        conv_w = sd[f'{prefix}.conv.weight']
        conv_b = sd[f'{prefix}.conv.bias']
        ln_w = sd[f'{prefix}.layer_norm.weight']
        ln_b = sd[f'{prefix}.layer_norm.bias']

        stride = 5 if i == 0 else 2
        h = F.conv1d(h, conv_w, conv_b, stride=stride)
        # LayerNorm over channel dim (last dim after transpose)
        h = h.transpose(1, 2)  # [B, T, C]
        h = F.layer_norm(h, [h.shape[-1]], ln_w, ln_b)
        h = F.gelu(h)
        h = h.transpose(1, 2)  # [B, C, T]

        if i == n_cnn - 1:
            cnn_out = h.squeeze(0).numpy()  # [C, T]
            np.save(os.path.join(args.output_dir, 'cnn_out.npy'), cnn_out)
            print(f"  cnn_out: {cnn_out.shape}, first5={cnn_out.ravel()[:5].tolist()}")

    # h is [1, 512, T_cnn]
    h = h.transpose(1, 2)  # [1, T, 512]

    # Post-extract LayerNorm
    pe_ln_w = sd['encoder_frontend.post_extract_layer_norm.weight']
    pe_ln_b = sd['encoder_frontend.post_extract_layer_norm.bias']
    h = F.layer_norm(h, [h.shape[-1]], pe_ln_w, pe_ln_b)
    np.save(os.path.join(args.output_dir, 'post_extract_ln.npy'), h.squeeze(0).numpy())
    print(f"  post_extract_ln: {h.shape}")

    # Projection: 512 → d_model (1024)
    proj_w = sd['encoder_frontend.model_dim_proj.weight']
    proj_b = sd['encoder_frontend.model_dim_proj.bias']
    h = F.linear(h, proj_w, proj_b)  # [1, T, 1024]
    np.save(os.path.join(args.output_dir, 'proj_out.npy'), h.squeeze(0).numpy())
    print(f"  proj: {h.shape}")

    # Positional encoding: grouped conv1d + GELU + residual
    # Weight normalization: w = g * v / ||v||
    wv = sd['encoder_frontend.pos_encoder.conv.weight_v']  # [1024, 64, 128]
    wg = sd['encoder_frontend.pos_encoder.conv.weight_g']  # [1, 1, 128]
    bias = sd['encoder_frontend.pos_encoder.conv.bias']    # [1024]
    v_norm = wv.reshape(wv.shape[0], -1).norm(dim=1).reshape(-1, 1, 1)
    w_combined = (wg / (v_norm + 1e-12)) * wv

    # Grouped conv1d: groups=16, kernel=128, padding=K//2=64
    h_conv = h.transpose(1, 2)  # [1, 1024, T]
    pad = 128 // 2  # 64
    h_conv = F.pad(h_conv, (pad, pad))
    pos = F.conv1d(h_conv, w_combined, bias, groups=16)
    # Trim to match input length
    if pos.shape[2] > h.shape[1]:
        pos = pos[:, :, :h.shape[1]]
    pos = F.gelu(pos)
    pos = pos.transpose(1, 2)  # [1, T, 1024]
    h = h + pos  # residual
    np.save(os.path.join(args.output_dir, 'pos_conv_out.npy'), h.squeeze(0).numpy())
    print(f"  pos_conv: {h.shape}")

    # Transformer encoder layers
    n_enc = min(args.max_enc_layers,
                max(int(k.split('.')[2]) for k in sd if k.startswith('encoder.layers.')) + 1)
    T = h.shape[1]
    d = h.shape[2]
    n_heads = d // 64

    for i in range(n_enc):
        prefix = f'encoder.layers.{i}'
        # Pre-norm
        attn_ln_w = sd[f'{prefix}.self_attn_layer_norm.weight']
        attn_ln_b = sd[f'{prefix}.self_attn_layer_norm.bias']
        residual = h
        h_norm = F.layer_norm(h, [d], attn_ln_w, attn_ln_b)

        # Self-attention
        q_w = sd[f'{prefix}.self_attn.q_proj.weight']
        q_b = sd[f'{prefix}.self_attn.q_proj.bias']
        k_w = sd[f'{prefix}.self_attn.k_proj.weight']
        k_b = sd[f'{prefix}.self_attn.k_proj.bias']
        v_w = sd[f'{prefix}.self_attn.v_proj.weight']
        v_b = sd[f'{prefix}.self_attn.v_proj.bias']
        o_w = sd[f'{prefix}.self_attn.output_proj.weight']
        o_b = sd[f'{prefix}.self_attn.output_proj.bias']

        Q = F.linear(h_norm, q_w, q_b)  # [1, T, d]
        K = F.linear(h_norm, k_w, k_b)
        V = F.linear(h_norm, v_w, v_b)

        # Multi-head reshape
        head_dim = d // n_heads
        Q = Q.view(1, T, n_heads, head_dim).transpose(1, 2)  # [1, nh, T, hd]
        K = K.view(1, T, n_heads, head_dim).transpose(1, 2)
        V = V.view(1, T, n_heads, head_dim).transpose(1, 2)

        # Scaled dot-product attention
        attn = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(head_dim)
        attn = F.softmax(attn, dim=-1)
        attn_out = torch.matmul(attn, V)

        attn_out = attn_out.transpose(1, 2).contiguous().view(1, T, d)
        attn_out = F.linear(attn_out, o_w, o_b)
        h = residual + attn_out

        # FFN
        ffn_ln_w = sd[f'{prefix}.ffn_layer_norm.weight']
        ffn_ln_b = sd[f'{prefix}.ffn_layer_norm.bias']
        residual = h
        h_norm = F.layer_norm(h, [d], ffn_ln_w, ffn_ln_b)
        up_w = sd[f'{prefix}.ffn.inner_proj.weight']
        up_b = sd[f'{prefix}.ffn.inner_proj.bias']
        down_w = sd[f'{prefix}.ffn.output_proj.weight']
        down_b = sd[f'{prefix}.ffn.output_proj.bias']
        ffn = F.linear(h_norm, up_w, up_b)
        ffn = F.gelu(ffn)
        ffn = F.linear(ffn, down_w, down_b)
        h = residual + ffn

        if i < 3 or i == n_enc - 1:
            np.save(os.path.join(args.output_dir, f'enc_layer_{i}.npy'), h.squeeze(0).numpy())
            print(f"  enc_layer_{i}: {h.shape}, first5={h[0,0,:5].tolist()}")

    # Final encoder LayerNorm
    enc_ln_w = sd['encoder.layer_norm.weight']
    enc_ln_b = sd['encoder.layer_norm.bias']
    h = F.layer_norm(h, [d], enc_ln_w, enc_ln_b)
    enc_out = h.squeeze(0).numpy()  # [T, d]
    np.save(os.path.join(args.output_dir, 'encoder_output.npy'), enc_out)
    print(f"  encoder_output: {enc_out.shape}, first5={enc_out[0,:5].tolist()}")

    # Encoder projection: d_model → d_dec (1024 → 4096)
    enc_proj_w = sd['encoder_proj.weight']
    enc_proj_b = sd['encoder_proj.bias']
    h_dec = F.linear(h, enc_proj_w, enc_proj_b)  # [1, T, 4096]
    enc_proj_out = h_dec.squeeze(0).numpy()
    np.save(os.path.join(args.output_dir, 'enc_proj_output.npy'), enc_proj_out)
    print(f"  enc_proj: {enc_proj_out.shape}, first5={enc_proj_out[0,:5].tolist()}")

    print(f"\nDumped to {args.output_dir}")

if __name__ == '__main__':
    main()
