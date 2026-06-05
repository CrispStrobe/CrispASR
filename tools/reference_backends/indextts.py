"""IndexTTS-1.5 reference dump backend for crispasr-diff.

Captures per-stage activations from the IndexTTS-1.5 PyTorch model so
the C++ implementation can be validated tensor-by-tensor.

Pipeline stages captured:
  text_tokens         — BPE token IDs (int32)
  prefix_embeds       — [cond_latents | text_embs+pos | start_mel+pos] embedding
  gpt_layer_0         — hidden state after GPT block 0
  gpt_layer_23        — hidden state after GPT block 23
  prefill_logits      — mel logits from last prefix position
  mel_codes           — full AR-generated mel code sequence (int32)
  latent_output       — GPT hidden states for mel positions [n_codes, D]

Usage:
    python tools/dump_reference.py --backend indextts \
        --model-dir /mnt/storage/models/IndexTTS-1.5 \
        --output /mnt/storage/indextts-ref.gguf \
        --text "Hello"
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, Set

import numpy as np

DEFAULT_STAGES = [
    "text_tokens",
    "prefix_embeds",
    "gpt_layer_0",
    "gpt_layer_23",
    "prefill_logits",
    "mel_codes",
    "latent_output",
]


def dump(*, model_dir: Path, audio: np.ndarray, stages: Set[str],
         max_new_tokens: int) -> Dict[str, np.ndarray]:
    import torch
    import torch.nn.functional as F

    out: Dict[str, np.ndarray] = {}

    text = os.environ.get("INDEXTTS_TEXT", "Hello")
    print(f"  text: {repr(text)}")

    # Load tokenizer
    import sentencepiece as spm
    sp = spm.SentencePieceProcessor()
    sp.Load(str(model_dir / "bpe.model"))
    text_tokens = sp.Encode(text)
    print(f"  tokens: {text_tokens}")
    if "text_tokens" in stages:
        out["text_tokens"] = np.array(text_tokens, dtype=np.int32)

    # Load GPT checkpoint
    ckpt = torch.load(str(model_dir / "gpt.pth"), map_location="cpu", weights_only=False)
    sd = ckpt["model"]
    sd = {k: v.float() for k, v in sd.items()}

    D = 1280
    n_heads = 20
    n_layers = 24
    head_dim = D // n_heads
    start_mel = 8192
    stop_mel = 8193

    # Conditioning latents (zeros — no reference audio)
    cond_latents = torch.zeros(32, D)

    # Build prefix embeddings
    text_emb = sd["text_embedding.weight"]
    text_pos = sd["text_pos_embedding.emb.weight"]
    mel_emb = sd["mel_embedding.weight"]
    mel_pos = sd["mel_pos_embedding.emb.weight"]

    text_ids = torch.tensor(text_tokens, dtype=torch.long)
    text_embedded = text_emb[text_ids] + text_pos[:len(text_tokens)]
    start_mel_embedded = mel_emb[start_mel:start_mel + 1] + mel_pos[0:1]

    prefix = torch.cat([cond_latents, text_embedded, start_mel_embedded], dim=0)
    if "prefix_embeds" in stages:
        out["prefix_embeds"] = prefix.numpy()

    # GPT-2 forward pass helper
    def gpt_forward(x_in, mask_in, capture_layers=False):
        x = x_in.unsqueeze(0)
        T = x.shape[1]
        layers_out = {}

        for il in range(n_layers):
            p = f"gpt.h.{il}"
            residual = x
            h = F.layer_norm(x, [D], sd[f"{p}.ln_1.weight"], sd[f"{p}.ln_1.bias"])

            # QKV (Conv1D: weight is [in, out])
            qkv = h @ sd[f"{p}.attn.c_attn.weight"] + sd[f"{p}.attn.c_attn.bias"]
            Q, K, V = qkv.chunk(3, dim=-1)
            Q = Q.view(1, T, n_heads, head_dim).transpose(1, 2)
            K = K.view(1, T, n_heads, head_dim).transpose(1, 2)
            V = V.view(1, T, n_heads, head_dim).transpose(1, 2)

            attn = torch.matmul(Q, K.transpose(-2, -1)) / (head_dim ** 0.5)
            attn = attn + mask_in
            attn = F.softmax(attn, dim=-1)
            attn_out = torch.matmul(attn, V)
            attn_out = attn_out.transpose(1, 2).contiguous().view(1, T, D)
            attn_out = attn_out @ sd[f"{p}.attn.c_proj.weight"] + sd[f"{p}.attn.c_proj.bias"]
            x = residual + attn_out

            residual = x
            h = F.layer_norm(x, [D], sd[f"{p}.ln_2.weight"], sd[f"{p}.ln_2.bias"])
            mlp = F.gelu(h @ sd[f"{p}.mlp.c_fc.weight"] + sd[f"{p}.mlp.c_fc.bias"],
                         approximate="tanh")
            mlp = mlp @ sd[f"{p}.mlp.c_proj.weight"] + sd[f"{p}.mlp.c_proj.bias"]
            x = residual + mlp

            if capture_layers and (il == 0 or il == n_layers - 1):
                layers_out[il] = x[0].detach().clone()

        # GPT-2 final LayerNorm (gpt.ln_f — HuggingFace GPT2Model applies this)
        if "gpt.ln_f.weight" in sd:
            x = F.layer_norm(x, [D], sd["gpt.ln_f.weight"], sd["gpt.ln_f.bias"])
        x_norm = F.layer_norm(x, [D], sd["final_norm.weight"], sd["final_norm.bias"])
        return x_norm[0], layers_out

    # Prefill pass
    T_prefix = prefix.shape[0]
    mask = torch.full((T_prefix, T_prefix), float("-inf"))
    mask = torch.triu(mask, diagonal=1)

    x_norm, layers = gpt_forward(prefix, mask, capture_layers=True)

    if "gpt_layer_0" in stages and 0 in layers:
        out["gpt_layer_0"] = layers[0].numpy()
    if "gpt_layer_23" in stages and 23 in layers:
        out["gpt_layer_23"] = layers[23].numpy()

    # Prefill logits
    mel_head_w = sd["mel_head.weight"]
    mel_head_b = sd["mel_head.bias"]
    logits = x_norm[-1] @ mel_head_w.t() + mel_head_b
    if "prefill_logits" in stages:
        out["prefill_logits"] = logits.detach().numpy()

    top5 = logits.topk(5)
    print(f"  prefill top5: ids={top5.indices.tolist()} vals={[f'{v:.3f}' for v in top5.values.tolist()]}")

    # AR decode — generate mel codes
    mel_codes = []
    cur_token = logits.argmax().item()
    mel_codes.append(cur_token)

    # For full AR we'd need incremental KV cache. For simplicity,
    # just do greedy argmax for a few steps to get reference mel codes.
    max_mel = min(max_new_tokens, 200)
    all_tokens = list(range(T_prefix))  # dummy — we rebuild the full sequence each step
    # Actually, let's do a simple full-recompute AR (slow but correct)
    full_seq = prefix.clone()
    for step in range(max_mel):
        new_emb = mel_emb[cur_token:cur_token + 1] + mel_pos[step + 1:step + 2]
        if step + 1 >= mel_pos.shape[0]:
            break
        full_seq = torch.cat([full_seq, new_emb], dim=0)
        T = full_seq.shape[0]
        mask = torch.full((T, T), float("-inf"))
        mask = torch.triu(mask, diagonal=1)
        x_norm, _ = gpt_forward(full_seq, mask)
        logits = x_norm[-1] @ mel_head_w.t() + mel_head_b
        top5 = logits.topk(5)
        cur_token = logits.argmax().item()
        if cur_token == stop_mel:
            break
        mel_codes.append(cur_token)
        if step < 10 or step % 20 == 0:
            print(f"  step {step}: token={cur_token} top5={top5.indices.tolist()} vals={[f'{v:.3f}' for v in top5.values.tolist()]}")

    print(f"  generated {len(mel_codes)} mel codes")
    if "mel_codes" in stages:
        out["mel_codes"] = np.array(mel_codes, dtype=np.int32)

    # Latent extraction — second forward pass on full sequence
    if "latent_output" in stages:
        T = full_seq.shape[0]
        mask = torch.full((T, T), float("-inf"))
        mask = torch.triu(mask, diagonal=1)
        x_norm, _ = gpt_forward(full_seq, mask)
        # Extract hidden states for mel positions only
        mel_start = T_prefix  # mel positions start after prefix
        mel_end = mel_start + len(mel_codes)
        latent = x_norm[mel_start - 1:mel_end - 1]  # shifted by 1 (input position predicts next)
        out["latent_output"] = latent.detach().numpy()
        print(f"  latent: {latent.shape}")

    return out
