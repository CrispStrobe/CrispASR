"""Per-stage reference dump for MADLAD-400 (google/madlad400-3b-mt), T5 1.1.

Blueprint, not a re-implementation of our runtime: this walks the checkpoint
with the semantics HF's `T5ForConditionalGeneration` has, so a disagreement
with `src/t5_translate.cpp` is a finding about the port.

⚠ LOADED LAZILY, ONE TENSOR AT A TIME. `model.safetensors` is **11.76 GB fp32**
and the box this is meant to run on (Kaggle) has ~13 GB of RAM — a plain
`from_pretrained` OOMs before it reaches the first layer. Every weight goes
through `_W()`, which `safe_open`s the file, slices out one tensor, and lets it
go. Weights are used once per layer and never cached.

The four T5 details that a shapes-compile-fine port gets wrong, all of which
this file encodes deliberately (HARD RULE #1):

  1. **No 1/sqrt(d_kv) attention scale.** T5 folds it into the q init, so the
     scores are a bare Q·Kᵀ. Dividing here would look "more correct" and be
     wrong.
  2. **Relative position bias, layer 0 only.** The bucket table lives on layer
     0 of each stack and is reused by every layer; the encoder is bidirectional
     (num_buckets split in half) and the decoder is causal (bidirectional=False,
     which halves the bucket range and changes the formula).
  3. **RMSNorm with no mean subtraction**, computed in fp32 regardless of the
     weight dtype, `x * rsqrt(mean(x²) + eps)` then scale.
  4. **Gated GELU with the tanh approximation** (`NewGELUActivation`), i.e.
     `gelu(gate(x)) * up(x)` — not the erf gelu, and not a plain FFN.

And one that only shows up at the end: MADLAD sets `tie_word_embeddings=false`,
so the lm_head is its OWN matrix and there is **no** `d_model**-0.5` rescale.
The original T5 rescaled precisely because it tied them; carrying that over is
a silent ~32x logit error here.

Stages are named to match the C++ side in `t5_translate_diff()`.

Env:
  MADLAD_TEXT   source text (default: the issue's own example sentence)
  MADLAD_TL     target language tag, without the angle brackets (default "de")
  MADLAD_LAYERS how many enc/dec layers to dump individually (default 4:
                0, 1, mid, last — a full 32+32 dump is ~1 GB of GGUF for no
                extra diagnostic value)
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Set

import numpy as np

DEFAULT_STAGES = [
    "enc_tokens",
    "enc_embed",
    "enc_pos_bias",
    "enc_blk_first",
    "enc_blk_second",
    "enc_blk_mid",
    "enc_blk_last",
    "enc_out",
    "cross_k_blk0",
    "cross_v_blk0",
    "dec_embed",
    "dec_blk_first",
    "dec_blk_last",
    "dec_out",
    "logits_step0",
    "greedy_tokens",
    "greedy_hit_eos",
    "generated_text",
]


def _rel_bucket(rel, bidirectional, num_buckets, max_distance):
    """T5's `_relative_position_bucket`, verbatim in semantics.

    The bidirectional/causal split is not cosmetic: bidirectional halves
    num_buckets and encodes the sign in the high half, causal clamps to
    non-positive. Getting it backwards puts every encoder score in the wrong
    bucket and still produces plausible activations.
    """
    import torch

    n = -rel
    ret = torch.zeros_like(n)
    if bidirectional:
        num_buckets //= 2
        ret = ret + (n < 0).long() * num_buckets
        n = torch.abs(n)
    else:
        n = torch.max(n, torch.zeros_like(n))
    max_exact = num_buckets // 2
    is_small = n < max_exact
    val_large = max_exact + (
        torch.log(n.float() / max_exact + 1e-20)
        / np.log(max_distance / max_exact)
        * (num_buckets - max_exact)
    ).long()
    val_large = torch.min(val_large, torch.full_like(val_large, num_buckets - 1))
    return ret + torch.where(is_small, n, val_large)


def dump(*, model_dir: Path, audio: np.ndarray, stages: Set[str],
         max_new_tokens: int) -> Dict[str, np.ndarray]:
    import torch
    from safetensors import safe_open

    del audio  # text model — the shared --audio arg is ignored

    model_dir = Path(model_dir)
    cfg = json.loads((model_dir / "config.json").read_text())
    D = cfg["d_model"]
    HD = cfg["d_kv"]
    NH = cfg["num_heads"]
    N_ENC = cfg["num_layers"]
    N_DEC = cfg.get("num_decoder_layers", N_ENC)
    N_BUCKETS = cfg["relative_attention_num_buckets"]
    MAX_DIST = cfg["relative_attention_max_distance"]
    EPS = cfg["layer_norm_epsilon"]
    TIED = cfg.get("tie_word_embeddings", False)
    START = cfg.get("decoder_start_token_id", 0)
    EOS = cfg.get("eos_token_id", 1)

    st_path = model_dir / "model.safetensors"
    idx_path = model_dir / "model.safetensors.index.json"
    is_sharded = idx_path.is_file()
    if is_sharded:
        idx = json.loads(idx_path.read_text(encoding="utf-8"))
        weight_map = idx.get("weight_map", {})
        keys = set(weight_map.keys())
        handle = None
    else:
        handle = safe_open(str(st_path), framework="pt")
        keys = set(handle.keys())

    def _W(name: str) -> "torch.Tensor":
        """One tensor, fp32, then let it go. Never cache — that is the whole
        point of this file (11.76 GB of weights vs ~13 GB of RAM).

        Fails with the near-misses listed, because a wrong key here costs a
        whole Kaggle run to discover: the first version of this file guessed
        `shared.weight` and died 14 minutes in with a bare KeyError."""
        if name not in keys:
            stem = name.split(".")[-2] if "." in name else name
            near = sorted(k for k in keys if stem in k)[:8]
            raise KeyError(f"{name} not in {model_dir.name}. Similar keys: {near}")
        if is_sharded:
            shard_path = model_dir / weight_map[name]
            with safe_open(str(shard_path), framework="pt") as h:
                return h.get_tensor(name).float()
        return handle.get_tensor(name).float()

    def _W_any(*names: str) -> "torch.Tensor":
        """First key that exists. The embedding is stored under one of three
        names depending on how the checkpoint was saved, which is exactly what
        `models/convert-madlad-to-gguf.py` already accounts for — this mirrors
        its list rather than assuming one of them."""
        for n in names:
            if n in keys:
                return _W(n)
        raise KeyError(f"none of {names} in {model_dir.name}")

    EMBED = ("shared.weight", "encoder.embed_tokens.weight", "decoder.embed_tokens.weight")

    def rms(x, w):
        # fp32, no mean subtraction. T5LayerNorm.
        v = x.float()
        return (v * torch.rsqrt(v.pow(2).mean(-1, keepdim=True) + EPS)) * w

    def gelu_tanh(x):
        # NewGELUActivation — the tanh approximation, which is what
        # feed_forward_proj="gated-gelu" selects.
        return 0.5 * x * (1.0 + torch.tanh(
            np.sqrt(2.0 / np.pi) * (x + 0.044715 * torch.pow(x, 3.0))))

    text = os.environ.get("MADLAD_TEXT", "Hello world, how are you today?")
    tl = os.environ.get("MADLAD_TL", "de")
    n_dump = int(os.environ.get("MADLAD_LAYERS", "4"))
    src = f"<2{tl}> {text}"

    # ── tokenize (SentencePiece unigram, spiece.model) ────────────────────────
    import sentencepiece as spm

    sp = spm.SentencePieceProcessor(model_file=str(model_dir / "spiece.model"))
    ids = sp.encode(src, out_type=int) + [EOS]
    enc_ids = torch.tensor(ids, dtype=torch.long)
    T = len(ids)
    print(f"  madlad: '{src}' -> {T} tokens, target=<2{tl}>")

    out: Dict[str, np.ndarray] = {}

    def put(name, t):
        if name in stages:
            out[name] = np.asarray(t.detach().cpu().float().numpy(), dtype=np.float32)

    put("enc_tokens", enc_ids.float())

    # ── encoder ───────────────────────────────────────────────────────────────
    x = _W_any(*EMBED)[enc_ids]  # (T, D)
    put("enc_embed", x)

    q_pos = torch.arange(T)[:, None]
    k_pos = torch.arange(T)[None, :]
    enc_bucket = _rel_bucket(k_pos - q_pos, True, N_BUCKETS, MAX_DIST)
    enc_bias = _W("encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight")
    # (T,T,NH) -> (NH,T,T): the bias is added per head to the score matrix.
    pos_bias = enc_bias[enc_bucket].permute(2, 0, 1)
    put("enc_pos_bias", pos_bias)

    marks = sorted({0, 1, N_ENC // 2, N_ENC - 1})[:n_dump]
    names = ["enc_blk_first", "enc_blk_second", "enc_blk_mid", "enc_blk_last"]
    for il in range(N_ENC):
        p = f"encoder.block.{il}.layer"
        h = rms(x, _W(f"{p}.0.layer_norm.weight"))
        q = (h @ _W(f"{p}.0.SelfAttention.q.weight").T).view(T, NH, HD).permute(1, 0, 2)
        k = (h @ _W(f"{p}.0.SelfAttention.k.weight").T).view(T, NH, HD).permute(1, 0, 2)
        v = (h @ _W(f"{p}.0.SelfAttention.v.weight").T).view(T, NH, HD).permute(1, 0, 2)
        # NO 1/sqrt(HD) — see the module docstring.
        scores = torch.matmul(q, k.transpose(-1, -2)) + pos_bias
        a = torch.softmax(scores.float(), dim=-1)
        ctx = torch.matmul(a, v).permute(1, 0, 2).reshape(T, NH * HD)
        x = x + ctx @ _W(f"{p}.0.SelfAttention.o.weight").T

        h = rms(x, _W(f"{p}.1.layer_norm.weight"))
        g = gelu_tanh(h @ _W(f"{p}.1.DenseReluDense.wi_0.weight").T)
        u = h @ _W(f"{p}.1.DenseReluDense.wi_1.weight").T
        x = x + (g * u) @ _W(f"{p}.1.DenseReluDense.wo.weight").T
        if il in marks:
            put(names[marks.index(il)], x)

    x = rms(x, _W("encoder.final_layer_norm.weight"))
    put("enc_out", x)
    enc_out = x

    # ── cross K/V for block 0 — the encoder-decoder joint, and the stage an
    # encoder-decoder port is most likely to get wrong. ───────────────────────
    ck = (enc_out @ _W("decoder.block.0.layer.1.EncDecAttention.k.weight").T)
    cv = (enc_out @ _W("decoder.block.0.layer.1.EncDecAttention.v.weight").T)
    put("cross_k_blk0", ck.view(T, NH, HD).permute(1, 0, 2))
    put("cross_v_blk0", cv.view(T, NH, HD).permute(1, 0, 2))

    # ── decoder, greedy ───────────────────────────────────────────────────────
    dec_bias_w = _W("decoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight")
    lm_w = _W_any(*EMBED) if TIED else _W("lm_head.weight")

    dec_ids = [START]
    gen: list[int] = []
    n_steps = max(1, min(int(max_new_tokens or 24), 64))
    logits0 = None
    dec_marks = sorted({0, N_DEC - 1})
    dec_names = ["dec_blk_first", "dec_blk_last"]

    for step in range(n_steps):
        L = len(dec_ids)
        y = _W_any(*EMBED)[torch.tensor(dec_ids, dtype=torch.long)]
        if step == 0:
            put("dec_embed", y)
        dq = torch.arange(L)[:, None]
        dk = torch.arange(L)[None, :]
        # causal: bidirectional=False, and a -inf mask on the future.
        dbucket = _rel_bucket(dk - dq, False, N_BUCKETS, MAX_DIST)
        dbias = dec_bias_w[dbucket].permute(2, 0, 1)
        causal = torch.full((L, L), float("-inf")).triu(1)

        for il in range(N_DEC):
            p = f"decoder.block.{il}.layer"
            h = rms(y, _W(f"{p}.0.layer_norm.weight"))
            q = (h @ _W(f"{p}.0.SelfAttention.q.weight").T).view(L, NH, HD).permute(1, 0, 2)
            k = (h @ _W(f"{p}.0.SelfAttention.k.weight").T).view(L, NH, HD).permute(1, 0, 2)
            v = (h @ _W(f"{p}.0.SelfAttention.v.weight").T).view(L, NH, HD).permute(1, 0, 2)
            s = torch.matmul(q, k.transpose(-1, -2)) + dbias + causal
            a = torch.softmax(s.float(), dim=-1)
            c = torch.matmul(a, v).permute(1, 0, 2).reshape(L, NH * HD)
            y = y + c @ _W(f"{p}.0.SelfAttention.o.weight").T

            h = rms(y, _W(f"{p}.1.layer_norm.weight"))
            cq = (h @ _W(f"{p}.1.EncDecAttention.q.weight").T).view(L, NH, HD).permute(1, 0, 2)
            xk = (enc_out @ _W(f"{p}.1.EncDecAttention.k.weight").T).view(T, NH, HD).permute(1, 0, 2)
            xv = (enc_out @ _W(f"{p}.1.EncDecAttention.v.weight").T).view(T, NH, HD).permute(1, 0, 2)
            # Cross-attention has NO position bias and NO mask.
            cs = torch.matmul(cq, xk.transpose(-1, -2))
            ca = torch.softmax(cs.float(), dim=-1)
            cc = torch.matmul(ca, xv).permute(1, 0, 2).reshape(L, NH * HD)
            y = y + cc @ _W(f"{p}.1.EncDecAttention.o.weight").T

            h = rms(y, _W(f"{p}.2.layer_norm.weight"))
            g = gelu_tanh(h @ _W(f"{p}.2.DenseReluDense.wi_0.weight").T)
            u = h @ _W(f"{p}.2.DenseReluDense.wi_1.weight").T
            y = y + (g * u) @ _W(f"{p}.2.DenseReluDense.wo.weight").T
            if step == 0 and il in dec_marks:
                put(dec_names[dec_marks.index(il)], y)

        y = rms(y, _W("decoder.final_layer_norm.weight"))
        if step == 0:
            put("dec_out", y)
        # tie_word_embeddings=false ⇒ NO d_model**-0.5 rescale here.
        logits = y[-1] @ lm_w.T
        if step == 0:
            logits0 = logits
            put("logits_step0", logits)
        nxt = int(torch.argmax(logits).item())
        if nxt == EOS:
            break
        gen.append(nxt)
        dec_ids.append(nxt)

    # Did the BLUEPRINT stop on its own? This is the question that decides
    # whether a runaway decode in the C++ is a port bug or the model's own
    # behaviour, and it costs nothing to record here (#333).
    hit_eos = len(gen) < n_steps
    print(f"  madlad reference: {len(gen)} tokens, "
          f"{'hit EOS' if hit_eos else f'RAN TO THE {n_steps}-STEP CAP (no EOS)'}")
    if "greedy_tokens" in stages:
        out["greedy_tokens"] = np.asarray(gen, dtype=np.float32)
    if "greedy_hit_eos" in stages:
        out["greedy_hit_eos"] = np.asarray([1.0 if hit_eos else 0.0], dtype=np.float32)
    if "generated_text" in stages:
        txt = sp.decode(gen)
        print(f"  madlad reference: {txt!r}")
        out["generated_text"] = np.frombuffer(txt.encode("utf-8"), dtype=np.uint8).astype(np.float32)

    del logits0
    return out
