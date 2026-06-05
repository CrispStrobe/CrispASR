#!/usr/bin/env python3
"""Convert FunAudioLLM/Fun-CosyVoice3-0.5B-2512 to GGUF.

CosyVoice3 is three sub-models tied together:

  1. LLM  (llm.pt, 2.0 GB)  — Qwen2-0.5B + speech_embedding (6761, 896)
                              + llm_decoder (6761, 896) speech-token head;
                              AR-decodes speech tokens from text.
  2. Flow (flow.pt, 1.3 GB) — input_embedding (6561, 80) → pre_lookahead
                              causal conv → DiT estimator (22 blocks,
                              AdaLN-Zero modulation, 1024-dim, 16 heads,
                              RoPE θ=1e6) → CausalConditionalCFM solver;
                              speech tokens → 80-dim mel.
  3. HiFT (hift.pt, 83 MB)  — CausalHiFTGenerator (HiFi-GAN-iSTFT hybrid):
                              80→512 conv_pre, 3 upsample stages with
                              Snake activations + NSF source modulator,
                              CausalConvRNNF0Predictor, conv_post → iSTFT
                              (n_fft=16, hop=4) → 24 kHz waveform.

Plus the Qwen2 BPE tokenizer (vocab=151936) lives in CosyVoice-BlankEN/.

Output: one GGUF per sub-model so users can mix quantisations later
(e.g. Q4_K LLM + F16 DiT/HiFT — flow/vocoder quality is more sensitive
to quant noise than the AR LLM is, mirroring qwen3-tts deployment
recipe).

Tensor naming:
  LLM  GGUF: llama.cpp-standard `blk.K.attn_q.weight` / `blk.K.ffn_*.weight`
             + `cosyvoice3.speech_embd.weight` + `cosyvoice3.speech_lm_head.weight`
             + standard `token_embd` / `output` / `output_norm`.
  Flow GGUF: `cosyvoice3.flow.input_embd.w`, `cosyvoice3.flow.pre_la.conv1.{w,b}`,
             `cosyvoice3.flow.spk_affine.{w,b}`,
             `cosyvoice3.flow.dit.input_embd.proj.{w,b}`,
             `cosyvoice3.flow.dit.input_embd.conv_pos.conv{1,2}.{w,b}`,
             `cosyvoice3.flow.dit.time_mlp.{0,2}.{w,b}`,
             `cosyvoice3.flow.dit.rotary_inv_freq`,
             `cosyvoice3.flow.dit.blk.K.adaln.{w,b}` (the 6144 modulation),
             `cosyvoice3.flow.dit.blk.K.attn.{q,k,v,o}.{w,b}`,
             `cosyvoice3.flow.dit.blk.K.ffn.{l1,l2}.{w,b}`,
             `cosyvoice3.flow.dit.norm_out.{w,b}`,
             `cosyvoice3.flow.dit.proj_out.{w,b}`.
  HiFT GGUF: `cosyvoice3.hift.conv_pre.{w,b}`, `cosyvoice3.hift.ups.K.{w,b}`,
             `cosyvoice3.hift.resblocks.K.convs{1,2}.J.{w,b,alpha}`,
             `cosyvoice3.hift.source_downs.K.{w,b}`,
             `cosyvoice3.hift.source_resblocks.K.convs{1,2}.J.{w,b,alpha}`,
             `cosyvoice3.hift.m_source.linear.{w,b}`,
             `cosyvoice3.hift.f0.condnet.K.{w,b}`,
             `cosyvoice3.hift.f0.classifier.{w,b}`,
             `cosyvoice3.hift.conv_post.{w,b}`.

Weight-norm parametrisation: hift uses `nn.utils.weight_norm` everywhere,
which splits each weight into `parametrizations.weight.original0` (g,
scale per output channel) + `parametrizations.weight.original1` (v,
direction). The converter materialises plain `w = g · v / ‖v‖` and
writes one tensor per conv — the runtime never sees the parametrised
form. Mirrors voxcpm2's `wn_reconstruct` cache pattern.

Usage:
  python models/convert-cosyvoice3-to-gguf.py \\
      --input FunAudioLLM/Fun-CosyVoice3-0.5B-2512 \\
      --output-dir /Volumes/backups/ai/crispasr-models/cosyvoice3-0.5b-2512/
"""

import argparse
import json
import os
import re
import sys

import numpy as np

try:
    import gguf
except ImportError:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ggml", "python"))
    import gguf


# ---------------------------------------------------------------------------
# Weight-norm materialiser (g, v) → w = g · v / ‖v‖
# ---------------------------------------------------------------------------

def wn_resolve(g: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Reconstruct PyTorch nn.utils.weight_norm (dim=0 for Conv1d) into
    a plain weight. `g` has shape (out_ch, 1, 1) for Conv1d or
    (out_ch, 1) for Linear; `v` carries the same shape as the final
    weight (out_ch, in_ch, kernel) for Conv1d."""
    g = g.astype(np.float32)
    v = v.astype(np.float32)
    # Flatten v's non-output dims, compute L2 norm per output channel,
    # then scale by g/‖v‖.
    out_ch = v.shape[0]
    v_flat = v.reshape(out_ch, -1)
    v_norm = np.linalg.norm(v_flat, axis=1, keepdims=True)  # (out_ch, 1)
    v_norm = np.maximum(v_norm, 1e-12)
    scale = (g.reshape(out_ch, 1) / v_norm).astype(np.float32)
    return (v_flat * scale).reshape(v.shape)


def to_np(t):
    import torch
    t = t.detach()
    if t.dtype == torch.bfloat16:
        return t.float().numpy()
    return t.numpy()


def add_tensor(writer, name, t, *, force_f32=False):
    assert len(name) < 64, f"GGUF name too long ({len(name)}): {name}"
    if not force_f32 and t.ndim >= 2:
        data = np.ascontiguousarray(t.astype(np.float16))
        dtype = "F16"
    else:
        data = np.ascontiguousarray(t.astype(np.float32))
        dtype = "F32"
    writer.add_tensor(name, data)
    return dtype


# ---------------------------------------------------------------------------
# LLM converter — Qwen2-0.5B + speech-token heads
# ---------------------------------------------------------------------------

LLM_DIRECT = {
    "llm.model.model.embed_tokens.weight": "token_embd.weight",
    "llm.model.model.norm.weight":         "output_norm.weight",
    "llm.model.lm_head.weight":            "output.weight",
}

LLM_LAYER_PATTERNS = [
    (r"llm\.model\.model\.layers\.(\d+)\.input_layernorm\.weight",
     "blk.{}.attn_norm.weight"),
    (r"llm\.model\.model\.layers\.(\d+)\.self_attn\.q_proj\.weight",
     "blk.{}.attn_q.weight"),
    (r"llm\.model\.model\.layers\.(\d+)\.self_attn\.q_proj\.bias",
     "blk.{}.attn_q.bias"),
    (r"llm\.model\.model\.layers\.(\d+)\.self_attn\.k_proj\.weight",
     "blk.{}.attn_k.weight"),
    (r"llm\.model\.model\.layers\.(\d+)\.self_attn\.k_proj\.bias",
     "blk.{}.attn_k.bias"),
    (r"llm\.model\.model\.layers\.(\d+)\.self_attn\.v_proj\.weight",
     "blk.{}.attn_v.weight"),
    (r"llm\.model\.model\.layers\.(\d+)\.self_attn\.v_proj\.bias",
     "blk.{}.attn_v.bias"),
    (r"llm\.model\.model\.layers\.(\d+)\.self_attn\.o_proj\.weight",
     "blk.{}.attn_output.weight"),
    (r"llm\.model\.model\.layers\.(\d+)\.post_attention_layernorm\.weight",
     "blk.{}.ffn_norm.weight"),
    (r"llm\.model\.model\.layers\.(\d+)\.mlp\.gate_proj\.weight",
     "blk.{}.ffn_gate.weight"),
    (r"llm\.model\.model\.layers\.(\d+)\.mlp\.up_proj\.weight",
     "blk.{}.ffn_up.weight"),
    (r"llm\.model\.model\.layers\.(\d+)\.mlp\.down_proj\.weight",
     "blk.{}.ffn_down.weight"),
]


def remap_llm(hf_name: str):
    if hf_name in LLM_DIRECT:
        return LLM_DIRECT[hf_name]
    for pat, tmpl in LLM_LAYER_PATTERNS:
        m = re.match(pat, hf_name)
        if m:
            return tmpl.format(m.group(1))
    return None


def convert_llm(base, output_path):
    import torch
    model_pt = os.path.join(base, "llm.pt")
    print(f"  Loading LLM state dict from {model_pt}")
    sd = torch.load(model_pt, map_location="cpu", weights_only=False)
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    print(f"  {len(sd)} tensors")

    qwen_cfg_path = os.path.join(base, "CosyVoice-BlankEN", "config.json")
    with open(qwen_cfg_path, encoding="utf-8") as f:
        qcfg = json.load(f)
    print(f"  Qwen2 config: hidden={qcfg['hidden_size']}, layers={qcfg['num_hidden_layers']}, "
          f"heads={qcfg['num_attention_heads']}, kv_heads={qcfg['num_key_value_heads']}, "
          f"vocab={qcfg['vocab_size']}")

    writer = gguf.GGUFWriter(output_path, "cosyvoice3-llm")
    writer.add_name("Fun-CosyVoice3-0.5B-2512-LLM")

    # Hparams under cosyvoice3.llm.* (mirror funasr.llm.* convention).
    writer.add_uint32("cosyvoice3.llm.n_layers", int(qcfg["num_hidden_layers"]))
    writer.add_uint32("cosyvoice3.llm.d_model",  int(qcfg["hidden_size"]))
    writer.add_uint32("cosyvoice3.llm.n_heads",  int(qcfg["num_attention_heads"]))
    writer.add_uint32("cosyvoice3.llm.n_kv_heads", int(qcfg["num_key_value_heads"]))
    writer.add_uint32("cosyvoice3.llm.head_dim", int(qcfg.get("head_dim",
                       qcfg["hidden_size"] // qcfg["num_attention_heads"])))
    writer.add_uint32("cosyvoice3.llm.ff_dim",   int(qcfg["intermediate_size"]))
    writer.add_float32("cosyvoice3.llm.rope_theta", float(qcfg.get("rope_theta", 1e6)))
    writer.add_float32("cosyvoice3.llm.rms_norm_eps", float(qcfg.get("rms_norm_eps", 1e-6)))
    writer.add_uint32("cosyvoice3.llm.vocab_size", int(qcfg["vocab_size"]))
    writer.add_uint32("cosyvoice3.llm.max_pos", int(qcfg.get("max_position_embeddings", 32768)))
    writer.add_uint32("cosyvoice3.llm.speech_vocab_size",
                      int(sd["llm_decoder.weight"].shape[0]))  # 6761
    writer.add_uint32("cosyvoice3.llm.speech_token_codebook", 6561)

    # Tokenizer (Qwen2 BPE, lives in CosyVoice-BlankEN/).
    with open(os.path.join(base, "CosyVoice-BlankEN", "vocab.json"), encoding="utf-8") as f:
        vocab_map = json.load(f)
    tokens = [None] * (max(vocab_map.values()) + 1)
    for s, i in vocab_map.items():
        if i < len(tokens):
            tokens[i] = s
    tokens = [t if t is not None else "" for t in tokens]
    print(f"  Qwen2 tokenizer: {len(tokens)} tokens")
    merges = []
    with open(os.path.join(base, "CosyVoice-BlankEN", "merges.txt"), encoding="utf-8") as f:
        for ln in f:
            ln = ln.rstrip("\n")
            if not ln or ln.startswith("#"):
                continue
            merges.append(ln)
    print(f"                   {len(merges)} BPE merges")
    writer.add_tokenizer_model("gpt2")
    writer.add_token_list(tokens)
    if merges:
        writer.add_token_merges(merges)

    n_written = n_f16 = n_f32 = 0
    def write(name, t, *, force_f32=False):
        nonlocal n_written, n_f16, n_f32
        dtype = add_tensor(writer, name, to_np(t), force_f32=force_f32)
        n_written += 1
        if dtype == "F16":
            n_f16 += 1
        else:
            n_f32 += 1

    # LLM body (Qwen2 → llama.cpp-standard names).
    for hf_name in sorted(k for k in sd if k.startswith("llm.")):
        gguf_name = remap_llm(hf_name)
        if gguf_name is None:
            print(f"  WARN: unmapped LLM tensor {hf_name!r}, skipping")
            continue
        # Norms and biases are F32 (small, sensitive); weights F16.
        is_norm_or_bias = ("norm" in gguf_name) or gguf_name.endswith(".bias")
        write(gguf_name, sd[hf_name], force_f32=is_norm_or_bias)

    # Speech-token embedding + decoder (the heads that distinguish CosyVoice's
    # LLM from a vanilla Qwen2 LM).
    write("cosyvoice3.speech_embd.weight",  sd["speech_embedding.weight"])
    write("cosyvoice3.speech_lm_head.weight", sd["llm_decoder.weight"])

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    size = os.path.getsize(output_path)
    print(f"  → {output_path}  ({size / 1e9:.2f} GB, {n_written} tensors; "
          f"F16: {n_f16}, F32: {n_f32})")


# ---------------------------------------------------------------------------
# Flow converter — input_embedding + pre_lookahead + spk_affine + DiT
# ---------------------------------------------------------------------------

def convert_flow(base, output_path):
    import torch
    model_pt = os.path.join(base, "flow.pt")
    print(f"  Loading Flow state dict from {model_pt}")
    sd = torch.load(model_pt, map_location="cpu", weights_only=False)
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    print(f"  {len(sd)} tensors")

    writer = gguf.GGUFWriter(output_path, "cosyvoice3-flow")
    writer.add_name("Fun-CosyVoice3-0.5B-2512-Flow")

    n_dit_layers = 22
    writer.add_uint32("cosyvoice3.flow.n_dit_layers", n_dit_layers)
    writer.add_uint32("cosyvoice3.flow.dit_dim", 1024)
    writer.add_uint32("cosyvoice3.flow.dit_heads", 16)
    writer.add_uint32("cosyvoice3.flow.dit_head_dim", 64)
    writer.add_uint32("cosyvoice3.flow.dit_ff_dim", 2048)
    writer.add_uint32("cosyvoice3.flow.dit_input_dim", 320)  # proj.weight (1024, 320)
    writer.add_uint32("cosyvoice3.flow.mel_dim", 80)
    writer.add_uint32("cosyvoice3.flow.spk_dim_in", 192)
    writer.add_uint32("cosyvoice3.flow.spk_dim_out", 80)
    writer.add_uint32("cosyvoice3.flow.speech_codebook", 6561)
    writer.add_uint32("cosyvoice3.flow.pre_lookahead_len", 3)
    writer.add_uint32("cosyvoice3.flow.token_mel_ratio", 2)
    writer.add_uint32("cosyvoice3.flow.input_frame_rate", 25)
    writer.add_uint32("cosyvoice3.flow.cfm_n_steps", 10)
    writer.add_float32("cosyvoice3.flow.cfm_inference_cfg_rate", 0.7)
    writer.add_float32("cosyvoice3.flow.cfm_sigma_min", 1e-6)
    writer.add_float32("cosyvoice3.flow.rope_theta", 10000.0)  # default DiT RoPE

    n_written = n_f16 = n_f32 = 0
    def write(name, t, *, force_f32=False):
        nonlocal n_written, n_f16, n_f32
        dtype = add_tensor(writer, name, to_np(t), force_f32=force_f32)
        n_written += 1
        if dtype == "F16":
            n_f16 += 1
        else:
            n_f32 += 1

    # Top-level (non-DiT) tensors.
    write("cosyvoice3.flow.input_embd.w",      sd["input_embedding.weight"])
    write("cosyvoice3.flow.pre_la.conv1.w",    sd["pre_lookahead_layer.conv1.weight"])
    write("cosyvoice3.flow.pre_la.conv1.b",    sd["pre_lookahead_layer.conv1.bias"], force_f32=True)
    write("cosyvoice3.flow.pre_la.conv2.w",    sd["pre_lookahead_layer.conv2.weight"])
    write("cosyvoice3.flow.pre_la.conv2.b",    sd["pre_lookahead_layer.conv2.bias"], force_f32=True)
    write("cosyvoice3.flow.spk_affine.w",      sd["spk_embed_affine_layer.weight"])
    write("cosyvoice3.flow.spk_affine.b",      sd["spk_embed_affine_layer.bias"], force_f32=True)

    # DiT input embedding (mel + token + spk concat → 1024-d hidden) + conv-pos.
    write("cosyvoice3.flow.dit.in_proj.w",       sd["decoder.estimator.input_embed.proj.weight"])
    write("cosyvoice3.flow.dit.in_proj.b",       sd["decoder.estimator.input_embed.proj.bias"], force_f32=True)
    write("cosyvoice3.flow.dit.conv_pos.c1.w",   sd["decoder.estimator.input_embed.conv_pos_embed.conv1.0.weight"])
    write("cosyvoice3.flow.dit.conv_pos.c1.b",   sd["decoder.estimator.input_embed.conv_pos_embed.conv1.0.bias"], force_f32=True)
    write("cosyvoice3.flow.dit.conv_pos.c2.w",   sd["decoder.estimator.input_embed.conv_pos_embed.conv2.0.weight"])
    write("cosyvoice3.flow.dit.conv_pos.c2.b",   sd["decoder.estimator.input_embed.conv_pos_embed.conv2.0.bias"], force_f32=True)

    # Time embedding (sinusoidal already implicit; this is the MLP).
    write("cosyvoice3.flow.dit.time_mlp.0.w",    sd["decoder.estimator.time_embed.time_mlp.0.weight"])
    write("cosyvoice3.flow.dit.time_mlp.0.b",    sd["decoder.estimator.time_embed.time_mlp.0.bias"], force_f32=True)
    write("cosyvoice3.flow.dit.time_mlp.2.w",    sd["decoder.estimator.time_embed.time_mlp.2.weight"])
    write("cosyvoice3.flow.dit.time_mlp.2.b",    sd["decoder.estimator.time_embed.time_mlp.2.bias"], force_f32=True)

    # Rotary inv_freq (head_dim/2 = 32). Stored as F32 since it's tiny.
    write("cosyvoice3.flow.dit.rope_inv_freq",   sd["decoder.estimator.rotary_embed.inv_freq"], force_f32=True)

    # 22 DiT transformer blocks.
    for L in range(n_dit_layers):
        p = f"decoder.estimator.transformer_blocks.{L}"
        o = f"cosyvoice3.flow.dit.blk.{L}"
        # AdaLN-Zero modulation: linear (6144, 1024) projecting time-emb to
        # 6 modulation params (γ_attn, β_attn, gate_attn, γ_ff, β_ff, gate_ff).
        write(f"{o}.adaln.w", sd[f"{p}.attn_norm.linear.weight"])
        write(f"{o}.adaln.b", sd[f"{p}.attn_norm.linear.bias"], force_f32=True)
        # MHA Q/K/V/O.
        write(f"{o}.attn.q.w",   sd[f"{p}.attn.to_q.weight"])
        write(f"{o}.attn.q.b",   sd[f"{p}.attn.to_q.bias"], force_f32=True)
        write(f"{o}.attn.k.w",   sd[f"{p}.attn.to_k.weight"])
        write(f"{o}.attn.k.b",   sd[f"{p}.attn.to_k.bias"], force_f32=True)
        write(f"{o}.attn.v.w",   sd[f"{p}.attn.to_v.weight"])
        write(f"{o}.attn.v.b",   sd[f"{p}.attn.to_v.bias"], force_f32=True)
        write(f"{o}.attn.o.w",   sd[f"{p}.attn.to_out.0.weight"])
        write(f"{o}.attn.o.b",   sd[f"{p}.attn.to_out.0.bias"], force_f32=True)
        # FFN (gated; ff.ff.0.0 = up, ff.ff.2 = down).
        write(f"{o}.ffn.l1.w",   sd[f"{p}.ff.ff.0.0.weight"])
        write(f"{o}.ffn.l1.b",   sd[f"{p}.ff.ff.0.0.bias"], force_f32=True)
        write(f"{o}.ffn.l2.w",   sd[f"{p}.ff.ff.2.weight"])
        write(f"{o}.ffn.l2.b",   sd[f"{p}.ff.ff.2.bias"], force_f32=True)

    # Final AdaLN-Zero + proj_out.
    write("cosyvoice3.flow.dit.norm_out.w",   sd["decoder.estimator.norm_out.linear.weight"])
    write("cosyvoice3.flow.dit.norm_out.b",   sd["decoder.estimator.norm_out.linear.bias"], force_f32=True)
    write("cosyvoice3.flow.dit.proj_out.w",   sd["decoder.estimator.proj_out.weight"])
    write("cosyvoice3.flow.dit.proj_out.b",   sd["decoder.estimator.proj_out.bias"], force_f32=True)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    size = os.path.getsize(output_path)
    print(f"  → {output_path}  ({size / 1e9:.2f} GB, {n_written} tensors; "
          f"F16: {n_f16}, F32: {n_f32})")


# ---------------------------------------------------------------------------
# HiFT converter — resolves weight-norm, mel → 24 kHz waveform via iSTFT.
# ---------------------------------------------------------------------------

def resolve_wn(sd, prefix):
    """Take a `<prefix>.parametrizations.weight.original{0,1}` pair, return
    plain reconstructed weight tensor."""
    g_key = f"{prefix}.parametrizations.weight.original0"
    v_key = f"{prefix}.parametrizations.weight.original1"
    if g_key not in sd or v_key not in sd:
        raise KeyError(f"missing weight_norm parametrisation for {prefix}")
    g = sd[g_key]
    v = sd[v_key]
    import torch
    g_np = g.detach().float().numpy() if isinstance(g, torch.Tensor) else g.astype(np.float32)
    v_np = v.detach().float().numpy() if isinstance(v, torch.Tensor) else v.astype(np.float32)
    return wn_resolve(g_np, v_np)


def convert_hift(base, output_path):
    import torch
    model_pt = os.path.join(base, "hift.pt")
    print(f"  Loading HiFT state dict from {model_pt}")
    sd = torch.load(model_pt, map_location="cpu", weights_only=False)
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    print(f"  {len(sd)} tensors (pre weight-norm resolution)")

    writer = gguf.GGUFWriter(output_path, "cosyvoice3-hift")
    writer.add_name("Fun-CosyVoice3-0.5B-2512-HiFT")

    # Architecture hparams (from cosyvoice3.yaml).
    writer.add_uint32("cosyvoice3.hift.sample_rate", 24000)
    writer.add_uint32("cosyvoice3.hift.mel_dim", 80)
    writer.add_uint32("cosyvoice3.hift.base_channels", 512)
    writer.add_uint32("cosyvoice3.hift.nb_harmonics", 8)
    writer.add_uint32("cosyvoice3.hift.istft_n_fft", 16)
    writer.add_uint32("cosyvoice3.hift.istft_hop", 4)
    writer.add_uint32("cosyvoice3.hift.n_upsample_stages", 3)
    # Upsample rates [8, 5, 3], kernels [16, 11, 7]
    for i, (r, k) in enumerate([(8, 16), (5, 11), (3, 7)]):
        writer.add_uint32(f"cosyvoice3.hift.upsample_rate.{i}", r)
        writer.add_uint32(f"cosyvoice3.hift.upsample_kernel.{i}", k)

    # 3 resblocks per upsample stage = 9 total.
    n_resblocks = 9

    n_written = 0
    def write_np(name, arr, *, force_f32=False):
        nonlocal n_written
        dtype = add_tensor(writer, name, arr, force_f32=force_f32)
        n_written += 1

    # conv_pre / conv_post
    write_np("cosyvoice3.hift.conv_pre.w", resolve_wn(sd, "conv_pre"))
    write_np("cosyvoice3.hift.conv_pre.b", to_np(sd["conv_pre.bias"]), force_f32=True)
    write_np("cosyvoice3.hift.conv_post.w", resolve_wn(sd, "conv_post"))
    write_np("cosyvoice3.hift.conv_post.b", to_np(sd["conv_post.bias"]), force_f32=True)

    # Upsample stages (ConvTranspose1d).
    for i in range(3):
        write_np(f"cosyvoice3.hift.ups.{i}.w", resolve_wn(sd, f"ups.{i}"))
        write_np(f"cosyvoice3.hift.ups.{i}.b", to_np(sd[f"ups.{i}.bias"]), force_f32=True)

    # ResBlocks (each has 3 convs1 + 3 convs2 + activations1/2 with alpha).
    for i in range(n_resblocks):
        for j in range(3):
            write_np(f"cosyvoice3.hift.resblocks.{i}.c1.{j}.w", resolve_wn(sd, f"resblocks.{i}.convs1.{j}"))
            write_np(f"cosyvoice3.hift.resblocks.{i}.c1.{j}.b", to_np(sd[f"resblocks.{i}.convs1.{j}.bias"]), force_f32=True)
            write_np(f"cosyvoice3.hift.resblocks.{i}.c2.{j}.w", resolve_wn(sd, f"resblocks.{i}.convs2.{j}"))
            write_np(f"cosyvoice3.hift.resblocks.{i}.c2.{j}.b", to_np(sd[f"resblocks.{i}.convs2.{j}.bias"]), force_f32=True)
            write_np(f"cosyvoice3.hift.resblocks.{i}.a1.{j}.alpha", to_np(sd[f"resblocks.{i}.activations1.{j}.alpha"]), force_f32=True)
            write_np(f"cosyvoice3.hift.resblocks.{i}.a2.{j}.alpha", to_np(sd[f"resblocks.{i}.activations2.{j}.alpha"]), force_f32=True)

    # Source side (NSF F0-conditioned chain). NB: `source_downs.*` are
    # plain Conv1d (no weight_norm); `source_resblocks.*` are weight-normed
    # just like the main `resblocks`. Asymmetric layout — easy trap.
    for i in range(3):
        write_np(f"cosyvoice3.hift.source_downs.{i}.w", to_np(sd[f"source_downs.{i}.weight"]))
        write_np(f"cosyvoice3.hift.source_downs.{i}.b", to_np(sd[f"source_downs.{i}.bias"]), force_f32=True)
        for j in range(3):
            write_np(f"cosyvoice3.hift.src_resblk.{i}.c1.{j}.w", resolve_wn(sd, f"source_resblocks.{i}.convs1.{j}"))
            write_np(f"cosyvoice3.hift.src_resblk.{i}.c1.{j}.b", to_np(sd[f"source_resblocks.{i}.convs1.{j}.bias"]), force_f32=True)
            write_np(f"cosyvoice3.hift.src_resblk.{i}.c2.{j}.w", resolve_wn(sd, f"source_resblocks.{i}.convs2.{j}"))
            write_np(f"cosyvoice3.hift.src_resblk.{i}.c2.{j}.b", to_np(sd[f"source_resblocks.{i}.convs2.{j}.bias"]), force_f32=True)
            write_np(f"cosyvoice3.hift.src_resblk.{i}.a1.{j}.alpha", to_np(sd[f"source_resblocks.{i}.activations1.{j}.alpha"]), force_f32=True)
            write_np(f"cosyvoice3.hift.src_resblk.{i}.a2.{j}.alpha", to_np(sd[f"source_resblocks.{i}.activations2.{j}.alpha"]), force_f32=True)

    # SineGen source linear projection.
    write_np("cosyvoice3.hift.m_source.l_linear.w", to_np(sd["m_source.l_linear.weight"]), force_f32=True)
    write_np("cosyvoice3.hift.m_source.l_linear.b", to_np(sd["m_source.l_linear.bias"]), force_f32=True)

    # F0 predictor: 5× conv1d (condnet.{0,2,4,6,8}) + classifier.
    for i, k in enumerate([0, 2, 4, 6, 8]):
        write_np(f"cosyvoice3.hift.f0.condnet.{i}.w", resolve_wn(sd, f"f0_predictor.condnet.{k}"))
        write_np(f"cosyvoice3.hift.f0.condnet.{i}.b", to_np(sd[f"f0_predictor.condnet.{k}.bias"]), force_f32=True)
    write_np("cosyvoice3.hift.f0.classifier.w", to_np(sd["f0_predictor.classifier.weight"]), force_f32=True)
    write_np("cosyvoice3.hift.f0.classifier.b", to_np(sd["f0_predictor.classifier.bias"]), force_f32=True)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    size = os.path.getsize(output_path)
    print(f"  → {output_path}  ({size / 1e9:.2f} GB, {n_written} tensors)")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="HF model ID or local snapshot dir")
    ap.add_argument("--output-dir", required=True,
                    help="Directory to write the three GGUFs into")
    ap.add_argument("--skip", choices=["llm", "flow", "hift"], action="append", default=[],
                    help="Skip a sub-model (repeatable; useful for iterating).")
    args = ap.parse_args()

    if os.path.isdir(args.input):
        base = args.input
    else:
        from huggingface_hub import snapshot_download
        print(f"Downloading {args.input}")
        base = snapshot_download(
            repo_id=args.input,
            allow_patterns=["*.pt", "*.yaml", "*.json", "*.txt", "CosyVoice-BlankEN/*"])
    print(f"Loading from {base}")
    os.makedirs(args.output_dir, exist_ok=True)

    if "llm" not in args.skip:
        print()
        print("=== LLM (Qwen2 + speech heads) ===")
        convert_llm(base, os.path.join(args.output_dir, "cosyvoice3-llm-f16.gguf"))

    if "flow" not in args.skip:
        print()
        print("=== Flow (DiT-CFM) ===")
        convert_flow(base, os.path.join(args.output_dir, "cosyvoice3-flow-f16.gguf"))

    if "hift" not in args.skip:
        print()
        print("=== HiFT (vocoder) ===")
        convert_hift(base, os.path.join(args.output_dir, "cosyvoice3-hift-f16.gguf"))

    print()
    print("Done.")


if __name__ == "__main__":
    main()
