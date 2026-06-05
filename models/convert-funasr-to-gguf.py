#!/usr/bin/env python3
"""Convert FunAudioLLM/Fun-ASR-Nano-2512 (and MLT-Nano variant) to GGUF.

Published architecture from upstream config.yaml mentions a CTC decoder + head;
the published model.pt **does not ship CTC weights** (verified across both
Fun-ASR-Nano-2512 and Fun-ASR-MLT-Nano-2512: state_dict has only audio_encoder /
audio_adaptor / llm prefixes, zero ctc keys). The only viable inference path
is therefore the LLM-decoder one.

Inference pipeline reproduced by the C++ runtime:

  audio(16kHz)
    → WavFrontend (kaldi hamming fbank + LFR(m=7,n=6))
                                                   (T_lfr, 560) F32
    → SenseVoiceEncoderSmall (70 SANM blocks)      (T_lfr, 512) F32
    → audio_adaptor                                (T_adapter, 1024) F32
        linear1 (512→2048)
        relu
        linear2 (2048→1024)
        2 × MultiHeadedAttention blocks (16 heads @ 1024)
    → splice adapter output into Qwen3-0.6B input embeds at the
      `<|startofspeech|>` / `<|endofspeech|>` slot of the ChatML prompt
    → Qwen3-0.6B AR decode with KV cache (max 512 new tokens)
    → detokenize via Qwen3 BPE


Tensor naming (kept under GGUF 64-char limit):

  Encoder (NEW prefix `funasr.enc.`):
    funasr.enc.blk.K.norm{1,2}.{w,b}        SANM block LN
    funasr.enc.blk.K.attn.qkv.{w,b}         fused Q/K/V linear (3*512 out)
    funasr.enc.blk.K.attn.out.{w,b}         output proj
    funasr.enc.blk.K.attn.fsmn.w            depthwise conv1d k=11
    funasr.enc.blk.K.ffn.l{1,2}.{w,b}       PositionwiseFeedForward 512↔2048
    funasr.enc.after_norm.{w,b}             LN between 50-block and 20-block
    funasr.enc.tp_norm.{w,b}                final encoder LN

  Audio adaptor (NEW prefix `funasr.adaptor.`):
    funasr.adaptor.linear{1,2}.{w,b}        512→2048→1024 prelude
    funasr.adaptor.blk.K.norm{1,2}.{w,b}    transformer block LN
    funasr.adaptor.blk.K.attn.{q,k,v,out}.{w,b}  separate Q/K/V/O linears
    funasr.adaptor.blk.K.ffn.l{1,2}.{w,b}   PositionwiseFeedForward 1024↔256

  LLM (llama.cpp-standard naming so the runtime can reuse the qwen3_asr /
  voxtral / granite_speech load patterns):
    token_embd.weight                       embed_tokens (151936, 1024)
    output.weight                           lm_head      (151936, 1024)
    output_norm.weight                      final RMSNorm
    blk.K.attn_norm.weight                  input_layernorm
    blk.K.attn_q.weight                     q_proj
    blk.K.attn_k.weight                     k_proj
    blk.K.attn_v.weight                     v_proj
    blk.K.attn_output.weight                o_proj
    blk.K.attn_q_norm.weight                q_norm (Qwen3-specific)
    blk.K.attn_k_norm.weight                k_norm (Qwen3-specific)
    blk.K.ffn_norm.weight                   post_attention_layernorm
    blk.K.ffn_gate.weight                   gate_proj
    blk.K.ffn_up.weight                     up_proj
    blk.K.ffn_down.weight                   down_proj


Usage:
  python models/convert-funasr-to-gguf.py \\
      --input FunAudioLLM/Fun-ASR-Nano-2512 \\
      --output /Volumes/backups/ai/crispasr-models/funasr-nano-2512/funasr-nano-2512-f16.gguf

  python models/convert-funasr-to-gguf.py \\
      --input FunAudioLLM/Fun-ASR-MLT-Nano-2512 \\
      --output /Volumes/backups/ai/crispasr-models/funasr-mlt-nano-2512/funasr-mlt-nano-2512-f16.gguf
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
# Encoder block index ↔ state-dict key family
# ---------------------------------------------------------------------------

def enc_block_path(layer_idx: int) -> str:
    """Returns the upstream state-dict path prefix for the SANM block at
    flat index `layer_idx` (0..69). 0 is the in-size-560 entry block."""
    if layer_idx == 0:
        return "audio_encoder.encoders0.0"
    if layer_idx < 50:
        return f"audio_encoder.encoders.{layer_idx - 1}"
    return f"audio_encoder.tp_encoders.{layer_idx - 50}"


# ---------------------------------------------------------------------------
# LLM tensor renaming (HF Qwen3 → llama.cpp standard)
# ---------------------------------------------------------------------------

LLM_DIRECT = {
    "llm.model.embed_tokens.weight": "token_embd.weight",
    "llm.model.norm.weight":         "output_norm.weight",
    "llm.lm_head.weight":            "output.weight",
}

LLM_LAYER_PATTERNS = [
    (r"llm\.model\.layers\.(\d+)\.input_layernorm\.weight",
     "blk.{}.attn_norm.weight"),
    (r"llm\.model\.layers\.(\d+)\.self_attn\.q_proj\.weight",
     "blk.{}.attn_q.weight"),
    (r"llm\.model\.layers\.(\d+)\.self_attn\.k_proj\.weight",
     "blk.{}.attn_k.weight"),
    (r"llm\.model\.layers\.(\d+)\.self_attn\.v_proj\.weight",
     "blk.{}.attn_v.weight"),
    (r"llm\.model\.layers\.(\d+)\.self_attn\.o_proj\.weight",
     "blk.{}.attn_output.weight"),
    (r"llm\.model\.layers\.(\d+)\.self_attn\.q_norm\.weight",
     "blk.{}.attn_q_norm.weight"),
    (r"llm\.model\.layers\.(\d+)\.self_attn\.k_norm\.weight",
     "blk.{}.attn_k_norm.weight"),
    (r"llm\.model\.layers\.(\d+)\.post_attention_layernorm\.weight",
     "blk.{}.ffn_norm.weight"),
    (r"llm\.model\.layers\.(\d+)\.mlp\.gate_proj\.weight",
     "blk.{}.ffn_gate.weight"),
    (r"llm\.model\.layers\.(\d+)\.mlp\.up_proj\.weight",
     "blk.{}.ffn_up.weight"),
    (r"llm\.model\.layers\.(\d+)\.mlp\.down_proj\.weight",
     "blk.{}.ffn_down.weight"),
]


def remap_llm(hf_name: str) -> str | None:
    if hf_name in LLM_DIRECT:
        return LLM_DIRECT[hf_name]
    for pat, tmpl in LLM_LAYER_PATTERNS:
        m = re.match(pat, hf_name)
        if m:
            return tmpl.format(m.group(1))
    return None


def to_np(t):
    """Detach + bf16-safe convert to numpy F32 array."""
    import torch
    t = t.detach()
    if t.dtype == torch.bfloat16:
        return t.float().numpy()
    return t.numpy()


def add_tensor(writer, name: str, t: np.ndarray, *, force_f32: bool = False):
    """Write `t` to GGUF. F16 for multi-dim weights, F32 for everything else
    (LNs, biases, 1-D tensors, and embeddings when force_f32 is set)."""
    assert len(name) < 64, f"GGUF name too long ({len(name)}): {name}"
    if not force_f32 and t.ndim >= 2:
        data = np.ascontiguousarray(t.astype(np.float16))
        dtype = "F16"
    else:
        data = np.ascontiguousarray(t.astype(np.float32))
        dtype = "F32"
    writer.add_tensor(name, data)
    return dtype


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True,
                    help="HF model ID or local snapshot dir")
    ap.add_argument("--output", required=True, help="Output GGUF path")
    args = ap.parse_args()

    import torch
    from huggingface_hub import snapshot_download

    if os.path.isdir(args.input):
        base = args.input
    else:
        print(f"Downloading {args.input}")
        base = snapshot_download(
            repo_id=args.input,
            allow_patterns=["model.pt", "config.yaml", "configuration.json",
                            "multilingual.tiktoken", "Qwen3-0.6B/*"],
        )
    print(f"Loading from {base}")

    model_pt = os.path.join(base, "model.pt")
    qwen_dir = os.path.join(base, "Qwen3-0.6B")
    qwen_cfg_path = os.path.join(qwen_dir, "config.json")

    print("  Loading state dict ...")
    raw = torch.load(model_pt, map_location="cpu", weights_only=False)
    sd = raw["state_dict"] if isinstance(raw, dict) and "state_dict" in raw else raw
    print(f"  {len(sd)} tensors")

    # ---- Read Qwen3-0.6B config (LLM hyperparameters) ----
    with open(qwen_cfg_path, encoding="utf-8") as f:
        llm_cfg = json.load(f)
    print(f"  Qwen3 config: hidden={llm_cfg['hidden_size']}, layers={llm_cfg['num_hidden_layers']}, "
          f"heads={llm_cfg['num_attention_heads']}, kv_heads={llm_cfg['num_key_value_heads']}, "
          f"vocab={llm_cfg['vocab_size']}")

    # ---- Hyperparameters under funasr.* (encoder + adaptor) and llm.* (Qwen3) ----
    enc_hp = dict(
        n_mels=80, lfr_m=7, lfr_n=6, sample_rate=16000,
        frame_length_ms=25, frame_shift_ms=10,
        d_model=512, n_heads=4, ffn_dim=2048,
        n_blocks_base=50, n_blocks_tp=20, sanm_kernel=11,
    )
    ada_hp = dict(
        ada_d_in=512, ada_ffn=2048, ada_d_out=1024,
        ada_n_layers=2, ada_n_heads=8, ada_ffn_inner=256,
    )

    # Read use_low_frame_rate from config.yaml (audio_adaptor_conf section).
    # Fun-ASR-Nano-2512 sets it to true; Fun-ASR-MLT-Nano-2512 omits it
    # (upstream default is false). Controls how many adaptor output frames
    # are spliced into the LLM prompt — with true only the first
    # ((T-1)/2+1 …)×3 frames are used; with false all T frames are used.
    import yaml
    cfg_path = os.path.join(base, "config.yaml")
    use_low_frame_rate = False  # upstream default when key is absent
    if os.path.isfile(cfg_path):
        with open(cfg_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        ada_conf = cfg.get("audio_adaptor_conf", {})
        use_low_frame_rate = ada_conf.get("use_low_frame_rate", False)

    # ---- Open GGUF writer ----
    short_name = ("Fun-ASR-MLT-Nano-2512" if "MLT" in args.input
                  else "Fun-ASR-Nano-2512" if "Nano-2512" in args.input
                  else "Fun-ASR")
    writer = gguf.GGUFWriter(args.output, "funasr")
    writer.add_name(short_name)
    for k, v in {**enc_hp, **ada_hp}.items():
        writer.add_uint32(f"funasr.{k}", int(v))
    writer.add_bool("funasr.use_low_frame_rate", use_low_frame_rate)

    # LLM hyperparameters — match qwen3_asr conventions but under funasr.llm.*
    writer.add_uint32("funasr.llm.n_layers", int(llm_cfg["num_hidden_layers"]))
    writer.add_uint32("funasr.llm.d_model",  int(llm_cfg["hidden_size"]))
    writer.add_uint32("funasr.llm.n_heads",  int(llm_cfg["num_attention_heads"]))
    writer.add_uint32("funasr.llm.n_kv_heads", int(llm_cfg["num_key_value_heads"]))
    writer.add_uint32("funasr.llm.head_dim", int(llm_cfg.get("head_dim", llm_cfg["hidden_size"] // llm_cfg["num_attention_heads"])))
    writer.add_uint32("funasr.llm.ff_dim",   int(llm_cfg["intermediate_size"]))
    writer.add_float32("funasr.llm.rope_theta", float(llm_cfg.get("rope_theta", 1e6)))
    writer.add_float32("funasr.llm.rms_norm_eps", float(llm_cfg.get("rms_norm_eps", 1e-6)))
    writer.add_uint32("funasr.llm.vocab_size", int(llm_cfg["vocab_size"]))
    writer.add_uint32("funasr.llm.max_pos", int(llm_cfg.get("max_position_embeddings", 32768)))

    # ---- Tokenizer (Qwen3-0.6B GPT2-style BPE) ----
    with open(os.path.join(qwen_dir, "tokenizer.json"), encoding="utf-8") as f:
        tok = json.load(f)
    vocab_map = tok["model"]["vocab"]                     # str -> int
    tokens = [None] * (max(vocab_map.values()) + 1)
    for s, i in vocab_map.items():
        if i < len(tokens):
            tokens[i] = s
    # Added tokens (FunASR-specific specials live here, e.g. <|startofspeech|>)
    added: list[dict] = tok.get("added_tokens", []) or []
    for at in added:
        i = at["id"]
        s = at["content"]
        if i >= len(tokens):
            tokens.extend([None] * (i - len(tokens) + 1))
        tokens[i] = s
    tokens = [t if t is not None else "" for t in tokens]
    merges = [" ".join(m) if isinstance(m, list) else m for m in tok["model"].get("merges", [])]
    print(f"  Tokenizer: {len(tokens)} tokens, {len(merges)} merges, {len(added)} added")

    writer.add_tokenizer_model("gpt2")
    writer.add_token_list(tokens)
    if merges:
        writer.add_token_merges(merges)
    # Record FunASR-specific special-token IDs so the runtime can find them.
    name_to_id = {at["content"]: at["id"] for at in added}
    if "<|startofspeech|>" in name_to_id:
        writer.add_uint32("funasr.audio_start_token_id", name_to_id["<|startofspeech|>"])
    if "<|endofspeech|>" in name_to_id:
        writer.add_uint32("funasr.audio_end_token_id", name_to_id["<|endofspeech|>"])
    # EOS / pad come from generation_config.json or tokenizer_config.json
    try:
        with open(os.path.join(qwen_dir, "generation_config.json"), encoding="utf-8") as f:
            gcfg = json.load(f)
        if "eos_token_id" in gcfg:
            eos = gcfg["eos_token_id"]
            if isinstance(eos, list):
                eos = eos[0]
            writer.add_uint32("funasr.eos_token_id", int(eos))
        if "pad_token_id" in gcfg:
            writer.add_uint32("funasr.pad_token_id", int(gcfg["pad_token_id"]))
    except FileNotFoundError:
        pass

    # ---- Tensor write helpers ----
    n_written = 0
    n_f16 = 0
    n_f32 = 0

    def write(name, tensor, *, force_f32=False):
        nonlocal n_written, n_f16, n_f32
        t = to_np(tensor)
        dtype = add_tensor(writer, name, t, force_f32=force_f32)
        if dtype == "F16":
            n_f16 += 1
        else:
            n_f32 += 1
        n_written += 1

    # ---- SANM encoder (70 blocks) ----
    for L in range(enc_hp["n_blocks_base"] + enc_hp["n_blocks_tp"]):
        p = enc_block_path(L)
        o = f"funasr.enc.blk.{L}"
        write(f"{o}.norm1.w",     sd[f"{p}.norm1.weight"], force_f32=True)
        write(f"{o}.norm1.b",     sd[f"{p}.norm1.bias"], force_f32=True)
        write(f"{o}.norm2.w",     sd[f"{p}.norm2.weight"], force_f32=True)
        write(f"{o}.norm2.b",     sd[f"{p}.norm2.bias"], force_f32=True)
        write(f"{o}.attn.qkv.w",  sd[f"{p}.self_attn.linear_q_k_v.weight"])
        write(f"{o}.attn.qkv.b",  sd[f"{p}.self_attn.linear_q_k_v.bias"], force_f32=True)
        write(f"{o}.attn.out.w",  sd[f"{p}.self_attn.linear_out.weight"])
        write(f"{o}.attn.out.b",  sd[f"{p}.self_attn.linear_out.bias"], force_f32=True)
        # Depthwise conv weight (n_feat, 1, K). Squeeze the middle dim so the
        # GGUF carries (n_feat, K) which is what ggml_conv_2d_dw_direct expects.
        fsmn = to_np(sd[f"{p}.self_attn.fsmn_block.weight"])
        if fsmn.ndim == 3 and fsmn.shape[1] == 1:
            fsmn = fsmn.squeeze(1)
        dt = add_tensor(writer, f"{o}.attn.fsmn.w", fsmn)
        n_written += 1
        (n_f16 if dt == "F16" else n_f32)  # tally indirectly below
        if dt == "F16":
            n_f16 += 1
        else:
            n_f32 += 1
        write(f"{o}.ffn.l1.w",    sd[f"{p}.feed_forward.w_1.weight"])
        write(f"{o}.ffn.l1.b",    sd[f"{p}.feed_forward.w_1.bias"], force_f32=True)
        write(f"{o}.ffn.l2.w",    sd[f"{p}.feed_forward.w_2.weight"])
        write(f"{o}.ffn.l2.b",    sd[f"{p}.feed_forward.w_2.bias"], force_f32=True)

    write("funasr.enc.after_norm.w", sd["audio_encoder.after_norm.weight"], force_f32=True)
    write("funasr.enc.after_norm.b", sd["audio_encoder.after_norm.bias"], force_f32=True)
    write("funasr.enc.tp_norm.w",    sd["audio_encoder.tp_norm.weight"], force_f32=True)
    write("funasr.enc.tp_norm.b",    sd["audio_encoder.tp_norm.bias"], force_f32=True)

    # ---- Audio adaptor (linear1 + linear2 + 2 transformer blocks) ----
    write("funasr.adaptor.linear1.w", sd["audio_adaptor.linear1.weight"])
    write("funasr.adaptor.linear1.b", sd["audio_adaptor.linear1.bias"], force_f32=True)
    write("funasr.adaptor.linear2.w", sd["audio_adaptor.linear2.weight"])
    write("funasr.adaptor.linear2.b", sd["audio_adaptor.linear2.bias"], force_f32=True)
    for L in range(ada_hp["ada_n_layers"]):
        pp = f"audio_adaptor.blocks.{L}"
        o = f"funasr.adaptor.blk.{L}"
        write(f"{o}.norm1.w",   sd[f"{pp}.norm1.weight"], force_f32=True)
        write(f"{o}.norm1.b",   sd[f"{pp}.norm1.bias"], force_f32=True)
        write(f"{o}.norm2.w",   sd[f"{pp}.norm2.weight"], force_f32=True)
        write(f"{o}.norm2.b",   sd[f"{pp}.norm2.bias"], force_f32=True)
        write(f"{o}.attn.q.w",  sd[f"{pp}.self_attn.linear_q.weight"])
        write(f"{o}.attn.q.b",  sd[f"{pp}.self_attn.linear_q.bias"], force_f32=True)
        write(f"{o}.attn.k.w",  sd[f"{pp}.self_attn.linear_k.weight"])
        write(f"{o}.attn.k.b",  sd[f"{pp}.self_attn.linear_k.bias"], force_f32=True)
        write(f"{o}.attn.v.w",  sd[f"{pp}.self_attn.linear_v.weight"])
        write(f"{o}.attn.v.b",  sd[f"{pp}.self_attn.linear_v.bias"], force_f32=True)
        write(f"{o}.attn.out.w", sd[f"{pp}.self_attn.linear_out.weight"])
        write(f"{o}.attn.out.b", sd[f"{pp}.self_attn.linear_out.bias"], force_f32=True)
        write(f"{o}.ffn.l1.w",  sd[f"{pp}.feed_forward.w_1.weight"])
        write(f"{o}.ffn.l1.b",  sd[f"{pp}.feed_forward.w_1.bias"], force_f32=True)
        write(f"{o}.ffn.l2.w",  sd[f"{pp}.feed_forward.w_2.weight"])
        write(f"{o}.ffn.l2.b",  sd[f"{pp}.feed_forward.w_2.bias"], force_f32=True)

    # ---- LLM (Qwen3-0.6B) — llama.cpp-standard naming ----
    n_llm = 0
    for hf_name in sorted(k for k in sd if k.startswith("llm.")):
        gguf_name = remap_llm(hf_name)
        if gguf_name is None:
            # ignore any unmapped LLM tensors (lm_head.bias etc.); print so
            # we know if upstream adds new keys.
            print(f"  WARN: unmapped LLM tensor {hf_name!r}, skipping")
            continue
        is_norm = "norm" in gguf_name or gguf_name.endswith(".bias")
        write(gguf_name, sd[hf_name], force_f32=is_norm)
        n_llm += 1
    print(f"  LLM: {n_llm} tensors written")

    print()
    print(f"  Total {n_written} tensors  (F16: {n_f16}, F32: {n_f32})")

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    size = os.path.getsize(args.output)
    print(f"\nDone: {args.output} ({size / 1e9:.2f} GB)")


if __name__ == "__main__":
    main()
