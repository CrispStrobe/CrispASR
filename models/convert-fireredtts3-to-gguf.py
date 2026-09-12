#!/usr/bin/env python3
"""Convert FireRedTeam/FireRedTTS3 → GGUF (#377).

FireRedTTS3 is a continuous-latent AR zero-shot TTS system:
  - Qwen3-1.7B LLM backbone (28L, hidden=2048, 16Q/8KV heads, head_dim=128,
    QK-norm, rope_theta=1e6) consuming [spk_emb, text tokens, latent patches]
  - PatchEncoder: 8L bidirectional transformer (hidden=1024, ratio 4); each
    patch of 4 RedAE latent frames + a [CLS] token → one 2048-d LLM embedding
  - DiT flow-matching head: 11L AdaLN blocks (hidden=1024, ratio 3, +conv
    branch), input = concat(latents 64, llm cond 1024, spk cond 512) = 1600;
    denoises one 4-frame patch per AR step over a 2-patch clean history
  - stop_head: Linear(2048→1) sigmoid stop predictor
  - RedAE: continuous 64-d 25 Hz latent autoencoder at 24 kHz
      encoder: 480-sample patches (50 Hz) → 18L Qwen3 (896, SW=64) →
               CLS 2x downsample (4L Qwen3) → Linear → 64-d
      decoder: 64 → 2x upsample MLP → 18L Qwen3 (896, SW=64) →
               Vocos-style ISTFT head (n_fft=1920, hop=480, exp-mag clip 100)
  - CAM++ speaker encoder (3D-Speaker CAMPPlus, 80 fbank → 512-d x-vector)

Outputs two GGUFs:
  fireredtts3-<variant>-f16.gguf  — LLM + PatchEncoder + DiT + heads + tokenizer
                                    (+ optional baked default voice prompt)
  fireredtts3-redae-f16.gguf      — RedAE encoder+decoder + CAM++ (campplus.*)

The three unused 151936x896 `embed_tokens` tables inside RedAE's Qwen3 stacks
(inputs are always inputs_embeds) are dropped — 1.6 GB of dead fp32 weight.

The checkpoint SHIPS the ISTFT hann window (decoder.istft_head.istft.window);
it is copied verbatim per HARD RULE #2b.

Tokenizer: Qwen2 byte-level BPE from text_tokenizer/tokenizer.json, plus the
255 FireRed special tokens appended at runtime by
fireredtts3.utils.text_tokenizer.load_text_tokenizer — replicated here in the
exact same order so C++ ids match transformers' ids.

Usage:
  python models/convert-fireredtts3-to-gguf.py \
      --model-dir /path/to/FireRedTTS3-snapshot \
      --output-dir out/ [--variant base] \
      [--default-prompt-npz prompt.npz --default-prompt-text "..." \
       --default-prompt-language English]
"""

import argparse
import json
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


# ── FireRed runtime special tokens (fireredtts3/utils/text_tokenizer.py) ──
# Order matters: appended to the tokenizer in exactly this order.
MULTI_LANG_TAGS = [
    "<|Chinese|>", "<|English|>", "<|Cantonese|>",
    "<|Japanese|>", "<|Korean|>", "<|Spanish|>",
    "<|French|>", "<|Russian|>", "<|Arabic|>",
    "<|Turkish|>", "<|Indonesian|>", "<|Portuguese|>",
    "<|Italian|>", "<|Dutch|>", "<|Vietnamese|>",
    "<|German|>", "<|Ukrainian|>", "<|Thai|>",
    "<|Polish|>", "<|Romanian|>", "<|Greek|>",
    "<|Czech|>", "<|Finnish|>", "<|Hindi|>",
]
MULTI_DIALECT_TAGS = [
    "<|ZH_Anhui|>", "<|ZH_Fujian|>", "<|ZH_Gansu|>",
    "<|ZH_Guizhou|>", "<|ZH_Hebei|>", "<|ZH_Henan|>",
    "<|ZH_Hubei|>", "<|ZH_Hunan|>", "<|ZH_Jiangxi|>",
    "<|ZH_Liaoning|>", "<|ZH_Minnan|>", "<|ZH_Ningxia|>",
    "<|ZH_Shaanxi|>", "<|ZH_Shandong|>", "<|ZH_Shanghai|>",
    "<|ZH_Shanxi|>", "<|ZH_Sichuan|>", "<|ZH_Tianjin|>",
    "<|ZH_Wenzhou|>", "<|ZH_Wu|>", "<|ZH_Yunnan|>",
]
SPECIAL_TOKENS = (
    [
        "<|sosp|>", "<|eosp|>", "<|empty|>", "<|Human|>", "<|SpeechLM|>",
        "<|sostm|>", "<|eostm|>", "<|sot|>", "<|eot|>",
        "<|TEXT_ONLY|>", "<|AUDIO_ONLY|>", "<|ASR|>", "<|TTS|>",
        "<|INTERLEAVE|>", "<|UNDERSTANDING|>",
    ]
    + [f"<|placeholder_{i:03d}|>" for i in range(1, 193)]
    + MULTI_LANG_TAGS
    + MULTI_DIALECT_TAGS
    + ["<|edit|>", "<|frame_patch|>", "<|end_edit|>"]
)


# ── Helpers ──────────────────────────────────────────────────────────

def to_f16(t: torch.Tensor) -> np.ndarray:
    return t.detach().to(torch.float16).numpy()


def to_f32(t: torch.Tensor) -> np.ndarray:
    return t.detach().to(torch.float32).numpy()


def choose_dtype(name: str, t: torch.Tensor):
    """Norms/biases/small/conditioning → F32; bulk matmul weights → F16."""
    n = t.numel()
    if t.dim() <= 1 or n < 256:
        return to_f32(t), GGMLQuantizationType.F32
    always_f32 = (
        "norm" in name
        or "bias" in name
        or "adaln" in name
        or "time_mlp" in name
        or "cls_tok" in name
        or "istft" in name
        or "stop_head" in name
        or "dprompt" in name
        or "running_" in name  # campplus BN stats
    )
    if always_f32:
        return to_f32(t), GGMLQuantizationType.F32
    return to_f16(t), GGMLQuantizationType.F16


def add_tensor(w: GGUFWriter, name: str, t: torch.Tensor):
    assert len(name) < 64, f"tensor name too long ({len(name)}): {name}"
    data, _ = choose_dtype(name, t)
    w.add_tensor(name, data)


# ── Name mapping ─────────────────────────────────────────────────────

def _map_qwen3_suffix(n: str) -> str:
    n = n.replace(".self_attn.q_proj.", ".q.")
    n = n.replace(".self_attn.k_proj.", ".k.")
    n = n.replace(".self_attn.v_proj.", ".v.")
    n = n.replace(".self_attn.o_proj.", ".o.")
    n = n.replace(".self_attn.q_norm.", ".q_norm.")
    n = n.replace(".self_attn.k_norm.", ".k_norm.")
    n = n.replace(".input_layernorm.", ".attn_norm.")
    n = n.replace(".post_attention_layernorm.", ".ffn_norm.")
    n = n.replace(".mlp.gate_proj.", ".gate.")
    n = n.replace(".mlp.up_proj.", ".up.")
    n = n.replace(".mlp.down_proj.", ".down.")
    return n


def _map_ditblock_suffix(n: str) -> str:
    """Common DiTBlock/Attention/FeedForward suffixes (modules.py + dit.py)."""
    n = n.replace(".attn.to_q.", ".q.")
    n = n.replace(".attn.to_k.", ".k.")
    n = n.replace(".attn.to_v.", ".v.")
    n = n.replace(".attn.to_out.0.", ".o.")
    n = n.replace(".mlp.ff.0.0.", ".ffn_up.")
    n = n.replace(".mlp.ff.2.", ".ffn_down.")
    n = n.replace(".adaLN_modulation.1.", ".adaln.")
    # dit.py ConvBlock: Sequential(Conv1d, Mish, Conv1d)
    n = n.replace(".conv.block.0.", ".conv0.")
    n = n.replace(".conv.block.2.", ".conv2.")
    return n


def map_core_name(k: str) -> str | None:
    n = k
    if n.startswith("backbone_llm."):
        n = n.replace("backbone_llm.embed_tokens.", "llm.tok_emb.")
        n = n.replace("backbone_llm.", "llm.")
        n = _map_qwen3_suffix(n)
        return "frt." + n
    if n.startswith("patch_encoder."):
        n = n.replace("patch_encoder.", "penc.")
        n = n.replace("penc.blocks.", "penc.blk.")
        n = n.replace("penc.out_proj.norm_final.", "penc.out_norm.")
        n = n.replace("penc.out_proj.linear.", "penc.out_proj.")
        n = _map_ditblock_suffix(n)
        return "frt." + n
    if n.startswith("dit."):
        n = n.replace("dit.blocks.", "dit.blk.")
        n = n.replace("dit.final_layer.adaLN_modulation.1.", "dit.final_adaln.")
        n = n.replace("dit.final_layer.linear.", "dit.final_proj.")
        n = n.replace("dit.t_embedder.time_mlp.0.", "dit.time_mlp0.")
        n = n.replace("dit.t_embedder.time_mlp.2.", "dit.time_mlp2.")
        n = _map_ditblock_suffix(n)
        return "frt." + n
    for p in ("spk_proj_llm.", "spk_proj_dit.", "dit_head.", "stop_head."):
        if n.startswith(p):
            return "frt." + n
    return None


def map_redae_name(k: str) -> str | None:
    n = k
    if ".embed_tokens." in n:
        return None  # unused in RedAE (inputs are always inputs_embeds)
    if n.startswith("encoder.downsample."):
        n = n.replace("encoder.downsample.qwen3.layers.", "encds.layers.")
        n = n.replace("encoder.downsample.qwen3.norm.", "encds.norm.")
        n = n.replace("encoder.downsample.cls_tok", "encds.cls_tok")
        n = _map_qwen3_suffix(n)
        return "frt." + n
    if n.startswith("encoder."):
        n = n.replace("encoder.qwen3.layers.", "enc.layers.")
        n = n.replace("encoder.qwen3.norm.", "enc.norm.")
        n = n.replace("encoder.in_proj.0.", "enc.in_proj0.")
        n = n.replace("encoder.in_proj.1.", "enc.in_proj1.")
        n = n.replace("encoder.out_proj.", "enc.out_proj.")
        n = _map_qwen3_suffix(n)
        return "frt." + n
    if n.startswith("decoder."):
        n = n.replace("decoder.qwen3.layers.", "dec.layers.")
        n = n.replace("decoder.qwen3.norm.", "dec.norm.")
        n = n.replace("decoder.in_proj.", "dec.in_proj.")
        n = n.replace("decoder.istft_head.out.", "dec.istft_out.")
        n = n.replace("decoder.istft_head.istft.window", "dec.istft_window")
        n = _map_qwen3_suffix(n)
        return "frt." + n
    return None


# ── Tokenizer ────────────────────────────────────────────────────────

def load_tokenizer(tok_dir: Path):
    """Replicate load_text_tokenizer(): tokenizer.json vocab + added_tokens,
    then append the FireRed specials (in order) that aren't already present.

    Returns (tokens ordered by id, merges, sot_id, eot_id, n_added)."""
    tj = json.loads((tok_dir / "tokenizer.json").read_text(encoding="utf-8"))
    vocab = dict(tj["model"]["vocab"])  # token → id
    for at in tj.get("added_tokens", []):
        vocab.setdefault(at["content"], at["id"])
    next_id = max(vocab.values()) + 1
    n_added = 0
    for tok in SPECIAL_TOKENS:
        if tok not in vocab:
            vocab[tok] = next_id
            next_id += 1
            n_added += 1
    # Dense id space check
    ids = sorted(vocab.values())
    assert ids[0] == 0 and ids[-1] == len(ids) - 1, "non-dense token id space"
    tokens = [None] * len(ids)
    for tok, i in vocab.items():
        tokens[i] = tok
    merges = []
    for m in tj["model"].get("merges", []):
        merges.append(" ".join(m) if isinstance(m, list) else str(m))
    return tokens, merges, vocab["<|sot|>"], vocab["<|eot|>"], n_added


# ── Main ─────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Convert FireRedTTS3 → GGUF")
    p.add_argument("--model-dir", type=Path, required=True,
                   help="FireRedTTS3 snapshot root (fireredtts3_base/, redae/, "
                        "campp/, text_tokenizer/)")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--variant", choices=["base", "instruct"], default="base")
    p.add_argument("--skip-core", action="store_true")
    p.add_argument("--skip-redae", action="store_true")
    p.add_argument("--default-prompt-npz", type=Path, default=None,
                   help="npz with latents [T,64] f32 + spk_emb [512] f32 "
                        "(precomputed by torch on the reference prompt WAV)")
    p.add_argument("--default-prompt-text", type=str, default="")
    p.add_argument("--default-prompt-language", type=str, default="English")
    args = p.parse_args()

    md = args.model_dir
    args.output_dir.mkdir(parents=True, exist_ok=True)

    core_dir = md / f"fireredtts3_{args.variant}"
    core_cfg = json.loads((core_dir / "config.json").read_text())
    redae_cfg = json.loads((md / "redae" / "config.json").read_text())

    # ── Core GGUF ────────────────────────────────────────────────────
    if not args.skip_core:
        out = args.output_dir / f"fireredtts3-{args.variant}-f16.gguf"
        print(f"=== core → {out}")
        w = GGUFWriter(str(out), arch="fireredtts3")
        w.add_string("frt.variant", args.variant)

        # Qwen3-1.7B backbone constants (Qwen3_1_7B_ConfigDict in
        # fireredtts3_base.py — NOT in the checkpoint config.json).
        w.add_int32("frt.llm.n_layers", 28)
        w.add_int32("frt.llm.hidden_size", 2048)
        w.add_int32("frt.llm.intermediate_size", 6144)
        w.add_int32("frt.llm.n_heads", 16)
        w.add_int32("frt.llm.n_kv_heads", 8)
        w.add_int32("frt.llm.head_dim", 128)
        w.add_int32("frt.llm.vocab_size", 151936)
        w.add_float32("frt.llm.rope_theta", 1000000.0)
        w.add_float32("frt.llm.rms_norm_eps", 1e-6)
        # PatchEncoder / DiT / shared (checkpoint config.json)
        w.add_int32("frt.penc.n_layers", core_cfg["patch_encoder_depth"])
        w.add_int32("frt.penc.hidden_size", core_cfg["patch_encoder_hidden_size"])
        w.add_int32("frt.penc.n_heads", core_cfg["patch_encoder_num_heads"])
        w.add_int32("frt.penc.mlp_ratio", core_cfg["patch_encoder_mlp_ratio"])
        w.add_int32("frt.dit.n_layers", core_cfg["dit_depth"])
        w.add_int32("frt.dit.hidden_size", core_cfg["dit_hidden_size"])
        w.add_int32("frt.dit.n_heads", core_cfg["dit_num_heads"])
        w.add_int32("frt.dit.mlp_ratio", core_cfg["dit_mlp_ratio"])
        w.add_int32("frt.redae_dim", core_cfg["redae_dim"])
        w.add_int32("frt.spk_dim", core_cfg["spk_in_dim"])
        w.add_int32("frt.patch_size", core_cfg["patch_size"])
        w.add_int32("frt.n_history_patches", core_cfg["num_history_patches"])

        # Tokenizer
        tokens, merges, sot_id, eot_id, n_added = load_tokenizer(md / "text_tokenizer")
        w.add_string("frt.tokenizer.tokens", "\n".join(tokens))
        w.add_int32("frt.tokenizer.n_tokens", len(tokens))
        w.add_string("frt.tokenizer.merges", "\n".join(merges))
        w.add_int32("frt.tokenizer.n_merges", len(merges))
        w.add_int32("frt.token.sot", sot_id)
        w.add_int32("frt.token.eot", eot_id)
        print(f"  tokenizer: {len(tokens)} tokens ({n_added} FireRed specials "
              f"appended), {len(merges)} merges, sot={sot_id} eot={eot_id}")
        assert len(tokens) <= 151936, "vocab exceeds embedding rows"

        # Baked default voice prompt (optional)
        if args.default_prompt_npz:
            npz = np.load(args.default_prompt_npz)
            lat = npz["latents"].astype(np.float32)
            spk = npz["spk_emb"].astype(np.float32).reshape(-1)
            assert lat.ndim == 2 and lat.shape[1] == core_cfg["redae_dim"], lat.shape
            assert spk.shape[0] == core_cfg["spk_in_dim"], spk.shape
            assert lat.shape[0] % core_cfg["patch_size"] == 0, lat.shape
            w.add_tensor("frt.dprompt.latents", lat)
            w.add_tensor("frt.dprompt.spk_emb", spk)
            w.add_string("frt.dprompt.text", args.default_prompt_text)
            w.add_string("frt.dprompt.language", args.default_prompt_language)
            print(f"  default prompt: latents {lat.shape}, "
                  f"text={args.default_prompt_text[:50]!r}")

        n = 0
        with safe_open(str(core_dir / "model.safetensors"), framework="pt") as f:
            for k in sorted(f.keys()):
                g = map_core_name(k)
                if g is None:
                    print(f"  SKIP {k}")
                    continue
                add_tensor(w, g, f.get_tensor(k))
                n += 1
        print(f"  core tensors: {n}")
        w.write_header_to_file()
        w.write_kv_data_to_file()
        w.write_tensors_to_file()
        w.close()
        print(f"  → {out} ({out.stat().st_size/2**20:.1f} MiB)")

    # ── RedAE + CAM++ GGUF ───────────────────────────────────────────
    if not args.skip_redae:
        out = args.output_dir / "fireredtts3-redae-f16.gguf"
        print(f"=== redae → {out}")
        w = GGUFWriter(str(out), arch="fireredtts3-redae")
        c = redae_cfg
        w.add_int32("frt.audio_patch_size", c["audio_patch_size"])   # 480
        w.add_int32("frt.sample_rate", c["audio_sample_rate"])       # 24000
        w.add_int32("frt.bottleneck_dim", c["bottleneck_dim"])       # 64
        for side, pre in (("enc", "enc_"), ("dec", "dec_")):
            w.add_int32(f"frt.{side}.n_layers", c[f"{pre}num_hidden_layers"])
            w.add_int32(f"frt.{side}.hidden_size", c[f"{pre}hidden_size"])
            w.add_int32(f"frt.{side}.intermediate_size", c[f"{pre}intermediate_size"])
            w.add_int32(f"frt.{side}.n_heads", c[f"{pre}num_attention_heads"])
            w.add_int32(f"frt.{side}.n_kv_heads", c[f"{pre}num_key_value_heads"])
            w.add_int32(f"frt.{side}.sliding_window", c[f"{pre}sliding_window"])
        w.add_int32("frt.enc.extra_downsample_rate", c["enc_extra_downsample_rate"])
        w.add_int32("frt.encds.n_layers", c["enc_downsample_num_hidden_layers"])
        # Qwen3Config defaults used by RedAE stacks (not in config.json):
        w.add_int32("frt.redae.head_dim", 128)
        w.add_float32("frt.redae.rope_theta", 10000.0)
        w.add_float32("frt.redae.rms_norm_eps", 1e-6)

        n = 0
        with safe_open(str(md / "redae" / "model.safetensors"), framework="pt") as f:
            for k in sorted(f.keys()):
                g = map_redae_name(k)
                if g is None:
                    continue
                add_tensor(w, g, f.get_tensor(k))
                n += 1
        print(f"  redae tensors: {n} (3 unused embed_tokens dropped)")

        # CAM++ speaker encoder — raw torch names under campplus.*
        # (confucius4_bind_campplus expects this exact layout).
        sd = torch.load(str(md / "campp" / "campplus_voxceleb.bin"),
                        weights_only=True, map_location="cpu")
        ncp = 0
        for k, t in sorted(sd.items()):
            if "num_batches_tracked" in k:
                continue
            add_tensor(w, "campplus." + k, t)
            ncp += 1
        w.add_int32("frt.campplus.embedding_size", 512)
        print(f"  campplus tensors: {ncp}")

        w.write_header_to_file()
        w.write_kv_data_to_file()
        w.write_tensors_to_file()
        w.close()
        print(f"  → {out} ({out.stat().st_size/2**20:.1f} MiB)")

    print("Done.")


if __name__ == "__main__":
    main()
