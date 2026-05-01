#!/usr/bin/env python3
"""
Convert hexgrad/Kokoro-82M (and yl4579/StyleTTS2-LJSpeech, which shares
the same module architecture) to GGUF F16 for the CrispASR ``kokoro``
TTS backend.

Architecture — confirmed against the public config.json/config.yml
(April 2026):

  Top-level (kokoro-v1_0.pth is a dict of 5 component state_dicts;
  StyleTTS2's epoch_2nd_*.pth wraps them under a 'net' key alongside
  training-only modules — auto-detected and unwrapped):

    bert/         — Custom ALBERT-base text encoder over phonemes.
                    1 albert_layer_group with 1 albert_layer, applied
                    num_hidden_layers=12 times with shared weights.
                    embedding_size=128 (factorised) → hidden=768.
    bert_encoder/ — Single Linear(768 → hidden_dim=512) projecting the
                    BERT pooled output into the prosody predictor space.
    predictor/    — ProsodyPredictor: DurationEncoder (3× {LSTM,AdaLN})
                    → LSTM → duration_proj (Linear 512→max_dur=50)
                    → shared LSTM → F0 (3× AdainResBlk1d) → F0_proj
                    → N  (3× AdainResBlk1d) → N_proj.
    decoder/      — iSTFTNet: encode (AdainResBlk1d) → 4× AdainResBlk1d
                    (decode.0..3) → Generator (m_source, ups, resblocks,
                    noise_convs, noise_res, conv_post → iSTFT).
    text_encoder/ — Phoneme TextEncoder: Embedding(178,512) → 3× (Conv1d
                    k=5, LayerNorm) → bidirectional LSTM (1L, 256/dir).

  All conv/linear weights inside predictor + decoder use PyTorch's
  ``weight_norm`` reparameterisation (weight_g + weight_v). The
  converter pre-fuses these into a single ``weight`` tensor so the C++
  runtime sees normal layers — same approach as the official Kokoro
  ONNX export.

  Voice files (one .pt per voice in ``voices/*.pt``) are *not* part of
  this checkpoint. Use ``convert-kokoro-voice-to-gguf.py`` to convert
  them into the per-voice GGUF the runtime loads via ``--voice``.

  Phoneme input: Kokoro consumes IPA phonemes produced by `misaki`
  (which falls back to espeak-ng for OOD English / non-English).
  StyleTTS2 uses espeak-ng directly. Both feed into the same 178-symbol
  IPA vocab. Kokoro's config.json embeds {phoneme: id} (sparse — gaps
  in the id range); StyleTTS2 ships only a config.yml without the vocab,
  so we fall back to the canonical 178-symbol set from yl4579's
  text_utils.py.

  *NOT* supported by this script:
   - StyleTTS2-LibriTTS (HiFi-GAN decoder with 4-stage upsampling
     [10,5,3,2] — needs a separate code path; this script will exit
     cleanly if it sees ``decoder.type == hifigan``).
   - The diffusion-style-sampler module bundled with StyleTTS2-LJSpeech
     (used at inference to draw a random style vector from text). Either
     pre-bake a voice with the upstream PyTorch model and run it through
     ``convert-kokoro-voice-to-gguf.py``, or extend this converter.

Usage:

    # Kokoro (config.json + 1 .pth)
    python models/convert-kokoro-to-gguf.py \\
        --input hexgrad/Kokoro-82M \\
        --output kokoro-82m.gguf

    # StyleTTS2-LJSpeech (config.yml + 1 .pth, ``net.*`` wrapped)
    python models/convert-kokoro-to-gguf.py \\
        --input /path/to/StyleTTS2-LJSpeech/Models/LJSpeech \\
        --output styletts2-ljspeech.gguf
"""

from __future__ import annotations

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
    import torch
except ImportError:
    sys.exit("pip install torch")

try:
    from huggingface_hub import snapshot_download
except ImportError:
    sys.exit("pip install huggingface_hub")

# StyleTTS2's standard 178-symbol IPA vocab (from yl4579/StyleTTS2 text_utils.py).
# Identical to the {"vocab": ...} block in Kokoro's config.json — used as a
# fallback when converting raw StyleTTS2 checkpoints that don't ship a Kokoro-
# style config.json. The id assignments match Kokoro byte-for-byte.
_STYLETTS2_PAD = "$"
_STYLETTS2_PUNCT = ';:,.!?¡¿—…"«»“” '
_STYLETTS2_LETTERS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
_STYLETTS2_IPA = (
    "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽ"
    "ʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘’̃"
)
STYLETTS2_DEFAULT_VOCAB = (
    [_STYLETTS2_PAD] + list(_STYLETTS2_PUNCT)
    + list(_STYLETTS2_LETTERS) + list(_STYLETTS2_IPA)
)


def load_yaml(path: Path) -> dict:
    try:
        import yaml  # type: ignore
    except ImportError:
        sys.exit("pip install pyyaml (needed to read StyleTTS2 config.yml)")
    with open(path) as f:
        return yaml.safe_load(f)


def styletts2_to_kokoro_cfg(yaml_cfg: dict) -> dict:
    """Translate a StyleTTS2 config.yml (model_params block) into the flat
    structure Kokoro's config.json uses, so the rest of this converter
    needs only one path. Vocab falls back to the canonical 178-symbol set."""
    mp = yaml_cfg.get("model_params", {}) or {}
    dec = mp.get("decoder", {}) or {}
    if dec.get("type") != "istftnet":
        sys.exit(
            f"StyleTTS2 decoder type {dec.get('type')!r} is not iSTFTNet — "
            "this converter currently only supports the iSTFTNet decoder "
            "(used by Kokoro and StyleTTS2-LJSpeech). HiFi-GAN (StyleTTS2-"
            "LibriTTS) needs a separate code path."
        )
    return {
        "dim_in": mp.get("dim_in", 64),
        "hidden_dim": mp.get("hidden_dim", 512),
        "style_dim": mp.get("style_dim", 128),
        "max_conv_dim": mp.get("max_conv_dim", 512),
        "max_dur": mp.get("max_dur", 50),
        "n_layer": mp.get("n_layer", 3),
        "n_mels": mp.get("n_mels", 80),
        "n_token": mp.get("n_token", 178),
        "text_encoder_kernel_size": mp.get("text_encoder_kernel_size", 5),
        "multispeaker": mp.get("multispeaker", False),
        "istftnet": {
            "upsample_rates": dec.get("upsample_rates", [10, 6]),
            "upsample_kernel_sizes": dec.get("upsample_kernel_sizes", [20, 12]),
            "upsample_initial_channel": dec.get("upsample_initial_channel", 512),
            "resblock_kernel_sizes": dec.get("resblock_kernel_sizes", [3, 7, 11]),
            "resblock_dilation_sizes": dec.get(
                "resblock_dilation_sizes", [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
            ),
            "gen_istft_n_fft": dec.get("gen_istft_n_fft", 20),
            "gen_istft_hop_size": dec.get("gen_istft_hop_size", 5),
        },
        "plbert": {  # PL-BERT defaults — matches papercup-ai config
            "hidden_size": 768,
            "num_attention_heads": 12,
            "intermediate_size": 2048,
            "max_position_embeddings": 512,
            "num_hidden_layers": 12,
        },
        "vocab": {tok: i for i, tok in enumerate(STYLETTS2_DEFAULT_VOCAB)},
    }


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_model_dir(model_id: str) -> Path:
    p = Path(model_id)
    if p.is_dir():
        return p
    print(f"Downloading {model_id}…", file=sys.stderr)
    return Path(snapshot_download(model_id, allow_patterns=[
        "*.pth", "*.pt", "*.json", "*.txt", "*.yml", "voices/*",
    ]))


# ---------------------------------------------------------------------------
# weight_norm fusion
# ---------------------------------------------------------------------------

def fuse_weight_norm(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Replace weight_norm reparameterisations with a single fused X.weight,
    matching ``torch.nn.utils.weight_norm``'s definition:

        weight = v * (g / ||v||)   (||·|| computed over all dims except 0)

    Handles BOTH naming conventions:
    - Legacy ``torch.nn.utils.weight_norm`` (PyTorch <2.1 default):
      ``X.weight_g`` / ``X.weight_v``  (Kokoro-82M, StyleTTS2-LJSpeech).
    - Modern ``torch.nn.utils.parametrize.register_parametrization`` with
      ``WeightNorm`` (PyTorch ≥2.1, used by recent re-trains like
      dida-80b/kokoro-german-hui-multispeaker-base):
      ``X.parametrizations.weight.original0`` (g) / ``original1`` (v).

    Operates in-place semantics — returns a new dict with fused entries.
    """
    out: dict[str, torch.Tensor] = {}
    pairs: dict[str, dict[str, torch.Tensor]] = {}
    for k, v in state.items():
        if k.endswith(".weight_g"):
            pairs.setdefault(k[: -len(".weight_g")], {})["g"] = v
        elif k.endswith(".weight_v"):
            pairs.setdefault(k[: -len(".weight_v")], {})["v"] = v
        elif k.endswith(".parametrizations.weight.original0"):
            stem = k[: -len(".parametrizations.weight.original0")]
            pairs.setdefault(stem, {})["g"] = v
        elif k.endswith(".parametrizations.weight.original1"):
            stem = k[: -len(".parametrizations.weight.original1")]
            pairs.setdefault(stem, {})["v"] = v
        else:
            out[k] = v
    fused = 0
    for stem, parts in pairs.items():
        if "g" not in parts or "v" not in parts:
            for sub, t in parts.items():
                out[f"{stem}.weight_{sub}"] = t  # leave orphan as-is
            continue
        g = parts["g"].to(torch.float32)
        v = parts["v"].to(torch.float32)
        # norm over all dims except dim 0 (PyTorch's norm_except_dim)
        flat = v.reshape(v.shape[0], -1)
        norm = flat.norm(p=2, dim=1).reshape(g.shape)  # match g's shape
        out[f"{stem}.weight"] = v * (g / norm).clamp_min(1e-12)
        fused += 1
    print(f"  weight_norm fused: {fused} layers", file=sys.stderr)
    return out


# ---------------------------------------------------------------------------
# Tensor name remapping
# ---------------------------------------------------------------------------

import re


def map_bert(n: str) -> str:
    # ALBERT-style: embeddings.* / encoder.* / pooler.*
    n = n.replace("embeddings.word_embeddings.", "bert.embd.tok.")
    n = n.replace("embeddings.position_embeddings.", "bert.embd.pos.")
    n = n.replace("embeddings.token_type_embeddings.", "bert.embd.tt.")
    n = n.replace("embeddings.LayerNorm.", "bert.embd.ln.")
    n = n.replace("encoder.embedding_hidden_mapping_in.", "bert.embd_proj.")
    n = n.replace(
        "encoder.albert_layer_groups.0.albert_layers.0.attention.query.",
        "bert.attn_q.",
    )
    n = n.replace(
        "encoder.albert_layer_groups.0.albert_layers.0.attention.key.",
        "bert.attn_k.",
    )
    n = n.replace(
        "encoder.albert_layer_groups.0.albert_layers.0.attention.value.",
        "bert.attn_v.",
    )
    n = n.replace(
        "encoder.albert_layer_groups.0.albert_layers.0.attention.dense.",
        "bert.attn_o.",
    )
    n = n.replace(
        "encoder.albert_layer_groups.0.albert_layers.0.attention.LayerNorm.",
        "bert.attn_ln.",
    )
    n = n.replace(
        "encoder.albert_layer_groups.0.albert_layers.0.full_layer_layer_norm.",
        "bert.ffn_ln.",
    )
    n = n.replace(
        "encoder.albert_layer_groups.0.albert_layers.0.ffn_output.",
        "bert.ffn_down.",
    )
    n = n.replace(
        "encoder.albert_layer_groups.0.albert_layers.0.ffn.",
        "bert.ffn_up.",
    )
    n = n.replace("pooler.", "bert.pooler.")
    return n


def map_predictor(n: str) -> str:
    # text_encoder = DurationEncoder (alternating LSTM, AdaLayerNorm).
    # Even indices (0, 2, 4) are LSTMs; odd indices (1, 3, 5) are AdaLN.
    def _de(match):
        idx = int(match.group(1))
        rest = match.group(2)
        if idx % 2 == 0:
            return f"pred.dur_enc.{idx // 2}.lstm.{rest}"
        # AdaLayerNorm: only ".fc.weight" and ".fc.bias" exist on disk
        return f"pred.dur_enc.{idx // 2}.adaln.{rest.replace('fc.', '')}"
    n = re.sub(r"^text_encoder\.lstms\.(\d+)\.(.+)$", _de, n)

    n = n.replace("lstm.", "pred.lstm.", 1) if n.startswith("lstm.") else n
    n = n.replace("shared.", "pred.shared.", 1) if n.startswith("shared.") else n
    n = n.replace("duration_proj.linear_layer.", "pred.dur_proj.")
    n = re.sub(r"^F0\.(\d+)\.", r"pred.F0.\1.", n)
    n = re.sub(r"^N\.(\d+)\.", r"pred.N.\1.", n)
    n = n.replace("F0_proj.", "pred.F0_proj.")
    n = n.replace("N_proj.", "pred.N_proj.")
    # AdainResBlk1d submodules: .conv1, .conv2, .conv1x1, .pool, .norm{1,2}.fc
    n = n.replace(".norm1.fc.", ".adain1.")
    n = n.replace(".norm2.fc.", ".adain2.")
    return n


def map_decoder(n: str) -> str:
    # encode.* — pre-generator AdainResBlk1d
    if n.startswith("encode."):
        n = "dec." + n
    # decode.{i}.* — 4× AdainResBlk1d
    n = re.sub(r"^decode\.(\d+)\.", r"dec.decode.\1.", n)
    # asr_res.0.* — Conv1d on ASR features (passes through)
    n = n.replace("asr_res.0.", "dec.asr_res.")
    # F0_conv / N_conv — input projections for F0 and energy
    n = n.replace("F0_conv.", "dec.F0_conv.")
    n = n.replace("N_conv.", "dec.N_conv.")
    # generator.*
    if n.startswith("generator."):
        sub = n[len("generator."):]
        sub = sub.replace("m_source.l_linear.", "m_source.")
        sub = re.sub(r"^ups\.(\d+)\.", r"ups.\1.", sub)
        sub = re.sub(r"^noise_convs\.(\d+)\.", r"noise_convs.\1.", sub)
        # noise_res.{i}.adain1.{j}.fc.{w} → noise_res.{i}.adain1.{j}.{w}
        sub = re.sub(
            r"^(noise_res|resblocks)\.(\d+)\.(adain[12])\.(\d+)\.fc\.",
            r"\1.\2.\3.\4.",
            sub,
        )
        sub = re.sub(
            r"^(noise_res|resblocks)\.(\d+)\.(alpha[12])\.(\d+)$",
            r"\1.\2.\3.\4",
            sub,
        )
        sub = re.sub(
            r"^(noise_res|resblocks)\.(\d+)\.(convs[12])\.(\d+)\.",
            r"\1.\2.\3.\4.",
            sub,
        )
        n = "dec.gen." + sub
    # AdainResBlk1d AdaIN: .norm1.fc → .adain1, .norm2.fc → .adain2
    # (applied AFTER the dec. prefix since AdainResBlk1d also lives inside dec.encode/decode)
    n = n.replace(".norm1.fc.", ".adain1.")
    n = n.replace(".norm2.fc.", ".adain2.")
    return n


def map_text_encoder(n: str) -> str:
    n = n.replace("embedding.", "text_enc.embd.")
    # cnn.{i}.0 = Conv1d, cnn.{i}.1 = LayerNorm (gamma/beta)
    n = re.sub(r"^cnn\.(\d+)\.0\.", r"text_enc.cnn.\1.conv.", n)
    n = re.sub(r"^cnn\.(\d+)\.1\.", r"text_enc.cnn.\1.ln.", n)
    n = re.sub(r"^lstm\.", "text_enc.lstm.", n)
    return n


def map_tensor_name(component: str, hf_name: str) -> str | None:
    """Map a (component, key) pair from kokoro-v1_0.pth → GGUF tensor name."""
    n = hf_name
    if n.startswith("module."):
        n = n[len("module."):]

    if component == "bert":
        return map_bert(n)
    if component == "bert_encoder":
        # Linear 768→512 — only .weight and .bias
        return f"bert_proj.{n}"
    if component == "predictor":
        return map_predictor(n)
    if component == "decoder":
        return map_decoder(n)
    if component == "text_encoder":
        return map_text_encoder(n)

    return None  # unknown component → skip


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Convert Kokoro-82M to GGUF")
    ap.add_argument("--input", required=True,
                    help="HF model ID (e.g. hexgrad/Kokoro-82M) or local dir")
    ap.add_argument("--output", required=True, help="Output GGUF path")
    ap.add_argument("--outtype", default="f16", choices=["f32", "f16"])
    ap.add_argument("--checkpoint", default=None,
                    help="Override the .pth filename (default: auto-detect)")
    ap.add_argument("--config", default=None,
                    help="Override config.json (useful when the model dir ships only "
                         "a HF-hub stub config without vocab/architecture, e.g. "
                         "dida-80b/kokoro-german-hui-multispeaker-base — point this at "
                         "hexgrad/Kokoro-82M's config.json since the vocab IDs match)")
    args = ap.parse_args()

    model_dir = load_model_dir(args.input)

    json_cfg = Path(args.config) if args.config else (model_dir / "config.json")
    yml_cfg = next(
        (p for p in (model_dir / "config.yml", model_dir / "config.yaml")
         if p.is_file()),
        None,
    )
    if json_cfg.is_file():
        with open(json_cfg) as f:
            cfg = json.load(f)
        print(f"  config:  {json_cfg.name} (Kokoro layout)", file=sys.stderr)
    elif yml_cfg is not None:
        cfg = styletts2_to_kokoro_cfg(load_yaml(yml_cfg))
        print(
            f"  config:  {yml_cfg.name} (StyleTTS2 layout — translated;"
            f" vocab falls back to canonical 178-symbol IPA set)",
            file=sys.stderr,
        )
    else:
        sys.exit(f"no config.json or config.yml in {model_dir}")

    # Locate the checkpoint.
    ckpt_path = (
        Path(args.checkpoint)
        if args.checkpoint
        else next((p for p in model_dir.glob("*.pth")), None)
    )
    if ckpt_path is None or not ckpt_path.is_file():
        sys.exit(f"no .pth checkpoint in {model_dir}")
    print(f"Loading {ckpt_path}…", file=sys.stderr)
    state = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    if not isinstance(state, dict):
        sys.exit(f"unexpected checkpoint format: {type(state).__name__}")

    # StyleTTS2-LJSpeech wraps the 5 inference components under a top-level
    # 'net' key alongside training-only modules (text_aligner, pitch_extractor,
    # discriminators, diffusion, style_encoder, …). Unwrap and ignore extras —
    # only bert/bert_encoder/predictor/decoder/text_encoder are needed for the
    # core synthesis path (Kokoro is exactly that subset, pre-extracted).
    if "net" in state and isinstance(state["net"], dict) and "bert" in state["net"]:
        print("  detected StyleTTS2-style wrapper — unwrapping 'net.*'",
              file=sys.stderr)
        net = state["net"]
        extras = sorted(set(net.keys()) - {
            "bert", "bert_encoder", "predictor", "decoder", "text_encoder",
        })
        if extras:
            print(f"  ignoring training/aux components: {extras}",
                  file=sys.stderr)
        state = net

    expected = {"bert", "bert_encoder", "predictor", "decoder", "text_encoder"}
    missing = expected - set(state.keys())
    if missing:
        sys.exit(f"missing components in checkpoint: {missing}")

    istft = cfg.get("istftnet", {})
    plbert = cfg.get("plbert", {})
    print(
        f"\nKokoro: hidden={cfg.get('hidden_dim')}  style={cfg.get('style_dim')}  "
        f"n_token={cfg.get('n_token')}  n_mels={cfg.get('n_mels')}  "
        f"max_dur={cfg.get('max_dur')}  n_layer={cfg.get('n_layer')}"
    )
    print(
        f"  iSTFTNet: ups={istft.get('upsample_rates')}  k={istft.get('upsample_kernel_sizes')}  "
        f"init_ch={istft.get('upsample_initial_channel')}  fft={istft.get('gen_istft_n_fft')}  "
        f"hop={istft.get('gen_istft_hop_size')}"
    )
    print(
        f"  PL-BERT:  {plbert.get('num_hidden_layers')}L  hidden={plbert.get('hidden_size')}  "
        f"heads={plbert.get('num_attention_heads')}  ff={plbert.get('intermediate_size')}"
    )

    # ------------------------------------------------------------------
    # Tensor pipeline
    # ------------------------------------------------------------------
    out_dtype = np.float16 if args.outtype == "f16" else np.float32
    out_qt = (
        GGMLQuantizationType.F16 if args.outtype == "f16" else GGMLQuantizationType.F32
    )
    w = GGUFWriter(args.output, arch="kokoro", use_temp_file=True)
    w.add_name(Path(args.input).name or "kokoro")

    # Hyperparameters --------------------------------------------------
    def u32(k, v): w.add_uint32(k, int(v))
    def f32(k, v): w.add_float32(k, float(v))

    u32("kokoro.dim_in",                cfg.get("dim_in", 64))
    u32("kokoro.hidden_dim",            cfg.get("hidden_dim", 512))
    u32("kokoro.style_dim",             cfg.get("style_dim", 128))
    u32("kokoro.max_conv_dim",          cfg.get("max_conv_dim", 512))
    u32("kokoro.max_dur",               cfg.get("max_dur", 50))
    u32("kokoro.n_token",               cfg.get("n_token", 178))
    u32("kokoro.n_mels",                cfg.get("n_mels", 80))
    u32("kokoro.n_layer",               cfg.get("n_layer", 3))
    u32("kokoro.text_encoder_kernel_size",
        cfg.get("text_encoder_kernel_size", 5))
    u32("kokoro.sample_rate",           24000)

    # iSTFTNet decoder
    u32("kokoro.istft.init_channel",    istft.get("upsample_initial_channel", 512))
    u32("kokoro.istft.n_fft",           istft.get("gen_istft_n_fft", 20))
    u32("kokoro.istft.hop_size",        istft.get("gen_istft_hop_size", 5))
    w.add_array("kokoro.istft.upsample_rates",
                list(istft.get("upsample_rates", [10, 6])))
    w.add_array("kokoro.istft.upsample_kernel_sizes",
                list(istft.get("upsample_kernel_sizes", [20, 12])))
    w.add_array("kokoro.istft.resblock_kernel_sizes",
                list(istft.get("resblock_kernel_sizes", [3, 7, 11])))
    # resblock_dilation_sizes is a 2-D list; flatten with row count for the runtime.
    rd = istft.get("resblock_dilation_sizes", [[1, 3, 5]] * 3)
    u32("kokoro.istft.resblock_n_dilations", len(rd[0]) if rd else 0)
    w.add_array("kokoro.istft.resblock_dilation_sizes",
                [d for row in rd for d in row])

    # PL-BERT (custom ALBERT) — parameter-shared across layers.
    # Word-embedding tensor lives at "module.embeddings.word_embeddings.weight"
    # in DataParallel-wrapped checkpoints (Kokoro-82M, StyleTTS2-LJSpeech) and
    # at the unprefixed key for non-DataParallel re-trains (dida-80b).
    bert_emb = (
        state["bert"].get("module.embeddings.word_embeddings.weight")
        or state["bert"].get("embeddings.word_embeddings.weight")
    )
    if bert_emb is None:
        sys.exit("bert state has no embeddings.word_embeddings.weight tensor")
    u32("kokoro.plbert.embedding_size",     bert_emb.shape[1])
    u32("kokoro.plbert.hidden_size",        plbert.get("hidden_size", 768))
    u32("kokoro.plbert.num_hidden_layers",  plbert.get("num_hidden_layers", 12))
    u32("kokoro.plbert.num_attention_heads",
        plbert.get("num_attention_heads", 12))
    u32("kokoro.plbert.intermediate_size",  plbert.get("intermediate_size", 2048))
    u32("kokoro.plbert.max_position_embeddings",
        plbert.get("max_position_embeddings", 512))
    u32("kokoro.plbert.vocab_size",         bert_emb.shape[0])

    # Vocab — Kokoro embeds {phoneme: id} in config.json. Build a flat
    # token list of size (max_id + 1); empty slots stay "".
    vocab = cfg.get("vocab", {})
    if not vocab:
        sys.exit("config.json has no vocab — refusing to write a tokenless GGUF")
    max_id = max(vocab.values())
    tokens = [""] * (max_id + 1)
    for tok, idx in vocab.items():
        tokens[idx] = tok
    w.add_token_list(tokens)
    u32("kokoro.vocab_size", len(tokens))
    print(f"  vocab:   {len(tokens)} entries from config.json (max id={max_id})")

    # ------------------------------------------------------------------
    # Tensors
    # ------------------------------------------------------------------
    n_mapped = 0
    n_skipped = 0
    unmapped: list[str] = []

    for component in ("bert", "bert_encoder", "predictor", "decoder", "text_encoder"):
        comp_state = {
            k.replace("module.", "", 1): v
            for k, v in state[component].items()
        }
        comp_state = fuse_weight_norm(comp_state)
        for k in sorted(comp_state.keys()):
            gn = map_tensor_name(component, k)
            if gn is None:
                n_skipped += 1
                continue
            if gn.endswith(".weight_g") or gn.endswith(".weight_v"):
                # An orphan weight_g/weight_v we couldn't fuse — abort
                # rather than silently writing it under a half-broken name.
                sys.exit(f"orphan weight_norm tensor: {component}/{k}")
            t = comp_state[k]
            arr = t.to(torch.float32).numpy()
            if arr.ndim <= 1:
                arr = np.ascontiguousarray(arr.astype(np.float32))
                w.add_tensor(gn, arr, raw_dtype=GGMLQuantizationType.F32)
            else:
                arr = np.ascontiguousarray(arr.astype(out_dtype))
                w.add_tensor(gn, arr, raw_dtype=out_qt)
            n_mapped += 1
            if n_mapped <= 30 or n_mapped % 100 == 0:
                print(f"  [{n_mapped}] {gn:60s} {tuple(arr.shape)}  {arr.dtype}")
            if not gn.startswith((
                "bert.", "bert_proj.", "pred.", "dec.", "text_enc.",
            )):
                unmapped.append(f"{component}/{k} → {gn}")

    if unmapped:
        print(
            f"\nWARNING: {len(unmapped)} tensors landed under an unexpected prefix:",
            file=sys.stderr,
        )
        for u in unmapped[:20]:
            print(f"  {u}", file=sys.stderr)

    print(f"\nMapped: {n_mapped}, skipped: {n_skipped}")
    print(f"Writing {args.output}…")
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()

    sz = Path(args.output).stat().st_size / 1e6
    print(f"Done: {args.output}  ({sz:.1f} MB, {n_mapped} tensors)")


if __name__ == "__main__":
    main()
