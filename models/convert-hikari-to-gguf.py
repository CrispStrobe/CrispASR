#!/usr/bin/env python3
"""
Convert sbintuitions/hikari-medium (MIT) to GGUF.

Hikari is a Whisper-medium encoder-decoder retrained for SIMULTANEOUS speech
translation (EN->DE/JA/RU) and streaming English ASR:
  - encoder: Whisper conv stem + 24 pre-LN layers, self-attention CAUSAL
    (config encoder_is_causal); learned positions [1500, 1024]
  - decoder: 24 pre-LN layers, learned positions [375, 1024], causal self-attn,
    cross-attention masked so decoder position i sees encoder frames j < 4*i
    (config cross_attention_is_causal + decoder_time_dilation = 4)
  - lm head tied to the token embedding
  - the decoder emits ONE token per 80 ms of audio; token 93 ("~") means WAIT

Reference: https://github.com/sbintuitions/hikari (src/hikari/server/model_wrapper.py,
models/modeling_hikari.py). Only model.safetensors + the json configs are read;
optimizer.pt / rng_state_*.pth / scheduler.pt / training_args.bin are training
state and are ignored.

The bf16 safetensors are read with numpy directly (no torch needed). Matrices
are written F16 (bf16 -> f16 is exact for every weight in range), norms,
biases and position tables F32.

Usage:
  python models/convert-hikari-to-gguf.py --input /path/to/hikari-medium \\
      --output hikari-medium-f16.gguf

GGUF tensor naming:
  audio.mel_filters                 F32 [80, 201]   (whisper mel_80, slaney)
  audio.mel_window                  F32 [400]       (periodic hann)
  enc.conv{1,2}.weight              F16 [OC, IC, 3]
  enc.conv{1,2}.bias                F32
  enc.pos_emb                       F32 [1500, 1024]
  enc.blk.N.attn_ln.{weight,bias}   F32
  enc.blk.N.attn_{q,v,o}.{weight,bias} / attn_k.weight
  enc.blk.N.ffn_ln / ffn_up / ffn_down
  enc.out_ln.{weight,bias}
  dec.tok_emb                       F16 [51865, 1024]  (also the lm head)
  dec.pos_emb                       F32 [375, 1024]
  dec.blk.N.attn_* / cross_* / ffn_*  (as the encoder; cross_k has no bias)
  dec.out_ln.{weight,bias}
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

import numpy as np

try:
    import gguf
except ImportError:
    sys.exit("pip install gguf")


# ---------------------------------------------------------------------------
# safetensors (bf16 / f16 / f32) without torch
# ---------------------------------------------------------------------------

def read_safetensors(path: Path) -> dict[str, np.ndarray]:
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
        base = 8 + n
    data = np.memmap(path, dtype=np.uint8, mode="r")
    out: dict[str, np.ndarray] = {}
    for name, info in header.items():
        if name == "__metadata__":
            continue
        a, b = info["data_offsets"]
        raw = data[base + a: base + b]
        shape = tuple(info["shape"])
        dt = info["dtype"]
        if dt == "BF16":
            u16 = np.frombuffer(raw.tobytes(), dtype=np.uint16)
            arr = (u16.astype(np.uint32) << 16).view(np.float32)
        elif dt == "F16":
            arr = np.frombuffer(raw.tobytes(), dtype=np.float16).astype(np.float32)
        elif dt == "F32":
            arr = np.frombuffer(raw.tobytes(), dtype=np.float32).copy()
        else:
            sys.exit(f"unsupported dtype {dt} for {name}")
        out[name] = arr.reshape(shape)
    return out


# ---------------------------------------------------------------------------
# Whisper mel filterbank (librosa slaney, n_fft=400, sr=16k)
# ---------------------------------------------------------------------------

def slaney_mel_filters(sr: int = 16000, n_fft: int = 400, n_mels: int = 80) -> np.ndarray:
    """librosa.filters.mel(sr, n_fft, n_mels) with htk=False, norm='slaney'."""
    def hz_to_mel(f):
        f = np.asanyarray(f, dtype=np.float64)
        f_sp = 200.0 / 3
        mels = f / f_sp
        min_log_hz = 1000.0
        min_log_mel = min_log_hz / f_sp
        logstep = np.log(6.4) / 27.0
        return np.where(f >= min_log_hz, min_log_mel + np.log(np.maximum(f, 1e-10) / min_log_hz) / logstep, mels)

    def mel_to_hz(m):
        m = np.asanyarray(m, dtype=np.float64)
        f_sp = 200.0 / 3
        freqs = f_sp * m
        min_log_hz = 1000.0
        min_log_mel = min_log_hz / f_sp
        logstep = np.log(6.4) / 27.0
        return np.where(m >= min_log_mel, min_log_hz * np.exp(logstep * (m - min_log_mel)), freqs)

    n_freqs = n_fft // 2 + 1
    fftfreqs = np.linspace(0, sr / 2, n_freqs)
    mel_pts = mel_to_hz(np.linspace(hz_to_mel(0.0), hz_to_mel(sr / 2.0), n_mels + 2))
    fdiff = np.diff(mel_pts)
    ramps = mel_pts[:, None] - fftfreqs[None, :]
    weights = np.zeros((n_mels, n_freqs))
    for i in range(n_mels):
        lower = -ramps[i] / fdiff[i]
        upper = ramps[i + 2] / fdiff[i + 1]
        weights[i] = np.maximum(0, np.minimum(lower, upper))
    enorm = 2.0 / (mel_pts[2: n_mels + 2] - mel_pts[:n_mels])
    weights *= enorm[:, None]
    return weights.astype(np.float32)


def whisper_mel_filters(n_mels: int) -> np.ndarray:
    # Prefer the exact array openai-whisper ships (what the reference uses);
    # fall back to recomputing it.
    # Located without importing the package (its __init__ pulls in torch/tiktoken).
    try:
        import importlib.util
        import os
        spec = importlib.util.find_spec("whisper")
        if spec is None or not spec.submodule_search_locations:
            raise ImportError
        p = os.path.join(list(spec.submodule_search_locations)[0], "assets", "mel_filters.npz")
        with np.load(p, allow_pickle=False) as f:
            fb = f[f"mel_{n_mels}"].astype(np.float32)
        calc = slaney_mel_filters(n_mels=n_mels)
        print(f"  mel filters: openai-whisper mel_{n_mels} (max |diff| vs recomputed {np.abs(fb - calc).max():.2e})")
        return fb
    except Exception:
        print("  mel filters: recomputed (openai-whisper not importable)")
        return slaney_mel_filters(n_mels=n_mels)


# ---------------------------------------------------------------------------
# Names
# ---------------------------------------------------------------------------

_SUB = {
    "self_attn_layer_norm": "attn_ln",
    "self_attn.q_proj": "attn_q",
    "self_attn.k_proj": "attn_k",
    "self_attn.v_proj": "attn_v",
    "self_attn.out_proj": "attn_o",
    "encoder_attn_layer_norm": "cross_ln",
    "encoder_attn.q_proj": "cross_q",
    "encoder_attn.k_proj": "cross_k",
    "encoder_attn.v_proj": "cross_v",
    "encoder_attn.out_proj": "cross_o",
    "final_layer_norm": "ffn_ln",
    "fc1": "ffn_up",
    "fc2": "ffn_down",
}


def remap(name: str) -> str | None:
    if not name.startswith("model."):
        return None
    n = name[len("model."):]
    side, rest = n.split(".", 1)
    pre = {"encoder": "enc", "decoder": "dec"}[side]
    if rest.startswith("layers."):
        _, idx, sub = rest.split(".", 2)
        mod, wb = sub.rsplit(".", 1)
        if mod not in _SUB:
            return None
        return f"{pre}.blk.{idx}.{_SUB[mod]}.{wb}"
    fixed = {
        "conv1.weight": "conv1.weight", "conv1.bias": "conv1.bias",
        "conv2.weight": "conv2.weight", "conv2.bias": "conv2.bias",
        "embed_positions.weight": "pos_emb",
        "embed_tokens.weight": "tok_emb",
        "layer_norm.weight": "out_ln.weight", "layer_norm.bias": "out_ln.bias",
    }
    if rest in fixed:
        return f"{pre}.{fixed[rest]}"
    return None


def keep_f32(gname: str, arr: np.ndarray) -> bool:
    return arr.ndim == 1 or gname.endswith("pos_emb") or gname.startswith("audio.")


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

def build_vocab(model_dir: Path, vocab_size: int) -> list[str]:
    tj = json.loads((model_dir / "tokenizer.json").read_text(encoding="utf-8"))
    tokens = [""] * vocab_size
    for tok, idx in tj["model"]["vocab"].items():
        if idx < vocab_size:
            tokens[idx] = tok
    for at in tj.get("added_tokens", []):
        if at["id"] < vocab_size:
            tokens[at["id"]] = at["content"]
    missing = [i for i, t in enumerate(tokens) if t == ""]
    for i in missing:
        tokens[i] = f"<|unused{i}|>"
    print(f"  vocab: {vocab_size} ids ({len(missing)} unnamed)")
    return tokens


# ---------------------------------------------------------------------------

def convert(input_dir: Path, out_path: Path) -> None:
    cfg = json.loads((input_dir / "config.json").read_text())
    gen = json.loads((input_dir / "generation_config.json").read_text())

    if not cfg.get("encoder_is_causal") or not cfg.get("cross_attention_is_causal"):
        sys.exit("not a causal Hikari checkpoint (encoder_is_causal / cross_attention_is_causal unset)")
    if cfg.get("conformer") or cfg.get("mimi") or cfg.get("use_rope"):
        sys.exit("conformer / mimi / rope Hikari variants are not supported")

    vocab_size = int(cfg["vocab_size"])
    d_model = int(cfg["d_model"])
    n_mels = int(cfg["num_mel_bins"])
    print(f"  hikari: d={d_model} enc={cfg['encoder_layers']}L dec={cfg['decoder_layers']}L "
          f"heads={cfg['encoder_attention_heads']} audio_ctx={cfg['max_source_positions']} "
          f"text_ctx={cfg['max_target_positions']} dilation={cfg['decoder_time_dilation']}")

    sd = read_safetensors(input_dir / "model.safetensors")
    print(f"  tensors: {len(sd)}")
    tokens = build_vocab(input_dir, vocab_size)

    w = gguf.GGUFWriter(str(out_path), arch="hikari")
    w.add_name("hikari-medium")
    A = "hikari"
    w.add_uint32(f"{A}.vocab_size", vocab_size)
    w.add_uint32(f"{A}.d_model", d_model)
    w.add_uint32(f"{A}.n_mels", n_mels)
    w.add_uint32(f"{A}.encoder.n_layers", int(cfg["encoder_layers"]))
    w.add_uint32(f"{A}.encoder.n_heads", int(cfg["encoder_attention_heads"]))
    w.add_uint32(f"{A}.encoder.ffn_dim", int(cfg["encoder_ffn_dim"]))
    w.add_uint32(f"{A}.decoder.n_layers", int(cfg["decoder_layers"]))
    w.add_uint32(f"{A}.decoder.n_heads", int(cfg["decoder_attention_heads"]))
    w.add_uint32(f"{A}.decoder.ffn_dim", int(cfg["decoder_ffn_dim"]))
    w.add_uint32(f"{A}.n_audio_ctx", int(cfg["max_source_positions"]))
    w.add_uint32(f"{A}.n_text_ctx", int(cfg["max_target_positions"]))
    w.add_uint32(f"{A}.decoder_time_dilation", int(cfg["decoder_time_dilation"]))
    # Whisper special-token ids (generation_config.json / model_wrapper.py)
    w.add_uint32(f"{A}.token.eot", int(cfg["eos_token_id"]))
    w.add_uint32(f"{A}.token.sot", int(cfg["decoder_start_token_id"]))
    w.add_uint32(f"{A}.token.translate", int(gen["task_to_id"]["translate"]))
    w.add_uint32(f"{A}.token.transcribe", int(gen["task_to_id"]["transcribe"]))
    w.add_uint32(f"{A}.token.notimestamps", int(gen["no_timestamps_token_id"]))
    # The WAIT token is a property of the training recipe, not of the configs:
    # model_wrapper.py hardcodes 93 ("~"). Recorded here so the runtime never
    # has to guess; it is checked against the vocab below.
    wait_id = 93
    if tokens[wait_id] != "~":
        sys.exit(f"token {wait_id} is {tokens[wait_id]!r}, expected '~' (the Hikari WAIT token)")
    w.add_uint32(f"{A}.token.wait", wait_id)
    langs = sorted(gen["lang_to_id"].items(), key=lambda kv: kv[1])
    w.add_array(f"{A}.lang_codes", [k.strip("<|>") for k, _ in langs])
    w.add_array(f"{A}.lang_token_ids", [int(v) for _, v in langs])
    w.add_string("tokenizer.ggml.model", "gpt2")
    w.add_array("tokenizer.ggml.tokens", tokens)

    n16 = n32 = 0

    def put(name: str, arr: np.ndarray) -> None:
        nonlocal n16, n32
        if keep_f32(name, arr):
            arr = np.ascontiguousarray(arr, dtype=np.float32)
            n32 += 1
        else:
            arr = np.ascontiguousarray(arr, dtype=np.float16)
            n16 += 1
        w.add_tensor(name, arr)

    fb = whisper_mel_filters(n_mels)  # [n_mels, 201]
    put("audio.mel_filters", fb)
    n = np.arange(400, dtype=np.float64)
    put("audio.mel_window", (0.5 - 0.5 * np.cos(2.0 * np.pi * n / 400)).astype(np.float32))  # torch.hann_window(400)

    skipped = []
    for k in sorted(sd):
        g = remap(k)
        if g is None:
            skipped.append(k)
            continue
        put(g, sd[k])
    if skipped:
        print(f"  skipped {len(skipped)}: {skipped[:6]}")

    L_e, L_d = int(cfg["encoder_layers"]), int(cfg["decoder_layers"])
    # audio(2) + enc conv/pos/out_ln(7) + enc layers + dec tok/pos/out_ln(4) + dec layers
    expected = 2 + 7 + L_e * 15 + 4 + L_d * 24
    total = n16 + n32
    print(f"  written: {total} (F16 {n16}, F32 {n32}); expected {expected}")
    if total != expected:
        sys.exit("tensor count mismatch")

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"Done: {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")


def main() -> None:
    p = argparse.ArgumentParser(description="sbintuitions/hikari-medium -> GGUF")
    p.add_argument("--input", required=True, type=Path, help="HF model directory (model.safetensors + configs)")
    p.add_argument("--output", required=True, type=Path)
    a = p.parse_args()
    convert(a.input, a.output)


if __name__ == "__main__":
    main()
