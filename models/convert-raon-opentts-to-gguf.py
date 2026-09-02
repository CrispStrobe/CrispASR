#!/usr/bin/env python3
"""Convert KRAFTON Raon-OpenTTS (F5-TTS DiT + HiFi-GAN) → CrispASR f5-tts GGUF.

Raon-OpenTTS IS an F5-TTS DiT (blueprint verified: post_norm=False and
qk_norm=null in both configs, so norm_type=rmsnorm is dead metadata — the
DiT is byte-identical to the stock F5 backbone). The two real deltas vs the
existing f5-tts GGUF are handled here, both outside the transformer:

  1. Vocoder is HiFi-GAN (speechbrain/tts-hifigan-libritts-16kHz generator),
     not Vocos. Its weight-normed state_dict is fused and emitted as `voc.*`
     tensors in the shared core_hifigan layout (same as SpeechT5/FastPitch),
     with the standard v1 hparams as `voc.*` GGUF keys.
  2. Mel is sbhifigan16k: 16 kHz, 80-bin, **slaney** filterbank + slaney mel
     scale, STFT center=False, log(clamp(x,1e-5)). The slaney filterbank is
     a scale-blind trap, so per the copy-the-shipped-filterbank rule this
     script computes fb + window with torchaudio and ships them as
     `f5.mel_fb` / `f5.mel_window`; the runtime does STFT + matmul, never
     rebuilds slaney in C++.

The DiT weights come from the checkpoint's `ema_model_state_dict` (the EMA
weights F5 inference uses); the multi-GB optimizer_state_dict is skipped, and
tensors are read lazily via mmap so a 5.4 GB .pt converts in <2 GB RSS.

Usage:
  python models/convert-raon-opentts-to-gguf.py \
      --checkpoint model_225000.pt --config config.yaml --vocab vocab.txt \
      --hifigan generator.ckpt --output raon-opentts-0.3b-f16.gguf --quant f16
"""

import argparse
import sys
from pathlib import Path

import numpy as np

try:
    import torch
    from safetensors.torch import safe_open  # noqa: F401 (parity with sibling)
except ImportError:
    sys.exit("pip install torch")

try:
    import yaml
except ImportError:
    sys.exit("pip install pyyaml")

try:
    from gguf import GGUFWriter, GGMLQuantizationType
    from gguf import quants as gguf_quants

    _HAS_QUANTS = True
except ImportError:
    try:
        from gguf import GGUFWriter, GGMLQuantizationType

        _HAS_QUANTS = False
    except ImportError:
        sys.exit("pip install gguf")


# ── dtype helpers (mirrors convert-f5-tts-to-gguf.py) ──────────────────────
def to_f16(a):
    return a.detach().to(torch.float16).numpy()


def to_f32(a):
    return a.detach().to(torch.float32).numpy()


def _quant(a, qtype):
    f32 = a.detach().to(torch.float32).numpy()
    if _HAS_QUANTS:
        try:
            return gguf_quants.quantize(f32, qtype), qtype
        except Exception:
            return f32.astype(np.float16), GGMLQuantizationType.F16
    return f32.astype(np.float16), GGMLQuantizationType.F16


# Same conditioning-pathway F32 protection as the f5 converter.
_ALWAYS_F32 = (
    "text_emb", "freqs_cis", "inv_freq", "time_", "conv_pos", "input_proj",
    "input_embed.proj", "final_adaln", "final_proj", "adaln", ".layer_scale",
)


def choose_dtype(name, t, quant):
    n = int(np.prod(list(t.shape)))
    if t.ndim <= 1 or n < 256:
        return to_f32(t), GGMLQuantizationType.F32
    if any(s in name for s in _ALWAYS_F32):
        return to_f32(t), GGMLQuantizationType.F32
    if quant == "q8_0":
        return _quant(t, GGMLQuantizationType.Q8_0)
    if quant == "q4_k":
        return _quant(t, GGMLQuantizationType.Q4_K)
    return to_f16(t), GGMLQuantizationType.F16


# ── DiT name map (verbatim from convert-f5-tts-to-gguf.py) ─────────────────
def map_f5tts_name(hf_name):
    n = hf_name.replace("ema_model.", "").replace("transformer.", "")
    n = n.replace("transformer_blocks.", "blk.")
    n = n.replace(".attn.to_q.", ".attn_q.").replace(".attn.to_k.", ".attn_k.")
    n = n.replace(".attn.to_v.", ".attn_v.").replace(".attn.to_out.0.", ".attn_o.")
    n = n.replace(".attn_norm.linear.", ".adaln.")
    n = n.replace(".ff.ff.0.0.", ".ffn_up.").replace(".ff.ff.2.", ".ffn_down.")
    n = n.replace("text_embed.text_embed.", "text_emb.")
    n = n.replace("text_embed.text_blocks.", "text_blk.")
    n = n.replace(".dwconv.", ".dw.").replace(".pwconv1.", ".pw_up.").replace(".pwconv2.", ".pw_down.")
    n = n.replace(".grn.gamma", ".grn_gamma").replace(".grn.beta", ".grn_beta")
    n = n.replace("time_embed.time_mlp.0.", "time_mlp_0.").replace("time_embed.time_mlp.2.", "time_mlp_1.")
    n = n.replace("input_embed.proj.", "input_proj.")
    n = n.replace("input_embed.conv_pos_embed.conv1d.0.", "conv_pos_0.")
    n = n.replace("input_embed.conv_pos_embed.conv1d.2.", "conv_pos_1.")
    n = n.replace("norm_out.linear.", "final_adaln.").replace("proj_out.", "final_proj.")
    n = n.replace("rotary_embed.inv_freq", "rotary_inv_freq")
    return "f5." + n


# ── HiFi-GAN weight-norm fuse (from convert-speecht5-to-gguf.py) ───────────
def fuse_weight_norm(sd):
    fused, done = {}, set()
    for key in sorted(sd):
        if key.endswith(".weight_g"):
            base = key[:-len(".weight_g")]
            vk = base + ".weight_v"
            if vk in sd:
                g, v = sd[key], sd[vk]
                norm = torch.linalg.norm(v.view(v.shape[0], -1), dim=1)
                w = g.view(-1, *([1] * (v.dim() - 1))) * v / norm.view(-1, *([1] * (v.dim() - 1)))
                fused[base + ".weight"] = w
                done.add(key)
                done.add(vk)
    out = {k: v for k, v in sd.items() if k not in done}
    out.update(fused)
    return out


def map_hifigan_name(n):
    # Raon/speechbrain HiFiGAN wraps each conv in ".conv"; drop it to reach
    # the shared core_hifigan `voc.*` names (SpeechT5/FastPitch layout).
    n = n.replace(".conv.", ".")
    if n.startswith(("conv_pre.", "conv_post.")):
        return "voc." + n
    if n.startswith("ups."):
        return "voc." + n
    if n.startswith("resblocks."):
        return "voc." + n
    return None  # inference_padding buffers etc.


# ── slaney mel filterbank + Hann window (shipped, never rebuilt in C++) ────
def build_mel_buffers(sr, n_fft, n_mels, f_min, f_max):
    import torchaudio

    ms = torchaudio.transforms.MelScale(
        sample_rate=sr, n_stft=n_fft // 2 + 1, n_mels=n_mels,
        f_min=f_min, f_max=f_max, norm="slaney", mel_scale="slaney",
    )
    fb = ms.fb.detach().numpy()  # (n_freqs, n_mels), slaney-normalized
    win = torch.hann_window(n_fft, periodic=True).numpy()  # torchaudio Spectrogram default
    return fb.astype(np.float32), win.astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--vocab", required=True, type=Path)
    ap.add_argument("--hifigan", required=True, type=Path, help="speechbrain generator.ckpt")
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--quant", choices=["f16", "q8_0", "q4_k"], default="f16")
    ap.add_argument("--name", default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    arch = cfg["model"]["arch"]
    mel = cfg["model"]["mel_spec"]
    dim = arch["dim"]
    depth = arch["depth"]
    heads = arch["heads"]
    ff_mult = arch["ff_mult"]
    text_dim = arch["text_dim"]
    conv_layers = arch["conv_layers"]
    n_mels = mel["n_mel_channels"]
    sr = mel["target_sample_rate"]
    hop = mel["hop_length"]
    win_length = mel["win_length"]
    n_fft = mel["n_fft"]
    mel_type = mel["mel_spec_type"]
    assert mel_type == "sbhifigan16k", f"expected sbhifigan16k, got {mel_type}"
    head_dim = dim // heads

    vocab = [ln.rstrip("\n") for ln in open(args.vocab, encoding="utf-8")]
    print(f"arch: dim={dim} depth={depth} heads={heads}x{head_dim} ff_mult={ff_mult} "
          f"n_mels={n_mels} sr={sr} vocab={len(vocab)}", flush=True)

    print("loading DiT (ema) via mmap …", flush=True)
    ckpt = torch.load(str(args.checkpoint), map_location="cpu", weights_only=True, mmap=True)
    ema = ckpt["ema_model_state_dict"]
    dit = {k: v for k, v in ema.items()
           if k.startswith("ema_model.transformer.") and k not in ("initted", "step")}
    print(f"  {len(dit)} DiT tensors", flush=True)

    print("loading HiFi-GAN …", flush=True)
    voc_sd = fuse_weight_norm(torch.load(str(args.hifigan), map_location="cpu", weights_only=True))

    fb, window = build_mel_buffers(sr, n_fft, n_mels, mel.get("fmin", 0.0), mel.get("fmax", 8000.0))
    print(f"  mel fb {fb.shape} (slaney), window {window.shape}", flush=True)

    w = GGUFWriter(str(args.output), arch="f5-tts")
    w.add_name(args.name or args.output.stem)
    # DiT hparams
    w.add_int32("f5.dim", dim)
    w.add_int32("f5.depth", depth)
    w.add_int32("f5.heads", heads)
    w.add_int32("f5.dim_head", head_dim)
    w.add_int32("f5.ff_mult", ff_mult)
    w.add_int32("f5.text_dim", text_dim)
    w.add_int32("f5.text_num_embeds", len(vocab) + 1)
    w.add_int32("f5.conv_layers", conv_layers)
    w.add_int32("f5.mel_dim", n_mels)
    w.add_int32("f5.sample_rate", sr)
    w.add_int32("f5.hop_length", hop)
    w.add_int32("f5.win_length", win_length)
    w.add_int32("f5.n_fft", n_fft)
    # ODE defaults (F5 standard)
    w.add_int32("f5.ode_steps", 32)
    w.add_float32("f5.cfg_strength", 2.0)
    w.add_float32("f5.sway_sampling_coef", -1.0)
    w.add_int32("f5.conv_pos_kernel", 31)
    w.add_int32("f5.conv_pos_groups", 16)
    # Raon deltas — the runtime branches on these
    w.add_string("f5.mel_spec_type", "sbhifigan16k")
    w.add_string("f5.vocoder", "hifigan")
    w.add_string("f5.norm_type", arch.get("norm_type", "layernorm"))  # informational
    w.add_bool("f5.mel_center", False)  # STFT center=False
    # HiFi-GAN v1 hparams (from Raon vocoder.py / speechbrain config)
    w.add_int32("f5.voc_upsample_initial_ch", 512)
    w.add_array("f5.voc_upsample_rates", [8, 8, 2, 2])
    w.add_array("f5.voc_upsample_kernel_sizes", [16, 16, 4, 4])
    w.add_array("f5.voc_resblock_kernel_sizes", [3, 7, 11])
    w.add_bool("f5.voc_normalize_before", False)
    w.add_array("f5.vocab", vocab)

    def add(name, arr, dt):
        if dt not in (GGMLQuantizationType.F32, GGMLQuantizationType.F16):
            w.add_tensor(name, arr, raw_dtype=dt)
        else:
            w.add_tensor(name, arr)

    add("f5.mel_fb", np.ascontiguousarray(fb), GGMLQuantizationType.F32)
    add("f5.mel_window", np.ascontiguousarray(window), GGMLQuantizationType.F32)

    n_dit = 0
    for hf, t in sorted(dit.items()):
        data, dt = choose_dtype(map_f5tts_name(hf), t, args.quant)
        add(map_f5tts_name(hf), data, dt)
        n_dit += 1

    n_voc = 0
    for hf, t in sorted(voc_sd.items()):
        g = map_hifigan_name(hf)
        if g is None:
            continue
        # vocoder convs: F16 bulk, F32 for the tiny conv_post
        if t.ndim <= 1 or int(np.prod(list(t.shape))) < 256:
            data, dt = to_f32(t), GGMLQuantizationType.F32
        else:
            data, dt = to_f16(t), GGMLQuantizationType.F16
        add(g, data, dt)
        n_voc += 1

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"wrote {args.output}: {n_dit} DiT + {n_voc} HiFi-GAN + mel fb/window, "
          f"{len(vocab)} vocab", flush=True)


if __name__ == "__main__":
    main()
