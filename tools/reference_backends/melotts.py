#!/usr/bin/env python
"""
Reference backend for MeloTTS (VITS2) — dumps intermediate stage outputs
for diff-testing against the C++ melotts runtime.

Usage:
    python tools/reference_backends/melotts.py \
        --ckpt /mnt/storage/melotts-en/checkpoint.pth \
        --config /mnt/storage/melotts-en/config.json \
        --text "Hello world." \
        --output /tmp/melotts-ref.gguf \
        --seed 42

Stages dumped:
    phoneme_ids, tone_ids, lang_ids     input sequences (I32)
    speaker_emb                         emb_g output (F32)
    enc_output                          raw encoder output before proj (F32)
    enc_mean, enc_logvar                projected mean/logvar (F32)
    sdp_logw                            SDP log-durations (F32)
    dp_logw                             DP log-durations (F32)
    durations                           final integer durations (I32)
    z_p                                 sampled latent (F32)
    z_dec                               after flow inverse (F32)
    audio                               final PCM (F32)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

# Ensure we can import melo
MELO_DIR = Path(__file__).resolve().parent.parent.parent.parent / "melotts-ref"
if MELO_DIR.exists():
    sys.path.insert(0, str(MELO_DIR))

try:
    from gguf import GGUFWriter, GGMLQuantizationType
except ImportError:
    sys.exit("pip install gguf")


def main():
    ap = argparse.ArgumentParser(description="MeloTTS reference dump")
    ap.add_argument("--ckpt", required=True, help="checkpoint.pth")
    ap.add_argument("--config", required=True, help="config.json")
    ap.add_argument("--text", default="Hello world.", help="Text to synthesize")
    ap.add_argument("--speaker-id", type=int, default=0)
    ap.add_argument("--noise-scale", type=float, default=0.667)
    ap.add_argument("--length-scale", type=float, default=1.0)
    ap.add_argument("--noise-w", type=float, default=0.8)
    ap.add_argument("--sdp-ratio", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", required=True, help="Output GGUF path")
    ap.add_argument("--language", default="EN")
    args = ap.parse_args()

    import os
    os.environ.setdefault("HF_HOME", "/mnt/storage/huggingface")
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

    from melo.models import SynthesizerTrn
    from melo.text import cleaned_text_to_sequence
    from melo.text.english import g2p, text_normalize
    from melo import commons

    # ── Load config ──
    with open(args.config) as f:
        cfg = json.load(f)

    class HParams:
        def __init__(self, **kw):
            for k, v in kw.items():
                if isinstance(v, dict):
                    v = HParams(**v)
                setattr(self, k, v)

    hps = HParams(**cfg)
    symbols = cfg["symbols"]
    symbol_to_id = {s: i for i, s in enumerate(symbols)}

    # ── Build and load model ──
    model = SynthesizerTrn(
        len(symbols),
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        num_tones=cfg["num_tones"],
        num_languages=cfg["num_languages"],
        **vars(hps.model),
    )
    model.eval()
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"], strict=True)
    del ckpt
    print(f"Model loaded: {sum(p.numel() for p in model.parameters())/1e6:.1f}M params")

    # ── Text processing ──
    norm_text = text_normalize(args.text)
    phones, tones, word2ph = g2p(norm_text)
    print(f"Phones: {phones}")

    phone_ids, tone_ids, lang_ids = cleaned_text_to_sequence(
        phones, tones, args.language, symbol_to_id
    )
    if hps.data.add_blank:
        phone_ids = commons.intersperse(phone_ids, 0)
        tone_ids = commons.intersperse(tone_ids, 0)
        lang_ids = commons.intersperse(lang_ids, 0)
        for i in range(len(word2ph)):
            word2ph[i] *= 2
        word2ph[0] += 1

    T = len(phone_ids)
    print(f"Sequence length (with blanks): {T}")

    # ── BERT features ──
    # For English: ja_bert(768) carries BERT features, bert(1024) is zeros
    from melo.text import get_bert

    bert_features = get_bert(norm_text, word2ph, args.language, "cpu")
    print(f"BERT features: {bert_features.shape}")

    if args.language in ["EN", "ZH_MIX_EN", "JP", "KR", "SP", "ES", "FR"]:
        ja_bert = bert_features
        bert = torch.zeros(1024, T)
    else:
        bert = bert_features
        ja_bert = torch.zeros(768, T)

    x = torch.LongTensor(phone_ids).unsqueeze(0)
    t_tone = torch.LongTensor(tone_ids).unsqueeze(0)
    l_lang = torch.LongTensor(lang_ids).unsqueeze(0)
    x_len = torch.LongTensor([T])
    sid = torch.LongTensor([args.speaker_id])
    b = bert.unsqueeze(0)
    jb = ja_bert.unsqueeze(0)

    # ── Set seed for reproducibility ──
    torch.manual_seed(args.seed)

    # ── Instrumented inference ──
    # We replicate SynthesizerTrn.infer() step by step to capture intermediates
    stages = {}

    stages["phoneme_ids"] = np.array(phone_ids, dtype=np.int32)
    stages["tone_ids"] = np.array(tone_ids, dtype=np.int32)
    stages["lang_ids"] = np.array(lang_ids, dtype=np.int32)
    stages["ja_bert_features"] = ja_bert.cpu().numpy().astype(np.float32)

    with torch.no_grad():
        # 1. Speaker embedding
        g = model.emb_g(sid).unsqueeze(-1)  # (1, 256, 1)
        stages["speaker_emb"] = g[0].cpu().numpy().astype(np.float32)

        g_p = g  # use_vc=False

        # 2. Text encoder
        enc_x, m_p, logs_p, x_mask = model.enc_p(
            x, x_len, t_tone, l_lang, b, jb, g=g_p
        )
        stages["enc_output"] = enc_x[0].cpu().numpy().astype(np.float32)
        stages["enc_mean"] = m_p[0].cpu().numpy().astype(np.float32)
        stages["enc_logvar"] = logs_p[0].cpu().numpy().astype(np.float32)

        # 3. Duration prediction
        torch.manual_seed(args.seed + 1)  # separate seed for SDP noise
        sdp_logw = model.sdp(enc_x, x_mask, g=g, reverse=True, noise_scale=args.noise_w)
        dp_logw = model.dp(enc_x, x_mask, g=g)
        stages["sdp_logw"] = sdp_logw[0, 0].cpu().numpy().astype(np.float32)
        stages["dp_logw"] = dp_logw[0, 0].cpu().numpy().astype(np.float32)

        logw = sdp_logw * args.sdp_ratio + dp_logw * (1 - args.sdp_ratio)
        w = torch.exp(logw) * x_mask * args.length_scale
        w_ceil = torch.ceil(w)
        y_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()
        y_mask = torch.unsqueeze(
            commons.sequence_mask(y_lengths, None), 1
        ).to(x_mask.dtype)
        attn_mask = torch.unsqueeze(x_mask, 2) * torch.unsqueeze(y_mask, -1)
        attn = commons.generate_path(w_ceil, attn_mask)

        durations = w_ceil[0, 0].cpu().numpy().astype(np.int32)
        stages["durations"] = durations

        # 4. Expand prior
        m_p_exp = torch.matmul(attn.squeeze(1), m_p.transpose(1, 2)).transpose(1, 2)
        logs_p_exp = torch.matmul(attn.squeeze(1), logs_p.transpose(1, 2)).transpose(
            1, 2
        )

        # 5. Sample latent
        torch.manual_seed(args.seed + 2)
        z_p = m_p_exp + torch.randn_like(m_p_exp) * torch.exp(logs_p_exp) * args.noise_scale
        stages["z_p"] = z_p[0].cpu().numpy().astype(np.float32)

        # 6. Flow inverse
        z = model.flow(z_p, y_mask, g=g, reverse=True)
        stages["z_dec"] = z[0].cpu().numpy().astype(np.float32)

        # 7. Decoder
        o = model.dec((z * y_mask)[:, :, :], g=g)
        audio_np = o[0, 0].cpu().numpy().astype(np.float32)
        stages["audio"] = audio_np

    print(f"Audio: {audio_np.shape}, dur={len(audio_np)/44100:.2f}s")

    # ── Write GGUF ──
    writer = GGUFWriter(args.output, arch="melotts-ref")
    writer.add_uint32("melotts.seed", args.seed)
    writer.add_string("melotts.text", args.text)
    writer.add_uint32("melotts.speaker_id", args.speaker_id)

    for name, arr in sorted(stages.items()):
        print(f"  {name}: {arr.shape} {arr.dtype}")
        if arr.dtype == np.int32:
            writer.add_tensor(name, arr)
        else:
            writer.add_tensor(name, arr.astype(np.float32))

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
