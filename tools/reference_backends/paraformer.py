"""FunASR Paraformer-zh reference dump.

NAR encoder-decoder: Kaldi fbank → LFR(7,6) → CMVN → 50 SANM encoder
blocks → CifPredictorV2 → 16 decoder blocks (FFN → FSMN → cross-attn) →
decoders3 post block → after_norm → output_layer → argmax → characters.

Stages exposed (all optional; absent → silent skip):

  raw_audio                     (N,)             F32 PCM
  mel_features                  (T_lfr, 560)     F32 post-WavFrontend+LFR
  encoder_layer_K   K=0..49     (T_lfr, 512)     F32 after SANM block K
  encoder_output                (T_lfr, 512)     F32 after encoder.after_norm
  cif_alphas                    (T_cif,)         F32 CIF alpha per timestep
  acoustic_embeds               (N_tokens, 512)  F32 CIF-fired token vectors
  decoder_layer_K   K=0..15     (N_tokens, 512)  F32 after decoder block K
  decoder_output                (N_tokens, 512)  F32 after decoder.after_norm
  generated_text                (utf-8 bytes)    decoded transcript

Usage:

  python tools/dump_reference.py --backend paraformer \\
      --model-dir /mnt/storage/paraformer-zh-upstream \\
      --audio /mnt/storage/paraformer-zh-upstream/example/asr_example.wav \\
      --output /mnt/storage/paraformer-zh/paraformer-zh-ref.gguf

Acceptance bar (elementwise vs C++):
  mel_features         cos >= 0.999  (frontend bit-near-exact)
  encoder_layer_K      cos >= 0.99   (F16, 50 blocks accumulate)
  encoder_output       cos >= 0.999
  acoustic_embeds      cos >= 0.99   (CIF sequential float accumulation)
  generated_text       byte-identical
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Set

import numpy as np


DEFAULT_STAGES = (
    [
        "raw_audio",
        "mel_features",
        "encoder_output",
        "cif_alphas",
        "acoustic_embeds",
        "decoder_output",
        "generated_text",
    ]
    + [f"encoder_layer_{i}" for i in range(50)]
    + [f"decoder_layer_{i}" for i in range(16)]
)


def dump(*, model_dir: Path, audio: np.ndarray, stages: Set[str],
         max_new_tokens: int) -> Dict[str, np.ndarray]:
    """Load Paraformer-zh, run inference on `audio`, return captured
    activations as a dict[str, np.ndarray]."""
    import torch
    from funasr import AutoModel
    from funasr.frontends.wav_frontend import WavFrontend

    pretrained = str(model_dir)
    print(f"  loading Paraformer-zh from {pretrained}")
    m = AutoModel(model=pretrained, device='cpu')
    model = m.model
    model.eval()

    # Deterministic frontend
    fe = WavFrontend(
        cmvn_file=None, fs=16000, window="hamming", n_mels=80,
        frame_length=25, frame_shift=10, lfr_m=7, lfr_n=6,
        dither=0.0, upsacle_samples=True, snip_edges=True,
    )
    fe.eval()

    sig = torch.from_numpy(audio.astype(np.float32))[None, :]
    sig_len = torch.tensor([audio.shape[0]])

    out: Dict[str, np.ndarray] = {}
    if "raw_audio" in stages:
        out["raw_audio"] = audio.astype(np.float32)

    # Hook plumbing
    from . import _hooks
    captured: Dict = {}
    stage_modules = []

    enc = model.encoder
    base_layers = list(enc.encoders0) + list(enc.encoders)
    for i, layer in enumerate(base_layers):
        name = f"encoder_layer_{i}"
        if name in stages:
            stage_modules.append((name, layer))
    if "encoder_output" in stages and hasattr(enc, "after_norm"):
        stage_modules.append(("encoder_output", enc.after_norm))

    dec = model.decoder
    for i, layer in enumerate(dec.decoders):
        name = f"decoder_layer_{i}"
        if name in stages:
            stage_modules.append((name, layer))
    if "decoder_output" in stages and hasattr(dec, "after_norm"):
        stage_modules.append(("decoder_output", dec.after_norm))

    handles = _hooks.capture_modules(captured, stage_modules)

    # CIF capture via monkey-patching
    cif_captures = {}
    need_cif = ("cif_alphas" in stages) or ("acoustic_embeds" in stages)
    if need_cif:
        orig_calc = model.calc_predictor.__func__

        def patched_calc(self, encoder_out, encoder_out_lens, *args, **kwargs):
            result = orig_calc(self, encoder_out, encoder_out_lens, *args, **kwargs)
            # result = (acoustic_embeds, token_num, alphas, cif_peak)
            cif_captures['alphas'] = result[2].detach().cpu().float().numpy()
            cif_captures['acoustic_embeds'] = result[0].detach().cpu().float().numpy()
            return result

        model.calc_predictor = patched_calc.__get__(model)

    with torch.no_grad():
        feats, feats_lens = fe(sig, sig_len)
        T_lfr = int(feats_lens.item())
        if "mel_features" in stages:
            out["mel_features"] = feats[0, :T_lfr].cpu().float().numpy()

        # Run full forward to trigger all hooks + CIF capture
        enc_result = model.encoder(feats, feats_lens)
        encoder_out, encoder_out_lens = enc_result[0], enc_result[1]
        T_enc = int(encoder_out_lens.item())

        # CIF predictor
        pre_acoustic_embeds, pre_token_num, alphas, pre_peak = \
            model.calc_predictor(encoder_out, encoder_out_lens)

        # Decoder
        pre_token_num_int = pre_token_num.floor().int()
        decoder_outs = model.decoder(
            encoder_out, encoder_out_lens,
            pre_acoustic_embeds, pre_token_num_int,
        )

    _hooks.drop_hooks(handles)
    out.update(_hooks.finalize(captured, T_max=max(T_lfr, T_enc)))

    # CIF outputs
    if "cif_alphas" in stages and 'alphas' in cif_captures:
        out["cif_alphas"] = cif_captures['alphas'][0]  # squeeze batch
    if "acoustic_embeds" in stages and 'acoustic_embeds' in cif_captures:
        N_tok = int(pre_token_num.floor().int().item())
        out["acoustic_embeds"] = cif_captures['acoustic_embeds'][0, :N_tok].astype(np.float32)

    # Full inference for generated_text
    if "generated_text" in stages:
        try:
            import tempfile, wave
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                tmp_wav = f.name
            with wave.open(tmp_wav, "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(16000)
                pcm16 = np.clip(audio * 32767, -32768, 32767).astype(np.int16)
                w.writeframes(pcm16.tobytes())
            res = m.generate(input=tmp_wav)
            txt = res[0]['text'] if res else ""
            out["generated_text"] = txt
            print(f"  generated_text: {txt!r}")
        except Exception as e:
            print(f"  WARN: generate() failed: {e}; skipping generated_text")

    return out
