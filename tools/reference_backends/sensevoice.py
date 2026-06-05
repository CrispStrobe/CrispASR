"""FunAudioLLM/SenseVoiceSmall reference dump for the diff harness.

Encoder-only multi-task ASR: kaldi-fbank (hamming) + LFR(7, 6) → prepend
4 query embeds → 70-block SANM encoder → CTC head → greedy decode →
SentencePiece detokenize.

Stages exposed:

  raw_audio              (N,)              F32 PCM
  mel_features           (T_lfr, 560)      F32 post-WavFrontend+LFR
  encoder_input          (T_lfr+4, 560)    F32 after 4-query-embed prepend
  encoder_layer_K K=0..69 (T_lfr+4, 512)   F32 after SANM block K
  encoder_main_out       (T_lfr+4, 512)    F32 after `after_norm`
  encoder_output         (T_lfr+4, 512)    F32 after `tp_norm`
  ctc_logits             (T_lfr+4, 25055)  F32 CTC head logits
  generated_text         str               final transcript (with rich prefix)

Usage:

  HF_HOME=/Volumes/backups/ai/huggingface-hub \\
  HUGGINGFACE_HUB_CACHE=/Volumes/backups/ai/huggingface-hub \\
  python tools/dump_reference.py --backend sensevoice \\
      --model-dir FunAudioLLM/SenseVoiceSmall \\
      --audio samples/jfk.wav \\
      --output /tmp/sensevoice-ref.gguf
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
        "encoder_input",
        "encoder_main_out",
        "encoder_output",
        "ctc_logits",
        "generated_text",
    ]
    + [f"encoder_layer_{i}" for i in range(70)]
)


def dump(*, model_dir: Path, audio: np.ndarray, stages: Set[str],
         max_new_tokens: int) -> Dict[str, np.ndarray]:
    import torch
    from funasr import AutoModel
    from funasr.models.sense_voice.model import SenseVoiceSmall  # noqa: F401
    from funasr.frontends.wav_frontend import WavFrontend

    pretrained = str(model_dir)
    print(f"  loading SenseVoiceSmall via AutoModel from {pretrained}")
    m = AutoModel(model=pretrained, disable_update=True, trust_remote_code=True)
    sv = m.model  # underlying torch nn.Module
    tokenizer = m.kwargs.get("tokenizer")
    sv.eval()

    fe = WavFrontend(
        cmvn_file=None, fs=16000, window="hamming", n_mels=80,
        frame_length=25, frame_shift=10, lfr_m=7, lfr_n=6, dither=0.0,
        upsacle_samples=True, snip_edges=True,
    )
    fe.eval()

    sig = torch.from_numpy(audio.astype(np.float32))[None, :]
    sig_len = torch.tensor([audio.shape[0]])

    out: Dict[str, np.ndarray] = {}
    if "raw_audio" in stages:
        out["raw_audio"] = audio.astype(np.float32)

    from . import _hooks
    captured: Dict = {}
    stage_modules = []

    enc = sv.encoder
    base_layers = list(enc.encoders0) + list(enc.encoders)
    for i, layer in enumerate(base_layers):
        name = f"encoder_layer_{i}"
        if name in stages:
            stage_modules.append((name, layer))
    n_base = len(base_layers)
    for j, layer in enumerate(getattr(enc, "tp_encoders", [])):
        name = f"encoder_layer_{n_base + j}"
        if name in stages:
            stage_modules.append((name, layer))
    if "encoder_main_out" in stages and hasattr(enc, "after_norm"):
        stage_modules.append(("encoder_main_out", enc.after_norm))

    handles = _hooks.capture_modules(captured, stage_modules)

    with torch.no_grad():
        # 1. Frontend → (T_lfr, 560)
        feats, feats_lens = fe(sig, sig_len)
        T_lfr = int(feats_lens.item())
        if "mel_features" in stages:
            out["mel_features"] = feats[0, :T_lfr].cpu().float().numpy()

        # 2. Prepend the 4 query embeds (textnorm + language + event + emotion).
        # Replicate SenseVoiceSmall.inference's prepend order: first textnorm
        # is concat'd to speech, then [language, event=1, emo=2] is prepended.
        # End result: [lang, event, emo, textnorm, speech_features].
        lang = "en"  # auto-detect via diff against ASR backend's first prefix
        # The reference dump targets the multilingual "auto" path —
        # match what the runtime uses: language="auto" → lang_id=0.
        textnorm = "withitn"
        language_query = sv.embed(
            torch.LongTensor([[sv.lid_dict["auto"]]])
        ).repeat(feats.size(0), 1, 1)
        textnorm_query = sv.embed(
            torch.LongTensor([[sv.textnorm_dict[textnorm]]])
        ).repeat(feats.size(0), 1, 1)
        event_emo_query = sv.embed(
            torch.LongTensor([[1, 2]])
        ).repeat(feats.size(0), 1, 1)

        # Compose the prepended sequence in the same order the runtime does:
        # [lang, event, emo, textnorm, fbank...].
        speech_with_textnorm = torch.cat((textnorm_query, feats), dim=1)
        speech = torch.cat((language_query, event_emo_query, speech_with_textnorm), dim=1)
        speech_lens = feats_lens + 4
        T_total = int(speech_lens.item())

        if "encoder_input" in stages:
            out["encoder_input"] = speech[0, :T_total].cpu().float().numpy()

        # 3. Encoder forward
        encoder_out, encoder_out_lens = sv.encoder(speech, speech_lens)
        if isinstance(encoder_out, tuple):
            encoder_out = encoder_out[0]
        if "encoder_output" in stages:
            out["encoder_output"] = encoder_out[0, :T_total].cpu().float().numpy()

        # 4. CTC head
        if "ctc_logits" in stages:
            ctc = sv.ctc.ctc_lo(encoder_out)
            out["ctc_logits"] = ctc[0, :T_total].cpu().float().numpy()

    _hooks.drop_hooks(handles)
    out.update(_hooks.finalize(captured, T_max=T_total))

    # 5. Full inference for the text comparison
    if "generated_text" in stages and tokenizer is not None:
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
            res = m.inference(input=tmp_wav, language="auto", use_itn=True,
                              cache={}, batch_size_s=60, merge_vad=False)
            txt = ""
            try:
                if isinstance(res, tuple):
                    res = res[0]
                txt = res[0]["text"] if isinstance(res[0], dict) else str(res[0])
            except Exception:
                pass
            out["generated_text"] = txt
            print(f"  generated_text: {txt!r}")
        except Exception as e:
            print(f"  WARN: inference() failed: {e}; skipping generated_text")

    return out
