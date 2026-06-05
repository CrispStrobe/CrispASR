"""FunAudioLLM/Fun-ASR-Nano-2512 (and Fun-ASR-MLT-Nano-2512) reference dump.

LLM-decoder path only — the published model.pt for both releases ships
**only** `audio_encoder.* + audio_adaptor.* + llm.*` (no CTC decoder /
head weights). Stage 1 is therefore the full LLM-decoder pipeline:
WavFrontend (hamming kaldi-fbank + LFR(7,6)) → 70 SANM encoder blocks
→ 2-layer transformer audio_adaptor → ChatML prompt with audio embeds
spliced at the `<|startofspeech|>`/`<|endofspeech|>` slot → Qwen3-0.6B
AR decode.

Stages exposed (all optional; absent → silent skip):

  raw_audio                     (N,)             F32 PCM
  mel_features                  (T_lfr, 560)     F32 post-WavFrontend+LFR
  encoder_layer_K   K=0..69     (T_lfr, 512)     F32 after SANM block K
  encoder_main_out              (T_lfr, 512)     F32 after `after_norm`
  encoder_output                (T_lfr, 512)     F32 after `tp_norm`
  audio_adaptor_layer_K K=0..1  (T_adapter, 1024) F32 after adaptor block K
  audio_adaptor_output          (T_adapter, 1024) F32 final adaptor output
  generated_text                (utf-8 bytes)    decoded transcript

Note: per-step llm_logits during AR decode are not captured here yet —
adding them requires intercepting Qwen3's generate() loop. The encoder
+ adaptor stages are sufficient to gate the C++ runtime up to the LLM
splice point; LLM correctness is checked against `generated_text` byte
similarity.

Usage:

  HF_HOME=/Volumes/backups/ai/huggingface-hub \\
  HUGGINGFACE_HUB_CACHE=/Volumes/backups/ai/huggingface-hub \\
  python tools/dump_reference.py --backend funasr \\
      --model-dir FunAudioLLM/Fun-ASR-Nano-2512 \\
      --audio samples/jfk.wav \\
      --output /Volumes/backups/ai/huggingface-hub/funasr/funasr-ref.gguf

Acceptance bar (elementwise vs C++):
  mel_features         cos >= 0.999 (frontend bit-near-exact)
  encoder_layer_K      cos >= 0.99  (F16, 70 blocks accumulate)
  audio_adaptor_output cos >= 0.99
  generated_text       byte-identical (English / Chinese transcripts)
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
        "encoder_main_out",
        "encoder_output",
        "audio_adaptor_output",
        "generated_text",
    ]
    + [f"encoder_layer_{i}" for i in range(70)]
    + [f"audio_adaptor_layer_{i}" for i in range(2)]
)


def _ensure_fun_asr_nano_importable() -> None:
    """funasr 1.3.1's fun_asr_nano/model.py uses `from ctc import CTC` instead
    of the relative form. Without this shim the import fails with
    `ModuleNotFoundError: No module named 'ctc'`. Prepending the package
    directory to sys.path makes the (mis-spelt) absolute import resolve to
    the same ctc.py we'd want anyway. Upstream packaging bug; remove this
    workaround when funasr ships a relative import.
    """
    import funasr

    fan = Path(funasr.__file__).parent / "models" / "fun_asr_nano"
    s = str(fan)
    if s not in sys.path:
        sys.path.insert(0, s)


def dump(*, model_dir: Path, audio: np.ndarray, stages: Set[str],
         max_new_tokens: int) -> Dict[str, np.ndarray]:
    """Load Fun-ASR-Nano-2512 (or MLT variant), run the LLM path on `audio`,
    return captured activations as a dict[str, np.ndarray]."""
    import torch

    _ensure_fun_asr_nano_importable()
    from funasr.models.fun_asr_nano.model import FunASRNano
    from funasr.frontends.wav_frontend import WavFrontend

    pretrained = str(model_dir)
    print(f"  loading FunASRNano from {pretrained}")
    m, kwargs = FunASRNano.from_pretrained(model=pretrained, device="cpu")
    m.eval()

    # Deterministic frontend — overrides the loaded WavFrontend's dither
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

    # Hook plumbing
    from . import _hooks
    captured: Dict = {}
    stage_modules = []

    enc = m.audio_encoder
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

    ada = getattr(m, "audio_adaptor", None)
    if ada is not None and ada.blocks is not None:
        for i, b in enumerate(ada.blocks):
            name = f"audio_adaptor_layer_{i}"
            if name in stages:
                stage_modules.append((name, b))

    handles = _hooks.capture_modules(captured, stage_modules)

    with torch.no_grad():
        feats, feats_lens = fe(sig, sig_len)
        T_lfr = int(feats_lens.item())
        if "mel_features" in stages:
            out["mel_features"] = feats[0, :T_lfr].cpu().float().numpy()

        encoder_out, encoder_out_lens = m.audio_encoder(feats, feats_lens)
        T_enc = int(encoder_out_lens.item())
        if "encoder_output" in stages:
            out["encoder_output"] = encoder_out[0, :T_enc].cpu().float().numpy()

        if ada is not None:
            adaptor_out, adaptor_out_lens = m.audio_adaptor(encoder_out, encoder_out_lens)
            T_ada = int(adaptor_out_lens.item())
            if "audio_adaptor_output" in stages:
                out["audio_adaptor_output"] = adaptor_out[0, :T_ada].cpu().float().numpy()

    _hooks.drop_hooks(handles)
    out.update(_hooks.finalize(captured, T_max=T_lfr))

    # Full inference: capture generated_text. Skips if the upstream model is
    # missing its tokenizer (kwargs dict from from_pretrained carries it),
    # which can happen when the snapshot path doesn't have the Qwen3 dir.
    if "generated_text" in stages:
        try:
            # Save audio as a temp wav so data_load_speech can read it.
            import tempfile, wave
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                tmp_wav = f.name
            with wave.open(tmp_wav, "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(16000)
                pcm16 = np.clip(audio * 32767, -32768, 32767).astype(np.int16)
                w.writeframes(pcm16.tobytes())
            res = m.inference(data_in=[tmp_wav], **kwargs)
            # res shape: list[list[dict]] from inference_llm; the outer list
            # is from generate(), inner is per-utterance
            txt = ""
            try:
                txt = res[0][0]["text"] if isinstance(res[0], list) else res[0]["text"]
            except Exception:
                pass
            # Routed into the GGUF metadata KV table (not a tensor) — the
            # dump_reference top-level scans for str-typed captures and
            # moves them there, where the C++ diff harness reads it via
            # Ref.meta("generated_text").
            out["generated_text"] = txt
            print(f"  generated_text: {txt!r}")
        except Exception as e:
            print(f"  WARN: inference() failed: {e}; skipping generated_text")

    return out
