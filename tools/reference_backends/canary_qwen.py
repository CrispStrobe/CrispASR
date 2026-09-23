"""Canary-Qwen (nvidia/canary-qwen-2.5b) reference dump backend.

SALM architecture:
  * FastConformer encoder (32L, d=1024) from canary-1b-flash
  * Linear projection (1024 -> 2048) inside the perception module
  * Qwen3-1.7B LLM decoder with merged LoRA (28L, d=2048, GQA 16/8)

Captures, for crispasr-diff parity against the C++ runtime:

  raw_audio        (N,)             input PCM (added by dump_reference.py)
  mel_spectrogram  (T_mel, n_mels)  perception.preprocessor output, batch-
                                    stripped and transposed to the C++
                                    TimeMels layout (n_mels fast axis).
  projected        (T_enc, 2048)    perception() output = FastConformer encoder
                                    + linear projection. This is what
                                    `canary_qwen_run_encoder` returns on the
                                    C++ side (the encoder graph fuses the
                                    projection), so it is the primary parity
                                    stage. NOT `encoder_output` — the C++ side
                                    has no separate 1024-d capture.
  encoder_1024     (T_enc, 1024)    pre-projection FastConformer output
                                    (diagnostic only; C++ does not compare it).
  llm_argmax       (n_gen,) int32   greedy-decoded token ids (informational).

Requires NeMo >= 2.5.0 (speechlm2). On this env NeMo's transitive httpcore ->
trio import crashes with "unsupported platform"; we neutralise trio (httpcore
falls back to asyncio) before importing NeMo.

Usage:
  python tools/dump_reference.py --backend canary-qwen \\
      --model-dir nvidia/canary-qwen-2.5b \\
      --audio samples/jfk.wav \\
      --output /tmp/canary-qwen-ref.gguf
"""

from __future__ import annotations
try:
    from reference_backends._safe_capture import own as _own
except ImportError:  # run as a standalone script from this directory
    from _safe_capture import own as _own

import sys
from pathlib import Path
from typing import Dict, Set

import numpy as np

# The env's trio raises "unsupported platform" at import; NeMo pulls it in via
# httpcore/httpx, which fall back to asyncio when trio is unavailable.
sys.modules.setdefault("trio", None)

DEFAULT_STAGES = [
    "raw_audio",
    "mel_spectrogram",
    "projected",
    "encoder_1024",
    "llm_argmax",
]


def dump(*, model_dir: Path, audio: np.ndarray, stages: Set[str],
         max_new_tokens: int) -> Dict[str, np.ndarray]:
    """Run canary-qwen (SALM) reference forward and return stage captures."""
    import torch

    sys.modules.setdefault("trio", None)
    try:
        from nemo.collections.speechlm2.models import SALM
    except ImportError as e:
        raise SystemExit(
            "NeMo speechlm2 required.\n"
            "Install: pip install 'nemo_toolkit[asr,tts]'\n"
            f"(import error: {e})")

    pretrained = str(model_dir)
    print(f"  loading NeMo SALM model from {pretrained}")
    model = SALM.from_pretrained(pretrained)
    model.eval()

    # Deterministic mel: disable dither / pad_to on the perception preprocessor.
    feat = getattr(getattr(model.perception, "preprocessor", None), "featurizer", None)
    if feat is not None:
        if hasattr(feat, "dither"):
            feat.dither = 0.0
        if hasattr(feat, "pad_to"):
            feat.pad_to = 0

    dev = next(model.parameters()).device
    audio = audio.astype(np.float32)
    audios = torch.from_numpy(audio).unsqueeze(0).to(dev)
    audio_lens = torch.tensor([audio.shape[0]], dtype=torch.int64, device=dev)

    out: Dict[str, np.ndarray] = {}
    if "raw_audio" in stages:
        out["raw_audio"] = audio

    # Hook the preprocessor (mel) and encoder (pre-projection 1024) inside the
    # perception module; perception() itself returns the projected 2048 embeds.
    cap: Dict[str, object] = {}

    def _hook(name):
        def fn(_m, _inp, o):
            cap[name] = _own(o)
        return fn

    handles = []
    if "mel_spectrogram" in stages and hasattr(model.perception, "preprocessor"):
        handles.append(model.perception.preprocessor.register_forward_hook(_hook("mel")))
    if "encoder_1024" in stages and hasattr(model.perception, "encoder"):
        handles.append(model.perception.encoder.register_forward_hook(_hook("enc")))

    with torch.no_grad():
        projected, proj_len = model.perception(audios, audio_lens)

    for h in handles:
        h.remove()

    # projected: (B=1, T_enc, 2048) -> (T_enc, 2048)
    T_enc = int(proj_len[0].item())
    if "projected" in stages:
        p = projected[0, :T_enc].contiguous().detach().cpu().float().numpy()
        out["projected"] = p

    # mel: hook output is (feats, feat_len); feats (B, n_mels, T_mel) ->
    # (T_mel, n_mels) TimeMels flat layout to match canary_qwen_compute_mel.
    if "mel_spectrogram" in stages and "mel" in cap:
        feats, feat_len = cap["mel"]
        T_valid = int(feat_len[0].item())
        m = feats[0, :, :T_valid].transpose(0, 1).contiguous()
        out["mel_spectrogram"] = m.detach().cpu().float().numpy()

    # encoder_1024: hook output (encoded, len); encoded (B, 1024, T_enc) ->
    # (T_enc, 1024). Diagnostic only.
    if "encoder_1024" in stages and "enc" in cap:
        encoded, enc_len = cap["enc"]
        te = int(enc_len[0].item())
        e = encoded[0, :, :te].transpose(0, 1).contiguous()
        out["encoder_1024"] = e.detach().cpu().float().numpy()

    # Greedy decode ids (informational parity of the decoded token stream).
    if "llm_argmax" in stages:
        with torch.no_grad():
            ids = model.generate(
                prompts=[[{"role": "user",
                           "content": f"Transcribe the following: {model.audio_locator_tag}"}]],
                audios=audios,
                audio_lens=audio_lens,
                max_new_tokens=max_new_tokens,
            )
        out["llm_argmax"] = ids[0].detach().cpu().numpy().astype(np.int32)
        try:
            print(f"  reference transcript: {model.tokenizer.ids_to_text(ids[0].cpu())!r}")
        except Exception:
            pass

    return out
