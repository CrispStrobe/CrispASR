"""VibeVoice-ASR-7B reference dump backend.

Captures every stage needed to validate the C++ ggml pipeline:

  audio_norm          (N,)          normalized PCM at 24kHz (-25 dBFS)
  at_enc_mean         (1, T', 64)   acoustic encoder VAE mean (pre-sampling)
  at_tokens           (1, T', 64)   acoustic tokens after sampling
  st_enc_mean         (1, T', 128)  semantic encoder output (mean, no sampling)
  at_conn_out         (1, T', D)    after acoustic SpeechConnector
  st_conn_out         (1, T', D)    after semantic SpeechConnector
  speech_features     (1, T', D)    combined (at_conn + st_conn)
  llm_argmax          (T_gen,)      greedy generated token IDs
  generated_text      str           decoded transcript

Key facts (7B ASR model):
  - Sample rate: 24 kHz (NOT 16 kHz like most ASR models)
  - Audio normalization: -25 dBFS before encoding
  - vae_tok_len = ceil(samples / 3200)  [product of encoder ratios]
  - Acoustic: std_dist_type='gaussian', fix_std=0.5, vae_dim=64
  - Semantic:  std_dist_type='none' (mean only), vae_dim=128
  - SpeechConnector: Linear(vae_dim→D) + RMSNorm + Linear(D→D)
  - Combined = acoustic_connector(at_tokens) + semantic_connector(st_mean)
  - Prompt: system + user(<speech_start>N×<speech_pad><speech_end>
            + "This is a X.XX seconds audio, please transcribe it with
            these keys: Start time, End time, Speaker ID, Content")
  - Output format: JSON with Start time / End time / Speaker ID / Content

Usage:
  python tools/dump_reference.py --backend vibevoice \\
      --model-dir /Volumes/backups/ai/hub/models--microsoft--VibeVoice-ASR/snapshots/<hash> \\
      --audio samples/jfk_24k.wav \\
      --output /tmp/vibevoice-ref.gguf
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Dict, Set

import numpy as np

DEFAULT_STAGES = [
    "audio_norm",
    "at_enc_mean",
    "at_tokens",
    "st_enc_mean",
    "at_conn_out",
    "st_conn_out",
    "speech_features",
    "llm_argmax",
    "generated_text",
]

_SR = 24000
_COMPRESS_RATIO = 3200   # product of encoder ratios [8,5,5,4,2,2]


def _normalize_audio(audio: np.ndarray, target_dB_FS: float = -25.0, eps: float = 1e-6) -> np.ndarray:
    rms = np.sqrt(np.mean(audio ** 2))
    audio = audio * (10 ** (target_dB_FS / 20) / (rms + eps))
    max_val = np.abs(audio).max()
    if max_val > 1.0:
        audio = audio / (max_val + eps)
    return audio.astype(np.float32)


def _resample_to_24k(audio: np.ndarray, src_sr: int) -> np.ndarray:
    """Resample from src_sr to 24 kHz. Uses librosa if available, else linear."""
    if src_sr == _SR:
        return audio
    try:
        import librosa
        return librosa.resample(audio, orig_sr=src_sr, target_sr=_SR).astype(np.float32)
    except ImportError:
        new_len = int(len(audio) * _SR / src_sr)
        return np.interp(np.linspace(0, len(audio) - 1, new_len),
                         np.arange(len(audio)), audio).astype(np.float32)


def dump(*, model_dir: Path, audio: np.ndarray, stages: Set[str],
         max_new_tokens: int) -> Dict[str, np.ndarray]:
    """Run VibeVoice-ASR forward and return stage captures.

    `audio` arrives as 16 kHz mono float32 from the shared loader in
    dump_reference.py. We resample to 24 kHz internally.
    """
    import torch

    # vibevoice package lives in the cloned repo, try both locations
    for vv_path in ["/tmp/VibeVoice", str(Path(__file__).parent.parent.parent / "third_party/VibeVoice")]:
        if Path(vv_path).exists() and vv_path not in sys.path:
            sys.path.insert(0, vv_path)
    try:
        from vibevoice.modular.modeling_vibevoice_asr import VibeVoiceASRForConditionalGeneration
        from vibevoice.processor.vibevoice_asr_processor import VibeVoiceASRProcessor
    except ImportError as e:
        raise SystemExit(
            "vibevoice package not found.\n"
            "Clone with: git clone https://github.com/microsoft/VibeVoice /tmp/VibeVoice\n"
            f"(import error: {e})"
        )

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    dtype = torch.float32   # MPS/CPU: bfloat16 not fully supported

    print(f"  loading VibeVoice-ASR from {model_dir}  [{device}, {dtype}]")
    model = VibeVoiceASRForConditionalGeneration.from_pretrained(
        str(model_dir), torch_dtype=dtype, trust_remote_code=True,
        attn_implementation="sdpa",
    ).to(device).eval()

    processor = VibeVoiceASRProcessor.from_pretrained(
        str(model_dir), language_model_pretrained_name="Qwen/Qwen2.5-7B"
    )

    # ── Resample from 16kHz → 24kHz and normalize ────────────────────────────
    audio24 = _resample_to_24k(audio, src_sr=16000)
    norm_audio = _normalize_audio(audio24)
    duration = len(norm_audio) / _SR

    out: Dict[str, np.ndarray] = {}
    if "audio_norm" in stages:
        out["audio_norm"] = norm_audio

    # ── Stage-by-stage encoder captures ──────────────────────────────────────
    speech_t = torch.tensor(norm_audio, dtype=dtype, device=device).unsqueeze(0)  # [1, T]

    at = model.model.acoustic_tokenizer
    st = model.model.semantic_tokenizer
    at_conn = model.model.acoustic_connector
    st_conn = model.model.semantic_connector

    with torch.no_grad():
        # Acoustic encoder → VAE mean
        at_enc_out = at.encode(speech_t.unsqueeze(1))   # input [1, 1, T]
        at_mean = at_enc_out.mean                        # [1, T', 64]
        if "at_enc_mean" in stages:
            out["at_enc_mean"] = at_mean.cpu().float().numpy().squeeze(0)  # (T', 64)

        # Acoustic sampling (gaussian: mean + per-batch-std * noise)
        # For deterministic reference dumps, use mean directly.
        at_tokens = at_mean          # skip noise for reproducible diffs
        if "at_tokens" in stages:
            out["at_tokens"] = at_tokens.cpu().float().numpy().squeeze(0)  # (T', 64)

        # Semantic encoder → mean only (std_dist_type='none')
        st_enc_out = st.encode(speech_t.unsqueeze(1))
        st_mean = st_enc_out.mean                        # [1, T', 128]
        if "st_enc_mean" in stages:
            out["st_enc_mean"] = st_mean.cpu().float().numpy().squeeze(0)  # (T', 128)

        # Connectors
        at_feat = at_conn(at_tokens)                     # [1, T', D]
        if "at_conn_out" in stages:
            out["at_conn_out"] = at_feat.cpu().float().numpy().squeeze(0)  # (T', D)

        st_feat = st_conn(st_mean)                       # [1, T', D]
        if "st_conn_out" in stages:
            out["st_conn_out"] = st_feat.cpu().float().numpy().squeeze(0)  # (T', D)

        combined = at_feat + st_feat                     # [1, T', D]
        if "speech_features" in stages:
            out["speech_features"] = combined.cpu().float().numpy().squeeze(0)  # (T', D)

    # ── Full generation ───────────────────────────────────────────────────────
    want_argmax = "llm_argmax" in stages
    want_text   = "generated_text" in stages

    if want_argmax or want_text:
        # Build prompt using the audio file path or the raw array + sr
        # The processor expects either a file path or a (array, sr) tuple.
        # We pass the normalized array at 24k.
        inputs = processor(
            audio=(norm_audio, _SR),   # tuple: (array, sample_rate)
            return_tensors="pt",
            add_generation_prompt=True,
        )
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                  for k, v in inputs.items()}

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=processor.pad_id,
                eos_token_id=processor.tokenizer.eos_token_id,
            )

        input_len = inputs["input_ids"].shape[1]
        gen_ids = output_ids[0, input_len:]

        if want_argmax:
            out["llm_argmax"] = gen_ids.cpu().int().numpy().astype(np.int32)

        if want_text:
            text = processor.decode(gen_ids, skip_special_tokens=True)
            out["generated_text"] = text
            print(f"\n  Transcript:\n{text}\n")

    return out
