"""ARK-ASR-3B reference dump backend (PLAN §ARK).

Instruments AutoArk-AI/ARK-ASR-3B (ArkasrForConditionalGeneration, custom
remote code) and emits stage activations following the contract in
tools/dump_reference.py. The C++ side (crispasr-diff arkasr) compares:

  mel_spectrogram  — WhisperFeatureExtractor log-mel (128, T)
  audio_embeds     — encoder + MLP adapter output (N, hidden) [validates the
                     conv stem, partial RoPE, and adapter]
  first_logits     — decoder logits at the last prompt position (vocab,)
                     [validates the Qwen2 decoder + audio-token injection]
  generated_text   — greedy transcript (optional, end-to-end sanity)
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Set

import numpy as np

DEFAULT_STAGES = [
    "mel_spectrogram",
    "audio_embeds",
    "first_logits",
    "generated_text",
]


def dump(*, model_dir: Path, audio: np.ndarray, stages: Set[str], max_new_tokens: int) -> Dict[str, np.ndarray]:
    import os

    import torch
    from transformers import AutoModelForCausalLM, AutoProcessor

    # Footprint control on a 16 GB M1: only load the 3B model for stages that
    # actually need it. `mel_spectrogram` needs just the feature extractor
    # (~tiny); the encoder-only `audio_embeds` could in principle load only the
    # submodule, but transformers gives us the whole model, so audio_embeds /
    # first_logits / generated_text all trigger the full load. Run mel alone when
    # RAM is tight; defer the model-heavy stages to a free machine.
    needs_model = bool(stages & {"audio_embeds", "first_logits", "llm_input_ids", "generated_text"})

    # Default to bfloat16: a float32 3B forward needs ~12 GB RAM and OOM-crashes
    # a 16 GB box. bf16 (~6 GB weights) keeps the cosine gate valid (>0.99 vs the
    # F16 ggml port). Override with ARKASR_REF_DTYPE=float32 on a big machine.
    dtype_name = os.environ.get("ARKASR_REF_DTYPE", "bfloat16")
    dtype = getattr(torch, dtype_name)
    processor = AutoProcessor.from_pretrained(str(model_dir), trust_remote_code=True)
    model = None
    if needs_model:
        print(f"  loading ARK-ASR from {model_dir} (dtype={dtype_name})")
        model = AutoModelForCausalLM.from_pretrained(
            str(model_dir), trust_remote_code=True, torch_dtype=dtype, device_map="cpu"
        )
        model.eval()
    else:
        print("  mel-only dump: skipping 3B model load (low footprint)")

    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    out: Dict[str, np.ndarray] = {}

    # ---- mel ----
    feat = processor.feature_extractor(
        [audio], sampling_rate=16000, return_tensors="pt",
        padding="longest", return_attention_mask=False,
    )
    input_features = feat["input_features"].to(torch.float32)  # (1, 128, T)
    if "mel_spectrogram" in stages:
        out["mel_spectrogram"] = input_features[0].detach().cpu().numpy()  # (128, T)

    # ---- encoder + adapter (audio_embeds) ----
    if "audio_embeds" in stages:
        with torch.no_grad():
            emb = model.audio_encoder(input_features.to(dtype))  # (1, N, hidden)
        out["audio_embeds"] = emb[0].detach().cpu().float().numpy()  # (N, hidden)

    # ---- prompt build + first-token logits ----
    if {"first_logits", "llm_input_ids"} & stages:
        batch = processor.apply_chat_template(
            [{"role": "user", "content": [{"type": "audio", "array": audio}]}],
            add_generation_prompt=True, tokenize=True, return_tensors="pt",
        )
        input_ids = batch["input_ids"]
        if "llm_input_ids" in stages:
            out["llm_input_ids"] = input_ids[0].detach().cpu().numpy().astype(np.float32)
        with torch.no_grad():
            mout = model(
                input_ids=input_ids,
                audios=batch.get("audios"),
                attention_mask=batch.get("attention_mask"),
                use_cache=False,
            )
        last = mout.logits[0, -1].detach().cpu().float().numpy()  # (vocab,)
        if "first_logits" in stages:
            out["first_logits"] = last

    # ---- end-to-end greedy transcript ----
    if "generated_text" in stages:
        batch = processor.apply_chat_template(
            [{"role": "user", "content": [{"type": "audio", "array": audio}]}],
            add_generation_prompt=True, tokenize=True, return_tensors="pt",
        )
        with torch.no_grad():
            gen = model.generate(
                input_ids=batch["input_ids"],
                audios=batch.get("audios"),
                attention_mask=batch.get("attention_mask"),
                max_new_tokens=max_new_tokens or 256,
                do_sample=False,
            )
        new_ids = gen[0, batch["input_ids"].shape[1]:]
        text = processor.tokenizer.decode(new_ids, skip_special_tokens=True)
        out["generated_text"] = text
        print(f"  reference transcript: {text!r}")

    return out
