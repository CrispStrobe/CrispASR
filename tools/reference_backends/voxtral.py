"""Voxtral-Mini-3B-2507 reference dump backend.

Mechanical port of `models/voxtral-encoder-dump.py` into the modular
interface. Captures the audio encoder + 4-frame-stack projector path
via forward hooks on the HF `VoxtralForConditionalGeneration` model.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Set

import numpy as np

DEFAULT_STAGES = [
    "raw_audio",
    "mel_spectrogram",
    "conv1_out",
    "conv2_out",
    "pos_embed",
    "enc_blk00_out",
    "enc_blk31_out",  # last (32-layer encoder)
    "layer_norm_out",
    "proj1_out",
    "proj2_out",
    "encoder_output",
    # Text-only LLM stages for voxtral-test-llm. When these are requested
    # the backend builds a fixed text prompt via apply_chat_template() (no
    # audio), runs one forward pass through language_model + lm_head, and
    # emits per-token logits in ne-order [vocab, T] plus the token ids so
    # the C++ driver can feed them into voxtral_run_llm and diff.
    "llm_input_ids",
    "llm_logits",
]

# Fixed text-only prompt used for the LLM ground truth. Matches the legacy
# models/voxtral-llm-dump.py so any hash-based verification cross-checks.
_LLM_TEXT_PROMPT = "Why should AI models be open-sourced?"


def dump(*, model_dir: Path, audio: np.ndarray, stages: Set[str],
         max_new_tokens: int) -> Dict[str, np.ndarray]:
    """Run Voxtral 3B reference forward and return stage captures."""
    import torch
    try:
        from transformers import VoxtralForConditionalGeneration, AutoProcessor
    except ImportError as e:
        raise SystemExit(
            "transformers with Voxtral support required.\n"
            f"Install: pip install 'transformers>=4.50'\n"
            f"(import error: {e})")

    print(f"  loading Voxtral 3B from {model_dir}")
    model = VoxtralForConditionalGeneration.from_pretrained(
        str(model_dir), torch_dtype=torch.float32, device_map="cpu",
    ).eval()
    processor = AutoProcessor.from_pretrained(str(model_dir))

    # ---- Mel via WhisperFeatureExtractor ----
    feat = processor.feature_extractor(
        audio, sampling_rate=16000, return_tensors="pt",
        padding="max_length", truncation=True,
    )
    mel = feat["input_features"]  # (1, 128, 3000) — Voxtral pads to 30s
    out: Dict[str, np.ndarray] = {}
    if "mel_spectrogram" in stages:
        out["mel_spectrogram"] = mel.detach().cpu().float().numpy()

    # ---- Resolve the audio tower and projector ----
    # HF Voxtral layout: model.audio_tower + model.multi_modal_projector
    audio_tower = model.audio_tower
    projector = model.multi_modal_projector

    captures: Dict[str, np.ndarray] = {}

    def cap(name: str):
        def hook(_mod, _inp, output):
            t = output[0] if isinstance(output, tuple) else output
            captures[name] = t.detach().cpu().float().numpy()
        return hook

    handles = []
    hook_map = {
        "conv1_out":      (audio_tower.conv1, "conv1"),
        "conv2_out":      (audio_tower.conv2, "conv2"),
        "enc_blk00_out":  (audio_tower.layers[0], "layers[0]"),
        "enc_blk31_out":  (audio_tower.layers[-1], f"layers[{len(audio_tower.layers)-1}]"),
        "layer_norm_out": (audio_tower.layer_norm, "layer_norm"),
        "proj1_out":      (projector.linear_1, "projector.linear_1"),
        "proj2_out":      (projector.linear_2, "projector.linear_2"),
    }
    for stage_name, (mod, _) in hook_map.items():
        if stage_name in stages:
            handles.append(mod.register_forward_hook(cap(stage_name)))

    with torch.no_grad():
        enc_out = audio_tower(mel)
        # encoder output → projector (4-frame stack handled inside projector)
        # We explicitly run the projector too so proj1/proj2 hooks fire.
        enc_hidden = enc_out.last_hidden_state if hasattr(enc_out, "last_hidden_state") else enc_out
        stacked = enc_hidden.reshape(enc_hidden.shape[0],
                                      enc_hidden.shape[1] // 4,
                                      enc_hidden.shape[2] * 4)
        proj_out = projector(stacked)

    if "encoder_output" in stages:
        out["encoder_output"] = enc_hidden.detach().cpu().float().numpy()
    if "projector_output" in stages:
        out["projector_output"] = proj_out.detach().cpu().float().numpy()

    for h in handles:
        h.remove()
    out.update(captures)

    # ---- Text-only LLM forward for voxtral-test-llm ----
    want_llm = bool(stages & {"llm_input_ids", "llm_logits"})
    if want_llm:
        print("  running text-only LLM forward (no audio)")
        text_model = model.language_model
        # HF layout: model.language_model can be either the stack directly
        # or have a .model inner + .lm_head outer. Normalise.
        text_inner = text_model.model if hasattr(text_model, "model") else text_model
        lm_head = (text_model.lm_head if hasattr(text_model, "lm_head")
                   else model.lm_head)

        # apply_chat_template wraps the user text in Voxtral's Tekken
        # control tokens the model was trained with. This matches what the
        # Python inference path does for plain-text prompts and is the same
        # template used by the deleted models/voxtral-llm-dump.py script.
        conversation = [{
            "role": "user",
            "content": [{"type": "text", "text": _LLM_TEXT_PROMPT}],
        }]
        chat_inputs = processor.apply_chat_template(conversation)
        input_ids_t = chat_inputs.input_ids  # (1, T)
        input_ids_np = input_ids_t[0].detach().cpu().numpy().astype(np.int32)

        if "llm_input_ids" in stages:
            # Stored as F32 because core_gguf loads floats only; the C++
            # driver casts back to int32 on read.
            out["llm_input_ids"] = input_ids_np.astype(np.float32)

        if "llm_logits" in stages:
            with torch.no_grad():
                lm_out = text_inner(input_ids=input_ids_t, use_cache=False)
                hidden = lm_out.last_hidden_state  # (1, T, D)
                logits = lm_head(hidden)            # (1, T, V)
            # ne-order: [vocab, T] — matches the qwen3 archive layout and
            # what the C++ voxtral-test-llm driver already parses.
            full = logits[0].detach().cpu().float().numpy()  # (T, V)
            out["llm_logits"] = full.T.copy()                 # (V, T)

    return out
