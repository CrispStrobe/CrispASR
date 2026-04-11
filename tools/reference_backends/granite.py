"""Granite Speech 4.0-1B reference dump backend.

Port of `models/granite-speech-kaggle-groundtruth.py` into the modular
tools/dump_reference.py interface. That legacy script was a one-off
Kaggle notebook that auto-downloaded the model, dumped stats to a gist,
and wrote scattered .npy files — this module keeps the relevant
forward-hook logic and drops the Kaggle-specific plumbing (HF_TOKEN
secret, gist upload, Kaggle-style `install()` helper).

Captures:

  mel_spectrogram            input_features from the processor
                             (stacked 2-frame, shape (160, T/2))
  enc_after_input_linear     after encoder.input_linear
  enc_after_layer1,4,8,mid,last
                             per-layer Conformer encoder outputs
                             (a handful of checkpoints, not every layer)
  encoder_out                final encoder hidden state
  projector_out              BLIP-2 Q-Former output (fixed-length
                             query tokens, ready for the LLM)
  llm_argmax                 greedy token IDs from model.generate()
  generated_text             decoded transcript

The model uses µP multipliers internally (`embedding_multiplier=12.0`,
`attention_multiplier=1/128`, `residual_multiplier=0.22`,
`logits_scaling=8.0`); those are baked into the PyTorch reference's
forward pass, so the captured activations already reflect them.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Set

import numpy as np

DEFAULT_STAGES = [
    "raw_audio",
    "mel_spectrogram",
    "enc_after_input_linear",
    "enc_after_layer1",
    "enc_after_layer4",
    "enc_after_layer8",
    "enc_after_layer_mid",
    "enc_after_layer_last",
    "encoder_out",
    "projector_out",
    "llm_argmax",
    "generated_text",
]


def dump(*, model_dir: Path, audio: np.ndarray, stages: Set[str],
         max_new_tokens: int) -> Dict[str, np.ndarray]:
    """Run Granite Speech 1B reference forward and return stage captures."""
    import torch
    try:
        from transformers import GraniteSpeechForConditionalGeneration, AutoProcessor
    except ImportError as e:
        raise SystemExit(
            "transformers with GraniteSpeech support required.\n"
            "Install: pip install 'transformers>=4.52.1'\n"
            f"(import error: {e})")

    print(f"  loading Granite Speech 1B from {model_dir}")
    processor = AutoProcessor.from_pretrained(str(model_dir))
    model = GraniteSpeechForConditionalGeneration.from_pretrained(
        str(model_dir), torch_dtype=torch.float32, device_map="cpu",
    ).eval()

    # Granite expects the audio as a torch tensor shape (1, N) via torchaudio,
    # but a float32 numpy array of shape (N,) works with .unsqueeze(0) too.
    wav = torch.from_numpy(audio.astype(np.float32)).unsqueeze(0)

    tokenizer = processor.tokenizer
    chat = [{"role": "user",
             "content": "<|audio|>can you transcribe the speech into a written format?"}]
    prompt = tokenizer.apply_chat_template(chat, tokenize=False,
                                            add_generation_prompt=True)
    inputs = processor(prompt, wav, device="cpu", return_tensors="pt")

    out: Dict[str, np.ndarray] = {}
    if "mel_spectrogram" in stages and "input_features" in inputs:
        feats = inputs["input_features"][0]   # drop batch dim
        out["mel_spectrogram"] = feats.detach().cpu().float().numpy()

    encoder = model.encoder

    # ---- Hook encoder layers + input_linear for per-stage captures ----
    layer_outputs: Dict[int, torch.Tensor] = {}

    def make_hook(idx: int):
        def hook(_mod, _inp, output):
            t = output[0] if isinstance(output, tuple) else output
            layer_outputs[idx] = t.detach().clone()
        return hook

    handles = []
    for i, layer in enumerate(encoder.layers):
        handles.append(layer.register_forward_hook(make_hook(i)))

    input_linear_out: Dict[str, torch.Tensor] = {}
    def il_hook(_m, _i, output):
        t = output[0] if isinstance(output, tuple) else output
        input_linear_out["v"] = t.detach().clone()
    handles.append(encoder.input_linear.register_forward_hook(il_hook))

    enc_hidden: torch.Tensor | None = None
    with torch.no_grad():
        feats = inputs.get("input_features", inputs.get("input_values"))
        if feats is not None:
            enc_out = encoder(feats)
            if hasattr(enc_out, "last_hidden_state"):
                enc_hidden = enc_out.last_hidden_state
            elif isinstance(enc_out, tuple):
                enc_hidden = enc_out[0]
            else:
                enc_hidden = enc_out

    for h in handles:
        h.remove()

    if "enc_after_input_linear" in stages and "v" in input_linear_out:
        out["enc_after_input_linear"] = (
            input_linear_out["v"][0].detach().cpu().float().numpy())

    if enc_hidden is not None:
        num_layers = len(encoder.layers)
        checkpoints = {
            "enc_after_layer1":      1,
            "enc_after_layer4":      4,
            "enc_after_layer8":      8,
            "enc_after_layer_mid":   max(1, num_layers // 2),
            "enc_after_layer_last":  num_layers,
        }
        for name, layer_num in checkpoints.items():
            if name not in stages: continue
            idx = layer_num - 1
            if idx in layer_outputs:
                out[name] = (
                    layer_outputs[idx][0].detach().cpu().float().numpy())

        if "encoder_out" in stages:
            out["encoder_out"] = enc_hidden[0].detach().cpu().float().numpy()

        # ---- Projector (BLIP-2 Q-Former) ----
        if "projector_out" in stages:
            projector = model.projector
            try:
                mask = inputs.get("input_features_mask")
                proj = projector(enc_hidden, encoder_attention_mask=mask)
            except TypeError:
                try:
                    proj = projector(enc_hidden, inputs.get("input_features_mask"))
                except TypeError:
                    proj = projector(enc_hidden)
            if hasattr(proj, "last_hidden_state"):
                proj_t = proj.last_hidden_state
            elif isinstance(proj, tuple):
                proj_t = proj[0]
            else:
                proj_t = proj
            out["projector_out"] = proj_t[0].detach().cpu().float().numpy()

    # ---- Generation ----
    want_argmax = "llm_argmax" in stages
    want_text   = "generated_text" in stages
    if want_argmax or want_text:
        with torch.no_grad():
            gen = model.generate(
                **inputs, max_new_tokens=max_new_tokens,
                do_sample=False, num_beams=1,
            )
        if want_argmax:
            out["llm_argmax"] = gen[0].detach().cpu().int().numpy().astype(np.int32)
        if want_text:
            n_in = inputs["input_ids"].shape[-1]
            new_tokens = gen[0, n_in:].unsqueeze(0)
            decoded = tokenizer.batch_decode(
                new_tokens, add_special_tokens=False, skip_special_tokens=True)
            out["generated_text"] = decoded[0] if decoded else ""

    return out
