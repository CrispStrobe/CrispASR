"""Cohere Transcribe (CohereLabs/cohere-transcribe-03-2026) reference backend.

Cohere's ASR model ships its processor + modeling code inside the HF repo
itself and is loaded via transformers with `trust_remote_code=True`. It
is a classic encoder–decoder architecture:

  * 48-layer Conformer encoder (d_model=1280, 8 heads)
    – conv subsampling (5×, total 8× temporal downsampling)
    – macaron FFN + rel-pos MHSA + Conformer conv + macaron FFN
  * Linear encoder→decoder projection (1280 → 1024)
  * 8-layer Transformer decoder (hidden=1024, 8 heads, 16384 vocab)
  * 9-token prompt to prime the decoder KV cache before autoregressive decoding

This reference module is patterned after voxtral.py / granite.py. It uses
forward hooks to capture encoder activations and runs `model.generate()`
for the argmax path. Field names below (`.encoder` / `.audio_tower` /
`.model.encoder`) are resolved dynamically since the exact submodule
layout depends on the version of the remote code that ships with the
model snapshot.

The model is gated — log in with `huggingface-cli login` and accept the
license at https://huggingface.co/CohereLabs/cohere-transcribe-03-2026
before running this backend.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Set

import numpy as np

DEFAULT_STAGES = [
    "raw_audio",
    "mel_spectrogram",
    "enc_pre_subsample_out",
    "enc_blk00_out",
    "enc_blk_mid_out",
    "enc_blk_last_out",
    "encoder_output",      # final encoder hidden state (post enc→dec proj)
    "llm_argmax",          # greedy decoded token IDs from model.generate()
    "generated_text",      # decoded transcript (special tokens stripped)
]


def _resolve_encoder(model):
    """Return the Conformer encoder submodule.

    Tries several attribute paths that appear in different revisions of
    CohereLabs/cohere-transcribe-03-2026's remote modeling code. The
    first one that resolves to an nn.Module with a `.layers` attribute
    (the 48 ConformerLayers) wins.
    """
    candidates = [
        ("encoder",),
        ("audio_tower",),
        ("model", "encoder"),
        ("model", "audio_tower"),
        ("asr_model", "encoder"),
    ]
    for path in candidates:
        cur = model
        ok = True
        for p in path:
            if not hasattr(cur, p):
                ok = False
                break
            cur = getattr(cur, p)
        if ok and hasattr(cur, "layers"):
            return cur, ".".join(path)
    raise RuntimeError(
        "cohere reference: could not resolve the encoder submodule. "
        "Tried: encoder, audio_tower, model.encoder, model.audio_tower, "
        "asr_model.encoder. Inspect `print(model)` and add the real "
        "path to _resolve_encoder().")


def dump(*, model_dir: Path, audio: np.ndarray, stages: Set[str],
         max_new_tokens: int) -> Dict[str, np.ndarray]:
    """Run Cohere Transcribe reference forward and return stage captures."""
    import torch
    try:
        from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
    except ImportError as e:
        raise SystemExit(
            "transformers required for the cohere reference backend.\n"
            "Install: pip install 'transformers>=4.45'\n"
            f"(import error: {e})")

    print(f"  loading Cohere Transcribe from {model_dir}")
    processor = AutoProcessor.from_pretrained(
        str(model_dir), trust_remote_code=True)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        str(model_dir), trust_remote_code=True,
        torch_dtype=torch.float32, device_map="cpu",
        low_cpu_mem_usage=True,
    ).eval()

    # ---- Feature extraction (processor handles pre-emphasis + mel + norm) ----
    # CohereProcessor mirrors WhisperFeatureExtractor's API.
    inputs = processor(
        audio, sampling_rate=16000, return_tensors="pt", language="en")
    # Cast to the same dtype as the model to match the rust reference's
    # bf16→f32 path (we load in f32 here so this is a no-op).
    if "input_features" in inputs:
        inputs["input_features"] = inputs["input_features"].to(torch.float32)

    out: Dict[str, np.ndarray] = {}
    if "mel_spectrogram" in stages and "input_features" in inputs:
        mel = inputs["input_features"][0]           # drop batch → (n_mels, T)
        # Transpose to (T, n_mels) matching C++ Layout::TimeMels convention.
        out["mel_spectrogram"] = mel.transpose(0, 1).contiguous().detach().cpu().float().numpy()

    # ---- Resolve the encoder and register hooks ----
    encoder, enc_path = _resolve_encoder(model)
    n_layers = len(encoder.layers)
    print(f"  resolved encoder at model.{enc_path} with {n_layers} layers")

    captures: Dict[str, torch.Tensor] = {}

    def cap(name: str):
        def hook(_mod, _inp, output):
            t = output[0] if isinstance(output, tuple) else output
            captures[name] = t.detach().clone()
        return hook

    handles = []
    # Per-layer hooks (only the three we expose in DEFAULT_STAGES).
    layer_hooks = {
        "enc_blk00_out":    0,
        "enc_blk_mid_out":  max(1, n_layers // 2),
        "enc_blk_last_out": n_layers - 1,
    }
    for stage_name, idx in layer_hooks.items():
        if stage_name in stages:
            handles.append(
                encoder.layers[idx].register_forward_hook(cap(stage_name)))

    # The conv subsampling module lives at encoder.pre_encode in the
    # rust reference; accept any of the common attribute names.
    pre_encode = None
    for name in ("pre_encode", "conv_subsampling", "subsample", "conv_subsample"):
        if hasattr(encoder, name):
            pre_encode = getattr(encoder, name)
            break
    if pre_encode is not None and "enc_pre_subsample_out" in stages:
        handles.append(
            pre_encode.register_forward_hook(cap("enc_pre_subsample_out")))

    # ---- Encoder forward ----
    enc_hidden = None
    with torch.no_grad():
        feats = inputs.get("input_features", inputs.get("input_values"))
        if feats is None:
            raise RuntimeError("cohere reference: processor produced no features")
        enc_out = encoder(feats)
        enc_hidden = (
            enc_out.last_hidden_state if hasattr(enc_out, "last_hidden_state")
            else (enc_out[0] if isinstance(enc_out, tuple) else enc_out))

    for h in handles:
        h.remove()

    for k, v in captures.items():
        out[k] = v[0].detach().cpu().float().numpy()
    if "encoder_output" in stages and enc_hidden is not None:
        out["encoder_output"] = enc_hidden[0].detach().cpu().float().numpy()

    # ---- Generation (greedy) ----
    want_argmax = "llm_argmax" in stages
    want_text   = "generated_text" in stages
    if want_argmax or want_text:
        print("  running greedy generate() for argmax capture")
        with torch.no_grad():
            gen = model.generate(
                **inputs, max_new_tokens=max_new_tokens,
                do_sample=False, num_beams=1,
            )
        gen_ids = gen[0].detach().cpu().int().numpy().astype(np.int32)
        if want_argmax:
            out["llm_argmax"] = gen_ids
        if want_text:
            # CohereAsrProcessor ships its own `.decode()` wrapping the
            # custom TokenizersBackend — going through it (instead of
            # `tokenizer.decode`) is what test_cohere_9.py does and is
            # the only path that handles the 9-token decoder prompt and
            # SentencePiece ▁-boundary detokenization correctly.
            try:
                decoded = processor.decode(
                    gen[0], skip_special_tokens=True)
            except Exception:
                decoded = processor.batch_decode(
                    gen, skip_special_tokens=True)[0]
            out["generated_text"] = decoded

    return out
