"""Google Gemma-4-E2B reference dump backend.

Loads `google/gemma-4-E2B-it` via HuggingFace Transformers and captures
mel features and encoder output for crispasr-diff comparison against the
C++ runtime.

Memory note: the full Gemma4ForConditionalGeneration model is ~9.6 GB
(F32) / ~5 GB (bf16). On 8 GB RAM machines, loading it triggers heavy
swap thrashing. We default to AUDIO-ONLY mode that loads just the
audio_tower submodule (~1 GB in bf16) and the feature extractor — that
fits comfortably and produces the activations the audio-side diff
actually needs.

Set CRISPASR_REF_FULL=1 to load the full model and capture LLM hidden
states as well (only useful on machines with ≥16 GB RAM or after the
diff harness verifies the encoder is correct).

Stages:

  raw_audio                 (N,)            input PCM
  mel_spectrogram           (n_mels, T_mel) Gemma4AudioFeatureExtractor output
  encoder_output            (T_enc, d_llm)  audio_tower → embed_audio output
                                            (i.e. post audio_embed_proj.norm + linear)

Audio-only mode skips the LLM hidden states. Use the full mode flag
when you need them.

Usage:

  python tools/dump_reference.py --backend gemma4 \\
      --model-dir google/gemma-4-E2B-it \\
      --audio samples/jfk.wav \\
      --output /tmp/gemma4-ref.gguf
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Set

import numpy as np

DEFAULT_STAGES = [
    "raw_audio",
    "mel_spectrogram",
    "audio_subsample_output",       # post-conv2d subsample (pre-conformer)
    "audio_layer_0",                # post-conformer-layer-0 (pre-output_proj)
    "audio_layer_1",
    "audio_layer_5",
    "audio_layer_11",               # last conformer layer output
    "audio_tower_output",           # post-output_proj (pre-adapter)
    "encoder_output",               # post-adapter (final, fed to LLM)
    "llm_hidden_layer_0",
    "llm_hidden_layer_1",
    "llm_hidden_layer_8",
    "llm_hidden_layer_mid",
    "llm_hidden_layer_last",
    "llm_logits",
    "llm_argmax",
    "generated_text",
]


def _dump_audio_only(model_dir: Path, audio: np.ndarray, stages: Set[str]) -> Dict[str, np.ndarray]:
    """Audio-tower-only dump — fits in <2 GB RAM."""
    import os, torch
    from transformers import AutoProcessor, Gemma4AudioModel
    from transformers.models.gemma4.modeling_gemma4 import Gemma4MultimodalEmbedder
    from transformers.models.gemma4.configuration_gemma4 import Gemma4AudioConfig, Gemma4TextConfig

    pretrained = str(model_dir)
    print(f"  loading audio_tower only from {pretrained}")
    processor = AutoProcessor.from_pretrained(pretrained, trust_remote_code=True)

    # Build the audio config (and the text config we need for the embed_audio
    # adapter's text_hidden_size). These are small — just metadata.
    from transformers import AutoConfig
    full_cfg = AutoConfig.from_pretrained(pretrained, trust_remote_code=True)
    audio_cfg = full_cfg.audio_config
    text_cfg = full_cfg.text_config

    # Allocate the audio_tower + the audio→LLM adapter only.
    dtype = torch.bfloat16 if os.environ.get("CRISPASR_REF_DTYPE", "bf16") in ("bf16", "bfloat16") else torch.float32
    audio_tower = Gemma4AudioModel(audio_cfg)
    embed_audio = Gemma4MultimodalEmbedder(audio_cfg, text_cfg)

    # Load only the audio_tower / embed_audio shards. safetensors are
    # already split per-tensor name, so we filter on prefix.
    from safetensors import safe_open
    import json
    snap_dir = None
    for p in Path(pretrained).rglob("config.json"):
        snap_dir = p.parent
        break
    if snap_dir is None:
        # processor.save_pretrained may have downloaded to HF cache; resolve via HF
        from huggingface_hub import snapshot_download
        snap = snapshot_download(pretrained, allow_patterns=["*.safetensors", "config.json"])
        snap_dir = Path(snap)

    safetensor_files = sorted(snap_dir.glob("*.safetensors"))
    print(f"  loading audio weights from {len(safetensor_files)} safetensor file(s) at {snap_dir}")

    audio_state, embed_audio_state = {}, {}
    for f in safetensor_files:
        with safe_open(str(f), framework="pt") as h:
            for k in h.keys():
                if k.startswith("model.audio_tower."):
                    short = k.replace("model.audio_tower.", "")
                    audio_state[short] = h.get_tensor(k).to(dtype)
                elif k.startswith("model.embed_audio."):
                    short = k.replace("model.embed_audio.", "")
                    embed_audio_state[short] = h.get_tensor(k).to(dtype)

    print(f"  audio_tower tensors: {len(audio_state)}, embed_audio tensors: {len(embed_audio_state)}")
    # Some tensor keys in HF state_dict have a `.linear.weight` indirection
    # for Gemma4ClippableLinear; the modeling code maps to plain `.weight`
    # via _post_load. Just load with strict=False and warn on missing.
    missing, unexpected = audio_tower.load_state_dict(audio_state, strict=False)
    if missing:
        print(f"  audio_tower missing keys: {len(missing)} (first 3: {missing[:3]})")
    if unexpected:
        print(f"  audio_tower unexpected keys: {len(unexpected)} (first 3: {unexpected[:3]})")
    missing, unexpected = embed_audio.load_state_dict(embed_audio_state, strict=False)
    if missing:
        print(f"  embed_audio missing: {len(missing)}")
    audio_tower = audio_tower.to(dtype).eval()
    embed_audio = embed_audio.to(dtype).eval()

    out: Dict[str, np.ndarray] = {}
    if "raw_audio" in stages:
        out["raw_audio"] = audio.astype(np.float32)

    # --- Mel features via the processor's feature extractor ---
    fe = processor.feature_extractor
    # Pass as a list of arrays: HF FE's `is_batched` path skips the extra
    # wrapping that confuses the spectrogram squeeze on a single 1-D input.
    feat_inputs = fe([audio.astype(np.float32)], sampling_rate=fe.sampling_rate, return_tensors="pt")
    input_features = feat_inputs["input_features"]
    input_features_mask = feat_inputs.get("input_features_mask",
                                          feat_inputs.get("attention_mask"))
    if input_features_mask is None:
        input_features_mask = torch.ones(input_features.shape[:2], dtype=torch.bool)

    if "mel_spectrogram" in stages:
        # HF FE returns (seq_len, n_mels) per item. We store it AS-IS
        # (frame-major, mel-fast) so the diff harness's per-row cos
        # comparison aligns one row per frame (row_w = n_mels = 128).
        m = input_features[0].detach().cpu().float().numpy()
        if m.ndim == 2 and m.shape[1] != 128 and m.shape[0] == 128:
            m = m.T  # only flip if FE returned (n_mels, T_mel)
        out["mel_spectrogram"] = np.ascontiguousarray(m)

    # --- Encoder forward with per-stage hooks ---
    capture: Dict[str, "torch.Tensor"] = {}
    handles = []

    def make_hook(name):
        def hook(_m, _i, output):
            t = output[0] if isinstance(output, tuple) else output
            if hasattr(t, "last_hidden_state"):
                t = t.last_hidden_state
            capture[name] = t.detach().clone()
        return hook

    # Hook the subsample (full block) and selected conformer layers.
    if hasattr(audio_tower, "subsample_conv_projection"):
        handles.append(audio_tower.subsample_conv_projection.register_forward_hook(
            make_hook("audio_subsample_output")))
    if hasattr(audio_tower, "layers"):
        n_audio_layers = len(audio_tower.layers)
        for idx, label in [(0, "audio_layer_0"), (1, "audio_layer_1"),
                           (5, "audio_layer_5"), (n_audio_layers - 1, "audio_layer_11")]:
            if 0 <= idx < n_audio_layers:
                handles.append(audio_tower.layers[idx].register_forward_hook(
                    make_hook(label)))

    with torch.no_grad():
        audio_features = input_features.to(dtype)
        if input_features_mask.dtype != torch.bool:
            input_features_mask = input_features_mask.to(torch.bool)
        try:
            audio_out = audio_tower(input_features=audio_features, attention_mask=input_features_mask)
        except TypeError:
            audio_out = audio_tower(audio_features, input_features_mask)
        last_hidden = audio_out.last_hidden_state if hasattr(audio_out, "last_hidden_state") else audio_out[0]
        capture["audio_tower_output"] = last_hidden.detach().clone()
        adapted = embed_audio(inputs_embeds=last_hidden)
        capture["encoder_output"] = adapted.detach().clone()

    for h in handles:
        h.remove()

    for name, tens in capture.items():
        if name in stages:
            t = tens
            if t.dim() >= 3 and t.shape[0] == 1:
                t = t[0]
            elif t.dim() == 4 and t.shape[0] == 1:
                t = t[0]  # subsample output: (B, C, H, W) → (C, H, W) — leave as 3D
            arr = t.cpu().float().numpy()
            out[name] = np.ascontiguousarray(arr)

    return out


def _dump_full(model_dir: Path, audio: np.ndarray, stages: Set[str], max_new_tokens: int) -> Dict[str, np.ndarray]:
    """Full-model dump — needs ≥10 GB RAM for bf16."""
    import os, torch
    from transformers import AutoProcessor, AutoModelForImageTextToText

    pretrained = str(model_dir)
    dtype = torch.bfloat16 if os.environ.get("CRISPASR_REF_DTYPE", "bf16") in ("bf16", "bfloat16") else torch.float32
    print(f"  loading full Gemma-4-E2B (dtype={dtype}) from {pretrained}")
    processor = AutoProcessor.from_pretrained(pretrained, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        pretrained, torch_dtype=dtype, trust_remote_code=True,
    ).eval()
    dev = next(model.parameters()).device

    chat = [{"role": "user", "content": [
        {"type": "audio", "audio": audio.astype(np.float32)},
        {"type": "text", "text": "Transcribe this audio."},
    ]}]
    inputs = processor.apply_chat_template(
        chat, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt",
    ).to(dev)

    out: Dict[str, np.ndarray] = {}
    if "raw_audio" in stages:
        out["raw_audio"] = audio.astype(np.float32)
    if "mel_spectrogram" in stages and "input_features" in inputs:
        m = inputs["input_features"][0].detach().cpu().float().numpy()
        if m.ndim == 2 and m.shape[0] != getattr(model.config.audio_config, "feature_size", 128):
            m = m.T
        out["mel_spectrogram"] = np.ascontiguousarray(m)

    enc_out = {}
    def enc_hook(_m, _i, output):
        t = output[0] if isinstance(output, tuple) else output
        enc_out["v"] = t.detach().clone()
    enc_handle = model.model.audio_tower.register_forward_hook(enc_hook) if hasattr(model.model, "audio_tower") else None

    layer_outputs: Dict[int, torch.Tensor] = {}
    handles = []
    layers = getattr(getattr(getattr(model, "model", model), "language_model", None), "layers", []) \
             or getattr(getattr(model, "model", model), "layers", [])
    for i, layer in enumerate(layers):
        def make(idx):
            def h(_m, _i, output):
                t = output[0] if isinstance(output, tuple) else output
                layer_outputs[idx] = t.detach().clone()
            return h
        handles.append(layer.register_forward_hook(make(i)))

    with torch.no_grad():
        gen = model.generate(
            **inputs, max_new_tokens=max(1, max_new_tokens),
            do_sample=False, num_beams=1, output_hidden_states=False,
            return_dict_in_generate=True,
        )

    if enc_handle is not None:
        enc_handle.remove()
    for h in handles:
        h.remove()

    if "encoder_output" in stages and "v" in enc_out:
        e = enc_out["v"]
        if e.dim() == 3 and e.shape[0] == 1:
            e = e[0]
        out["encoder_output"] = e.detach().cpu().float().numpy()

    if layer_outputs:
        n_layers = len(layers)
        checkpoints = {
            "llm_hidden_layer_0":     0,
            "llm_hidden_layer_1":     1,
            "llm_hidden_layer_8":     min(8, n_layers - 1),
            "llm_hidden_layer_mid":   max(0, n_layers // 2 - 1),
            "llm_hidden_layer_last":  n_layers - 1,
        }
        for name, idx in checkpoints.items():
            if name in stages and idx in layer_outputs:
                out[name] = layer_outputs[idx][0].detach().cpu().float().numpy()

    if "generated_text" in stages or "llm_argmax" in stages:
        seq = gen.sequences if hasattr(gen, "sequences") else gen
        n_in = inputs["input_ids"].shape[-1]
        new_ids = seq[0, n_in:]
        if "llm_argmax" in stages:
            out["llm_argmax"] = new_ids.detach().cpu().int().numpy().astype(np.int32)
        if "generated_text" in stages:
            text = processor.tokenizer.decode(new_ids, skip_special_tokens=True)
            out["generated_text"] = np.array([ord(c) for c in text], dtype=np.int32)

    return out


def dump(*, model_dir: Path, audio: np.ndarray, stages: Set[str],
         max_new_tokens: int) -> Dict[str, np.ndarray]:
    """Run Gemma-4-E2B reference forward and return stage captures.

    Defaults to audio-only (fits on 8 GB RAM). Set CRISPASR_REF_FULL=1
    to load the full model and capture LLM hidden states.
    """
    import os
    if os.environ.get("CRISPASR_REF_FULL", "0") == "1":
        return _dump_full(model_dir, audio, stages, max_new_tokens)
    return _dump_audio_only(model_dir, audio, stages)
