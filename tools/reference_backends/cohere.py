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

import os
from pathlib import Path
from typing import Dict, Set

import numpy as np

DEFAULT_STAGES = [
    "raw_audio",
    "mel_spectrogram",
    "enc_pre_subsample_out",
    "L0_ff1",              # layer 0: after FF1 + residual
    "L0_attn",             # layer 0: after self-attention + residual
    "L0_conv",             # layer 0: after conv + residual
    "L0_ff2",              # layer 0: after FF2 + residual (before out_norm)
] + [f"encoder_layer_{i}" for i in range(48)] + [
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

    # Workaround: from_pretrained with trust_remote_code + the Cohere
    # CohereAsrPreTrainedModel._init_weights re-initializes all Linear/Conv
    # weights to normal(0, 0.02), overwriting the pretrained values.  Detect
    # and patch by reloading ALL weight tensors from the safetensors file.
    # Find the first conv-like weight anywhere in the resolved encoder
    # (base model: encoder.pre_encode.conv[0]; Parakeet-style Arabic model:
    # encoder.subsampling.layers[0]). Structure varies, so search generically.
    _enc_probe, _ = _resolve_encoder(model)
    _conv0 = None
    for _m in _enc_probe.modules():
        _w = getattr(_m, "weight", None)
        if _w is not None and hasattr(_w, "dim") and _w.dim() >= 3:
            _conv0 = _w
            break
    _conv0_rms = float(_conv0.norm() / _conv0.numel() ** 0.5) if _conv0 is not None else None
    if _conv0_rms is not None and _conv0_rms < 0.1:
        print(f"  WARNING: weights appear random-init (rms={_conv0_rms:.4f}), patching ALL from safetensors")
        from safetensors import safe_open
        sf_path = Path(model_dir) / "model.safetensors"
        model_sd = dict(model.named_parameters())
        with safe_open(str(sf_path), framework="pt") as sf:
            for key in sf.keys():
                if key in model_sd:
                    model_sd[key].data.copy_(sf.get_tensor(key).float())

    # Workaround #2: some checkpoints (e.g. cohere-transcribe-arabic-07-2026)
    # ship ~10 encoder LayerNorm weights ZEROED in model.safetensors
    # (encoder.layers.N.norm_self_att / norm_conv / norm_feed_forward1, ...).
    # Left as zeros they null those Conformer sub-layers -> the decoder emits a
    # 🎵/repetition loop instead of a transcript. Patch them from a known-good
    # GGUF given by CRISPASR_COHERE_NORM_PATCH_GGUF (the corrected converter's
    # output, whose norms are non-zero). Verified: patching only these 10 makes
    # both this Python reference and the C++ port transcribe correctly.
    import torch as _torch
    _patch_gguf = os.environ.get("CRISPASR_COHERE_NORM_PATCH_GGUF")
    if _patch_gguf and Path(_patch_gguf).exists():
        import torch.nn as _nn
        from gguf import GGUFReader as _GR, dequantize as _dq
        _gg = {t.name: t for t in _GR(_patch_gguf).tensors}
        _sfx = {"norm_self_att": "attn.norm", "norm_conv": "conv.norm",
                "norm_feed_forward1": "ff1.norm", "norm_feed_forward2": "ff2.norm",
                "norm_out": "out_norm"}
        _npatch = 0
        for _n, _mod in model.named_modules():
            if isinstance(_mod, _nn.LayerNorm) and float(_mod.weight.abs().max()) < 1e-6:
                _p = _n.split(".")
                _gn = (f"enc.blk.{_p[2]}.{_sfx[_p[3]]}.weight"
                       if len(_p) >= 4 and _p[2].isdigit() and _p[3] in _sfx else None)
                if _gn and _gn in _gg:
                    _t = _gg[_gn]
                    _v = np.asarray(_dq(_t.data, _t.tensor_type), dtype=np.float32)
                    _mod.weight.data.copy_(_torch.from_numpy(_v.copy()).reshape_as(_mod.weight))
                    _npatch += 1
        if _npatch:
            print(f"  patched {_npatch} zeroed encoder LayerNorm weights from "
                  f"{Path(_patch_gguf).name}")

    # ---- Feature extraction (processor handles pre-emphasis + mel + norm) ----
    # CohereProcessor mirrors WhisperFeatureExtractor's API. Language for the
    # decoder prompt is overridable (Arabic model needs "ar", not "en").
    inputs = processor(
        audio, sampling_rate=16000, return_tensors="pt",
        language=os.environ.get("CRISPASR_COHERE_REF_LANG", "en"))
    # Cast to the same dtype as the model to match the rust reference's
    # bf16→f32 path (we load in f32 here so this is a no-op).
    if "input_features" in inputs:
        inputs["input_features"] = inputs["input_features"].to(torch.float32)
    # Newer transformers' generate() extends every *attention_mask key it is
    # handed ("'NoneType' object has no attribute 'new_ones'", rebake
    # 2026-09-28): drop None-valued entries and give a decoder prompt an
    # explicit all-ones mask, which is what an absent one meant before.
    inputs = {k: v for k, v in dict(inputs).items() if v is not None}
    if "decoder_input_ids" in inputs and "decoder_attention_mask" not in inputs:
        inputs["decoder_attention_mask"] = torch.ones_like(inputs["decoder_input_ids"])

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
    # Per-layer hooks for full diagnosis
    for i in range(n_layers):
        stage_name = f"encoder_layer_{i}"
        if stage_name in stages:
            handles.append(
                encoder.layers[i].register_forward_hook(cap(stage_name)))
    # Legacy stage names (keep for backwards compat with old refs)
    legacy_hooks = {
        "enc_blk00_out":    0,
        "enc_blk_mid_out":  max(1, n_layers // 2),
        "enc_blk_last_out": n_layers - 1,
    }
    for stage_name, idx in legacy_hooks.items():
        if stage_name in stages:
            handles.append(
                encoder.layers[idx].register_forward_hook(cap(stage_name)))

    # The conv subsampling module lives at encoder.pre_encode in the
    # rust reference; accept any of the common attribute names.
    pre_encode = None
    for name in ("pre_encode", "subsampling", "conv_subsampling", "subsample", "conv_subsample"):
        if hasattr(encoder, name):
            pre_encode = getattr(encoder, name)
            break
    if pre_encode is not None and "enc_pre_subsample_out" in stages:
        handles.append(
            pre_encode.register_forward_hook(cap("enc_pre_subsample_out")))

    # ---- Layer-0 sub-stage monkey-patch ----
    # ParakeetEncoderBlock.forward has a different signature than the base
    # ConformerLayer, so the sub-stage monkey-patch below can't replicate it.
    # Skip it — the per-layer encoder_layer_i captures localize divergence.
    want_substages = set()  # was: stages & {"L0_ff1", "L0_attn", "L0_conv", "L0_ff2"}
    if want_substages:
        layer0 = encoder.layers[0]
        _orig_forward = layer0.forward

        def _patched_forward(x, pos_emb, mask=None, pad_mask=None):
            # Replicate ConformerLayer.forward with intermediate captures.
            residual = x
            x = layer0.norm_feed_forward1(x)
            x = residual + 0.5 * layer0.feed_forward1(x)
            if "L0_ff1" in stages:
                captures["L0_ff1"] = x.detach().clone()

            residual = x
            x = layer0.norm_self_att(x)
            x = residual + layer0.self_attn(x, pos_emb, mask)
            if "L0_attn" in stages:
                captures["L0_attn"] = x.detach().clone()

            residual = x
            x = layer0.norm_conv(x)
            x = residual + layer0.conv(x, pad_mask=pad_mask)
            if "L0_conv" in stages:
                captures["L0_conv"] = x.detach().clone()

            residual = x
            x = layer0.norm_feed_forward2(x)
            x = residual + 0.5 * layer0.feed_forward2(x)
            if "L0_ff2" in stages:
                captures["L0_ff2"] = x.detach().clone()

            return layer0.norm_out(x)

        layer0.forward = _patched_forward

    # ---- Encoder forward ----
    # Run the FULL model (encoder + 1 decode step) rather than calling
    # encoder(feats) directly: the encoder's forward signature varies by
    # model variant (ParakeetEncoder needs a length/mask), whereas
    # model.generate() drives the encoder through its own known call path.
    # The per-layer + encoder-output hooks capture during this forward.
    enc_hidden = None
    _enc_hidden_holder = {}

    def _enc_out_hook(_m, _i, o):
        t = (o.last_hidden_state if hasattr(o, "last_hidden_state")
             else (o[0] if isinstance(o, tuple) else o))
        _enc_hidden_holder["h"] = t.detach().clone()

    _eh = encoder.register_forward_hook(_enc_out_hook)
    with torch.no_grad():
        model.generate(**inputs, max_new_tokens=1, do_sample=False, num_beams=1)
    _eh.remove()
    enc_hidden = _enc_hidden_holder.get("h")

    for h in handles:
        h.remove()
    if want_substages:
        layer0.forward = _orig_forward

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
