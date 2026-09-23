"""Qwen3-ASR 0.6B reference dump backend.

Instruments the HuggingFace `Qwen3ASRForConditionalGeneration` forward
pass with per-layer hooks and emits a dict of captured activations
following the stage contract in `tools/dump_reference.py`.

Mechanical port of `models/qwen3-asr-reference-dump.py` into the new
modular interface. The only differences from the legacy script:

  1. dump() takes `stages: set[str]` and only captures the stages the
     user asked for (the legacy script dumped everything unconditionally).
  2. dump() returns an in-memory dict instead of writing .npy files.
     The parent dumper collects the dict and serializes it to a single
     GGUF tensor archive.
  3. No CLI argument parsing inside this file — that lives in the
     unified tools/dump_reference.py dispatcher.
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
    "conv3_out",
    "conv_out",
    "enc_blk00_out",
    "enc_blk17_out",
    "ln_post_out",
    "proj1_out",
    "proj2_out",
    "encoder_output",
    "llm_input_ids",
    "llm_logits",
    "llm_argmax",
    "generated_text",
    # Trace stages for the full-prompt end-to-end diff (qwen3-asr-test-trace).
    # These go through processor.apply_chat_template() to build a prompt that
    # matches what the wrapper uses internally, splice audio embeds in, and
    # run one forward pass so the C++ side can diff a bit-identical prompt.
    "trace_input_ids",
    "trace_audio_pad_pos",
    "trace_inputs_embeds",
    "trace_first_logits",
    "trace_generated_ids",
]

# Token id for <|audio_pad|> in Qwen3-ASR's tokenizer. Used to locate splice
# positions in the chat-template prompt without having to query the tokenizer.
_AUDIO_PAD_ID = 151676


def dump(*, model_dir: Path, audio: np.ndarray, stages: Set[str],
         max_new_tokens: int) -> Dict[str, np.ndarray]:
    """Run Qwen3-ASR reference forward pass, return captured stage tensors."""
    import torch
    try:
        from qwen_asr import Qwen3ASRModel
    except ImportError as e:
        raise SystemExit(
            "qwen_asr package not found. Install with: pip install qwen-asr\n"
            f"(original import error: {e})")

    print(f"  loading Qwen3-ASR model from {model_dir}")
    # bfloat16 is the model's native dtype — the reference forward runs in
    # bf16 (activations are cast to f32 only at capture time). Set
    # QWEN3_REF_DTYPE=float32 for an f32 reference when memory allows.
    import os
    dtype = os.environ.get("QWEN3_REF_DTYPE", "bfloat16")
    wrapper = Qwen3ASRModel.from_pretrained(
        str(model_dir), dtype=dtype, device_map="cpu",
    )
    processor = wrapper.processor
    model = wrapper.model
    model.eval()

    thinker = model.thinker
    audio_tower = thinker.audio_tower

    out: Dict[str, np.ndarray] = {}

    # ---- Mel spectrogram via WhisperFeatureExtractor ----
    # We go through the feature_extractor directly because processor.__call__
    # requires text input as well.
    feat = processor.feature_extractor(
        audio, sampling_rate=16000, return_tensors="pt",
        padding=True, truncation=False,
    )
    mel = feat["input_features"]  # (1, 128, T)
    if "mel_spectrogram" in stages:
        out["mel_spectrogram"] = mel.detach().cpu().float().numpy()

    # ---- Register forward hooks on the audio encoder ----
    captures: Dict[str, np.ndarray] = {}

    def cap(name: str):
        def hook(_mod, _inp, output):
            t = output[0] if isinstance(output, tuple) else output
            captures[name] = t.detach().cpu().float().numpy()
        return hook

    handles = []
    hook_map = {
        "conv1_out":     (audio_tower.conv2d1, "conv2d1"),
        "conv2_out":     (audio_tower.conv2d2, "conv2d2"),
        "conv3_out":     (audio_tower.conv2d3, "conv2d3"),
        "conv_out":      (audio_tower.conv_out, "conv_out"),
        "ln_post_out":   (audio_tower.ln_post, "ln_post"),
        "proj1_out":     (audio_tower.proj1, "proj1"),
        "proj2_out":     (audio_tower.proj2, "proj2"),
    }
    # Every encoder block under its REAL index. This used to capture only
    # layers[0] and layers[-1], naming the last one "enc_blk17_out" — right
    # for the 18-layer 0.6B, but for the 24-layer 1.7B the C++ side compared
    # its block 17 against the reference's block 23 (#445). Asking for any
    # enc_blk* stage (or the legacy enc_blk17_out) captures them all.
    want_blocks = any(st.startswith("enc_blk") for st in stages)
    if want_blocks:
        for i, layer in enumerate(audio_tower.layers):
            hook_map[f"enc_blk{i:02d}_out"] = (layer, f"layers[{i}]")
    for stage_name, (mod, readable) in hook_map.items():
        if stage_name in stages or (want_blocks and stage_name.startswith("enc_blk")):
            handles.append(mod.register_forward_hook(cap(stage_name)))

    # ---- Run the audio encoder ----
    # HF's audio_tower.forward expects a 2D mel (128, T) plus feature_lens
    # because it internally does .T.split(...).
    #
    # QWEN3_REF_WINDOWED=1: reference for the WINDOWED encoder-attention
    # path. The vendored eager/SDPA attention ignores cu_seqlens (full
    # attention); FlashAttention-2 — and the C++ port's windowed mode —
    # restrict attention to n_window_infer blocks. Reproduce that on CPU by
    # monkey-patching each attention layer to receive the block-diagonal
    # mask the modeling's own _prepare_attention_mask() builds from
    # cu_seqlens.
    import os as _os
    if _os.environ.get("QWEN3_REF_WINDOWED", "") not in ("", "0"):
        print("  windowed-attention reference (block-diagonal cu_seqlens mask)")
        _orig_layer_forwards = []
        for layer in audio_tower.layers:
            _orig_layer_forwards.append(layer.forward)

        def _windowed_layer_forward(_self_layer, hidden_states, cu_seqlens, attention_mask=None, **kw):
            if attention_mask is None:
                attention_mask = audio_tower._prepare_attention_mask(hidden_states, cu_seqlens)
            return type(_self_layer).forward(
                _self_layer, hidden_states, cu_seqlens, attention_mask=attention_mask, **kw)

        import functools
        for layer in audio_tower.layers:
            layer.forward = functools.partial(_windowed_layer_forward, layer)

    mel_2d = mel.squeeze(0).to(model.dtype)  # (128, T), model-native dtype
    feature_lens = torch.tensor([mel_2d.shape[-1]], dtype=torch.long)
    with torch.no_grad():
        enc_out = audio_tower(mel_2d, feature_lens=feature_lens)

    final = enc_out.last_hidden_state if hasattr(enc_out, "last_hidden_state") else enc_out
    if "encoder_output" in stages:
        out["encoder_output"] = final.detach().cpu().float().numpy()

    for h in handles:
        h.remove()
    out.update(captures)

    # ---- Trace stages: chat-template prompt + spliced embeds + logits ----
    # This block runs the same pipeline the Python wrapper uses for inference
    # (apply_chat_template → processor → embed → splice audio → text_model
    # forward → lm_head). It gives the C++ side a bit-identical ground truth
    # for the "prompt build + first-forward" path, which is what
    # qwen3-asr-test-trace checks.
    trace_keys = {
        "trace_input_ids", "trace_audio_pad_pos", "trace_inputs_embeds",
        "trace_first_logits", "trace_generated_ids", "llm_input_ids",
    }
    want_trace = bool(stages & trace_keys)
    if want_trace:
        print("  capturing trace stages (chat-template prompt + splice + forward)")
        text_model = thinker.model
        lm_head    = thinker.lm_head

        chat_text = processor.apply_chat_template(
            [{"role": "user", "content": [{"type": "audio", "audio": ""}]}],
            tokenize=False,
            add_generation_prompt=True,
        )
        proc_inputs = processor(
            text=chat_text, audio=audio, sampling_rate=16000, return_tensors="pt",
        )
        input_ids_t = proc_inputs["input_ids"]  # (1, T)
        input_ids_np = input_ids_t[0].detach().cpu().numpy().astype(np.int32)

        # Locate audio_pad positions in the prompt
        pad_pos_np = np.nonzero(input_ids_np == _AUDIO_PAD_ID)[0].astype(np.int32)

        if "llm_input_ids" in stages:
            # Stored as F32 so the existing core_gguf loader can read it.
            # The C++ side casts back to int32 on load.
            out["llm_input_ids"] = input_ids_np.astype(np.float32)
        if "trace_input_ids" in stages:
            out["trace_input_ids"] = input_ids_np.astype(np.float32)
        if "trace_audio_pad_pos" in stages:
            out["trace_audio_pad_pos"] = pad_pos_np.astype(np.float32)

        # Embed text tokens, then splice audio embeds into the audio_pad slots.
        with torch.no_grad():
            inputs_embeds = text_model.embed_tokens(input_ids_t)      # (1, T, D)
            n_splice = min(len(pad_pos_np), final.shape[0])
            inputs_embeds_spliced = inputs_embeds.clone()
            inputs_embeds_spliced[0, pad_pos_np[:n_splice]] = final[:n_splice]

        if "trace_inputs_embeds" in stages:
            out["trace_inputs_embeds"] = (
                inputs_embeds_spliced[0].detach().cpu().float().numpy())

        # One forward pass through the text model + lm_head to capture the
        # last-token logits used as "first logits" by the trace test.
        # lm_head over all T positions is a (T, 151936) tensor — only
        # materialize it when a full-logits stage was requested.
        want_full_logits = bool(stages & {"llm_logits", "llm_argmax"})
        with torch.no_grad():
            lm_out = text_model(
                inputs_embeds=inputs_embeds_spliced, use_cache=False,
            )
            hidden = lm_out.last_hidden_state                         # (1, T, D)
            last_logits_t = lm_head(hidden[:, -1:])                   # (1, 1, V)
        last_logits = last_logits_t[0, -1].detach().cpu().float().numpy()

        if "trace_first_logits" in stages:
            out["trace_first_logits"] = last_logits
        if want_full_logits:
            with torch.no_grad():
                logits = lm_head(hidden)                              # (1, T, V)
            if "llm_logits" in stages:
                # Full per-token logits in ne-order [vocab, T] so the existing
                # qwen3-asr-test-llm driver can compare element-wise.
                full = logits[0].detach().cpu().float().numpy()       # (T, V)
                out["llm_logits"] = full.T.copy()                      # (V, T)
            if "llm_argmax" in stages:
                out["llm_argmax"] = (
                    logits[0].argmax(dim=-1).detach().cpu().numpy().astype(np.int32))

    # ---- End-to-end transcribe() for generated text + ids ----
    want_text     = "generated_text" in stages
    want_gen_ids  = "trace_generated_ids" in stages
    if want_text or want_gen_ids:
        print("  running wrapper.transcribe() for generated text/ids")
        # qwen_asr's transcribe() accepts (waveform, sr) tuples via
        # normalize_audio_input — pass the already-loaded 16k mono audio.
        result = wrapper.transcribe(audio=[(audio, 16000)])
        if isinstance(result, list):
            result = result[0]
        text = getattr(result, "text", str(result))
        if want_text:
            out["generated_text"] = text
        if want_gen_ids:
            # Best-effort: re-tokenize the decoded text. The wrapper strips
            # language/control tokens internally, so the C++ side looks for
            # the longest contiguous match rather than a prefix match.
            gen_ids = processor.tokenizer(
                text, return_tensors="pt", add_special_tokens=False,
            ).input_ids[0].detach().cpu().numpy().astype(np.int32)
            out["trace_generated_ids"] = gen_ids.astype(np.float32)

    return out
