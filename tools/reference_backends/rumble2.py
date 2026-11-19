"""Reson8/Rumble-2-Flash (#602) reference dump via native transformers.

Rumble-2-Flash is a Qwen3-ASR fine-tune (audio 18L d=896, LLM 14L d=1024)
with a re-indexed 99 630-token vocabulary and Reson8 control tokens in the
system turn. The `qwen3` reference backend loads the qwen_asr package's
`thinker.*` layout; this one loads the HF layout the repo ships
(Qwen3ASRForConditionalGeneration) and builds the prompt with the repo's own
chat_template.jinja, so the system turn is "<digits><nofiller><verbatim>"
exactly as the model card's defaults produce it.

Stage names match `reference_backends/qwen3.py`, so `crispasr-diff qwen3`
consumes this dump unchanged. Native transformers applies the n_window_infer
(8 s) encoder attention windows in every attention implementation, which is
also the C++ encoder's default.

RUMBLE_REF_LANGUAGE=<English name> forces the language through the template's
`language` kwarg (assistant prefill "language X<asr_text>"), the documented
forcing path and the one `crispasr -l` uses.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Set

import numpy as np

DEFAULT_STAGES = [
    "raw_audio",
    "mel_spectrogram",
    "conv_out",
    "enc_blk00_out",
    "ln_post_out",
    "proj1_out",
    "proj2_out",
    "encoder_output",
    "trace_input_ids",
    "trace_audio_pad_pos",
    "trace_inputs_embeds",
    "trace_first_logits",
    "generated_text",
]


def dump(*, model_dir: Path, audio: np.ndarray, stages: Set[str], max_new_tokens: int = 256) -> Dict[str, np.ndarray]:
    import torch
    from transformers import AutoProcessor, Qwen3ASRForConditionalGeneration

    dtype = getattr(torch, os.environ.get("RUMBLE_REF_DTYPE", "float32"))
    processor = AutoProcessor.from_pretrained(str(model_dir))
    model = Qwen3ASRForConditionalGeneration.from_pretrained(str(model_dir), dtype=dtype).eval()
    tower = model.model.audio_tower
    proj = model.model.multi_modal_projector
    audio_pad_id = int(model.config.audio_token_id)

    out: Dict[str, np.ndarray] = {}
    if "raw_audio" in stages:
        out["raw_audio"] = audio.astype(np.float32)

    tmpl_kw = {}
    lang = os.environ.get("RUMBLE_REF_LANGUAGE", "")
    if lang:
        tmpl_kw["language"] = lang
    chat_text = processor.apply_chat_template(
        [{"role": "user", "content": [{"type": "audio", "audio": ""}]}],
        tokenize=False, add_generation_prompt=True, **tmpl_kw,
    )
    print(f"  prompt template: {chat_text!r}")
    inputs = processor(text=chat_text, audio=audio, sampling_rate=16000, return_tensors="pt")
    feats = inputs["input_features"].to(dtype)
    fmask = inputs["input_features_mask"]
    n_valid = int(fmask.sum())
    if "mel_spectrogram" in stages:
        # The C++ front end produces only the valid frames; the processor
        # right-pads the mel axis to a multiple of 100.
        out["mel_spectrogram"] = feats[:, :, :n_valid].float().numpy()

    caps: Dict[str, np.ndarray] = {}

    def cap(name):
        def hook(_m, _i, o):
            t = o[0] if isinstance(o, tuple) else o
            caps[name] = t.detach().float().numpy()
        return hook

    hooks = {"conv_out": tower.conv_out, "ln_post_out": tower.ln_post,
             "proj1_out": proj.linear_1, "proj2_out": proj.linear_2}
    if any(s.startswith("enc_blk") for s in stages):
        for i, layer in enumerate(tower.layers):
            hooks[f"enc_blk{i:02d}_out"] = layer
    handles = [m.register_forward_hook(cap(n)) for n, m in hooks.items()
               if n in stages or n.startswith("enc_blk")]
    with torch.no_grad():
        audio_out = model.model.get_audio_features(input_features=feats, input_features_mask=fmask)
    for h in handles:
        h.remove()
    enc = audio_out.pooler_output  # (N_enc, 1024)
    out.update({k: v for k, v in caps.items()})
    if "encoder_output" in stages:
        out["encoder_output"] = enc.float().numpy()

    ids = inputs["input_ids"]
    ids_np = ids[0].numpy().astype(np.int32)
    pad_pos = np.nonzero(ids_np == audio_pad_id)[0].astype(np.int32)
    assert len(pad_pos) == enc.shape[0], (len(pad_pos), enc.shape)
    out["trace_input_ids"] = ids_np.astype(np.float32)
    out["trace_audio_pad_pos"] = pad_pos.astype(np.float32)
    with torch.no_grad():
        emb = model.model.language_model.embed_tokens(ids).clone()
        emb[0, pad_pos] = enc.to(emb.dtype)
        hid = model.model.language_model(inputs_embeds=emb, use_cache=False).last_hidden_state
        last = model.lm_head(hid[:, -1:])[0, -1]
    if "trace_inputs_embeds" in stages:
        out["trace_inputs_embeds"] = emb[0].float().numpy()
    out["trace_first_logits"] = last.float().numpy()
    print(f"  first argmax {int(last.argmax())}")

    if "generated_text" in stages or "trace_generated_ids" in stages:
        inputs = {k: (v.to(dtype) if v.is_floating_point() else v) for k, v in inputs.items()}
        with torch.no_grad():
            gen = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        new = gen[:, ids.shape[1]:]
        raw = processor.decode(new[0], skip_special_tokens=False)
        text = processor.decode(new, return_format="transcription_only")[0]
        print(f"  raw: {raw!r}")
        print(f"  transcription: {text!r}")
        out["generated_text"] = text
        out["trace_generated_ids"] = new[0].numpy().astype(np.float32)
    return out
