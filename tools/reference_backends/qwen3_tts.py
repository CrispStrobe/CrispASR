"""Qwen3-TTS-12Hz-{0.6B,1.7B}-Base reference dump backend.

Captures stage-by-stage activations from the official `qwen_tts` package
(`pip install -U qwen-tts` + clone of QwenLM/Qwen3-TTS) so we can diff
the CrispASR talker against the bit-true PyTorch path. The test prompt
is fixed and embedded in the dump so both sides see exactly the same
inputs.

Stages dumped (subset selectable via tools/dump_reference.py --stages):

  text_input_ids        — the synth text, tokenised by the official processor
  ref_input_ids         — the ref text (voice clone prompt) tokenised
  text_proj_out         — text_embedding(input_ids) → text_projection
                          ⇒ (T, hidden_size). Pure-text path that doesn't
                          depend on speaker_embed / codec splice, and
                          therefore the easiest first-line numerical
                          check on the CrispASR side.
  talker_layer_0_out    — output of decoder layer 0 on the prefill mix
  talker_layer_27_out   — output of the last decoder layer
  talker_output_norm    — final RMSNorm output
  talker_logits         — codec_head(last hidden state) for the prefill
                          tail position
  generated_codes       — first N greedy codebook-0 codes from
                          generate_voice_clone(do_sample=False)

The "audio" arg in tools/dump_reference.py is repurposed for TTS: pass
a 16 kHz mono WAV that's BOTH the reference audio (for voice cloning)
AND a placeholder so the existing dispatcher's audio-loading path
doesn't break. The synth text and ref text are env-configurable
(QWEN3_TTS_REF_TEXT / QWEN3_TTS_SYN_TEXT) with sensible defaults.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Set

import numpy as np

from . import _hooks

DEFAULT_STAGES = [
    "text_input_ids",
    "ref_input_ids",
    "text_proj_out",
    "icl_role",
    "icl_bridge",
    "icl_codec_input",
    "icl_text_embed",
    "icl_codec_embed",
    "icl_input",
    "ref_spk_embedding",
    "ref_codes",
    # ICL-mode stages — capture the prefill embedding the official
    # generate_voice_clone path actually feeds into talker.model, plus
    # codec_head logits at the last position (= what greedy AR decode
    # would sample first). Together these isolate "is our talker graph
    # bit-equivalent given the same prefill" from "does our C++ prefill
    # builder match the PyTorch one".
    "talker_inputs_embeds",
    "talker_logits",
    "talker_layer_0_out",
    "talker_layer_27_out",
    "talker_output_norm",
    "generated_codes",
    # Per-step code-predictor stages (frame 0 of generate_voice_clone).
    # cp_step{i}_input_embed is the (T, cp_d_model) tensor fed to the
    # code_predictor's small_to_mtp_projection at AR step i — T=2 for
    # step 0 (past_hidden + last_id_hidden), T=1 for steps 1..14.
    # cp_step{i}_logits is the lm_head[i] output at the LAST position
    # (= what greedy AR sampling would pick from). Together they pin
    # the T=1 AR loop so the diff harness can bisect any O15 graph
    # change to the exact step where it diverges.
    *(f"cp_step{i}_input_embed" for i in range(15)),
    *(f"cp_step{i}_logits" for i in range(15)),
]

# Defaults match the official examples/test_model_12hz_base.py smoke
# test so the diff is reproducible without arguments.
_DEFAULT_REF_TEXT = (
    "Okay. Yeah. I resent you. I love you. I respect you. "
    "But you know what? You blew it! And thanks to you."
)
_DEFAULT_SYN_TEXT = "Good one. Okay, fine, I'm just gonna leave this sock monkey here. Goodbye."
_DEFAULT_LANG = "Auto"


def _load_audio_preserve_sr(path: Path) -> tuple[np.ndarray, int]:
    import soundfile as sf

    wav, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if wav.ndim > 1:
        wav = wav.mean(axis=-1)
    return np.ascontiguousarray(wav.astype(np.float32)), int(sr)


def dump(*, model_dir: Path, audio: np.ndarray, stages: Set[str],
         max_new_tokens: int) -> Dict[str, np.ndarray]:
    """Run Qwen3-TTS forward + greedy decode, return captured stage tensors.

    `audio` is the reference audio (16 kHz mono, float32). `model_dir`
    is the HF repo id or local snapshot path. The synth text and ref
    text come from QWEN3_TTS_SYN_TEXT / QWEN3_TTS_REF_TEXT env vars,
    falling back to the official sample defaults.
    """
    import torch

    try:
        from qwen_tts import Qwen3TTSModel
    except ImportError as e:
        raise SystemExit(
            "qwen_tts package not found. Install with: pip install -U qwen-tts\n"
            f"(original import error: {e})")

    syn_text = os.environ.get("QWEN3_TTS_SYN_TEXT", _DEFAULT_SYN_TEXT)
    ref_text = os.environ.get("QWEN3_TTS_REF_TEXT", _DEFAULT_REF_TEXT)
    language = os.environ.get("QWEN3_TTS_LANG", _DEFAULT_LANG)
    ref_wav_override = os.environ.get("QWEN3_TTS_REF_WAV", "")

    print(f"  loading Qwen3-TTS Base from {model_dir} (CPU, fp32, eager attn)")
    tts = Qwen3TTSModel.from_pretrained(
        str(model_dir),
        device_map="cpu",
        dtype=torch.float32,
        attn_implementation="eager",
    )
    model = tts.model
    model.eval()
    talker = model.talker
    processor = tts.processor

    out: Dict[str, np.ndarray] = {}

    prompt_audio = np.ascontiguousarray(audio.astype(np.float32))
    prompt_sr = 16000
    if ref_wav_override:
        prompt_audio, prompt_sr = _load_audio_preserve_sr(Path(ref_wav_override))

    # ---- Tokenise the chat-template prompt + ref text ----
    syn_chat = tts._build_assistant_text(syn_text)  # noqa: SLF001
    ref_chat = tts._build_ref_text(ref_text)        # noqa: SLF001
    syn_ids = processor(text=syn_chat, return_tensors="pt", padding=True)["input_ids"]
    ref_ids = processor(text=ref_chat, return_tensors="pt", padding=True)["input_ids"]
    if syn_ids.dim() == 1:
        syn_ids = syn_ids.unsqueeze(0)
    if ref_ids.dim() == 1:
        ref_ids = ref_ids.unsqueeze(0)
    if "text_input_ids" in stages:
        out["text_input_ids"] = syn_ids[0].detach().cpu().numpy().astype(np.int32).astype(np.float32)
    if "ref_input_ids" in stages:
        out["ref_input_ids"] = ref_ids[0].detach().cpu().numpy().astype(np.int32).astype(np.float32)

    # ---- Pure-text stage: text_embedding + text_projection on the syn prompt ----
    # This isolates the resize-MLP path from the codec splice, so a
    # mismatch here implicates only the text_proj fc1/fc2 + the
    # text_embedding lookup. Done WITHOUT inference_mode so we can
    # also call layer_0 hooks below — but we don't need grads here, so
    # wrap in no_grad to save a few cycles.
    if "text_proj_out" in stages:
        with torch.no_grad():
            text_embeds = talker.get_text_embeddings()(syn_ids)
            text_proj_out = talker.text_projection(text_embeds)
        out["text_proj_out"] = text_proj_out[0].detach().cpu().numpy().astype(np.float32)

    # ---- Per-layer + lm_head dump via forward hooks on a real prefill ----
    #
    # The reference path is `tts.generate_voice_clone(..., do_sample=False)`
    # which internally runs `Qwen3TTSForConditionalGeneration.generate` →
    # builds the ICL prefill (text role + codec sentinels + speaker
    # embedding + first-frame codec sum + tail) → calls
    # `talker.forward(inputs_embeds=...)` → samples codebook-0 from
    # `codec_head` and steps the AR loop.
    #
    # We hook the talker decoder layers + final norm + codec_head AND
    # also intercept `talker.model.forward` to capture the actual
    # `inputs_embeds` argument it received. That `inputs_embeds` is
    # exactly what our C++ talker should consume; the matching
    # `codec_head` output at position [-1] is what greedy decode would
    # then sample from.

    # Capture the FIRST call to each module — every layer fires once for the
    # full prefill and again per AR step with shape (1, 1, *); the first call
    # is the only meaningful one for our diff. `first_call_only=True` is the
    # _hooks.py knob for that pattern (see also `_iter_capture` if/when we
    # add a per-step talker diff).
    captures: Dict[str, "torch.Tensor | np.ndarray"] = {}
    handles = []

    cp_in_names = [f"cp_step{i}_input_embed" for i in range(15)]
    cp_out_names = [f"cp_step{i}_logits" for i in range(15)]

    layer_hook_map = [
        ("talker_layer_0_out",  talker.model.layers[0]),
        ("talker_layer_27_out", talker.model.layers[-1]),
        ("talker_output_norm",  talker.model.norm),
        ("talker_logits",       talker.codec_head),
    ]
    hook_stage_names = [name for name, _mod in layer_hook_map]
    handles.extend(_hooks.capture_modules(
        captures,
        [(name, mod) for name, mod in layer_hook_map if name in stages],
        first_call_only=True,
    ))

    # Pre-hook on talker.model captures the kwargs (specifically inputs_embeds)
    # so we can write the exact prefill back out. _hooks doesn't have a
    # pre-hook helper yet (only post-hooks), so we register inline; the
    # first-call-only logic mirrors the post-hook helper.
    if "talker_inputs_embeds" in stages:
        def cap_embeds(_mod, args, kwargs):
            if "talker_inputs_embeds" in captures:
                return
            embeds = kwargs.get("inputs_embeds", None)
            if embeds is None and len(args) >= 5:
                embeds = args[4]  # signature: (input_ids, attention_mask, position_ids, past_key_values, inputs_embeds)
            if embeds is not None:
                captures["talker_inputs_embeds"] = embeds[0].detach().cpu().float()
        handles.append(talker.model.register_forward_pre_hook(cap_embeds, with_kwargs=True))

    # ---- Per-step code-predictor capture ----
    #
    # The code_predictor.generate(...) call inside the talker AR loop runs
    # 15 forward passes per talker frame (1 prefill at T=2 + 14 gen steps
    # at T=1). The C++ run_code_pred_kv mirrors this exactly. We capture
    # the input embeds (pre small_to_mtp_projection — for the 0.6B base
    # this is Identity, so equivalent to the post-projection tensor our
    # C++ graph consumes) and the lm_head[i] output at each step, but
    # only for the FIRST talker frame. Subsequent frames overflow the
    # counter and are skipped, since the diff harness only compares
    # frame 0.
    if any(name in stages for name in cp_in_names + cp_out_names):
        cp_step_counter = {"i": 0}

        def cp_input_pre_hook(_mod, args):
            i = cp_step_counter["i"]
            if i >= 15:
                return
            x = args[0]
            if isinstance(x, torch.Tensor):
                # x shape: (1, T, hidden_size). Save flat (T, hidden_size).
                captures[f"cp_step{i}_input_embed"] = x[0].detach().cpu().float()
            cp_step_counter["i"] += 1

        handles.append(
            talker.code_predictor.small_to_mtp_projection
                  .register_forward_pre_hook(cp_input_pre_hook)
        )

        def make_lm_head_hook(idx):
            def hook(_mod, _args, output):
                key = f"cp_step{idx}_logits"
                if key in captures:
                    return
                if isinstance(output, torch.Tensor):
                    # output: (1, T, vocab). Last position only — matches
                    # what build_graph_code_pred_kv emits at the "logits"
                    # output node.
                    captures[key] = output[0, -1].detach().cpu().float()
            return hook

        for i in range(len(talker.code_predictor.lm_head)):
            handles.append(
                talker.code_predictor.lm_head[i].register_forward_hook(make_lm_head_hook(i))
            )

    # ---- Build the official ICL prefill sub-blocks for component diffing ----
    prompt_items = None
    if any(s in stages for s in (
        "icl_role",
        "icl_bridge",
        "icl_codec_input",
        "icl_text_embed",
        "icl_codec_embed",
        "icl_input",
        "ref_spk_embedding",
        "ref_codes",
        *hook_stage_names,
        "talker_inputs_embeds",
        "generated_codes",
        *cp_in_names,
        *cp_out_names,
    )):
        prompt_items = tts.create_voice_clone_prompt(
            ref_audio=(prompt_audio, prompt_sr),
            ref_text=ref_text,
            x_vector_only_mode=False,
        )

    if prompt_items is not None and any(s in stages for s in (
        "icl_role",
        "icl_bridge",
        "icl_codec_input",
        "icl_text_embed",
        "icl_codec_embed",
        "icl_input",
        "ref_spk_embedding",
        "ref_codes",
    )):
        import torch

        with torch.no_grad():
            prompt0 = prompt_items[0]
            ref_code = prompt0.ref_code.to(talker.device).clone()
            ref_spk = prompt0.ref_spk_embedding.to(talker.device).to(talker.dtype).clone()

            if "ref_spk_embedding" in stages:
                out["ref_spk_embedding"] = ref_spk.detach().cpu().numpy().astype(np.float32)
            if "ref_codes" in stages:
                out["ref_codes"] = ref_code.detach().cpu().numpy().astype(np.int32).astype(np.float32)

            tts_bos_embed, tts_eos_embed, tts_pad_embed = talker.text_projection(
                talker.get_text_embeddings()(
                    torch.tensor(
                        [[model.config.tts_bos_token_id, model.config.tts_eos_token_id, model.config.tts_pad_token_id]],
                        device=talker.device,
                        dtype=syn_ids.dtype,
                    )
                )
            ).chunk(3, dim=1)

            codec_prefill_ids = torch.tensor(
                [[
                    model.config.talker_config.codec_nothink_id,
                    model.config.talker_config.codec_think_bos_id,
                    model.config.talker_config.codec_think_eos_id,
                ]],
                device=talker.device,
                dtype=syn_ids.dtype,
            )
            codec_tail_ids = torch.tensor(
                [[
                    model.config.talker_config.codec_pad_id,
                    model.config.talker_config.codec_bos_id,
                ]],
                device=talker.device,
                dtype=syn_ids.dtype,
            )

            codec_input_embedding = torch.cat([
                talker.get_input_embeddings()(codec_prefill_ids),
                ref_spk.view(1, 1, -1),
                talker.get_input_embeddings()(codec_tail_ids),
            ], dim=1)

            role = talker.text_projection(talker.get_text_embeddings()(syn_ids[:, :3]))
            bridge = torch.cat((
                tts_pad_embed.expand(-1, codec_input_embedding.shape[1] - 2, -1),
                tts_bos_embed,
            ), dim=1) + codec_input_embedding[:, :-1]

            text_id = syn_ids[:, 3:-5]
            ref_id = ref_ids[:, 3:-2]
            text_embed = talker.text_projection(
                talker.get_text_embeddings()(torch.cat([ref_id, text_id], dim=-1))
            )
            text_embed = torch.cat([text_embed, tts_eos_embed], dim=1)

            codec_rows = []
            for i in range(talker.config.num_code_groups):
                if i == 0:
                    codec_rows.append(talker.get_input_embeddings()(ref_code[:, :1]))
                else:
                    codec_rows.append(talker.code_predictor.get_input_embeddings()[i - 1](ref_code[:, i:i + 1]))
            codec_embed = torch.cat(codec_rows, dim=1).sum(1).unsqueeze(0)
            codec_bos = talker.get_input_embeddings()(
                torch.tensor(
                    [[model.config.talker_config.codec_bos_id]],
                    device=talker.device,
                    dtype=syn_ids.dtype,
                )
            )
            codec_embed = torch.cat([codec_bos, codec_embed], dim=1)

            icl_input, _trailing = model.generate_icl_prompt(
                text_id=text_id,
                ref_id=ref_id,
                ref_code=ref_code,
                tts_pad_embed=tts_pad_embed,
                tts_eos_embed=tts_eos_embed,
                non_streaming_mode=False,
            )

            if "icl_role" in stages:
                out["icl_role"] = role[0].detach().cpu().numpy().astype(np.float32)
            if "icl_bridge" in stages:
                out["icl_bridge"] = bridge[0].detach().cpu().numpy().astype(np.float32)
            if "icl_codec_input" in stages:
                out["icl_codec_input"] = codec_input_embedding[0].detach().cpu().numpy().astype(np.float32)
            if "icl_text_embed" in stages:
                out["icl_text_embed"] = text_embed[0].detach().cpu().numpy().astype(np.float32)
            if "icl_codec_embed" in stages:
                out["icl_codec_embed"] = codec_embed[0].detach().cpu().numpy().astype(np.float32)
            if "icl_input" in stages:
                out["icl_input"] = icl_input[0].detach().cpu().numpy().astype(np.float32)

    # ---- Run generate (greedy) for deterministic codes + activations ----
    if any(s in stages for s in (*hook_stage_names,
                                 "talker_inputs_embeds",
                                 "generated_codes",
                                 *cp_in_names,
                                 *cp_out_names)):
        assert prompt_items is not None
        with torch.no_grad():
            tts.generate_voice_clone(
                text=syn_text,
                language=language,
                voice_clone_prompt=prompt_items,
                max_new_tokens=int(max(8, max_new_tokens or 0)),
                do_sample=False,
                temperature=1.0,  # ignored when do_sample=False
                top_k=1,
            )

    _hooks.drop_hooks(handles)

    # Convert captures (torch.Tensor) -> numpy via _hooks.finalize, but ours
    # are already (B?, T, D) per-stage and the consumers expect raw arrays
    # without the (T, D) dim-sniff that finalize does for NeMo encoders.
    # So we just .numpy() each tensor here.
    import torch
    for name, t in captures.items():
        if isinstance(t, torch.Tensor):
            out[name] = t.numpy()
        else:
            out[name] = t

    return out
