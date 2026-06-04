"""MOSS-Audio-4B-Instruct reference dump backend.

Captures stage-by-stage activations from the official PyTorch
``MossAudioModel`` forward pass so the C++ runtime in
src/moss_audio.cpp can be compared element-wise via crispasr-diff.

Architecture (OpenMOSS-Team/MOSS-Audio-4B-Instruct):

  audio PCM (16 kHz mono)
    → WhisperFeatureExtractor  128-bin log-mel (n_fft=400, hop=160)
    → MossAudioEncoder:
        3× Conv2d stride-2 → stem_proj (7680→1280)
        + sinusoidal pos → 32 WhisperEncoderLayers (d=1280, 20h, FFN 5120)
        → layer_norm → out_proj (Identity for 4B)
        DeepStack taps at layers [8, 16, 24]
    → audio_adapter GatedMLP (1280→8192→2560, SiLU)
    → masked_scatter into text embeddings at audio_token positions
    → 3× deepstack_audio_merger GatedMLP (1280→8192→2560)
        injected as residuals at LM layers [0, 1, 2] via forward hooks
    → 36L Qwen3 LM (2560d, 32Q/8KV, head_dim=128, QK-norm, SwiGLU 9728,
                     RoPE θ=1M, vocab=151936)
    → lm_head → logits → greedy decode

Stages dumped (all (T, D) row-major where applicable):

  mel_spectrogram         (128, T_mel)         F32
  enc_post_conv           (T_down, 7680)       F32 — after 3 conv + flatten
  enc_post_stem_proj      (T_down, 1280)       F32 — after stem_proj + pos
  enc_layer_8             (T_down, 1280)       F32 — DeepStack tap #0
  enc_layer_16            (T_down, 1280)       F32 — DeepStack tap #1
  enc_layer_24            (T_down, 1280)       F32 — DeepStack tap #2
  encoder_output          (T_down, 1280)       F32 — final encoder output
  adapter_output          (T_down, 2560)       F32 — after audio_adapter
  deepstack_proj_0        (T_down, 2560)       F32 — merger 0 output
  deepstack_proj_1        (T_down, 2560)       F32 — merger 1 output
  deepstack_proj_2        (T_down, 2560)       F32 — merger 2 output
  prefill_inputs_embeds   (T_total, 2560)      F32 — LM input after scatter
  prefill_last_hidden     (2560,)              F32 — LM last-position hidden
  prefill_logits_step0    (151936,)            F32 — lm_head at last position
  prefill_argmax_step0    (1,)                 I32
  generated_text          str

Environment:
  MOSS_AUDIO_DIR        — HF snapshot or local path to MOSS-Audio-4B-Instruct
  MOSS_AUDIO_GITHUB     — path to cloned OpenMOSS/MOSS-Audio GitHub repo
                          (default: ref/moss_audio/github)
  MOSS_AUDIO_PROMPT     — override prompt (default "Transcribe this audio.")
  MOSS_AUDIO_MAX_NEW    — max_new_tokens for generate() (default 128)
"""

from __future__ import annotations

import gc
import os
import sys
from pathlib import Path
from typing import Dict, Set

import numpy as np

DEFAULT_STAGES = [
    "mel_spectrogram",
    "enc_post_stem_proj",
    "enc_layer_8",
    "enc_layer_16",
    "enc_layer_24",
    "encoder_output",
    "adapter_output",
    "deepstack_proj_0",
    "deepstack_proj_1",
    "deepstack_proj_2",
    "prefill_inputs_embeds",
    "prefill_last_hidden",
    "prefill_logits_step0",
    "prefill_argmax_step0",
    "generated_text",
]


def _ensure_github_on_path():
    """Put the MOSS-Audio GitHub clone on sys.path so we can import
    ``src.modeling_moss_audio`` and ``src.configuration_moss_audio``.

    Expected layout: ref/moss_audio/github/ containing the cloned repo
    with src/__init__.py, src/modeling_moss_audio.py, etc."""
    github_dir = os.environ.get("MOSS_AUDIO_GITHUB")
    if github_dir:
        p = Path(github_dir)
    else:
        p = Path(__file__).resolve().parents[2] / "ref" / "moss_audio" / "github"
    if p.is_dir() and str(p) not in sys.path:
        sys.path.insert(0, str(p))
    return p


def dump(*, model_dir: Path, audio: np.ndarray, stages: Set[str],
         max_new_tokens: int) -> Dict[str, np.ndarray]:
    """Run MOSS-Audio-4B-Instruct reference forward and capture activations."""
    import torch

    github_path = _ensure_github_on_path()
    model_dir = Path(os.environ.get("MOSS_AUDIO_DIR", str(model_dir)))
    prompt = os.environ.get("MOSS_AUDIO_PROMPT", "Transcribe this audio.")
    max_new = int(os.environ.get("MOSS_AUDIO_MAX_NEW", str(max_new_tokens)))

    # ---- Import model classes ----
    # The HF repo only ships config + processor; modeling code comes from
    # the GitHub clone under ref/moss_audio/github/src/.
    from src.configuration_moss_audio import MossAudioConfig
    from src.modeling_moss_audio import MossAudioModel
    from src.processing_moss_audio import MossAudioProcessor

    print(f"  loading MOSS-Audio-4B-Instruct from {model_dir}")
    print(f"  GitHub source: {github_path}")

    # ---- Load model ----
    # Load config manually (bypasses HF auto_map trust_remote_code prompt)
    # then use from_pretrained with the pre-built config object.
    import json as _json
    with open(Path(model_dir) / "config.json") as f:
        cfg_dict = _json.load(f)
    config = MossAudioConfig(
        audio_config=cfg_dict.get("audio_config"),
        language_config=cfg_dict.get("language_config"),
        adapter_hidden_size=cfg_dict.get("adapter_hidden_size", 8192),
        deepstack_num_inject_layers=cfg_dict.get("deepstack_num_inject_layers"),
    )

    # Use CPU — Kaggle may assign a P100 (sm_60) incompatible with
    # current PyTorch CUDA. bfloat16 on CPU requires PyTorch >=2.0.
    device = "cpu"
    dtype = torch.bfloat16
    # Load weights with the manually-built config — no auto_map lookup
    model = MossAudioModel(config)
    # Load sharded safetensors weights
    from safetensors.torch import load_file
    import glob as _glob
    st_files = sorted(_glob.glob(str(Path(model_dir) / "model-*.safetensors")))
    if not st_files:
        st_files = sorted(_glob.glob(str(Path(model_dir) / "*.safetensors")))
    state_dict = {}
    for sf in st_files:
        state_dict.update(load_file(sf))
    model.load_state_dict(state_dict, strict=False)
    del state_dict
    gc.collect()
    model = model.to(device=device, dtype=dtype).eval()
    print(f"  device={device}, dtype={dtype}, params={sum(p.numel() for p in model.parameters())/1e6:.0f}M")

    # Processor — also bypass HF auto_map by loading from the JSON config
    with open(Path(model_dir) / "processor_config.json") as f:
        proc_cfg = _json.load(f)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)
    # Override mel_dtype to float32 for CPU compatibility
    mel_cfg = dict(proc_cfg.get("mel_config", {}))
    mel_cfg["mel_dtype"] = "float32"
    processor = MossAudioProcessor(
        tokenizer=tokenizer,
        mel_config=mel_cfg,
        enable_time_marker=True,
        audio_token_id=proc_cfg.get("audio_token_id", 151654),
        audio_start_id=proc_cfg.get("audio_start_id", 151669),
        audio_end_id=proc_cfg.get("audio_end_id", 151670),
    )

    # ---- Prepare inputs ----
    # Use WhisperFeatureExtractor directly to avoid _np_extract_fbank_features
    # compatibility issues across transformers versions.
    from transformers import WhisperFeatureExtractor
    fe = WhisperFeatureExtractor(
        feature_size=128, sampling_rate=16000, hop_length=160, n_fft=400,
    )
    mel_features = fe(audio, sampling_rate=16000, return_tensors="pt")
    mel_input = mel_features.input_features  # (1, 128, T)

    # Build prompt with the processor (text-only path — no audio mel)
    audio_seq_len = processor._conv3_downsample_len(mel_input.shape[-1])
    prompt_text = processor._build_default_prompt(prompt, has_audio=True)
    input_ids = processor._build_input_from_prompt(prompt_text, [audio_seq_len])
    input_ids_tensor = torch.tensor([input_ids], dtype=torch.long)
    attention_mask = torch.ones_like(input_ids_tensor)

    inputs = {
        "input_ids": input_ids_tensor.to(device),
        "attention_mask": attention_mask.to(device),
        "audio_data": mel_input.to(device=device, dtype=dtype),
        "audio_data_seqlens": torch.tensor([mel_input.shape[-1]], dtype=torch.long, device=device),
    }

    audio_input_mask = inputs["input_ids"] == processor.audio_token_id
    inputs["audio_input_mask"] = audio_input_mask

    out: Dict[str, np.ndarray] = {}

    # ---- Stage: mel_spectrogram ----
    if "mel_spectrogram" in stages:
        # mel_input is (1, 128, T) from WhisperFeatureExtractor
        out["mel_spectrogram"] = mel_input[0].detach().cpu().float().numpy()

    with torch.no_grad():
        audio_data = inputs.get("audio_data")
        audio_data_seqlens = inputs.get("audio_data_seqlens")

        if audio_data is not None:
            audio_data = audio_data.float()

        # ---- Encoder stages ----
        want_enc = any(s in stages for s in [
            "enc_post_conv", "enc_post_stem_proj",
            "enc_layer_8", "enc_layer_16", "enc_layer_24",
            "encoder_output",
        ])
        want_adapter = "adapter_output" in stages
        want_deepstack = any(f"deepstack_proj_{i}" in stages for i in range(3))

        encoder_last_hidden = None
        deepstack_hidden = None

        if want_enc or want_adapter or want_deepstack:
            # Run encoder with hooks for intermediate captures
            encoder = model.audio_encoder
            hook_captures = {}
            handles = []

            # Hook after stem_proj (before transformer layers)
            if "enc_post_stem_proj" in stages or "enc_post_conv" in stages:
                def _stem_hook(module, input, output):
                    hook_captures["stem_proj_out"] = output.detach().clone()
                handles.append(encoder.stem_proj.register_forward_hook(_stem_hook))

            # Hook encoder layers for DeepStack taps
            for tap_layer in [8, 16, 24]:
                stage_name = f"enc_layer_{tap_layer}"
                if stage_name in stages:
                    def _make_layer_hook(name):
                        def _hook(module, input, output):
                            if isinstance(output, tuple):
                                hook_captures[name] = output[0].detach().clone()
                            else:
                                hook_captures[name] = output.detach().clone()
                        return _hook
                    # Encoder layers are 0-indexed, config taps at [8,16,24]
                    handles.append(
                        encoder.layers[tap_layer].register_forward_hook(
                            _make_layer_hook(stage_name)
                        )
                    )

            # Full encoder forward
            enc_out = model.audio_encoder(
                input_features=audio_data,
                feature_lens=audio_data_seqlens,
                output_deepstack_hidden_states=True,
            )
            encoder_last_hidden = enc_out.last_hidden_state  # [1, T_enc, 1280]
            deepstack_hidden = list(enc_out.hidden_states) if enc_out.hidden_states else []

            # Clean up hooks
            for h in handles:
                h.remove()

            # Capture encoder stages
            if "enc_post_stem_proj" in stages and "stem_proj_out" in hook_captures:
                out["enc_post_stem_proj"] = hook_captures["stem_proj_out"][0].cpu().float().numpy()
            for tap_layer in [8, 16, 24]:
                stage_name = f"enc_layer_{tap_layer}"
                if stage_name in stages and stage_name in hook_captures:
                    out[stage_name] = hook_captures[stage_name][0].cpu().float().numpy()
            if "encoder_output" in stages:
                out["encoder_output"] = encoder_last_hidden[0].cpu().float().numpy()

        # ---- Adapter stage ----
        if want_adapter and encoder_last_hidden is not None:
            adapter_out = model.audio_adapter(encoder_last_hidden)
            out["adapter_output"] = adapter_out[0].detach().cpu().float().numpy()

        # ---- DeepStack merger stages ----
        if want_deepstack and deepstack_hidden:
            for i, (ds_hidden, merger) in enumerate(
                zip(deepstack_hidden[:len(model.deepstack_audio_merger_list)],
                    model.deepstack_audio_merger_list)
            ):
                stage_name = f"deepstack_proj_{i}"
                if stage_name in stages:
                    ds_proj = merger(ds_hidden)
                    out[stage_name] = ds_proj[0].detach().cpu().float().numpy()

        # ---- Full forward for LM stages ----
        want_lm = any(s in stages for s in [
            "prefill_inputs_embeds", "prefill_last_hidden",
            "prefill_logits_step0", "prefill_argmax_step0",
        ])
        want_text = "generated_text" in stages

        if want_lm:
            # We need to intercept the full forward to capture inputs_embeds
            # and the LM hidden state. Run the model's forward() manually.
            input_ids = inputs["input_ids"]
            attention_mask = inputs["attention_mask"]

            # Step 1: Build inputs_embeds with audio scatter
            inputs_embeds = model.get_input_embeddings()(input_ids)

            if encoder_last_hidden is None:
                enc_out = model.audio_encoder(
                    input_features=audio_data,
                    feature_lens=audio_data_seqlens,
                    output_deepstack_hidden_states=True,
                )
                encoder_last_hidden = enc_out.last_hidden_state
                deepstack_hidden = list(enc_out.hidden_states) if enc_out.hidden_states else []

            audio_embeds = model.audio_adapter(encoder_last_hidden)

            # Scatter audio embeddings into text embedding positions
            mask_expanded = audio_input_mask.unsqueeze(-1).expand_as(inputs_embeds)
            inputs_embeds = inputs_embeds.clone()
            inputs_embeds.masked_scatter_(mask_expanded, audio_embeds)

            if "prefill_inputs_embeds" in stages:
                out["prefill_inputs_embeds"] = inputs_embeds[0].detach().cpu().float().numpy()

            # Step 2: Prepare deepstack hooks
            deepstack_audio_embeds = []
            if deepstack_hidden and len(model.deepstack_audio_merger_list) > 0:
                for i, ds_h in enumerate(deepstack_hidden[:len(model.deepstack_audio_merger_list)]):
                    deepstack_audio_embeds.append(model.deepstack_audio_merger_list[i](ds_h))

            hook_handles = model._register_llm_deepstack_hooks(
                audio_input_mask, deepstack_audio_embeds
            ) if deepstack_audio_embeds else []

            try:
                # Step 3: Run LM forward
                lm_outputs = model.language_model(
                    input_ids=None,
                    attention_mask=attention_mask,
                    inputs_embeds=inputs_embeds,
                    use_cache=False,
                    output_hidden_states=False,
                    return_dict=True,
                )
            finally:
                for h in hook_handles:
                    h.remove()

            hidden_states = lm_outputs[0]  # [1, T, 2560]
            logits = model.lm_head(hidden_states)  # [1, T, 151936]

            if "prefill_last_hidden" in stages:
                out["prefill_last_hidden"] = hidden_states[0, -1].cpu().float().numpy()
            if "prefill_logits_step0" in stages:
                out["prefill_logits_step0"] = logits[0, -1].cpu().float().numpy()
            if "prefill_argmax_step0" in stages:
                argmax = int(logits[0, -1].argmax().item())
                out["prefill_argmax_step0"] = np.array([argmax], dtype=np.int32)

        # ---- Full generation for text ----
        if want_text:
            gen = model.generate(
                **inputs,
                max_new_tokens=max_new,
                do_sample=False,
                num_beams=1,
                use_cache=True,
            )
            input_len = inputs["input_ids"].shape[1]
            decoded = processor.decode(gen[0, input_len:], skip_special_tokens=True)
            out["generated_text"] = decoded
            print(f"  generated: {decoded[:200]}")

    return out
