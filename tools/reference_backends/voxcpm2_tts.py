"""VoxCPM2 TTS reference dump backend.

Captures stage-by-stage activations from the official `voxcpm` package
(pip install voxcpm) so we can diff the CrispASR voxcpm2-tts C++ backend
against the bit-true PyTorch path.

Stages dumped (subset selectable via tools/dump_reference.py --stages):

  text_input_ids        — tokenized text input (after CJK char splitting)
  locenc_in             — LocEnc input patches [B, T, P, D]
  locenc_out            — LocEnc CLS output [B, T, hidden_enc]
  enc_to_lm             — enc_to_lm_proj output [B, T, hidden_lm]
  tslm_prefill_out      — TSLM full-sequence output (after FSQ masking)
  tslm_layer_0_out      — TSLM decoder layer 0 output
  tslm_layer_27_out     — TSLM last decoder layer output
  ralm_prefill_out      — RALM full-sequence output
  lm_to_dit_hidden      — projected LM hidden for DiT
  res_to_dit_hidden     — projected residual hidden for DiT
  dit_step0_input       — DiT input for first AR step (concat of mu, t, cond, x)
  dit_step0_output      — DiT velocity prediction for first AR step
  cfm_step0_z           — initial noise for first CFM solve
  cfm_step0_result      — CFM Euler result after N timesteps (first AR step)
  stop_logits_step0     — stop predictor logits at step 0
  generated_latent      — full generated latent sequence [B, D, T*P]
  decoded_audio         — final VAE-decoded 48kHz audio (first 1s)

The "audio" arg in tools/dump_reference.py is repurposed: pass a 16 kHz
mono WAV as reference audio for voice cloning. The synth text is
env-configurable (VOXCPM2_SYN_TEXT) with a sensible default.

Usage:
    python tools/dump_reference.py --backend voxcpm2-tts \\
        --model-dir /path/to/VoxCPM2 \\
        --audio samples/jfk.wav \\
        --output /tmp/voxcpm2-ref.gguf
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Set

import numpy as np
import torch

from . import _hooks

DEFAULT_STAGES = [
    "text_input_ids",
    "locenc_in",
    "locenc_out",
    "enc_to_lm",
    "tslm_prefill_out",
    "tslm_layer_0_out",
    "tslm_layer_27_out",
    "ralm_prefill_out",
    "lm_to_dit_hidden",
    "res_to_dit_hidden",
    "dit_step0_input",
    "dit_step0_output",
    "cfm_step0_z",
    "cfm_step0_result",
    "stop_logits_step0",
    "generated_latent",
    "decoded_audio",
]

# Default synth text
DEFAULT_SYN_TEXT = "Hello, this is a test of the VoxCPM2 text to speech system."


def dump(
    model_dir: str,
    audio: str = "",
    audio_path: str = "",
    stages: Set[str] | None = None,
    device: str = "cpu",
    **kwargs: Any,
) -> Dict[str, np.ndarray]:
    """Run VoxCPM2 inference and capture intermediate activations.

    Args:
        model_dir: Path to local VoxCPM2 model directory.
        audio_path: Path to reference audio (16 kHz mono WAV) for voice cloning.
        stages: Set of stage names to capture (None = all DEFAULT_STAGES).
        device: "cpu" or "cuda".

    Returns:
        Dict mapping stage name → numpy array.
    """
    if stages is None:
        stages = set(DEFAULT_STAGES)

    # Import voxcpm
    try:
        from voxcpm.model.voxcpm2 import VoxCPM2Model
    except ImportError:
        raise ImportError("pip install voxcpm  (needed for VoxCPM2 reference backend)")

    audio_path = audio_path or audio or ""
    syn_text = os.environ.get("VOXCPM2_SYN_TEXT", DEFAULT_SYN_TEXT)
    print(f"  synth text: {syn_text!r}")
    print(f"  ref audio:  {audio_path}")

    # Load model (disable torch.compile for hook compatibility)
    print(f"  loading model from {model_dir} ...")
    model = VoxCPM2Model.from_local(model_dir, optimize=False, device=device)
    model.eval()

    results: Dict[str, np.ndarray] = {}

    # --- Stage: text_input_ids ---
    if "text_input_ids" in stages:
        ids = model.text_tokenizer(syn_text)
        results["text_input_ids"] = np.array(ids, dtype=np.int32)

    # --- Prepare inputs (zero-shot or voice clone) ---
    use_ref = audio_path and os.path.exists(audio_path)

    # Build the input tensors manually to capture intermediates
    text_token = torch.LongTensor(model.text_tokenizer(syn_text))
    text_token = torch.cat([
        text_token,
        torch.tensor([model.audio_start_token], dtype=torch.int32),
    ], dim=-1)
    text_length = text_token.shape[0]

    if use_ref:
        ref_feat = model._encode_wav(audio_path, padding_mode="right")
        ref_tokens, ref_feats, ref_t_mask, ref_a_mask = model._make_ref_prefix(
            ref_feat, text_token.device
        )
        text_pad_feat = torch.zeros(
            (text_length, model.patch_size, model.audio_vae.latent_dim),
            dtype=torch.float32,
        )
        text_token = torch.cat([ref_tokens, text_token])
        audio_feat = torch.cat([ref_feats, text_pad_feat], dim=0)
        text_mask = torch.cat([
            ref_t_mask,
            torch.ones(text_length, dtype=torch.int32),
        ])
        audio_mask = torch.cat([
            ref_a_mask,
            torch.zeros(text_length, dtype=torch.int32),
        ])
    else:
        audio_feat = torch.zeros(
            (text_length, model.patch_size, model.audio_vae.latent_dim),
            dtype=torch.float32,
        )
        text_mask = torch.ones(text_length, dtype=torch.int32)
        audio_mask = torch.zeros(text_length, dtype=torch.int32)

    # Add batch dim and move to device
    text_token = text_token.unsqueeze(0).to(device)
    text_mask = text_mask.unsqueeze(0).to(device)
    audio_feat = audio_feat.unsqueeze(0).to(device).to(torch.bfloat16 if device == "cuda" else torch.float32)
    audio_mask = audio_mask.unsqueeze(0).to(device)

    dtype = next(model.parameters()).dtype

    # --- Run inference with hooks ---
    with torch.inference_mode():
        B, T, P, D = audio_feat.shape

        # LocEnc
        if "locenc_in" in stages:
            results["locenc_in"] = audio_feat[0, :8].cpu().numpy().astype(np.float32)

        feat_embed = model.feat_encoder(audio_feat.to(dtype))
        if "locenc_out" in stages:
            results["locenc_out"] = feat_embed[0, :8].cpu().float().numpy()

        feat_embed = model.enc_to_lm_proj(feat_embed)
        if "enc_to_lm" in stages:
            results["enc_to_lm"] = feat_embed[0, :8].cpu().float().numpy()

        # TSLM prefill
        scale_emb = model.config.lm_config.scale_emb if model.config.lm_config.use_mup else 1.0
        text_embed = model.base_lm.embed_tokens(text_token) * scale_emb
        combined_embed = text_mask.unsqueeze(-1) * text_embed + audio_mask.unsqueeze(-1) * feat_embed

        # Hook layer 0 and last layer
        layer0_out = [None]
        layer27_out = [None]

        def hook_layer0(module, input, output):
            layer0_out[0] = output[0].detach()

        def hook_layer27(module, input, output):
            layer27_out[0] = output[0].detach()

        h0 = model.base_lm.layers[0].register_forward_hook(hook_layer0)
        n_layers = len(model.base_lm.layers)
        h27 = model.base_lm.layers[n_layers - 1].register_forward_hook(hook_layer27)

        enc_outputs, kv_cache_tuple = model.base_lm(
            inputs_embeds=combined_embed.to(dtype), is_causal=True
        )
        model.base_lm.kv_cache.fill_caches(kv_cache_tuple)

        h0.remove()
        h27.remove()

        if "tslm_layer_0_out" in stages and layer0_out[0] is not None:
            results["tslm_layer_0_out"] = layer0_out[0][0, :8].cpu().float().numpy()
        if "tslm_layer_27_out" in stages and layer27_out[0] is not None:
            results["tslm_layer_27_out"] = layer27_out[0][0, :8].cpu().float().numpy()

        enc_outputs = enc_outputs.to(dtype)
        enc_outputs = model.fsq_layer(enc_outputs) * audio_mask.unsqueeze(-1) + enc_outputs * text_mask.unsqueeze(-1)
        lm_hidden = enc_outputs[:, -1, :]

        if "tslm_prefill_out" in stages:
            results["tslm_prefill_out"] = enc_outputs[0, :8].cpu().float().numpy()

        # RALM prefill
        residual_enc_inputs = model.fusion_concat_proj(
            torch.cat((enc_outputs, audio_mask.unsqueeze(-1) * feat_embed.to(dtype)), dim=-1)
        )
        residual_enc_outputs, residual_kv = model.residual_lm(
            inputs_embeds=residual_enc_inputs, is_causal=True
        )
        model.residual_lm.kv_cache.fill_caches(residual_kv)
        residual_hidden = residual_enc_outputs[:, -1, :]

        if "ralm_prefill_out" in stages:
            results["ralm_prefill_out"] = residual_enc_outputs[0, :8].cpu().float().numpy()

        # First AR step — DiT + CFM
        dit_hidden_1 = model.lm_to_dit_proj(lm_hidden)
        dit_hidden_2 = model.res_to_dit_proj(residual_hidden)
        dit_hidden = torch.cat((dit_hidden_1, dit_hidden_2), dim=-1)

        if "lm_to_dit_hidden" in stages:
            results["lm_to_dit_hidden"] = dit_hidden_1[0].cpu().float().numpy()
        if "res_to_dit_hidden" in stages:
            results["res_to_dit_hidden"] = dit_hidden_2[0].cpu().float().numpy()

        # CFM solve (manually to capture internals)
        prefix_feat_cond = audio_feat[:, -1, ...].to(dtype)
        patch_size = model.patch_size
        cfm = model.feat_decoder

        # Initial noise
        z = torch.randn(
            (B, cfm.in_channels, patch_size),
            device=dit_hidden.device, dtype=dit_hidden.dtype,
        )
        if "cfm_step0_z" in stages:
            results["cfm_step0_z"] = z[0].cpu().float().numpy()

        # Run one full CFM solve for step 0
        pred_feat = cfm(
            mu=dit_hidden,
            patch_size=patch_size,
            cond=prefix_feat_cond.transpose(1, 2).contiguous(),
            n_timesteps=10,
            cfg_value=2.0,
        ).transpose(1, 2)  # [B, P, D]

        if "cfm_step0_result" in stages:
            results["cfm_step0_result"] = pred_feat[0].cpu().float().numpy()

        # Stop logits
        stop_logits = model.stop_head(model.stop_actn(model.stop_proj(lm_hidden)))
        if "stop_logits_step0" in stages:
            results["stop_logits_step0"] = stop_logits[0].cpu().float().numpy()

    # --- Full generation (limited length for reference) ---
    max_steps = int(os.environ.get("VOXCPM2_MAX_STEPS", "20"))
    print(f"  running full generation (max {max_steps} steps)...")

    with torch.inference_mode():
        # Reset KV caches
        model.base_lm.kv_cache.reset()
        model.residual_lm.kv_cache.reset()

        # Use the model's generate method with limited length
        if use_ref:
            wav = model.generate(
                target_text=syn_text,
                reference_wav_path=audio_path,
                max_len=max_steps,
                inference_timesteps=10,
                cfg_value=2.0,
            )
        else:
            wav = model.generate(
                target_text=syn_text,
                max_len=max_steps,
                inference_timesteps=10,
                cfg_value=2.0,
            )

        if "decoded_audio" in stages and wav is not None:
            # Capture first 48000 samples (1 second at 48kHz)
            audio_np = wav[0, :48000].numpy().astype(np.float32)
            results["decoded_audio"] = audio_np

    print(f"  captured {len(results)} stages")
    return results
