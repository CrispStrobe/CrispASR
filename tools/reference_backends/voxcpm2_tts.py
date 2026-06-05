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

import gc
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Set

import numpy as np
import torch

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
    "cfm_mu",
    "dit_input_seq",
    "dit_single_fwd",
    "cfm_step0_z",
    "cfm_step0_result",
    "stop_logits_step0",
    "generated_latent",
    "decoded_audio",
]

# Stages that require the manual prefill path (hook-based intermediate capture)
_PREFILL_STAGES = {
    "locenc_in", "locenc_out", "enc_to_lm",
    "tslm_prefill_out", "tslm_last_hidden", "tslm_layer_0_out", "tslm_layer_27_out",
    "ralm_prefill_out",
    "lm_to_dit_hidden", "res_to_dit_hidden", "cfm_mu",
    "dit_input_seq", "dit_single_fwd", "cfm_step0_z", "cfm_step0_result",
    "stop_logits_step0",
}

# Stages that require model.generate() (full AR loop + VAE decode)
_GENERATE_STAGES = {"decoded_audio", "generated_latent"}

# Default synth text
DEFAULT_SYN_TEXT = "Hello, this is a test of the VoxCPM2 text to speech system."


def _write_wav_16k(pcm_f32: np.ndarray, path: str) -> None:
    """Write float32 PCM to a 16 kHz mono 16-bit WAV file."""
    import wave
    pcm_i16 = np.clip(pcm_f32 * 32767, -32768, 32767).astype(np.int16)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(pcm_i16.tobytes())


def dump(
    *,
    model_dir: Path,
    audio: np.ndarray,
    stages: Set[str],
    max_new_tokens: int = 20,
) -> Dict[str, np.ndarray]:
    """Run VoxCPM2 inference and capture intermediate activations.

    Args:
        model_dir: Path to local VoxCPM2 model directory.
        audio: 16 kHz mono float32 PCM (used as voice-clone reference).
        stages: Set of stage names to capture.
        max_new_tokens: Max AR generation steps.

    Returns:
        Dict mapping stage name → numpy array.
    """
    # Ensure voxcpm is importable (installed at /tmp/voxcpm_src)
    voxcpm_path = "/tmp/voxcpm_src"
    if voxcpm_path not in sys.path:
        sys.path.insert(0, voxcpm_path)

    try:
        from voxcpm.model.voxcpm2 import VoxCPM2Model
    except ImportError:
        raise ImportError(
            "pip install --no-deps --target=/tmp/voxcpm_src voxcpm  "
            "(needed for VoxCPM2 reference backend)"
        )

    syn_text = os.environ.get("VOXCPM2_SYN_TEXT", DEFAULT_SYN_TEXT)
    use_ref = os.environ.get("VOXCPM2_USE_REF", "0") == "1"
    seed = int(os.environ.get("VOXCPM2_SEED", "42"))
    print(f"  synth text: {syn_text!r}")
    print(f"  use ref audio: {use_ref}")

    # Write audio to temp WAV if we need voice cloning
    ref_wav_path = ""
    if use_ref and audio is not None and len(audio) > 0:
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        _write_wav_16k(audio, tmp.name)
        ref_wav_path = tmp.name
        print(f"  ref audio written to: {ref_wav_path}")

    device = "cpu"
    use_f32 = os.environ.get("VOXCPM2_F32", "0") == "1"
    print(f"  loading model from {model_dir} ...")
    model = VoxCPM2Model.from_local(str(model_dir), optimize=False, device=device)
    if use_f32:
        model = model.float()
        print("  forced F32 inference mode (VOXCPM2_F32=1)")
    model.eval()

    results: Dict[str, np.ndarray] = {}

    # --- Stage: text_input_ids ---
    if "text_input_ids" in stages:
        ids = model.text_tokenizer(syn_text)
        results["text_input_ids"] = np.array(ids, dtype=np.int32)

    # Decide which paths to run
    need_prefill = bool(stages & _PREFILL_STAGES)
    need_generate = bool(stages & _GENERATE_STAGES)

    # --- Manual prefill path: capture intermediate activations via hooks ---
    if need_prefill:
        print(f"  running manual prefill for {len(stages & _PREFILL_STAGES)} stages...")
        _run_prefill(model, syn_text, use_ref, ref_wav_path, device, seed, stages, results)

        # Free prefill intermediates before full generation
        if need_generate:
            gc.collect()

    # --- Full generation path: model.generate() for end-to-end audio ---
    if need_generate:
        max_steps = int(os.environ.get("VOXCPM2_MAX_STEPS", str(max_new_tokens)))
        print(f"  running full generation (max {max_steps} steps)...")

        # Re-init KV caches (prefill may have filled them, or they need fresh init)
        cache_dtype = next(model.parameters()).dtype
        model.base_lm.setup_cache(1, model.config.max_length, device, cache_dtype)
        model.residual_lm.setup_cache(1, model.config.max_length, device, cache_dtype)

        # Hook the VAE decode to capture the latent that goes IN — that's the
        # boundary between AR generation (TSLM/RALM/LocDiT/CFM) and VAE. Lets
        # us isolate VAE drift from upstream drift downstream by feeding the
        # captured latent into a C++ VAE-only test path.
        capture_lat = {"val": None}
        orig_decode = model.audio_vae.decode

        def hook_decode(latent):
            # latent: torch.Tensor shape [B, D, T_lat]
            capture_lat["val"] = latent.detach().cpu().float().numpy()
            return orig_decode(latent)

        model.audio_vae.decode = hook_decode

        try:
            with torch.inference_mode():
                torch.manual_seed(seed)
                if use_ref and ref_wav_path:
                    wav = model.generate(
                        target_text=syn_text,
                        reference_wav_path=ref_wav_path,
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
                    wav_1d = wav.squeeze()
                    audio_np = wav_1d[:48000].numpy().astype(np.float32)
                    results["decoded_audio"] = audio_np

                if "generated_latent" in stages and capture_lat["val"] is not None:
                    # Saved as [B=1, D=64, T_lat]; drop batch dim → [D, T_lat]
                    lat = capture_lat["val"][0]
                    results["generated_latent"] = lat.astype(np.float32)
        finally:
            model.audio_vae.decode = orig_decode

    # Cleanup temp WAV
    if ref_wav_path:
        try:
            os.unlink(ref_wav_path)
        except OSError:
            pass

    print(f"  captured {len(results)} stages")
    return results


def _run_prefill(
    model,
    syn_text: str,
    use_ref: bool,
    ref_wav_path: str,
    device: str,
    seed: int,
    stages: Set[str],
    results: Dict[str, np.ndarray],
) -> None:
    """Manual prefill with hooks to capture intermediate activations."""

    # Build input tensors (mirrors _generate zero-shot / ref-only path)
    text_token = torch.LongTensor(model.text_tokenizer(syn_text))
    text_token = torch.cat([
        text_token,
        torch.tensor([model.audio_start_token], dtype=torch.int32),
    ], dim=-1)
    text_length = text_token.shape[0]

    if use_ref and ref_wav_path:
        ref_feat = model._encode_wav(ref_wav_path, padding_mode="right")
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
    audio_feat = audio_feat.unsqueeze(0).to(device)
    audio_mask = audio_mask.unsqueeze(0).to(device)

    dtype = next(model.parameters()).dtype

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
        layer_last_out = [None]

        def hook_layer0(module, input, output):
            layer0_out[0] = output[0].detach() if isinstance(output, tuple) else output.detach()

        def hook_layer_last(module, input, output):
            layer_last_out[0] = output[0].detach() if isinstance(output, tuple) else output.detach()

        h0 = model.base_lm.layers[0].register_forward_hook(hook_layer0)
        n_layers = len(model.base_lm.layers)
        h_last = model.base_lm.layers[n_layers - 1].register_forward_hook(hook_layer_last)

        enc_outputs, kv_cache_tuple = model.base_lm(
            inputs_embeds=combined_embed.to(dtype), is_causal=True
        )
        model.base_lm.kv_cache.fill_caches(kv_cache_tuple)

        h0.remove()
        h_last.remove()

        if "tslm_layer_0_out" in stages and layer0_out[0] is not None:
            results["tslm_layer_0_out"] = layer0_out[0][0, :8].cpu().float().numpy()
        if "tslm_layer_27_out" in stages and layer_last_out[0] is not None:
            results["tslm_layer_27_out"] = layer_last_out[0][0, :8].cpu().float().numpy()

        # Free hook captures
        del layer0_out, layer_last_out, kv_cache_tuple

        enc_outputs = enc_outputs.to(dtype)
        enc_outputs = model.fsq_layer(enc_outputs) * audio_mask.unsqueeze(-1) + enc_outputs * text_mask.unsqueeze(-1)
        lm_hidden = enc_outputs[:, -1, :]

        if "tslm_prefill_out" in stages:
            results["tslm_prefill_out"] = enc_outputs[0, :8].cpu().float().numpy()
        if "tslm_last_hidden" in stages:
            results["tslm_last_hidden"] = lm_hidden[0].cpu().float().numpy()

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

        # Free large intermediates
        del enc_outputs, residual_enc_inputs, residual_enc_outputs, residual_kv

        # First AR step — DiT + CFM
        dit_hidden_1 = model.lm_to_dit_proj(lm_hidden)
        dit_hidden_2 = model.res_to_dit_proj(residual_hidden)
        dit_hidden = torch.cat((dit_hidden_1, dit_hidden_2), dim=-1)

        if "lm_to_dit_hidden" in stages:
            results["lm_to_dit_hidden"] = dit_hidden_1[0].cpu().float().numpy()
        if "res_to_dit_hidden" in stages:
            results["res_to_dit_hidden"] = dit_hidden_2[0].cpu().float().numpy()
        if "cfm_mu" in stages:
            results["cfm_mu"] = dit_hidden[0].cpu().float().numpy()

        # Dump the full LocDiT input sequence (for debugging transformer layers)
        if "dit_input_seq" in stages:
            # Generate z (same as cfm_step0_z) if not already done
            rng_dit = torch.Generator(device=device)
            rng_dit.manual_seed(seed)
            cfm = model.feat_decoder
            z_dit = torch.randn(
                (1, cfm.in_channels, model.patch_size),
                device=device, dtype=dit_hidden.dtype, generator=rng_dit,
            )
            # Build the full input sequence at the FIRST real LocDiT step
            t_span_local = torch.linspace(1, 0, 11, device=device, dtype=dtype)
            t_span_local = t_span_local + 1.0 * (torch.cos(torch.pi / 2 * t_span_local) - 1 + t_span_local)
            t_val = t_span_local[1:2]  # First real step (after zero-init)
            dt_val_local = torch.tensor([0.0], device=device, dtype=dtype)
            cfm_est = model.feat_decoder.estimator
            x_proj = cfm_est.in_proj(z_dit.transpose(1, 2).contiguous())
            # For zero-shot first step, cond is the last position's audio_feat = zeros
            zero_cond = torch.zeros(1, cfm.in_channels, model.patch_size, device=device, dtype=dtype)
            cond_proj_dit = cfm_est.cond_proj(zero_cond.transpose(1, 2).contiguous())
            t_emb_dit = cfm_est.time_embeddings(t_val).to(dtype)
            t_emb_dit = cfm_est.time_mlp(t_emb_dit)
            dt_emb_dit = cfm_est.time_embeddings(dt_val_local).to(dtype)
            dt_emb_dit = cfm_est.delta_time_mlp(dt_emb_dit)
            t_emb_dit = t_emb_dit + dt_emb_dit
            mu_reshaped = dit_hidden.view(1, -1, x_proj.size(-1))
            dit_seq = torch.cat([mu_reshaped, t_emb_dit.unsqueeze(1), cond_proj_dit, x_proj], dim=1)
            results["dit_input_seq"] = dit_seq[0].cpu().float().numpy()

        # Single LocDiT forward pass (for debugging transformer layers in isolation)
        if "dit_single_fwd" in stages:
            # Use the same inputs as dit_input_seq but run through the full estimator
            rng_sf = torch.Generator(device=device)
            rng_sf.manual_seed(seed)
            cfm = model.feat_decoder
            z_sf = torch.randn(
                (1, cfm.in_channels, model.patch_size),
                device=device, dtype=dit_hidden.dtype, generator=rng_sf,
            )
            t_span_sf = torch.linspace(1, 0, 11, device=device, dtype=dtype)
            t_span_sf = t_span_sf + 1.0 * (torch.cos(torch.pi / 2 * t_span_sf) - 1 + t_span_sf)
            t_val_sf = t_span_sf[1:2]  # First real step (step 2, after zero-init)
            dt_val_sf = torch.tensor([0.0], device=device, dtype=dtype)
            # Zero cond for first step
            zero_cond_sf = torch.zeros(1, cfm.in_channels, model.patch_size, device=device, dtype=dtype)
            # Run the full estimator forward (this includes projections + 12 layers + out_proj)
            vel_sf = cfm.estimator(z_sf, dit_hidden, t_val_sf, zero_cond_sf, dt_val_sf)
            # vel_sf is [B, C=64, T=4] channels-first
            results["dit_single_fwd"] = vel_sf[0].cpu().float().numpy()

        # CFM solve (manually to capture internals)
        prefix_feat_cond = audio_feat[:, -1, ...].to(dtype)
        cfm = model.feat_decoder

        # Initial noise — use fixed seed for reproducibility
        rng = torch.Generator(device=device)
        rng.manual_seed(seed)
        z = torch.randn(
            (B, cfm.in_channels, model.patch_size),
            device=device, dtype=dit_hidden.dtype,
            generator=rng,
        )
        if "cfm_step0_z" in stages:
            results["cfm_step0_z"] = z[0].cpu().float().numpy()

        # Run CFM solve for step 0
        torch.manual_seed(seed)
        pred_feat = cfm(
            mu=dit_hidden,
            patch_size=model.patch_size,
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
