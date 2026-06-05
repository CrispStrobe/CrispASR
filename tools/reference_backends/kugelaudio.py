"""KugelAudio-0-Open TTS reference dump for crispasr-diff.

Captures intermediate activations at each pipeline stage:
  - text_token_ids: tokenized prompt
  - lm_hidden_last: LM hidden state at last position (diffusion condition)
  - pred_t_emb_step0: timestep embedding at diffusion step 0
  - pred_output_step0: prediction head output at step 0
  - diffusion_latent: final denoised latent after 20 DPM-Solver++ steps
  - scaled_latent: after unscaling (z / scaling - bias)
  - decoded_audio: acoustic decoder output PCM

Usage:
  python tools/dump_reference.py --backend kugelaudio \\
      --model-dir kugelaudio/kugelaudio-0-open \\
      --audio samples/jfk.wav \\
      --output /tmp/kugelaudio-ref.gguf

Env vars:
  KUGELAUDIO_TEXT     synth text (default: "Hello, this is a test.")
  KUGELAUDIO_SEED     RNG seed (default: 42)
  KUGELAUDIO_STEPS    diffusion steps (default: 20)
  KUGELAUDIO_CFG      cfg scale (default: 3.0)
"""

import os
import numpy as np
from pathlib import Path

DEFAULT_STAGES = [
    "text_token_ids",
    "lm_hidden_last",
    "pred_t_emb_step0",
    "pred_cond_step0",
    "pred_output_step0",
    "diffusion_latent",
    "scaled_latent",
    "decoded_audio",
    # Per-diffusion-step captures (diagnostic)
    "diff_step0_noisy",
    "diff_step0_denoised",
]


def dump(model_dir: Path, audio: np.ndarray, stages: set, **kwargs) -> dict:
    """Run KugelAudio TTS and capture intermediate tensors."""
    import torch

    syn_text = os.environ.get("KUGELAUDIO_TEXT", "Hello, this is a test of the speech synthesis system.")
    seed = int(os.environ.get("KUGELAUDIO_SEED", "42"))
    num_steps = int(os.environ.get("KUGELAUDIO_STEPS", "20"))
    cfg_scale = float(os.environ.get("KUGELAUDIO_CFG", "3.0"))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    print(f"kugelaudio ref: text={syn_text!r} seed={seed} steps={num_steps} cfg={cfg_scale}")
    print(f"kugelaudio ref: device={device} dtype={dtype}")

    # Load model
    try:
        from kugelaudio_open.models import KugelAudioForConditionalGenerationInference
        from kugelaudio_open.processors import KugelAudioProcessor
    except ImportError:
        raise RuntimeError("kugelaudio_open package required. pip install kugelaudio-open")

    model = KugelAudioForConditionalGenerationInference.from_pretrained(
        str(model_dir), torch_dtype=dtype,
    ).to(device)
    model.eval()
    processor = KugelAudioProcessor.from_pretrained(str(model_dir))

    config = model.config
    results = {}
    # Metadata stored as 1-element float arrays (dump_reference.py expects ndarrays)
    results["kugelaudio_syn_text"] = np.array([0.0], dtype=np.float32)  # placeholder for GGUF meta string

    # Prepare inputs
    inputs = processor(text=syn_text, return_tensors="pt")
    text_ids = inputs["text_ids"].to(device)

    if "text_token_ids" in stages:
        results["text_token_ids"] = text_ids[0].cpu().numpy().astype(np.float32)

    # Hook captures
    captures = {}

    def make_hook(name):
        def hook_fn(module, inp, out):
            if isinstance(out, tuple):
                captures[name] = out[0].detach().cpu().float()
            elif hasattr(out, "last_hidden_state"):
                captures[name] = out.last_hidden_state.detach().cpu().float()
            else:
                captures[name] = out.detach().cpu().float()
        return hook_fn

    hooks = []
    hooks.append(model.model.language_model.register_forward_hook(make_hook("lm_output")))
    hooks.append(model.model.prediction_head.t_embedder.register_forward_hook(make_hook("t_embedder")))
    hooks.append(model.model.prediction_head.cond_proj.register_forward_hook(make_hook("cond_proj")))
    hooks.append(model.model.prediction_head.register_forward_hook(make_hook("pred_output")))

    # Intercept diffusion loop
    original_sample = model.sample_speech_tokens
    diff_captures = {}

    @torch.no_grad()
    def sample_instrumented(condition, neg_condition, cfg_scale_=3.0):
        scheduler = model.model.noise_scheduler
        scheduler.set_timesteps(model.ddpm_inference_steps)

        diff_captures["condition"] = condition.detach().cpu().float().numpy()

        if cfg_scale_ == 1.0:
            speech = torch.randn(condition.shape[0], config.acoustic_vae_dim).to(condition)
            diff_captures["noise_init"] = speech.detach().cpu().float().numpy()

            for step_idx, t in enumerate(scheduler.timesteps):
                eps = model.model.prediction_head(
                    speech, t.repeat(speech.shape[0]).to(speech), condition=condition
                )
                if step_idx == 0:
                    diff_captures["step0_noisy"] = speech.detach().cpu().float().numpy()
                    diff_captures["step0_pred"] = eps.detach().cpu().float().numpy()

                result = scheduler.step(eps, t, speech)
                speech = result.prev_sample

                if step_idx == 0:
                    diff_captures["step0_denoised"] = speech.detach().cpu().float().numpy()

            diff_captures["final_latent"] = speech.detach().cpu().float().numpy()
            return speech

        # CFG path
        combined_condition = torch.cat([condition, neg_condition], dim=0).to(
            model.model.prediction_head.device
        )
        speech = torch.randn(combined_condition.shape[0], config.acoustic_vae_dim).to(
            combined_condition
        )
        diff_captures["noise_init"] = speech[:len(speech)//2].detach().cpu().float().numpy()

        for step_idx, t in enumerate(scheduler.timesteps):
            half = speech[:len(speech) // 2]
            combined = torch.cat([half, half], dim=0)
            eps = model.model.prediction_head(
                combined, t.repeat(combined.shape[0]).to(combined),
                condition=combined_condition
            )
            cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
            half_eps = uncond_eps + cfg_scale_ * (cond_eps - uncond_eps)
            eps_combined = torch.cat([half_eps, half_eps], dim=0)

            if step_idx == 0:
                diff_captures["step0_noisy"] = half.detach().cpu().float().numpy()
                diff_captures["step0_pred_cond"] = cond_eps.detach().cpu().float().numpy()
                diff_captures["step0_pred_uncond"] = uncond_eps.detach().cpu().float().numpy()
                diff_captures["step0_pred_cfg"] = half_eps.detach().cpu().float().numpy()

            result = scheduler.step(eps_combined, t, speech)
            speech = result.prev_sample

            if step_idx == 0:
                diff_captures["step0_denoised"] = speech[:len(speech)//2].detach().cpu().float().numpy()

        final = speech[:len(speech) // 2]
        diff_captures["final_latent"] = final.detach().cpu().float().numpy()
        return final

    model.sample_speech_tokens = sample_instrumented

    # Run generation
    model.set_ddpm_inference_steps(num_steps)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            cfg_scale=cfg_scale,
            max_new_tokens=2048,
            do_sample=False,
            show_progress=True,
        )
    model.sample_speech_tokens = original_sample

    # Collect results
    if "lm_hidden_last" in stages and "lm_output" in captures:
        h = captures["lm_output"][0]  # [T, d_lm]
        results["lm_hidden_last"] = h[-1].numpy()  # last token

    if "pred_t_emb_step0" in stages and "t_embedder" in captures:
        results["pred_t_emb_step0"] = captures["t_embedder"].numpy().flatten()

    if "pred_cond_step0" in stages and "cond_proj" in captures:
        results["pred_cond_step0"] = captures["cond_proj"].numpy().flatten()

    if "pred_output_step0" in stages and "pred_output" in captures:
        results["pred_output_step0"] = captures["pred_output"].numpy().flatten()

    if "diff_step0_noisy" in stages and "step0_noisy" in diff_captures:
        results["diff_step0_noisy"] = diff_captures["step0_noisy"].flatten()

    if "diff_step0_denoised" in stages and "step0_denoised" in diff_captures:
        results["diff_step0_denoised"] = diff_captures["step0_denoised"].flatten()

    if "diffusion_latent" in stages and "final_latent" in diff_captures:
        results["diffusion_latent"] = diff_captures["final_latent"].flatten()

    if "scaled_latent" in stages and "final_latent" in diff_captures:
        sf = model.speech_scaling_factor.cpu().float().item()
        bf = model.speech_bias_factor.cpu().float().item()
        lat = diff_captures["final_latent"].flatten()
        results["scaled_latent"] = lat / sf - bf

    if "decoded_audio" in stages and outputs.speech_outputs and outputs.speech_outputs[0] is not None:
        results["decoded_audio"] = outputs.speech_outputs[0].cpu().float().numpy().flatten()

    # Cleanup hooks
    for h in hooks:
        h.remove()

    return results
