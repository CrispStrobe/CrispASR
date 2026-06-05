"""F5-TTS reference dump backend for crispasr-diff.

Captures per-stage activations from the SWivid/F5-TTS DiT-based
flow-matching TTS model, including the Vocos mel-24khz vocoder,
so the C++ native implementation can be validated tensor-by-tensor.

Pipeline stages captured:
  text_tokens          — character-level token IDs (int32)
  text_embed           — after TextEmbedding (text_dim=512, with ConvNeXtV2)
  time_embed           — sinusoidal timestep embedding at first ODE step
  input_embed          — after InputEmbedding proj + ConvPosEmbed (first step)
  dit_layer_{0..21}    — DiT block output after each of 22 layers (first step)
  dit_output           — final DiT output before ODE step update (first step)
  ode_step_{0,8,16,24,31} — x state after selected Euler steps
  ref_mel              — mel spectrogram of reference audio (T_ref, 100)
  conditioning_input   — zero-padded ref mel (T_total, 100)
  vocos_input          — mel going into Vocos vocoder (100, T_gen)
  vocos_backbone_out   — Vocos backbone output before ISTFTHead
  audio                — final 24 kHz PCM waveform

Environment variables:
  F5_TTS_SYN_TEXT  — text to synthesize (default: "Hello world.")
  F5_TTS_REF_TEXT  — reference audio transcript (default: "")
  F5_TTS_SEED      — random seed (default: 42)
  F5_TTS_STEPS     — ODE steps (default: 32)
  F5_TTS_CFG       — CFG strength (default: 2.0)
  F5_TTS_SWAY      — sway sampling coefficient (default: -1.0)
  F5_TTS_VOCOS_DIR — path to vocos model dir (default: <model_dir>/vocos)

Usage:
    python tools/dump_reference.py --backend f5-tts \\
        --model-dir /mnt/storage/f5-tts \\
        --audio samples/jfk.wav \\
        --output /mnt/storage/f5-tts/f5-tts-ref.gguf
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, Set

import numpy as np
import torch
import torchaudio

DEFAULT_STAGES = [
    "text_tokens",
    "text_embed",
    "time_embed",
    "input_embed",
    "ref_mel",
    "conditioning_input",
    "dit_layer_0",
    "dit_layer_5",
    "dit_layer_10",
    "dit_layer_15",
    "dit_layer_21",
    "dit_output",
    "ode_step_0",
    "ode_step_8",
    "ode_step_16",
    "ode_step_24",
    "ode_step_31",
    "vocos_input",
    "vocos_backbone_out",
    "audio",
]


def dump(model_dir: Path, audio: np.ndarray, stages: Set[str],
         max_new_tokens: int = 0) -> Dict[str, np.ndarray]:
    """Run F5-TTS reference forward pass and capture intermediate tensors."""
    from importlib.resources import files as pkg_files

    from f5_tts.model import CFM
    from f5_tts.model.backbones.dit import DiT
    from f5_tts.model.utils import convert_char_to_pinyin, get_tokenizer
    from vocos import Vocos

    device = "cpu"  # reproducible, no GPU needed for diff

    # --- Config ---
    syn_text = os.environ.get("F5_TTS_SYN_TEXT", "Hello world.")
    ref_text = os.environ.get("F5_TTS_REF_TEXT", "")
    seed = int(os.environ.get("F5_TTS_SEED", "42"))
    n_steps = int(os.environ.get("F5_TTS_STEPS", "32"))
    cfg_strength = float(os.environ.get("F5_TTS_CFG", "2.0"))
    sway_coef = float(os.environ.get("F5_TTS_SWAY", "-1.0"))
    vocos_dir = os.environ.get("F5_TTS_VOCOS_DIR",
                               str(model_dir / "vocos"))

    # --- Mel spec config ---
    target_sample_rate = 24000
    n_mel_channels = 100
    hop_length = 256
    win_length = 1024
    n_fft = 1024

    # --- Load vocab ---
    vocab_file = str(pkg_files("f5_tts").joinpath("infer/examples/vocab.txt"))
    vocab_char_map, vocab_size = get_tokenizer(vocab_file, "custom")
    print(f"  vocab_size={vocab_size}")

    # --- Load F5-TTS model ---
    model_cfg = dict(
        dim=1024,
        depth=22,
        heads=16,
        ff_mult=2,
        text_dim=512,
        text_mask_padding=True,
        qk_norm=None,
        conv_layers=4,
        pe_attn_head=None,
        attn_backend="torch",
        attn_mask_enabled=False,
    )
    model = CFM(
        transformer=DiT(**model_cfg, text_num_embeds=vocab_size, mel_dim=n_mel_channels),
        mel_spec_kwargs=dict(
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            n_mel_channels=n_mel_channels,
            target_sample_rate=target_sample_rate,
            mel_spec_type="vocos",
        ),
        odeint_kwargs=dict(method="euler"),
        vocab_char_map=vocab_char_map,
    ).to(device)

    # Load weights
    ckpt_path = model_dir / "F5TTS_v1_Base" / "model_1250000.safetensors"
    if not ckpt_path.exists():
        ckpt_path = model_dir / "model_1250000.safetensors"
    assert ckpt_path.exists(), f"Model not found: {ckpt_path}"

    from safetensors.torch import load_file
    state_dict = load_file(str(ckpt_path), device=device)
    clean_sd = {k.replace("ema_model.", ""): v for k, v in state_dict.items()
                if k not in ["initted", "step"]}
    for key in ["mel_spec.mel_stft.mel_scale.fb", "mel_spec.mel_stft.spectrogram.window"]:
        clean_sd.pop(key, None)
    model.load_state_dict(clean_sd)
    model.eval()
    del state_dict, clean_sd
    print(f"  Loaded F5-TTS from {ckpt_path}")

    # --- Load Vocos ---
    vocos_config = os.path.join(vocos_dir, "config.yaml")
    vocos_model_path = os.path.join(vocos_dir, "pytorch_model.bin")
    vocoder = Vocos.from_hparams(vocos_config)
    voc_sd = torch.load(vocos_model_path, map_location="cpu", weights_only=True)
    vocoder.load_state_dict(voc_sd)
    vocoder.eval()
    del voc_sd
    print(f"  Loaded Vocos from {vocos_dir}")

    # --- Prepare reference audio ---
    # dump_reference.py provides 16kHz mono. F5-TTS needs 24kHz.
    audio_t = torch.from_numpy(audio).unsqueeze(0)  # (1, N)
    resampler = torchaudio.transforms.Resample(16000, target_sample_rate)
    audio_24k = resampler(audio_t)
    # Normalize RMS
    rms = torch.sqrt(torch.mean(torch.square(audio_24k)))
    target_rms = 0.1
    if rms < target_rms:
        audio_24k = audio_24k * target_rms / rms
    audio_24k = audio_24k.to(device)

    # --- Prepare text ---
    if ref_text and not ref_text.endswith(". ") and not ref_text.endswith("。"):
        if ref_text.endswith("."):
            ref_text += " "
        else:
            ref_text += ". "
    full_text = ref_text + syn_text
    text_list = [full_text]
    final_text_list = convert_char_to_pinyin(text_list)
    print(f"  text: {final_text_list[0][:80]}...")

    captures: Dict[str, np.ndarray] = {}
    transformer = model.transformer

    # --- Tokenize (for text_tokens capture) ---
    from f5_tts.model.utils import list_str_to_idx
    text_tensor = list_str_to_idx(final_text_list, vocab_char_map).to(device)
    if "text_tokens" in stages:
        captures["text_tokens"] = text_tensor[0].numpy().astype(np.int32)

    # --- Compute reference mel for capture ---
    with torch.no_grad():
        ref_mel = model.mel_spec(audio_24k)  # (1, 100, T_ref)
        ref_mel = ref_mel.permute(0, 2, 1)  # (1, T_ref, 100)
    ref_audio_len = ref_mel.shape[1]
    if "ref_mel" in stages:
        captures["ref_mel"] = ref_mel[0].numpy()

    # --- Hook infrastructure ---
    # step_counter tracks ODE step index (each call to transformer.forward
    # inside the ODE solver increments it). CFG mode calls forward ONCE with
    # batch=2 (cond+uncond packed) per ODE step.
    step_counter = [0]
    first_step_captures = {}
    dit_layer_captures = {}
    hooks = []

    # Hook: text_embed (captured on first forward call, cond path)
    if "text_embed" in stages:
        def text_embed_hook(module, inp, output):
            if "text_embed" not in first_step_captures:
                first_step_captures["text_embed"] = output.detach().cpu().float()
        hooks.append(transformer.text_embed.register_forward_hook(text_embed_hook))

    # Hook: time_embed
    if "time_embed" in stages:
        def time_embed_hook(module, inp, output):
            if "time_embed" not in first_step_captures:
                first_step_captures["time_embed"] = output.detach().cpu().float()
        hooks.append(transformer.time_embed.register_forward_hook(time_embed_hook))

    # Hook: input_embed (captured on first step, conditioned path)
    if "input_embed" in stages:
        def input_embed_hook(module, inp, output):
            if step_counter[0] == 0 and "input_embed" not in first_step_captures:
                first_step_captures["input_embed"] = output.detach().cpu().float()
        hooks.append(transformer.input_embed.register_forward_hook(input_embed_hook))

    # Hook: DiT layers (captured on first ODE step)
    for i, block in enumerate(transformer.transformer_blocks):
        stage_name = f"dit_layer_{i}"
        if stage_name in stages:
            def make_hook(name):
                def hook(module, inp, output):
                    if step_counter[0] == 0 and name not in dit_layer_captures:
                        dit_layer_captures[name] = output.detach().cpu().float()
                return hook
            hooks.append(block.register_forward_hook(make_hook(stage_name)))

    # Monkey-patch transformer.forward to capture dit_output and count steps
    orig_forward = transformer.forward

    def patched_forward(*args, **kwargs):
        output = orig_forward(*args, **kwargs)
        if step_counter[0] == 0 and "dit_output" in stages:
            # cfg_infer packs (cond, uncond) → 2*batch. Take first.
            if output.shape[0] > 1:
                first_step_captures["dit_output"] = output[:1].detach().cpu().float()
            else:
                first_step_captures["dit_output"] = output.detach().cpu().float()
        step_counter[0] += 1
        return output

    transformer.forward = patched_forward

    # --- Compute duration ---
    ref_text_len = max(len(ref_text.encode("utf-8")), 1)
    gen_text_len = len(syn_text.encode("utf-8"))
    duration = ref_audio_len + int(ref_audio_len / ref_text_len * gen_text_len)
    print(f"  duration={duration} (ref={ref_audio_len}, gen_text_len={gen_text_len})")

    # --- Run full synthesis ---
    with torch.no_grad():
        generated, trajectory = model.sample(
            cond=audio_24k,
            text=final_text_list,
            duration=duration,
            steps=n_steps,
            cfg_strength=cfg_strength,
            sway_sampling_coef=sway_coef,
            seed=seed,
            use_epss=True,
        )

    # --- Restore hooks ---
    transformer.forward = orig_forward
    for h in hooks:
        h.remove()

    # --- Extract captures from hooks ---
    for name, tensor in first_step_captures.items():
        if name == "text_embed":
            # text_embed: first call is cond (drop_text=False), shape (1, T, 512)
            captures[name] = tensor[0].numpy()
        elif name == "time_embed":
            captures[name] = tensor[0].numpy()
        elif name == "input_embed":
            # cfg_infer=True: packed (2, T, 1024). Take conditioned half.
            if tensor.shape[0] > 1:
                captures[name] = tensor[0].numpy()
            else:
                captures[name] = tensor[0].numpy()
        elif name == "dit_output":
            captures[name] = tensor[0].numpy()

    for name, tensor in dit_layer_captures.items():
        # cfg_infer=True: (2, T, 1024). Take conditioned half.
        if tensor.shape[0] > 1:
            captures[name] = tensor[0].numpy()
        else:
            captures[name] = tensor[0].numpy()

    # --- ODE trajectory captures ---
    if trajectory is not None:
        for step_idx in range(trajectory.shape[0]):
            stage_name = f"ode_step_{step_idx}"
            if stage_name in stages:
                captures[stage_name] = trajectory[step_idx, 0].numpy()

    # --- Conditioning input ---
    total_duration = generated.shape[1]
    if "conditioning_input" in stages:
        with torch.no_grad():
            cond_padded = torch.nn.functional.pad(
                ref_mel, (0, 0, 0, total_duration - ref_audio_len), value=0.0
            )
        captures["conditioning_input"] = cond_padded[0].numpy()

    # --- Vocoder ---
    with torch.no_grad():
        generated = generated.to(torch.float32)
        gen_mel = generated[:, ref_audio_len:, :]  # (1, T_gen, 100)
        gen_mel_t = gen_mel.permute(0, 2, 1)  # (1, 100, T_gen)

        if "vocos_input" in stages:
            captures["vocos_input"] = gen_mel_t[0].numpy()

        # Vocos backbone hook
        if "vocos_backbone_out" in stages:
            backbone_out = [None]
            def voc_hook(module, inp, output):
                backbone_out[0] = output.detach().cpu().float()
            h = vocoder.backbone.register_forward_hook(voc_hook)
            audio_out = vocoder.decode(gen_mel_t)
            h.remove()
            if backbone_out[0] is not None:
                captures["vocos_backbone_out"] = backbone_out[0][0].numpy()
        else:
            audio_out = vocoder.decode(gen_mel_t)

        if "audio" in stages:
            captures["audio"] = audio_out.squeeze().numpy()

    print(f"  Generated audio: {captures.get('audio', np.array([])).shape}")
    print(f"  Duration: {total_duration} frames, Ref: {ref_audio_len}, Gen: {total_duration - ref_audio_len}")
    print(f"  ODE steps tracked: {step_counter[0]}")

    return captures
