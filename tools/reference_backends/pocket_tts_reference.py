"""Pocket TTS reference backend for diff testing.

Loads the official kyutai/pocket-tts model, runs inference, and dumps
intermediate activations for comparison with the C++ runtime.

Stages dumped:
  text_tokens       - SentencePiece token ids after prepare(), int32
  text_embeddings   - LUT conditioner output, (L_text, 1024)
  backbone_out      - transformer output before flow head, (1, 1024)
  flow_net_out      - consistency head output (one-step LSD), (1, 32)
  latent_sequence   - full AR-generated latent sequence, (N_frames, 32)
  mimi_quantizer_out - quantizer projection output, (512, N_frames)
  audio_out         - final 24 kHz PCM, (T_samples,)

Environment variables:
  POCKET_TTS_LANG       language config (default: "english")
  POCKET_TTS_TEXT       text to synthesize (default: "Hello world.")
  POCKET_TTS_VOICE      voice conditioning .wav or .safetensors path
  POCKET_TTS_TEMP       temperature (default: 0.7)
  POCKET_TTS_STEPS      LSD decode steps (default: 1)
  POCKET_TTS_SEED       random seed for reproducibility
  POCKET_TTS_MAX_FRAMES max AR frames to generate (default: 50)

Usage:
  python tools/reference_backends/pocket_tts_reference.py \\
      --output-dir /mnt/storage/pocket-tts/ref-dumps
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

try:
    import torch
except ImportError:
    sys.exit("pip install torch")

try:
    from pocket_tts import TTSModel
    from pocket_tts.conditioners.base import TokenizedText
    from pocket_tts.modules.stateful_module import init_states, increment_steps
except ImportError:
    sys.exit("pip install pocket-tts")


def dump_tensor(path: Path, name: str, t: torch.Tensor | np.ndarray):
    """Save tensor as .npy for diff testing."""
    if isinstance(t, torch.Tensor):
        t = t.detach().cpu().float().numpy()
    outpath = path / f"{name}.npy"
    np.save(str(outpath), t)
    print(f"  {name}: shape={t.shape} dtype={t.dtype} -> {outpath}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    lang = os.environ.get("POCKET_TTS_LANG", "english")
    text = os.environ.get("POCKET_TTS_TEXT", "Hello world.")
    voice_path = os.environ.get("POCKET_TTS_VOICE", "")
    temp = float(os.environ.get("POCKET_TTS_TEMP", "0.7"))
    lsd_steps = int(os.environ.get("POCKET_TTS_STEPS", "1"))
    seed_str = os.environ.get("POCKET_TTS_SEED", "")
    max_frames = int(os.environ.get("POCKET_TTS_MAX_FRAMES", "50"))

    if seed_str:
        torch.manual_seed(int(seed_str))

    print(f"Loading Pocket TTS model (language={lang})...")
    model = TTSModel.load_model(language=lang, temp=temp, lsd_decode_steps=lsd_steps)
    model.eval()

    # ── Stage 1: Text tokenization ──
    print(f"\nText: '{text}'")
    prepared = model.flow_lm.conditioner.prepare(text)
    tokens = prepared.tokens
    dump_tensor(out, "text_tokens", tokens.to(torch.int32))
    print(f"  Token count: {tokens.shape[1]}")

    # ── Stage 2: Text embeddings ──
    text_embeddings = model.flow_lm.conditioner(TokenizedText(tokens))
    dump_tensor(out, "text_embeddings", text_embeddings.squeeze(0))

    # ── Stage 3: Voice conditioning (if provided) ──
    if voice_path:
        print(f"\nVoice conditioning: {voice_path}")
        model_state = model.get_state_for_audio_prompt(voice_path)
    else:
        print("\nNo voice conditioning (using init state)")
        model_state = init_states(model.flow_lm, batch_size=1, sequence_length=1000)

    # ── Stage 4: Autoregressive generation with intermediate dumps ──
    print(f"\nGenerating (max_frames={max_frames})...")

    # Prompt text
    model._run_flow_lm_and_increment_step(
        model_state=model_state,
        text_tokens=tokens,
    )

    # AR loop
    latents = []
    bos_input = torch.full(
        (1, 1, model.flow_lm.ldim),
        fill_value=float("NaN"),
        device=next(iter(model.flow_lm.parameters())).device,
        dtype=model.flow_lm.dtype,
    )
    backbone_input = bos_input

    for step in range(max_frames):
        next_latent, is_eos = model._run_flow_lm_and_increment_step(
            model_state=model_state,
            backbone_input_latents=backbone_input,
        )

        latents.append(next_latent.squeeze(0).squeeze(0))

        if step == 0:
            dump_tensor(out, "first_latent", next_latent.squeeze(0))

        if is_eos.item():
            print(f"  EOS at step {step}")
            # Generate a few more frames
            for extra in range(3):
                next_latent, _ = model._run_flow_lm_and_increment_step(
                    model_state=model_state,
                    backbone_input_latents=backbone_input,
                )
                latents.append(next_latent.squeeze(0).squeeze(0))
                backbone_input = next_latent
            break

        backbone_input = next_latent

    latent_seq = torch.stack(latents, dim=0)  # (N, 32)
    dump_tensor(out, "latent_sequence", latent_seq)
    print(f"  Generated {len(latents)} frames")

    # ── Stage 5: Mimi decode ──
    print("\nDecoding with Mimi...")

    # Denormalize: latent * emb_std + emb_mean
    denorm = latent_seq * model.flow_lm.emb_std + model.flow_lm.emb_mean
    dump_tensor(out, "latent_denormalized", denorm)

    # Quantizer projection (Conv1d: 32 -> 512)
    denorm_3d = denorm.unsqueeze(0).transpose(1, 2)  # (1, 32, N)
    quantized = model.mimi.quantizer(denorm_3d)       # (1, 512, N)
    dump_tensor(out, "mimi_quantizer_out", quantized.squeeze(0))

    # Full decode
    mimi_state = init_states(model.mimi, batch_size=1, sequence_length=10000)
    audio = model.mimi.decode_from_latent(quantized, mimi_state)
    audio_np = audio.squeeze().detach().cpu().numpy()
    dump_tensor(out, "audio_out", audio_np)

    # Save as WAV
    try:
        import scipy.io.wavfile
        wav_path = out / "output.wav"
        scipy.io.wavfile.write(str(wav_path), 24000, audio_np)
        print(f"  Saved WAV: {wav_path}")
    except ImportError:
        print("  (scipy not available, skipping WAV save)")

    # Summary
    audio_duration = len(audio_np) / 24000
    print(f"\nSummary:")
    print(f"  Text: '{text}'")
    print(f"  Frames: {len(latents)}")
    print(f"  Audio: {audio_duration:.2f}s ({len(audio_np)} samples @ 24kHz)")


if __name__ == "__main__":
    main()
