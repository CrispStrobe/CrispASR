"""Decode CrispASR's saved per-frame latents through the official VibeVoice
acoustic decoder. If the click is in our decoder, this run produces clean
output; if the click is in our latents, this run reproduces it.

Usage:
  python tools/decode_latents_official.py \\
      --latent-dir .local/issue40/cpp_dump_long_emma/  \\
      --output     /tmp/our_latents_official_decoder.wav
"""
import argparse
import sys
import wave
from pathlib import Path

import numpy as np
import torch

import vibevoice  # noqa
from vibevoice.modular.modeling_vibevoice_streaming_inference import (
    VibeVoiceStreamingForConditionalGenerationInference,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--latent-dir", required=True,
                    help="Directory of perframe_latent_f<NNN>.bin files (float32, vae_dim each)")
    ap.add_argument("--output", required=True, help="Output WAV path")
    ap.add_argument("--model", default="microsoft/VibeVoice-Realtime-0.5B")
    ap.add_argument("--vae-dim", type=int, default=64)
    args = ap.parse_args()

    # Load all perframe_latent_f*.bin in order
    latent_dir = Path(args.latent_dir)
    files = sorted(latent_dir.glob("perframe_latent_f*.bin"))
    if not files:
        print(f"no perframe_latent_f*.bin in {latent_dir}", file=sys.stderr)
        return 1

    frames = []
    for fp in files:
        arr = np.fromfile(fp, dtype=np.float32)
        if arr.size != args.vae_dim:
            print(f"warning: {fp.name} has {arr.size} floats, expected {args.vae_dim}")
        frames.append(arr.reshape(-1))
    latents = np.stack(frames, axis=0)  # (N, vae_dim)
    print(f"loaded {len(frames)} frames of latents from {latent_dir}, shape {latents.shape}")

    # Load model (F32, CPU) — only need .model.acoustic_tokenizer.decoder
    print(f"loading {args.model} ...")
    model = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
        args.model, dtype=torch.float32, device_map="cpu"
    )
    model.eval()

    sf = model.model.speech_scaling_factor.item()
    bf = model.model.speech_bias_factor.item()
    print(f"  speech_scaling={sf}  speech_bias={bf}")

    # Apply scaling: scaled = latent / scaling - bias  (matches inference path)
    scaled = latents / sf - bf
    # Decoder expects (B, C, T) — our latents are (T, C), so transpose
    scaled_pt = torch.from_numpy(scaled).T.unsqueeze(0).float()  # (1, C, T)
    print(f"  scaled latents -> {tuple(scaled_pt.shape)}")

    # Decode through the OFFICIAL decoder. The acoustic_tokenizer.decode
    # method takes (1, C, T), runs forward + head, returns (1, 1, T*hop).
    with torch.no_grad():
        pcm = model.model.acoustic_tokenizer.decode(scaled_pt)
    print(f"  decoder output -> {tuple(pcm.shape)}")
    audio = pcm.detach().cpu().float().squeeze().numpy()  # (T_pcm,)

    # Save
    audio_clip = np.clip(audio, -1, 1)
    samples = (audio_clip * 32767).astype(np.int16)
    sr = 24000
    with wave.open(args.output, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(samples.tobytes())
    print(f"  wrote {len(samples)} samples = {len(samples)/sr:.2f}s -> {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
