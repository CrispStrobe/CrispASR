#!/usr/bin/env python3
"""
SpeechT5 TTS reference backend — loads microsoft/speecht5_tts +
microsoft/speecht5_hifigan from HuggingFace, runs TTS inference,
and optionally dumps per-layer intermediates for diff-testing.

Usage:
    python tools/reference_backends/speecht5_tts.py \
        --text "Hallo, wie geht es Ihnen?" \
        --model sjdata/speecht5_finetuned_common_voice_11_de \
        --output /mnt/storage/speecht5/ref_output.wav

    # With intermediate dumps:
    python tools/reference_backends/speecht5_tts.py \
        --text "Hello, how are you?" \
        --dump-dir /mnt/storage/speecht5/ref_intermediates/ \
        --output /mnt/storage/speecht5/ref_output.wav
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
    import soundfile as sf
except ImportError:
    sf = None

try:
    from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
except ImportError:
    sys.exit("pip install transformers sentencepiece")

try:
    from datasets import load_dataset
except ImportError:
    load_dataset = None


def get_speaker_embedding(speaker_id: int = 7306) -> torch.Tensor:
    """Load a speaker embedding from the CMU Arctic x-vectors dataset."""
    if load_dataset is None:
        sys.exit("pip install datasets  (needed for speaker embeddings)")
    embeddings_dataset = load_dataset(
        "Matthijs/cmu-arctic-xvectors", split="validation",
        trust_remote_code=True,
    )
    speaker_embedding = torch.tensor(
        embeddings_dataset[speaker_id]["xvector"]
    ).unsqueeze(0)
    return speaker_embedding


def run_inference(args):
    device = "cpu"

    print(f"Loading processor from {args.model}...", file=sys.stderr)
    processor = SpeechT5Processor.from_pretrained(args.model)

    print(f"Loading TTS model from {args.model}...", file=sys.stderr)
    model = SpeechT5ForTextToSpeech.from_pretrained(args.model).to(device)
    model.eval()

    print(f"Loading vocoder from {args.vocoder}...", file=sys.stderr)
    vocoder = SpeechT5HifiGan.from_pretrained(args.vocoder).to(device)
    vocoder.eval()

    # Get speaker embedding
    print("Loading speaker embedding...", file=sys.stderr)
    speaker_embedding = get_speaker_embedding(args.speaker_id).to(device)

    # Tokenize
    inputs = processor(text=args.text, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    print(f"Input tokens: {input_ids.shape} = {input_ids[0].tolist()}", file=sys.stderr)

    dump_dir = Path(args.dump_dir) if args.dump_dir else None
    if dump_dir:
        dump_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        # --- Encoder ---
        encoder_attention_mask = 1 - (input_ids == model.config.pad_token_id).int()

        # Text prenet
        encoder_prenet_out = model.speecht5.encoder.prenet(input_ids)
        if dump_dir:
            np.save(dump_dir / "encoder_prenet_output.npy",
                    encoder_prenet_out.cpu().numpy())

        # Encoder forward
        encoder_out = model.speecht5.encoder(
            input_values=input_ids,
            attention_mask=encoder_attention_mask,
            return_dict=True,
        )
        encoder_hidden = encoder_out.last_hidden_state
        if dump_dir:
            np.save(dump_dir / "encoder_output.npy",
                    encoder_hidden.cpu().numpy())
            print(f"Encoder output: {encoder_hidden.shape}", file=sys.stderr)

        # --- Full generation (with vocoder) ---
        speech = model.generate_speech(
            input_ids,
            speaker_embeddings=speaker_embedding,
            vocoder=vocoder,
        )
        waveform = speech.cpu().numpy()
        print(f"Generated waveform: {waveform.shape} samples "
              f"({len(waveform) / 16000:.2f}s at 16kHz)", file=sys.stderr)

        # --- Also generate mel spectrogram without vocoder ---
        mel_output = model.generate_speech(
            input_ids,
            speaker_embeddings=speaker_embedding,
        )
        if dump_dir:
            np.save(dump_dir / "mel_spectrogram.npy",
                    mel_output.cpu().numpy())
            print(f"Mel spectrogram: {mel_output.shape}", file=sys.stderr)

        # --- Vocoder intermediates ---
        if dump_dir:
            # Run vocoder step-by-step
            spec = mel_output.unsqueeze(0)
            if vocoder.config.normalize_before:
                spec_norm = (spec - vocoder.mean) / vocoder.scale
            else:
                spec_norm = spec
            np.save(dump_dir / "vocoder_input_normalized.npy",
                    spec_norm.cpu().numpy())

            h = spec_norm.transpose(2, 1)
            h = vocoder.conv_pre(h)
            np.save(dump_dir / "vocoder_conv_pre.npy", h.cpu().numpy())

            for i in range(vocoder.num_upsamples):
                h = torch.nn.functional.leaky_relu(h, vocoder.config.leaky_relu_slope)
                h = vocoder.upsampler[i](h)

                res_state = vocoder.resblocks[i * vocoder.num_kernels](h)
                for j in range(1, vocoder.num_kernels):
                    res_state += vocoder.resblocks[i * vocoder.num_kernels + j](h)
                h = res_state / vocoder.num_kernels

                np.save(dump_dir / f"vocoder_upsample_{i}.npy", h.cpu().numpy())

            h = torch.nn.functional.leaky_relu(h)
            h = vocoder.conv_post(h)
            h = torch.tanh(h)
            np.save(dump_dir / "vocoder_output.npy", h.cpu().numpy())

        # --- Speaker embedding ---
        if dump_dir:
            np.save(dump_dir / "speaker_embedding.npy",
                    speaker_embedding.cpu().numpy())

    # Save waveform
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if sf is not None:
            sf.write(str(out_path), waveform, 16000)
            print(f"Saved: {out_path}", file=sys.stderr)
        else:
            # Fallback: save as raw numpy
            np.save(str(out_path).replace(".wav", ".npy"), waveform)
            print(f"Saved as .npy (install soundfile for .wav): "
                  f"{str(out_path).replace('.wav', '.npy')}", file=sys.stderr)

    return waveform


def main():
    ap = argparse.ArgumentParser(description="SpeechT5 TTS reference inference")
    ap.add_argument("--text", default="Hello, how are you today?",
                    help="Text to synthesize")
    ap.add_argument("--model", default="microsoft/speecht5_tts",
                    help="HF model ID for SpeechT5 TTS")
    ap.add_argument("--vocoder", default="microsoft/speecht5_hifigan",
                    help="HF model ID for HiFi-GAN vocoder")
    ap.add_argument("--speaker-id", type=int, default=7306,
                    help="Speaker index in CMU Arctic x-vectors dataset")
    ap.add_argument("--output", "-o", default=None,
                    help="Output WAV file path")
    ap.add_argument("--dump-dir", default=None,
                    help="Directory for intermediate tensor dumps")
    args = ap.parse_args()
    run_inference(args)


if __name__ == "__main__":
    main()
