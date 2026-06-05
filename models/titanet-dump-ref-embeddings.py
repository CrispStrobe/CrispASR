#!/usr/bin/env python3
"""Dump reference TitaNet-Large embeddings for numerical agreement testing.

Loads the NeMo model, runs it on test audio files, and saves:
  1. Reference 192-d embeddings as .npy files
  2. Reference mel spectrogram features as .npy files (for debugging)
  3. Cosine similarity matrix between all pairs

Usage:
  # Install NeMo first:
  #   pip install 'git+https://github.com/NVIDIA/NeMo.git@main#egg=nemo_toolkit[asr]'
  #
  # Then run on test WAV files:
  python models/titanet-dump-ref-embeddings.py \
      --wavs speaker_a_1.wav speaker_a_2.wav speaker_b_1.wav \
      --outdir /mnt/storage/titanet-ref/

  # Or use the two built-in test files from the HF repo:
  python models/titanet-dump-ref-embeddings.py --use-hf-samples --outdir /mnt/storage/titanet-ref/
"""

import argparse
import os
import sys

import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Dump TitaNet reference embeddings")
    parser.add_argument("--wavs", nargs="+", help="Input WAV files (16kHz mono)")
    parser.add_argument("--use-hf-samples", action="store_true", help="Use the two sample WAVs from HF repo")
    parser.add_argument("--outdir", required=True, help="Output directory for .npy files")
    parser.add_argument("--model", default="nvidia/speakerverification_en_titanet_large", help="HF model ID")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # Collect WAV files
    wav_files = []
    if args.use_hf_samples:
        from huggingface_hub import hf_hub_download
        for fname in ["an255-fash-b.wav", "cen7-fash-b.wav"]:
            wav_files.append(hf_hub_download(args.model, fname))
    elif args.wavs:
        wav_files = args.wavs
    else:
        print("ERROR: provide --wavs or --use-hf-samples")
        sys.exit(1)

    print(f"Loading NeMo model: {args.model}")
    import torch
    import nemo.collections.asr as nemo_asr

    model = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained(args.model)
    model.eval()

    embeddings = []
    names = []

    for wav_path in wav_files:
        name = os.path.splitext(os.path.basename(wav_path))[0]
        names.append(name)
        print(f"\nProcessing: {wav_path} ({name})")

        # Get embedding
        with torch.no_grad():
            emb = model.get_embedding(wav_path)
        emb_np = emb.cpu().numpy().flatten()
        print(f"  embedding shape: {emb_np.shape}, norm: {np.linalg.norm(emb_np):.6f}")
        print(f"  first 10: {emb_np[:10]}")

        # Save embedding
        np.save(os.path.join(args.outdir, f"{name}_emb.npy"), emb_np)
        embeddings.append(emb_np)

        # Also dump the preprocessed mel features for fbank debugging
        try:
            import torchaudio
            waveform, sr = torchaudio.load(wav_path)
            if sr != 16000:
                waveform = torchaudio.functional.resample(waveform, sr, 16000)
                sr = 16000
            # Save raw PCM for C++ to load
            pcm = waveform[0].numpy()
            np.save(os.path.join(args.outdir, f"{name}_pcm.npy"), pcm.astype(np.float32))
            print(f"  saved PCM: {pcm.shape} samples ({len(pcm)/16000:.2f}s)")

            # Run preprocessor to get mel features
            with torch.no_grad():
                processed, processed_len = model.preprocessor(
                    input_signal=waveform.unsqueeze(0),
                    length=torch.tensor([waveform.shape[1]])
                )
            mel_np = processed[0].cpu().numpy()  # (n_mels, T)
            np.save(os.path.join(args.outdir, f"{name}_mel.npy"), mel_np)
            print(f"  mel features: {mel_np.shape}")
        except Exception as e:
            print(f"  WARNING: could not dump mel features: {e}")

    # Print cosine similarity matrix
    print("\n=== Cosine Similarity Matrix ===")
    header = "        " + "  ".join(f"{n:>12s}" for n in names)
    print(header)
    for i, (name_i, emb_i) in enumerate(zip(names, embeddings)):
        row = f"{name_i:>8s}"
        for j, emb_j in enumerate(embeddings):
            sim = np.dot(emb_i, emb_j) / (np.linalg.norm(emb_i) * np.linalg.norm(emb_j))
            row += f"  {sim:12.6f}"
        print(row)

    print(f"\nSaved {len(embeddings)} embeddings to {args.outdir}")


if __name__ == "__main__":
    main()
