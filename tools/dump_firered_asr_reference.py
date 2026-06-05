#!/usr/bin/env python3
"""Dump FireRedASR2-AED intermediates for stage-by-stage diff-testing.

Usage:
  python tools/dump_firered_asr_reference.py \
      --model FireRedTeam/FireRedASR2-AED \
      --audio samples/jfk.wav \
      --outdir /mnt/storage/firered_ref
"""

import argparse
import os
import struct
import sys

import numpy as np
import torch


def parse_cmvn_ark(path):
    """Parse Kaldi binary CMVN ark file."""
    with open(path, "rb") as f:
        data = f.read()
    for i in range(len(data) - 3):
        if data[i : i + 3] in (b"BDM", b"BFM"):
            mtype = data[i : i + 3]
            idx = i + 3
            if data[idx] == 0x20:
                idx += 1
            assert data[idx] == 4
            idx += 1
            rows = struct.unpack("<i", data[idx : idx + 4])[0]
            idx += 4
            assert data[idx] == 4
            idx += 1
            cols = struct.unpack("<i", data[idx : idx + 4])[0]
            idx += 4
            elem_size = 8 if mtype == b"BDM" else 4
            dtype = "<f8" if mtype == b"BDM" else "<f4"
            vals = np.frombuffer(data[idx : idx + rows * cols * elem_size], dtype=dtype).reshape(rows, cols)
            count = vals[0, -1]
            mean = (vals[0, :-1] / count).astype(np.float32)
            var = vals[1, :-1] / count - mean.astype(np.float64) ** 2
            std = np.sqrt(np.maximum(var, 1e-10)).astype(np.float32)
            return mean, std
    raise ValueError("Could not parse CMVN ark")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--audio", required=True)
    parser.add_argument("--outdir", required=True)
    args = parser.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    from fireredasr.models.fireredasr_aed import FireRedAsrAed
    import kaldi_native_fbank as knf
    import scipy.io.wavfile as wav_io
    from huggingface_hub import hf_hub_download

    # Resolve paths
    if os.path.isdir(args.model):
        base = args.model
    else:
        p = hf_hub_download(args.model, "model.pth.tar")
        base = os.path.dirname(p)
        for f in ["dict.txt", "train_bpe1000.model", "cmvn.ark"]:
            hf_hub_download(args.model, f)

    # Load model
    ckpt = torch.load(os.path.join(base, "model.pth.tar"), map_location="cpu", weights_only=False)
    model = FireRedAsrAed(ckpt["args"])
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval()

    # Compute features
    sr, wav_data = wav_io.read(args.audio)
    if wav_data.dtype == np.int16:
        wav_data = wav_data.astype(np.float32) / 32768.0
    print(f"Audio: {len(wav_data)} samples @ {sr} Hz")

    opts = knf.FbankOptions()
    opts.frame_opts.samp_freq = 16000
    opts.frame_opts.frame_length_ms = 25.0
    opts.frame_opts.frame_shift_ms = 10.0
    opts.mel_opts.num_bins = 80
    fbank = knf.OnlineFbank(opts)
    fbank.accept_waveform(16000, wav_data.tolist())
    fbank.input_finished()
    n_frames = fbank.num_frames_ready
    features = np.zeros((n_frames, 80), dtype=np.float32)
    for i in range(n_frames):
        features[i] = fbank.get_frame(i)

    # CMVN
    mean, std = parse_cmvn_ark(os.path.join(base, "cmvn.ark"))
    features = (features - mean) / std
    print(f"Features: {features.shape}")
    np.save(os.path.join(args.outdir, "00_features.npy"), features)

    padded = torch.from_numpy(features).unsqueeze(0)
    lengths = torch.tensor([n_frames], dtype=torch.long)

    # =========================================================================
    # Stage 1: Encoder
    # =========================================================================
    print("\n=== Encoder ===")
    with torch.no_grad():
        enc_out, _, enc_mask = model.encoder(padded, lengths)
    print(f"Encoder output: {enc_out.shape}")
    np.save(os.path.join(args.outdir, "01_encoder_output.npy"), enc_out[0].numpy())

    # Encoder output shape: [1, T/4, 1280] after 4x subsampling

    # =========================================================================
    # Stage 2: Decoder (greedy)
    # =========================================================================
    print("\n=== Decoder ===")
    with torch.no_grad():
        results = model.transcribe(padded, lengths, beam_size=1)

    tokens = results[0][0]["yseq"].tolist()
    print(f"Tokens ({len(tokens)}): {tokens[:10]}...")
    np.save(os.path.join(args.outdir, "02_tokens.npy"), np.array(tokens))

    # Decode to text
    vocab = []
    with open(os.path.join(base, "dict.txt")) as f:
        for line in f:
            parts = line.strip().split()
            if parts:
                vocab.append(parts[0])

    text = ""
    for tid in tokens:
        if tid < len(vocab) and tid not in [0, 2, 3, 4]:
            text += vocab[tid]
    text = text.replace("\u2581", " ").strip()
    print(f"Text: \"{text}\"")

    with open(os.path.join(args.outdir, "transcript.txt"), "w") as f:
        f.write(text)

    # CTC: run manually from encoder output using checkpoint weights
    print("\n=== CTC ===")
    with torch.no_grad():
        ctc_w = ckpt["model_state_dict"]["ctc.ctc_lo.weight"]
        ctc_b = ckpt["model_state_dict"]["ctc.ctc_lo.bias"]
        ctc_out = torch.nn.functional.linear(enc_out, ctc_w.float(), ctc_b.float())
        ctc_out = torch.nn.functional.log_softmax(ctc_out, dim=-1)
    print(f"CTC output: {ctc_out.shape}")
    ctc_ids = ctc_out[0].argmax(dim=-1)
    prev = -1
    ctc_text = ""
    for t in range(ctc_ids.shape[0]):
        tid = ctc_ids[t].item()
        if tid != prev and tid != 0:
            if tid < len(vocab) and tid not in [2, 3, 4]:
                ctc_text += vocab[tid]
        prev = tid
    ctc_text = ctc_text.replace("\u2581", " ").strip()
    print(f"CTC text: \"{ctc_text}\"")

    # Save CMVN for C++ to use
    np.savez(os.path.join(args.outdir, "cmvn.npz"), mean=mean, std=std)

    print(f"\nDumped to {args.outdir}")


if __name__ == "__main__":
    main()
