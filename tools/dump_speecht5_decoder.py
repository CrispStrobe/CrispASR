#!/usr/bin/env python
"""
SpeechT5 decoder-level reference dump for diff-testing the C++ backend.

Runs the manual AR decoder loop, dumping per-step intermediates that can be
compared 1:1 with the C++ SPEECHT5_DUMP_DIR output.

Usage:
    SPEECHT5_DUMP_DIR=/mnt/storage/speecht5/ref_dec python tools/dump_speecht5_decoder.py \
        --text "Hello." --speaker /mnt/storage/speecht5/speaker.bin --max-steps 5

Dumps per step:
    step{N}_prenet.npy      - (1, 768) prenet output (after speaker proj)
    step{N}_hidden.npy      - (1, 768) decoder hidden (after 6 layers, before feat_out)
    step{N}_mel.npy         - (160,)   feat_out output (reduction*mel_bins)
    step{N}_self_k_L{l}.npy - self-attn K for layer l at step N
    step{N}_self_v_L{l}.npy - self-attn V for layer l at step N

Final:
    mel_pre_postnet.npy     - (T_mel, 80)
    mel_post_postnet.npy    - (T_mel, 80)
    ref_output.wav          - 16 kHz PCM
"""

from __future__ import annotations

import argparse
import os
import struct
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("HF_HOME", "/mnt/storage/huggingface")

import torch
import torch.nn as nn
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan


def load_speaker(path: str | None, seed: int = 42) -> torch.Tensor:
    """Load speaker embedding from .bin file or generate one."""
    if path and Path(path).exists():
        raw = Path(path).read_bytes()
        spk = np.frombuffer(raw, dtype=np.float32).copy()
        return torch.from_numpy(spk).unsqueeze(0)
    torch.manual_seed(seed)
    spk = torch.randn(1, 512)
    return spk


def write_wav(path: Path, pcm: np.ndarray, sr: int = 16000):
    """Write minimal WAV."""
    pcm16 = np.clip(pcm * 32767, -32768, 32767).astype(np.int16)
    data = pcm16.tobytes()
    with open(path, "wb") as f:
        f.write(b"RIFF")
        f.write(struct.pack("<I", 36 + len(data)))
        f.write(b"WAVE")
        f.write(b"fmt ")
        f.write(struct.pack("<IHHIIHH", 16, 1, 1, sr, sr * 2, 2, 16))
        f.write(b"data")
        f.write(struct.pack("<I", len(data)))
        f.write(data)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", default="Hello.")
    ap.add_argument("--speaker", default=None, help="Path to 512-d F32 .bin file")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-steps", type=int, default=0, help="0=auto")
    ap.add_argument("--no-dropout", action="store_true", default=True,
                    help="Disable prenet dropout (default: yes, for deterministic comparison)")
    ap.add_argument("--with-dropout", action="store_true", help="Enable prenet dropout")
    args = ap.parse_args()

    dump_dir = os.environ.get("SPEECHT5_DUMP_DIR", "")
    if not dump_dir:
        sys.exit("Set SPEECHT5_DUMP_DIR env var to output directory")
    dump_dir = Path(dump_dir)
    dump_dir.mkdir(parents=True, exist_ok=True)

    no_dropout = not args.with_dropout

    print(f"Loading models...", file=sys.stderr)
    processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
    model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
    vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")
    model.eval()
    vocoder.eval()

    if no_dropout:
        prenet = model.speecht5.decoder.prenet
        prenet._consistent_dropout = lambda x, p: x
        prenet.encode_positions.dropout = nn.Dropout(0.0)
        print("Prenet dropout DISABLED for deterministic comparison", file=sys.stderr)

    speaker_embedding = load_speaker(args.speaker, args.seed)
    print(f"Speaker embedding: shape={speaker_embedding.shape}, norm={speaker_embedding.norm():.4f}", file=sys.stderr)

    # Save speaker embedding for C++ to use
    spk_np = speaker_embedding.numpy().flatten()
    spk_path = dump_dir / "speaker.bin"
    with open(spk_path, "wb") as f:
        f.write(spk_np.tobytes())
    print(f"Speaker saved to {spk_path}", file=sys.stderr)

    inputs = processor(text=args.text, return_tensors="pt")
    input_ids = inputs["input_ids"]
    print(f"Text: {args.text!r}", file=sys.stderr)
    print(f"Token IDs: {input_ids[0].tolist()}", file=sys.stderr)

    config = model.config
    reduction_factor = config.reduction_factor
    num_mel_bins = config.num_mel_bins

    with torch.no_grad():
        # Encoder
        encoder_attention_mask = 1 - (input_ids == config.pad_token_id).int()
        encoder_out = model.speecht5.encoder(
            input_values=input_ids,
            attention_mask=encoder_attention_mask,
            return_dict=True,
        )
        encoder_hidden = encoder_out.last_hidden_state
        np.save(dump_dir / "encoder_output.npy", encoder_hidden.cpu().numpy())
        print(f"Encoder: {encoder_hidden.shape}, first 4: {encoder_hidden[0, 0, :4].tolist()}", file=sys.stderr)

        # Manual AR decoder loop
        bsz = 1
        output_sequence = torch.zeros(bsz, 1, num_mel_bins)
        past_key_values = None
        spectrogram = []

        max_steps = args.max_steps if args.max_steps > 0 else \
            int(encoder_hidden.size(1) * 20.0 / reduction_factor)

        # Hook to capture per-layer self-attention K/V
        layer_kv = {}

        def make_self_attn_hook(layer_idx):
            def hook(module, input, output):
                # output is (attn_output, attn_weights, past_key_value)
                if isinstance(output, tuple) and len(output) >= 3:
                    pkv = output[2]  # (key, value) tuple
                    if pkv is not None:
                        layer_kv[layer_idx] = (pkv[0].cpu(), pkv[1].cpu())
            return hook

        # Register hooks on decoder self-attention layers
        hooks = []
        wrapped_dec = model.speecht5.decoder.wrapped_decoder
        for i, layer in enumerate(wrapped_dec.layers):
            h = layer.self_attn.register_forward_hook(make_self_attn_hook(i))
            hooks.append(h)

        for step in range(max_steps):
            layer_kv.clear()

            # Prenet on full output sequence
            decoder_hidden_states = model.speecht5.decoder.prenet(
                output_sequence, speaker_embedding
            )

            # Dump prenet output for current step (last position only)
            prenet_last = decoder_hidden_states[:, -1:, :]  # (1, 1, 768)
            np.save(dump_dir / f"step{step}_prenet.npy",
                    prenet_last.squeeze(0).cpu().numpy())  # (1, 768)

            # Decoder (only last position, with KV cache)
            decoder_out = model.speecht5.decoder.wrapped_decoder(
                hidden_states=decoder_hidden_states[:, -1:],
                attention_mask=None,
                encoder_hidden_states=encoder_hidden,
                encoder_attention_mask=encoder_attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
                output_attentions=False,
                return_dict=True,
            )

            last_hidden = decoder_out.last_hidden_state  # (1, 1, 768)
            past_key_values = decoder_out.past_key_values

            # Dump decoder hidden
            np.save(dump_dir / f"step{step}_hidden.npy",
                    last_hidden.squeeze(0).cpu().numpy())  # (1, 768)

            # Dump self-attn K/V from hooks
            for l_idx, (k, v) in layer_kv.items():
                # k shape: (1, n_heads, T_kv, head_dim)
                if step <= 2:
                    np.save(dump_dir / f"step{step}_self_k_L{l_idx}.npy", k.numpy())
                    np.save(dump_dir / f"step{step}_self_v_L{l_idx}.npy", v.numpy())

            # feat_out
            last_squeeze = last_hidden.squeeze(1)  # (1, 768)
            spectrum = model.speech_decoder_postnet.feat_out(last_squeeze)  # (1, 160)
            mel_arr = spectrum.cpu().numpy().flatten()
            np.save(dump_dir / f"step{step}_mel.npy", mel_arr)

            # Reshape and accumulate
            spectrum_reshaped = spectrum.view(bsz, reduction_factor, num_mel_bins)
            spectrogram.append(spectrum_reshaped)

            # Update output sequence
            new_frame = spectrum_reshaped[:, -1, :].view(bsz, 1, num_mel_bins)
            output_sequence = torch.cat((output_sequence, new_frame), dim=1)

            # Stop probability
            prob = torch.sigmoid(model.speech_decoder_postnet.prob_out(last_squeeze))
            prob_sum = torch.sum(prob, dim=-1).item()

            if step < 5 or step % 20 == 0:
                print(f"  step {step}: stop_prob={prob_sum:.4f} mel_rms={mel_arr.std():.4f} "
                      f"mel[:4]={mel_arr[:4].tolist()}", file=sys.stderr)

            if prob_sum >= 0.5 and step > 0:
                print(f"Stopped at step {step+1} (prob={prob_sum:.3f})", file=sys.stderr)
                break

        # Remove hooks
        for h in hooks:
            h.remove()

        # Stack mel
        all_mel = torch.stack(spectrogram).transpose(0, 1).flatten(1, 2)  # (1, T_mel, 80)
        mel_pre = all_mel.squeeze(0).cpu().numpy()
        np.save(dump_dir / "mel_pre_postnet.npy", mel_pre)
        print(f"mel_pre_postnet: {mel_pre.shape}", file=sys.stderr)

        # Postnet
        mel_post_t = model.speech_decoder_postnet.postnet(all_mel)
        mel_post = mel_post_t.squeeze(0).cpu().numpy()
        np.save(dump_dir / "mel_post_postnet.npy", mel_post)
        print(f"mel_post_postnet: {mel_post.shape}", file=sys.stderr)

        # Vocoder
        waveform = vocoder(mel_post_t.squeeze(0))
        pcm = waveform.cpu().numpy()
        np.save(dump_dir / "vocoder_pcm.npy", pcm)
        write_wav(dump_dir / "ref_output.wav", pcm, 16000)
        print(f"vocoder_pcm: {pcm.shape} ({len(pcm)/16000:.2f}s)", file=sys.stderr)
        print(f"Saved to {dump_dir}", file=sys.stderr)


if __name__ == "__main__":
    main()
