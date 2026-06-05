#!/usr/bin/env python3
"""Convert VibeVoice voice prompt (.pt) to GGUF.

The voice prompt contains pre-computed KV caches that establish speaker identity.
Without it, the TTS model generates random-sounding speech.

Usage:
  python models/convert-vibevoice-voice-to-gguf.py \
      --input demo/voices/streaming_model/en-Emma_woman.pt \
      --output en-Emma_woman.gguf
"""

import argparse
import os
import sys

import numpy as np
import torch

try:
    import gguf
except ImportError:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ggml", "python"))
    import gguf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Voice prompt .pt file")
    parser.add_argument("--output", required=True, help="Output GGUF file")
    args = parser.parse_args()

    voice = torch.load(args.input, map_location="cpu", weights_only=False)

    writer = gguf.GGUFWriter(args.output, "vibevoice-voice")
    writer.add_name(os.path.basename(args.input).replace(".pt", ""))

    tensor_count = 0
    for prefix in ["lm", "tts_lm", "neg_lm", "neg_tts_lm"]:
        obj = voice[prefix]
        kv = obj.past_key_values
        n_layers = len(kv.key_cache)
        seq_len = kv.key_cache[0].shape[2]

        writer.add_uint32(f"voice.{prefix}.n_layers", n_layers)
        writer.add_uint32(f"voice.{prefix}.seq_len", seq_len)

        for il in range(n_layers):
            for kv_type, cache in [("k", kv.key_cache), ("v", kv.value_cache)]:
                # Shape: [1, n_kv_heads, seq_len, head_dim] -> squeeze batch -> [n_kv_heads, seq_len, head_dim]
                data = cache[il].squeeze(0).to(torch.float16).numpy()
                name = f"voice.{prefix}.{il}.{kv_type}"
                writer.add_tensor(name, data)
                tensor_count += 1

        print(f"  {prefix}: {n_layers} layers, seq_len={seq_len}")

    print(f"\n  total: {tensor_count} tensors")

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    sz = os.path.getsize(args.output)
    print(f"\nDone: {args.output} ({sz / 1e6:.1f} MB, {tensor_count} tensors)")


if __name__ == "__main__":
    main()
