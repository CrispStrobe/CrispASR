#!/usr/bin/env python3
"""Render a Qwen3-TTS codebook stream → 24 kHz WAV via the official codec.

Reads a flat int32 stream of `T_frames * 16` codes from stdin or a file
(one int per line, OR space/comma-separated, OR binary if --binary), packs
them into shape (T, 16), feeds them through `qwen_tts`'s built-in
`speech_tokenizer.decode`, and writes the audio.

Used as a temporary bridge until the C++ codec decoder lands.

    # Pipe codes from the C++ test driver:
    ./build/bin/test_qwen3_tts_talker MODEL VOICE "Hello world." 50 \\
        | tools/render_qwen3_tts_codes.py --model Qwen/Qwen3-TTS-12Hz-0.6B-Base \\
                                          -o /tmp/our.wav
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np


def parse_codes(text: str) -> np.ndarray:
    nums = re.findall(r"-?\d+", text)
    arr = np.array([int(n) for n in nums], dtype=np.int32)
    if arr.size % 16 != 0:
        sys.exit(f"need a multiple of 16 codes, got {arr.size}")
    return arr.reshape(-1, 16)


def main() -> None:
    ap = argparse.ArgumentParser(description="Render Qwen3-TTS codes → WAV")
    ap.add_argument("--model", required=True, help="HF repo id or local path")
    ap.add_argument("-o", "--output", required=True, help="output WAV path")
    ap.add_argument("--binary", action="store_true",
                    help="read codes as raw int32 bytes instead of text")
    ap.add_argument("--prepend-voice-pack",
                    help="prepend voice-pack ref_code before decoding "
                         "(matches PyTorch's generate_voice_clone path). "
                         "Format: <gguf_path>:<voice_name>")
    ap.add_argument("input", nargs="?", default="-",
                    help="codes file (default: stdin)")
    args = ap.parse_args()

    if args.input == "-":
        if args.binary:
            raw = sys.stdin.buffer.read()
            codes = np.frombuffer(raw, dtype=np.int32)
            if codes.size % 16 != 0:
                sys.exit(f"need a multiple of 16 codes, got {codes.size}")
            codes = codes.reshape(-1, 16)
        else:
            codes = parse_codes(sys.stdin.read())
    else:
        if args.binary:
            codes = np.fromfile(args.input, dtype=np.int32)
            if codes.size % 16 != 0:
                sys.exit(f"need a multiple of 16 codes, got {codes.size}")
            codes = codes.reshape(-1, 16)
        else:
            codes = parse_codes(Path(args.input).read_text())

    print(f"got {codes.shape[0]} frames × 16 codebooks", file=sys.stderr)

    ref_codes_np = None
    if args.prepend_voice_pack:
        from gguf import GGUFReader
        gguf_path, _, voice_name = args.prepend_voice_pack.partition(":")
        if not voice_name:
            sys.exit("--prepend-voice-pack expects <path>:<voice_name>")
        r = GGUFReader(gguf_path)
        key = f"voicepack.code.{voice_name}.codes"
        for t in r.tensors:
            if t.name == key:
                # GGUF ne[] is reverse of numpy: stored as (16, T_codec).
                ref_codes_np = t.data.reshape(t.shape[1], t.shape[0]).astype(np.int32)
                break
        if ref_codes_np is None:
            sys.exit(f"voice '{voice_name}' not in pack '{gguf_path}'")
        print(f"prepending {ref_codes_np.shape[0]} ref frames from {voice_name}", file=sys.stderr)

    import torch
    import soundfile as sf
    from qwen_tts import Qwen3TTSModel

    print(f"loading {args.model} on CPU…", file=sys.stderr)
    tts = Qwen3TTSModel.from_pretrained(
        args.model, device_map="cpu", dtype=torch.float32, attn_implementation="eager",
    )
    model = tts.model

    decode_codes = codes
    if ref_codes_np is not None:
        decode_codes = np.concatenate([ref_codes_np, codes], axis=0)
        print(f"concat codec input: {decode_codes.shape}", file=sys.stderr)

    codes_t = torch.from_numpy(decode_codes).long()
    wavs, sr = model.speech_tokenizer.decode([{"audio_codes": codes_t}])
    wav = np.asarray(wavs[0], dtype=np.float32)
    print(f"decoded → {len(wav)} samples @ {sr} Hz ({len(wav)/sr:.2f}s)", file=sys.stderr)

    if ref_codes_np is not None:
        # Trim the prepended-ref portion (matches PyTorch's wav[cut:])
        cut = int(ref_codes_np.shape[0] / decode_codes.shape[0] * wav.shape[0])
        print(f"trim first {cut} samples (ref portion)", file=sys.stderr)
        wav = wav[cut:]

    sf.write(args.output, wav, sr, subtype="PCM_16")
    print(f"wrote {args.output} ({len(wav)/sr:.2f}s)", file=sys.stderr)


if __name__ == "__main__":
    main()
