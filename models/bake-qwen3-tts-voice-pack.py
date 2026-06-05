#!/usr/bin/env python3
"""
Bake a "voice pack" GGUF for Qwen3-TTS-12Hz-Base.

The Base model is voice-cloning-only — every synthesis call needs a
reference audio + reference text. Two pieces are extracted from the
reference at clone time:

  1. ref_spk_embedding  (1024-d float32)
       From `Qwen3TTSSpeakerEncoder` (ECAPA-TDNN; mel(24kHz) → TDNN →
       3 SE-Res2Net → MFA → ASP → Conv1×1 → 1024-d).

  2. ref_code  (T_codec, 16) int32
       From `Qwen3TTSTokenizer.encode` (the 12 Hz RVQ codec encoder
       that turns the reference waveform into 16-codebook codes).

Re-running these in C++ requires porting both the ECAPA chain and the
codec encoder — substantial conv work that is NOT on the critical
path for verifying the talker / code_predictor / codec-decoder
forward passes.

This script bakes them once via the Apache-2.0 `qwen-tts` package and
writes them as named GGUF tensors into a "voice pack" archive. The
runtime then loads them by name and uses them in the ICL prefill
splice exactly as `Qwen3TTSForConditionalGeneration.generate_icl_prompt`
does — no Python at synthesis time.

Usage:
    python models/bake-qwen3-tts-voice-pack.py \\
        --model Qwen/Qwen3-TTS-12Hz-0.6B-Base \\
        --voice clone:samples/qwen3_tts/clone.wav:Okay. Yeah. I resent you... \\
        --voice clone_1:samples/qwen3_tts/clone_1.wav:甚至出现交易几乎停滞的情况。 \\
        --output /tmp/qwen3-tts-voice-pack.gguf

Each --voice arg is "name:wav_path:ref_text". The voice pack stores
per-voice tensors `spk.<name>.embd` and `code.<name>.codes` plus a
metadata array of names.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np


def parse_voice(arg: str) -> Tuple[str, str, str]:
    parts = arg.split(":", 2)
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            f"--voice expects 'name:wav_path:ref_text', got: {arg!r}")
    name, wav_path, ref_text = parts
    if not name or not wav_path or not ref_text:
        raise argparse.ArgumentTypeError(
            f"--voice all three fields required: {arg!r}")
    return name, wav_path, ref_text


def main() -> None:
    ap = argparse.ArgumentParser(description="Bake a Qwen3-TTS voice pack GGUF")
    ap.add_argument("--model", required=True,
                    help="HF repo id or local path of Qwen3-TTS-12Hz-{0.6B,1.7B}-Base")
    ap.add_argument("--voice", action="append", required=True, type=parse_voice,
                    help="name:wav_path:ref_text (repeatable, one per voice)")
    ap.add_argument("--output", required=True, help="Output GGUF path")
    args = ap.parse_args()

    # Lazy imports so --help works without the heavy stack
    try:
        import torch
        import soundfile as sf
        from qwen_tts import Qwen3TTSModel
    except ImportError as e:
        sys.exit(f"missing python dep: {e}. install: pip install -U qwen-tts soundfile torch")

    try:
        from gguf import GGUFWriter, GGMLQuantizationType
    except ImportError:
        sys.exit("missing python dep: gguf. install: pip install gguf")

    print(f"loading {args.model} on CPU (eager attn, fp32)…")
    tts = Qwen3TTSModel.from_pretrained(
        args.model,
        device_map="cpu",
        dtype=torch.float32,
        attn_implementation="eager",
    )
    model = tts.model

    voices: List[Tuple[str, str, str]] = list(args.voice)
    print(f"baking {len(voices)} voice(s)")

    out_path = Path(args.output)
    w = GGUFWriter(str(out_path), arch="qwen3tts.voicepack", use_temp_file=True)
    w.add_description("Qwen3-TTS voice prompt pack (spk_embedding + ref_code per voice)")
    w.add_array("voicepack.names", [name for name, _, _ in voices])
    w.add_array("voicepack.ref_texts", [t for _, _, t in voices])

    for name, wav_path, ref_text in voices:
        print(f"\n--- voice '{name}' ---")
        print(f"  wav:      {wav_path}")
        print(f"  ref_text: {ref_text!r}")

        audio, sr = sf.read(wav_path, dtype="float32", always_2d=False)
        if audio.ndim > 1:
            audio = audio.mean(axis=-1)
        print(f"  audio:    {len(audio)} samples @ {sr} Hz ({len(audio)/sr:.2f}s)")

        with torch.no_grad():
            prompt_items = tts.create_voice_clone_prompt(
                ref_audio=(audio.astype(np.float32), int(sr)),
                ref_text=ref_text,
                x_vector_only_mode=False,  # ICL mode — needs both spk_embed AND ref_code
            )

        # Each prompt_items[i] has ref_code (T, 16) int + ref_spk_embedding (1024,) float
        spk = prompt_items[0].ref_spk_embedding.detach().cpu().float().numpy()
        spk = np.ascontiguousarray(spk.reshape(-1).astype(np.float32))
        print(f"  spk:      shape={spk.shape}  dtype={spk.dtype}  norm={np.linalg.norm(spk):.4f}")

        code = prompt_items[0].ref_code.detach().cpu().numpy()
        code = np.ascontiguousarray(code.astype(np.int32))
        print(f"  code:     shape={code.shape}  dtype={code.dtype}  range=[{code.min()},{code.max()}]")

        # Persist int32 codes as int32 (gguf supports it). spk as F32.
        w.add_tensor(f"voicepack.spk.{name}.embd", spk, raw_dtype=GGMLQuantizationType.F32)
        # gguf treats integer tensors as int8/16/32 via raw_dtype; use int32 (I32).
        w.add_tensor(f"voicepack.code.{name}.codes", code, raw_dtype=GGMLQuantizationType.I32)

    print(f"\nwriting {out_path}")
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    sz = out_path.stat().st_size / 1e6
    print(f"done: {out_path}  ({sz:.2f} MB, {len(voices)} voice(s))")


if __name__ == "__main__":
    main()
