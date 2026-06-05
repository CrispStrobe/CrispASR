#!/usr/bin/env python3
"""Bake a Chatterbox voice GGUF from a reference WAV.

Mirrors the vibevoice voice-pack workflow (see
``models/convert-vibevoice-voice-to-gguf.py``): the WAV is run through the
upstream Resemble Chatterbox pipeline (VoiceEncoder → 256-d spkr_emb,
S3Tokenizer → speech-cond prompt tokens, S3Gen.embed_ref → 24 kHz prompt
mel + CAMPPlus 192-d x-vector). The resulting ``Conditionals`` bundle is
then dumped under the same tensor names the C++ runtime already accepts
for the built-in voice (see ``src/chatterbox.cpp:846-851``):

    conds.t3.speaker_emb           f32  (1, 256)
    conds.t3.speech_prompt_tokens  i32  (T_prompt,)
    conds.gen.prompt_token         i32  (1, T_speech_tokens)
    conds.gen.prompt_feat          f32  (1, T_mel, 80)
    conds.gen.embedding            f32  (1, 192)

plus float32 metadata:

    chatterbox.conds.emotion_adv          (default exaggeration scalar)
    chatterbox.conds.gen_prompt_token_len (= T_speech_tokens)

The baked voice loads via ``chatterbox_set_voice_from_wav(ctx, path)`` —
the C++ side recognizes a ``.gguf`` extension and routes to the GGUF path.
WAV → cond extraction in C++ (porting VE/CAMPPlus/S3Tokenizer to ggml) is
a separate, larger refactor; this baker is the pragmatic short path.

Usage:
    python models/bake-chatterbox-voice-from-wav.py \\
        --input /path/to/reference.wav \\
        --output my_voice.gguf \\
        [--exaggeration 0.5]

Requirements: the upstream ``chatterbox-tts`` package and ``gguf`` (ships
with llama.cpp / ggml). Set ``RESEMBLE_CHATTERBOX_SRC`` if you have a
local clone of resemble-ai/chatterbox to use instead of the installed pkg.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

# Disable xformers BEFORE diffusers tries to import its flash3 path against
# a torch that does not export _flash_attention_backward_flop. Same trick the
# diff harness uses (tools/dump_reference.py and tools/reference_backends/
# chatterbox.py via env-driven shim).
try:
    import diffusers.utils.import_utils as _iu  # type: ignore[import]
    _iu._xformers_available = False
except Exception:  # pragma: no cover — diffusers absent is fine
    pass


def _find_chatterbox_src() -> None:
    """Push a local resemble-chatterbox clone onto sys.path if requested."""
    src = os.environ.get("RESEMBLE_CHATTERBOX_SRC")
    if not src:
        return
    p = Path(src).resolve()
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, type=Path,
                        help="reference WAV (any sample rate, mono or stereo)")
    parser.add_argument("--output", required=True, type=Path,
                        help="output voice GGUF path")
    parser.add_argument("--model-dir", type=Path, default=None,
                        help="local Chatterbox snapshot dir; if omitted, "
                             "ChatterboxTTS.from_pretrained() is used")
    parser.add_argument("--exaggeration", type=float, default=0.5,
                        help="emotion-adv scalar baked into the voice (0.0-2.0, default 0.5)")
    parser.add_argument("--device", default="cpu",
                        help="device for the baking forward pass (default cpu)")
    args = parser.parse_args()

    _find_chatterbox_src()

    try:
        import torch
    except ImportError as e:
        raise SystemExit("torch is required to bake a chatterbox voice") from e
    try:
        from chatterbox.tts import ChatterboxTTS
    except ImportError as e:
        raise SystemExit(
            "chatterbox-tts (resemble-ai/chatterbox) not importable. "
            "Install with: pip install chatterbox-tts\n"
            "Or set RESEMBLE_CHATTERBOX_SRC=/path/to/chatterbox/src to use a local clone."
        ) from e
    try:
        import gguf
    except ImportError as e:
        raise SystemExit("gguf python package required (ships with llama.cpp / ggml)") from e

    if not args.input.exists():
        raise SystemExit(f"input WAV not found: {args.input}")

    print(f"loading Chatterbox ({'from-local: ' + str(args.model_dir) if args.model_dir else 'from-pretrained'})")
    if args.model_dir is not None:
        model = ChatterboxTTS.from_local(str(args.model_dir), device=args.device)
    else:
        model = ChatterboxTTS.from_pretrained(device=args.device)

    print(f"running prepare_conditionals on {args.input}")
    model.prepare_conditionals(str(args.input), exaggeration=args.exaggeration)
    conds = model.conds
    if conds is None:
        raise SystemExit("prepare_conditionals returned None — upstream API change?")

    t3 = conds.t3
    gen = conds.gen

    # Validate shapes
    speaker_emb = t3.speaker_emb.detach().cpu().float().numpy()      # (1, 256)
    if speaker_emb.shape != (1, 256):
        raise SystemExit(f"unexpected speaker_emb shape: {speaker_emb.shape}, expected (1, 256)")
    if t3.cond_prompt_speech_tokens is None:
        raise SystemExit("t3.cond_prompt_speech_tokens is None — model.t3.hp.speech_cond_prompt_len is 0?")
    speech_prompt_tokens = t3.cond_prompt_speech_tokens.detach().cpu().int().numpy()  # (1, T_prompt)
    speech_prompt_tokens = np.ascontiguousarray(speech_prompt_tokens.reshape(-1).astype(np.int32))

    # gen.prompt_token: (1, T) int → flatten to (T,)
    prompt_token = gen["prompt_token"].detach().cpu().int().numpy()
    prompt_token = np.ascontiguousarray(prompt_token.reshape(-1).astype(np.int32))

    # gen.prompt_feat: (1, T_mel, 80) or (1, 80, T_mel) depending on upstream version.
    # Upstream s3gen.embed_ref returns (1, 80, T_mel) then transposes to (1, T_mel, 80)
    # before storing in the dict. Match the shape baked by convert-chatterbox-to-gguf.py
    # at ``conds.gen.prompt_feat`` (currently writes the tensor as-is from the dict).
    prompt_feat = gen["prompt_feat"].detach().cpu().float().numpy()
    if prompt_feat.ndim == 3 and prompt_feat.shape[0] == 1:
        prompt_feat = prompt_feat[0]
    prompt_feat = np.ascontiguousarray(prompt_feat.astype(np.float32))

    # gen.embedding: (1, 192) f32
    embedding = gen["embedding"].detach().cpu().float().numpy()
    if embedding.ndim == 2 and embedding.shape[0] == 1:
        embedding = embedding[0:1]
    embedding = np.ascontiguousarray(embedding.astype(np.float32))

    # Build the GGUF
    args.output.parent.mkdir(parents=True, exist_ok=True)
    print(f"writing voice GGUF: {args.output}")
    writer = gguf.GGUFWriter(str(args.output), arch="chatterbox-voice")
    writer.add_name(args.output.stem)
    writer.add_description("Chatterbox voice conditioning baked from " + str(args.input))

    writer.add_float32("chatterbox.conds.emotion_adv", float(args.exaggeration))
    writer.add_uint32("chatterbox.conds.gen_prompt_token_len", int(prompt_token.shape[0]))

    writer.add_tensor("conds.t3.speaker_emb",
                      speaker_emb.astype(np.float32),
                      raw_dtype=gguf.GGMLQuantizationType.F32)
    writer.add_tensor("conds.t3.speech_prompt_tokens",
                      speech_prompt_tokens,
                      raw_dtype=gguf.GGMLQuantizationType.I32)
    writer.add_tensor("conds.gen.prompt_token",
                      prompt_token,
                      raw_dtype=gguf.GGMLQuantizationType.I32)
    writer.add_tensor("conds.gen.prompt_feat",
                      prompt_feat,
                      raw_dtype=gguf.GGMLQuantizationType.F32)
    writer.add_tensor("conds.gen.embedding",
                      embedding,
                      raw_dtype=gguf.GGMLQuantizationType.F32)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    sz = args.output.stat().st_size
    print(f"  wrote {args.output} ({sz / 1e6:.1f} MB)")
    print(f"  speaker_emb            {speaker_emb.shape} f32")
    print(f"  speech_prompt_tokens   {speech_prompt_tokens.shape} i32 "
          f"(first 10: {speech_prompt_tokens[:10].tolist()})")
    print(f"  prompt_token           {prompt_token.shape} i32 "
          f"(first 10: {prompt_token[:10].tolist()})")
    print(f"  prompt_feat            {prompt_feat.shape} f32")
    print(f"  embedding              {embedding.shape} f32")


if __name__ == "__main__":
    main()
