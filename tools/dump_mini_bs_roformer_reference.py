#!/usr/bin/env python3
"""Dump original Mini-BS-RoFormer stages from native 44100-Hz stereo audio.

Unlike the shared ASR reference CLI, this entry point preserves stereo and the
model sample rate. ``--samples`` makes short forward fixtures explicit; omit it
for a complete input. Inputs longer than a trained chunk should be validated
separately with the original ``model.separate`` overlap/add method.

By default, hooks stream F32 stages into one atomic GGUF archive in bounded
blocks. This avoids retaining every intermediate activation during inference;
the original forward computation and all selected tensor values are unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from dump_reference import write_gguf_archive
from reference_backends._streaming_gguf import StreamingGGUFArchive
from reference_backends.mini_bs_roformer import (
    DEFAULT_STAGES, MODEL_ID, _load, dump_waveform, reference_metadata, stage_shapes,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", default=MODEL_ID)
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, help="explicitly trim to this many native-rate samples")
    parser.add_argument("--stages", default="", help="comma-separated stages; default captures all boundaries")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--in-memory", action="store_true", help="retain all stages before writing (diagnostic only)")
    args = parser.parse_args()
    if args.threads <= 0:
        parser.error("--threads must be positive")
    import soundfile as sf
    import torch

    torch.set_num_threads(args.threads)
    audio, sample_rate = sf.read(args.audio, dtype="float32", always_2d=True)
    if sample_rate != 44100 or audio.shape[1] != 2:
        parser.error("input must be native 44100-Hz stereo; convert the fixture explicitly first")
    if args.samples is not None:
        if args.samples <= 2048 or args.samples > len(audio):
            parser.error("--samples must be greater than 2048 and no longer than the input")
        audio = audio[:args.samples]
    audio = np.ascontiguousarray(audio.T)
    stages = set(filter(None, (item.strip() for item in args.stages.split(",")))) if args.stages else set(DEFAULT_STAGES)
    model = _load(args.model_dir, device=args.device)
    shapes = stage_shapes(model.config, audio.shape[1], stages)
    metadata = reference_metadata(model)
    metadata.update({
        "backend": "mini-bs-roformer",
        "sample_rate": sample_rate,
        "n_samples": int(audio.shape[1]),
        "audio_channels": int(audio.shape[0]),
        "input_pcm_sha256": hashlib.sha256(audio.astype("<f4").tobytes()).hexdigest(),
        "sources_json": json.dumps(["bass", "drums", "other", "vocals"]),
        "threads": args.threads,
    })
    if args.in_memory:
        captures = dump_waveform(model, audio, stages=stages)
        captures = {name: value for name, value in captures.items() if not isinstance(value, str)}
        write_gguf_archive(captures, metadata, args.output)
    else:
        with StreamingGGUFArchive(args.output, shapes, metadata) as archive:
            dump_waveform(model, audio, stages=stages, capture=archive.capture)
        print(f"Largest serialization block: {archive.max_block_bytes} bytes")
    print(f"Wrote {args.output}: {len(shapes)} stages, {audio.shape[1]} stereo samples")
    for name, shape in shapes.items():
        print(f"  {name:24s} {list(shape)}")


if __name__ == "__main__":
    main()
