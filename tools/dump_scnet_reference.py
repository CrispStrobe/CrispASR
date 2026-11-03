#!/usr/bin/env python3
"""Capture the original pinned SCNet inference and all 35 required stages.

Input must already be native 44100-Hz stereo. By default the original helper
normalizes the waveform and runs 20-second segments, 50% overlap, one fresh
random shift per overlapping segment, and seed zero. --raw selects only the
original SCNet.forward diagnostic. No resampling, tiling, gain correction,
reference-math replacement, or missing-stage substitution is performed.

  python tools/dump_scnet_reference.py --source-dir ../SCNet-upstream \
      --checkpoint checkpoint.th --checkpoint-sha256 <64-hex-digest> \
      --audio music.wav --output scnet-reference.gguf
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from pathlib import Path

from reference_backends.scnet import (
    CHECKPOINT_URLS, DEFAULT_STAGES, capture_original, load_original,
    reference_metadata, sha256, validate_profile,
)


class ReferenceArchive:
    """Spool one owned F32 stage at a time; publish only a complete archive.

    GGUFWriter(use_temp_file=True) serializes each add_tensor synchronously.
    This bounds retained activation memory to one stage while accepting hooks
    in execution order. A failed capture leaves the previous output intact.
    """

    def __init__(self, output, metadata):
        from gguf import GGUFWriter

        self.output = Path(output)
        self.output.parent.mkdir(parents=True, exist_ok=True)
        handle, path = tempfile.mkstemp(prefix=f".{self.output.name}.", suffix=".partial", dir=self.output.parent)
        os.close(handle)
        self.temporary = Path(path)
        self.writer = None
        self.shapes = {}
        try:
            self.writer = GGUFWriter(path, "scnet-reference", use_temp_file=True)
            # Keep temporary tensor storage beside the output, including for
            # workers where /tmp is a different, capacity-limited filesystem.
            self.writer.temp_file = tempfile.SpooledTemporaryFile(max_size=8 * 1024 * 1024,
                                                                dir=self.output.parent)
            self.writer.add_description("Pinned original SCNet first-forward stages and full inference output")
            for key, value in metadata.items():
                if isinstance(value, bool):
                    self.writer.add_bool(key, value)
                elif key == "scnet.reference.shift_seed":
                    self.writer.add_uint64(key, value)
                elif isinstance(value, int):
                    self.writer.add_uint32(key, value)
                elif isinstance(value, float):
                    self.writer.add_float32(key, value)
                elif isinstance(value, str):
                    self.writer.add_string(key, value)
                else:
                    raise ValueError(f"unsupported reference metadata value: {key}")
        except BaseException:
            self.abort()
            raise

    def capture(self, name, value):
        import numpy as np

        if self.writer is None:
            raise RuntimeError("reference archive is closed")
        if name not in DEFAULT_STAGES or name in self.shapes:
            raise ValueError(f"unexpected or duplicate reference stage: {name}")
        if not isinstance(value, np.ndarray):
            value = value.detach().float().cpu().numpy()
        array = np.asarray(value, dtype=np.float32, order="C")
        if array.size == 0 or not np.isfinite(array).all():
            raise ValueError(f"empty or nonfinite reference stage: {name}")
        if name.endswith("_stats"):
            valid = array.shape == (2,)
        elif name.endswith("audio") or name.startswith(("output_", "raw_output_")):
            valid = array.ndim == 2 and array.shape[0] == 2
        else:
            valid = array.ndim == 4 and array.shape[0] == 1
        if not valid:
            raise ValueError(f"invalid original layout for {name}: {array.shape}")
        self.writer.add_tensor(name, array)
        self.shapes[name] = list(array.shape)

    def finish(self):
        missing = set(DEFAULT_STAGES) - set(self.shapes)
        if missing:
            raise ValueError(f"reference missing required stages: {sorted(missing)}")
        if self.writer is None:
            raise RuntimeError("reference archive is closed")
        self.writer.write_header_to_file()
        self.writer.write_kv_data_to_file()
        self.writer.write_tensors_to_file()
        self.writer.close()
        self.writer = None
        os.replace(self.temporary, self.output)
        self.temporary = None

    def abort(self):
        if self.writer is not None:
            self.writer.close()
            if self.writer.temp_file is not None:
                self.writer.temp_file.close()
            self.writer = None
        if self.temporary is not None:
            self.temporary.unlink(missing_ok=True)
            self.temporary = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, _exc, _traceback):
        if exc_type is None:
            try:
                self.finish()
            except BaseException:
                self.abort()
                raise
        else:
            self.abort()
        return False


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--variant", choices=tuple(CHECKPOINT_URLS), default="scnet")
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, help="explicitly trim the original waveform to this many frames")
    parser.add_argument("--segment-seconds", type=int, default=20)
    parser.add_argument("--shifts", type=int, default=1)
    parser.add_argument("--shift-seed", type=int, default=0)
    parser.add_argument("--raw", action="store_true", help="diagnostic original forward, without outer normalization/chunks")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if args.threads <= 0 or args.threads > 256:
        parser.error("--threads must be in [1,256]")
    try:
        validate_profile(args.segment_seconds, args.shifts, args.shift_seed)
    except ValueError as error:
        parser.error(str(error))
    import numpy as np
    import soundfile as sf
    import torch

    torch.set_num_threads(args.threads)
    audio, rate = sf.read(args.audio, dtype="float32", always_2d=True)
    if rate != 44100 or audio.shape[1] != 2:
        parser.error("input must be native 44100-Hz stereo; convert the fixture explicitly first")
    if args.samples is not None:
        if args.samples <= 0 or args.samples > len(audio):
            parser.error("--samples must be positive and no longer than the input")
        audio = audio[:args.samples]
    audio = np.ascontiguousarray(audio.T)
    model = load_original(args.source_dir, args.checkpoint, args.variant, args.device,
                          expected_sha256=args.checkpoint_sha256)
    profile = dict(segment_seconds=args.segment_seconds, shifts=args.shifts,
                   shift_seed=args.shift_seed, no_segment=args.raw)
    metadata = reference_metadata(model, audio, **profile)
    metadata["scnet.reference.threads"] = args.threads
    metadata["scnet.reference.device"] = str(next(model.parameters()).device)
    metadata["scnet.reference.input_file_sha256"] = sha256(args.audio)
    start = time.perf_counter()
    with ReferenceArchive(args.output, metadata) as archive:
        capture_original(model, audio, capture=archive.capture, **profile)
    receipt = {
        "metadata": metadata, "stages": archive.shapes, "reference_sha256": sha256(args.output),
        "capture_wall_seconds": time.perf_counter() - start,
        "complete": True, "note": "Capture duration includes hooks and disk output; not a performance measurement.",
    }
    args.output.with_name(args.output.name + ".json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote {args.output}: {len(archive.shapes)} original stages, {audio.shape[1]} stereo frames")


if __name__ == "__main__":
    main()
