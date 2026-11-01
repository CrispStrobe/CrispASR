#!/usr/bin/env python3
"""Actual external-CrispEmbed adapter controls; no model or Python ML stack."""

import argparse
import math
import os
from pathlib import Path
import struct
import subprocess
import tempfile


STAGES = ["input_audio", "stft_input", "stft_output", "band_split", "time_compressed",
          *(f"layer{i}_{axis}" for i in range(6) for axis in ("time", "freq")),
          "encoder_output", "freq_pooled", "stem_embeddings", "mask_raw",
          "output_bass", "output_drums", "output_other", "output_vocals"]
OUTPUTS = ["input_audio", "output_bass", "output_drums", "output_other", "output_vocals"]


def string(value):
    data = value.encode()
    return struct.pack("<Q", len(data)) + data


def archive(path, names, changed=None, shape=None):
    changed = changed or {}
    payload = {name: struct.pack("<16f", *changed.get(name, [1, 0, 0, 0] * 4)) for name in names}
    with path.open("wb") as stream:
        stream.write(b"GGUF" + struct.pack("<IQQ", 3, len(names), 1))
        stream.write(string("general.architecture") + struct.pack("<I", 8) + string("crispasr.reference"))
        for index, name in enumerate(names):
            dimensions = shape if name == "band_split" and shape else (4, 4)
            stream.write(string(name) + struct.pack("<IQQIQ", 2, *dimensions, 0, index * 64))
        stream.write(b"\0" * (-stream.tell() % 32))
        for value in payload.values():
            stream.write(value)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("adapter", type=Path)
    args = parser.parse_args()
    binary = str(args.adapter.resolve())
    with tempfile.TemporaryDirectory(prefix="mini-bs-crispembed-diff-") as directory:
        root = Path(directory)
        reference, native = root / "ref.gguf", root / "native.gguf"
        archive(reference, STAGES)

        def check(label, accepted, *options, environment=None):
            result = subprocess.run([binary, str(reference), str(native), "q8_0", *options],
                                    capture_output=True, text=True, timeout=30, env=environment)
            assert (result.returncode == 0) == accepted, (label, result.stdout, result.stderr)
            print("PASS", label)

        archive(native, STAGES)
        check("identical complete captures", True)
        archive(native, STAGES[:-1])
        check("missing stem rejected", False)
        archive(native, STAGES, shape=(2, 8))
        check("equal element count with wrong shape rejected", False)
        archive(native, STAGES, changed={"band_split": [2, 0, 0, 0] * 4})
        check("cosine-one gain error rejected", False)
        archive(native, STAGES, changed={"band_split": [math.nan, 0, 0, 0] * 4})
        check("nonfinite data rejected", False)
        archive(native, STAGES, changed={"band_split": [0.5, math.sqrt(0.75), 0, 0] * 4})
        env = dict(os.environ, CRISPEMBED_DIFF_COS_THRESHOLD="0.01")
        check("environment cannot weaken user threshold", False, environment=env)
        archive(reference, OUTPUTS)
        archive(native, OUTPUTS)
        check("outputs-only cannot masquerade as full stage proof", False)
        check("explicit complete outputs-only accepted", True, "--outputs-only")
        archive(native, OUTPUTS[:-1])
        check("outputs-only still requires every stem", False, "--outputs-only")


if __name__ == "__main__":
    main()
