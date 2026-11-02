#!/usr/bin/env python3
"""Check SCNet's storage policy using the real quantizer and tiny GGUFs.

The recurrent/convolution matrices are declared in their actual 2D GEMM
layout. A Q4_K artifact must never silently replace a short row with Q4_0.
No checkpoint or model-quality claim is involved in this model-free test.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path
import struct
import subprocess
import tempfile


TENSORS = {
    "separation_net.dp_modules.0.intra_rnn.weight_ih_l0": ((256, 4), 1),
    "separation_net.dp_modules.0.intra_rnn.weight_hh_l0": ((128, 4), 1),
    "encoder.0.sd_layer.conv.weight": ((16, 4), 1),
    "encoder.1.sd_layer.conv.weight": ((96, 4), 1),
    "separation_net.dp_modules.0.intra_norm.weight": ((256,), 0),
    "separation_net.dp_modules.0.intra_rnn.bias_ih_l0": ((256,), 0),
}


def string(value):
    encoded = value.encode()
    return struct.pack("<Q", len(encoded)) + encoded


def fixture(path, architecture):
    raw = {
        name: b"".join(struct.pack("<f" if kind == 0 else "<e", math.sin(i * 0.13))
                       for i in range(math.prod(shape)))
        for name, (shape, kind) in TENSORS.items()
    }
    with path.open("wb") as output:
        output.write(b"GGUF" + struct.pack("<IQQ", 3, len(TENSORS), 2))
        output.write(string("general.architecture") + struct.pack("<I", 8) + string(architecture))
        output.write(string("test.provenance") + struct.pack("<I", 8) + string("model-free-fixture"))
        offset = 0
        for name, (shape, kind) in TENSORS.items():
            output.write(string(name) + struct.pack("<I", len(shape)))
            output.write(struct.pack("<" + "Q" * len(shape), *shape))
            output.write(struct.pack("<IQ", kind, offset))
            offset += (len(raw[name]) + 31) // 32 * 32
        output.write(b"\0" * (-output.tell() % 32))
        for data in raw.values():
            output.write(data)
            output.write(b"\0" * (-len(data) % 32))
    return raw


def read_table(path):
    with path.open("rb") as stream:
        def number(fmt):
            return struct.unpack("<" + fmt, stream.read(struct.calcsize("<" + fmt)))[0]

        def text():
            return stream.read(number("Q")).decode()

        def value(kind):
            if kind == 8:
                return text()
            if kind == 9:
                subtype, length = number("I"), number("Q")
                return [value(subtype) for _ in range(length)]
            return number({0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f",
                           7: "B", 10: "Q", 11: "q", 12: "d"}[kind])

        assert stream.read(4) == b"GGUF" and number("I") == 3
        count, fields = number("Q"), number("Q")
        metadata = {}
        for _ in range(fields):
            key, kind = text(), number("I")
            metadata[key] = value(kind)
        table = {}
        for _ in range(count):
            name = text()
            shape = [number("Q") for _ in range(number("I"))]
            kind, offset = number("I"), number("Q")
            table[name] = {"shape": shape, "type": kind, "offset": offset}
        alignment = metadata.get("general.alignment", 32)
        start = (stream.tell() + alignment - 1) // alignment * alignment
        for row in table.values():
            if row["type"] in (0, 1):
                stream.seek(start + row["offset"])
                row["bytes"] = stream.read(math.prod(row["shape"]) * (4 if row["type"] == 0 else 2))
        return metadata, table


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("quantizer", type=Path)
    args = parser.parse_args()
    quantizer = str(args.quantizer.resolve(strict=True))
    with tempfile.TemporaryDirectory(prefix="scnet-quant-policy-") as directory:
        work = Path(directory)
        source = work / "source.gguf"
        original = fixture(source, "scnet")

        def run(label, input_path, quant, *extra):
            output = work / (label + ".gguf")
            result = subprocess.run([quantizer, str(input_path), str(output), quant, *extra],
                                    capture_output=True, text=True, timeout=30)
            assert result.returncode == 0, (label, result.stdout, result.stderr)
            metadata, table = read_table(output)
            assert metadata["test.provenance"] == "model-free-fixture", label
            return output, table

        for quant, wide_type in (("q8_0", 8), ("q4_k", 12)):
            output, table = run(quant, source, quant)
            for name, (shape, original_type) in TENSORS.items():
                if len(shape) == 1 or shape[0] < 32:
                    assert table[name]["type"] == original_type, (quant, name, table[name])
                    assert table[name]["bytes"] == original[name], (quant, name)
                else:
                    expected = wide_type if shape[0] % 256 == 0 else 8
                    assert table[name]["type"] == expected, (quant, name, table[name])
                assert table[name]["shape"] == list(shape), (quant, name)
            if quant == "q8_0":
                q8_path = output
            print("PASS", quant, "GEMM layouts, Q8 row fallback and byte-exact scalar/small tensors")

        _, table = run("requantize", q8_path, "q4_k")
        assert table[next(iter(TENSORS))]["type"] == 12
        for name, (shape, _) in TENSORS.items():
            if len(shape) == 2 and 32 <= shape[0] < 256:
                assert table[name]["type"] == 8, name
        print("PASS Q8_0-to-Q4_K policy")

        # Explicit calibration overrides remain authoritative. The production
        # default alone prevents implicit fallback; deliberate experiments can
        # select another type and must be labelled and measured independently.
        narrow = "separation_net.dp_modules.0.intra_rnn.weight_hh_l0"
        _, table = run("override", source, "q4_k", "--tensor-type", r"weight_hh_l0$=q4_0")
        assert table[narrow]["type"] == 2
        other = work / "other.gguf"
        fixture(other, "unrelated-architecture")
        _, table = run("other", other, "q4_k")
        assert table[narrow]["type"] == 2
        print("PASS explicit overrides and unrelated architecture policy")


if __name__ == "__main__":
    main()
