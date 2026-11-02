#!/usr/bin/env python3
"""Check SCNet's storage policy using the real quantizer and tiny GGUFs.

The recurrent/convolution matrices are declared in their actual 2D GEMM
layout. A Q4_K artifact must never silently replace a short row with Q4_0.
The four final reconstruction matrices stay F16; neighboring decoder matrices
remain quantized, and explicit tensor-type overrides remain authoritative.
No checkpoint or model-quality claim is involved in this model-free test.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path
import struct
import subprocess
import tempfile


RECONSTRUCTION = {
    # Actual flattened GEMM geometry from the small checkpoint, in ggml order.
    "decoder.2.0.conv.weight": (576, 64),
    "decoder.2.1.convtrs.0.weight": (32, 48),
    "decoder.2.1.convtrs.1.weight": (32, 64),
    "decoder.2.1.convtrs.2.weight": (32, 256),
}
TENSORS = {
    "separation_net.dp_modules.0.intra_rnn.weight_ih_l0": ((256, 4), 1),
    "separation_net.dp_modules.0.intra_rnn.weight_hh_l0": ((128, 4), 1),
    "encoder.0.sd_layer.conv.weight": ((16, 4), 1),
    "encoder.1.sd_layer.conv.weight": ((96, 4), 1),
    "separation_net.dp_modules.0.intra_norm.weight": ((256,), 0),
    "separation_net.dp_modules.0.intra_rnn.bias_ih_l0": ((256,), 0),
    **{name: (shape, 1) for name, shape in RECONSTRUCTION.items()},
    "decoder.2.0.conv.bias": ((64,), 0),
    "decoder.2.0.norm.weight": ((64,), 0),
    "decoder.1.0.conv.weight": ((256, 4), 1),
    "decoder.1.1.convtrs.0.weight": ((128, 4), 1),
    "decoder.20.0.conv.weight": ((256, 4), 1),
    "decoder.2.1.convtrs.3.weight": ((128, 4), 1),
    "decoder.2.0.conv_extra.weight": ((256, 4), 1),
}


def string(value):
    encoded = value.encode()
    return struct.pack("<Q", len(encoded)) + encoded


def fixture(path, architecture, reconstruction_type=1):
    tensor_types = {name: reconstruction_type if name in RECONSTRUCTION else kind
                    for name, (_, kind) in TENSORS.items()}
    raw = {
        name: b"".join(struct.pack("<f" if tensor_types[name] == 0 else "<e", math.sin(i * 0.13))
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
            output.write(struct.pack("<IQ", tensor_types[name], offset))
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
                if name in RECONSTRUCTION or len(shape) == 1 or shape[0] < 32:
                    assert table[name]["type"] == original_type, (quant, name, table[name])
                    assert table[name]["bytes"] == original[name], (quant, name)
                else:
                    expected = wide_type if shape[0] % 256 == 0 else 8
                    assert table[name]["type"] == expected, (quant, name, table[name])
                assert table[name]["shape"] == list(shape), (quant, name)
            if quant == "q8_0":
                q8_path = output
            print("PASS", quant, "exact reconstruction F16 bytes, neighboring matrices, "
                  "Q8 row fallback and unchanged scalar/small tensors")

        _, table = run("requantize", q8_path, "q4_k")
        assert table[next(iter(TENSORS))]["type"] == 12
        for name, (shape, _) in TENSORS.items():
            if name in RECONSTRUCTION:
                assert table[name]["type"] == 1 and table[name]["bytes"] == original[name], name
            elif len(shape) == 2 and 32 <= shape[0] < 256:
                assert table[name]["type"] == 8, name
        print("PASS Q8_0-to-Q4_K policy")

        # F32 input must produce the same F16 preservation policy, not retain
        # F32 merely because the selected matrices were skipped by quantization.
        f32_source = work / "f32-source.gguf"
        f32_values = fixture(f32_source, "scnet", reconstruction_type=0)
        for quant in ("q8_0", "q4_k"):
            _, table = run("f32-input-" + quant, f32_source, quant)
            for name in RECONSTRUCTION:
                expected = b"".join(struct.pack("<e", value[0])
                                    for value in struct.iter_unpack("<f", f32_values[name]))
                assert table[name]["type"] == 1 and table[name]["bytes"] == expected, (quant, name)
        print("PASS F32 reconstruction matrices convert to F16")

        # Explicit calibration overrides remain authoritative. The production
        # default alone prevents implicit fallback; deliberate experiments can
        # select another type and must be labelled and measured independently.
        narrow = "separation_net.dp_modules.0.intra_rnn.weight_hh_l0"
        _, table = run("override", source, "q4_k", "--tensor-type", r"weight_hh_l0$=f16")
        assert table[narrow]["type"] == 1
        assert table[narrow]["bytes"] == original[narrow]
        for quant in ("q8_0", "q4_k"):
            _, table = run("head-override-" + quant, source, quant,
                           "--tensor-type", r"^decoder\.2\.0\.conv\.weight$=f32",
                           "--tensor-type", r"^decoder\.2\..*\.weight$=q8_0")
            for name in RECONSTRUCTION:
                expected = 0 if name == "decoder.2.0.conv.weight" else 8
                assert table[name]["type"] == expected, (quant, name)
            for name in ("decoder.2.0.conv.bias", "decoder.2.0.norm.weight"):
                assert table[name]["type"] == 0 and table[name]["bytes"] == original[name]
        print("PASS reconstruction overrides and first-match precedence; norms/biases remain F32")

        other = work / "other-source.gguf"
        fixture(other, "unrelated-architecture")
        for quant, expected in (("q8_0", 8), ("q4_k", 2)):
            _, table = run("other-" + quant, other, quant)
            assert table[narrow]["type"] == expected
            for name in RECONSTRUCTION:
                assert table[name]["type"] == expected, (quant, name)
        _, table = run("other-target-type", source, "q4_0")
        for name in RECONSTRUCTION:
            assert table[name]["type"] == 2, name
        print("PASS unrelated architecture and non-Q8_0/Q4_K target policies unchanged")


if __name__ == "__main__":
    main()
