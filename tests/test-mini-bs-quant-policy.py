#!/usr/bin/env python3
"""Exercise Mini BS RoFormer's precision policy through the real quantizer."""

import argparse
import math
from pathlib import Path
import struct
import subprocess
import tempfile


PREFIX = "freq_domain_model."
PROTECTED = [
    PREFIX + "mask_model.band_split.to_features.0.1.weight",
    PREFIX + "mask_model.mask_estimator.to_freq_mlps.0.weight",
    PREFIX + "mask_model.mask_estimator.to_gate_mlps.0.weight",
]
FLOORED = [
    PREFIX + "mask_model.time_conv.gate_proj.weight",
    PREFIX + "mask_model.time_deconv.down_proj.weight",
    PREFIX + "freq_pooling.attn.q_proj.weight",
    PREFIX + "time_pooling.mlp.down_proj.weight",
]
BODY = PREFIX + "mask_model.encoder.layers.0.time_transformer.0.self_attn.q_proj.weight"
NORM = PREFIX + "mask_model.encoder.final_norm.scale"
PROBE = PREFIX + "time_pooling.probe"
REGISTERS = PREFIX + "mask_model.encoder.register_tokens"
SMALL = PREFIX + "mask_model.band_split.to_features.1.1.weight"


def string(value):
    encoded = value.encode()
    return struct.pack("<Q", len(encoded)) + encoded


def fixture(path, architecture):
    tensors = {name: ((256, 4), 1) for name in PROTECTED + FLOORED + [BODY]}
    tensors.update({NORM: ((256,), 0), PROBE: ((256, 4), 0),
                    REGISTERS: ((256, 4, 4), 0), SMALL: ((8, 4), 1)})
    raw = {}
    for name, (shape, kind) in tensors.items():
        raw[name] = b"".join(struct.pack("<f" if kind == 0 else "<e", math.sin(i * .13))
                             for i in range(math.prod(shape)))
    with path.open("wb") as output:
        output.write(b"GGUF" + struct.pack("<IQQ", 3, len(tensors), 2))
        output.write(string("general.architecture") + struct.pack("<I", 8) + string(architecture))
        output.write(string("test.provenance") + struct.pack("<I", 8) + string("model-free-fixture"))
        offset = 0
        for name, (shape, kind) in tensors.items():
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
    quantizer = str(args.quantizer.resolve())
    with tempfile.TemporaryDirectory(prefix="mini-bs-quant-policy-") as directory:
        root = Path(directory)
        source = root / "source.gguf"
        original = fixture(source, "mini-bs-roformer")

        def run(label, input_path, quant, *extra):
            output = root / (label + ".gguf")
            result = subprocess.run([quantizer, str(input_path), str(output), quant, *extra],
                                    capture_output=True, text=True, timeout=30)
            assert result.returncode == 0, (label, result.stdout, result.stderr)
            metadata, table = read_table(output)
            assert metadata["test.provenance"] == "model-free-fixture", label
            return output, table

        for quant, body_type in (("q8_0", 8), ("q4_k", 12)):
            output, table = run(quant, source, quant)
            assert table[BODY]["type"] == body_type, (quant, table[BODY])
            for name in PROTECTED + [SMALL]:
                assert table[name]["type"] == 1 and table[name]["bytes"] == original[name], (quant, name)
            for name in FLOORED:
                assert table[name]["type"] == 8, (quant, name, table[name])
            for name in (NORM, PROBE, REGISTERS):
                assert table[name]["type"] == 0 and table[name]["bytes"] == original[name], (quant, name)
            if quant == "q8_0":
                q8_path = output
            print("PASS", quant, "body, Q8 floors, byte-exact F16/F32 carve-outs")

        _, requantized = run("q8-to-q4", q8_path, "q4_k")
        assert requantized[BODY]["type"] == 12
        assert all(requantized[name]["type"] == 8 for name in FLOORED)
        assert all(requantized[name]["bytes"] == original[name] for name in PROTECTED)
        print("PASS q8-to-q4 retains the precision policy")

        # Real user overrides must remain possible for calibration experiments.
        _, overridden = run("override", source, "q4_k", "--tensor-type",
                            r"^freq_domain_model\.mask_model\.mask_estimator\..*\.weight$=q8_0",
                            "--tensor-type", r"time_conv\.gate_proj\.weight$=q4_k")
        assert overridden[PROTECTED[1]]["type"] == 8
        assert overridden[PROTECTED[2]]["type"] == 8
        assert overridden[FLOORED[0]]["type"] == 12
        print("PASS explicit calibration overrides")

        other = root / "other-source.gguf"
        fixture(other, "unrelated-architecture")
        _, table = run("other", other, "q4_k")
        assert all(table[name]["type"] == 12 for name in PROTECTED + FLOORED + [BODY])
        print("PASS unrelated architectures retain their original policy")


if __name__ == "__main__":
    main()
