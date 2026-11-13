#!/usr/bin/env python3
"""Inspect published Nemotron 3.5 tensor layouts without loading model weights.

Reuse existing bounded prefix files with --prefix-dir, or fetch at most 8 MiB
per decoder and 1 MiB for the encoder. This reports storage compatibility only;
it does not convert weights or evaluate recognition/translation quality.
"""
import argparse
from collections import defaultdict
import importlib.util
import json
import math
from pathlib import Path
import urllib.request

REVISION = "dbd06c5cad7a2f33c8e06be2fb3ef814dbd9aa59"
REPOSITORY = "AtomicChat/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-GGUF"
STEM = "NVIDIA-Nemotron-3.5-Transcribe-30B-A4B"
FILES = {
    "decoder": (STEM + "-Q8_0.gguf", 8 << 20),
    "iq4": (STEM + "-AD-IQ4_NL.gguf", 8 << 20),
    "encoder": (STEM + "-Encoder-F16.gguf", 1 << 20),
}


def summarize(header):
    groups = defaultdict(lambda: {"count": 0, "elements": 0, "examples": []})
    quantized = []
    for name, dims, dtype, offset, size in header["tensors"]:
        group = groups[(tuple(dims), dtype)]
        group["count"] += 1
        group["elements"] += math.prod(dims)
        if len(group["examples"]) < 2:
            group["examples"].append(name)
        if len(dims) >= 2 and dtype in ("Q8_0", "IQ4_NL"):
            quantized.append((name, dims, math.prod(dims)))
    total = sum(t[2] for t in quantized)
    compatibility = {}
    for block in (32, 256):
        fitting = [t for t in quantized if t[1][0] % block == 0]
        elements = sum(t[2] for t in fitting)
        compatibility[str(block)] = {"tensors": len(fitting), "elements": elements,
                                     "fraction": elements / total if total else None}
    experts = [t for t in quantized if "_exps.weight" in t[0]]
    return {
        "architecture": header["kv"].get("general.architecture"),
        "header_bytes": header["header_bytes"], "tensor_count": header["n_tensors"],
        "dtype_counts": header["dtype_hist"],
        "quantized_matrix_tensors": len(quantized), "quantized_matrix_elements": total,
        "expert_bank_tensors": len(experts), "expert_bank_elements": sum(t[2] for t in experts),
        "row_block_compatibility": compatibility,
        "layout_groups": [{"ggml_dimensions": list(dims), "dtype": dtype, **value}
                          for (dims, dtype), value in sorted(groups.items())],
    }


def inspect(prefix_dir=None):
    spec = importlib.util.spec_from_file_location("remote_gguf", Path(__file__).with_name("verify-remote-gguf.py"))
    remote = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(remote)
    report = {"repository": REPOSITORY, "revision": REVISION,
              "scope": "Published tensor headers only; no model execution or quality measurement", "files": {}}
    for role, (name, limit) in FILES.items():
        url = f"https://huggingface.co/{REPOSITORY}/resolve/{REVISION}/{name}"
        if prefix_dir:
            data = (prefix_dir / (role + "-prefix.gguf")).read_bytes()
        else:
            request = urllib.request.Request(url, headers={"Range": f"bytes=0-{limit - 1}"})
            with urllib.request.urlopen(request) as response:
                if response.status != 206:
                    raise RuntimeError("Server ignored bounded Range request")
                data = response.read(limit)
        reader = remote.RangeReader(lambda start, end: data[start:end + 1], len(data))
        header = remote.parse_gguf_header(reader)
        report["files"][role] = {"file": name, "url": url, "prefix_bytes": len(data), **summarize(header)}
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix-dir", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = json.dumps(inspect(args.prefix_dir), indent=2) + "\n"
    if args.output:
        args.output.write_text(result)
    else:
        print(result, end="")
