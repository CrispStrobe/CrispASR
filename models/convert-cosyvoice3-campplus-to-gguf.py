#!/usr/bin/env python3
"""Convert CosyVoice3 CAMPPlus ONNX to GGUF.

This writes the native speaker-encoder bundle used by the Phase 6 WAV
clone path. The source ONNX export mixes named tensors for the bulk of
the xvector chain with an anonymous initializer tail for the FCM head,
TDNN, CAMDense blocks, and transit3 projection. We preserve the weight
layout and emit the tensor names that the C++ runtime expects:

  - `s3.se.head.*`
  - `s3.se.xv.tdnn.*`
  - `s3.se.xv.transit{1,2,3}.*`
  - `s3.se.xv.block{1,2,3}.tdnnd*.*`
  - `s3.se.xv.dense.*`

Usage:
  python models/convert-cosyvoice3-campplus-to-gguf.py \
      --input /Volumes/backups/ai/upstream/cosyvoice3-onnx/campplus.onnx \
      --output /tmp/cosyvoice3-campplus-f16.gguf
"""

import argparse
import os
import re
from pathlib import Path

import numpy as np

try:
    import onnx
    from onnx import numpy_helper
except ImportError as exc:  # pragma: no cover - env dependency
    raise SystemExit("onnx is required: pip install onnx") from exc

try:
    import gguf
except ImportError:
    import sys

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ggml", "python"))
    import gguf


def _load_model(input_path: Path) -> onnx.ModelProto:
    if input_path.is_dir():
        candidate = input_path / "campplus.onnx"
        if not candidate.exists():
            candidate = input_path / "model.onnx"
        input_path = candidate
    if not input_path.exists():
        raise SystemExit(f"ONNX model not found: {input_path}")
    return onnx.load(str(input_path), load_external_data=True)


def _as_float_array(arr: np.ndarray, *, force_f32: bool = False) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.dtype.kind in "iu":
        return np.ascontiguousarray(arr.astype(np.int32))
    if force_f32 or arr.ndim <= 1:
        return np.ascontiguousarray(arr.astype(np.float32))
    return np.ascontiguousarray(arr.astype(np.float16))


def _expect(shape: tuple[int, ...], expected: tuple[int, ...], name: str) -> None:
    if shape != expected:
        raise SystemExit(f"unexpected tensor shape for {name}: got {shape}, expected {expected}")


def _shape(arr: np.ndarray) -> tuple[int, ...]:
    return tuple(int(x) for x in arr.shape)


def _map_named_tensor(name: str) -> str | None:
    m = re.match(r"^xvector\.block([123])\.tdnnd(\d+)\.nonlinear1\.batchnorm\.(weight|bias|running_mean|running_var)$", name)
    if m:
        blk, layer, suffix = m.groups()
        suffix_map = {
            "weight": "nonl1.bn.weight",
            "bias": "nonl1.bn.bias",
            "running_mean": "nonl1.bn.running_mean",
            "running_var": "nonl1.bn.running_var",
        }
        return f"s3.se.xv.block{blk}.tdnnd{layer}.{suffix_map[suffix]}"

    m = re.match(r"^xvector\.block([123])\.tdnnd(\d+)\.cam_layer\.linear_local\.weight$", name)
    if m:
        blk, layer = m.groups()
        return f"s3.se.xv.block{blk}.tdnnd{layer}.cam.ll.weight"

    m = re.match(r"^xvector\.block([123])\.tdnnd(\d+)\.cam_layer\.linear1\.(weight|bias)$", name)
    if m:
        blk, layer, kind = m.groups()
        return f"s3.se.xv.block{blk}.tdnnd{layer}.cam.l1.{kind}"

    m = re.match(r"^xvector\.block([123])\.tdnnd(\d+)\.cam_layer\.linear2\.(weight|bias)$", name)
    if m:
        blk, layer, kind = m.groups()
        return f"s3.se.xv.block{blk}.tdnnd{layer}.cam.l2.{kind}"

    m = re.match(r"^xvector\.transit([123])\.nonlinear\.batchnorm\.(weight|bias|running_mean|running_var)$", name)
    if m:
        idx, suffix = m.groups()
        suffix_map = {
            "weight": "nl.bn.weight",
            "bias": "nl.bn.bias",
            "running_mean": "nl.bn.running_mean",
            "running_var": "nl.bn.running_var",
        }
        return f"s3.se.xv.transit{idx}.{suffix_map[suffix]}"

    m = re.match(r"^xvector\.transit([123])\.linear\.weight$", name)
    if m:
        return f"s3.se.xv.transit{m.group(1)}.linear.weight"

    if name == "xvector.dense.linear.weight":
        return "s3.se.xv.dense.linear.weight"
    if name == "xvector.dense.nonlinear.batchnorm.running_mean":
        return "s3.se.xv.dense.nl.bn.running_mean"
    if name == "xvector.dense.nonlinear.batchnorm.running_var":
        return "s3.se.xv.dense.nl.bn.running_var"

    return None


def _anon_name(idx: int, arr: np.ndarray) -> str:
    shp = _shape(arr)

    # FCM head.
    if idx == 485:
        _expect(shp, (32, 1, 3, 3), "head.conv1.weight")
        return "s3.se.head.conv1.weight"
    if idx == 486:
        _expect(shp, (32,), "head.conv1.bias")
        return "s3.se.head.conv1.bias"
    if idx == 487:
        _expect(shp, (32, 32, 3, 3), "head.layer1.0.conv1.weight")
        return "s3.se.head.layer1.0.conv1.weight"
    if idx == 488:
        _expect(shp, (32,), "head.layer1.0.conv1.bias")
        return "s3.se.head.layer1.0.conv1.bias"
    if idx == 489:
        _expect(shp, (32, 32, 3, 3), "head.layer1.0.conv2.weight")
        return "s3.se.head.layer1.0.conv2.weight"
    if idx == 490:
        _expect(shp, (32,), "head.layer1.0.conv2.bias")
        return "s3.se.head.layer1.0.conv2.bias"
    if idx == 491:
        _expect(shp, (32, 32, 1, 1), "head.layer1.0.shortcut.0.weight")
        return "s3.se.head.layer1.0.shortcut.0.weight"
    if idx == 492:
        _expect(shp, (32,), "head.layer1.0.shortcut.0.bias")
        return "s3.se.head.layer1.0.shortcut.0.bias"
    if idx == 493:
        _expect(shp, (32, 32, 3, 3), "head.layer1.1.conv1.weight")
        return "s3.se.head.layer1.1.conv1.weight"
    if idx == 494:
        _expect(shp, (32,), "head.layer1.1.conv1.bias")
        return "s3.se.head.layer1.1.conv1.bias"
    if idx == 495:
        _expect(shp, (32, 32, 3, 3), "head.layer1.1.conv2.weight")
        return "s3.se.head.layer1.1.conv2.weight"
    if idx == 496:
        _expect(shp, (32,), "head.layer1.1.conv2.bias")
        return "s3.se.head.layer1.1.conv2.bias"
    if idx == 497:
        _expect(shp, (32, 32, 3, 3), "head.layer2.0.conv1.weight")
        return "s3.se.head.layer2.0.conv1.weight"
    if idx == 498:
        _expect(shp, (32,), "head.layer2.0.conv1.bias")
        return "s3.se.head.layer2.0.conv1.bias"
    if idx == 499:
        _expect(shp, (32, 32, 3, 3), "head.layer2.0.conv2.weight")
        return "s3.se.head.layer2.0.conv2.weight"
    if idx == 500:
        _expect(shp, (32,), "head.layer2.0.conv2.bias")
        return "s3.se.head.layer2.0.conv2.bias"
    if idx == 501:
        _expect(shp, (32, 32, 1, 1), "head.layer2.0.shortcut.0.weight")
        return "s3.se.head.layer2.0.shortcut.0.weight"
    if idx == 502:
        _expect(shp, (32,), "head.layer2.0.shortcut.0.bias")
        return "s3.se.head.layer2.0.shortcut.0.bias"
    if idx == 503:
        _expect(shp, (32, 32, 3, 3), "head.layer2.1.conv1.weight")
        return "s3.se.head.layer2.1.conv1.weight"
    if idx == 504:
        _expect(shp, (32,), "head.layer2.1.conv1.bias")
        return "s3.se.head.layer2.1.conv1.bias"
    if idx == 505:
        _expect(shp, (32, 32, 3, 3), "head.layer2.1.conv2.weight")
        return "s3.se.head.layer2.1.conv2.weight"
    if idx == 506:
        _expect(shp, (32,), "head.layer2.1.conv2.bias")
        return "s3.se.head.layer2.1.conv2.bias"
    if idx == 507:
        _expect(shp, (32, 32, 3, 3), "head.conv2.weight")
        return "s3.se.head.conv2.weight"
    if idx == 508:
        _expect(shp, (32,), "head.conv2.bias")
        return "s3.se.head.conv2.bias"

    # TDNN.
    if idx == 509:
        _expect(shp, (128, 320, 5), "tdnn.linear.weight")
        return "s3.se.xv.tdnn.linear.weight"
    if idx == 510:
        _expect(shp, (128,), "tdnn.linear.bias")
        return "s3.se.xv.tdnn.linear.bias"

    # Dense blocks: l1 weights and biases only.
    if 511 <= idx <= 534:
        layer = (idx - 511) // 2 + 1
        is_bias = (idx % 2) == 0
        in_ch = 128 + (layer - 1) * 32
        if not is_bias:
            _expect(shp, (128, in_ch, 1), f"block1.tdnnd{layer}.l1.weight")
            return f"s3.se.xv.block1.tdnnd{layer}.l1.weight"
        _expect(shp, (128,), f"block1.tdnnd{layer}.l1.bias")
        return f"s3.se.xv.block1.tdnnd{layer}.l1.bias"

    if 535 <= idx <= 582:
        layer = (idx - 535) // 2 + 1
        is_bias = (idx % 2) == 0
        in_ch = 256 + (layer - 1) * 32
        if not is_bias:
            _expect(shp, (128, in_ch, 1), f"block2.tdnnd{layer}.l1.weight")
            return f"s3.se.xv.block2.tdnnd{layer}.l1.weight"
        _expect(shp, (128,), f"block2.tdnnd{layer}.l1.bias")
        return f"s3.se.xv.block2.tdnnd{layer}.l1.bias"

    if 583 <= idx <= 614:
        layer = (idx - 583) // 2 + 1
        is_bias = (idx % 2) == 0
        in_ch = 512 + (layer - 1) * 32
        if not is_bias:
            _expect(shp, (128, in_ch, 1), f"block3.tdnnd{layer}.l1.weight")
            return f"s3.se.xv.block3.tdnnd{layer}.l1.weight"
        _expect(shp, (128,), f"block3.tdnnd{layer}.l1.bias")
        return f"s3.se.xv.block3.tdnnd{layer}.l1.bias"

    # transit3.
    if idx == 615:
        _expect(shp, (512, 1024, 1), "transit3.linear.weight")
        return "s3.se.xv.transit3.linear.weight"
    if idx == 616:
        _expect(shp, (512,), "transit3.linear.bias")
        return "s3.se.xv.transit3.linear.bias"

    raise SystemExit(f"unhandled anonymous tensor at index {idx}: shape={shp}")


def _renamed_tensors(model: onnx.ModelProto) -> list[tuple[str, np.ndarray]]:
    tensors: list[tuple[str, np.ndarray]] = []
    for idx, init in enumerate(model.graph.initializer):
        arr = numpy_helper.to_array(init)
        name = _map_named_tensor(init.name)
        if name is None:
            name = _anon_name(idx, arr)
        tensors.append((name, arr))
    return tensors


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="campplus.onnx path or directory")
    ap.add_argument("--output", required=True, help="Output GGUF path")
    args = ap.parse_args()

    model = _load_model(Path(args.input))
    tensors = _renamed_tensors(model)
    if len(tensors) != len(model.graph.initializer):
        raise SystemExit(f"tensor count mismatch: renamed {len(tensors)} / {len(model.graph.initializer)}")

    writer = gguf.GGUFWriter(args.output, "cosyvoice3-campplus")
    writer.add_name("Fun-CosyVoice3-0.5B-2512-CAMPPlus")
    writer.add_uint32("cosyvoice3.campplus.sample_rate", 16000)
    writer.add_uint32("cosyvoice3.campplus.n_mels", 80)
    writer.add_uint32("cosyvoice3.campplus.xvector_dim", 192)
    writer.add_uint32("cosyvoice3.campplus.head_channels", 32)
    writer.add_uint32("cosyvoice3.campplus.tdnn_channels", 128)
    writer.add_uint32("cosyvoice3.campplus.block1_layers", 12)
    writer.add_uint32("cosyvoice3.campplus.block2_layers", 24)
    writer.add_uint32("cosyvoice3.campplus.block3_layers", 16)

    n_written = 0
    n_f16 = 0
    n_f32 = 0
    for name, arr in tensors:
        force_f32 = arr.ndim <= 1 or name.endswith(".bias") or ".bn." in name or name.endswith(".running_mean") or name.endswith(".running_var")
        data = _as_float_array(arr, force_f32=force_f32)
        writer.add_tensor(name, data)
        n_written += 1
        if data.dtype == np.float16:
            n_f16 += 1
        else:
            n_f32 += 1

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"wrote {n_written} tensors to {args.output} (F16={n_f16}, F32={n_f32})")


if __name__ == "__main__":
    main()
