#!/usr/bin/env python3
"""Convert CosyVoice3 speech_tokenizer_v3 ONNX to GGUF.

This is the native GGUF converter for the Phase 6 speech tokenizer.
The ONNX graph is exported with a mix of named tensors (the 12 FSMN
blocks) and anonymous MatMul/Conv initializers. We preserve the raw
tensor data and write the key CosyVoice3 speech-tokenizer metadata the
runtime needs:

  - 12 FSMN blocks
  - 1280-d model width
  - 20 attention heads
  - 5120-d FFN
  - 31-tap FSMN memory kernel
  - 8-axis FSQ head with 3 levels per axis
  - 6561 speech-codebook size

The output is a float GGUF intended as the F16 reference model. Use
`crispasr-quantize` after this step if you want a Q4_K variant later.

Usage:
  python models/convert-cosyvoice3-s3tok-to-gguf.py \
      --input /Volumes/backups/ai/upstream/cosyvoice3-onnx/speech_tokenizer_v3.onnx \
      --output /tmp/cosyvoice3-s3tok-f16.gguf
"""

import argparse
import os
from pathlib import Path

import numpy as np

try:
    import onnx
    from onnx import numpy_helper
except ImportError as exc:  # pragma: no cover - local env dependency
    raise SystemExit("onnx is required: pip install onnx") from exc

try:
    import gguf
except ImportError:
    import sys

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ggml", "python"))
    import gguf


def _load_model(input_path: Path) -> onnx.ModelProto:
    if input_path.is_dir():
        candidate = input_path / "speech_tokenizer_v3.onnx"
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


def _shape(arr: np.ndarray) -> tuple[int, ...]:
    return tuple(int(x) for x in arr.shape)


def _expect(shape: tuple[int, ...], expected: tuple[int, ...], name: str) -> None:
    if shape != expected:
        raise SystemExit(f"unexpected tensor shape for {name}: got {shape}, expected {expected}")


def _rename_anon_tensor(idx: int, arr: np.ndarray, *, blk_idx: int | None, slot: int | None) -> str:
    shp = _shape(arr)

    if idx == 0:
        _expect(shp, (1280, 128, 3), "subsample.conv0.w")
        return "cosyvoice3.s3tok.subsample.conv0.w"
    if idx == 1:
        _expect(shp, (1280,), "subsample.conv0.b")
        return "cosyvoice3.s3tok.subsample.conv0.b"
    if idx == 2:
        _expect(shp, (1280, 1280, 3), "subsample.conv1.w")
        return "cosyvoice3.s3tok.subsample.conv1.w"
    if idx == 3:
        _expect(shp, (1280,), "subsample.conv1.b")
        return "cosyvoice3.s3tok.subsample.conv1.b"

    if blk_idx is not None:
        base = f"cosyvoice3.s3tok.blk.{blk_idx}."
        seq = [
            ("attn_ln.w", (1280,)),
            ("attn_ln.b", (1280,)),
            ("attn_q.b", (1280,)),
            ("attn_q.w", (1280, 1280)),
            ("attn_k.w", (1280, 1280)),
            ("attn_v.b", (1280,)),
            ("attn_v.w", (1280, 1280)),
            ("attn_o.b", (1280,)),
            ("attn_o.w", (1280, 1280)),
            ("mlp_ln.w", (1280,)),
            ("mlp_ln.b", (1280,)),
            ("mlp_up.b", (5120,)),
            ("mlp_up.w", (1280, 5120)),
            ("mlp_dn.b", (1280,)),
            ("mlp_dn.w", (5120, 1280)),
        ]
        if slot is None or slot < 0 or slot >= len(seq):
            raise SystemExit(f"invalid block slot {slot} for tensor {idx}")
        suffix, expected = seq[slot]
        _expect(shp, expected, f"blk.{blk_idx}.{suffix}")
        return base + suffix

    if idx == 2 + 2 + 12 * 15:
        _expect(shp, (1280, 8), "fsq.proj.w")
        return "cosyvoice3.s3tok.fsq.proj.w"

    raise SystemExit(f"unhandled anonymous tensor at index {idx}: shape={shp}")


def _renamed_tensors(model: onnx.ModelProto) -> list[tuple[str, np.ndarray]]:
    named: list[tuple[str, np.ndarray]] = []
    anon: list[np.ndarray] = []
    for init in model.graph.initializer:
        arr = numpy_helper.to_array(init)
        if init.name.startswith("blocks.") and init.name.endswith(".attn.fsmn_block.weight"):
            blk = init.name.split(".")[1]
            # FSMN depthwise conv kernel. onnx layout is (C, 1, KW); ggml_conv_1d_dw
            # wants ne=[KW, 1, C]. gguf-py reverses the numpy shape when writing to
            # ggml ne, so pass the onnx array as-is: reverse(C,1,KW) -> ne=(KW,1,C).
            named.append((f"cosyvoice3.s3tok.blk.{blk}.attn.fsmn_block.w", np.ascontiguousarray(arr)))
        elif init.name == "quantizer.project_in.bias":
            named.append(("cosyvoice3.s3tok.fsq.proj.b", arr))
        else:
            anon.append(arr)

    # gguf-py reverses the numpy shape to obtain the ggml ne ordering, and the
    # C++ runtime consumes every tensor as-is (no transpose at load). So we must
    # emit each weight in the numpy layout whose reverse is the ggml layout the
    # graph expects:
    #   - conv1d kernels:        ggml ne=[KW, IC, OC]  <- onnx (OC, IC, KW) as-is
    #   - 2D MatMul/Linear .w:   ggml ne=[in, out]     <- onnx (in, out) -> .T (out, in)
    #   - 1D bias / LayerNorm:   unchanged
    matmul_w_suffixes = (".attn_q.w", ".attn_k.w", ".attn_v.w", ".attn_o.w", ".mlp_up.w", ".mlp_dn.w", ".fsq.proj.w")

    def _layout(name: str, arr: np.ndarray) -> np.ndarray:
        if arr.ndim == 2 and name.endswith(matmul_w_suffixes):
            return np.ascontiguousarray(arr.T)
        # conv1d kernels (ndim==3) and everything else: keep onnx order; gguf's
        # reverse yields the [KW, IC, OC] / [in, out] the runtime needs.
        return np.ascontiguousarray(arr)

    out: list[tuple[str, np.ndarray]] = []
    for i, arr in enumerate(anon):
        if i < 4:
            name = _rename_anon_tensor(i, arr, blk_idx=None, slot=None)
            out.append((name, _layout(name, arr)))
            continue
        if i < 4 + 12 * 15:
            rel = i - 4
            blk = rel // 15
            slot = rel % 15
            name = _rename_anon_tensor(i, arr, blk_idx=blk, slot=slot)
            out.append((name, _layout(name, arr)))
            continue
        name = _rename_anon_tensor(i, arr, blk_idx=None, slot=None)
        out.append((name, _layout(name, arr)))

    return named + out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="speech_tokenizer_v3.onnx path or directory")
    ap.add_argument("--output", required=True, help="Output GGUF path")
    args = ap.parse_args()

    model = _load_model(Path(args.input))

    writer = gguf.GGUFWriter(args.output, "cosyvoice3-s3tok")
    writer.add_name("Fun-CosyVoice3-0.5B-2512-speech_tokenizer_v3")
    writer.add_uint32("cosyvoice3.s3tok.n_blocks", 12)
    writer.add_uint32("cosyvoice3.s3tok.d_model", 1280)
    writer.add_uint32("cosyvoice3.s3tok.n_heads", 20)
    writer.add_uint32("cosyvoice3.s3tok.ff_dim", 5120)
    writer.add_uint32("cosyvoice3.s3tok.fsmn_kernel", 31)
    writer.add_uint32("cosyvoice3.s3tok.codebook_dim", 8)
    writer.add_uint32("cosyvoice3.s3tok.codebook_levels", 3)
    writer.add_uint32("cosyvoice3.s3tok.codebook_size", 6561)

    tensors = _renamed_tensors(model)
    if len(tensors) != len(model.graph.initializer):
        raise SystemExit(f"tensor count mismatch: renamed {len(tensors)} / {len(model.graph.initializer)}")
    n_written = 0
    for name, arr in tensors:
        data = _as_float_array(arr, force_f32=(arr.ndim <= 1 or name.endswith(".b")))
        writer.add_tensor(name, data)
        n_written += 1

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"wrote {n_written} tensors to {args.output}")


if __name__ == "__main__":
    main()
