#!/usr/bin/env python3
"""Convert Silero Language Classifier 95 (ONNX) to GGUF.

Source: https://huggingface.co/deepghs/silero-lang95-onnx

Architecture: 8 stage pairs (conv + transformer), attention pooling, 2 classifiers.
See task #56 in the task list for the full architecture description.

ONNX numeric initializer assignment (confirmed by consumer-node walk):
  Transformer blocks appear in order. Each block has 4 MatMul weights:
    1. QKV combined (dim, 3*dim)
    2. out_proj (dim, dim)
    3. linear1 / FFN (dim, dim) — no expansion, bottleneck FFN
    4. linear2 / FFN (dim, dim)
  Plus a 1×1 Conv projection per stage boundary.

  Biases are named (encoder.N.attention.QKV.bias, etc.) and pass through directly.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np

try: import onnx
except ImportError: sys.exit("pip install onnx")
try: import gguf
except ImportError: sys.exit("pip install gguf")


# Stage layout: (conv_stage_id, tx_stage_id, dim)
STAGES = [
    (0,  1,  128), (5,  6,  128), (10, 11, 128), (15, 16, 128),
    (20, 21, 192), (25, 26, 192), (30, 31, 192), (35, 36, 192),
]


def convert(input_dir: Path, out_path: Path) -> None:
    onnx_path = input_dir / "lang_classifier_95.onnx"
    print(f"Loading: {onnx_path}")
    model = onnx.load(str(onnx_path))

    # Language + group dicts
    with open(input_dir / "lang_dict_95.json") as f:
        lang_dict = json.load(f)
    with open(input_dir / "lang_group_dict_95.json") as f:
        group_dict = json.load(f)
    lang_list = [""] * 95
    for k, v in lang_dict.items(): lang_list[int(k)] = v
    group_list = [""] * 58
    for k, v in group_dict.items():
        group_list[int(k)] = json.dumps(v) if not isinstance(v, str) else v

    # Collect all initializers
    inits = {}
    for w in model.graph.initializer:
        inits[w.name] = onnx.numpy_helper.to_array(w).astype(np.float32)

    # The initial frame-extraction Conv has stride=160 and uses a Constant
    # node (not an initializer). Extract it from the ONNX graph.
    for n in model.graph.node:
        if n.op_type == 'Constant' and n.output[0] in [
            inp for node in model.graph.node if node.op_type == 'Conv'
            for inp in node.input[1:2]
            if any(a.name == 'strides' and list(a.ints) == [160] for a in node.attribute)
        ]:
            for a in n.attribute:
                if a.name == 'value':
                    arr = onnx.numpy_helper.to_array(a.t).astype(np.float32)
                    inits['_frontend_conv_weight'] = arr
                    print(f"  frontend conv: {arr.shape} (stride=160, learned STFT)")
                    break

    # Sort numeric IDs
    numeric_ids = sorted([int(k) for k in inits if k.isdigit()])

    # Separate by role: 1×1 Conv weight (3D) vs MatMul weight (2D)
    conv1x1_w_ids = [i for i in numeric_ids if len(inits[str(i)].shape) == 3]
    matmul_ids    = [i for i in numeric_ids if len(inits[str(i)].shape) == 2]

    # Pair each 1×1 Conv weight with its bias (the next numeric ID, 1D, same dim).
    conv1x1_pairs = []
    for wid in conv1x1_w_ids:
        bid = wid + 1  # biases are always the next ID (2040→2041, 2043→2044, ...)
        if str(bid) in inits and len(inits[str(bid)].shape) == 1:
            conv1x1_pairs.append((wid, bid))
        else:
            conv1x1_pairs.append((wid, None))

    # MatMul: 4 per transformer block × 8 blocks = 32
    # Order: QKV, out_proj, linear1, linear2
    matmul_groups = []
    for bi in range(0, len(matmul_ids), 4):
        matmul_groups.append(matmul_ids[bi:bi+4])

    print(f"  conv1x1 pairs: {len(conv1x1_pairs)}")
    print(f"  matmul groups: {len(matmul_groups)} (× 4 each)")
    print(f"  stages: {len(STAGES)}")

    # ---- Write GGUF ----
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = gguf.GGUFWriter(str(out_path), arch="silero_lid")

    writer.add_uint32("silero_lid.sample_rate", 16000)
    writer.add_uint32("silero_lid.n_langs", 95)
    writer.add_uint32("silero_lid.n_groups", 58)
    writer.add_uint32("silero_lid.n_stages", 8)
    writer.add_uint32("silero_lid.conv_blocks_per_stage", 12)
    writer.add_uint32("silero_lid.conv_kernel", 5)
    # Front-end: learned Conv1d(1→322, kernel=320, stride=160) = 100 Hz framing
    writer.add_uint32("silero_lid.frontend_channels", 322)
    writer.add_uint32("silero_lid.frontend_kernel", 320)
    writer.add_uint32("silero_lid.frontend_stride", 160)
    # Stage-boundary stride-2 downsampling after each of the first 4 (128-dim) stages
    writer.add_uint32("silero_lid.n_downsample_stages", 4)
    writer.add_array("silero_lid.lang_strs", lang_list)
    writer.add_array("silero_lid.group_strs", group_list)

    n_written = 0

    def write(name, arr):
        nonlocal n_written
        writer.add_tensor(name, arr.astype(np.float32))
        n_written += 1

    # ---- Named tensors: conv stages ----
    for si, (conv_id, tx_id, dim) in enumerate(STAGES):
        # Conv blocks (12 per stage)
        for bi in range(12):
            prefix = f"encoder.{conv_id}.{bi}"
            for sub in ["dw_conv.0.weight", "dw_conv.0.bias", "pw_conv.0.weight", "pw_conv.0.bias"]:
                key = f"{prefix}.{sub}"
                if key in inits:
                    gguf_name = f"lid.conv.{si}.{bi}.{sub.replace('.0.', '.')}"
                    write(gguf_name, inits[key])
            # Residual proj (only on last block of some stages)
            for sub in ["proj.weight", "proj.bias"]:
                key = f"{prefix}.{sub}"
                if key in inits:
                    write(f"lid.conv.{si}.{bi}.{sub}", inits[key])

        # Transformer block — named biases
        for sub, gguf_sub in [
            ("attention.QKV.bias", "tx.qkv.bias"),
            ("attention.out_proj.bias", "tx.out.bias"),
            ("linear1.bias", "tx.ff1.bias"),
            ("linear2.bias", "tx.ff2.bias"),
            ("norm1.weight", "tx.norm1.weight"),
            ("norm1.bias", "tx.norm1.bias"),
            ("norm2.weight", "tx.norm2.weight"),
            ("norm2.bias", "tx.norm2.bias"),
        ]:
            key = f"encoder.{tx_id}.{sub}"
            if key in inits:
                write(f"lid.{si}.{gguf_sub}", inits[key])

    # ---- Numeric tensor assignment ----
    # Conv1x1 projections: one per stage
    for si, (wid, bid) in enumerate(conv1x1_pairs[:len(STAGES)]):
        write(f"lid.{si}.tx.conv1x1.weight", inits[str(wid)])
        if bid is not None:
            write(f"lid.{si}.tx.conv1x1.bias", inits[str(bid)])

    # MatMul weights: 4 per block (QKV, out, ff1, ff2)
    for si, group in enumerate(matmul_groups[:len(STAGES)]):
        roles = ["tx.qkv.weight", "tx.out.weight", "tx.ff1.weight", "tx.ff2.weight"]
        for role, mid in zip(roles, group):
            write(f"lid.{si}.{role}", inits[str(mid)])

    # ---- Front-end Conv weight (learned STFT, stride=160) ----
    if '_frontend_conv_weight' in inits:
        write("lid.frontend.weight", inits['_frontend_conv_weight'])
    else:
        print("  WARNING: frontend conv weight not found in ONNX graph!")

    # ---- Top-level tensors ----
    write("lid.adaptive_norm.filter", inits["adaptive_normalization.filter_"])
    write("lid.pool.weight", inits["attention.attention_weights"])
    write("lid.lang.weight", inits["lang_classifier.weight"])
    write("lid.lang.bias", inits["lang_classifier.bias"])
    write("lid.group.weight", inits["group_classifier.weight"])
    write("lid.group.bias", inits["group_classifier.bias"])

    print(f"\n  total: {n_written} tensors")
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"Done: {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()
    convert(args.input, args.output)
