#!/usr/bin/env python3
"""Convert X-ASR (icefall streaming Zipformer2 transducer) to GGUF — #436.

The source is the sherpa-onnx export upstream ships
(`deployment/models/chunk-*ms-model/{encoder,decoder,joiner}-*.onnx` +
`tokens.txt`). The training checkpoint in the repo (`streaming_exp/pretrained.pt`)
is NOT the exported model — every one of its tensors differs from the ONNX
weights — so it is not used.

  python models/convert-xasr-to-gguf.py --models-dir deployment/models \\
      --output x-asr-zh-en-f16.gguf

The four chunk exports share every weight bit for bit (checked by
tools/kaggle/xasr-onnx-graph); they differ only in chunk size and left
context, so one GGUF serves all four and records their table.

Recovering names from the ONNX graph (docs/xasr/PLAN.md):
  * named initializers keep their PyTorch names;
  * the 344 Linear weights are anonymous (`onnx::MatMul_*`, stored transposed)
    and appear in the encoder's execution order: embed.out, then per layer
    in_proj, linear_pos, ff1 in/out, nonlin in/out, attn1 in/out, conv1 in/out,
    ff2 in/out, attn2 in/out, conv2 in/out, ff3 in/out, then encoder_proj —
    matched in order and checked shape by shape;
  * each chunkwise-conv scale table (2, C, K) was split by constant folding
    into two anonymous (C, K) rows (left edge, right edge), matched in
    first-use order and restacked;
  * the downsample softmax weights were constant-folded to (ds, 1, 1)
    constants; they are stored as log-weights so softmax() restores them.

Tensor names are shortened to fit GGUF's 64-byte limit (see rename()).
Matrices go to F16; convolutions, 1-D tensors and scale tables stay F32.
"""
import argparse
import re
import sys
from pathlib import Path

import numpy as np

try:
    import gguf
except ImportError:
    sys.exit("pip install gguf")

LAYER_LINEARS = [
    "self_attn_weights.in_proj", "self_attn_weights.linear_pos", "feed_forward1.in_proj", "feed_forward1.out_proj",
    "nonlin_attention.in_proj", "nonlin_attention.out_proj", "self_attn1.in_proj", "self_attn1.out_proj",
    "conv_module1.in_proj", "conv_module1.out_proj", "feed_forward2.in_proj", "feed_forward2.out_proj",
    "self_attn2.in_proj", "self_attn2.out_proj", "conv_module2.in_proj", "conv_module2.out_proj",
    "feed_forward3.in_proj", "feed_forward3.out_proj",
]

SUB = [
    ("self_attn_weights.in_proj.", "aw.in."),
    ("self_attn_weights.linear_pos.", "aw.pos."),
    ("self_attn1.in_proj.", "sa1.in."),
    ("self_attn1.out_proj.", "sa1.out."),
    ("self_attn2.in_proj.", "sa2.in."),
    ("self_attn2.out_proj.", "sa2.out."),
    ("feed_forward1.in_proj.", "ff1.in."),
    ("feed_forward1.out_proj.", "ff1.out."),
    ("feed_forward2.in_proj.", "ff2.in."),
    ("feed_forward2.out_proj.", "ff2.out."),
    ("feed_forward3.in_proj.", "ff3.in."),
    ("feed_forward3.out_proj.", "ff3.out."),
    ("nonlin_attention.in_proj.", "na.in."),
    ("nonlin_attention.out_proj.", "na.out."),
    ("depthwise_conv.chunkwise_conv_scale", "chunk_scale"),
    ("depthwise_conv.causal_conv.", "causal."),
    ("depthwise_conv.chunkwise_conv.", "chunk."),
    ("conv_module1.in_proj.", "cv1.in."),
    ("conv_module1.out_proj.", "cv1.out."),
    ("conv_module2.in_proj.", "cv2.in."),
    ("conv_module2.out_proj.", "cv2.out."),
    ("conv_module1.", "cv1."),
    ("conv_module2.", "cv2."),
    ("bypass_mid.bypass_scale", "bypass_mid"),
    ("bypass.bypass_scale", "bypass"),
]


def ints(s):
    return [int(x) for x in str(s).split(",")]


def layer_prefix(s, l, ds):
    return f"encoder.encoders.{s}." + ("encoder." if ds[s] > 1 else "") + f"layers.{l}."


def onnx_state_dict(chunk_dir):
    """PyTorch-named state dict + hyper-parameters from one sherpa-onnx chunk export."""
    import onnx
    from onnx import numpy_helper

    chunk_dir = Path(chunk_dir)
    enc = next(chunk_dir.glob("encoder-*.onnx"))
    dec = next(chunk_dir.glob("decoder-*.onnx"))
    joi = next(chunk_dir.glob("joiner-*.onnx"))
    m = onnx.load(str(enc))
    meta = {p.key: p.value for p in m.metadata_props}
    inits = {t.name: t for t in m.graph.initializer}
    consts = {}
    first_use = {}
    for i, nd in enumerate(m.graph.node):
        if nd.op_type == "Constant":
            for a in nd.attribute:
                if a.name == "value":
                    consts[nd.output[0]] = numpy_helper.to_array(a.t)
        for x in nd.input:
            first_use.setdefault(x, i)
    arr = lambda n: numpy_helper.to_array(inits[n])  # noqa: E731

    hp = {
        "n_layers": ints(meta["num_encoder_layers"]), "dims": ints(meta["encoder_dims"]),
        "kernel": ints(meta["cnn_module_kernels"]), "n_heads": ints(meta["num_heads"]),
        "qd": ints(meta["query_head_dims"]), "vd": ints(meta["value_head_dims"]),
        "decode_chunk_len": int(meta["decode_chunk_len"]), "T": int(meta["T"]),
    }
    left = ints(meta["left_context_len"])
    hp["left_context_frames"] = left[0]
    hp["downsample"] = [left[0] // v for v in left]
    ds = hp["downsample"]
    S = len(hp["dims"])

    sd = {n: arr(n) for n in inits if not n.startswith("onnx::")}
    sd = {re.sub(r"^encoder_proj\.", "joiner.encoder_proj.", k): v for k, v in sd.items()}

    # Linear weights: anonymous MatMul initializers in execution order.
    names = ["encoder_embed.out.weight"]
    for s in range(S):
        for l in range(hp["n_layers"][s]):
            names += [layer_prefix(s, l, ds) + n + ".weight" for n in LAYER_LINEARS]
    names.append("joiner.encoder_proj.weight")
    anon = [nd.input[1] for nd in m.graph.node
            if nd.op_type == "MatMul" and len(nd.input) > 1 and nd.input[1] in inits and nd.input[1].startswith("onnx::")]
    if len(anon) != len(names):
        sys.exit(f"{enc}: {len(anon)} anonymous MatMul weights, expected {len(names)}")
    for n, a in zip(names, anon):
        w = arr(a).T.copy()  # MatMul(x, W^T): stored (in, out)
        b = n[:-len("weight")] + "bias"
        if b in sd and sd[b].shape[0] != w.shape[0]:
            sys.exit(f"order mismatch at {n}: weight out {w.shape[0]} vs bias {sd[b].shape[0]}")
        sd[n] = w
    for s in range(S):
        d, H = hp["dims"][s], hp["n_heads"][s]
        pre = layer_prefix(s, 0, ds)
        exp = {"self_attn_weights.linear_pos": None, "nonlin_attention.in_proj": (3 * (3 * d // 4), d),
               "self_attn1.in_proj": (H * hp["vd"][s], d), "conv_module1.in_proj": (2 * d, d)}
        for k, shp in exp.items():
            if shp and sd[pre + k + ".weight"].shape != shp:
                sys.exit(f"shape check failed: {pre + k} {sd[pre + k + '.weight'].shape} != {shp}")

    # Chunkwise-conv scale tables. Constant folding split each (2, C, K)
    # parameter into its two rows, left_edge = scale[0] and right_edge = scale[1]:
    # two anonymous (C, K) initializers that feed a Slice (chunk < K) or a Concat
    # (chunk >= K). _get_chunk_scale touches the left edge first, so in first-use
    # order they come as (left, right) per conv module.
    mm_inputs = {nd.input[1] for nd in m.graph.node if nd.op_type == "MatMul" and len(nd.input) > 1}
    edges = sorted((first_use.get(n, 1 << 30), n) for n, t in inits.items()
                   if n.startswith("onnx::") and len(t.dims) == 2 and n not in mm_inputs)
    want = [(layer_prefix(s, l, ds) + f"conv_module{c}.depthwise_conv.chunkwise_conv_scale", (hp["dims"][s], hp["kernel"][s]))
            for s in range(S) for l in range(hp["n_layers"][s]) for c in (1, 2)]
    if len(edges) != 2 * len(want):
        sys.exit(f"{enc}: {len(edges)} anonymous 2-D non-MatMul initializers, expected {2 * len(want)} (C, K) edges")
    for i, (n, shp) in enumerate(want):
        left, right = arr(edges[2 * i][1]), arr(edges[2 * i + 1][1])
        if left.shape != shp or right.shape != shp:
            sys.exit(f"chunk scale order mismatch at {n}: {left.shape}/{right.shape} != {shp}")
        sd[n] = np.stack([left, right])

    # Downsample weights: constant-folded softmax(bias) of shape (ds, 1, 1).
    folded = []
    for n, v in list(consts.items()) + [(n, arr(n)) for n in inits if n.startswith("onnx::")]:
        if v.ndim == 3 and v.shape[1:] == (1, 1) and v.shape[0] in (2, 4, 8) and abs(float(v.sum()) - 1.0) < 1e-4:
            folded.append((first_use.get(n, 1 << 30), v.reshape(-1)))
    folded.sort(key=lambda x: x[0])
    want_ds = [(f"encoder.encoders.{s}.downsample.bias", ds[s]) for s in range(S) if ds[s] > 1]
    want_ds.append(("encoder.downsample_output.bias", 2))
    if [len(v) for _, v in folded] != [k for _, k in want_ds]:
        sys.exit(f"downsample weights: found sizes {[len(v) for _, v in folded]}, expected {[k for _, k in want_ds]}")
    for (n, _), (_, w) in zip(want_ds, folded):
        sd[n] = np.log(w.astype(np.float64)).astype(np.float32)  # softmax(log w) == w

    md = onnx.load(str(dec))
    dmeta = {p.key: p.value for p in md.metadata_props}
    for t in md.graph.initializer:
        k = re.sub(r"^decoder_proj\.", "joiner.decoder_proj.", t.name)
        sd[k] = numpy_helper.to_array(t)
    mj = onnx.load(str(joi))
    for t in mj.graph.initializer:
        sd["joiner." + t.name] = numpy_helper.to_array(t)

    hp["context_size"] = int(dmeta["context_size"])
    hp["vocab_size"] = int(dmeta["vocab_size"])
    hp["pos_dim"] = sd[layer_prefix(0, 0, ds) + "self_attn_weights.linear_pos.weight"].shape[1]
    hp["pd"] = [sd[layer_prefix(s, 0, ds) + "self_attn_weights.linear_pos.weight"].shape[0] // hp["n_heads"][s] for s in range(S)]
    hp["ffn_dim"] = [sd[layer_prefix(s, 0, ds) + "feed_forward2.in_proj.weight"].shape[0] for s in range(S)]
    hp["decoder_dim"] = sd["decoder.embedding.weight"].shape[1]
    hp["joiner_dim"] = sd["joiner.output_linear.weight"].shape[1]
    hp["feature_dim"] = [d.dim_value for d in m.graph.input[0].type.tensor_type.shape.dim][2]
    return sd, hp


def rename(k):
    """PyTorch name -> GGUF name, or None to drop."""
    m = re.match(r"^encoder\.encoders\.(\d+)\.(?:encoder\.)?layers\.(\d+)\.(.+)$", k)
    if m:
        s, l, rest = m.groups()
        for a, b in SUB:
            rest = rest.replace(a, b)
        if not re.fullmatch(r"(aw|sa[12]|ff[123]|na|cv[12])\.[a-z_.]+|norm\.(bias|log_scale)|bypass(_mid)?", rest):
            raise KeyError(f"unmapped layer tensor: {k} -> {rest}")
        return f"z.{s}.{l}.{rest}"
    m = re.match(r"^encoder\.encoders\.(\d+)\.(downsample\.bias|out_combiner\.bypass_scale)$", k)
    if m:
        return f"z.{m.group(1)}." + ("ds_bias" if m.group(2).startswith("downsample") else "combiner")
    if k == "encoder.downsample_output.bias":
        return "z.out_ds_bias"
    if k.startswith("encoder_embed."):
        r = k[len("encoder_embed."):]
        r = r.replace("conv.0.", "conv0.").replace("conv.4.", "conv1.").replace("conv.7.", "conv2.")
        r = r.replace("convnext.depthwise_conv.", "cnx.dw.").replace("convnext.pointwise_conv1.", "cnx.pw1.")
        r = r.replace("convnext.pointwise_conv2.", "cnx.pw2.").replace("out_norm.", "norm.")
        return "emb." + r
    if k.startswith("decoder."):
        return "dec." + k[len("decoder."):].replace("embedding.", "emb.")
    if k.startswith("joiner."):
        return "join." + k[len("joiner."):]
    raise KeyError(f"unmapped tensor: {k}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models-dir", required=True, help="deployment/models (holds chunk-*ms-model/)")
    ap.add_argument("--output", required=True)
    ap.add_argument("--name", default="x-asr-zh-en")
    a = ap.parse_args()

    dirs = sorted(Path(a.models_dir).glob("chunk-*ms-model"), key=lambda p: int(re.search(r"(\d+)ms", p.name).group(1)))
    if not dirs:
        sys.exit(f"no chunk-*ms-model/ under {a.models_dir}")
    sd, hp = onnx_state_dict(dirs[0])
    table = []
    import onnx
    for d in dirs:
        meta = {p.key: p.value for p in onnx.load(str(next(d.glob("encoder-*.onnx"))), load_external_data=False).metadata_props}
        for k in ("num_encoder_layers", "encoder_dims", "cnn_module_kernels", "num_heads"):
            if ints(meta[k]) != hp[{"num_encoder_layers": "n_layers", "encoder_dims": "dims",
                                    "cnn_module_kernels": "kernel", "num_heads": "n_heads"}[k]]:
                sys.exit(f"{d}: architecture differs from {dirs[0]}")
        table.append((int(meta["decode_chunk_len"]) * 10, ints(meta["left_context_len"])[0]))

    toks = []
    for line in open(dirs[0] / "tokens.txt", encoding="utf-8"):
        line = line.rstrip("\n")
        if not line:
            continue
        tok, idx = line.rsplit(" ", 1)
        if int(idx) != len(toks):
            sys.exit(f"tokens.txt: id {idx} out of order")
        toks.append(tok)
    if len(toks) != hp["vocab_size"]:
        sys.exit(f"tokens.txt has {len(toks)} entries, vocab_size is {hp['vocab_size']}")

    w = gguf.GGUFWriter(a.output, "xasr")
    w.add_name(a.name)
    for key, gk in (("n_layers", "n_layers"), ("downsample", "downsample"), ("ffn_dim", "ffn_dim"),
                    ("n_heads", "n_heads"), ("dims", "dims"), ("qd", "query_head_dim"), ("vd", "value_head_dim"),
                    ("pd", "pos_head_dim"), ("kernel", "conv_kernel")):
        w.add_array(f"xasr.{gk}", [int(x) for x in hp[key]])
    for key in ("pos_dim", "decoder_dim", "joiner_dim", "context_size", "vocab_size", "feature_dim"):
        w.add_uint32(f"xasr.{key}", int(hp[key]))
    w.add_uint32("xasr.blank_id", 0)
    w.add_uint32("xasr.unk_id", toks.index("<unk>") if "<unk>" in toks else 0xFFFFFFFF)
    w.add_array("xasr.chunk_ms", [c for c, _ in table])
    w.add_array("xasr.left_context_frames", [l for _, l in table])
    w.add_array("tokenizer.ggml.tokens", toks)

    cnt = 0
    for k in sorted(sd):
        name = rename(k)
        if name is None:
            continue
        arr = np.asarray(sd[k], dtype=np.float32)
        if name.startswith(("z.", "join.", "dec.emb", "emb.out.")) and arr.ndim == 2:
            arr = arr.astype(np.float16)
        if arr.ndim == 0:
            arr = arr.reshape(1)
        w.add_tensor(name, arr)
        cnt += 1
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"wrote {a.output}: {cnt} tensors, chunks {table}, vocab {len(toks)}")


if __name__ == "__main__":
    main()
