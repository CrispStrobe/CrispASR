#!/usr/bin/env python3
"""Convert X-ASR (icefall streaming Zipformer2 transducer) to GGUF — #436.

Input is the upstream training checkpoint `streaming_exp/pretrained.pt`
(model hyper-parameters are stored in it) plus any chunk folder's
`tokens.txt`. One GGUF serves every chunk size: chunk size and left context
are runtime choices of a Zipformer2, and the four sherpa-onnx exports differ
only in them (docs/xasr/PLAN.md).

  python models/convert-xasr-to-gguf.py --pt pretrained.pt --tokens tokens.txt \\
      --state model_avg --output x-asr-zh-en-f16.gguf

`--state` picks which weights of the checkpoint to export (`model` or
`model_avg`); use the one the sherpa-onnx exports carry — the reference
kernel checks this against the ONNX initializers before converting.

Tensor names are shortened to fit GGUF's 64-byte limit:
  encoder_embed.*                          -> emb.*
  encoder.encoders.S[.encoder].layers.L.*  -> z.S.L.*   (module names abbreviated)
  encoder.encoders.S.downsample.bias       -> z.S.ds_bias
  encoder.encoders.S.out_combiner.*        -> z.S.combiner
  encoder.downsample_output.bias           -> z.out_ds_bias
  decoder.* / joiner.*                     -> dec.* / join.*
Matrices go to F16; convolutions, every 1-D tensor and the small scale
tables stay F32. simple_am_proj / simple_lm_proj (pruned-RNNT training
heads) and the unused per-layer `bypass_scale` are dropped.
"""
import argparse
import re
import sys

import numpy as np
import torch

try:
    import gguf
except ImportError:
    sys.exit("pip install gguf")

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


def rename(k):
    """checkpoint name -> GGUF name, or None to drop."""
    if k.startswith(("simple_am_proj.", "simple_lm_proj.")):
        return None
    m = re.match(r"^encoder\.encoders\.(\d+)\.(?:encoder\.)?layers\.(\d+)\.(.+)$", k)
    if m:
        s, l, rest = m.groups()
        if rest == "bypass_scale":  # "TODO: remove it" in icefall; never read
            return None
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


def ints(v):
    if isinstance(v, int):
        return [v]
    return [int(x) for x in str(v).split(",")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pt", required=True)
    ap.add_argument("--tokens", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--state", default="model_avg", choices=["model", "model_avg"])
    ap.add_argument("--name", default="x-asr-zh-en")
    a = ap.parse_args()

    ck = torch.load(a.pt, map_location="cpu", weights_only=False)
    sd = ck[a.state]
    if not ck.get("causal", False):
        sys.exit("only the causal (streaming) Zipformer2 is implemented")
    if ck.get("use_ctc") or ck.get("use_attention_decoder") or not ck.get("use_transducer", True):
        sys.exit("expected a transducer-only checkpoint")

    toks = []
    for line in open(a.tokens, encoding="utf-8"):
        line = line.rstrip("\n")
        if not line:
            continue
        tok, idx = line.rsplit(" ", 1)
        if int(idx) != len(toks):
            sys.exit(f"tokens.txt: id {idx} out of order")
        toks.append(tok)
    if len(toks) != int(ck["vocab_size"]):
        sys.exit(f"tokens.txt has {len(toks)} entries, vocab_size is {ck['vocab_size']}")

    w = gguf.GGUFWriter(a.output, "xasr")
    w.add_name(a.name)
    ds = ints(ck["downsampling_factor"])
    n = len(ds)

    def per_stack(key):
        v = ints(ck[key])
        return v * n if len(v) == 1 else v

    for key, gk in (("num_encoder_layers", "n_layers"), ("downsampling_factor", "downsample"),
                    ("feedforward_dim", "ffn_dim"), ("num_heads", "n_heads"), ("encoder_dim", "dims"),
                    ("query_head_dim", "query_head_dim"), ("value_head_dim", "value_head_dim"),
                    ("pos_head_dim", "pos_head_dim"), ("cnn_module_kernel", "conv_kernel")):
        w.add_array(f"xasr.{gk}", per_stack(key))
    for key in ("pos_dim", "decoder_dim", "joiner_dim", "context_size", "vocab_size", "blank_id", "feature_dim"):
        w.add_uint32(f"xasr.{key}", int(ck[key]))
    w.add_uint32("xasr.unk_id", toks.index("<unk>") if "<unk>" in toks else 0xFFFFFFFF)
    # Chunk sizes (100 Hz frames per decode chunk) and the left context (50 Hz
    # frames) the four upstream sherpa-onnx exports were made with.
    w.add_array("xasr.chunk_ms", [160, 480, 960, 1920])
    w.add_array("xasr.left_context_frames", [96, 256, 256, 256])
    w.add_array("tokenizer.ggml.tokens", toks)

    cnt = 0
    for k, t in sd.items():
        name = rename(k)
        if name is None:
            continue
        arr = t.detach().float().numpy()
        if name.startswith(("z.", "join.", "dec.emb", "emb.out.")) and arr.ndim == 2 and "chunk_scale" not in name:
            arr = arr.astype(np.float16)
        else:
            arr = arr.astype(np.float32)
        if arr.ndim == 0:
            arr = arr.reshape(1)
        w.add_tensor(name, arr)
        cnt += 1
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"wrote {a.output}: {cnt} tensors from '{a.state}', vocab {len(toks)}")


if __name__ == "__main__":
    main()
