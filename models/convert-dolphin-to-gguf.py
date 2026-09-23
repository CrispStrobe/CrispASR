#!/usr/bin/env python3
"""Convert a Dolphin (DataoceanAI) WeNet checkpoint to GGUF — #436.

Covers the Dolphin family (base / small / cn-dialect / cn-dialect-streaming):
E-Branchformer encoder + Transformer decoder + CTC. Inputs are the files the
HF repo ships: `<name>.pt`, `train.yaml`, `units.txt`.

  python models/convert-dolphin-to-gguf.py --pt small.cn.streaming.pt \\
      --yaml train.yaml --units units.txt --output dolphin-cn-dialect-small-streaming-f16.gguf

Matrices go to F16, everything 1-D (norms, biases, CMVN, pos_bias_*) and the
positional tables stay F32. The optional hotword `context_module` is skipped
(v1 has no deep biasing). Read docs/dolphin/PLAN.md for the blueprint facts
this layout encodes — notably that "rel_pos" uses ABSOLUTE interleaved
sinusoids (`pe` is shipped verbatim) and that sos/eos come from the special
token table (<sos>=2, <eos>=3), not vocab-1.
"""
import argparse
import re
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

try:
    import gguf
except ImportError:
    sys.exit("pip install gguf")


def load_units(path):
    toks = []
    for line in open(path, encoding="utf-8"):
        line = line.rstrip("\n")
        if not line:
            continue
        tok, idx = line.rsplit(" ", 1)
        idx = int(idx)
        if idx != len(toks):
            sys.exit(f"units.txt: id {idx} out of order at '{tok}'")
        toks.append(tok)
    return toks


def rename(k):
    """WeNet state-dict name -> GGUF name, or None to skip."""
    if k.startswith("context_module."):
        return None
    r = k
    r = r.replace("encoder.global_cmvn.", "enc.cmvn.")
    r = re.sub(r"^encoder\.embed\.conv\.(\d+)\.", lambda m: f"enc.sub.conv{int(m.group(1)) // 2}.", r)
    r = r.replace("encoder.embed.out.0.", "enc.sub.out.")
    r = r.replace("encoder.embed.pos_enc.pe", "enc.pe")
    r = r.replace("encoder.after_norm.", "enc.norm_out.")
    r = re.sub(r"^encoder\.encoders\.(\d+)\.", r"enc.blk.\1.", r)
    r = r.replace(".attn.linear_q.", ".attn.q.").replace(".attn.linear_k.", ".attn.k.")
    r = r.replace(".attn.linear_v.", ".attn.v.").replace(".attn.linear_out.", ".attn.o.")
    r = r.replace(".attn.linear_pos.", ".attn.pos.")
    r = r.replace(".cgmlp.channel_proj1.0.", ".cgmlp.proj1.").replace(".cgmlp.channel_proj2.", ".cgmlp.proj2.")
    r = r.replace(".cgmlp.csgu.norm.", ".cgmlp.csgu_norm.").replace(".cgmlp.csgu.conv.", ".cgmlp.csgu_conv.")
    r = r.replace(".feed_forward_macaron.w_1.", ".ffm.w1.").replace(".feed_forward_macaron.w_2.", ".ffm.w2.")
    r = r.replace(".feed_forward.w_1.", ".ff.w1.").replace(".feed_forward.w_2.", ".ff.w2.")
    r = r.replace(".depthwise_conv_fusion.", ".merge_conv.").replace(".merge_proj.", ".merge_proj.")
    r = r.replace("decoder.embed.0.weight", "dec.embed.weight").replace("decoder.embed.1.pe", "dec.pe")
    r = r.replace("decoder.after_norm.", "dec.norm_out.").replace("decoder.output_layer.", "dec.out.")
    r = re.sub(r"^decoder\.decoders\.(\d+)\.", r"dec.blk.\1.", r)
    r = r.replace(".self_attn.linear_q.", ".self_attn.q.").replace(".self_attn.linear_k.", ".self_attn.k.")
    r = r.replace(".self_attn.linear_v.", ".self_attn.v.").replace(".self_attn.linear_out.", ".self_attn.o.")
    r = r.replace(".src_attn.linear_q.", ".src_attn.q.").replace(".src_attn.linear_k.", ".src_attn.k.")
    r = r.replace(".src_attn.linear_v.", ".src_attn.v.").replace(".src_attn.linear_out.", ".src_attn.o.")
    r = r.replace("ctc.ctc_lo.", "ctc.")
    if r == k:
        raise KeyError(f"unmapped tensor: {k}")
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pt", required=True)
    ap.add_argument("--yaml", required=True)
    ap.add_argument("--units", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--name", default="dolphin")
    a = ap.parse_args()

    cfg = yaml.safe_load(open(a.yaml))
    enc, dec = cfg["encoder_conf"], cfg["decoder_conf"]
    if cfg.get("encoder") != "e_branchformer" or cfg.get("decoder") != "transformer":
        sys.exit(f"unsupported arch: encoder={cfg.get('encoder')} decoder={cfg.get('decoder')}")
    if enc.get("pos_enc_layer_type") != "rel_pos" or enc.get("input_layer") != "conv2d":
        sys.exit("only rel_pos (WeNet legacy) + conv2d subsampling are implemented")
    if enc.get("gate_activation", "identity") != "identity" or enc.get("use_linear_after_conv"):
        sys.exit("cgMLP gate must be identity without linear_after_conv")
    toks = load_units(a.units)
    special = cfg.get("tokenizer_conf", {}).get("special_tokens", {})
    vocab = cfg["output_dim"]
    if len(toks) != vocab:
        sys.exit(f"units.txt has {len(toks)} tokens, output_dim is {vocab}")

    sd = torch.load(a.pt, map_location="cpu", weights_only=True, mmap=True)
    if isinstance(sd, dict) and "model" in sd and isinstance(sd["model"], dict):
        sd = sd["model"]

    w = gguf.GGUFWriter(a.output, "dolphin")
    w.add_name(a.name)
    kv = {
        "dolphin.n_mels": cfg.get("input_dim", 80),
        "dolphin.d_model": enc["output_size"],
        "dolphin.enc_heads": enc["attention_heads"],
        "dolphin.enc_layers": enc["num_blocks"],
        "dolphin.enc_ffn": enc["linear_units"],
        "dolphin.cgmlp_units": enc["cgmlp_linear_units"],
        "dolphin.cgmlp_kernel": enc["cgmlp_conv_kernel"],
        "dolphin.merge_kernel": enc["merge_conv_kernel"],
        "dolphin.dec_heads": dec["attention_heads"],
        "dolphin.dec_layers": dec["num_blocks"],
        "dolphin.dec_ffn": dec["linear_units"],
        "dolphin.vocab": vocab,
        "dolphin.blank_id": cfg.get("ctc_conf", {}).get("ctc_blank_id", 0),
        "dolphin.sos_id": special.get("<sos>", vocab - 1),
        "dolphin.eos_id": special.get("<eos>", vocab - 1),
        "dolphin.asr_id": toks.index("<asr>"),
        "dolphin.notimestamp_id": toks.index("<notimestamp>"),
    }
    for k, v in kv.items():
        w.add_uint32(k, int(v))
    w.add_bool("dolphin.causal", bool(enc.get("causal", False)))
    fb = cfg["dataset_conf"]["fbank_conf"]
    w.add_uint32("dolphin.frame_length_ms", int(fb["frame_length"]))
    w.add_uint32("dolphin.frame_shift_ms", int(fb["frame_shift"]))
    w.add_array("tokenizer.ggml.tokens", toks)

    n = 0
    for k, t in sd.items():
        name = rename(k)
        if name is None:
            continue
        arr = t.detach().float().numpy()
        if name.endswith(".pe"):
            arr = arr.reshape(arr.shape[-2], arr.shape[-1])  # (5000, d)
        if name.endswith(("csgu_conv.weight", "merge_conv.weight")):
            arr = arr.reshape(arr.shape[0], arr.shape[-1])  # (C, K) depthwise
        # The two depthwise convs run as ggml_ssm_conv, which requires F32
        # kernels; they are tiny (C x 31), so keep them F32 with the 1-D tensors.
        big = arr.ndim >= 2 and not name.endswith((".pe", "pos_bias_u", "pos_bias_v", "csgu_conv.weight",
                                                   "merge_conv.weight"))
        w.add_tensor(name, arr.astype(np.float16 if big else np.float32))
        n += 1
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"wrote {a.output}: {n} tensors, vocab {vocab}, sos={kv['dolphin.sos_id']} eos={kv['dolphin.eos_id']}")


if __name__ == "__main__":
    main()
