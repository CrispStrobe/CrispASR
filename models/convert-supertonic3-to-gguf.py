#!/usr/bin/env python3
"""
Convert Supertone Supertonic-3 (ONNX-only distribution) → one GGUF
(arch "supertonic-tts").

Source: https://huggingface.co/Supertone/supertonic-3 (OpenRAIL-M)
    onnx/{duration_predictor,text_encoder,vector_estimator,vocoder}.onnx
    onnx/{tts.json,unicode_indexer.json}
    voice_styles/{F1..F5,M1..M5}.json

Everything the C++ runtime needs lands in ONE file:
  * the four networks' weights (renamed to a stable dotted scheme),
  * the unicode→id indexer (i32[65536]),
  * NFKD decomposition + canonical-combining-class tables generated from
    Python's unicodedata at convert time (so the C++ text preprocessor
    needs no Unicode library),
  * all ten preset voices (tiny: style_ttl [50,256] + style_dp [8,16]).

Tensor naming (numpy shapes; GGUFWriter writes ne[] reversed):
  dp.*    duration predictor        (from tts.dp.*)
  te.*    text encoder              (from tts.ttl.text_encoder.*)
  te.spte.*  speech-prompted text encoder (tts.ttl.speech_prompted_text_encoder.*)
  te.style_key  [50,256]            (tts.ttl.style_encoder.style_token_layer.style_key)
  vf.*    vector field              (vector_estimator.tts.ttl.vector_field.*)
  vf.uncond.{text,style_key,style_value}  CFG uncond special tokens
  vf.time_freqs [32], vf.rotary_theta [32]
  voc.*   vocoder / AE decoder      (tts.ae.decoder.*), voc.latent_{mean,std} [24]
  text.unicode_indexer i32[65536]
  text.nfkd_{keys,offsets,codepoints} i32, text.ccc_{keys,vals} i32
  voice.<NAME>.ttl [50,256] f32, voice.<NAME>.dp [8,16] f32

Layout conventions:
  * Linear (MatMul B-matrix [in,out]) → transposed to (out,in).
  * Gemm transB=1 weights already (out,in) → unchanged.
  * Conv k=1 → squeezed to (out,in) (used as matmul).
  * Conv k>1 → ONNX (out, in/groups, K) kept; gguf ne = (K, in/g, out)
    which is exactly what ggml_conv_1d / ggml_conv_1d_dw expect.
  * Embedding (vocab, dim) kept (ne = (dim, vocab), get_rows-ready).

Anonymous initializers (onnx::MatMul_*, onnx::Conv_*, onnx::PRelu_*) are
resolved by walking the graph to the consuming node and deriving the
canonical module path from the node name — never by initializer name.

The converter VERIFIES (hard asserts) every constant the C++ hardcodes:
CFG 4/3, rotary theta, time freqs, attention score scales, dwconv pads,
softmax mask values, LayerNorm epsilons.

Usage:
    python models/convert-supertonic3-to-gguf.py \
        --model /mnt/storage/models/supertonic-3 \
        --output supertonic3-f16.gguf [--dtype f16|f32]
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import unicodedata

import numpy as np

try:
    import onnx
    from onnx import numpy_helper
except ImportError:
    sys.exit("pip install onnx")

try:
    from gguf import GGUFWriter, GGMLQuantizationType
except ImportError:
    sys.exit("pip install gguf")

LANGS = ["en", "ko", "ja", "ar", "bg", "cs", "da", "de", "el", "es", "et",
         "fi", "fr", "hi", "hr", "hu", "id", "it", "lt", "lv", "nl", "pl",
         "pt", "ro", "ru", "sk", "sl", "sv", "tr", "uk", "vi", "na"]
VOICES = ["F1", "F2", "F3", "F4", "F5", "M1", "M2", "M3", "M4", "M5"]


def load_graph(path):
    m = onnx.load(path)
    g = m.graph
    inits = {i.name: i for i in g.initializer}
    consts = {}
    for n in g.node:
        if n.op_type == "Constant":
            for a in n.attribute:
                if a.name == "value":
                    consts[n.output[0]] = numpy_helper.to_array(a.t)
    return g, inits, consts


def node_canonical(node_name: str) -> str:
    """'/a/b/W_query/linear/MatMul' -> 'a.b.W_query.linear'"""
    parts = [p for p in node_name.strip("/").split("/") if p]
    return ".".join(parts[:-1])


def resolve_weights(g, inits, prefix_map, out: dict, fname: str):
    """Collect all float initializers under canonical names.

    prefix_map: list of (old_prefix, new_prefix) applied to NAMED initializers.
    Anonymous ones resolve through their consuming node.
    """
    consumed_anon = {}
    for n in g.node:
        for idx, inp in enumerate(n.input):
            if inp in inits and (inp.startswith("onnx::") or inp.startswith("/")):
                base = node_canonical(n.name)
                if n.op_type in ("MatMul", "Conv", "Gemm"):
                    suffix = ".weight" if idx == 1 else ".bias"
                elif n.op_type == "PRelu":
                    suffix = ".weight"
                else:
                    continue  # shape/pad constants etc.
                consumed_anon.setdefault(inp, (base + suffix, n.op_type))

    for name, init in inits.items():
        if init.data_type != onnx.TensorProto.FLOAT:
            continue
        arr = numpy_helper.to_array(init)
        if name in consumed_anon:
            canon, op = consumed_anon[name]
        else:
            canon, op = name, None
            for old, new in prefix_map:
                if canon.startswith(old):
                    canon = new + canon[len(old):]
                    break
            else:
                if name.startswith("/") or name.startswith("onnx::"):
                    continue  # unconsumed graph constant
        # apply prefix map to node-derived names too
        for old, new in prefix_map:
            if canon.startswith(old):
                canon = new + canon[len(old):]
                break
        out[canon] = (arr, op)
    return out


def find_scalar_div_after_matmul(g, consts, inits, module_prefix):
    """Return the scalar divisor of the Div fed by a MatMul inside module."""
    outs = {}
    for n in g.node:
        for o in n.output:
            outs[o] = n
    for n in g.node:
        if n.op_type == "Div" and n.name.startswith(module_prefix):
            src = outs.get(n.input[0])
            if src is not None and src.op_type == "MatMul":
                c = consts.get(n.input[1])
                if c is None and n.input[1] in inits:
                    c = numpy_helper.to_array(inits[n.input[1]])
                if c is not None and np.asarray(c).size == 1:
                    return float(np.asarray(c).flatten()[0])
    return None


def collect_ln_eps(g, module_filter):
    eps = set()
    for n in g.node:
        if n.op_type == "LayerNormalization" and module_filter(n.name):
            e = 1e-5
            for a in n.attribute:
                if a.name == "epsilon":
                    e = a.f
            eps.add(round(e, 12))
    return eps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="supertonic-3 snapshot dir")
    ap.add_argument("--output", required=True)
    ap.add_argument("--dtype", default="f16", choices=["f16", "f32"])
    args = ap.parse_args()

    onnx_dir = os.path.join(args.model, "onnx")
    cfg = json.load(open(os.path.join(onnx_dir, "tts.json")))
    indexer = json.load(open(os.path.join(onnx_dir, "unicode_indexer.json")))
    assert len(indexer) == 65536

    tensors: dict[str, tuple[np.ndarray, str | None]] = {}
    kv_f: dict[str, float] = {}

    # ---------------- duration_predictor ----------------
    g, inits, consts = load_graph(os.path.join(onnx_dir, "duration_predictor.onnx"))
    resolve_weights(g, inits, [("tts.dp.", "dp."),
                               ("sentence_encoder.", "dp.sentence_encoder."),
                               ("predictor.", "dp.predictor.")], tensors, "dp")
    s = find_scalar_div_after_matmul(g, consts, inits,
                                     "/sentence_encoder/attn_encoder/attn_layers.0")
    # VITS divides K (not the QK product) — fall back to Div after conv_k
    kv_f["dp.attn_scale"] = s if s is not None else -1.0
    eps_cn = collect_ln_eps(g, lambda n: "convnext" in n)
    eps_at = collect_ln_eps(g, lambda n: "attn_encoder" in n)
    assert len(eps_cn) == 1 and len(eps_at) == 1, (eps_cn, eps_at)
    kv_f["dp.convnext_ln_eps"] = eps_cn.pop()
    kv_f["dp.attn_ln_eps"] = eps_at.pop()
    del g, inits, consts
    gc.collect()

    # ---------------- text_encoder ----------------
    g, inits, consts = load_graph(os.path.join(onnx_dir, "text_encoder.onnx"))
    resolve_weights(
        g, inits,
        [("tts.ttl.text_encoder.", "te."),
         ("tts.ttl.speech_prompted_text_encoder.", "te.spte."),
         ("tts.ttl.style_encoder.style_token_layer.style_key", "te.style_key"),
         ("text_encoder.", "te."),
         ("speech_prompted_text_encoder.", "te.spte.")], tensors, "te")
    s1 = find_scalar_div_after_matmul(g, consts, inits,
                                      "/speech_prompted_text_encoder/attention1")
    s2 = find_scalar_div_after_matmul(g, consts, inits,
                                      "/speech_prompted_text_encoder/attention2")
    assert s1 is not None and s1 == s2, (s1, s2)
    kv_f["te.spte_attn_scale"] = s1
    eps_cn = collect_ln_eps(g, lambda n: "convnext" in n)
    eps_at = collect_ln_eps(g, lambda n: "attn_encoder" in n)
    eps_sp = collect_ln_eps(g, lambda n: "speech_prompted" in n)
    assert len(eps_cn) == 1 and len(eps_at) == 1 and len(eps_sp) == 1
    kv_f["te.convnext_ln_eps"] = eps_cn.pop()
    kv_f["te.attn_ln_eps"] = eps_at.pop()
    kv_f["te.spte_ln_eps"] = eps_sp.pop()
    del g, inits, consts
    gc.collect()

    # ---------------- vector_estimator ----------------
    g, inits, consts = load_graph(os.path.join(onnx_dir, "vector_estimator.onnx"))

    def cval(name):
        if name in consts:
            return numpy_helper.to_array(consts[name]) if hasattr(consts[name], "dims") else consts[name]
        if name in inits:
            return numpy_helper.to_array(inits[name])
        return None

    cfg_c = np.asarray(cval("/Constant_3_output_0")).item()
    cfg_u = np.asarray(cval("/Constant_4_output_0")).item()
    assert cfg_c == 4.0 and cfg_u == 3.0, (cfg_c, cfg_u)
    kv_f["vf.cfg_cond"] = cfg_c
    kv_f["vf.cfg_uncond"] = cfg_u

    theta = np.asarray(cval("vector_estimator.tts.ttl.vector_field.main_blocks.3.attn.theta")).reshape(-1)
    assert theta.shape == (32,)
    ref_theta = 10.0 * (10000.0 ** (-np.arange(32) / 32.0))
    assert np.allclose(theta, ref_theta, rtol=1e-4), "rotary theta formula changed"

    tfreqs = np.asarray(cval("/vector_estimator/vector_field/time_encoder/sinusoidal/Constant_3_output_0")).reshape(-1)
    assert tfreqs.shape == (32,)
    ref_freqs = 10000.0 ** (-np.arange(32) / 31.0)
    assert np.allclose(tfreqs, ref_freqs, rtol=1e-4), "time freqs formula changed"
    tscale = np.asarray(cval("/vector_estimator/vector_field/time_encoder/sinusoidal/Constant_2_output_0")).item()
    assert tscale == 1000.0

    attn_scale = np.asarray(cval("/vector_estimator/vector_field/main_blocks.3/attn/Constant_51_output_0")).item()
    assert attn_scale == 16.0
    kv_f["vf.text_attn_scale"] = attn_scale
    s = find_scalar_div_after_matmul(g, consts, inits,
                                     "/vector_estimator/vector_field/main_blocks.5/attention")
    assert s is not None
    kv_f["vf.style_attn_scale"] = s

    # dwconv pads: symmetric 2*dil for k5
    for i, dil in enumerate([1, 2, 4, 8]):
        p = np.asarray(cval(f"/vector_estimator/vector_field/main_blocks.0/convnext.{i}/dwconv/Cast_output_0"))
        assert list(p) == [0, 0, 2 * dil, 0, 0, 2 * dil], (i, list(p))

    eps_cn = collect_ln_eps(g, lambda n: "convnext" in n)
    eps_bl = collect_ln_eps(g, lambda n: "/norm/" in n and "convnext" not in n)
    assert len(eps_cn) == 1 and len(eps_bl) == 1
    kv_f["vf.convnext_ln_eps"] = eps_cn.pop()
    kv_f["vf.block_ln_eps"] = eps_bl.pop()

    resolve_weights(
        g, inits,
        [("vector_estimator.tts.ttl.vector_field.", "vf."),
         ("vector_estimator.tts.ttl.uncond_masker.text_special_token", "vf.uncond.text"),
         ("vector_estimator.tts.ttl.uncond_masker.style_key_special_token", "vf.uncond.style_key"),
         ("vector_estimator.tts.ttl.uncond_masker.style_value_special_token", "vf.uncond.style_value"),
         ("vector_estimator.vector_field.", "vf.")], tensors, "vf")

    # drop the per-block theta duplicate; store canonical copies
    for k in [k for k in tensors if k.endswith(".attn.theta")]:
        del tensors[k]
    tensors["vf.rotary_theta"] = (theta.astype(np.float32), None)
    tensors["vf.time_freqs"] = (tfreqs.astype(np.float32), None)

    style_key_vf = np.asarray(cval("/vector_estimator/Expand_output_0"))
    del g, inits, consts
    gc.collect()

    # style_key must match the text encoder's prototype
    te_sk = tensors["te.style_key"][0]
    assert np.allclose(style_key_vf, te_sk), "vf style_key != te style_key"

    # ---------------- vocoder ----------------
    g, inits, consts = load_graph(os.path.join(onnx_dir, "vocoder.onnx"))
    resolve_weights(g, inits,
                    [("tts.ae.decoder.", "voc."),
                     ("tts.ae.latent_mean", "voc.latent_mean"),
                     ("tts.ae.latent_std", "voc.latent_std"),
                     ("tts.ttl.normalizer.scale", "voc.normalizer_scale"),
                     ("decoder.", "voc.")], tensors, "voc")
    eps_cn = collect_ln_eps(g, lambda n: "convnext" in n)
    assert len(eps_cn) == 1
    kv_f["voc.convnext_ln_eps"] = eps_cn.pop()
    bn_eps = None
    for n in g.node:
        if n.op_type == "BatchNormalization":
            for a in n.attribute:
                if a.name == "epsilon":
                    bn_eps = a.f
    assert bn_eps is not None
    kv_f["voc.final_bn_eps"] = bn_eps
    del g, inits, consts
    gc.collect()

    norm_scale = tensors.pop("voc.normalizer_scale")[0]
    assert float(np.asarray(norm_scale).flatten()[0]) == cfg["ttl"]["normalizer"]["scale"]
    kv_f["vf.normalizer_scale"] = float(np.asarray(norm_scale).flatten()[0])

    # ---------------- text tables ----------------
    idx_arr = np.asarray(indexer, dtype=np.int32)
    nfkd_keys, nfkd_offs, nfkd_cps = [], [0], []
    ccc_keys, ccc_vals = [], []
    for cp in range(0x10000):
        if 0xD800 <= cp <= 0xDFFF:
            continue
        ch = chr(cp)
        dec = unicodedata.normalize("NFKD", ch)
        if dec != ch:
            nfkd_keys.append(cp)
            nfkd_cps.extend(ord(c) for c in dec)
            nfkd_offs.append(len(nfkd_cps))
        cc = unicodedata.combining(ch)
        if cc:
            ccc_keys.append(cp)
            ccc_vals.append(cc)

    # ---------------- voices ----------------
    voice_tensors = {}
    for v in VOICES:
        d = json.load(open(os.path.join(args.model, "voice_styles", v + ".json")))
        ttl = np.asarray(d["style_ttl"]["data"], dtype=np.float32).reshape(d["style_ttl"]["dims"][1:])
        dp = np.asarray(d["style_dp"]["data"], dtype=np.float32).reshape(d["style_dp"]["dims"][1:])
        assert ttl.shape == (50, 256) and dp.shape == (8, 16)
        voice_tensors[f"voice.{v}.ttl"] = ttl
        voice_tensors[f"voice.{v}.dp"] = dp

    # ---------------- layout transforms ----------------
    out_tensors = {}
    for name, (arr, op) in tensors.items():
        a = np.asarray(arr)
        if op == "MatMul":
            assert a.ndim == 2, name
            a = a.T.copy()  # (in,out) -> (out,in)
        elif op == "Conv" and name.endswith(".weight") and a.ndim == 3 and a.shape[2] == 1:
            a = a[:, :, 0].copy()  # k=1 conv -> (out,in) matmul
        elif name.endswith(".weight") and op is None and a.ndim == 3 and a.shape[2] == 1 \
                and ("pwconv" in name or "proj_out" in name or "proj_in" in name
                     or "conv_q" in name or "conv_k" in name or "conv_v" in name
                     or "conv_o" in name or "conv_1" in name or "conv_2" in name
                     or name.endswith("head.layer2.weight")):
            a = a[:, :, 0].copy()
        out_tensors[name] = a
    out_tensors.update(voice_tensors)

    # ---------------- write ----------------
    w = GGUFWriter(args.output, "supertonic-tts")
    w.add_uint32("supertonic.sample_rate", cfg["ae"]["sample_rate"])
    w.add_uint32("supertonic.base_chunk_size", cfg["ae"]["base_chunk_size"])
    w.add_uint32("supertonic.chunk_compress_factor", cfg["ttl"]["chunk_compress_factor"])
    w.add_uint32("supertonic.latent_dim", cfg["ttl"]["latent_dim"])
    w.add_uint32("supertonic.default_steps", 8)
    w.add_float32("supertonic.default_speed", 1.05)
    w.add_string("supertonic.languages", ",".join(LANGS))
    w.add_string("supertonic.voices", ",".join(VOICES))
    w.add_string("supertonic.tts_version", cfg.get("tts_version", ""))
    w.add_string("general.license", "openrail")
    w.add_string("general.source_hf_repo", "Supertone/supertonic-3")
    for k, v in kv_f.items():
        w.add_float32("supertonic." + k, float(v))

    f16 = args.dtype == "f16"
    n_f16 = n_f32 = 0
    for name, a in sorted(out_tensors.items()):
        keep32 = (a.ndim <= 1 or a.size < 4096 or "norm" in name
                  or name.endswith(".bias") or "gamma" in name
                  or name.startswith("voice.") or "style_key" in name
                  or "uncond" in name or "latent_mean" in name
                  or "latent_std" in name or "emb_rel" in name
                  or "sentence_token" in name)
        if f16 and not keep32:
            w.add_tensor(name, a.astype(np.float16), raw_dtype=GGMLQuantizationType.F16)
            n_f16 += 1
        else:
            w.add_tensor(name, a.astype(np.float32), raw_dtype=GGMLQuantizationType.F32)
            n_f32 += 1

    w.add_tensor("text.unicode_indexer", idx_arr, raw_dtype=GGMLQuantizationType.I32)
    w.add_tensor("text.nfkd_keys", np.asarray(nfkd_keys, dtype=np.int32), raw_dtype=GGMLQuantizationType.I32)
    w.add_tensor("text.nfkd_offsets", np.asarray(nfkd_offs, dtype=np.int32), raw_dtype=GGMLQuantizationType.I32)
    w.add_tensor("text.nfkd_codepoints", np.asarray(nfkd_cps, dtype=np.int32), raw_dtype=GGMLQuantizationType.I32)
    w.add_tensor("text.ccc_keys", np.asarray(ccc_keys, dtype=np.int32), raw_dtype=GGMLQuantizationType.I32)
    w.add_tensor("text.ccc_vals", np.asarray(ccc_vals, dtype=np.int32), raw_dtype=GGMLQuantizationType.I32)

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"wrote {args.output}: {len(out_tensors) + 6} tensors "
          f"({n_f16} f16, {n_f32 + 6} f32), nfkd {len(nfkd_keys)} keys / "
          f"{len(nfkd_cps)} cps, ccc {len(ccc_keys)}")


if __name__ == "__main__":
    main()
