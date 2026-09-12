#!/usr/bin/env python
"""
Reference backend for Supertonic-3 TTS — dumps per-stage intermediates from
the AUTHORITATIVE onnxruntime pipeline (the model ships ONNX-only; the
upstream MIT sample code in supertone-inc/supertonic py/helper.py is the
blueprint and this file mirrors it exactly).

Usage:
    python tools/reference_backends/supertonic_tts.py \
        --model-dir /mnt/storage/models/supertonic-3 \
        --text "The quick brown fox jumps over the lazy dog." \
        --lang en --voice M1 --steps 8 --speed 1.05 --seed 1234 \
        --output supertonic-ref.gguf

Stages dumped (all batch=1):
    text_ids        I32 [L]        indexer ids AFTER preprocessing + <lang> wrap
    dur             F32 [1]        duration seconds AFTER /speed
    te_convnext     F32 [256, L]   text encoder after ConvNeXt stack
    te_pre_spte     F32 [256, L]   after attn_encoder + global residual, *mask
    text_emb        F32 [256, L]   final text encoder output
    xt0             F32 [144, N]   the seeded gaussian noise (masked) — the C++
                                   diff run MUST inject this exact tensor
    xt_1 .. xt_K    F32 [144, N]   latent after each of K flow steps
    vf_projin_s0    F32 [512, N]   vector-field proj_in output (cond half),
                                   step 0 — first-divergence probe
    audio           F32 [T]        final wav (44.1 kHz), trimmed to sr*dur
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from unicodedata import normalize

import numpy as np

try:
    import onnxruntime as ort
    import onnx
except ImportError:
    sys.exit("pip install onnxruntime onnx")
try:
    from gguf import GGUFWriter, GGMLQuantizationType
except ImportError:
    sys.exit("pip install gguf")

AVAILABLE_LANGS = ["en", "ko", "ja", "ar", "bg", "cs", "da", "de", "el", "es",
                   "et", "fi", "fr", "hi", "hr", "hu", "id", "it", "lt", "lv",
                   "nl", "pl", "pt", "ro", "ru", "sk", "sl", "sv", "tr", "uk",
                   "vi", "na"]


def preprocess_text(text: str, lang: str) -> str:
    """Verbatim port of upstream UnicodeProcessor._preprocess_text."""
    text = normalize("NFKD", text)
    emoji_pattern = re.compile(
        "[\U0001f600-\U0001f64f\U0001f300-\U0001f5ff\U0001f680-\U0001f6ff"
        "\U0001f700-\U0001f77f\U0001f780-\U0001f7ff\U0001f800-\U0001f8ff"
        "\U0001f900-\U0001f9ff\U0001fa00-\U0001fa6f\U0001fa70-\U0001faff"
        "☀-⛿✀-➿\U0001f1e6-\U0001f1ff]+", flags=re.UNICODE)
    text = emoji_pattern.sub("", text)
    replacements = {"–": "-", "‑": "-", "—": "-", "_": " ",
                    "“": '"', "”": '"', "‘": "'", "’": "'",
                    "´": "'", "`": "'", "[": " ", "]": " ", "|": " ", "/": " ",
                    "#": " ", "→": " ", "←": " "}
    for k, v in replacements.items():
        text = text.replace(k, v)
    text = re.sub(r"[♥☆♡©\\]", "", text)
    for k, v in {"@": " at ", "e.g.,": "for example, ", "i.e.,": "that is, "}.items():
        text = text.replace(k, v)
    for pat, rep in [(r" ,", ","), (r" \.", "."), (r" !", "!"), (r" \?", "?"),
                     (r" ;", ";"), (r" :", ":"), (r" '", "'")]:
        text = re.sub(pat, rep, text)
    while '""' in text:
        text = text.replace('""', '"')
    while "''" in text:
        text = text.replace("''", "'")
    while "``" in text:
        text = text.replace("``", "`")
    text = re.sub(r"\s+", " ", text).strip()
    if not re.search(r"[.!?;:,'\"')\]}…。」』】〉》›»]$", text):
        text += "."
    assert lang in AVAILABLE_LANGS, lang
    return f"<{lang}>" + text + f"</{lang}>"


def with_extra_outputs(path: str, extra: list[str]) -> "ort.InferenceSession":
    m = onnx.load(path)
    existing = {o.name for o in m.graph.output}
    for name in extra:
        if name not in existing:
            m.graph.output.append(onnx.ValueInfoProto(name=name))
    opts = ort.SessionOptions()
    return ort.InferenceSession(m.SerializeToString(), sess_options=opts,
                                providers=["CPUExecutionProvider"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--text", default="The quick brown fox jumps over the lazy dog.")
    ap.add_argument("--lang", default="en")
    ap.add_argument("--voice", default="M1")
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--speed", type=float, default=1.05)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    onnx_dir = os.path.join(args.model_dir, "onnx")
    cfg = json.load(open(os.path.join(onnx_dir, "tts.json")))
    indexer = json.load(open(os.path.join(onnx_dir, "unicode_indexer.json")))
    sr = cfg["ae"]["sample_rate"]
    chunk = cfg["ae"]["base_chunk_size"] * cfg["ttl"]["chunk_compress_factor"]
    ldim = cfg["ttl"]["latent_dim"] * cfg["ttl"]["chunk_compress_factor"]

    vs = json.load(open(os.path.join(args.model_dir, "voice_styles", args.voice + ".json")))
    style_ttl = np.asarray(vs["style_ttl"]["data"], np.float32).reshape(1, 50, 256)
    style_dp = np.asarray(vs["style_dp"]["data"], np.float32).reshape(1, 8, 16)

    text = preprocess_text(args.text, args.lang)
    ids = np.asarray([[indexer[ord(c)] for c in text]], dtype=np.int64)
    L = ids.shape[1]
    text_mask = np.ones((1, 1, L), dtype=np.float32)

    stages: dict[str, np.ndarray] = {
        "text_ids": ids[0].astype(np.int32),
    }

    dp = ort.InferenceSession(os.path.join(onnx_dir, "duration_predictor.onnx"),
                              providers=["CPUExecutionProvider"])
    dur, = dp.run(None, {"text_ids": ids, "style_dp": style_dp, "text_mask": text_mask})
    dur = dur / args.speed
    stages["dur"] = dur.astype(np.float32)
    del dp

    te_probes = ["/text_encoder/convnext/convnext.5/Mul_3_output_0",
                 "/text_encoder/proj_out/Mul_output_0"]
    te = with_extra_outputs(os.path.join(onnx_dir, "text_encoder.onnx"), te_probes)
    te_out = te.run(None, {"text_ids": ids, "style_ttl": style_ttl,
                           "text_mask": text_mask})
    te_names = [o.name for o in te.get_outputs()]
    te_map = dict(zip(te_names, te_out))
    stages["text_emb"] = te_map["text_emb"][0].astype(np.float32)
    stages["te_convnext"] = te_map[te_probes[0]][0].astype(np.float32)
    stages["te_pre_spte"] = te_map[te_probes[1]][0].astype(np.float32)
    text_emb = te_map["text_emb"]
    del te

    # noise — verbatim sample_noisy_latent
    rng = np.random.RandomState(args.seed)
    wav_len_max = dur.max() * sr
    wav_lengths = (dur * sr).astype(np.int64)
    latent_len = int((wav_len_max + chunk - 1) // chunk)
    xt = rng.randn(1, ldim, latent_len).astype(np.float32)
    latent_lengths = (wav_lengths + chunk - 1) // chunk
    lmask = (np.arange(latent_len)[None, :] < latent_lengths[:, None])
    latent_mask = lmask.astype(np.float32).reshape(1, 1, latent_len)
    xt = xt * latent_mask
    stages["xt0"] = xt[0].copy()

    vf_probe = "/vector_estimator/vector_field/proj_in/Mul_output_0"
    ve = with_extra_outputs(os.path.join(onnx_dir, "vector_estimator.onnx"), [vf_probe])
    ve_names = [o.name for o in ve.get_outputs()]
    total = np.asarray([args.steps], dtype=np.float32)
    for step in range(args.steps):
        cur = np.asarray([step], dtype=np.float32)
        outs = ve.run(None, {"noisy_latent": xt, "text_emb": text_emb,
                             "style_ttl": style_ttl, "text_mask": text_mask,
                             "latent_mask": latent_mask, "current_step": cur,
                             "total_step": total})
        omap = dict(zip(ve_names, outs))
        if step == 0:
            # cond half only (batch was doubled inside; row 0 = cond)
            stages["vf_projin_s0"] = omap[vf_probe][0].astype(np.float32)
        xt = omap["denoised_latent"]
        stages[f"xt_{step + 1}"] = xt[0].astype(np.float32)
    del ve

    voc = ort.InferenceSession(os.path.join(onnx_dir, "vocoder.onnx"),
                               providers=["CPUExecutionProvider"])
    wav, = voc.run(None, {"latent": xt})
    n = int(sr * float(dur[0]))
    stages["audio"] = wav[0][:n].astype(np.float32)
    del voc

    w = GGUFWriter(args.output, "supertonic-tts-ref")
    w.add_string("supertonic_ref.text", args.text)
    w.add_string("supertonic_ref.lang", args.lang)
    w.add_string("supertonic_ref.voice", args.voice)
    w.add_uint32("supertonic_ref.steps", args.steps)
    w.add_float32("supertonic_ref.speed", args.speed)
    w.add_uint32("supertonic_ref.seed", args.seed)
    w.add_uint32("supertonic_ref.sample_rate", sr)
    for name, arr in stages.items():
        if arr.dtype == np.int32:
            w.add_tensor(name, np.ascontiguousarray(arr), raw_dtype=GGMLQuantizationType.I32)
        else:
            w.add_tensor(name, np.ascontiguousarray(arr.astype(np.float32)),
                         raw_dtype=GGMLQuantizationType.F32)
        print(f"  stage {name}: shape {arr.shape} |x|={np.abs(arr).mean():.6f}")
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"wrote {args.output} ({len(stages)} stages)")


if __name__ == "__main__":
    main()
