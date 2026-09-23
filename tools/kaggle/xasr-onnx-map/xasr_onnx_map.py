#!/usr/bin/env python3
"""CrispASR #436 (X-ASR) — can the weights come from the ONNX exports?

pretrained.pt does not carry the exported weights (named ONNX initializers
differ from both `model` and `model_avg` by up to 24..34). Map every checkpoint
key to an ONNX source — named initializer, or an anonymous MatMul/Gemm/Conv
initializer named by its node path — for each chunk export, and report what is
missing (likely constant-folded) plus sample node names around it.
"""
import collections, json, re, sys, subprocess, traceback
from pathlib import Path
OUT = Path("/kaggle/working/out"); OUT.mkdir(parents=True, exist_ok=True)
res = {"errors": []}
def save(): (OUT / "onnx_map.json").write_text(json.dumps(res, indent=1, default=str))
try:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "onnx"])
    import numpy as np, onnx, torch
    from onnx import numpy_helper
    from huggingface_hub import hf_hub_download
    R = "GilgameshWind/X-ASR-zh-en"
    pt = hf_hub_download(R, "streaming_exp/pretrained.pt", local_dir="/tmp/x")
    ck = torch.load(pt, map_location="cpu", weights_only=False)
    ref = {k: v.float().numpy() for k, v in ck["model_avg"].items()}
    del ck
    for ch in (480, 1920, 160):
        sd, src = {}, {}
        for part in ("encoder", "decoder", "joiner"):
            p = hf_hub_download(R, f"deployment/models/chunk-{ch}ms-model/{part}-{ch}ms.onnx", local_dir="/tmp/x")
            m = onnx.load(p)
            inits = {t.name: t for t in m.graph.initializer}
            for n, t in inits.items():
                if not n.startswith("onnx::"):
                    sd[n] = numpy_helper.to_array(t); src[n] = "named"
            for nd in m.graph.node:
                if nd.op_type in ("MatMul", "Gemm", "Conv"):
                    for i in nd.input:
                        if i in inits and i.startswith("onnx::"):
                            mod = ".".join(nd.name.strip("/").split("/")[:-1])
                            a = numpy_helper.to_array(inits[i])
                            key = mod + ".weight"
                            sd[key] = a.T if nd.op_type == "MatMul" else a; src[key] = nd.op_type + ":" + nd.name
            if ch == 480:
                res.setdefault("sample_nodes", {})[part] = [nd.name for nd in m.graph.node if nd.op_type in ("MatMul", "Gemm", "Conv", "Softmax")][:40]
            del m
        # align prefixes to checkpoint names
        fix = {}
        for k, v in sd.items():
            k2 = re.sub(r"^encoder_proj\.", "joiner.encoder_proj.", k)
            k2 = re.sub(r"^decoder_proj\.", "joiner.decoder_proj.", k2)
            k2 = re.sub(r"^output_linear\.", "joiner.output_linear.", k2)
            fix[k2] = v
        want = [k for k in ref if not k.startswith(("simple_am_proj", "simple_lm_proj"))]
        missing = [k for k in want if k not in fix]
        shape_bad = [(k, list(fix[k].shape), list(ref[k].shape)) for k in want if k in fix and fix[k].shape != ref[k].shape]
        extra = [k for k in fix if k not in ref][:30]
        diffs = sorted(((float(np.abs(fix[k] - ref[k]).max()), k) for k in want if k in fix and fix[k].shape == ref[k].shape), reverse=True)
        res[f"chunk{ch}"] = {"n_have": len(fix), "n_want": len(want), "missing": missing, "missing_patterns": dict(collections.Counter(re.sub(r"\d+", "N", k) for k in missing)),
                             "shape_bad": shape_bad[:20], "extra": extra, "diff_top": diffs[:15],
                             "n_identical": sum(1 for d, _ in diffs if d == 0.0), "n_compared": len(diffs)}
        if ch == 480:
            np.savez_compressed(OUT / "onnx480_sd_sample.npz", **{k: fix[k] for k in list(fix)[:5]})
        save()
except BaseException:
    res["errors"].append(traceback.format_exc())
finally:
    save()
