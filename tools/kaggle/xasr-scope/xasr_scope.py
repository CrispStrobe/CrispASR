#!/usr/bin/env python3
"""CrispASR #436 (X-ASR) — scope the icefall streaming Zipformer2 before porting.

Dumps: ONNX metadata_props + op histogram + initializer shapes for each chunk
model, the pretrained.pt state-dict keys/shapes (+ any embedded args), and the
sherpa-onnx greedy transcripts of jfk + the zh sample for all four chunk sizes.
"""
import json, os, subprocess, sys, traceback, collections
from pathlib import Path
OUT = Path("/kaggle/working/out"); OUT.mkdir(parents=True, exist_ok=True)
res = {"errors": []}
def save(): (OUT / "xasr_scope.json").write_text(json.dumps(res, indent=1, ensure_ascii=False, default=str))
try:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "onnx", "onnxruntime", "sherpa-onnx", "soundfile"])
    from huggingface_hub import hf_hub_download, snapshot_download
    R = "GilgameshWind/X-ASR-zh-en"
    loc = snapshot_download(R, allow_patterns=["deployment/models/*"], local_dir="/tmp/xasr")
    import onnx
    for ch in (160, 480, 960, 1920):
        d = Path(loc) / f"deployment/models/chunk-{ch}ms-model"
        info = {}
        for part in ("encoder", "decoder", "joiner"):
            m = onnx.load(str(d / f"{part}-{ch}ms.onnx"))
            ops = collections.Counter(n.op_type for n in m.graph.node)
            info[part] = {"meta": {p.key: p.value for p in m.metadata_props}, "ops": dict(ops.most_common()),
                          "inputs": [(i.name, [x.dim_value or x.dim_param for x in i.type.tensor_type.shape.dim]) for i in m.graph.input],
                          "outputs": [(o.name, [x.dim_value or x.dim_param for x in o.type.tensor_type.shape.dim]) for o in m.graph.output]}
            if ch == 160:
                info[part]["inits"] = [(t.name, list(t.dims), t.data_type) for t in m.graph.initializer]
            del m
        res[f"chunk{ch}"] = info; save()
    for c, url in (("jfk", "https://github.com/CrispStrobe/CrispASR/raw/main/samples/jfk.wav"),
                   ("zh", "https://github.com/CrispStrobe/CrispASR/raw/main/samples/paraformer_zh.wav")):
        subprocess.check_call(["wget", "-q", "-O", f"/tmp/{c}.wav", url])
    import sherpa_onnx, soundfile as sf, numpy as np
    for ch in (160, 480, 960, 1920):
        d = Path(loc) / f"deployment/models/chunk-{ch}ms-model"
        rec = sherpa_onnx.OnlineRecognizer.from_transducer(tokens=str(d / "tokens.txt"), encoder=str(d / f"encoder-{ch}ms.onnx"),
              decoder=str(d / f"decoder-{ch}ms.onnx"), joiner=str(d / f"joiner-{ch}ms.onnx"), num_threads=4, feature_dim=80,
              decoding_method="greedy_search")
        for c in ("jfk", "zh"):
            a, sr = sf.read(f"/tmp/{c}.wav", dtype="float32")
            s = rec.create_stream(); s.accept_waveform(sr, a)
            s.accept_waveform(sr, np.zeros(int(0.66 * sr), dtype=np.float32)); s.input_finished()
            while rec.is_ready(s): rec.decode_stream(s)
            res.setdefault("text", {})[f"{ch}/{c}"] = rec.get_result(s)
        save()
    import torch
    p = hf_hub_download(R, "streaming_exp/pretrained.pt", local_dir="/tmp/xasr")
    ck = torch.load(p, map_location="cpu", weights_only=False)
    res["ckpt_top"] = list(ck.keys()) if isinstance(ck, dict) else str(type(ck))
    sd = ck.get("model", ck) if isinstance(ck, dict) else ck
    res["ckpt_params"] = sum(v.numel() for v in sd.values() if hasattr(v, "numel"))
    res["ckpt_keys"] = [(k, list(v.shape), str(v.dtype)) for k, v in sd.items() if hasattr(v, "shape")]
    for k in ck:
        if k != "model" and not hasattr(ck[k], "shape"):
            res.setdefault("ckpt_other", {})[k] = str(ck[k])[:3000]
except BaseException:
    res["errors"].append(traceback.format_exc())
finally:
    save()
