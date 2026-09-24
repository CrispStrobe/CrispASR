#!/usr/bin/env python3
"""#459 — publish the F16+F32-embedding fullstop-punc GGUF named in the model card.

convert-fullstop-punc-to-gguf.py (its default: F16 weights, F32 norms/biases,
F32 embeddings) -> fullstop-punc-f32emb.gguf. Before uploading, cross-check
against the published fullstop-punc-q8_0.gguf (known good): same metadata
keys and values, same tensor names and shapes, and every tensor's
dequantised values agree with cos >= 0.9999.
"""
import json, os, subprocess, sys, traceback
from pathlib import Path
OUT = Path("/kaggle/working/out"); OUT.mkdir(parents=True, exist_ok=True)
REPO = Path("/kaggle/working/CrispASR")
res = {"errors": []}
def save(): (OUT / "r459.json").write_text(json.dumps(res, indent=1, default=str))
try:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "gguf", "sentencepiece", "protobuf"])
    subprocess.check_call(["git", "clone", "--depth", "1", "-b", "fix/459-f32emb", "https://github.com/CrispStrobe/CrispASR.git", str(REPO)])
    sys.path.insert(0, str(REPO / "tools" / "kaggle"))
    import kaggle_harness as kh
    kh.resolve_hf_token()
    out = Path("/tmp/fullstop-punc-f32emb.gguf")
    r = subprocess.run([sys.executable, str(REPO / "models/convert-fullstop-punc-to-gguf.py"),
                        "--input", "oliverguhr/fullstop-punctuation-multilang-large", "--output", str(out)],
                       capture_output=True, text=True)
    res["convert_rc"] = r.returncode; res["convert_tail"] = (r.stdout + r.stderr)[-3000:]; save()
    assert r.returncode == 0
    res["size"] = out.stat().st_size
    import gguf, numpy as np
    from huggingface_hub import hf_hub_download, HfApi
    q8 = hf_hub_download("cstr/fullstop-punc-multilang-GGUF", "fullstop-punc-q8_0.gguf")
    A, B = gguf.GGUFReader(str(out)), gguf.GGUFReader(q8)
    def kv(rd):
        d = {}
        for k, f in rd.fields.items():
            if k.startswith("GGUF.") or k == "general.file_type" or k.startswith("general.quantization"):
                continue
            try:
                d[k] = [f.parts[i].tolist() for i in f.data][:50] if len(f.data) > 1 else f.parts[f.data[0]].tolist()
            except Exception:
                d[k] = None
        return d
    ka, kb = kv(A), kv(B)
    res["kv_only_new"] = sorted(set(ka) - set(kb)); res["kv_only_q8"] = sorted(set(kb) - set(ka))
    res["kv_diff"] = [k for k in ka if k in kb and ka[k] != kb[k]]
    ta = {t.name: t for t in A.tensors}; tb = {t.name: t for t in B.tensors}
    res["tensors_only_new"] = sorted(set(ta) - set(tb)); res["tensors_only_q8"] = sorted(set(tb) - set(ta))
    worst, types = 1.0, {}
    from gguf.quants import dequantize
    for n in ta:
        if n not in tb:
            continue
        a = dequantize(ta[n].data, ta[n].tensor_type).astype(np.float64).ravel()
        b = dequantize(tb[n].data, tb[n].tensor_type).astype(np.float64).ravel()
        assert a.shape == b.shape, n
        c = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30)) if np.any(a) or np.any(b) else 1.0
        worst = min(worst, c)
        types[ta[n].tensor_type.name] = types.get(ta[n].tensor_type.name, 0) + 1
        if "tok_emb" in n: res["tok_emb_type"] = ta[n].tensor_type.name
    res["worst_cos"] = worst; res["new_types"] = types
    ok = (not res["kv_diff"] and not res["tensors_only_new"] and not res["tensors_only_q8"]
          and worst >= 0.9999 and res.get("tok_emb_type") == "F32")
    res["gate"] = ok; save()
    if ok:
        HfApi().upload_file(path_or_fileobj=str(out), path_in_repo=out.name, repo_id="cstr/fullstop-punc-multilang-GGUF",
                            repo_type="model")
        res["uploaded"] = True
except BaseException:
    res["errors"].append(traceback.format_exc()[-3000:])
finally:
    save()
