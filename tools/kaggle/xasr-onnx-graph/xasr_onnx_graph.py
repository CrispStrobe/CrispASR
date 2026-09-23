#!/usr/bin/env python3
"""CrispASR #436 (X-ASR) — the structure an ONNX-sourced converter must follow.

For the 480 ms encoder: ordered list of MatMul/Conv nodes that take an
initializer (name, shape), and of Mul/Add nodes that take a constant
(initializer or Constant node) with >= 2 elements — these are the
constant-folded chunk scales, downsample weights and (maybe) pos projections.
Also: whether the four exports share their named weights bit-for-bit.
"""
import collections, json, sys, subprocess, traceback
from pathlib import Path
OUT = Path("/kaggle/working/out"); OUT.mkdir(parents=True, exist_ok=True)
res = {"errors": []}
def save(): (OUT / "onnx_graph.json").write_text(json.dumps(res, indent=1, default=str))
try:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "onnx"])
    import numpy as np, onnx
    from onnx import numpy_helper
    from huggingface_hub import hf_hub_download
    R = "GilgameshWind/X-ASR-zh-en"
    ms = {}
    for ch in (160, 480, 960, 1920):
        ms[ch] = onnx.load(hf_hub_download(R, f"deployment/models/chunk-{ch}ms-model/encoder-{ch}ms.onnx", local_dir="/tmp/x"))
    m = ms[480]
    inits = {t.name: t for t in m.graph.initializer}
    consts = {}
    for nd in m.graph.node:
        if nd.op_type == "Constant":
            for a in nd.attribute:
                if a.name == "value":
                    consts[nd.output[0]] = numpy_helper.to_array(a.t)
    def cshape(n):
        if n in inits: return list(inits[n].dims), "init", n
        if n in consts: return list(consts[n].shape), "const", n
        return None
    mm, muls = [], []
    for i, nd in enumerate(m.graph.node):
        if nd.op_type in ("MatMul", "Conv", "Gemm"):
            for j, x in enumerate(nd.input):
                c = cshape(x)
                if c and c[1] == "init":
                    mm.append([i, nd.op_type, nd.name, j, c[2], c[0]])
    res["n_nodes"] = len(m.graph.node)
    res["n_weight_matmul_anon"] = sum(1 for x in mm if x[1] == "MatMul" and x[4].startswith("onnx::"))
    # every anonymous NON-MatMul initializer / constant with >= 16 elements: shape + the op chain it feeds
    for ch in (160, 1920):
        mc = ms[ch]
        ini = {t.name: t for t in mc.graph.initializer}
        cst = {}
        for nd in mc.graph.node:
            if nd.op_type == "Constant":
                for a in nd.attribute:
                    if a.name == "value":
                        cst[nd.output[0]] = numpy_helper.to_array(a.t)
        cons = collections.defaultdict(list)
        for i, nd in enumerate(mc.graph.node):
            for x in nd.input:
                cons[x].append(i)
        nodes = mc.graph.node
        def chain(name, depth=4):
            out, cur = [], name
            for _ in range(depth):
                if not cons[cur]:
                    break
                nd = nodes[cons[cur][0]]
                out.append(f"{nd.op_type}({nd.name})")
                cur = nd.output[0]
            return " > ".join(out)
        rows = []
        for n, t in ini.items():
            if not n.startswith("onnx::"):
                continue
            if any(nodes[i].op_type == "MatMul" for i in cons[n]):
                continue
            size = int(np.prod(t.dims)) if len(t.dims) else 1
            if size >= 16:
                rows.append(["init", n, list(t.dims), cons[n][0] if cons[n] else -1, chain(n)])
        for n, v in cst.items():
            if v.size >= 16:
                rows.append(["const", n, list(v.shape), cons[n][0] if cons[n] else -1, chain(n)])
        rows.sort(key=lambda r: r[3])
        res[f"anon_nonmatmul_{ch}"] = rows
        res[f"anon_nonmatmul_shapes_{ch}"] = dict(collections.Counter(str(r[2]) for r in rows))
        save()
    # do exports share weights?
    named = {t.name: numpy_helper.to_array(t) for t in m.graph.initializer if not t.name.startswith("onnx::")}
    for ch in (160, 960, 1920):
        other = {t.name: numpy_helper.to_array(t) for t in ms[ch].graph.initializer if not t.name.startswith("onnx::")}
        res[f"same_named_{ch}"] = all(k in other and np.array_equal(named[k], other[k]) for k in named)
        a = [numpy_helper.to_array(inits[x[4]]) for x in mm if x[1] == "MatMul" and x[4].startswith("onnx::")]
        oi = {t.name: t for t in ms[ch].graph.initializer}
        b = []
        for nd in ms[ch].graph.node:
            if nd.op_type == "MatMul":
                for x in nd.input:
                    if x in oi and x.startswith("onnx::"):
                        b.append(numpy_helper.to_array(oi[x]))
        res[f"anon_matmul_equal_{ch}"] = len(a) == len(b) and all(np.array_equal(p, q) for p, q in zip(a, b))
        res[f"anon_matmul_count_{ch}"] = len(b)
    save()
except BaseException:
    res["errors"].append(traceback.format_exc())
finally:
    save()
