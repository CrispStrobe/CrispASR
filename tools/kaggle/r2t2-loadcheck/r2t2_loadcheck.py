#!/usr/bin/env python3
"""CrispASR #445 — which op does this machine's torch compute differently?

The R2T2 audio tower's block 3 comes out different on Kaggle (torch 2.10 CPU)
than on the dev box (torch 2.11), in float64 NumPy, and in CrispASR — same code,
weights, and input. Recompute block 3 op by op from its (agreed) input in torch
float32 and NumPy float64 and report the first op that disagrees, plus the
torch build config.
"""
import json, os, subprocess, sys, traceback
from pathlib import Path
OUT = Path("/kaggle/working/out"); OUT.mkdir(parents=True, exist_ok=True)
res = {}
try:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "gguf", "safetensors", "qwen-asr"])
    import numpy as np, torch, math
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open
    p = hf_hub_download("netease-youdao/Confucius4-R2T2", "model.safetensors", revision="185ce639118ad1362d049ca0d8ed04b6ec5cd6c9", local_dir="/tmp/r2t2")
    x_path = hf_hub_download("cstr/crispasr-regression-fixtures", "r2t2/jfk/blk02_in_blk03.npy", repo_type="dataset", local_dir="/tmp/fx")
    f = safe_open(p, "pt")
    W = lambda n: f.get_tensor("thinker.audio_tower.layers.3." + n).float()
    x = torch.from_numpy(np.load(x_path)).float()          # block-2 output == block-3 input (N, 1024)
    res["torch"] = torch.__version__; res["config"] = torch.__config__.show()[-1500:]
    res["cpu_capability"] = torch.backends.cpu.get_cpu_capability()
    def rel(a, b):
        a = np.asarray(a, np.float64); b = np.asarray(b, np.float64)
        return float(np.abs(a - b).max() / (np.abs(b).max() + 1e-30))
    ops = {}
    with torch.no_grad():
        ln1 = torch.nn.functional.layer_norm(x, (1024,), W("self_attn_layer_norm.weight"), W("self_attn_layer_norm.bias"), 1e-5)
        xd = x.double(); m = xd.mean(-1, keepdim=True); v = ((xd - m) ** 2).mean(-1, keepdim=True)
        ln1_ref = (xd - m) / torch.sqrt(v + 1e-5) * W("self_attn_layer_norm.weight").double() + W("self_attn_layer_norm.bias").double()
        ops["layer_norm"] = rel(ln1, ln1_ref)
        for pr in ("q", "k", "v"):
            wt, bs = W(f"self_attn.{pr}_proj.weight"), W(f"self_attn.{pr}_proj.bias")
            y = torch.nn.functional.linear(ln1_ref.float(), wt, bs)
            y_ref = ln1_ref @ wt.double().T + bs.double()
            ops[f"linear_{pr}"] = rel(y, y_ref)
            ops[f"matmul_{pr}"] = rel(ln1_ref.float() @ wt.T + bs, y_ref)
        q = (ln1_ref @ W("self_attn.q_proj.weight").double().T + W("self_attn.q_proj.bias").double()).reshape(-1, 16, 64).transpose(0, 1)
        k = (ln1_ref @ W("self_attn.k_proj.weight").double().T + W("self_attn.k_proj.bias").double()).reshape(-1, 16, 64).transpose(0, 1)
        vv = (ln1_ref @ W("self_attn.v_proj.weight").double().T + W("self_attn.v_proj.bias").double()).reshape(-1, 16, 64).transpose(0, 1)
        att_ref = torch.softmax(q @ k.transpose(1, 2) / 8.0, -1) @ vv
        att = torch.nn.functional.scaled_dot_product_attention(q.float()[None], k.float()[None], vv.float()[None])[0]
        ops["sdpa"] = rel(att, att_ref)
        ops["softmax_matmul_f32"] = rel(torch.softmax(q.float() @ k.float().transpose(1, 2) / 8.0, -1) @ vv.float(), att_ref)
        fc1w, fc1b = W("fc1.weight"), W("fc1.bias")
        h = torch.randn(143, 1024, generator=torch.Generator().manual_seed(0)).double()
        ops["linear_fc1_rand"] = rel(torch.nn.functional.linear(h.float(), fc1w, fc1b), h @ fc1w.double().T + fc1b.double())
        u = h @ fc1w.double().T + fc1b.double()
        ops["gelu"] = rel(torch.nn.functional.gelu(u.float()), 0.5 * u * (1 + torch.erf(u / math.sqrt(2))))
    res["ops_rel_err"] = ops
    # The real module: layer 3 of the real audio tower on the same input, each
    # submodule hooked and compared with a float64 recompute of ITS OWN input.
    from qwen_asr.core.transformers_backend.configuration_qwen3_asr import Qwen3ASRAudioEncoderConfig
    from qwen_asr.core.transformers_backend.modeling_qwen3_asr import Qwen3ASRAudioEncoder
    cfg = json.load(open(hf_hub_download("netease-youdao/Confucius4-R2T2", "config.json", revision="185ce639118ad1362d049ca0d8ed04b6ec5cd6c9", local_dir="/tmp/r2t2")))["thinker_config"]["audio_config"]
    c = Qwen3ASRAudioEncoderConfig(**cfg); c._attn_implementation = "sdpa"
    enc = Qwen3ASRAudioEncoder(c).float().eval()
    enc.load_state_dict({k[len("thinker.audio_tower."):]: f.get_tensor(k).float() for k in f.keys() if k.startswith("thinker.audio_tower.")})
    L = enc.layers[3]
    mod = {}
    def hk(name):
        def h(m, inp, out):
            o = out[0] if isinstance(out, tuple) else out
            mod[name] = (inp[0].detach().clone() if inp else None, o.detach().clone())
        return h
    for name, sm in L.named_modules():
        if name:
            sm.register_forward_hook(hk(name))
    cu = torch.tensor([0, x.shape[0]], dtype=torch.int32)
    with torch.no_grad():
        y = L(x, cu)[0]
    np.save(OUT / "kaggle_layer3_out.npy", y.numpy())
    per = {}
    torch.set_grad_enabled(False)
    for name, (i, o) in mod.items():
        sm = dict(L.named_modules())[name]
        if isinstance(sm, torch.nn.Linear):
            ref = i.double() @ sm.weight.double().T + sm.bias.double()
        elif isinstance(sm, torch.nn.LayerNorm):
            idd = i.double(); m = idd.mean(-1, keepdim=True); v = ((idd - m) ** 2).mean(-1, keepdim=True)
            ref = (idd - m) / torch.sqrt(v + 1e-5) * sm.weight.double() + sm.bias.double()
        else:
            continue
        per[name] = rel(o, ref)
    res["module_layer3_submodule_rel_err"] = per
    # Attention core: out_proj's input vs attention recomputed from the
    # captured q/k/v outputs, and SDPA replayed on the module's exact views.
    qo, ko, vo = mod["self_attn.q_proj"][1], mod["self_attn.k_proj"][1], mod["self_attn.v_proj"][1]
    core = mod["self_attn.out_proj"][0]
    N = qo.shape[0]
    qd, kd, vd = [t.double().reshape(N, 16, 64).transpose(0, 1) for t in (qo, ko, vo)]
    att64 = (torch.softmax(qd @ kd.transpose(1, 2) / 8.0, -1) @ vd).transpose(0, 1).reshape(N, 1024)
    res["attn_core_rel_err"] = rel(core, att64)
    qv, kv_, vv_ = [t.reshape(N, 16, 64).transpose(0, 1).unsqueeze(0) for t in (qo, ko, vo)]
    res["strides"] = [list(t.stride()) for t in (qv, kv_, vv_)]
    sd_strided = torch.nn.functional.scaled_dot_product_attention(qv, kv_, vv_, scale=0.125)
    sd_contig = torch.nn.functional.scaled_dot_product_attention(qv.contiguous(), kv_.contiguous(), vv_.contiguous(), scale=0.125)
    ref64 = torch.softmax(qd @ kd.transpose(1, 2) / 8.0, -1) @ vd
    res["sdpa_strided_rel_err"] = rel(sd_strided[0], ref64)
    res["sdpa_contig_rel_err"] = rel(sd_contig[0], ref64)
    res["matmul_strided_rel_err"] = rel(torch.softmax(torch.matmul(qv, kv_.transpose(2, 3)) * 0.125, -1) @ vv_, ref64[None])
    import transformers.integrations.sdpa_attention as sa, inspect
    res["sdpa_attention_forward_src"] = inspect.getsource(sa.sdpa_attention_forward)[:3000]
    res["module_layer3_vs_local_truth_input_norm"] = float(x.norm())
except BaseException:
    res["error"] = traceback.format_exc()
(OUT / "opcheck.json").write_text(json.dumps(res, indent=1, default=str))
print(json.dumps(res, indent=1, default=str)[:6000])
