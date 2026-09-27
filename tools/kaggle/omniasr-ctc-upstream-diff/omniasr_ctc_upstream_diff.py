#!/usr/bin/env python3
"""omniASR-CTC: where does C++ diverge from the official pipeline as input grows?

Window A/B (f689d277): unsplit, 300M-v2 is garbage already on jfk 11 s and
1B-v2 degrades at 33 s, while the official pipeline accepts up to 40 s. So:
official omnilingual-asr (fairseq2, own uv venv) with forward hooks on the
encoder frontend / every encoder layer / final_proj, vs CRISPASR_OMNIASR_DUMP_DIR
(chunking off), per-frame cosine per stage, jfk (11 s) and jfk x3 (33 s).
"""
import json, os, re, subprocess, sys, traceback
from pathlib import Path
OUT = Path("/kaggle/working/out"); OUT.mkdir(parents=True, exist_ok=True)
REPO = Path("/tmp/CrispASR"); G = Path("/tmp/g"); G.mkdir(exist_ok=True)
REF = os.environ.get("CRISPASR_REF", "fix/omniasr-ctc-chunk")
res = {"errors": [], "upstream": {}, "cpp": {}, "cmp": {}}
def save(): (OUT / "result.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
CARDS = {"300m-v2": ("omniASR_CTC_300M_v2", "cstr/omniASR-CTC-300M-v2-GGUF"),
         "1b-v2": ("omniASR_CTC_1B_v2", "cstr/omniASR-CTC-1B-v2-GGUF")}
UP = r'''
import json, sys, re, numpy as np, torch
from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline
card, outdir, files = sys.argv[1], sys.argv[2], sys.argv[3:]
p = ASRInferencePipeline(model_card=card, device="cpu", dtype=torch.float32)
model = getattr(p, "model", None)
names = [n for n, _ in model.named_modules()] if model is not None else []
print("@@NAMES" + json.dumps([n for n in names if n.count(".") <= 3][:400]))
pat = re.compile(r"(encoder_frontend(\.[a-z_]+)?|encoder\.layers\.\d+|encoder\.layer_norm|final_proj)$")
caps = {}
def mk(n):
    def h(m, i, o):
        t = o[0] if isinstance(o, (tuple, list)) else o
        if hasattr(t, "seqs"): t = t.seqs
        if torch.is_tensor(t): caps[n] = t.detach().float().cpu().numpy()
    return h
if model is not None:
    for n, m in model.named_modules():
        if pat.search(n): m.register_forward_hook(mk(n))
texts = {}
for f in files:
    caps.clear()
    texts[f] = p.transcribe([f], batch_size=1)[0]
    tag = f.split("/")[-1].rsplit(".", 1)[0]
    for n, a in caps.items():
        np.save(f"{outdir}/{tag}__{n}.npy", a.squeeze())
print("@@TEXT" + json.dumps(texts))
'''
try:
    subprocess.check_call(["git", "clone", "-b", REF, "https://github.com/CrispStrobe/CrispASR.git", str(REPO)])
    subprocess.run(["git", "submodule", "update", "--init", "--recursive", "--depth", "1"], cwd=str(REPO))
    res["commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(REPO), text=True).strip()
    # upstream env FIRST, fully separate (pip-before-import; never touch this interpreter)
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "uv"])
    V = G / "venv"
    subprocess.check_call([sys.executable, "-m", "uv", "venv", "--python", "3.11", str(V)])
    py = str(V / "bin" / "python")
    # torch + torchaudio from the SAME cu128 index (fairseq2n refuses a CPU
    # torch, v2); unpinned, uv paired torch 2.8
    # with a CUDA-13 torchaudio 2.11 -> "libcudart.so.13: cannot open" (v1).
    r = subprocess.run([sys.executable, "-m", "uv", "pip", "install", "--python", py, "omnilingual-asr", "numpy",
                        "soundfile", "torch==2.8.0", "torchaudio==2.8.0",
                        "--extra-index-url", "https://download.pytorch.org/whl/cu128", "--index-strategy", "unsafe-best-match"],
                       capture_output=True, text=True, timeout=2400)
    res["uv_install"] = (r.stdout + r.stderr)[-1500:]; save()
    sys.path.insert(0, str(REPO / "tools" / "kaggle")); import kaggle_harness as kh
    tok = kh.resolve_hf_token()
    if tok: os.environ["HF_TOKEN"] = tok
    import numpy as np, soundfile as sf
    from huggingface_hub import hf_hub_download, HfApi
    a, sr = sf.read(REPO / "samples/jfk.wav", dtype="float32")
    audio = {"jfk": G / "jfk.wav", "jfk_x3": G / "jfk_x3.wav"}
    sf.write(audio["jfk"], a, sr, subtype="PCM_16"); sf.write(audio["jfk_x3"], np.concatenate([a] * 3), sr, subtype="PCM_16")
    for mk, (card, _) in CARDS.items():
        d = G / f"up_{mk}"; d.mkdir(exist_ok=True)
        (G / "up.py").write_text(UP)
        r = subprocess.run([py, str(G / "up.py"), card, str(d)] + [str(p) for p in audio.values()], capture_output=True,
                           text=True, timeout=3600)
        m = re.search(r"^@@TEXT(.*)$", r.stdout, re.M); n = re.search(r"^@@NAMES(.*)$", r.stdout, re.M)
        res["upstream"][mk] = {"texts": json.loads(m.group(1)) if m else None, "names": json.loads(n.group(1)) if n else None,
                               "npy": sorted(x.name for x in d.glob("*.npy")),
                               "err": None if m else (r.stdout + r.stderr)[-3000:]}
        save()

    kh.install_build_toolchain()
    arch = kh.detect_cuda_arch()
    kh.sh(f"cmake -S {REPO} -B {REPO}/build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCRISPASR_OPUS=OFF -DCRISPASR_AMR=OFF "
          + " ".join(kh.cuda_build_flags(arch) + kh.cache_and_link_flags()))
    with kh.build_heartbeat("cmake.build"):
        kh.sh(f"cmake --build {REPO}/build -j{kh.safe_build_jobs(gpu=True)} --target crispasr-cli")
    B = REPO / "build" / "bin" / "crispasr"
    for mk, (card, repo) in CARDS.items():
        fs = [f for f in HfApi().list_repo_files(repo) if f.endswith(".gguf") and not re.search(r"q\d|_k", f)]
        gg = hf_hub_download(repo, sorted(fs, key=len)[0], cache_dir=str(G))
        for ak, ap in audio.items():
            d = G / f"cpp_{mk}_{ak}"; d.mkdir(exist_ok=True)
            r = subprocess.run([str(B), "-m", gg, "-f", str(ap), "-np", "-ng", "--chunk-seconds", "120"], capture_output=True,
                               text=True, env=dict(os.environ, CRISPASR_OMNIASR_CTC_CHUNK_SEC="0", CRISPASR_OMNIASR_DUMP_DIR=str(d)),
                               timeout=1800)
            shapes = {m.group(1): (int(m.group(2)), int(m.group(3))) for m in re.finditer(r"DUMP (\S+) \[(\d+), (\d+)\]", r.stderr)}
            res["cpp"][f"{mk}|{ak}"] = {"gguf": Path(gg).name, "text": " ".join(r.stdout.split()), "shapes": shapes}
            # per-frame cosine vs upstream: C++ [C,T] frame-major -> (T,C)
            upd = G / f"up_{mk}"
            pairs = {"proj_out": None, "pos_conv_out": None, "logits": "final_proj"}
            for k in shapes:
                mm = re.match(r"enc_layer_(\d+)$", k)
                if mm: pairs[k] = f"encoder.layers.{mm.group(1)}"
            ups = {x.stem.split("__", 1)[1]: x for x in upd.glob(f"{ak}__*.npy")}
            cmp = {}
            for ck, uk in pairs.items():
                if ck not in shapes: continue
                cands = [u for u in ups if (uk and u.endswith(uk)) or (not uk and ck == "proj_out" and u.endswith("encoder_frontend"))]
                if not cands: continue
                U = np.load(ups[cands[0]]); C0, T0 = shapes[ck]
                Cm = np.fromfile(d / f"{ck}.bin", dtype=np.float32).reshape(T0, C0)
                if U.shape != Cm.shape:
                    cmp[ck] = {"up": cands[0], "shape_up": list(U.shape), "shape_cpp": list(Cm.shape)}; continue
                cf = (U * Cm).sum(1) / (np.linalg.norm(U, axis=1) * np.linalg.norm(Cm, axis=1) + 1e-30)
                cmp[ck] = {"up": cands[0], "cos_min": round(float(cf.min()), 5), "cos_mean": round(float(cf.mean()), 5),
                           "first_below_0.99": int(np.argmax(cf < 0.99)) if (cf < 0.99).any() else -1,
                           "n_below_0.99": int((cf < 0.99).sum()), "T": int(T0),
                           "rms_up": round(float(np.sqrt((U ** 2).mean())), 4), "rms_cpp": round(float(np.sqrt((Cm ** 2).mean())), 4)}
            res["cmp"][f"{mk}|{ak}"] = cmp
            print(mk, ak, res["cpp"][f"{mk}|{ak}"]["text"][:120], {k: v.get("cos_min") for k, v in cmp.items()}, flush=True)
            save()
        os.remove(gg)
except BaseException:
    res["errors"].append(traceback.format_exc())
finally:
    save()
