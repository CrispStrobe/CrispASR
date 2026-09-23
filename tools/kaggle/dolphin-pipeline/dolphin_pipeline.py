#!/usr/bin/env python3
"""CrispASR #436 — Dolphin-CN-Dialect (small, streaming) port pipeline, Kaggle CPU.

build (feat/436-dolphin) -> verify checkpoint sha256 -> convert F16 -> crispasr-diff
against the upstream references (cstr/crispasr-regression-fixtures dolphin/{zh,jfk})
-> quantize Q8_0 / Q4_K -> diff + transcripts. Every GGUF is uploaded to
cstr/dolphin-cn-dialect-small-streaming-GGUF as soon as it exists. Outputs are
small JSON/logs; clones are deleted.
"""
import hashlib, json, os, shutil, subprocess, sys, time, traceback
from pathlib import Path

WORK = Path("/kaggle/working"); OUT = WORK / "out"; OUT.mkdir(parents=True, exist_ok=True)
REPO = WORK / "CrispASR"; BUILD = REPO / "build"
BRANCH = os.environ.get("CRISPASR_REF", "feat/436-dolphin")
SHA = "bba8688ed33841b5f8b4578370be553f4557739c406dc7298f22985f7b061faf"
HF_REPO = "cstr/dolphin-cn-dialect-small-streaming-GGUF"
BASE = "https://huggingface.co/DataoceanAI1/dolphin-cn-dialect-small-streaming/resolve/main/"
G = Path("/tmp/gguf"); G.mkdir(exist_ok=True)
res = {"steps": {}, "errors": []}

def save(): (OUT / "pipeline.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
def run(cmd, log, timeout=None):
    t = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    (OUT / log).write_text(r.stdout[-200000:] + "\n--- stderr ---\n" + r.stderr[-100000:])
    return r.returncode, round(time.time() - t, 1), r.stdout

try:
    subprocess.check_call(["git", "clone", "--depth", "1", "--recurse-submodules", "--shallow-submodules", "-b", BRANCH,
                           "https://github.com/CrispStrobe/CrispASR.git", str(REPO)])
    res["commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(REPO), text=True).strip()
    sys.path.insert(0, str(REPO / "tools" / "kaggle"))
    import kaggle_harness as kh
    kh.resolve_hf_token(); os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "gguf", "pyyaml"])
    from huggingface_hub import HfApi, hf_hub_download
    api = HfApi()

    kh.install_build_toolchain()
    flags = kh.cache_and_link_flags()
    with kh.build_heartbeat("cmake.configure"):
        kh.sh(f"cmake -S {REPO} -B {BUILD} -G Ninja -DCMAKE_BUILD_TYPE=Release -DCRISPASR_OPUS=OFF -DCRISPASR_AMR=OFF " + " ".join(flags))
    with kh.build_heartbeat("cmake.build"):
        kh.sh(f"cmake --build {BUILD} -j$(nproc) --target crispasr crispasr-quantize crispasr-diff")
    bin_ = BUILD / "bin"; res["steps"]["build"] = "ok"; save()

    md = Path("/tmp/dolphin"); md.mkdir(exist_ok=True)
    for f in ("small.cn.streaming.pt", "train.yaml", "units.txt"):
        subprocess.check_call(["curl", "-sfL", "-o", str(md / f), BASE + f])
    h = hashlib.sha256(open(md / "small.cn.streaming.pt", "rb").read()).hexdigest()
    res["checkpoint_sha256_ok"] = h == SHA; save()
    if h != SHA: raise SystemExit("checkpoint sha256 mismatch")

    f16 = G / "dolphin-cn-dialect-small-streaming-f16.gguf"
    rc, s, _ = run([sys.executable, str(REPO / "models/convert-dolphin-to-gguf.py"), "--pt", str(md / "small.cn.streaming.pt"),
                    "--yaml", str(md / "train.yaml"), "--units", str(md / "units.txt"), "--output", str(f16),
                    "--name", "dolphin-cn-dialect-small-streaming"], "convert.log")
    res["steps"]["convert"] = {"rc": rc, "s": s}; save()
    if rc: raise SystemExit("convert failed")
    arts = {"f16": f16}

    refs = {c: hf_hub_download("cstr/crispasr-regression-fixtures", f"dolphin/{c}/ref.gguf", repo_type="dataset", local_dir="/tmp/fx") for c in ("zh", "jfk")}
    wav = {"zh": str(REPO / "samples/paraformer_zh.wav"), "jfk": str(REPO / "samples/jfk.wav")}
    res["diff"] = {}
    def diff(name, p):
        for c in ("zh", "jfk"):
            rc, s, out = run([str(bin_ / "crispasr-diff"), "dolphin", str(p), refs[c], wav[c]], f"diff-{name}-{c}.log", timeout=3600)
            res["diff"][f"{name}/{c}"] = [l for l in out.splitlines() if l.startswith(("[", "       "))] or [f"rc={rc}"]
            save()
    diff("f16", f16)   # parity first, before any quantization noise
    api.create_repo(HF_REPO, exist_ok=True)
    api.upload_file(path_or_fileobj=str(f16), path_in_repo=f16.name, repo_id=HF_REPO); res["steps"]["upload_f16"] = "ok"; save()
    for q in ("q8_0", "q4_k"):
        p = G / f"dolphin-cn-dialect-small-streaming-{q}.gguf"
        rc, s, _ = run([str(bin_ / "crispasr-quantize"), str(f16), str(p), q], f"quant-{q}.log")
        res["steps"][f"quant_{q}"] = {"rc": rc, "bytes": p.stat().st_size if p.exists() else 0}; save()
        if rc == 0:
            arts[q] = p
            api.upload_file(path_or_fileobj=str(p), path_in_repo=p.name, repo_id=HF_REPO); res["steps"][f"upload_{q}"] = "ok"
            diff(q, p)
    res["cli"] = {}
    for name, p in arts.items():
        for c, w in wav.items():
            rc, s, out = run([str(bin_ / "crispasr"), "--backend", "dolphin", "-m", str(p), "-f", w, "-t", "4", "--no-prints"], f"cli-{name}-{c}.log", timeout=3600)
            res["cli"][f"{name}/{c}"] = {"rc": rc, "s": s, "text": out.strip()[-500:]}
            save()
except BaseException:
    res["errors"].append(traceback.format_exc())
finally:
    save()
    shutil.rmtree(REPO, ignore_errors=True)
