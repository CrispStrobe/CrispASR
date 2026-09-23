#!/usr/bin/env python3
"""CrispASR #445 — Confucius4-R2T2 port pipeline on Kaggle CPU.

The dev VPS cannot hold this (4.7 GB F16, 3.6 GB converter peak), so the whole
convert -> quantize -> upload -> validate sequence runs here:

  1. build crispasr / crispasr-quantize / crispasr-diff from feat/445-r2t2
  2. download the pinned checkpoint and VERIFY its sha256 against the hub
     (a CIFS download on the dev box came back silently corrupted)
  3. convert (F16, --streaming-recipe r2t2), quantize Q8_0 / Q4_K; upload each
     artifact to cstr/confucius4-r2t2-GGUF the moment it exists
  4. crispasr-diff every artifact against the f32 qwen_asr references
     (cstr/crispasr-regression-fixtures r2t2/{jfk,zh}/ref.gguf)
  5. offline transcripts, and the realtime streaming session (the example.py
     schedule) traced call by call, for comparison with r2t2/stream_ref.json

Outputs: small JSON + logs in /kaggle/working/out. Clones are deleted.
"""
import hashlib, json, os, shutil, subprocess, sys, time, traceback
from pathlib import Path

WORK = Path("/kaggle/working"); OUT = WORK / "out"; OUT.mkdir(parents=True, exist_ok=True)
REPO = WORK / "CrispASR"; BUILD = REPO / "build"
BRANCH = os.environ.get("CRISPASR_REF", "feat/445-r2t2")
MODEL, REV = "netease-youdao/Confucius4-R2T2", "185ce639118ad1362d049ca0d8ed04b6ec5cd6c9"
SHA = "cc4d5324d386c80f98a8a7b09fbcdcc813ad08a6503fe3a586ebb144ec4610dc"
HF_REPO = "cstr/confucius4-r2t2-GGUF"
G = Path("/tmp/gguf"); G.mkdir(exist_ok=True)
res = {"steps": {}, "errors": []}

def save(): (OUT / "pipeline.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
def run(cmd, log, env=None, timeout=None):
    t = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=timeout)
    (OUT / log).write_text(r.stdout[-200000:] + "\n--- stderr ---\n" + r.stderr[-200000:])
    return r.returncode, round(time.time() - t, 1), r.stdout

try:
    subprocess.check_call(["git", "clone", "--depth", "1", "--recurse-submodules", "--shallow-submodules", "-b", BRANCH, "https://github.com/CrispStrobe/CrispASR.git", str(REPO)])
    res["commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(REPO), text=True).strip()
    sys.path.insert(0, str(REPO / "tools" / "kaggle"))
    import kaggle_harness as kh
    kh.resolve_hf_token(); os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "gguf", "safetensors", "sentencepiece"])
    from huggingface_hub import HfApi, hf_hub_download, snapshot_download
    api = HfApi()

    # 1. build
    kh.install_build_toolchain()
    flags = kh.cache_and_link_flags()
    with kh.build_heartbeat("cmake.configure"):
        kh.sh(f"cmake -S {REPO} -B {BUILD} -G Ninja -DCMAKE_BUILD_TYPE=Release -DCRISPASR_OPUS=OFF -DCRISPASR_AMR=OFF " + " ".join(flags))
    with kh.build_heartbeat("cmake.build"):
        kh.sh(f"cmake --build {BUILD} -j$(nproc) --target crispasr crispasr-quantize crispasr-diff")
    bin_ = BUILD / "bin"; res["steps"]["build"] = "ok"; save()

    # 2. checkpoint, verified
    d = snapshot_download(MODEL, revision=REV, local_dir="/tmp/r2t2")
    h = hashlib.sha256()
    with open(f"{d}/model.safetensors", "rb") as f:
        for c in iter(lambda: f.read(1 << 24), b""): h.update(c)
    res["checkpoint_sha256_ok"] = h.hexdigest() == SHA; save()
    if not res["checkpoint_sha256_ok"]: raise SystemExit("checkpoint sha256 mismatch")

    # 3. convert / quantize / upload
    api.create_repo(HF_REPO, exist_ok=True)
    lic = subprocess.run(["curl", "-sfL", "https://raw.githubusercontent.com/netease-youdao/Confucius4-R2T2/refs/heads/master/MODEL_LICENSE"], capture_output=True).stdout
    (G / "MODEL_LICENSE").write_bytes(lic)
    api.upload_file(path_or_fileobj=str(G / "MODEL_LICENSE"), path_in_repo="MODEL_LICENSE", repo_id=HF_REPO)
    f16 = G / "confucius4-r2t2-f16.gguf"
    rc, s, _ = run([sys.executable, str(REPO / "models/convert-qwen3-asr-to-gguf.py"), "--input", d, "--output", str(f16), "--streaming-recipe", "r2t2"], "convert.log")
    res["steps"]["convert"] = {"rc": rc, "s": s}; save()
    if rc: raise SystemExit("convert failed")
    arts = {"f16": f16}
    for q in ("q8_0", "q4_k"):
        p = G / f"confucius4-r2t2-{q}.gguf"
        rc, s, _ = run([str(bin_ / "crispasr-quantize"), str(f16), str(p), q], f"quant-{q}.log")
        res["steps"][f"quant_{q}"] = {"rc": rc, "s": s, "bytes": p.stat().st_size if p.exists() else 0}; save()
        if rc == 0: arts[q] = p
    for name, p in arts.items():
        api.upload_file(path_or_fileobj=str(p), path_in_repo=p.name, repo_id=HF_REPO,
                        commit_message=f"{p.name} (converted by CrispASR from rev {REV[:8]})")
        res["steps"][f"upload_{name}"] = "ok"; save()

    # 4. per-stage diff
    refs = {c: hf_hub_download("cstr/crispasr-regression-fixtures", f"r2t2/{c}/ref.gguf", repo_type="dataset", local_dir="/tmp/fx") for c in ("jfk", "zh")}
    wav = {"jfk": str(REPO / "samples/jfk.wav"), "zh": str(REPO / "samples/paraformer_zh.wav")}
    res["diff"] = {}
    for name, p in arts.items():
        for c in ("jfk", "zh"):
            rc, s, out = run([str(bin_ / "crispasr-diff"), "qwen3", str(p), refs[c], wav[c]], f"diff-{name}-{c}.log", timeout=3600)
            res["diff"][f"{name}/{c}"] = [l for l in out.splitlines() if l.startswith("[")]
            save()

    # 5. offline transcripts + streaming traces
    res["offline"] = {}
    for name, p in arts.items():
        for c, w in wav.items():
            rc, s, out = run([str(bin_ / "crispasr"), "--backend", "qwen3", "-m", str(p), "-f", w, "--no-prints", "-t", "4"], f"asr-{name}-{c}.log", timeout=3600)
            res["offline"][f"{name}/{c}"] = out.strip().splitlines()[-1] if out.strip() else f"rc={rc}"
            save()
    res["stream"] = {}
    for name in ("f16", "q4_k"):
        if name not in arts: continue
        for c, w in wav.items():
            o = OUT / f"stream-{name}-{c}.json"
            env = dict(os.environ, CRISPASR_QWEN3_STREAM_TRACE="2")
            rc, s, out = run([sys.executable, str(REPO / "tools/qwen3_stream_client.py"), "-m", str(arts[name]), "-f", w,
                              "-o", str(o), "--bin", str(bin_ / "crispasr"), "--port", "48400"], f"stream-{name}-{c}.log", env=env, timeout=7200)
            res["stream"][f"{name}/{c}"] = {"rc": rc, "s": s, "tail": out.strip()[-300:]}
            save()
except BaseException:
    res["errors"].append(traceback.format_exc())
finally:
    save()
    shutil.rmtree(REPO, ignore_errors=True)
