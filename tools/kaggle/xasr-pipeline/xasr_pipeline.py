#!/usr/bin/env python3
"""CrispASR #436 (X-ASR) — C++ parity on Kaggle.

build feat/436-xasr -> crispasr-diff xasr on every fixture (jfk, zh at 480 ms;
jfk-c160, zh-c160 at 160 ms) -> CLI transcripts -> Q8_0 / Q4_K quantized
transcripts against F16. Logs go to out/*.log, the summary to out/pipeline.json.
"""
import json, os, shutil, subprocess, sys, traceback
from pathlib import Path
WORK = Path("/kaggle/working"); OUT = WORK / "out"; OUT.mkdir(parents=True, exist_ok=True)
REPO = WORK / "CrispASR"; BUILD = REPO / "build"
res = {"diff": {}, "cli": {}, "errors": []}
def save(): (OUT / "pipeline.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
def run(cmd, log, env=None, timeout=3600):
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=env)
    (OUT / log).write_text(r.stdout + "\n--- stderr ---\n" + r.stderr[-20000:])
    return r.returncode, r.stdout, r.stderr
try:
    subprocess.check_call(["git", "clone", "--depth", "1", "--recurse-submodules", "--shallow-submodules", "-b", "feat/436-xasr",
                           "https://github.com/CrispStrobe/CrispASR.git", str(REPO)])
    res["head"] = subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip()
    sys.path.insert(0, str(REPO / "tools" / "kaggle"))
    import kaggle_harness as kh
    kh.resolve_hf_token(); os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
    kh.install_build_toolchain()
    flags = kh.cache_and_link_flags()
    with kh.build_heartbeat("cmake.configure"):
        kh.sh(f"cmake -S {REPO} -B {BUILD} -G Ninja -DCMAKE_BUILD_TYPE=Release -DCRISPASR_OPUS=OFF -DCRISPASR_AMR=OFF " + " ".join(flags))
    with kh.build_heartbeat("cmake.build"):
        kh.sh(f"cmake --build {BUILD} -j$(nproc) --target crispasr crispasr-diff crispasr-quantize")
    from huggingface_hub import hf_hub_download
    f16 = hf_hub_download("cstr/x-asr-zh-en-GGUF", "x-asr-zh-en-f16.gguf", local_dir="/tmp/g")
    wav = {"jfk": str(REPO / "samples/jfk.wav"), "zh": str(REPO / "samples/paraformer_zh.wav")}
    for tag in ("jfk", "zh", "jfk-c160", "zh-c160"):
        ref = hf_hub_download("cstr/crispasr-regression-fixtures", f"xasr/{tag}/ref.gguf", repo_type="dataset", local_dir="/tmp/fx")
        rc, out, err = run([str(BUILD / "bin/crispasr-diff"), "xasr", f16, ref, wav[tag.split("-")[0]]], f"diff-{tag}.log")
        res["diff"][tag] = {"rc": rc, "rows": [l for l in out.splitlines() if l.startswith(("[", "       "))]}
        save()
    models = {"f16": f16}
    for qt in ("q8_0", "q4_k"):
        p = f"/tmp/g/x-asr-zh-en-{qt}.gguf"
        rc, _, _ = run([str(BUILD / "bin/crispasr-quantize"), f16, p, qt], f"quant-{qt}.log")
        if rc == 0:
            models[qt] = p
            res[f"{qt}_bytes"] = Path(p).stat().st_size
    for q, m in models.items():
        for chunk in (480, 160):
            for c, w in wav.items():
                env = dict(os.environ, CRISPASR_XASR_CHUNK_MS=str(chunk))
                rc, out, err = run([str(BUILD / "bin/crispasr"), "--backend", "xasr", "-m", m, "-f", w, "-np", "-nt"],
                                   f"cli-{q}-{chunk}-{c}.log", env=env)
                res["cli"][f"{q}/{chunk}/{c}"] = out.strip() if rc == 0 else f"rc={rc}: {err[-400:]}"
        save()
except BaseException:
    res["errors"].append(traceback.format_exc())
finally:
    save()
    shutil.rmtree(REPO, ignore_errors=True)
