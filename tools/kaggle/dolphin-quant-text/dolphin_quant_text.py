#!/usr/bin/env python3
"""CrispASR #436 — Dolphin: does quantization change the transcript?

The encoder-output A/B showed chaotic worst-frame cosines (0.3..0.99, not
monotonic in what is kept at F16) while zh text stayed identical. Decide the
default by the metric users see: transcripts of F16 vs Q8_0 vs Q4_K on every
Mandarin clip at hand, CER against the F16 transcript.
"""
import json, os, shutil, subprocess, sys, traceback
from pathlib import Path
WORK = Path("/kaggle/working"); OUT = WORK / "out"; OUT.mkdir(parents=True, exist_ok=True)
REPO = WORK / "CrispASR"; BUILD = REPO / "build"
res = {"clips": {}, "cer_vs_f16": {}, "errors": []}
def save(): (OUT / "quant_text.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
def cer(a, b):
    a, b = a.replace(" ", ""), b.replace(" ", "")
    d = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        p, d[0] = d[0], i
        for j, cb in enumerate(b, 1):
            p, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, p + (ca != cb))
    return d[len(b)] / max(1, len(b))
try:
    subprocess.check_call(["git", "clone", "--depth", "1", "--recurse-submodules", "--shallow-submodules", "-b", "feat/436-dolphin",
                           "https://github.com/CrispStrobe/CrispASR.git", str(REPO)])
    sys.path.insert(0, str(REPO / "tools" / "kaggle"))
    import kaggle_harness as kh
    kh.resolve_hf_token(); os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
    kh.install_build_toolchain()
    flags = kh.cache_and_link_flags()
    with kh.build_heartbeat("cmake.configure"):
        kh.sh(f"cmake -S {REPO} -B {BUILD} -G Ninja -DCMAKE_BUILD_TYPE=Release -DCRISPASR_OPUS=OFF -DCRISPASR_AMR=OFF " + " ".join(flags))
    with kh.build_heartbeat("cmake.build"):
        kh.sh(f"cmake --build {BUILD} -j$(nproc) --target crispasr crispasr-quantize")
    from huggingface_hub import hf_hub_download, list_repo_files
    f16 = hf_hub_download("cstr/dolphin-cn-dialect-small-streaming-GGUF", "dolphin-cn-dialect-small-streaming-f16.gguf", local_dir="/tmp/g")
    models = {"f16": f16}
    for qt in ("q8_0", "q4_k"):
        p = f"/tmp/g/{qt}.gguf"
        subprocess.check_call([str(BUILD / "bin/crispasr-quantize"), f16, p, qt], stdout=subprocess.DEVNULL)
        models[qt] = p
    clips = {"paraformer_zh": str(REPO / "samples/paraformer_zh.wav"), "jfk": str(REPO / "samples/jfk.wav")}
    for repo in ("csukuangfj/sherpa-onnx-paraformer-zh-2023-09-14", "csukuangfj/sherpa-onnx-streaming-paraformer-trilingual-zh-cantonese-en"):
        try:
            for f in list_repo_files(repo):
                if f.startswith("test_wavs/") and f.endswith(".wav"):
                    src = hf_hub_download(repo, f, local_dir="/tmp/w/" + repo.split("/")[1])
                    dst = f"/tmp/w/{repo.split('-')[2]}_{Path(f).stem}.wav"
                    subprocess.check_call(["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", src, "-ar", "16000", "-ac", "1", dst])
                    clips[Path(dst).stem] = dst
        except Exception as e:
            res["errors"].append(f"{repo}: {e}")
    for c, w in clips.items():
        res["clips"][c] = {}
        for q, m in models.items():
            r = subprocess.run([str(BUILD / "bin/crispasr"), "--backend", "dolphin", "-m", m, "-f", w, "-np", "-nt"],
                               capture_output=True, text=True, timeout=900)
            res["clips"][c][q] = r.stdout.strip()
        save()
    for q in ("q8_0", "q4_k"):
        res["cer_vs_f16"][q] = {c: round(cer(v[q], v["f16"]), 4) for c, v in res["clips"].items()}
except BaseException:
    res["errors"].append(traceback.format_exc())
finally:
    save()
    shutil.rmtree(REPO, ignore_errors=True)
