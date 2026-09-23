#!/usr/bin/env python3
"""CrispASR #436 (X-ASR) — build crispasr-quantize at the branch head and upload
Q8_0 + Q4_K of the uploaded F16 (the same files the xasr-pipeline parity run
transcribed with)."""
import json, os, shutil, subprocess, sys, traceback
from pathlib import Path
WORK = Path("/kaggle/working"); OUT = WORK / "out"; OUT.mkdir(parents=True, exist_ok=True)
REPO = WORK / "CrispASR"; BUILD = REPO / "build"
res = {"errors": []}
def save(): (OUT / "quant_upload.json").write_text(json.dumps(res, indent=1))
try:
    subprocess.check_call(["git", "clone", "--depth", "1", "--recurse-submodules", "--shallow-submodules", "-b", "feat/436-xasr",
                           "https://github.com/CrispStrobe/CrispASR.git", str(REPO)])
    sys.path.insert(0, str(REPO / "tools" / "kaggle"))
    import kaggle_harness as kh
    kh.resolve_hf_token(); os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
    kh.install_build_toolchain()
    flags = kh.cache_and_link_flags()
    kh.sh(f"cmake -S {REPO} -B {BUILD} -G Ninja -DCMAKE_BUILD_TYPE=Release -DCRISPASR_OPUS=OFF -DCRISPASR_AMR=OFF " + " ".join(flags))
    kh.sh(f"cmake --build {BUILD} -j$(nproc) --target crispasr-quantize")
    from huggingface_hub import HfApi, hf_hub_download
    api = HfApi()
    f16 = hf_hub_download("cstr/x-asr-zh-en-GGUF", "x-asr-zh-en-f16.gguf", local_dir="/tmp/g")
    for qt in ("q8_0", "q4_k"):
        p = f"/tmp/g/x-asr-zh-en-{qt}.gguf"
        subprocess.check_call([str(BUILD / "bin/crispasr-quantize"), f16, p, qt])
        api.upload_file(path_or_fileobj=p, path_in_repo=Path(p).name, repo_id="cstr/x-asr-zh-en-GGUF", repo_type="model")
        res[qt] = Path(p).stat().st_size; save()
except BaseException:
    res["errors"].append(traceback.format_exc())
finally:
    save()
    shutil.rmtree(REPO, ignore_errors=True)
