#!/usr/bin/env python3
"""Why does PR #470's build fail to initialise the pinned legacy canary-1b-v2.gguf (exit 13)?
Builds the PR head (or CRISPASR_REF) and runs the legacy files with full stderr."""
import json, os, subprocess, sys, traceback
from pathlib import Path
OUT = Path("/kaggle/working/out"); OUT.mkdir(parents=True, exist_ok=True)
REPO = Path("/tmp/CrispASR"); res = {"runs": {}, "errors": []}
REF = os.environ.get("CRISPASR_REF", "pr470-work")
def save(): (OUT / "result.json").write_text(json.dumps(res, indent=1))
try:
    subprocess.check_call(["git", "clone", "https://github.com/CrispStrobe/CrispASR.git", str(REPO)])
    subprocess.check_call(["git", "checkout", REF], cwd=str(REPO))
    subprocess.run(["git", "submodule", "update", "--init", "--recursive", "--depth", "1"], cwd=str(REPO))
    res["commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(REPO), text=True).strip()
    sys.path.insert(0, str(REPO / "tools" / "kaggle")); import kaggle_harness as kh
    tok = kh.resolve_hf_token()
    from huggingface_hub import hf_hub_download
    kh.install_build_toolchain()
    kh.sh(f"cmake -S {REPO} -B {REPO}/build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCRISPASR_OPUS=OFF -DCRISPASR_AMR=OFF " + " ".join(kh.cache_and_link_flags()))
    kh.sh(f"cmake --build {REPO}/build -j$(nproc) --target crispasr-cli")
    for fname, rev in (("canary-1b-v2.gguf", "b3715a517928f8f68833142c90fc5810ad583210"), ("canary-1b-v2-q8_0.gguf", None)):
        g = hf_hub_download("cstr/canary-1b-v2-GGUF", fname, revision=rev, cache_dir="/tmp/g")
        for extra in ([], ["-sl", "en", "-tl", "en"]):
            r = subprocess.run([str(REPO / "build/bin/crispasr"), "-m", g, "-f", str(REPO / "samples/jfk.wav"), "-v"] + extra,
                               capture_output=True, text=True, timeout=1800)
            res["runs"][f"{fname} {' '.join(extra)}"] = {"rc": r.returncode, "stdout": r.stdout[-800:], "stderr": r.stderr[-6000:]}
            save()
        os.remove(g)
except BaseException:
    res["errors"].append(traceback.format_exc())
finally:
    save()
