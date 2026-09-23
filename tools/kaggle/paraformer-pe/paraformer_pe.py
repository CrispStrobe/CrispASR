#!/usr/bin/env python3
"""paraformer: sqrt(d)+PE fix vs the rebaked references (kaldi-mel-final's).
Build the branch head, crispasr-diff paraformer on zh + jfk, and CLI
transcripts at F16 / Q8_0 / Q4_K."""
import json, os, shutil, subprocess, sys, traceback
from pathlib import Path
WORK = Path("/kaggle/working"); OUT = WORK / "out"; OUT.mkdir(parents=True, exist_ok=True)
REPO = WORK / "CrispASR"; BUILD = REPO / "build"
res = {"diff": {}, "cli": {}, "errors": []}
def save(): (OUT / "paraformer_pe.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
try:
    subprocess.check_call(["git", "clone", "--depth", "1", "--recurse-submodules", "--shallow-submodules", "-b", "fix/kaldi-mel-domain",
                           "https://github.com/CrispStrobe/CrispASR.git", str(REPO)])
    res["head"] = subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip()
    sys.path.insert(0, str(REPO / "tools" / "kaggle"))
    import kaggle_harness as kh
    kh.resolve_hf_token(); os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
    kh.install_build_toolchain()
    kh.sh(f"cmake -S {REPO} -B {BUILD} -G Ninja -DCMAKE_BUILD_TYPE=Release -DCRISPASR_OPUS=OFF -DCRISPASR_AMR=OFF " + " ".join(kh.cache_and_link_flags()))
    kh.sh(f"cmake --build {BUILD} -j$(nproc) --target crispasr crispasr-diff")
    from huggingface_hub import hf_hub_download
    wav = {"jfk": str(REPO / "samples/jfk.wav"), "zh": str(REPO / "samples/paraformer_zh.wav")}
    g = {q: hf_hub_download("cstr/paraformer-zh-GGUF", f"paraformer-zh-{q}.gguf", local_dir="/tmp/g") for q in ("f16", "q8_0", "q4_k")}
    for c in ("zh", "jfk"):
        ref = hf_hub_download("cstr/crispasr-regression-fixtures", f"paraformer/{c}/ref.gguf", repo_type="dataset", local_dir="/tmp/fx", force_download=True)
        r = subprocess.run([str(BUILD / "bin/crispasr-diff"), "paraformer", g["f16"], ref, wav[c]], capture_output=True, text=True)
        (OUT / f"diff-{c}.log").write_text(r.stdout + r.stderr[-5000:])
        rows = [l for l in r.stdout.splitlines() if l.startswith("[")]
        res["diff"][c] = {"rc": r.returncode, "n_fail": sum(1 for l in rows if l.startswith("[FAIL")), "rows": rows}
        for q, m in g.items():
            r = subprocess.run([str(BUILD / "bin/crispasr"), "--backend", "paraformer", "-m", m, "-f", wav[c], "-np", "-nt"], capture_output=True, text=True)
            res["cli"][f"{q}/{c}"] = r.stdout.strip()
        save()
except BaseException:
    res["errors"].append(traceback.format_exc())
finally:
    save()
    shutil.rmtree(REPO, ignore_errors=True)
