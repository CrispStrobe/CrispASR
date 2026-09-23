#!/usr/bin/env python3
"""CrispASR #436 — which Dolphin tensors are quantization-sensitive?

Q8_0 loses up to cos 0.34 on single frames of the encoder output (jfk) while F16
is exact. Quantize the F16 GGUF with candidate tensor groups kept at F16 — one
group at a time, then all — and report the worst-frame encoder-output cosine,
the CTC top-1 agreement and whether the decoded text matches upstream.
"""
import json, os, shutil, subprocess, sys, traceback
from pathlib import Path
WORK = Path("/kaggle/working"); OUT = WORK / "out"; OUT.mkdir(parents=True, exist_ok=True)
REPO = WORK / "CrispASR"; BUILD = REPO / "build"
res = {"variants": {}, "errors": []}
def save(): (OUT / "quant_ab.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
GROUPS = {
    "base": [],
    "attn_pos": [r"attn\.pos\.weight"],
    "cgmlp": [r"cgmlp\.proj[12]\.weight"],
    "merge_proj": [r"merge_proj\.weight"],
    "sub_out": [r"enc\.sub\.out\.weight"],
    "ffn": [r"\.(ff|ffm)\.w[12]\.weight"],
}
GROUPS["all"] = [p for k in ("attn_pos", "cgmlp", "merge_proj", "sub_out") for p in GROUPS[k]]
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
        kh.sh(f"cmake --build {BUILD} -j$(nproc) --target crispasr-quantize crispasr-diff")
    from huggingface_hub import hf_hub_download
    f16 = hf_hub_download("cstr/dolphin-cn-dialect-small-streaming-GGUF", "dolphin-cn-dialect-small-streaming-f16.gguf", local_dir="/tmp/g")
    refs = {c: hf_hub_download("cstr/crispasr-regression-fixtures", f"dolphin/{c}/ref.gguf", repo_type="dataset", local_dir="/tmp/fx") for c in ("zh", "jfk")}
    wav = {"zh": str(REPO / "samples/paraformer_zh.wav"), "jfk": str(REPO / "samples/jfk.wav")}
    for qt in ("q8_0", "q4_k"):
        for name, pats in GROUPS.items():
            if qt == "q4_k" and name not in ("base", "all"):
                continue
            out = Path(f"/tmp/g/{qt}-{name}.gguf")
            cmd = [str(BUILD / "bin/crispasr-quantize"), f16, str(out), qt]
            for p in pats:
                cmd += ["--tensor-type", f"{p}=f16"]
            subprocess.run(cmd, capture_output=True, text=True)
            v = {"bytes": out.stat().st_size if out.exists() else 0}
            for c in ("zh", "jfk"):
                r = subprocess.run([str(BUILD / "bin/crispasr-diff"), "dolphin", str(out), refs[c], wav[c]], capture_output=True, text=True)
                rows = {l.split()[1]: l for l in r.stdout.splitlines() if l.startswith("[")}
                enc = rows.get("encoder_output(ref_fbank)", "")
                v[c] = {"enc": enc[enc.find("cos_min"):enc.find("max_abs")].strip(), "text_ok": rows.get("text", "").startswith("[PASS]"),
                        "blk11": rows.get("enc_blk_11", "")[rows.get("enc_blk_11", "").find("cos_min"):][:40]}
            res["variants"][f"{qt}/{name}"] = v
            out.unlink(missing_ok=True)
            save()
except BaseException:
    res["errors"].append(traceback.format_exc())
finally:
    save()
    shutil.rmtree(REPO, ignore_errors=True)
