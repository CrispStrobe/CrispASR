#!/usr/bin/env python3
"""omniasr-ctc-1b-v2: why does GH CPU emit '...Ask what you can do. كan do fo يu اoutي.'?

The pinned regression gate (WER 0.25) hides it. Same machine, same binary, CUDA
build: GPU vs CPU (-ng), threads 1/2/4, q4_k / q8_0 / F16. Then per-stage dumps
(CRISPASR_OMNIASR_DUMP_DIR) GPU vs CPU on the pinned q4_k, reported as per-frame
cosine so a tail-only divergence shows as the minimum over the last frames.
"""
import json, os, subprocess, sys, traceback
from pathlib import Path
OUT = Path("/kaggle/working/out"); OUT.mkdir(parents=True, exist_ok=True)
REPO = Path("/tmp/CrispASR"); G = Path("/tmp/g"); G.mkdir(exist_ok=True)
REF = os.environ.get("CRISPASR_REF", "main")
REPO_ID = "cstr/omniASR-CTC-1B-v2-GGUF"; PIN = "e05a1d261ca3f0dfe6932ee9d50794860d849ea0"
res = {"errors": [], "runs": {}, "dumps": {}}
def save(): (OUT / "result.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
try:
    subprocess.check_call(["git", "clone", "-b", REF, "https://github.com/CrispStrobe/CrispASR.git", str(REPO)])
    subprocess.run(["git", "submodule", "update", "--init", "--recursive", "--depth", "1"], cwd=str(REPO))
    res["commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(REPO), text=True).strip()
    sys.path.insert(0, str(REPO / "tools" / "kaggle")); import kaggle_harness as kh
    tok = kh.resolve_hf_token()
    if tok: os.environ["HF_TOKEN"] = tok
    import numpy as np
    from huggingface_hub import hf_hub_download
    kh.install_build_toolchain()
    arch = kh.detect_cuda_arch()
    kh.sh(f"cmake -S {REPO} -B {REPO}/build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCRISPASR_OPUS=OFF -DCRISPASR_AMR=OFF "
          + " ".join(kh.cuda_build_flags(arch) + kh.cache_and_link_flags()))
    with kh.build_heartbeat("cmake.build"):
        kh.sh(f"cmake --build {REPO}/build -j{kh.safe_build_jobs(gpu=True)} --target crispasr-cli")
    B = REPO / "build" / "bin" / "crispasr"
    res["cpu"] = subprocess.run("lscpu | grep -E 'Model name|Flags' | cut -c1-400", shell=True, capture_output=True, text=True).stdout
    jfk = REPO / "samples" / "jfk.wav"
    models = {"q4_k": hf_hub_download(REPO_ID, "omniasr-ctc-1b-v2-q4_k.gguf", revision=PIN, cache_dir=str(G))}
    save()

    def run(tag, model, extra, env=None):
        r = subprocess.run([str(B), "-m", model, "-f", str(jfk), "-np"] + extra, capture_output=True, text=True,
                           env=dict(os.environ, **(env or {})), timeout=900)
        lines = [l.strip() for l in r.stdout.splitlines() if l.strip()]
        res["runs"][tag] = {"rc": r.returncode, "text": lines[-1] if lines else "", "err": r.stderr[-600:] if r.returncode else ""}
        print(tag, res["runs"][tag]["text"], flush=True); save()
        return r

    run("q4_k.gpu", models["q4_k"], [])
    for t in (4, 2, 1):
        run(f"q4_k.cpu.t{t}", models["q4_k"], ["-ng", "-t", str(t)])
    # the regression's exact invocation minus -np: default threads
    run("q4_k.cpu.default", models["q4_k"], ["-ng"])

    # per-stage GPU vs CPU dumps. DUMP lines give [ne0, ne1]; ne0 is contiguous,
    # so for [C, T] stages each frame is one row and the per-frame cosine shows
    # whether a divergence is global or confined to the tail frames.
    import re
    dd, shapes = {}, {}
    for dev, extra in (("gpu", []), ("cpu", ["-ng", "-t", "4"])):
        d = G / f"dump_{dev}"; d.mkdir(exist_ok=True); dd[dev] = d
        r = run(f"q4_k.{dev}.dump", models["q4_k"], extra, {"CRISPASR_OMNIASR_DUMP_DIR": str(d)})
        for m in re.finditer(r"DUMP (\S+) \[(\d+), (\d+)\]", r.stderr):
            shapes[m.group(1)] = (int(m.group(2)), int(m.group(3)))
    res["shapes"] = shapes
    for f in sorted(dd["gpu"].glob("*.bin")):
        c = dd["cpu"] / f.name
        if not c.exists():
            continue
        a = np.fromfile(f, dtype=np.float32); b = np.fromfile(c, dtype=np.float32)
        ent = {"n": [int(a.size), int(b.size)]}
        if a.size == b.size:
            ent.update(cos_all=float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30)),
                       max_abs=float(np.nanmax(np.abs(a - b))), nan_cpu=int(np.isnan(b).sum()),
                       nan_gpu=int(np.isnan(a).sum()))
            sh = shapes.get(f.stem)
            if sh and f.stem != "cnn_out" and sh[0] * sh[1] == a.size:
                A = a.reshape(sh[1], sh[0]); Bm = b.reshape(sh[1], sh[0])
                cf = (A * Bm).sum(1) / (np.linalg.norm(A, axis=1) * np.linalg.norm(Bm, axis=1) + 1e-30)
                ent["cos_frame_min"] = float(cf.min()); ent["cos_frame_argmin"] = int(cf.argmin())
                ent["cos_frames_last8"] = [round(float(x), 5) for x in cf[-8:]]
                ent["n_frames_below_0.99"] = int((cf < 0.99).sum())
                ent["first_frame_below_0.99"] = int(np.argmax(cf < 0.99)) if (cf < 0.99).any() else -1
                ent["rms_cpu_last4"] = [round(float(x), 4) for x in np.sqrt((Bm[-4:] ** 2).mean(1))]
                ent["rms_gpu_last4"] = [round(float(x), 4) for x in np.sqrt((A[-4:] ** 2).mean(1))]
        res["dumps"][f.stem] = ent
    save()

    for q, fn in (("q8_0", "omniasr-ctc-1b-v2-q8_0.gguf"), ("f16", "omniasr-ctc-1b-v2.gguf")):
        try:
            models[q] = hf_hub_download(REPO_ID, fn, cache_dir=str(G))
            run(f"{q}.gpu", models[q], [])
            run(f"{q}.cpu.t4", models[q], ["-ng", "-t", "4"])
            os.remove(models[q])
        except Exception as e:
            res["errors"].append(f"{q}: {e}")
except BaseException:
    res["errors"].append(traceback.format_exc())
finally:
    save()
