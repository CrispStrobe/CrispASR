#!/usr/bin/env python3
"""CrispASR #466 perf A/B on a Kaggle CPU: Nemotron-3-Diarization streaming cost.

Two builds of crispasr-diff: default (GGML_NATIVE) and GGML_BLAS=ON (OpenBLAS,
used via CRISPASR_NEMOTRON3_DIAR_ACCEL=1). On the 60 s AMI clip, offline and
low_latency, every arm runs the full parity diff against transformers and
reports its timing split (graph build+alloc vs compute):
  attention: manual (default) | cont (contiguous Q/K) | flash (ggml_flash_attn_ext)
  weights:   F32 | F16 | NVIDIA q8_0
  BLAS:      off | on
plus the catch-up path (3 s blocks, up to 8 chunks per forward) with its DER.
"""
import json, os, shutil, subprocess, sys, time, traceback
from pathlib import Path

WORK = Path("/kaggle/working"); OUT = WORK / "out"; OUT.mkdir(parents=True, exist_ok=True)
REPO = WORK / "CrispASR"
BRANCH = os.environ.get("CRISPASR_REF", "main")
G = Path("/tmp/n3d"); G.mkdir(exist_ok=True)
res = {"steps": {}, "errors": [], "runs": {}, "der": {}}
GPU = shutil.which("nvidia-smi") is not None  # the -gpu bootstrap: CUDA build, GPU arms
res["gpu"] = GPU

def save(): (OUT / "perf.json").write_text(json.dumps(res, indent=1))
def run(cmd, log, timeout=None, env=None):
    t = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=env)
    (OUT / log).write_text(r.stdout[-200000:] + "\n--- stderr ---\n" + r.stderr[-100000:])
    return r.returncode, round(time.time() - t, 1), r.stdout, r.stderr

def read_rttm(p):
    return [(float(f[3]), float(f[3]) + float(f[4]), f[7]) for f in (l.split() for l in open(p)) if f and f[0] == "SPEAKER"]

def read_segtxt(txt):
    return [(float(a), float(b), c) for a, b, c in (l.split() for l in txt.strip().splitlines() if len(l.split()) == 3)]

def frame_der(ref, hyp):
    import numpy as np
    from scipy.optimize import linear_sum_assignment
    T = int(round(max([e for _, e, _ in ref + hyp] + [0]) * 100)) + 1
    def mat(segs):
        ids = sorted({s for _, _, s in segs}); m = np.zeros((T, max(1, len(ids))), bool)
        for s, e, k in segs:
            m[int(round(s * 100)):int(round(e * 100)), ids.index(k)] = True
        return m
    R, H = mat(ref), mat(hyp)
    ri, hi = linear_sum_assignment(-(R.T.astype(np.int64) @ H.astype(np.int64)))
    correct = sum((R[:, a] & H[:, b]).astype(np.int64) for a, b in zip(ri, hi))
    nr, nh = R.sum(1), H.sum(1)
    err = np.maximum(nr - nh, 0).sum() + np.maximum(nh - nr, 0).sum() + (np.minimum(nr, nh) - correct).sum()
    return round(float(err / nr.sum()), 4)

try:
    res["cpu"] = subprocess.run(["bash", "-c", "lscpu | grep -E 'Model name|^CPU\\(s\\)|Thread|Flags' | cut -c1-2000"],
                                capture_output=True, text=True).stdout
    res["avx512"] = "avx512f" in res["cpu"]
    subprocess.check_call(["git", "clone", "--depth", "1", "--recurse-submodules", "--shallow-submodules", "-b", BRANCH,
                           "https://github.com/CrispStrobe/CrispASR.git", str(REPO)])
    res["commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(REPO), text=True).strip()
    # before kaggle_harness imports huggingface_hub (see feedback: pip before import)
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "gguf", "safetensors", "librosa",
                           "git+https://github.com/huggingface/transformers.git"])
    subprocess.run(["bash", "-c", "apt-get install -y -q libopenblas-dev >/dev/null 2>&1 || sudo apt-get install -y -q libopenblas-dev"])
    sys.path.insert(0, str(REPO / "tools" / "kaggle"))
    import kaggle_harness as kh
    kh.resolve_hf_token(); os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
    from huggingface_hub import snapshot_download
    save()

    kh.install_build_toolchain()
    flags = kh.cache_and_link_flags()
    builds = {"native": []}  # OpenBLAS measured no gain (run 1) - dropped
    if GPU:
        builds = {"cuda": kh.cuda_build_flags()}
    bins = {}
    for name, extra in builds.items():
        b = REPO / f"build-{name}"
        try:
            with kh.build_heartbeat(f"build.{name}"):
                kh.sh(f"cmake -S {REPO} -B {b} -G Ninja -DCMAKE_BUILD_TYPE=Release -DCRISPASR_OPUS=OFF -DCRISPASR_AMR=OFF "
                      + " ".join(flags + extra))
                kh.sh(f"cmake --build {b} -j$(nproc) --target crispasr-diff")
            bins[name] = b / "bin" / "crispasr-diff"
            res["steps"][f"build_{name}"] = "ok"
        except Exception as e:
            res["steps"][f"build_{name}"] = f"FAILED {e}"
        save()

    md = Path(snapshot_download("nvidia/Nemotron-3-Diarization", local_dir=str(G / "hf")))
    arts = {"nvidia_q8_0": md / "Nemotron-3-Diarization.q8_0.gguf"}
    for ot in ("f32", "f16"):
        p = G / f"nemotron3-diar-{ot}.gguf"
        rc, s_, _, _ = run([sys.executable, str(REPO / "models/convert-nemotron3-diar-to-gguf.py"), "--input", str(md),
                            "--output", str(p), "--outtype", ot], f"convert-{ot}.log")
        if rc == 0: arts[ot] = p
    nsc = G / "nsc"
    subprocess.run(["git", "clone", "--depth", "1", "https://github.com/NVIDIA/NeMo-Speech.cpp.git", str(nsc)], check=False)
    subprocess.run(["git", "lfs", "pull", "--include", "test_files/diar/*"], cwd=str(nsc), check=False)
    ami = nsc / "test_files/diar/ami_en2002d_2132"
    wav, rttm = ami.with_suffix(".wav"), read_rttm(ami.with_suffix(".rttm"))

    refs = {}
    for mode in ("offline", "low_latency"):
        env = dict(os.environ); env["NEMOTRON3_DIAR_MODE"] = mode
        ref = G / f"ref-{mode}.gguf"
        rc, s_, _, _ = run([sys.executable, str(REPO / "tools/dump_reference.py"), "--backend", "nemotron3-diar",
                            "--model-dir", str(md), "--audio", str(wav), "--output", str(ref)], f"ref-{mode}.log",
                           timeout=3600, env=env)
        res["steps"][f"ref_{mode}"] = {"rc": rc, "s": s_}
        if rc == 0: refs[mode] = ref
    save()

    # run 2: the flash offline crash fixed; manual vs flash, plus the finer
    # streaming overhead split (mel / embed / cache update)
    # run 3: the shipped defaults ("default" sets no attention env var) plus the
    # q8_0 users actually download
    ARMS = [("native", w, "default", False) for w in ("f32", "nvidia_q8_0")]
    if GPU:
        ARMS = [("cuda", w, "default", False) for w in ("f32", "nvidia_q8_0")]
    for mode, ref in refs.items():
        for bname, wname, attn, accel in ARMS:
            if bname not in bins or wname not in arts:
                continue
            key = f"{mode}/{bname}/{wname}/{attn}/{'accel' if accel else 'cpu'}"
            env = dict(os.environ)
            env["CRISPASR_NEMOTRON3_DIAR_BENCH"] = "1"
            if attn != "default": env["CRISPASR_NEMOTRON3_DIAR_ATTN"] = attn
            if accel: env["CRISPASR_NEMOTRON3_DIAR_ACCEL"] = "1"
            if mode != "offline": env["CRISPASR_NEMOTRON3_DIAR_MODE"] = mode
            cu = G / f"catchup-{key.replace('/', '_')}.txt"
            env["CRISPASR_DIFF_CATCHUP_SEGMENTS_OUT"] = str(cu)
            rc, s_, out, err = run([str(bins[bname]), "nemotron3-diar", str(arts[wname]), str(ref), str(wav)],
                                   f"diff-{key.replace('/', '_')}.log", timeout=5400, env=env)
            res["runs"][key] = {"rc": rc, "s": s_,
                                "bench": [l.strip() for l in err.splitlines() if "nemotron3_diar_bench" in l or "using" in l],
                                "rows": [l[:120] for l in out.splitlines() if l.startswith("[")]}
            if cu.exists() and cu.read_text().strip():
                res["der"][key + "/catchup"] = frame_der(rttm, read_segtxt(cu.read_text()))
            save()
except BaseException:
    res["errors"].append(traceback.format_exc())
finally:
    save()
    shutil.rmtree(REPO, ignore_errors=True)
