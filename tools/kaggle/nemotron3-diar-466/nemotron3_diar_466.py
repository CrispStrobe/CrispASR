#!/usr/bin/env python3
"""CrispASR #466 — Nemotron-3-Diarization (streaming Sortformer v3) parity + DER, Kaggle CPU.

build (feat/466-nemotron3-diar) -> convert HF safetensors to F32 / F16 sortformer
GGUFs -> transformers-main reference dumps (tools/dump_reference.py) on
samples/multispeaker.wav and NeMo-Speech.cpp's AMI clip -> crispasr-diff
nemotron3-diar for our F32, F16 and NVIDIA's own q8_0 GGUF -> frame DER (10 ms,
no collar, overlap scored, optimal speaker mapping) of the C++ segments and of
the transformers segments against the AMI RTTM -> end-to-end crispasr
--diarize-method sortformer run. Outputs are small JSON/logs.
"""
import json, os, shutil, subprocess, sys, time, traceback
from pathlib import Path

WORK = Path("/kaggle/working"); OUT = WORK / "out"; OUT.mkdir(parents=True, exist_ok=True)
REPO = WORK / "CrispASR"; BUILD = REPO / "build"
BRANCH = os.environ.get("CRISPASR_REF", "main")
HF_MODEL = "nvidia/Nemotron-3-Diarization"
G = Path("/tmp/n3d"); G.mkdir(exist_ok=True)
res = {"steps": {}, "errors": []}
# GPU arm (the -gpu bootstrap kernel): CUDA build, offline + low_latency only,
# every arm timed on the GPU and, for the q8_0 GGUF, on the CPU as well.
GPU = shutil.which("nvidia-smi") is not None
res["gpu"] = GPU

def save(): (OUT / "pipeline.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
def run(cmd, log, timeout=None, env=None):
    t = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=env)
    (OUT / log).write_text(r.stdout[-200000:] + "\n--- stderr ---\n" + r.stderr[-100000:])
    return r.returncode, round(time.time() - t, 1), r.stdout

def read_rttm(p):
    segs = []
    for l in open(p):
        f = l.split()
        if f and f[0] == "SPEAKER":
            segs.append((float(f[3]), float(f[3]) + float(f[4]), f[7]))
    return segs

def read_segtxt(txt):
    segs = []
    for l in txt.strip().splitlines():
        f = l.split()
        if len(f) == 3:
            segs.append((float(f[0]), float(f[1]), f[2]))
    return segs

def frame_der(ref, hyp):
    import numpy as np
    from scipy.optimize import linear_sum_assignment
    T = int(round(max([e for _, e, _ in ref + hyp] + [0]) * 100)) + 1
    def mat(segs):
        ids = sorted({s for _, _, s in segs}); m = np.zeros((T, max(1, len(ids))), bool)
        for s, e, k in segs:
            m[int(round(s * 100)):int(round(e * 100)), ids.index(k)] = True
        return m, ids
    R, rid = mat(ref); H, hid = mat(hyp)
    ov = R.T.astype(np.int64) @ H.astype(np.int64)
    ri, hi = linear_sum_assignment(-ov)
    correct = np.zeros(T, np.int64)
    for a, b in zip(ri, hi):
        correct += R[:, a] & H[:, b]
    nr, nh = R.sum(1), H.sum(1)
    miss = np.maximum(nr - nh, 0).sum(); fa = np.maximum(nh - nr, 0).sum()
    conf = (np.minimum(nr, nh) - correct).sum()
    tot = nr.sum()
    return {"der": round(float((miss + fa + conf) / tot), 4), "miss": round(float(miss / tot), 4),
            "fa": round(float(fa / tot), 4), "conf": round(float(conf / tot), 4),
            "n_ref_spk": len(rid), "n_hyp_spk": len(hid)}

try:
    subprocess.check_call(["git", "clone", "--depth", "1", "--recurse-submodules", "--shallow-submodules", "-b", BRANCH,
                           "https://github.com/CrispStrobe/CrispASR.git", str(REPO)])
    res["commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(REPO), text=True).strip()
    # Before kaggle_harness: it imports huggingface_hub, and a pip upgrade of an
    # already-imported huggingface_hub leaves a half-old module in the process.
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "gguf", "safetensors", "librosa",
                           "git+https://github.com/huggingface/transformers.git"])
    sys.path.insert(0, str(REPO / "tools" / "kaggle"))
    import kaggle_harness as kh
    kh.resolve_hf_token(); os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
    import transformers
    res["transformers"] = transformers.__version__; save()
    from huggingface_hub import snapshot_download, hf_hub_download

    kh.install_build_toolchain()
    flags = kh.cache_and_link_flags()
    with kh.build_heartbeat("cmake.configure"):
        if GPU:
            flags = flags + kh.cuda_build_flags()
        kh.sh(f"cmake -S {REPO} -B {BUILD} -G Ninja -DCMAKE_BUILD_TYPE=Release -DCRISPASR_OPUS=OFF -DCRISPASR_AMR=OFF " + " ".join(flags))
    with kh.build_heartbeat("cmake.build"):
        kh.sh(f"cmake --build {BUILD} -j$(nproc) --target crispasr crispasr-diff")
    bin_ = BUILD / "bin"; res["steps"]["build"] = "ok"; save()

    md = Path(snapshot_download(HF_MODEL, local_dir=str(G / "hf")))
    q8 = md / "Nemotron-3-Diarization.q8_0.gguf"
    arts = {}
    for ot in ("f32", "f16"):
        p = G / f"nemotron3-diar-{ot}.gguf"
        rc, s, _ = run([sys.executable, str(REPO / "models/convert-nemotron3-diar-to-gguf.py"), "--input", str(md),
                        "--output", str(p), "--outtype", ot], f"convert-{ot}.log")
        res["steps"][f"convert_{ot}"] = {"rc": rc, "s": s, "bytes": p.stat().st_size if p.exists() else 0}; save()
        if rc == 0: arts[ot] = p
    if q8.exists(): arts["nvidia_q8_0"] = q8

    nsc = G / "nsc"
    subprocess.run(["git", "clone", "--depth", "1", "https://github.com/NVIDIA/NeMo-Speech.cpp.git", str(nsc)], check=False)
    subprocess.run(["git", "lfs", "pull", "--include", "test_files/diar/*"], cwd=str(nsc), check=False)
    ami = nsc / "test_files/diar/ami_en2002d_2132"
    clips = {"multispeaker": REPO / "samples/multispeaker.wav"}
    if (ami.with_suffix(".wav")).exists() and ami.with_suffix(".wav").stat().st_size > 10000:
        clips["ami"] = ami.with_suffix(".wav")
        for suf in (".json", ".rttm"):
            if ami.with_suffix(suf).exists(): shutil.copy(ami.with_suffix(suf), OUT / ("ami" + suf))
    res["clips"] = {k: str(v) for k, v in clips.items()}; save()

    MODES = ["offline", "low_latency"] if GPU else ["offline", "low_latency", "very_low_latency", "ultra_low_latency"]
    rttm = read_rttm(ami.with_suffix(".rttm")) if "ami" in clips and ami.with_suffix(".rttm").exists() else None
    res["diff"] = {}; res["der"] = {}
    import gguf
    for mode in MODES:
        env_ref = dict(os.environ); env_ref["NEMOTRON3_DIAR_MODE"] = mode
        refs = {}
        for c, w in clips.items():
            ref = G / f"ref-{mode}-{c}.gguf"
            rc, s_, out = run([sys.executable, str(REPO / "tools/dump_reference.py"), "--backend", "nemotron3-diar",
                               "--model-dir", str(md), "--audio", str(w), "--output", str(ref)],
                              f"ref-{mode}-{c}.log", timeout=3600, env=env_ref)
            res["steps"][f"ref_{mode}_{c}"] = {"rc": rc, "s": s_}; save()
            if rc == 0: refs[c] = ref
        # quantized arm only offline: parity there already bounds it
        arms = [(n, p, False) for n, p in arts.items()]
        if GPU and "nvidia_q8_0" in arts:
            arms.append(("nvidia_q8_0-cpu", arts["nvidia_q8_0"], True))
        for name, p, force_cpu in arms:
            if mode != "offline" and name == "nvidia_q8_0" and not GPU:
                continue
            for c, ref in refs.items():
                env = dict(os.environ); seg_out = G / f"seg-{mode}-{name}-{c}.txt"
                if force_cpu: env["CRISPASR_DIFF_NO_GPU"] = "1"
                env["CRISPASR_DIFF_SEGMENTS_OUT"] = str(seg_out); env["CRISPASR_NEMOTRON3_DIAR_BENCH"] = "1"
                if mode != "offline": env["CRISPASR_NEMOTRON3_DIAR_MODE"] = mode
                rc, s_, out = run([str(bin_ / "crispasr-diff"), "nemotron3-diar", str(p), str(ref), str(clips[c])],
                                  f"diff-{mode}-{name}-{c}.log", timeout=3600, env=env)
                bench = [l.strip() for l in (OUT / f"diff-{mode}-{name}-{c}.log").read_text().splitlines()
                         if "nemotron3_diar_bench" in l]
                res["diff"][f"{mode}/{name}/{c}"] = {"rc": rc, "s": s_, "bench": bench,
                    "rows": [l for l in out.splitlines() if l.startswith(("[", "  "))][:40]}
                if seg_out.exists(): shutil.copy(seg_out, OUT / seg_out.name)
                if c == "ami" and rttm and seg_out.exists():
                    res["der"][f"{mode}/cpp-{name}"] = frame_der(rttm, read_segtxt(seg_out.read_text()))
                save()
        if rttm and "ami" in refs:
            rd = gguf.GGUFReader(str(refs["ami"]))
            for k, f in rd.fields.items():
                if k.endswith("segments_text"):
                    txt = bytes(f.parts[f.data[0]]).decode()
                    (OUT / f"seg-{mode}-transformers-ami.txt").write_text(txt)
                    res["der"][f"{mode}/transformers"] = frame_der(rttm, read_segtxt(txt))
            save()

    # End to end on the AMI clip with whisper tiny: the legacy .bin path (whose
    # console now prints the diarizer's labels, not "(speaker ?)"), the unified
    # dispatcher, and a streaming preset.
    if "ami" in clips and "f16" in arts:
        asr = hf_hub_download("ggerganov/whisper.cpp", "ggml-tiny.en.bin", local_dir=str(G / "w"))
        res["cli"] = {}
        for route, extra in (("legacy", []), ("dispatch", ["--backend", "whisper"]),
                             ("legacy-low_latency", ["--sortformer-mode", "low_latency"])):
            of = G / f"cli-{route}"
            rc, s_, out = run([str(bin_ / "crispasr"), "-m", asr, "-f", str(clips["ami"]), "-t", "4", "--diarize",
                               "--diarize-method", "sortformer", "--diarize-model", str(arts["f16"]), "-oj", "-of",
                               str(of)] + extra, f"cli-{route}.log", timeout=3600)
            spk = []
            try:
                j = json.load(open(str(of) + ".json"))
                for sg in j.get("transcription", j.get("segments", [])):
                    spk.append([sg.get("timestamps", {}).get("from", sg.get("start")), sg.get("speaker"),
                                (sg.get("text") or "")[:60]])
                shutil.copy(str(of) + ".json", OUT / f"cli-{route}.json")
            except Exception as e:
                spk = [f"json read failed: {e}"]
            console = [l for l in out.splitlines() if l.startswith("[")]
            res["cli"][route] = {"rc": rc, "s": s_, "segments": spk, "console": console[:12],
                                 "console_unlabelled": sum("(speaker ?)" in l for l in console)}
            save()
except BaseException:
    res["errors"].append(traceback.format_exc())
finally:
    save()
    shutil.rmtree(REPO, ignore_errors=True)
