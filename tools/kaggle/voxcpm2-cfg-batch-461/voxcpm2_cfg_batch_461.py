#!/usr/bin/env python3
"""#461 — voxcpm2 CFG batch-2 LocDiT: A/B on CPU and CUDA.

Arms (same text, same seed, voxcpm2-q8_0.gguf):
  cpu_ref    CRISPASR_VOXCPM2_CFG_BATCH=0, -ng   (two LocDiT forwards per Euler step)
  cpu_batch  default, -ng                        (one batch-2 forward)
  gpu_ref    CRISPASR_VOXCPM2_CFG_BATCH=0, CUDA
  gpu_batch  default, CUDA
Reported: WAV md5, per-step cfm ms (CRISPASR_VOXCPM2_BENCH), total ms, and a
parakeet ASR roundtrip of every WAV. Exactness gate: cpu_batch WAV equals
cpu_ref (bitwise, or PCM cos >= 0.9999 if the batch changes a reduction order).
"""
import hashlib, json, os, re, subprocess, sys, traceback
from pathlib import Path
WORK = Path("/kaggle/working"); OUT = WORK / "out"; OUT.mkdir(parents=True, exist_ok=True)
REPO = Path("/tmp/CrispASR"); BUILD = REPO / "build"
res = {"errors": [], "arms": {}}
def save(): (OUT / "r461.json").write_text(json.dumps(res, indent=1, default=str))
TEXT = "Hello, this is a short test sentence. We are testing the speed of the speech synthesis."
try:
    subprocess.check_call(["git", "clone", "--depth", "1", "--recurse-submodules", "--shallow-submodules", "-b",
                           "perf/461-voxcpm2-cfg-batch", "https://github.com/CrispStrobe/CrispASR.git", str(REPO)])
    res["head"] = subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip()
    sys.path.insert(0, str(REPO / "tools" / "kaggle"))
    import kaggle_harness as kh
    kh.resolve_hf_token()
    kh.install_build_toolchain()
    arch = kh.detect_cuda_arch()
    kh.sh(f"cmake -S {REPO} -B {BUILD} -G Ninja -DCMAKE_BUILD_TYPE=Release -DCRISPASR_OPUS=OFF -DCRISPASR_AMR=OFF "
          + " ".join(kh.cuda_build_flags(arch) + kh.cache_and_link_flags()))
    with kh.build_heartbeat("build"):
        kh.sh(f"cmake --build {BUILD} -j{kh.safe_build_jobs(gpu=True)} --target crispasr-cli")
    res["gpu"] = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
    from huggingface_hub import hf_hub_download
    model = hf_hub_download("cstr/voxcpm2-GGUF", "voxcpm2-q8_0.gguf")
    asr = hf_hub_download("cstr/parakeet-tdt-0.6b-v2-GGUF", "parakeet-tdt-0.6b-v2-q4_k.gguf")
    cli = str(BUILD / "bin" / "crispasr")
    arms = [("cpu_ref", {"CRISPASR_VOXCPM2_CFG_BATCH": "0"}, ["-ng"]), ("cpu_batch", {}, ["-ng"]),
            ("gpu_ref", {"CRISPASR_VOXCPM2_CFG_BATCH": "0"}, []), ("gpu_batch", {}, [])]
    import numpy as np, soundfile as sf
    for name, env, extra in arms:
        wav = WORK / f"{name}.wav"
        e = dict(os.environ, CRISPASR_VOXCPM2_BENCH="1", **env)
        r = subprocess.run([cli, "--backend", "voxcpm2", "-m", model, "--tts", TEXT, "--tts-output", str(wav),
                            "--seed", "42", "-v"] + extra, capture_output=True, text=True, env=e, timeout=3600)
        log = r.stdout + r.stderr
        (OUT / f"{name}.log").write_text(log[-60000:])
        A = res["arms"].setdefault(name, {"rc": r.returncode})
        m = re.search(r"cfm\s+([\d.]+) ms", log); A["cfm_ms_per_step"] = float(m.group(1)) if m else None
        m = re.search(r"AR loop (\d+) steps, ([\d.]+) ms", log); A["ar"] = m.groups() if m else None
        m = re.search(r"voxcpm2: total ([\d.]+) ms", log); A["total_ms"] = float(m.group(1)) if m else None
        if wav.exists():
            A["md5"] = hashlib.md5(wav.read_bytes()).hexdigest()
            a, sr = sf.read(str(wav)); A["dur_s"] = len(a) / sr
            rr = subprocess.run([cli, "--backend", "parakeet", "-m", asr, "-f", str(wav), "-np", "-nt"],
                                capture_output=True, text=True, timeout=600)
            A["asr"] = rr.stdout.strip()
        save()
    def pcm(n): return sf.read(str(WORK / f"{n}.wav"))[0]
    for ref, b in (("cpu_ref", "cpu_batch"), ("gpu_ref", "gpu_batch")):
        try:
            x, y = pcm(ref), pcm(b); n = min(len(x), len(y))
            res[f"{b}_vs_{ref}"] = {"same_md5": res["arms"][ref].get("md5") == res["arms"][b].get("md5"),
                                   "len": [len(x), len(y)],
                                   "cos": float(np.dot(x[:n], y[:n]) / (np.linalg.norm(x[:n]) * np.linalg.norm(y[:n]) + 1e-12))}
        except Exception as ex:
            res[f"{b}_vs_{ref}"] = str(ex)
except BaseException:
    res["errors"].append(traceback.format_exc()[-3000:])
finally:
    save()
