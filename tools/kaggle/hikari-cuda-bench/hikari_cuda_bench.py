# CrispASR — hikari (sbintuitions/hikari-medium) on CUDA (Kaggle, GPU).
#
# Why a GPU: on an Apple M1 hikari runs at ~1.8 s per audio-second, and the M1
# was shared with other GPU jobs, so its timings could not separate slow
# kernels from contention. This measures the same code on an NVIDIA GPU.
#
# 1. Builds main with CUDA and runs the regression suite for hikari-medium
#    (q8_0 + Silero, English transcript) — the canonical
#    tools/kaggle/crispasr-regression.py, exec'd from the clone.
# 2. With that build: f16 and q8_0, GPU and CPU, on jfk (11 s) and jfk x3
#    (33 s), English -> German. Prints HIKARI_BENCH (per-stage ms, ms per
#    audio-second) and whether the German equals the reference text.
#
# Works on a P100 (ggml compiles its own kernels; no torch). One push, then
# read the log. Do not re-push in a loop (tools/kaggle/README.md).
import os
import subprocess
import sys
import wave
from pathlib import Path

os.environ["PYTHONUNBUFFERED"] = "1"
SCRIPT_VERSION = "hikari-cuda-bench v1"
print(SCRIPT_VERSION, flush=True)
os.environ.setdefault("CRISPASR_REF", "main")
os.environ.setdefault("CRISPASR_REGRESSION_MODE", "validate")
os.environ.setdefault("CRISPASR_REGRESSION_BUILD", "cuda")
os.environ.setdefault("CRISPASR_REGRESSION_BACKENDS", "hikari-medium")

subprocess.run(["nvidia-smi"], check=False)
subprocess.run(["nvidia-smi", "--query-gpu=name,compute_cap,memory.total", "--format=csv"], check=False)

WORK = Path("/kaggle/working")
REPO = WORK / "CrispASR-bootstrap"
if not REPO.exists():
    subprocess.check_call(
        ["git", "clone", "--depth", "50", "--no-single-branch", "https://github.com/CrispStrobe/CrispASR.git", str(REPO)]
    )
subprocess.check_call(["git", "checkout", os.environ["CRISPASR_REF"]], cwd=str(REPO))
print("bootstrap sha:", subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(REPO), text=True).strip(), flush=True)

script = REPO / "tools" / "kaggle" / "crispasr-regression.py"
print(f"exec {script}", flush=True)
sys.argv[0] = str(script)
suite_exit = 0
try:
    exec(compile(script.read_text(), str(script), "exec"))
except SystemExit as e:  # keep going: the benchmark below is the point
    suite_exit = e.code or 0
print(f"regression suite exit: {suite_exit}", flush=True)

GERMAN = "Und meine Mit-Amerikaner fragen nicht, was dein Land für dich tun kann. Frag, was du für dein Land tun kannst."

try:
    from huggingface_hub import hf_hub_download

    rev = "20311f04bd4691aefeebdabe182ac9ac770d6696"
    files = {f: hf_hub_download("cstr/hikari-medium-GGUF", f, revision=rev)
             for f in ("hikari-medium-f16.gguf", "hikari-medium-q8_0.gguf", "ggml-silero-v6.2.0.bin")}
    exe = "/kaggle/working/build/bin/crispasr"
    jfk = "/kaggle/working/CrispASR/samples/jfk.wav"
    jfk3 = str(WORK / "jfk3.wav")
    with wave.open(jfk) as w:
        params, frames = w.getparams(), w.readframes(w.getnframes())
    with wave.open(jfk3, "wb") as w:
        w.setparams(params)
        w.writeframes(frames * 3)
    env = dict(os.environ, HIKARI_BENCH="1", HIKARI_VAD_MODEL=files["ggml-silero-v6.2.0.bin"])
    for q in ("f16", "q8_0"):
        for dev, extra in (("GPU", []), ("CPU", ["-ng"])):
            for name, wav in (("jfk", jfk), ("jfk x3", jfk3)):
                if dev == "CPU" and name == "jfk x3":
                    continue  # CPU is the slow reference arm; one clip is enough
                r = subprocess.run([exe, "--backend", "hikari", "-m", files[f"hikari-medium-{q}.gguf"], "-l", "en",
                                    "-tl", "de", "-f", wav, "-np"] + extra,
                                   capture_output=True, text=True, timeout=3600, env=env)
                text = r.stdout.strip().splitlines()[-1] if r.stdout.strip() else ""
                bench = [l for l in r.stderr.splitlines() if "hikari_bench" in l]
                same = text.startswith(GERMAN) if name == "jfk" else None
                print(f"RESULT {q} {dev} {name}: rc={r.returncode} equal_ref={same}", flush=True)
                print(f"  bench: {bench[-1] if bench else '(none)'}", flush=True)
                print(f"  text: {text[:300]!r}", flush=True)
                if r.returncode != 0:
                    print(r.stderr[-1500:], flush=True)
except Exception as e:
    print(f"benchmark failed: {e!r}", flush=True)

sys.exit(suite_exit)
