#!/usr/bin/env python3
"""Raon 1B ODE trajectory watch (#387-adj) — observe where real synthesis blows up.

The DiT + both CFG arms are proven exact (magnitude+direction) for realistic
inputs, yet real 1B synthesis produces an RMS-hot mel. This runs a REAL synthesis
with CRISPASR_F5_DUMP_DIR and reports the std/max of x at every ODE step, plus the
final gen_mel (vocos_input) stats. If x grows monotonically → the ODE diverges;
if x is normal until the vocoder → the bug is post-ODE (gen-mel slice / vocoder).
Compares against the reference gen_mel std≈0.9. RAON_SIZE=1B|0.3B; chr1s4.
"""
import json, os, subprocess, sys, time, glob
from pathlib import Path
import numpy as np

WORK = Path("/kaggle/working"); TMP = Path("/kaggle/temp"); TMP.mkdir(exist_ok=True)
DUMP = TMP / "dump"; DUMP.mkdir(exist_ok=True)
SIZE = os.environ.get("RAON_SIZE", "1B")
CRISPASR_URL = "https://github.com/CrispStrobe/CrispASR.git"
CRISPASR_REF = os.environ.get("CRISPASR_REF", "feat/raon-opentts-1b")
CLONE = TMP / "CrispASR"


def sh(cmd, **kw): return subprocess.run(cmd, shell=True, capture_output=True, text=True, **kw)
def step(name, **kv): print(f"[{time.strftime('%H:%M:%S')}] {name} " + json.dumps(kv), flush=True)


if not CLONE.exists():
    for _ in range(4):
        if subprocess.run(["git", "clone", "--depth", "1", "--branch", CRISPASR_REF, CRISPASR_URL, str(CLONE)]).returncode == 0:
            break
        time.sleep(15)
for _ in range(4):
    if subprocess.run(["git", "submodule", "update", "--init", "--recursive", "ggml", "third_party/c2pa-audio"],
                      cwd=str(CLONE)).returncode == 0 or (CLONE / "ggml" / "CMakeLists.txt").exists():
        break
    time.sleep(15)
sys.path.insert(0, str(CLONE / "tools" / "kaggle"))
import kaggle_harness as kh  # noqa: E402
kh.init_progress()
HF_TOKEN = kh.resolve_hf_token()
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "huggingface_hub"], check=False)
from huggingface_hub import hf_hub_download  # noqa: E402

kh.install_build_toolchain()
arch = kh.detect_cuda_arch()
flags = ["-DCMAKE_BUILD_TYPE=Release"] + kh.cuda_build_flags(arch) + kh.cache_and_link_flags()
r = sh(f"cd {CLONE} && cmake -G Ninja -B build " + " ".join(flags), timeout=1200)
if r.returncode != 0:
    step("cmake_FAIL", err=r.stderr[-1500:]); sys.exit(1)
with kh.build_heartbeat("build", interval_s=30):
    kh.sh_with_progress(f"cmake --build build -j{kh.safe_build_jobs(gpu=True)} --target crispasr-cli", cwd=str(CLONE))
CLI = CLONE / "build" / "bin" / "crispasr"
if not CLI.exists():
    c = [p for p in (CLONE / "build").rglob("crispasr") if p.is_file() and os.access(p, os.X_OK)]
    CLI = c[0] if c else None
os.environ["LD_LIBRARY_PATH"] = str(CLI.parent) + ":" + os.environ.get("LD_LIBRARY_PATH", "")
gguf = hf_hub_download(f"cstr/raon-opentts-{SIZE.lower()}-GGUF", f"raon-opentts-{SIZE.lower()}-f16.gguf",
                       local_dir=str(TMP / "models"), token=HF_TOKEN or None)
ref_wav = CLONE / "samples" / "jfk.wav"
backend = {"0.3B": "raon", "1B": "raon-1b"}[SIZE]
step("built", gguf=os.path.basename(gguf), backend=backend)

env = dict(os.environ); env["CRISPASR_F5_DUMP_DIR"] = str(DUMP)
GEN = "The quick brown fox jumps over the lazy dog."
r = sh(f"{CLI} --backend {backend} -m {gguf} --voice {ref_wav} "
       f"--ref-text \"And so my fellow Americans\" --tts \"{GEN}\" --tts-output {WORK/'t.wav'} "
       f"-t 4 --seed 42 --i-have-rights -v", env=env, timeout=3600)
print(r.stdout[-1500:], r.stderr[-2000:], flush=True)

def stats(fn):
    a = np.fromfile(fn, dtype=np.float32)
    if a.size == 0:
        return None
    return {"n": int(a.size), "std": round(float(a.std()), 3), "max": round(float(np.abs(a).max()), 3),
            "mean": round(float(a.mean()), 3), "nan": bool(not np.isfinite(a).all())}

traj = {}
for f in sorted(glob.glob(str(DUMP / "ode_step_*.bin")), key=lambda p: int(p.split("_")[-1].split(".")[0])):
    k = int(Path(f).stem.split("_")[-1])
    traj[k] = stats(f)
result = {"size": SIZE,
          "n_ode_dumps": len(traj),
          "ode_step_stats": {str(k): traj[k] for k in sorted(traj)},
          "text_embed": stats(DUMP / "text_embed.bin") if (DUMP / "text_embed.bin").exists() else None,
          "cat_input": stats(DUMP / "cat_input.bin") if (DUMP / "cat_input.bin").exists() else None,
          "input_proj_out": stats(DUMP / "input_proj_out.bin") if (DUMP / "input_proj_out.bin").exists() else None,
          "gen_mel_vocos_input": stats(DUMP / "vocos_input.bin") if (DUMP / "vocos_input.bin").exists() else None,
          "reference_gen_mel_std": 0.9}
(WORK / "raon_traj.json").write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2), flush=True)
step("DONE", n=len(traj), gen_mel=result["gen_mel_vocos_input"])
