#!/usr/bin/env python3
"""Raon-OpenTTS end-to-end TTS→ASR roundtrip — the acceptance gate (#387-adj).

Builds CrispASR from feat/raon-opentts on Kaggle (GPU: fast DiT ODE that is
minutes-on-CPU on the VPS), synthesizes with `--backend raon` on a real
English reference voice, and transcribes the result with whisper-tiny. PASS =
the transcript covers the gen_text (proves DiT + sbhifigan mel + CPU HiFi-GAN
produce intelligible speech of the requested words). Proof-of-work: rc==0,
non-trivial wav, word overlap — a crash or silent no-op cannot mint a pass.

GPU build so the DiT runs on-device (ggml-cuda ships sm_60/75, so P100/T4 are
fine — unlike torch, which lacks P100 kernels). The vocoder stays CPU.
Datasets: chr1str/crispasr-hf-token, chr1str/crispasr-ccache.
"""

import json
import os
import re
import subprocess
import sys
import time
import wave
from pathlib import Path

WORK = Path("/kaggle/working")
TMP = Path("/kaggle/temp")
TMP.mkdir(parents=True, exist_ok=True)
MODELS = TMP / "models"
MODELS.mkdir(parents=True, exist_ok=True)
RESULTS = WORK / "raon_roundtrip.json"

CRISPASR_URL = "https://github.com/CrispStrobe/CrispASR.git"
CRISPASR_REF = os.environ.get("CRISPASR_REF", "feat/raon-opentts")
CLONE = TMP / "CrispASR"

GEN_TEXT = "The quick brown fox jumps over the lazy dog near the riverbank."


def sh(cmd, timeout=None, env=None, cwd=None):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout, env=env, cwd=cwd)


def step(name, **kv):
    print(f"[{time.strftime('%H:%M:%S')}] {name} " + json.dumps(kv), flush=True)


# Some Kaggle workers have flaky GitHub access (gotcha #18); clone the repo
# with retries, then init submodules separately (a failed submodule fetch must
# not abort the whole clone the way --recurse-submodules does).
if not CLONE.exists():
    for attempt in range(4):
        r = subprocess.run(["git", "clone", "--depth", "1", "--branch", CRISPASR_REF,
                            CRISPASR_URL, str(CLONE)], timeout=1800)
        if r.returncode == 0:
            break
        time.sleep(15)
    else:
        print("clone failed after retries", flush=True); sys.exit(1)
for attempt in range(4):
    r = subprocess.run(["git", "submodule", "update", "--init", "--recursive",
                        "ggml", "third_party/c2pa-audio"], cwd=str(CLONE), timeout=1800)
    if r.returncode == 0 or (CLONE / "ggml" / "CMakeLists.txt").exists():
        break
    time.sleep(15)
sys.path.insert(0, str(CLONE / "tools" / "kaggle"))
import kaggle_harness as kh  # noqa: E402

kh.init_progress()
HF_TOKEN = kh.resolve_hf_token()
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "huggingface_hub"], check=False)
from huggingface_hub import hf_hub_download  # noqa: E402

sh("nvidia-smi -L")
gpu = subprocess.run("nvidia-smi --query-gpu=name --format=csv,noheader", shell=True,
                     capture_output=True, text=True).stdout.strip()
step("gpu", gpu=gpu)

# ── build crispasr (GPU) ──────────────────────────────────────────────────
kh.install_build_toolchain()
arch = kh.detect_cuda_arch()
flags = (["-DCMAKE_BUILD_TYPE=Release", "-DCRISPASR_OPUS_FETCH=ON"]
         + kh.cuda_build_flags(arch) + kh.cache_and_link_flags())
r = sh(f"cd {CLONE} && cmake -G Ninja -B build " + " ".join(flags), timeout=1200)
if r.returncode != 0:
    step("cmake_FAILED", err=r.stderr[-2000:]); sys.exit(1)
with kh.build_heartbeat("build", interval_s=30):
    kh.sh_with_progress(f"cmake --build build -j{kh.safe_build_jobs(gpu=True)} --target crispasr-cli", cwd=str(CLONE))
CLI = CLONE / "build" / "bin" / "crispasr"
if not CLI.exists():
    cands = [c for c in (CLONE / "build").rglob("crispasr") if c.is_file() and os.access(c, os.X_OK)]
    CLI = cands[0] if cands else None
if not CLI:
    step("no_binary"); sys.exit(1)
os.environ["LD_LIBRARY_PATH"] = str(CLI.parent) + ":" + os.environ.get("LD_LIBRARY_PATH", "")
try:
    kh.export_ccache_tar()
except Exception:
    pass
step("built", cli=str(CLI))

# ── models: raon GGUF + whisper-tiny + a real English reference clip ───────
gguf = hf_hub_download("cstr/raon-opentts-0.3b-GGUF", "raon-opentts-0.3b-f16.gguf",
                       local_dir=str(MODELS), token=HF_TOKEN or None)
sh(f"bash {CLONE}/models/download-ggml-model.sh tiny.en {MODELS}", timeout=600)
whisper = MODELS / "ggml-tiny.en.bin"
# reference voice: the repo's jfk.wav (English), with its known transcript.
ref_wav = CLONE / "samples" / "jfk.wav"
ref_text = "And so my fellow Americans, ask not what your country can do for you, ask what you can do for your country."
step("models", gguf=os.path.basename(gguf), whisper=whisper.exists(), ref=ref_wav.exists())

# ── synth (GPU DiT + CPU HiFi-GAN) ─────────────────────────────────────────
out_wav = WORK / "raon_synth.wav"
synth_cmd = (f"{CLI} --backend raon -m {gguf} --voice {ref_wav} "
             f"--ref-text \"{ref_text}\" --tts \"{GEN_TEXT}\" --tts-output {out_wav} "
             f"-t 4 --seed 42 --i-have-rights -v")
t0 = time.time()
with kh.build_heartbeat("synth", interval_s=30):
    r = sh(synth_cmd, timeout=3600)
synth_s = round(time.time() - t0, 1)
print(r.stdout[-3000:], r.stderr[-3000:], flush=True)


def wav_info(p):
    try:
        with wave.open(str(p), "rb") as w:
            return w.getnframes(), w.getframerate()
    except Exception:
        return 0, 0


n_frames, sr = wav_info(out_wav)
step("synth", rc=r.returncode, wall_s=synth_s, wav_frames=n_frames, sr=sr,
     dur_s=round(n_frames / sr, 2) if sr else 0)
if r.returncode != 0 or n_frames < sr:  # <1 s of audio ⇒ failure, never a pass
    step("SYNTH_FAIL"); RESULTS.write_text(json.dumps({"pass": False, "stage": "synth"})); sys.exit(1)

# ── ASR roundtrip ──────────────────────────────────────────────────────────
r2 = sh(f"{CLI} -m {whisper} -nt --no-prints {out_wav}", timeout=600)
asr = " ".join(r2.stdout.split())
step("asr", text=asr[:200])


def norm(s):
    return {w.strip(".,!?;:\"'").lower() for w in s.split() if w.strip(".,!?;:\"'")}


gen_words = norm(GEN_TEXT)
asr_words = norm(asr)
overlap = round(len(gen_words & asr_words) / max(1, len(gen_words)), 3)
passed = overlap >= 0.7  # most content words present ⇒ intelligible, correct words

result = {"pass": bool(passed), "overlap": overlap, "gen_text": GEN_TEXT, "asr": asr,
          "synth_wall_s": synth_s, "dur_s": round(n_frames / sr, 2) if sr else 0, "gpu": gpu}
RESULTS.write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2), flush=True)

# upload the synth wav for local listening/inspection
try:
    from huggingface_hub import HfApi
    api = HfApi(token=HF_TOKEN)
    api.upload_file(path_or_fileobj=str(out_wav), repo_type="dataset",
                    repo_id="cstr/crispasr-regression-fixtures",
                    path_in_repo="raon-opentts/0.3B/raon_synth_ours.wav")
except Exception as e:
    step("upload_skip", err=str(e)[:200])

step("DONE", **{k: result[k] for k in ("pass", "overlap")})
if not passed:
    sys.exit(1)
