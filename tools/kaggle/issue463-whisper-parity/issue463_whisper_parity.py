#!/usr/bin/env python3
"""Issue #463: prove unified Whisper uses Whisper's native long-audio seek loop."""

import json
import os
import shutil
import subprocess
import sys
import time
import wave
from pathlib import Path

WORK = Path("/kaggle/working")
TEMP = Path("/kaggle/temp") if Path("/kaggle/temp").is_dir() else Path("/tmp")
BASE = TEMP / "base"
FIX = TEMP / "fix"
BASE_BUILD = TEMP / "build-base"
FIX_BUILD = TEMP / "build-fix"
MODEL_DIR = TEMP / "models"
BASE_SHA = "d1b0684a"
FIX_BRANCH = "fix/463-whisper-internal-chunking"
SCRIPT_VERSION = "v1"


def run(cmd, **kwargs):
    print("+", " ".join(map(str, cmd)), flush=True)
    return subprocess.run(cmd, check=True, text=True, **kwargs)


run(["git", "clone", "--depth", "1", "--branch", "main", "--recurse-submodules",
     "https://github.com/CrispStrobe/CrispASR.git", str(BASE)])
run(["git", "clone", "--depth", "1", "--branch", FIX_BRANCH, "--recurse-submodules",
     "https://github.com/CrispStrobe/CrispASR.git", str(FIX)])
assert run(["git", "rev-parse", "--short", "HEAD"], cwd=BASE, capture_output=True).stdout.strip() == BASE_SHA
fix_sha = run(["git", "rev-parse", "--short", "HEAD"], cwd=FIX, capture_output=True).stdout.strip()

sys.path.insert(0, str(FIX / "tools" / "kaggle"))
import kaggle_harness as kh  # noqa: E402

kh.init_progress(hf_progress_repo="cstr/crispasr-kaggle-progress")
step = kh.step
step("script.start", script=SCRIPT_VERSION, base=BASE_SHA, fix=fix_sha)
kh.install_build_toolchain()

ccache_run = TEMP / ".ccache"
warmed = WORK / ".ccache"
if warmed.exists():
    if ccache_run.exists():
        shutil.rmtree(ccache_run)
    shutil.move(str(warmed), str(ccache_run))
ccache_run.mkdir(parents=True, exist_ok=True)
os.environ["CCACHE_DIR"] = str(ccache_run)
flags = kh.cache_and_link_flags()


def build(src, out):
    run(["cmake", "-G", "Ninja", "-S", str(src), "-B", str(out),
         "-DCMAKE_BUILD_TYPE=Release", *flags])
    with kh.build_heartbeat(f"build.{src.name}"):
        kh.sh_with_progress(
            f"stdbuf -oL -eL cmake --build {out} --target crispasr-cli -j {kh.safe_build_jobs(False)}"
        )
    cli = out / "bin" / "crispasr"
    assert cli.is_file()
    return cli


base_cli = build(BASE, BASE_BUILD)
fix_cli = build(FIX, FIX_BUILD)
step("build.done")

run([sys.executable, "-m", "pip", "install", "--quiet", "huggingface_hub"])
from huggingface_hub import hf_hub_download  # noqa: E402

MODEL_DIR.mkdir(parents=True, exist_ok=True)
model = Path(hf_hub_download("ggerganov/whisper.cpp", "ggml-tiny.en.bin", local_dir=str(MODEL_DIR)))

# Four copies are 44 seconds: long enough to trigger the old 30-second generic
# dispatcher chunker, while Whisper's own seek loop remains deterministic.
src_wav = BASE / "samples" / "jfk.wav"
long_wav = TEMP / "jfk-x4.wav"
with wave.open(str(src_wav), "rb") as r:
    params = r.getparams()
    pcm = r.readframes(r.getnframes())
with wave.open(str(long_wav), "wb") as w:
    w.setparams(params)
    w.writeframes(pcm * 4)


def transcribe(cli, unified):
    cmd = [str(cli), "-m", str(model), "-f", str(long_wav), "-l", "en", "-t", "4", "-nt"]
    if unified:
        cmd[1:1] = ["--backend", "whisper"]
    t0 = time.perf_counter()
    p = subprocess.run(cmd, text=True, capture_output=True, timeout=900)
    if p.returncode:
        raise RuntimeError(p.stderr[-5000:])
    text = " ".join(line.strip() for line in p.stdout.splitlines() if line.strip())
    return {"text": text, "seconds": time.perf_counter() - t0,
            "generic_slices": "processing " in p.stderr and " slice(s)" in p.stderr,
            "stderr_tail": p.stderr[-2000:]}


result = {
    "script": SCRIPT_VERSION,
    "base_sha": BASE_SHA,
    "fix_sha": fix_sha,
    "base_legacy": transcribe(base_cli, False),
    "base_unified": transcribe(base_cli, True),
    "fix_legacy": transcribe(fix_cli, False),
    "fix_unified": transcribe(fix_cli, True),
}
result["base_mismatch"] = result["base_legacy"]["text"] != result["base_unified"]["text"]
result["fix_match"] = result["fix_legacy"]["text"] == result["fix_unified"]["text"]
result["passed"] = (result["base_mismatch"] and result["base_unified"]["generic_slices"] and
                    result["fix_match"] and not result["fix_unified"]["generic_slices"])
(WORK / "summary.json").write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2), flush=True)
assert result["passed"], result
step("proof.done", base_mismatch=True, fix_match=True)

run(["tar", "cf", str(WORK / "ccache.tar"), "-C", str(TEMP), ".ccache"])
step("ccache.packed")
