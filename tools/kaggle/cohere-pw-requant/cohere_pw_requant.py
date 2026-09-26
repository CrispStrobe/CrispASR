#!/usr/bin/env python3
"""Requantize Cohere's published default Q4 GGUF after the hidden-F16 A/B."""

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

WORK = Path("/kaggle/working")
TEMP = Path("/kaggle/temp") if Path("/kaggle/temp").is_dir() else Path("/tmp")
REPO = TEMP / "CrispASR"
BUILD = TEMP / "build"
MODELS = TEMP / "models"
SHA = "5159c3c1"
HF_REPO = "cstr/cohere-transcribe-03-2026-GGUF"
NAME = "cohere-transcribe-q4_k.gguf"

subprocess.check_call(["git", "clone", "--filter=blob:none", "--no-checkout",
                       "https://github.com/CrispStrobe/CrispASR.git", str(REPO)])
subprocess.check_call(["git", "checkout", SHA], cwd=REPO)
subprocess.check_call(["git", "submodule", "update", "--init", "--recursive"], cwd=REPO)
sys.path.insert(0, str(REPO / "tools" / "kaggle"))
import kaggle_harness as kh  # noqa: E402

kh.init_progress(hf_progress_repo="cstr/crispasr-kaggle-progress")
step = kh.step
step("script.start", sha=SHA, run="v1")
token = kh.resolve_hf_token("HF_TOKEN")
subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet",
                       "huggingface_hub", "hf_transfer", "gguf"])
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
from huggingface_hub import HfApi, hf_hub_download  # noqa: E402
from gguf import GGUFReader  # noqa: E402

kh.install_build_toolchain()
ccache_run = TEMP / ".ccache"
warmed = WORK / ".ccache"
if warmed.exists():
    if ccache_run.exists():
        shutil.rmtree(ccache_run)
    shutil.move(str(warmed), str(ccache_run))
ccache_run.mkdir(parents=True, exist_ok=True)
os.environ["CCACHE_DIR"] = str(ccache_run)

BUILD.mkdir(parents=True, exist_ok=True)
flags = " ".join(kh.cache_and_link_flags())
subprocess.check_call(f"cmake -G Ninja -S {REPO} -B {BUILD} -DCMAKE_BUILD_TYPE=Release {flags}", shell=True)
with kh.build_heartbeat("cmake.build"):
    kh.sh_with_progress(f"stdbuf -oL -eL cmake --build {BUILD} --target crispasr-cli crispasr-quantize "
                        f"-j {kh.safe_build_jobs(False)}")
cli = BUILD / "bin" / "crispasr"
quant = BUILD / "bin" / "crispasr-quantize"
assert cli.is_file() and quant.is_file()
step("build.done")

MODELS.mkdir(parents=True, exist_ok=True)
old = Path(hf_hub_download(HF_REPO, NAME, token=token, local_dir=str(MODELS)))
new = MODELS / (NAME + ".new")
subprocess.check_call([str(quant), str(old), str(new), "q4_k"])


def inventory(path):
    return {t.name: (t.tensor_type.name, t.data.tobytes()) for t in GGUFReader(str(path)).tensors}


before, after = inventory(old), inventory(new)
assert before.keys() == after.keys()
changed = []
for name in before:
    if before[name] == after[name]:
        continue
    assert name.endswith(("conv.pw1.weight", "conv.pw2.weight")), name
    assert before[name][0] == "F16" and after[name][0] == "Q8_0", (name, before[name][0], after[name][0])
    changed.append(name)
assert len(changed) == 96, len(changed)
step("structure.done", changed=len(changed))


def transcribe(path):
    env = dict(os.environ, CRISPASR_SCHED_PROFILE="1")
    cmd = [str(cli), "--no-gpu", "--backend", "cohere", "-m", str(path),
           "-f", str(REPO / "samples" / "jfk.wav"), "-l", "en", "-t", "4"]
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, env=env, text=True, capture_output=True, timeout=900)
    if proc.returncode:
        raise RuntimeError(proc.stderr[-4000:])
    text = " ".join(line.strip() for line in proc.stdout.splitlines() if line.strip())
    enc_total = None
    for line in proc.stderr.splitlines():
        if "sched_profile[cohere]: total" in line and enc_total is None:
            enc_total = float(line.rsplit(" ", 2)[1])
    return text, time.perf_counter() - t0, enc_total


old_text, old_s, old_enc_ms = transcribe(old)
new_text, new_s, new_enc_ms = transcribe(new)
assert old_text == new_text
assert new_enc_ms is not None and old_enc_ms is not None and new_enc_ms < old_enc_ms
step("quality.done", same_text=True, old_enc_ms=old_enc_ms, new_enc_ms=new_enc_ms)

HfApi(token=token).upload_file(path_or_fileobj=str(new), repo_id=HF_REPO,
                               path_in_repo=NAME, repo_type="model",
                               commit_message="Requantize hidden Conformer pointwise matmuls to Q8_0")
summary = {"sha": SHA, "file": NAME, "changed": len(changed), "same_text": True,
           "old_process_s": old_s, "new_process_s": new_s,
           "old_encoder_profile_ms": old_enc_ms, "new_encoder_profile_ms": new_enc_ms,
           "uploaded": True, "passed": True}
(WORK / "summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2), flush=True)
step("upload.done")

# Protocol: the clone/build live outside WORK, leaving only two retrievable outputs.
subprocess.check_call(["tar", "cf", str(WORK / "ccache.tar"), "-C", str(TEMP), ".ccache"])
step("ccache.packed")
