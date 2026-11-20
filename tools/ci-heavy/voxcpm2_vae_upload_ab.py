#!/usr/bin/env python3
"""A/B for #461: VAE upsampler weights uploaded once, already permuted.

Builds this checkout and a baseline commit and synthesizes the reporter's
sentence twice with each (q4_k, seed 2, CPU). CPU WAV bytes are not
reproducible even run-to-run on one build (run 44853321277), so the gate is
the same AR stop step and the same whisper base.en transcript on every run.
Prints the vae.wn_init bench lines.

    gh workflow run heavy-cpu.yml -f script=tools/ci-heavy/voxcpm2_vae_upload_ab.py \
        -f args=<baseline-sha> -f pip=huggingface_hub
"""
import hashlib
import os
import re
import subprocess
import sys
from pathlib import Path

OUT = Path(os.environ.get("HEAVY_OUT", "out"))
SCR = Path(os.environ.get("HEAVY_SCRATCH", "scratch"))
OUT.mkdir(parents=True, exist_ok=True)
SCR.mkdir(parents=True, exist_ok=True)
REPO = Path(__file__).resolve().parents[2]
TEXT = "Hello, this is a short test sentence."
BASE = sys.argv[1]
NJ = str(os.cpu_count() or 4)


def run(cmd, **kw):
    print("$", " ".join(map(str, cmd)), flush=True)
    return subprocess.run([str(c) for c in cmd], **kw)


def build(src, name):
    b = SCR / name
    run(["cmake", "-S", src, "-B", b, "-DCMAKE_BUILD_TYPE=Release", "-DCRISPASR_OPUS=OFF", "-DCRISPASR_AMR=OFF"],
        check=True, stdout=subprocess.DEVNULL)
    run(["cmake", "--build", b, "--target", "crispasr-cli", f"-j{NJ}"], check=True, stdout=subprocess.DEVNULL)
    return b / "bin" / "crispasr"


from huggingface_hub import hf_hub_download  # noqa: E402

run(["git", "-C", REPO, "fetch", "-q", "--depth=1", "origin", BASE], check=True)
base_src = SCR / "base-src"
run(["git", "-C", REPO, "worktree", "add", "-q", "--detach", base_src, BASE], check=True)
run(["git", "-C", base_src, "submodule", "update", "-q", "--init", "--depth=1", "ggml", "third_party/c2pa-audio"], check=True)
clis = {"base": build(base_src, "b-base"), "new": build(REPO, "b-new")}
model = hf_hub_download("cstr/voxcpm2-GGUF", "voxcpm2-q4_k.gguf")
env = dict(os.environ, CRISPASR_VOXCPM2_BENCH="1")
wm = hf_hub_download("ggerganov/whisper.cpp", "ggml-base.en.bin")
norm = lambda s: re.sub(r"[^a-z ]", "", s.lower()).split()
sha, lines, exact = {}, [], {}
for arm, cli in clis.items():
    for rep in (1, 2):  # two runs per arm separate a real change from run-to-run variation
        tag = f"{arm}{rep}"
        wav = SCR / f"{tag}.wav"
        r = run([cli, "--backend", "voxcpm2", "-m", model, "--tts", TEXT, "--tts-output", wav, "--seed", "2",
                 "-t", NJ], capture_output=True, text=True, timeout=3600, env=env)
        assert r.returncode == 0, r.stderr[-2000:]
        sha[tag] = hashlib.sha256(wav.read_bytes()).hexdigest()
        lines += [f"{tag}: {l.strip()}" for l in r.stderr.splitlines()
                  if "vae." in l or "vae_wn" in l or "stopped at step" in l or "AR loop" in l]
        a = run([clis["base"], "-m", wm, "-f", wav, "-np", "-nt"], capture_output=True, text=True, timeout=600)
        exact[tag] = norm(a.stdout)
        lines.append(f"{tag}: wav sha256 {sha[tag]} asr={a.stdout.strip()!r} ")
steps = {l.split(": ")[0] + ":" + l.rsplit("step", 1)[1] for l in lines if "stopped at step" in l}
same = len({" ".join(v) for v in exact.values()}) == 1 and len({s.split(":", 1)[1] for s in steps}) == 1
lines += [f"same stop step and transcript on all runs: {same}"]
(OUT / "summary.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
sys.exit(0 if same else 1)
