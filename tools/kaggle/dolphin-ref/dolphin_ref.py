#!/usr/bin/env python3
"""CrispASR #436 — Dolphin-CN-Dialect (small, streaming) reference dump.

Runs tools/dump_reference.py --backend dolphin (upstream DataoceanAI/Dolphin
package, dither=0) on the repo's Chinese sample and jfk.wav, verifying the
checkpoint sha256 first. Output: the -ref.gguf files + logs only.
"""
import hashlib, os, shutil, subprocess, sys, traceback
from pathlib import Path
WORK = Path("/kaggle/working"); OUT = WORK / "out"; OUT.mkdir(parents=True, exist_ok=True)
REPO = WORK / "CrispASR"
SHA = "bba8688ed33841b5f8b4578370be553f4557739c406dc7298f22985f7b061faf"
try:
    subprocess.check_call(["git", "clone", "--depth", "1", "-b", "feat/436-dolphin", "https://github.com/CrispStrobe/CrispASR.git", str(REPO)])
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "gguf",
                           "git+https://github.com/DataoceanAI/Dolphin.git@78ea615"])
    md = Path("/tmp/dolphin/small.cn.streaming"); md.mkdir(parents=True, exist_ok=True)
    base = "https://huggingface.co/DataoceanAI1/dolphin-cn-dialect-small-streaming/resolve/main/"
    for f in ("small.cn.streaming.pt", "train.yaml", "units.txt", "global_cmvn"):
        subprocess.check_call(["curl", "-sfL", "-o", str(md / f), base + f])
    h = hashlib.sha256(open(md / "small.cn.streaming.pt", "rb").read()).hexdigest()
    (OUT / "sha.txt").write_text(h + (" OK" if h == SHA else " MISMATCH"))
    if h != SHA:
        raise SystemExit("checkpoint sha256 mismatch")
    env = dict(os.environ, DOLPHIN_MODEL_NAME="small.cn.streaming")
    for name, wav in (("zh", "samples/paraformer_zh.wav"), ("jfk", "samples/jfk.wav")):
        r = subprocess.run([sys.executable, str(REPO / "tools/dump_reference.py"), "--backend", "dolphin",
                            "--model-dir", str(md), "--audio", str(REPO / wav),
                            "--output", str(OUT / f"dolphin-{name}-ref.gguf")], capture_output=True, text=True, env=env)
        (OUT / f"dump-{name}.log").write_text(r.stdout + "\n--- stderr ---\n" + r.stderr[-20000:])
except BaseException:
    (OUT / "error.txt").write_text(traceback.format_exc())
    raise
finally:
    shutil.rmtree(REPO, ignore_errors=True)
