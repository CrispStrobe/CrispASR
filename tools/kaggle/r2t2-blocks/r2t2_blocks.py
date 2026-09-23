#!/usr/bin/env python3
"""CrispASR #445 — Confucius4-R2T2 per-block encoder reference.

tools/dump_reference.py --backend qwen3 (float32, CPU) from the feature branch,
which now captures EVERY audio-encoder block under its real index, for
jfk.wav and for its first 0.32 s (the first streaming chunk). Output: the two
-ref.gguf files only; the repo clone is deleted so the output stays small.
"""
import os, shutil, subprocess, sys, traceback
from pathlib import Path

WORK = Path("/kaggle/working"); OUT = WORK / "out"; OUT.mkdir(parents=True, exist_ok=True)
REPO = WORK / "CrispASR"
BRANCH = "feat/445-r2t2"
MODEL, REV = "netease-youdao/Confucius4-R2T2", "185ce639118ad1362d049ca0d8ed04b6ec5cd6c9"
try:
    subprocess.check_call(["git", "clone", "--depth", "1", "-b", BRANCH, "https://github.com/CrispStrobe/CrispASR.git", str(REPO)])
    sys.path.insert(0, str(REPO / "tools" / "kaggle"))
    import kaggle_harness as kh
    kh.resolve_hf_token()
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "qwen-asr", "gguf"])
    from huggingface_hub import snapshot_download
    model_dir = snapshot_download(MODEL, revision=REV, local_dir="/tmp/r2t2")
    import soundfile as sf
    a, sr = sf.read(str(REPO / "samples" / "jfk.wav"), dtype="int16")
    sf.write(str(OUT / "jfk_032.wav"), a[:5120], 16000, subtype="PCM_16")
    env = dict(os.environ, QWEN3_REF_DTYPE="float32")
    for name, wav in (("jfk", REPO / "samples" / "jfk.wav"), ("jfk_032", OUT / "jfk_032.wav")):
        r = subprocess.run([sys.executable, str(REPO / "tools" / "dump_reference.py"), "--backend", "qwen3",
                            "--model-dir", model_dir, "--audio", str(wav), "--output", str(OUT / f"r2t2-{name}-blocks-ref.gguf")],
                           capture_output=True, text=True, env=env)
        (OUT / f"dump-{name}.log").write_text(r.stdout + "\n--- stderr ---\n" + r.stderr[-6000:])
except BaseException:
    (OUT / "error.txt").write_text(traceback.format_exc())
    raise
finally:
    shutil.rmtree(REPO, ignore_errors=True)
