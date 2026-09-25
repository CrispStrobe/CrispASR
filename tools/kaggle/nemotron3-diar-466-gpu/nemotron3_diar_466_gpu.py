#!/usr/bin/env python3
"""CrispASR #466 GPU arm: runs the canonical tools/kaggle/nemotron3-diar-466
script from the repo on a GPU worker, which switches it to a CUDA build and the
offline + low_latency modes, timing each GGUF on the GPU and q8_0 on the CPU."""
import os, subprocess, sys, urllib.request

ref = os.environ.get("CRISPASR_REF", "main")
url = f"https://raw.githubusercontent.com/CrispStrobe/CrispASR/{ref}/tools/kaggle/nemotron3-diar-466/nemotron3_diar_466.py"
code = urllib.request.urlopen(url, timeout=60).read().decode()
os.environ.setdefault("CRISPASR_REF", ref)
exec(compile(code, "nemotron3_diar_466.py", "exec"), {"__name__": "__main__"})
