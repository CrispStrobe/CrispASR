#!/usr/bin/env python3
"""CrispASR #466 perf A/B, GPU arm: runs tools/kaggle/nemotron3-diar-466-perf
from the repo on a GPU worker (CUDA build; attention x weights on the GPU)."""
import os, urllib.request

ref = os.environ.get("CRISPASR_REF", "main")
url = f"https://raw.githubusercontent.com/CrispStrobe/CrispASR/{ref}/tools/kaggle/nemotron3-diar-466-perf/nemotron3_diar_466_perf.py"
os.environ.setdefault("CRISPASR_REF", ref)
exec(compile(urllib.request.urlopen(url, timeout=60).read().decode(), "perf.py", "exec"), {"__name__": "__main__"})
