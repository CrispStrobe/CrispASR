#!/usr/bin/env python3
"""Hojo-ASR (#438) follow-up: close the two things the main run left open.

The main kernel established parity (all stages PASS at f16, byte-exact greedy
transcripts on both fixtures). It left two loose ends, and NEITHER needs the
12 GB checkpoint or a conversion -- the GGUFs are already on HF:

  1. docs/feature-matrix.md. The wiring audit failed the branch with
     `hojo-asr missing: feature-matrix(regen?)`. The matrix is generated from a
     live `crispasr --list-backends-json`, so it can only be regenerated where a
     binary exists -- and it must be a binary built from a branch rebased onto
     CURRENT main, or regeneration silently deletes backends merged since the
     branch was cut. (Measured: main's matrix has 125 backends, the pre-rebase
     branch build had 125 too -- 124 base + hojo -- so it would have dropped
     one.) This branch is now rebased, so the regenerated matrix should show 126.

  2. The one q4_k transcript divergence. german q4_k greedy ends 'meilen weit'
     where the matched greedy reference ends 'meilen breed'; f16 greedy matches
     the reference EXACTLY, and french matches on both precisions. The obvious
     reading is that 4-bit quantisation tipped a near-tie at the final token --
     but that is a hypothesis until someone measures the margin. So both
     precisions run with CRISPASR_HOJO_ASR_LOGIT_TRACE=1 and the per-step
     top1-top2 margins are diffed. If the margin at the divergent step is small
     next to the q4_k logit perturbation (max_abs 3.69 on this fixture), the
     hypothesis holds; if the margin is large, it is a defect and parity at f16
     was hiding it.

Push (chr1s4):
  export KAGGLE_API_TOKEN=<chr1s4 token>
  python -m kaggle kernels push -p tools/kaggle/hojo-asr-438-followup
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

SCRIPT_VERSION = "f1"
WORK = Path("/kaggle/working")
REPO = WORK / "CrispASR"
TEMP = Path("/kaggle/temp") if Path("/kaggle/temp").is_dir() else Path("/tmp")
BRANCH = "feat/438-hojo-asr"
GGUF_REPO = "cstr/Hojo-ASR-Multi-V1-GGUF"
DEMO_SPACE = "hugging-apps/hojo-asr-multi-v1-demo"

print(f"=== hojo-asr-438-followup {SCRIPT_VERSION} ===", flush=True)

if not REPO.exists():
    subprocess.check_call(["git", "clone", "--depth", "1", "--recursive", "--shallow-submodules",
                           "-b", BRANCH,
                           "https://github.com/CrispStrobe/CrispASR", str(REPO)])
subprocess.check_call(["git", "submodule", "update", "--init", "--recursive", "--depth", "1",
                       "ggml"], cwd=str(REPO))
subprocess.check_call(["git", "log", "--oneline", "-1"], cwd=str(REPO))
sys.path.insert(0, str(REPO / "tools" / "kaggle"))
import kaggle_harness as kh  # noqa: E402

kh.init_progress()


def sec(t):
    print(f"\n{'=' * 78}\n== {t}\n{'=' * 78}", flush=True)


kh.step("install deps")
kh.sh_with_progress("pip install -q huggingface_hub hf_transfer soundfile")
kh.step("resolve HF token")
hf_token = kh.resolve_hf_token()
if hf_token:
    os.environ["HF_TOKEN"] = hf_token
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
from huggingface_hub import hf_hub_download  # noqa: E402

sec("1. build the REBASED branch")
kh.install_build_toolchain()
kh.sh_with_progress(f"cmake -G Ninja -B {REPO}/build -S {REPO} -DCMAKE_BUILD_TYPE=Release "
                    + " ".join(kh.cache_and_link_flags()))
kh.sh_with_progress(f"cmake --build {REPO}/build -j{kh.safe_build_jobs(False)} --target crispasr-cli")
BIN = REPO / "build" / "bin"
if not (BIN / "crispasr").exists():
    raise SystemExit("build did not produce bin/crispasr (OUTPUT_NAME is 'crispasr', not 'crispasr-cli')")

sec("2. feature matrix, regenerated from a CURRENT binary")
before = (REPO / "docs" / "feature-matrix.md").read_text().splitlines()[5:6]
print(f"  committed matrix header : {before}")
subprocess.run([sys.executable, str(REPO / "tools" / "gen-feature-matrix.py")], cwd=str(REPO), check=True)
fm = REPO / "docs" / "feature-matrix.md"
after = fm.read_text().splitlines()[5:6]
print(f"  regenerated header      : {after}")
rows = [l for l in fm.read_text().splitlines() if "hojo" in l.lower()]
print(f"  hojo-asr row            : {rows if rows else 'ABSENT — regeneration failed'}")
# These are the backends merged into main while this branch was out; if the
# regenerated matrix has lost them, the build is stale and the file must NOT be
# committed (that is the silent-deletion failure this check exists to catch).
for must in ("breeze-tts-2", "voxtral", "moss-transcribe", "chatterbox"):
    present = any(f"`{must}" in l for l in fm.read_text().splitlines())
    print(f"  sanity: {must:16s} present in regenerated matrix: {present}")
shutil.copy(fm, WORK / "feature-matrix.md")
shutil.copy(REPO / "docs" / "feature-matrix.html", WORK / "feature-matrix.html")
print("  -> copied to /kaggle/working for commit back")

sec("3. wiring audit (expect the feature-matrix gap to be gone)")
aud = subprocess.run([sys.executable, str(REPO / "tools" / "check-backend-wiring.py"),
                      "--crispasr", str(BIN / "crispasr")], capture_output=True, text=True)
print(aud.stdout[-5000:])
print(f"  wiring audit rc={aud.returncode}  (0 = no required gaps)")

sec("4. logit-margin trace: is the q4_k divergence a near-tie?")
wav = TEMP / "german.wav"
p = hf_hub_download(repo_id=DEMO_SPACE, repo_type="space", filename="examples/german.wav",
                    cache_dir=str(TEMP / "demo"), token=hf_token)
import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
a, sr = sf.read(p, dtype="float32", always_2d=True)
mono = a.mean(axis=1)
pk = float(np.abs(mono).max()) if mono.size else 0.0
if pk > 1.0:
    mono = mono / pk
sf.write(wav, mono, 16000, subtype="PCM_16")
print(f"  audio: {wav} ({len(mono)/16000:.2f} s)")

traces = {}
for label, fname in (("f16", "hojo-asr-multi-v1-f16.gguf"), ("q4_k", "hojo-asr-multi-v1-q4_k.gguf")):
    g = hf_hub_download(repo_id=GGUF_REPO, filename=fname, cache_dir=str(TEMP / "gguf"), token=hf_token)
    env = dict(os.environ, CRISPASR_HOJO_ASR_LOGIT_TRACE="1")
    r = subprocess.run([str(BIN / "crispasr"), "-m", g, "--backend", "hojo-asr",
                        "-f", str(wav), "-bs", "1"], env=env, capture_output=True, text=True)
    steps = {}
    for line in r.stderr.splitlines():
        if "hojo_asr_trace:" in line:
            f = line.split()
            d = {k: v for k, v in (x.split("=", 1) for x in f if "=" in x)}
            steps[int(d["step"])] = line.strip()
    traces[label] = steps
    print(f"\n  [{label}] transcript: {r.stdout.strip()!r}")
    print(f"  [{label}] {len(steps)} decode steps traced")

sec("5. VERDICT on the divergence")
f16, q4 = traces.get("f16", {}), traces.get("q4_k", {})
if not f16 or not q4:
    print("  !! one or both traces missing — the trace env var did not fire; NOT a pass")
else:
    def top1(line):
        import re
        m = re.search(r"top1=(\d+)", line)
        return int(m.group(1)) if m else -1
    def margin(line):
        import re
        m = re.search(r"margin=([-\d.eE+]+)", line)
        return float(m.group(1)) if m else float("nan")
    common = sorted(set(f16) & set(q4))
    first_div = next((s for s in common if top1(f16[s]) != top1(q4[s])), None)
    print(f"  steps traced: f16={len(f16)} q4_k={len(q4)}; first differing step: {first_div}")
    if first_div is None:
        print("  the two precisions picked the SAME token at every traced step")
    else:
        print(f"    f16  : {f16[first_div]}")
        print(f"    q4_k : {q4[first_div]}")
        print(f"    f16 margin at that step  = {margin(f16[first_div]):.6f}")
        print(f"    q4_k margin at that step = {margin(q4[first_div]):.6f}")
        ms = sorted(margin(f16[s]) for s in common)
        med = ms[len(ms) // 2]
        print(f"    median f16 margin over all steps = {med:.6f}")
        print(f"    -> divergent-step margin is {margin(f16[first_div])/med:.3f}x the median")
        print("    A near-tie means the divergent-step margin is SMALL vs the median and")
        print("    vs the q4_k logit perturbation (max_abs 3.69 on this fixture).")
        after_div = [s for s in common if s > first_div and top1(f16[s]) != top1(q4[s])]
        print(f"    steps after the divergence that also differ: {len(after_div)} "
              f"(0 = single-token, no cascade)")
print("=== done ===", flush=True)
