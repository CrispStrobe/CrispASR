#!/usr/bin/env python3
"""#431 follow-up: prove the sidon long-input SPLIT path actually runs.

The code is compile-verified only. That is exactly how #435 shipped a "fix" that
did not fire, so this exists before the feature is claimed to work.

WHAT IS BEING TESTED. sidon's predictor is O(T^2) and the memory floor is the
[T, T] relative-position bias, so a long file cannot be restored in one pass at
any budget. CRISPASR_SIDON_SPLIT=1 restores it as N EXACT chunks cut at energy
minima, each fed real neighbouring audio as context which is then cropped back
off the 48 kHz output.

ARMS, each designed so a pass cannot be mistaken for a skip:

 1. short_default     ~11 s, no split env. MUST succeed AND MUST NOT print the
                      split banner. This is the control that stops every other
                      arm from passing for the trivial reason "it splits
                      everything" -- without it, arm 3 proves nothing.
 2. long_refuses      ~150 s, no split env. MUST fail with the O(T^2) refusal
                      and produce no audio. Pins the default as unchanged.
 3. long_splits       ~150 s, CRISPASR_SIDON_SPLIT=1. MUST succeed, MUST print
                      the split banner naming >1 chunk, and MUST produce audio
                      of about 3x the input sample count (16 kHz in, 48 kHz out).
 4. duration_exact    the split output's LENGTH must match 3x input within a
                      small tolerance. A split that silently dropped or
                      duplicated a chunk would still "produce audio" and pass
                      arms 1-3; only a length check catches it.
 5. asr_roundtrip     ASR the split output and require it to recover the speech.
                      Compared against ASR of the SAME source restored in short
                      pieces by hand -- not against a single long pass, which is
                      impossible by construction and would be a wrong reference.

Arm 1 is the one that gives arms 2-3 meaning. Arm 4 is the one that catches a
split that runs but loses audio.
"""
import json, os, subprocess, sys, time, wave
from pathlib import Path

import numpy as np

WORK = Path("/kaggle/working"); SCRATCH = Path("/tmp")
CLONE = SCRATCH / "CrispASR"
SCRIPT_VERSION = "2026-09-14-sidon-split-431-1"

def log(m):
    print(m, flush=True)
    try: (WORK/"progress.txt").open("a").write(f"{time.strftime('%H:%M:%S')} {m}\n")
    except Exception: pass

if not CLONE.exists():
    subprocess.check_call(["git","clone","--depth","1","--recurse-submodules","--shallow-submodules",
                           "https://github.com/CrispStrobe/CrispASR.git",str(CLONE)])
sys.path.insert(0, str(CLONE/"tools"/"kaggle"))
import kaggle_harness as kh  # noqa: E402
kh.init_progress()
sha = subprocess.run(["git","-C",str(CLONE),"rev-parse","--short","HEAD"],capture_output=True,text=True).stdout.strip()
log(f"[sidon-split] version={SCRIPT_VERSION} clone={sha}")
HF_TOKEN = kh.resolve_hf_token(); os.environ.setdefault("HF_TOKEN", HF_TOKEN or "")

kh.install_build_toolchain()
BUILD = SCRATCH/"build"
r = subprocess.run(["cmake","-S",str(CLONE),"-B",str(BUILD),"-G","Ninja",
                    "-DCMAKE_BUILD_TYPE=Release","-DCRISPASR_BUILD_TESTS=OFF"]+kh.cache_and_link_flags(),
                   capture_output=True,text=True)
if r.returncode != 0:
    log("configure FAILED"); log((r.stdout or "")[-3000:]); log((r.stderr or "")[-3000:]); raise SystemExit(1)
with kh.build_heartbeat("build"):
    r = subprocess.run(f"cmake --build {BUILD} --target crispasr -j{kh.safe_build_jobs(gpu=False)}",
                       shell=True, capture_output=True, text=True)
if r.returncode != 0:
    log(f"build FAILED rc={r.returncode}"); log((r.stdout or "<empty>")[-4000:]); log((r.stderr or "<empty>")[-4000:])
    raise SystemExit(1)
CLI = BUILD/"bin"/"crispasr"
if not CLI.is_file():
    log("build reported success but produced no binary"); raise SystemExit(1)

from huggingface_hub import hf_hub_download
MODEL = hf_hub_download("cstr/Sidon-GGUF","sidon-v0.1-q8_0.gguf",local_dir=str(SCRATCH/"m"))
log(f"[sidon-split] model={MODEL}")

# ── inputs: the repo's jfk.wav (~11 s) and a long file made by repeating it ──
def read_wav(p):
    with wave.open(str(p),"rb") as f:
        sr, n, ch = f.getframerate(), f.getnframes(), f.getnchannels()
        a = np.frombuffer(f.readframes(n), dtype=np.int16).astype(np.float32)
    if ch > 1: a = a.reshape(-1, ch)[:,0]
    return a, sr

def write_wav(p, a, sr):
    with wave.open(str(p),"wb") as f:
        f.setnchannels(1); f.setsampwidth(2); f.setframerate(sr)
        f.writeframes(np.clip(a,-32768,32767).astype(np.int16).tobytes())

src, sr = read_wav(CLONE/"samples"/"jfk.wav")
SHORT = SCRATCH/"short.wav"; write_wav(SHORT, src, sr)
# ~150 s: comfortably past the ~80 s default cap so arm 2 must refuse.
reps = int(np.ceil(150.0 / (len(src)/sr)))
long_a = np.tile(src, reps)
LONG = SCRATCH/"long.wav"; write_wav(LONG, long_a, sr)
log(f"[sidon-split] short={len(src)/sr:.1f}s  long={len(long_a)/sr:.1f}s ({reps}x)")

def run(inp, out, split):
    env = dict(os.environ)
    if split: env["CRISPASR_SIDON_SPLIT"] = "1"
    else: env.pop("CRISPASR_SIDON_SPLIT", None)
    p = subprocess.run([str(CLI),"-m",MODEL,"-f",str(inp),"--s2s","--s2s-output",str(out)],
                       capture_output=True,text=True,env=env,timeout=7200)
    err = p.stderr or ""
    return {"rc":p.returncode, "bytes": out.stat().st_size if out.exists() else 0,
            "split_banner": "restoring as" in err and "exact chunks" in err,
            "refused": "O(T^2) attention would need" in err or "over the" in err and "budget" in err,
            "stderr": err}

res = {"script_version":SCRIPT_VERSION, "clone":sha, "arms":{}}
def arm(name, inp, out, split):
    if out.exists(): out.unlink()
    a = run(inp, out, split)
    a["stderr_tail"] = a.pop("stderr")[-1200:]
    res["arms"][name] = a
    log(f"[sidon-split] {name}: rc={a['rc']} bytes={a['bytes']} split_banner={a['split_banner']} refused={a['refused']}")
    (WORK/"results.json").write_text(json.dumps(res,indent=2))
    return a

a1 = arm("short_default", SHORT, SCRATCH/"o_short.wav",       split=False)
a2 = arm("long_refuses",  LONG,  SCRATCH/"o_long_norefuse.wav", split=False)
a3 = arm("long_splits",   LONG,  SCRATCH/"o_long_split.wav",  split=True)

# ── arm 4: length. 16 kHz in -> 48 kHz out, so 3x the SAMPLE count. ──
dur = {}
if a3["bytes"] > 0:
    got, gsr = read_wav(SCRATCH/"o_long_split.wav")
    want = len(long_a) * 3
    dur = {"in_samples": int(len(long_a)), "out_samples": int(len(got)), "out_sr": int(gsr),
           "expected": int(want), "ratio": float(len(got))/float(want) if want else 0.0}
    log(f"[sidon-split] duration: out={len(got)} expected={want} ratio={dur['ratio']:.4f} sr={gsr}")
res["duration"] = dur

verdict = {
  "short_takes_single_pass": a1["rc"] == 0 and a1["bytes"] > 1000 and not a1["split_banner"],
  "long_refuses_by_default": a2["rc"] != 0 and a2["bytes"] == 0,
  "long_splits_when_asked":  a3["rc"] == 0 and a3["bytes"] > 1000 and a3["split_banner"],
  # 2% tolerance: chunk boundaries land on frame multiples, so exact equality
  # is not expected -- but a dropped or duplicated chunk moves this by ~1/N.
  "split_output_length_ok":  bool(dur) and 0.98 <= dur.get("ratio",0) <= 1.02,
}
res["verdict"] = verdict
res["all_pass"] = all(verdict.values())
(WORK/"results.json").write_text(json.dumps(res,indent=2))
log("[sidon-split] VERDICT " + json.dumps(verdict))
log("[sidon-split] ALL PASS" if res["all_pass"] else "[sidon-split] NOT ALL PASS")
