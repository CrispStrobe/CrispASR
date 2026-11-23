#!/usr/bin/env python3
"""
Bisect the T4 (Turing) FLASH_ATTN_EXT slowdown of the 2026-11 ggml sync.

ggml-sync-cuda-ab measured, old fork ef9e36f6 -> new 717f46e9, on a T4:
  hsk=hsv=512 decode (nb=1)  1.33-1.40x slower,  prefill 1.15-1.20x slower
  hsk=hsv=256                ~1.06x (prefill up to 1.16x)
  hsk 64/128                 neutral;  hsk 72  3.7x faster
gemma4-e2b runs FA with head_dim 512 on its full-attention layers, so this matters.

Method: standalone ggml (fork CrispStrobe/ggml), one build dir reused across
commits (ninja + ccache keep steps incremental), `git bisect` limited to
src/ggml-cuda. Each step times the hs512 decode cases (2 runs, min per case)
and compares with the ef9e36f6 baseline timed in the same session:
geomean ratio > BAD_AT => bad. Build failure => skip.
At the end the culprit and its parent get the full hs 256/512 FA perf.

Results: /kaggle/working/fa_bisect.json (rewritten after every step).
"""
import json, math, os, re, subprocess, sys, time
from pathlib import Path

WORK = Path("/kaggle/working")
TMP = Path("/kaggle/temp"); TMP.mkdir(parents=True, exist_ok=True)
RES = WORK / "fa_bisect.json"
GOOD, BAD = "ef9e36f69c3a79b713824f5a1e22b9836b4690c5", "717f46e9aa9480d3ac24bce6c98bbb9322483121"
BAD_AT = 1.18
SRC = TMP / "ggml"
BLD = TMP / "b"
FILTER_DECODE = r"hsk=512,hsv=512,nh=8,nr23=\[[48],1\],kv=(4096|8192|16384),nb=1,"
FILTER_FULL = r"hsk=(256|512),"

# harness from CrispASR main (progress, toolchain, ccache/mold)
CR = TMP / "CrispASR"
if not CR.exists():
    subprocess.run(["git", "clone", "--depth", "1", "https://github.com/CrispStrobe/CrispASR.git", str(CR)], check=True)
sys.path.insert(0, str(CR / "tools" / "kaggle"))
import kaggle_harness as kh
kh.init_progress()
kh.install_build_toolchain()

res = {"started": time.strftime("%F %T"), "good": GOOD, "bad": BAD, "bad_at": BAD_AT, "steps": []}
save = lambda: RES.write_text(json.dumps(res, indent=1))


def sh(cmd, cwd=None, timeout=None):
    return subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True, text=True, timeout=timeout)


if not SRC.exists():
    subprocess.run(["git", "clone", "-q", "https://github.com/CrispStrobe/ggml.git", str(SRC)], check=True)
    sh(f"git -C {SRC} fetch -q origin crispstrobe-ops")
res["gpu"] = sh("nvidia-smi --query-gpu=name,driver_version --format=csv,noheader").stdout.strip()
save()

flags = ["-G Ninja", "-DCMAKE_BUILD_TYPE=Release", "-DGGML_NATIVE=ON", "-DGGML_BUILD_TESTS=ON",
         "-DGGML_BUILD_EXAMPLES=OFF"] + kh.cuda_build_flags() + kh.cache_and_link_flags()
JOBS = kh.safe_build_jobs(gpu=True)


def build(rev):
    sh(f"git -C {SRC} checkout -q -f {rev}")
    r = sh(f"cmake -S {SRC} -B {BLD} " + " ".join(flags), timeout=1800)
    if r.returncode:
        return None, "configure: " + r.stderr[-800:]
    t0 = time.time()
    with kh.build_heartbeat(f"build.{rev[:8]}", interval_s=120):
        r = sh(f"cmake --build {BLD} -j{JOBS} --target test-backend-ops", timeout=3 * 3600)
    if r.returncode:
        return None, "build: " + (r.stdout + r.stderr)[-1500:]
    tbo = sorted(BLD.rglob("test-backend-ops"), key=lambda p: (not p.is_file(), len(str(p))))[0]
    return tbo, round(time.time() - t0)


PERF = re.compile(r"^\s*FLASH_ATTN_EXT\((.*)\):\s+\d+ runs -\s+([\d.]+) us/run")


def perf(tbo, filt, runs=2):
    best = {}
    for _ in range(runs):
        r = sh(f"{tbo} perf -b CUDA0 -o FLASH_ATTN_EXT -p '{filt}'", timeout=3600)
        for l in r.stdout.splitlines():
            m = PERF.match(l)
            if m:
                # newer test-backend-ops appends default-valued params; drop them so keys match
                k = re.sub(r",q_head_padding=0,mask_heads=1$", "", m.group(1))
                best[k] = min(float(m.group(2)), best.get(k, 1e30))
    return best


def ratio(cur, base):
    common = [k for k in cur if k in base]
    if not common:
        return None, 0
    return math.exp(sum(math.log(cur[k] / base[k]) for k in common) / len(common)), len(common)


# baseline
kh.step("baseline")
tbo, info = build(GOOD)
if tbo is None:
    res["baseline_error"] = info; save(); raise SystemExit(1)
base = perf(tbo, FILTER_DECODE, runs=3)
res["baseline"] = {"build_s": info, "n": len(base), "us": base}
save()

sh(f"git -C {SRC} bisect reset"); sh(f"git -C {SRC} checkout -q -f {BAD}")
out = sh(f"git -C {SRC} bisect start {BAD} {GOOD} -- src/ggml-cuda").stdout
res["bisect_start"] = out.strip(); save()

culprit = None
for step in range(40):
    rev = sh(f"git -C {SRC} rev-parse HEAD").stdout.strip()
    subj = sh(f"git -C {SRC} log -1 --format=%s {rev}").stdout.strip()
    kh.step(f"bisect_{step}", rev=rev[:10])
    tbo, info = build(rev)
    if tbo is None:
        verdict, r, n = "skip", None, 0
    else:
        r, n = ratio(perf(tbo, FILTER_DECODE), base)
        verdict = "skip" if r is None else ("bad" if r > BAD_AT else "good")
    res["steps"].append({"rev": rev, "subject": subj, "ratio": r and round(r, 3), "n": n,
                         "verdict": verdict, "build": info if isinstance(info, int) else str(info)[-600:]})
    save()
    out = sh(f"git -C {SRC} bisect {verdict}").stdout
    m = re.search(r"^([0-9a-f]{40}) is the first bad commit", out, re.M)
    if m:
        culprit = m.group(1); break
    if "only 'skip'ped commits left" in out or "There are only" in out:
        res["bisect_unresolved"] = out[-2000:]; break

res["bisect_log"] = sh(f"git -C {SRC} bisect log").stdout[-4000:]
sh(f"git -C {SRC} bisect reset")
if culprit:
    res["culprit"] = {"rev": culprit, "log": sh(f"git -C {SRC} show --stat {culprit}").stdout[:3000]}
    save()
    # full hs 256/512 picture, culprit vs its parent (first parent)
    kh.step("confirm")
    full = {}
    for tag, rv in [("parent", culprit + "^"), ("culprit", culprit)]:
        tbo, info = build(rv)
        full[tag] = perf(tbo, FILTER_FULL) if tbo else {}
    rows = []
    for k in full.get("culprit", {}):
        if k in full.get("parent", {}):
            rows.append((round(full["culprit"][k] / full["parent"][k], 3), k))
    rows.sort()
    g, n = ratio(full.get("culprit", {}), full.get("parent", {}))
    res["confirm"] = {"geomean_culprit_over_parent": g and round(g, 3), "n": n, "rows": rows}
res["finished"] = time.strftime("%F %T")
save()
kh.step("done")
print(json.dumps({k: v for k, v in res.items() if k not in ("baseline",)}, indent=1)[:6000])
