#!/usr/bin/env python3
"""
Settle the ggml-cuda-fa-bisect result: does bd37ba0d ("cuda : fattn mma head
sizes 40 72", llama/30285) really slow T4 FlashAttention at head size 512/256?

The bisect jumped 1.146 -> 1.36 at that commit, but its culprit-vs-parent
confirm was only 1.065 and also moved shapes the commit does not touch (q8_0 KV
on the vector kernel, up to 1.44), so T4 clock drift is in those numbers.

  1. deterministic: registers / spills / shared memory of the FA MMA kernels
     for head sizes 256 and 512, parent vs culprit (cuobjdump -res-usage);
  2. drift-proof timing: parent and culprit built side by side and timed in
     alternation (A B A B ...), so a clock change hits both equally.

Results: /kaggle/working/fa_ab.json.
"""
import json, math, re, subprocess, sys, time
from pathlib import Path

WORK = Path("/kaggle/working")
TMP = Path("/kaggle/temp"); TMP.mkdir(parents=True, exist_ok=True)
RES = WORK / "fa_ab.json"
CULPRIT = "bd37ba0dfd4bbedc63d2178f844e9488a52d5b04"
ROUNDS = 5
FILTERS = {
    "hs512_decode_f16": r"hsk=512,hsv=512,nh=8,nr23=\[[48],1\],kv=(4096|8192|16384),nb=1,.*type_K=f16",
    "hs512_prefill_f16": r"hsk=512,hsv=512,.*nb=(512|4096),.*type_K=f16",
    "hs256_decode_f16": r"hsk=256,hsv=256,.*nb=1,.*type_K=f16",
    "hs256_decode_q8_0": r"hsk=256,hsv=256,.*nb=1,.*type_K=q8_0",  # vector kernel: control, untouched by the commit
    "hs128_decode_f16": r"hsk=128,hsv=128,.*nb=1,.*type_K=f16",   # control
}

CR = TMP / "CrispASR"
if not CR.exists():
    subprocess.run(["git", "clone", "--depth", "1", "https://github.com/CrispStrobe/CrispASR.git", str(CR)], check=True)
sys.path.insert(0, str(CR / "tools" / "kaggle"))
import kaggle_harness as kh
kh.init_progress()
kh.install_build_toolchain()
res = {"started": time.strftime("%F %T"), "culprit": CULPRIT, "rounds": ROUNDS}
save = lambda: RES.write_text(json.dumps(res, indent=1))


def sh(cmd, cwd=None, timeout=None):
    return subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True, text=True, timeout=timeout)


res["gpu"] = sh("nvidia-smi --query-gpu=name,driver_version,clocks.max.sm --format=csv,noheader").stdout.strip()
flags = ["-G Ninja", "-DCMAKE_BUILD_TYPE=Release", "-DGGML_NATIVE=ON", "-DGGML_BUILD_TESTS=ON",
         "-DGGML_BUILD_EXAMPLES=OFF"] + kh.cuda_build_flags() + kh.cache_and_link_flags()
JOBS = kh.safe_build_jobs(gpu=True)

bins = {}
for tag, rev in [("parent", CULPRIT + "^"), ("culprit", CULPRIT)]:
    src, bld = TMP / f"ggml_{tag}", TMP / f"b_{tag}"
    if not src.exists():
        subprocess.run(["git", "clone", "-q", "https://github.com/CrispStrobe/ggml.git", str(src)], check=True)
        sh(f"git -C {src} fetch -q origin crispstrobe-ops")
    sh(f"git -C {src} checkout -q -f {rev}")
    res[f"{tag}_rev"] = sh(f"git -C {src} rev-parse HEAD").stdout.strip()
    kh.step(f"build_{tag}")
    r = sh(f"cmake -S {src} -B {bld} " + " ".join(flags), timeout=1800)
    with kh.build_heartbeat(f"build.{tag}", interval_s=120):
        r = sh(f"cmake --build {bld} -j{JOBS} --target test-backend-ops", timeout=3 * 3600)
    if r.returncode:
        res[f"{tag}_build_error"] = (r.stdout + r.stderr)[-2000:]; save(); raise SystemExit(1)
    bins[tag] = sorted(bld.rglob("test-backend-ops"), key=lambda p: (not p.is_file(), len(str(p))))[0]
    # 1. resource usage of the FA MMA kernels for head sizes 256 / 512
    lib = next(iter(sorted(bld.rglob("libggml-cuda.so*"))), None) or next(iter(sorted(bld.rglob("libggml-cuda.a"))), None)
    usage = {}
    if lib:
        out = sh(f"cuobjdump -res-usage {lib} 2>&1 | c++filt", timeout=1800).stdout.splitlines()
        fn = None
        for l in out:
            m = re.match(r"\s*Function (.*):\s*$", l)
            if m:
                fn = m.group(1); continue
            if fn and "flash_attn_ext_f16" in fn and re.search(r"\b(256|512), ?(256|512)\b", fn) and "REG:" in l:
                usage[fn[:200]] = l.strip()
                fn = None
    res[f"{tag}_res_usage"] = usage
    save()

# 2. alternating timing
PERF = re.compile(r"^\s*FLASH_ATTN_EXT\((.*)\):\s+\d+ runs -\s+([\d.]+) us/run")


def perf(tbo, filt):
    out = {}
    r = sh(f"{tbo} perf -b CUDA0 -o FLASH_ATTN_EXT -p '{filt}'", timeout=3600)
    for l in r.stdout.splitlines():
        m = PERF.match(l)
        if m:
            out[m.group(1)] = float(m.group(2))
    return out


def gm(xs):
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else None


res["timing"] = {}
for name, filt in FILTERS.items():
    per_round = []
    for i in range(ROUNDS):
        kh.step(f"time_{name}_{i}")
        order = ("parent", "culprit") if i % 2 == 0 else ("culprit", "parent")
        t = {tag: perf(bins[tag], filt) for tag in order}
        common = [k for k in t["culprit"] if k in t["parent"]]
        per_round.append(gm([t["culprit"][k] / t["parent"][k] for k in common]))
    vals = [v for v in per_round if v]
    res["timing"][name] = {"per_round": [round(v, 3) for v in vals], "geomean": vals and round(gm(vals), 3),
                           "min": vals and round(min(vals), 3), "max": vals and round(max(vals), 3)}
    save()

res["finished"] = time.strftime("%F %T")
save()
kh.step("done")
print(json.dumps({k: v for k, v in res.items()}, indent=1)[:8000])
