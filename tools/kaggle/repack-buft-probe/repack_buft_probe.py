#!/usr/bin/env python3
"""Kaggle kernel: does ggml's CPU repack extra buffer type exist, and does it pay?

Companion to docs/ggml-optimisation-playbook.md §4/§8 and to the local run on
the CrispASR VPS (Intel Xeon Skylake-SP: AVX-512F/DQ/CD/BW/VL, NO VNNI, NO AMX).

The point of running this on Kaggle is to move the one variable the VPS cannot:
the ISA. Kaggle workers are a different CPU generation and may have AVX-512
VNNI. So the FIRST thing this prints is the ISA it actually got — Kaggle does
not guarantee a consistent CPU between runs, and a number without its machine
is worthless.

CPU-only kernel. Kaggle CPU workers get NO internet (kaggle_usage.md gotcha #3),
so nothing is cloned: the ggml source and the probe .cpp arrive via
dataset_sources (chr1str/crispasr-repack-probe-src), which is the only delivery
route that survives no internet.
"""
import json
import os
import subprocess
import sys
import tarfile
import time
from pathlib import Path

WORK = Path("/kaggle/working")
OUT = WORK / "repack-probe-out"
OUT.mkdir(parents=True, exist_ok=True)


def sh(cmd, **kw):
    print(f"$ {cmd if isinstance(cmd, str) else ' '.join(cmd)}", flush=True)
    return subprocess.run(cmd, shell=isinstance(cmd, str), check=False,
                          capture_output=True, text=True, **kw)


def banner(t):
    print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78, flush=True)


# ---------------------------------------------------------------- machine ---
banner("MACHINE — report this with every number below")
cpuinfo = Path("/proc/cpuinfo").read_text()
model = ""
flags = ""
for line in cpuinfo.splitlines():
    if line.startswith("model name") and not model:
        model = line.split(":", 1)[1].strip()
    if line.startswith("flags") and not flags:
        flags = line.split(":", 1)[1].strip()
fl = set(flags.split())
ncpu = os.cpu_count()
print(f"model name : {model}")
print(f"nproc      : {ncpu}")
interesting = sorted(f for f in fl if f.startswith(("avx", "amx", "sse4", "fma", "f16c", "bmi")))
print(f"ISA flags  : {' '.join(interesting)}")
print()
for k in ["avx2", "avx512f", "avx512vl", "avx512bw", "avx512dq",
          "avx512_vnni", "avx_vnni", "amx_int8", "amx_tile", "amx_bf16"]:
    print(f"  {k:<12} = {int(k in fl)}")
machine = {"model": model, "nproc": ncpu,
           "has_avx512_vnni": int("avx512_vnni" in fl),
           "has_avx_vnni": int("avx_vnni" in fl),
           "has_amx_int8": int("amx_int8" in fl)}
try:
    print("\nmeminfo MemTotal:", [l for l in Path("/proc/meminfo").read_text().splitlines()
                                 if l.startswith("MemTotal")][0])
except Exception:
    pass

# ---------------------------------------------------------------- sources ---
banner("SOURCES — from dataset_sources (no internet on a Kaggle CPU worker)")
# `kaggle datasets create -r tar` makes Kaggle EXTRACT the archive at mount
# time, so the worker usually sees the tree directly and never a .tar.gz.
# Handle both, and print the mount tree when neither turns up.
ggml_dir = None
probe_cpp = None
for cand in Path("/kaggle/input").rglob("ggml/include/ggml.h"):
    ggml_dir = cand.parent.parent
    break
for cand in Path("/kaggle/input").rglob("crispasr_repack_probe.cpp"):
    probe_cpp = cand
    break

if ggml_dir is None:
    tar_path = next(iter(Path("/kaggle/input").rglob("repack-probe-src.tar.gz")), None)
    if tar_path is not None:
        print("extracting", tar_path)
        src = WORK / "src"
        src.mkdir(exist_ok=True)
        with tarfile.open(tar_path) as tf:
            tf.extractall(src)
        ggml_dir = src / "ggml"
        probe_cpp = src / "examples" / "cli" / "crispasr_repack_probe.cpp"

if ggml_dir is None or probe_cpp is None or not ggml_dir.exists() or not probe_cpp.exists():
    print("FATAL: could not locate ggml source and/or the probe .cpp under /kaggle/input.")
    for p in sorted(Path("/kaggle/input").glob("*/*/*"))[:80]:
        print("   ", p)
    sys.exit(1)

# The mount is read-only; CMake needs to read only, so an in-source path is
# fine, but copy anyway so nothing tries to write next to the sources.
if str(ggml_dir).startswith("/kaggle/input"):
    import shutil
    local = WORK / "src"
    local.mkdir(exist_ok=True)
    if not (local / "ggml").exists():
        shutil.copytree(ggml_dir, local / "ggml", symlinks=True, dirs_exist_ok=True)
    ggml_dir = local / "ggml"
    shutil.copy(probe_cpp, local / "crispasr_repack_probe.cpp")
    probe_cpp = local / "crispasr_repack_probe.cpp"

print("ggml :", ggml_dir)
print("probe:", probe_cpp)

# ------------------------------------------------------------------ build ---
banner("BUILD — ggml CPU backend with repack enabled, native ISA")
bld = WORK / "bld"
bld.mkdir(exist_ok=True)
t0 = time.time()
cfg = sh(["cmake", "-S", str(ggml_dir), "-B", str(bld),
          "-DCMAKE_BUILD_TYPE=Release",
          "-DGGML_NATIVE=ON",
          "-DGGML_CPU_REPACK=ON",
          "-DGGML_OPENMP=ON",
          "-DGGML_BUILD_TESTS=OFF",
          "-DGGML_BUILD_EXAMPLES=OFF"])
print(cfg.stdout[-4000:])
print(cfg.stderr[-4000:])
if cfg.returncode != 0:
    sys.exit("cmake configure failed")
b = sh(["cmake", "--build", str(bld), "--target", "ggml", "ggml-base", "ggml-cpu",
        "-j", str(ncpu)])
print(b.stdout[-6000:])
print(b.stderr[-6000:])
if b.returncode != 0:
    sys.exit("ggml build failed")
print(f"ggml built in {time.time()-t0:.0f}s")

libdir = None
for p in bld.rglob("libggml-cpu.so*"):
    libdir = p.parent
    break
print("libdir:", libdir)

# The probe reads ggml_tensor::extra, which sits immediately after
# name[GGML_MAX_NAME]. It MUST be compiled with the same GGML_MAX_NAME the
# library was built with or the read lands in the middle of the name and the
# probe silently reports "no repack traits" for everything. ggml's own default
# is 64 and this standalone ggml build does not override it, so no -D is needed
# here — unlike the in-tree CrispASR build, which uses 128.
maxname = 64
for line in (ggml_dir / "include" / "ggml.h").read_text().splitlines():
    if "define GGML_MAX_NAME" in line:
        maxname = int(line.split()[-1])
        break
print("GGML_MAX_NAME from header:", maxname)

exe = WORK / "repack_probe"
c = sh(["c++", "-O2", "-std=c++17", f"-DGGML_MAX_NAME={maxname}",
        "-I", str(ggml_dir / "include"), str(probe_cpp), "-o", str(exe),
        "-L", str(libdir), "-lggml", "-lggml-base", "-lggml-cpu",
        f"-Wl,-rpath,{libdir}"])
print(c.stdout[-3000:])
print(c.stderr[-3000:])
if c.returncode != 0:
    sys.exit("probe compile failed")

# ------------------------------------------------------------------- runs ---
env = dict(os.environ)
# CrispASR issue #453: pin MKL to one thread on every timing run.
env["MKL_NUM_THREADS"] = "1"
env["OPENBLAS_NUM_THREADS"] = "1"

results = {}
for threads in [1, min(4, ncpu)]:
    banner(f"RUN — threads={threads}, interleaved A/B, 25 reps per arm")
    r = subprocess.run([str(exe), "--threads", str(threads), "--reps", "25"],
                       capture_output=True, text=True, env=env)
    print(r.stdout)
    if r.stderr.strip():
        print("--- stderr (ggml debug) ---")
        print(r.stderr[-4000:])
    results[f"threads_{threads}"] = r.stdout
    (OUT / f"repack_probe_threads{threads}.txt").write_text(r.stdout + "\n" + r.stderr)

(OUT / "machine.json").write_text(json.dumps(machine, indent=2))
(OUT / "cpuinfo_flags.txt").write_text(flags)

banner("SUMMARY")
print(json.dumps(machine, indent=2))
print("\nRemember: a result here is a result for THIS machine. "
      "Do not generalise across ISAs.")
