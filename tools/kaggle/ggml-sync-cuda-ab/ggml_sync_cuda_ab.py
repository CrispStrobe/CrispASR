#!/usr/bin/env python3
"""
ggml sync 2026-11 (upstream ggml d9fe7bff + llama.cpp 1014775c), CUDA check on a T4.

CI only COMPILES the CUDA backend; this runs it.

  1. Correctness: full `test-backend-ops -b CUDA0` on the new ggml (CrispASR main's submodule).
  2. Speed A/B: `test-backend-ops perf` for MUL_MAT, MUL_MAT_ID, FLASH_ATTN_EXT,
     old ggml (fork ef9e36f6, pre-sync) vs new; runs alternate old/new/old/new and
     the faster of the two runs per case is kept.
  3. End-to-end: crispasr-cli on samples/jfk.wav, CUDA vs --no-gpu, same binary
     (parakeet-tdt-0.6b-v3 q4_k, whisper base.en).

Results: /kaggle/working/ggml_sync_cuda.json, logs alongside.
"""
import json, os, re, subprocess, sys, time
from pathlib import Path

WORK = Path("/kaggle/working")
TMP = Path("/kaggle/temp"); TMP.mkdir(parents=True, exist_ok=True)
MODELS = TMP / "models"; MODELS.mkdir(parents=True, exist_ok=True)
RESULTS = WORK / "ggml_sync_cuda.json"
CLONE = TMP / "CrispASR"
OLD_REV = "ef9e36f6"
if not CLONE.exists():
    subprocess.run(["git", "clone", "--branch", "main", "--recurse-submodules",
                    "https://github.com/CrispStrobe/CrispASR.git", str(CLONE)], check=True, timeout=1800)
sys.path.insert(0, str(CLONE / "tools" / "kaggle"))
import kaggle_harness as kh
kh.init_progress()
kh.install_build_toolchain()

res = {"started": time.strftime("%F %T")}


def save():
    RESULTS.write_text(json.dumps(res, indent=1))


def sh(cmd, cwd=None, timeout=None):
    return subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True, text=True, timeout=timeout)


res["crispasr_head"] = sh("git rev-parse HEAD", cwd=str(CLONE)).stdout.strip()
res["ggml_new"] = sh("git rev-parse HEAD", cwd=str(CLONE / "ggml")).stdout.strip()
res["gpu"] = sh("nvidia-smi --query-gpu=name,driver_version --format=csv,noheader").stdout.strip()
save()

JOBS = kh.safe_build_jobs(gpu=True)
CUDA = kh.cuda_build_flags()
CACHE = kh.cache_and_link_flags()


def build(src, bdir, targets, extra):
    flags = ["-G Ninja", "-DCMAKE_BUILD_TYPE=Release", "-DGGML_NATIVE=ON", "-DGGML_BUILD_TESTS=ON"] + CUDA + CACHE + extra
    r = sh(f"cmake -S {src} -B {bdir} " + " ".join(flags), timeout=1800)
    if r.returncode:
        kh.step(f"configure_FAILED_{Path(bdir).name}", err=r.stderr[-2000:])
        raise RuntimeError("configure " + bdir)
    t0 = time.time()
    with kh.build_heartbeat(f"build.{Path(bdir).name}", interval_s=60):
        kh.sh_with_progress(f"cmake --build {bdir} -j{JOBS} --target {' '.join(targets)}")
    res[f"build_s_{Path(bdir).name}"] = round(time.time() - t0)
    save()


# ── builds ───────────────────────────────────────────────────────────────────
NEW_B = TMP / "b-new"
build(str(CLONE), str(NEW_B), ["crispasr-cli", "test-backend-ops"], ["-DCRISPASR_OPUS_FETCH=ON"])
TBO_NEW = sorted(NEW_B.rglob("test-backend-ops"), key=lambda p: (not p.is_file(), len(str(p))))[0]
res["tbo_new_path"] = str(TBO_NEW)
CLI = NEW_B / "bin" / "crispasr"

OLD_SRC = TMP / "ggml-old"
if not OLD_SRC.exists():
    sh(f"git -C {CLONE / 'ggml'} fetch -q origin {OLD_REV} || git -C {CLONE / 'ggml'} fetch -q --unshallow origin")
    r = sh(f"git -C {CLONE / 'ggml'} worktree add {OLD_SRC} {OLD_REV}")
    res["old_worktree"] = (r.returncode, r.stderr[-500:])
OLD_B = TMP / "b-old"
build(str(OLD_SRC), str(OLD_B), ["test-backend-ops"], [])
TBO_OLD = sorted(OLD_B.rglob("test-backend-ops"), key=lambda p: (not p.is_file(), len(str(p))))[0]

# ── 1. correctness (new) ─────────────────────────────────────────────────────
kh.step("tbo_full_new")
log = WORK / "tbo_cuda_new.log"
t0 = time.time()
with kh.build_heartbeat("tbo.full", interval_s=120):
    r = sh(f"{TBO_NEW} -b CUDA0 > {log} 2>&1; echo rc=$?", timeout=5 * 3600)
txt = log.read_text(errors="replace")
res["correctness"] = {
    "rc": r.stdout.strip(),
    "seconds": round(time.time() - t0),
    "summary": re.findall(r"\d+/\d+ tests passed", txt),
    "failed": [l.strip() for l in txt.splitlines() if "FAIL" in l and "tests passed" not in l][:200],
    "backend_ok": "Backend CUDA0: \x1b[1;32mOK" in txt or "Backend CUDA0: OK" in txt,
}
save()

# ── 2. speed A/B ─────────────────────────────────────────────────────────────
PERF = re.compile(r"^\s*(\S.*?):\s+(\d+) runs -\s+([\d.]+) us/run")


def perf(binp, op, tag):
    out = WORK / f"perf_{tag}_{op}.log"
    sh(f"{binp} perf -b CUDA0 -o {op} > {out} 2>&1", timeout=3 * 3600)
    d = {}
    for l in out.read_text(errors="replace").splitlines():
        m = PERF.match(l)
        if m:
            d[m.group(1)] = float(m.group(3))
    return d


speed = {}
for op in ["MUL_MAT", "MUL_MAT_ID", "FLASH_ATTN_EXT"]:
    kh.step(f"perf_{op}")
    runs = {"old": [], "new": []}
    for i in range(2):
        runs["old"].append(perf(TBO_OLD, op, f"old{i}"))
        runs["new"].append(perf(TBO_NEW, op, f"new{i}"))
    best = {k: {} for k in runs}
    for k, lst in runs.items():
        for d in lst:
            for name, us in d.items():
                best[k][name] = min(us, best[k].get(name, us))
    common = sorted(set(best["old"]) & set(best["new"]))
    ratios = [(best["new"][n] / best["old"][n], n) for n in common if best["old"][n] > 0]
    ratios.sort()
    geo = (2 ** (sum(__import__("math").log2(r) for r, _ in ratios) / len(ratios))) if ratios else None
    speed[op] = {
        "n_common": len(common), "only_old": len(set(best["old"]) - set(best["new"])),
        "only_new": len(set(best["new"]) - set(best["old"])),
        "geomean_new_over_old": geo,
        "slower_10pct": [(round(r, 3), n, best["old"][n], best["new"][n]) for r, n in ratios if r > 1.10][-60:],
        "faster_10pct": [(round(r, 3), n, best["old"][n], best["new"][n]) for r, n in ratios if r < 0.90][:60],
    }
    res["speed"] = speed
    save()

# ── 3. end-to-end ASR, CUDA vs CPU ───────────────────────────────────────────
kh.step("e2e")
sys_ok = sh(f"{sys.executable} -m pip install -q huggingface_hub")
from huggingface_hub import hf_hub_download
HF = kh.resolve_hf_token() or None
models = {
    "parakeet": hf_hub_download("cstr/parakeet-tdt-0.6b-v3-GGUF", "parakeet-tdt-0.6b-v3-q4_k.gguf", local_dir=str(MODELS), token=HF),
    "whisper": hf_hub_download("ggerganov/whisper.cpp", "ggml-base.en.bin", local_dir=str(MODELS), token=HF),
}
wav = CLONE / "samples" / "jfk.wav"
e2e = {}
for name, m in models.items():
    for dev, flag in [("cuda", ""), ("cpu", "--no-gpu")]:
        t0 = time.time()
        r = sh(f"{CLI} -m {m} -f {wav} {flag} -nt", timeout=900)
        e2e[f"{name}_{dev}"] = {"rc": r.returncode, "s": round(time.time() - t0, 2),
                                "text": " ".join(l.strip() for l in r.stdout.splitlines() if l.strip()),
                                "backend_lines": [l for l in r.stderr.splitlines() if "CUDA" in l][:4],
                                "err_tail": r.stderr[-600:] if r.returncode else ""}
    e2e[f"{name}_same_text"] = e2e[f"{name}_cuda"]["text"] == e2e[f"{name}_cpu"]["text"]
res["e2e"] = e2e
res["finished"] = time.strftime("%F %T")
save()
kh.step("done")
print(json.dumps({k: v for k, v in res.items() if k != "speed"}, indent=1)[:4000])
for op, s in speed.items():
    print(op, s["n_common"], "geomean new/old", s["geomean_new_over_old"])
