#!/usr/bin/env python3
"""
Issue #416 — Sidon quantized GGUFs decode to a full-length file of SILENCE
while the f16 model restores the same clip correctly. PROOF kernel for the
CUDA arm, on a real CUDA box (Kaggle P100/T4).

The report (Disonantemus): `-m sidon-v0.1-f16.gguf --s2s` works;
`sidon-v0.1-q4_k.gguf` and `sidon-v0.1-q8_0.gguf` both write 1.80 s of pure
silence. rc=0, correct duration, correct sample rate, no error message.

Already eliminated before this kernel, so do not re-litigate them here:
  * x86-64 CPU — all three artifacts decode normally (f16 rms 0.128101,
    q8_0 0.127733, q4_k 0.126898 on the same clip). q8_0 is ~lossless, so the
    silence CANNOT be a precision effect; it is a code path only quantized
    weights take.
  * Vulkan — `distance_embedding` is the only quantized tensor feeding a
    broadcasting matmul (src0 [64,73,1,1] over H=16 heads), the classic
    CPU-tolerant/GPU-strict shape. Probed at sidon's exact shapes on lavapipe:
    F16 cos 0.999999, Q8_0/Q4_0 0.999985 vs CPU. test-backend-ops -o MUL_MAT
    passed 1536/1536. Hypothesis dead.

So this kernel asks exactly one question: does CUDA reproduce it?

Design notes that matter for trusting the answer:
  * EVERY model runs on BOTH backends on the SAME box, back to back. A quant
    that is silent on CUDA but fine under -ng on the same machine is a GPU-path
    fault; silent on both would mean the VPS CPU result was the outlier.
    Without the paired control a "silence" reading proves nothing.
  * rc=0 is NOT the verdict (HARD RULE #8 / #4a). Silence IS rc=0 here — that
    is the whole bug. The verdict comes from measuring the produced PCM:
    rms, peak, and the fraction of non-zero samples, read back out of the WAV.
  * main now warns when a stage handoff is degenerate and names WHICH stage
    (w2v-BERT predictor vs DAC decoder) and whether it is NaN or all-zero.
    Those lines are captured verbatim — if CUDA reproduces, they localize the
    fault to a stage without a second run.

Results -> /kaggle/working/sidon416_results.json
"""
import json
import os
import subprocess
import sys
import wave
from pathlib import Path

WORK = Path("/kaggle/working")
TMP = Path("/kaggle/temp/i416")
TMP.mkdir(parents=True, exist_ok=True)
MODELS = Path("/kaggle/temp/models")
MODELS.mkdir(parents=True, exist_ok=True)
RESULTS = WORK / "sidon416_results.json"

HERE = Path(__file__).resolve().parent
CRISPASR_URL = "https://github.com/CrispStrobe/CrispASR.git"
BRANCH = "main"
CLONE = Path("/kaggle/temp/CrispASR")
if not CLONE.exists():
    try:
        subprocess.run(["git", "clone", "--depth", "1", "--branch", BRANCH,
                        "--recurse-submodules", CRISPASR_URL, str(CLONE)],
                       check=True, timeout=1800)
    except Exception as e:  # fall through to the bundled harness copy
        print(f"clone failed: {e}", flush=True)
sys.path.insert(0, str(CLONE / "tools" / "kaggle") if CLONE.exists() else str(HERE))
import kaggle_harness as kh  # noqa: E402

kh.init_progress()

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "huggingface_hub"], check=False)
HF_TOKEN = kh.resolve_hf_token()
kh.step("hf.token", present=bool(HF_TOKEN))
from huggingface_hub import hf_hub_download  # noqa: E402

SIDON_REPO = "cstr/Sidon-GGUF"
# q6_k is included even though the reporter did not try it: if q8_0 and q4_k
# fail it says whether EVERY quant is affected or only some, which separates
# "quantized weights in general" from "one specific quant type's kernel".
LEGACY_FILE = "sidon-legacy-q8_0.gguf"  # rebuilt in-kernel; see the note in main()
SIDON_FILES = ["sidon-v0.1-f16.gguf", "sidon-v0.1-q8_0.gguf",
               "sidon-v0.1-q6_k.gguf", "sidon-v0.1-q4_k.gguf"]

CLIP = TMP / "in3s.wav"


def sh(cmd, timeout=None, env=None):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True,
                          timeout=timeout, env=env)


def build_crispasr():
    kh.install_build_toolchain()
    if not (CLONE / "CMakeLists.txt").exists():
        raise RuntimeError("repo clone missing")
    if not (CLONE / "ggml" / "CMakeLists.txt").exists():
        sh(f"cd {CLONE} && git submodule update --init ggml", timeout=1200)
    arch = kh.detect_cuda_arch()
    flags = (["-DCMAKE_BUILD_TYPE=Release", "-DCRISPASR_OPUS_FETCH=ON"]
             + kh.cuda_build_flags(arch) + kh.cache_and_link_flags())
    kh.step("build.configure", arch=arch)
    r = sh(f"cd {CLONE} && cmake -G Ninja -B build " + " ".join(flags), timeout=2400)
    if r.returncode != 0:
        kh.step("build.configure_FAILED", stderr=(r.stderr or "")[-2000:])
        raise RuntimeError("cmake configure failed")
    jobs = kh.safe_build_jobs(gpu=True)
    with kh.build_heartbeat("build.ninja", interval_s=30):
        kh.sh_with_progress(f"cmake --build build -j{jobs} --target crispasr-cli crispasr-quantize",
                            cwd=str(CLONE))
    binp = CLONE / "build" / "bin" / "crispasr"
    if not binp.exists():
        raise RuntimeError("crispasr binary not produced")
    kh.step("build.ready", path=str(binp))
    try:
        kh.export_ccache_tar()
    except Exception:
        pass
    return binp


def make_clip():
    """3 s of samples/jfk.wav at 16 kHz — near the reporter's 1.8 s, and short
    enough that 8 sidon runs fit the kernel budget (the DAC dominates)."""
    src = CLONE / "samples" / "jfk.wav"
    w = wave.open(str(src))
    n = int(3.0 * w.getframerate())
    data = w.readframes(n)
    o = wave.open(str(CLIP), "wb")
    o.setnchannels(w.getnchannels())
    o.setsampwidth(w.getsampwidth())
    o.setframerate(w.getframerate())
    o.writeframes(data)
    o.close()
    kh.step("clip.ready", seconds=3.0, rate=w.getframerate())


def measure(path):
    """Read the produced PCM back. This — not the exit code — is the verdict:
    the reported failure is a well-formed WAV whose samples are all zero."""
    if not Path(path).exists():
        return None
    w = wave.open(str(path))
    n = w.getnframes()
    raw = w.readframes(n)
    w.close()
    if n == 0:
        return dict(n=0, rms=0.0, peak=0.0, nonzero_frac=0.0)
    import array
    a = array.array("h")
    a.frombytes(raw)
    acc = 0.0
    peak = 0
    nz = 0
    for s in a:
        acc += float(s) * s
        av = -s if s < 0 else s
        if av > peak:
            peak = av
        if s:
            nz += 1
    return dict(n=len(a), rms=(acc / len(a)) ** 0.5 / 32768.0,
                peak=peak / 32768.0, nonzero_frac=nz / len(a))


def run_sidon(binp, model, tag, gpu, quant_rpe=False):
    out = TMP / f"out_{tag}.wav"
    if out.exists():
        out.unlink()
    ng = "" if gpu else "-ng "
    cmd = (f"{binp} {ng}-m {MODELS / model} -f {CLIP} --s2s "
           f"--s2s-output {out} -t 4")
    env = dict(os.environ)
    if quant_rpe:
        # Restore the PRE-FIX graph: keep the quantized weight in the
        # broadcasting RPE matmul. Without this the kernel would only be
        # testing the fix (main already contains it) and could never observe
        # the defect on CUDA.
        env["CRISPASR_SIDON_QUANT_RPE"] = "1"
    with kh.build_heartbeat(f"run.{tag}", interval_s=30):
        try:
            r = sh(cmd, timeout=2400, env=env)
            rc, err = r.returncode, (r.stderr or "")
        except subprocess.TimeoutExpired:
            rc, err = -99, "TIMEOUT"
    m = measure(out)
    # The #416 signature: rc=0, right length, no energy.
    silent = bool(m and m["n"] > 0 and m["peak"] == 0.0)
    warns = [ln.strip() for ln in err.splitlines() if "WARNING" in ln and "sidon" in ln]
    res = dict(tag=tag, model=model, gpu=gpu, quant_rpe=quant_rpe, rc=rc, measured=m,
               silent=silent, sidon_warnings=warns,
               tail=(err[-700:] if rc != 0 else ""))
    kh.step(f"run.{tag}", rc=rc, silent=silent,
            rms=(round(m["rms"], 6) if m else None),
            peak=(round(m["peak"], 6) if m else None),
            warns=len(warns))
    return res


SCRIPT_VERSION = "2026-09-03.2"  # bump when arms/capture/predicate change


def main():
    kh.provenance(SCRIPT_VERSION, clone_dir=CLONE)
    # Settle an open question at zero cost, since this prints before the sm_60
    # early-exit and therefore rides on any draw, good or bad: does
    # `kaggle kernels push` upload the WHOLE directory or only code_file?
    # ~/kaggle_usage.md says "everything in the push directory"; a peer inferred
    # "only code_file" from `kernels pull` returning a single .py — but pull is
    # selective by design (its -m flag GENERATES metadata rather than fetching
    # it), so that observation cannot settle it either way. This lists what the
    # script can actually see next to itself at runtime, which is the only thing
    # that decides whether the bundled kaggle_harness.py fallback can ever fire.
    try:
        kh.step("upload_probe", script_dir=str(HERE),
                siblings=sorted(p.name for p in HERE.iterdir()))
    except Exception as e:
        kh.step("upload_probe_failed", error=repr(e))
    smi = sh("nvidia-smi --query-gpu=name,driver_version --format=csv,noheader")
    gpu_name = (smi.stdout or "").strip()
    cc_out = sh("nvidia-smi --query-gpu=compute_cap --format=csv,noheader")
    cc_raw = (cc_out.stdout or "").strip().splitlines()[0] if cc_out.stdout else ""
    try:
        cc = float(cc_raw)
    except ValueError:
        cc = 0.0
    # ggml disables the integer MMQ path below compute capability 6.1:
    #   GGML_CUDA_CC_DP4A == 610, and ggml_cuda_should_use_mmq() returns false
    #   when highest_compiled_arch(cc) < that. __dp4a does not exist before 6.1.
    # Kaggle hands out P100 (sm_60, NO dp4a) or T4 (sm_75, dp4a) at random, so a
    # P100 run CANNOT exercise the path this kernel exists to test — it silently
    # measures dequant+cuBLAS on both arms and "proves" CUDA is fine.
    # This is the same capability-gap trap that made a lavapipe Vulkan sweep
    # falsely exonerate Vulkan for #416. Label the run instead of hiding it.
    mmq_reachable = cc >= 6.1
    kh.step("gpu", device=gpu_name, compute_cap=cc_raw, mmq_reachable=mmq_reachable)
    if not mmq_reachable:
        # Bail out BEFORE the ~20 min build. There is no working API-side T4
        # selector for script kernels: --accelerator nvidiaTeslaT4 was accepted
        # and ignored (3 runs), and "machine_shape": "GPU_T4_X2" in the metadata
        # was independently retested on another chr1s4 kernel and still landed
        # sm_60. So the accelerator is a lottery, and the only cheap play is to
        # lose it fast — a re-push then costs ~1 min instead of a whole run.
        RESULTS.write_text(json.dumps({
            "gpu": gpu_name, "compute_cap": cc_raw, "mmq_reachable": False,
            "verdict": {"conclusive": False,
                        "reason": "compute capability < 6.1: ggml disables MMQ "
                                  "(GGML_CUDA_CC_DP4A=610), so the pre-fix arm "
                                  "cannot exercise the path under test"},
        }, indent=2))
        print(f"P100_LOTTERY_RETRY: compute capability {cc_raw} < 6.1 — CUDA MMQ "
              f"is disabled on this GPU, so every arm would be VACUOUS. "
              f"Exiting early; re-push to redraw.", flush=True)
        return

    binp = build_crispasr()
    make_clip()

    for f in SIDON_FILES:
        with kh.build_heartbeat(f"models.{f}", interval_s=30):
            hf_hub_download(repo_id=SIDON_REPO, filename=f,
                            local_dir=str(MODELS), token=HF_TOKEN or None)
    kh.step("models.ready", files=SIDON_FILES)

    # THE PUBLISHED QUANTS CAN NO LONGER REPRODUCE THIS. cstr/Sidon-GGUF was
    # re-quantized and re-uploaded 2026-09-03, so distance_embedding now ships
    # as F16 — which makes CRISPASR_SIDON_QUANT_RPE=1 a NO-OP on those files:
    # the gate only bites when the tensor is actually quantized. Running the
    # arms on the HF q8_0 would yield a confident "CUDA is fine" from an arm
    # that never reached the code (and would fail validity gate V3 below,
    # identical arms). Rebuild the pre-fix file locally; --tensor-type
    # deliberately overrides the arch guards.
    legacy = MODELS / LEGACY_FILE
    qbin = CLONE / "build" / "bin" / "crispasr-quantize"
    if not qbin.exists():
        raise RuntimeError("crispasr-quantize not built — cannot rebuild the legacy quant")
    rq = sh(f"{qbin} {MODELS}/sidon-v0.1-f16.gguf {legacy} q8_0 "
            f"--tensor-type 'distance_embedding=q8_0'", timeout=1800)
    if rq.returncode != 0 or not legacy.exists():
        kh.step("legacy_quant_FAILED", stderr=(rq.stderr or "")[-1500:])
        raise RuntimeError("legacy quant rebuild failed")
    kh.step("legacy_quant.ready", path=str(legacy), bytes=legacy.stat().st_size)

    results = {"gpu": gpu_name, "compute_cap": cc_raw,
               "mmq_reachable": mmq_reachable, "branch": BRANCH, "runs": {}}

    # main already carries the fix (the RPE table is gathered to F32 before the
    # multiply), so the interesting arm is the PRE-FIX graph via
    # CRISPASR_SIDON_QUANT_RPE=1. Three arms per quant:
    #   cuda        — fixed graph on CUDA (must be healthy: the fix must not
    #                 regress the backend it was not written for)
    #   cudaq       — pre-fix quantized RPE matmul on CUDA (does CUDA's MMQ
    #                 have the same defect Vulkan does?)
    #   cpu         — control on the same box
    # f16 needs no cudaq arm: its table is not quantized, so the gate is a no-op.
    swept = SIDON_FILES + [LEGACY_FILE]
    for f in swept:
        short = f.replace("sidon-v0.1-", "").replace(".gguf", "")
        arms = [("cuda", True, False), ("cpu", False, False)]
        if "f16" not in f:
            arms.insert(1, ("cudaq", True, True))
        for suffix, gpu, qrpe in arms:
            tag = f"{short}_{suffix}"
            try:
                results["runs"][tag] = run_sidon(binp, f, tag, gpu, quant_rpe=qrpe)
            except Exception as e:
                results["runs"][tag] = dict(tag=tag, error=repr(e))
            RESULTS.write_text(json.dumps(results, indent=2))

    # Verdict. Reproduced == a quant is silent on CUDA while its own CPU arm on
    # this same box is not, with f16 healthy on CUDA (otherwise the fault is not
    # quantization-specific and the framing is wrong).
    def ok(tag):
        r = results["runs"].get(tag) or {}
        m = r.get("measured") or {}
        return bool(r.get("rc") == 0 and m.get("peak", 0) > 0.01)

    # Derive from what was actually SWEPT, not from SIDON_FILES. The published
    # quants can no longer reproduce (their distance_embedding is F16, so the
    # gate is a no-op), so a verdict computed over SIDON_FILES alone would read
    # "not reproduced" from three arms that never reached the code — while
    # silently excluding sidon-legacy-q8_0, the only model that CAN reproduce.
    quants = [f.replace("sidon-v0.1-", "").replace(".gguf", "")
              for f in swept if "f16" not in f]
    # The defect on CUDA: pre-fix arm silent while its own CPU control is fine.
    repro = [q for q in quants
             if not ok(f"{q}_cudaq") and ok(f"{q}_cpu") and ok("f16_cuda")]
    # The fix must leave CUDA healthy whether or not CUDA ever had the defect.
    fixed_ok = [q for q in quants if ok(f"{q}_cuda")]
    results["verdict"] = {
        "f16_cuda_ok": ok("f16_cuda"),
        "f16_cpu_ok": ok("f16_cpu"),
        "prefix_silent_on_cuda_only": repro,
        "cuda_defect_reproduced": bool(repro),
        # Without MMQ the pre-fix arm never differs from the fixed one, so a
        # "not reproduced" here says nothing about CUDA. Two independent tells:
        # the capability check, and gated/ungated arms measuring identically.
        "conclusive": bool(mmq_reachable),
        "gate_had_no_effect": [
            q for q in quants
            if (results["runs"].get(f"{q}_cuda", {}).get("measured") or {}).get("rms")
            == (results["runs"].get(f"{q}_cudaq", {}).get("measured") or {}).get("rms")
        ],
        "fixed_graph_healthy_on_cuda": sorted(fixed_ok),
        "fix_regresses_cuda": sorted(set(quants) - set(fixed_ok)),
    }
    RESULTS.write_text(json.dumps(results, indent=2))
    kh.step("verdict", **results["verdict"])
    print(json.dumps(results["verdict"], indent=2), flush=True)


if __name__ == "__main__":
    main()
