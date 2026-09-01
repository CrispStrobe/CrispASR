#!/usr/bin/env python3
"""PR #414 / #413 — htdemucs GPU-by-default: FUSED parity + engagement proof.

The #398 kernel proved GPU per-layer-graph parity vs CPU/BLAS (cos
0.9997-1.0, |ratio| 1.000, f16 + q8_0). #414 flips the DEFAULT to the
FUSED graph on GPU hosts — the arm #398 never covered. This kernel, on a
real CUDA box, from a single build of the integr/pr-414 branch:

  A) cpu_ref f16   : CRISPASR_HTDEMUCS_GPU=0  -> gates line "graph=0",
                     legacy BLAS reference, 4 stems
  B) default_f16   : NO envs                  -> gates line
                     "graph=1 fused=1 gpu=1" (the new default engages by
                     itself), 4 stems, wall time
  C) cpu_ref_q8    : GPU=0, q8_0
  D) default_q8    : no envs, q8_0
  E) parity        : per-stem cosine + magnitude ratio + max|diff|,
                     B-vs-A and D-vs-C (cosine is scale-blind — HARD
                     RULE #2b — so magnitudes are checked too)

PASS bar mirrors #398: cos > 0.98 with magnitude ratio in (0.9, 1.1) per
stem; engagement asserted from the gates line, wall speedup reported (not
gated — box GPUs vary). Results: /kaggle/working/htdemucs414_results.json
"""

import array
import json
import math
import os
import re
import subprocess
import sys
import time
import wave
from pathlib import Path

WORK = Path("/kaggle/working")
TMP = Path("/kaggle/temp/i414"); TMP.mkdir(parents=True, exist_ok=True)
MODELS = Path("/kaggle/temp/models"); MODELS.mkdir(parents=True, exist_ok=True)
RESULTS = WORK / "htdemucs414_results.json"

HERE = Path(__file__).resolve().parent
CRISPASR_URL = "https://github.com/CrispStrobe/CrispASR.git"
BRANCH = "integr/pr-414"  # switch to "main" after the merge for re-runs
CLONE = Path("/kaggle/temp/CrispASR")
if not CLONE.exists():
    try:
        subprocess.run(["git", "clone", "--depth", "1", "--branch", BRANCH,
                        "--recurse-submodules", CRISPASR_URL, str(CLONE)],
                       check=True, timeout=1200)
    except Exception as e:
        print(f"clone failed: {e}", flush=True)
sys.path.insert(0, str(CLONE / "tools" / "kaggle") if CLONE.exists() else str(HERE))
import kaggle_harness as kh  # noqa: E402

kh.init_progress()

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "huggingface_hub"], check=False)
HF_TOKEN = kh.resolve_hf_token()
from huggingface_hub import hf_hub_download  # noqa: E402

HTD_REPO = "cstr/htdemucs-GGUF"
STEMS = ("drums", "bass", "other", "vocals")
GATES_RE = re.compile(r"htdemucs: gates graph=(\d) fused=(\d) gpu=(\d)")


def sh(cmd, timeout=None, env=None):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True,
                          timeout=timeout, env=env)


def build_crispasr():
    kh.install_build_toolchain()
    if not (CLONE / "ggml" / "CMakeLists.txt").exists():
        sh(f"cd {CLONE} && git submodule update --init ggml", timeout=900)
    arch = kh.detect_cuda_arch()
    flags = (["-DCMAKE_BUILD_TYPE=Release", "-DCRISPASR_OPUS_FETCH=ON"]
             + kh.cuda_build_flags(arch) + kh.cache_and_link_flags())
    kh.step("build.configure", arch=arch)
    r = sh(f"cd {CLONE} && cmake -G Ninja -B build " + " ".join(flags), timeout=1200)
    if r.returncode != 0:
        kh.step("build.configure_FAILED", stderr=r.stderr[-2000:])
        raise RuntimeError("cmake configure failed")
    jobs = kh.safe_build_jobs(gpu=True)
    with kh.build_heartbeat("build.ninja", interval_s=30):
        kh.sh_with_progress(f"cmake --build build -j{jobs} --target crispasr-cli", cwd=str(CLONE))
    binp = CLONE / "build" / "bin" / "crispasr"
    if not binp.exists():
        raise RuntimeError("crispasr binary not produced")
    os.environ["LD_LIBRARY_PATH"] = str(binp.parent) + ":" + os.environ.get("LD_LIBRARY_PATH", "")
    kh.step("build.ready", path=str(binp))
    try:
        kh.export_ccache_tar()
    except Exception:
        pass
    return str(binp)


def make_test_wav():
    """44.1 kHz stereo from samples/jfk.wav (mirrors the #398/#413 input shape)."""
    src = CLONE / "samples" / "jfk.wav"
    w = wave.open(str(src), "rb")
    sr = w.getframerate()
    a = array.array("h"); a.frombytes(w.readframes(w.getnframes())); w.close()
    x = [s / 32768.0 for s in a]
    tgt_sr = 44100
    n_out = int(len(x) * tgt_sr / sr)
    y = array.array("h")
    for i in range(n_out):
        t = i * (len(x) - 1) / max(1, n_out - 1)
        i0 = int(t); f = t - i0
        v = x[i0] * (1 - f) + x[min(i0 + 1, len(x) - 1)] * f
        s = max(-32768, min(32767, int(v * 32767)))
        y.append(s); y.append(s)
    p = TMP / "test-44k.wav"
    o = wave.open(str(p), "wb"); o.setnchannels(2); o.setsampwidth(2)
    o.setframerate(tgt_sr); o.writeframes(y.tobytes()); o.close()
    kh.step("input.ready", seconds=round(n_out / tgt_sr, 2))
    return p


def read_wav_f32(path):
    w = wave.open(str(path), "rb")
    a = array.array("h"); a.frombytes(w.readframes(w.getnframes())); w.close()
    return [s / 32768.0 for s in a]


def stem_paths(inp):
    stem_base = str(inp)[: -len(".wav")]
    return {s: Path(f"{stem_base}_{s}.wav") for s in STEMS}


def run_separate(binp, model, inp, tag, env_extra, timeout=3000):
    arm_inp = TMP / f"{tag}.wav"
    arm_inp.write_bytes(Path(inp).read_bytes())
    env = dict(os.environ); env.update(env_extra)
    cmd = f"{binp} --separate --backend htdemucs -m {model} -f {arm_inp}"
    t0 = time.time()
    with kh.build_heartbeat(f"sep.{tag}", interval_s=30):
        try:
            r = sh(cmd, timeout=timeout, env=env)
            rc, out = r.returncode, (r.stderr or "") + (r.stdout or "")
        except subprocess.TimeoutExpired:
            rc, out = -99, "TIMEOUT"
    wall = round(time.time() - t0, 2)
    stems = stem_paths(arm_inp)
    m = GATES_RE.search(out)
    res = dict(tag=tag, rc=rc, wall_s=wall,
               stems_written=sum(p.exists() for p in stems.values()),
               gates=(m.group(0) if m else None), tail=out[-400:])
    kh.step(f"sep.{tag}", rc=rc, wall=wall, stems=res["stems_written"], gates=res["gates"])
    return res, stems


def cosine(a, b):
    n = min(len(a), len(b))
    if n == 0:
        return 0.0, 0.0, 0.0
    dot = sum(a[i] * b[i] for i in range(n))
    na = math.sqrt(sum(v * v for v in a[:n])); nb = math.sqrt(sum(v * v for v in b[:n]))
    if na == 0 or nb == 0:
        return 0.0, na, nb
    return dot / (na * nb), na, nb


def parity(stems_new, stems_ref):
    out = {}
    for s in STEMS:
        if stems_new[s].exists() and stems_ref[s].exists():
            a = read_wav_f32(stems_new[s]); b = read_wav_f32(stems_ref[s])
            cos, na, nb = cosine(a, b)
            n = min(len(a), len(b))
            mx = max((abs(a[i] - b[i]) for i in range(n)), default=1.0)
            out[s] = dict(cos=round(cos, 6), mag_ratio=round(na / nb, 4) if nb else None,
                          max_abs_diff=round(mx, 6))
    return out


def parity_ok(p):
    return bool(p) and all(v["cos"] > 0.98 and v["mag_ratio"] and 0.9 < v["mag_ratio"] < 1.1
                           for v in p.values())


def main():
    binp = build_crispasr()
    for f in ("htdemucs-f16.gguf", "htdemucs-q8_0.gguf"):
        hf_hub_download(repo_id=HTD_REPO, filename=f, local_dir=str(MODELS), token=HF_TOKEN or None)
    f16 = MODELS / "htdemucs-f16.gguf"
    q8 = MODELS / "htdemucs-q8_0.gguf"
    inp = make_test_wav()

    results = {"branch": BRANCH, "runs": {}, "parity": {}}
    OFF = {"CRISPASR_HTDEMUCS_GPU": "0"}

    cpu16, stems_cpu16 = run_separate(binp, f16, inp, "cpu_ref_f16", OFF)
    results["runs"]["cpu_ref_f16"] = cpu16
    RESULTS.write_text(json.dumps(results, indent=2))

    def16, stems_def16 = run_separate(binp, f16, inp, "default_f16", {})
    results["runs"]["default_f16"] = def16
    RESULTS.write_text(json.dumps(results, indent=2))

    cpuq8, stems_cpuq8 = run_separate(binp, q8, inp, "cpu_ref_q8", OFF)
    results["runs"]["cpu_ref_q8"] = cpuq8
    RESULTS.write_text(json.dumps(results, indent=2))

    defq8, stems_defq8 = run_separate(binp, q8, inp, "default_q8", {})
    results["runs"]["default_q8"] = defq8

    results["parity"]["f16"] = parity(stems_def16, stems_cpu16)
    results["parity"]["q8_0"] = parity(stems_defq8, stems_cpuq8)

    verdict = dict(
        cpu_ref_stays_blas=bool(cpu16["gates"] and "graph=0" in cpu16["gates"]),
        default_engages_fused_gpu=bool(def16["gates"] and "graph=1 fused=1 gpu=1" in def16["gates"]),
        default_f16_ok=bool(def16["rc"] == 0 and def16["stems_written"] == 4),
        default_q8_ok=bool(defq8["rc"] == 0 and defq8["stems_written"] == 4),
        parity_f16_ok=parity_ok(results["parity"]["f16"]),
        parity_q8_ok=parity_ok(results["parity"]["q8_0"]),
        speedup_f16=round(cpu16["wall_s"] / def16["wall_s"], 2) if def16["wall_s"] else None,
        speedup_q8=round(cpuq8["wall_s"] / defq8["wall_s"], 2) if defq8["wall_s"] else None,
    )
    results["verdict"] = verdict
    RESULTS.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2), flush=True)
    kh.step("DONE", **verdict)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        kh.step("FATAL", err=str(e)[:300])
        traceback.print_exc()
        sys.exit(1)
