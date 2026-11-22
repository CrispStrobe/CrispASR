#!/usr/bin/env python3
"""
outetts repetition-penalty A/B (nightly regression-extended failure, runs
44732629120 / 44999859659: "...over the le the crit doll the dou b...").

A = main (penalty divided once per OCCURRENCE -> penalty^k)
B = fix/outetts-rep-penalty (once per distinct token over prompt + generated,
    as HF generate(repetition_penalty) / llama.cpp repeat_penalty upstream)

CPU only, as the nightly runs it. One build; B is an incremental rebuild.
Seeds 1234 (the nightly's) + 1..11. Roundtrip ASR: parakeet-tdt-0.6b-v3 q4_k,
the nightly's ASR. WAVs for both arms land in /kaggle/working.
"""
import os, sys, subprocess, json, re, shutil
from pathlib import Path

WORK = Path("/kaggle/working")
TMP = Path("/kaggle/temp/oute"); TMP.mkdir(parents=True, exist_ok=True)
MODELS = Path("/kaggle/temp/models"); MODELS.mkdir(parents=True, exist_ok=True)
RESULTS = WORK / "oute_ab.json"
URL = "https://github.com/CrispStrobe/CrispASR.git"
CLONE = Path("/kaggle/temp/CrispASR")
BRANCH_B = "fix/outetts-rep-penalty"
if not CLONE.exists():
    subprocess.run(["git", "clone", "--branch", "main", "--recurse-submodules",
                    URL, str(CLONE)], check=True, timeout=1500)
sys.path.insert(0, str(CLONE / "tools" / "kaggle"))
import kaggle_harness as kh
kh.init_progress()
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "huggingface_hub"], check=False)
HF_TOKEN = kh.resolve_hf_token()
from huggingface_hub import hf_hub_download

PHRASE = "The quick brown fox jumps over the lazy dog"
SEEDS = [1234] + list(range(1, 12))
OUTE_REPO, OUTE_REV = "cstr/outetts-0.3-1b-GGUF", "fdd6560ab48bb197ad4a2b92d2a1fd93ac115f05"


def sh(cmd, cwd=None, timeout=None):
    return subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True, text=True, timeout=timeout)


def build():
    flags = ["-DCMAKE_BUILD_TYPE=Release", "-DGGML_NATIVE=ON", "-DCRISPASR_OPUS_FETCH=ON"] + kh.cache_and_link_flags()
    if not (CLONE / "build" / "build.ninja").exists():
        r = sh("cmake -G Ninja -B build " + " ".join(flags), cwd=str(CLONE), timeout=1200)
        if r.returncode:
            kh.step("configure_FAILED", err=r.stderr[-1500:]); raise RuntimeError("configure")
    with kh.build_heartbeat("build", interval_s=30):
        kh.sh_with_progress(f"cmake --build build -j{kh.safe_build_jobs(gpu=False)} --target crispasr-cli", cwd=str(CLONE))
    return str(CLONE / "build" / "bin" / "crispasr")


def norm(s):
    return re.findall(r"[a-z']+", s.lower())


def wer(ref, hyp):
    r, h = norm(ref), norm(hyp)
    d = list(range(len(h) + 1))
    for i, rw in enumerate(r, 1):
        prev, d[0] = d[0], i
        for j, hw in enumerate(h, 1):
            cur = min(d[j] + 1, d[j - 1] + 1, prev + (rw != hw))
            prev, d[j] = d[j], cur
    return round(d[len(h)] / max(1, len(r)), 4)


def asr(binp, model, wav):
    r = sh(f"{binp} -m {model} -f {wav} --no-gpu -nt --no-prints", timeout=600)
    lines = [l.strip() for l in r.stdout.splitlines() if l.strip()]
    return lines[-1] if lines else ""


def sweep(binp, arm, oute, codec, parakeet, results):
    for seed in SEEDS:
        out = TMP / f"{arm}-{seed}.wav"
        cmd = (f"{binp} -m {oute} --codec-model {codec} --tts \"{PHRASE}\" --tts-output {out} "
               f"--seed {seed} --voice default --no-gpu --no-prints")
        with kh.build_heartbeat(f"tts.{arm}.{seed}", interval_s=60):
            r = sh(cmd, timeout=900)
        hyp = asr(binp, parakeet, out) if out.exists() else ""
        if out.exists():
            shutil.copy(out, WORK / out.name)
        w = wer(PHRASE, hyp)
        results[arm][str(seed)] = dict(rc=r.returncode, asr=hyp, wer=w)
        kh.step(f"{arm}.{seed}", wer=w, asr=hyp[:100])
        RESULTS.write_text(json.dumps(results, indent=2))


def main():
    kh.install_build_toolchain()
    oute = hf_hub_download(OUTE_REPO, "outetts-0.3-1b-q4_k.gguf", revision=OUTE_REV, local_dir=str(MODELS), token=HF_TOKEN or None)
    codec = hf_hub_download(OUTE_REPO, "wavtokenizer-decoder-f16.gguf", revision=OUTE_REV, local_dir=str(MODELS), token=HF_TOKEN or None)
    parakeet = hf_hub_download("cstr/parakeet-tdt-0.6b-v3-GGUF", "parakeet-tdt-0.6b-v3-q4_k.gguf", local_dir=str(MODELS), token=HF_TOKEN or None)
    results = {"A_main": {}, "B_fix": {}}
    binp = build()
    results["A_sha"] = sh("git rev-parse --short HEAD", cwd=str(CLONE)).stdout.strip()
    sweep(binp, "A_main", oute, codec, parakeet, results)
    sh(f"git fetch origin {BRANCH_B} && git checkout -q FETCH_HEAD && git submodule update --init ggml", cwd=str(CLONE), timeout=900)
    results["B_sha"] = sh("git rev-parse --short HEAD", cwd=str(CLONE)).stdout.strip()
    binp = build()
    sweep(binp, "B_fix", oute, codec, parakeet, results)
    for arm in ("A_main", "B_fix"):
        ws = [v["wer"] for v in results[arm].values()]
        results[arm + "_summary"] = dict(mean_wer=round(sum(ws) / len(ws), 4),
                                         pass_0_3=sum(w <= 0.3 for w in ws), n=len(ws))
    RESULTS.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2), flush=True)
    kh.step("DONE", A=results["A_main_summary"], B=results["B_fix_summary"])


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        kh.step("FATAL", err=str(e)[:300]); traceback.print_exc(); sys.exit(1)
