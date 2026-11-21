#!/usr/bin/env python3
"""VoxCPM2 quantisation ablation: which tensors make Q4_K babble?

On CPU, voxcpm2-q4_k + seed 2 adds words after the sentence ("... or another
rule"); F16 and the official bf16 package do not (heavy-cpu run 44871392490).
Every arm is built from the published F16 GGUF with crispasr-quantize
--tensor-type overrides and synthesizes the same sentence over 10 seeds; each
WAV is transcribed with whisper base.en. Reported per arm: exact transcripts,
stop steps, size. llama.cpp-omni's published recipe (only the base LM Q8_0,
all acoustic modules F16) is one of the arms.

  gh workflow run heavy-cpu.yml --ref <branch> -f script=tools/ci-heavy/voxcpm2_quant_ablation.py \
     -f pip="huggingface_hub" -f timeout_minutes=330
"""
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sweep_controls_common as C  # noqa: E402

TEXT = "Hello, this is a short test sentence."
# args: "seeds=A-B" and "arms=0,2,4" (indices into ARMS) select a subset.
_ARG = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
_lo, _hi = (int(x) for x in _ARG.get("seeds", "1-10").split("-"))
SEEDS = list(range(_lo, _hi + 1))
ACOUSTIC_F16 = [r"^(locenc|locdit|ralm|fsq|stop|proj)\.=f16"]
ARMS = {
    "f16": None,
    "q8_0": ("q8_0", []),
    "q4_k": ("q4_k", []),
    "q4_k+stop,fsq f16": ("q4_k", [r"^(stop|fsq)\.=f16"]),
    "q4_k+stop,fsq,ralm f16": ("q4_k", [r"^(stop|fsq|ralm)\.=f16"]),
    "q4_k tslm only": ("q4_k", ACOUSTIC_F16),
    "q8_0 tslm only (omni)": ("q8_0", ACOUSTIC_F16),
}
if "arms" in _ARG:
    _keep = {int(i) for i in _ARG["arms"].split(",")}
    ARMS = {k: v for i, (k, v) in enumerate(ARMS.items()) if i in _keep}


def norm(s):
    return re.sub(r"[^a-z ]", "", (s or "").lower()).split()


def main():
    bd = C.SCRATCH / "build"
    C.run(["cmake", "-S", C.ROOT, "-B", bd, "-G", "Ninja", "-DCMAKE_BUILD_TYPE=Release", "-DGGML_NATIVE=OFF",
           "-DBUILD_SHARED_LIBS=OFF"], C.OUT / "cmake.log")
    C.run(["cmake", "--build", bd, "-j", str(os.cpu_count() or 4), "--target", "crispasr", "crispasr-quantize"],
          C.OUT / "build.log")
    b, quant = bd / "bin" / "crispasr", bd / "bin" / "crispasr-quantize"
    wm = C.hf("ggerganov/whisper.cpp", "ggml-base.en.bin", None)
    f16 = C.hf("cstr/voxcpm2-GGUF", "voxcpm2-f16.gguf", None)
    res = {"text": TEXT, "seeds": SEEDS, "arms": {}}
    for i, (arm, spec) in enumerate(ARMS.items()):
        g = f16
        if spec:
            g = C.SCRATCH / f"arm{i}.gguf"
            cmd = [quant, f16, g, spec[0]]
            for rule in spec[1]:
                cmd += ["--tensor-type", rule]
            C.run(cmd, C.OUT / f"quant-arm{i}.log")
        rows = {}
        for s in SEEDS:
            wav = C.SCRATCH / f"a{i}-s{s}.wav"
            r = C.run([b, "--backend", "voxcpm2", "-m", g, "--tts", TEXT, "--tts-output", wav, "--seed", str(s),
                       "-t", str(os.cpu_count() or 4)], C.OUT / f"tts-a{i}-s{s}.log", check=False, timeout=1800)
            stop = re.findall(r"stopped at step (\d+)", r.stderr)
            heard = C.cli(b, ["-m", wm, "-f", wav, "-np", "-nt"], C.OUT / f"asr-a{i}-s{s}.log")["text"] \
                if wav.exists() else None
            rows[s] = {"stop": int(stop[-1]) if stop else None, "asr": heard, "exact": norm(heard) == norm(TEXT)}
            wav.unlink(missing_ok=True)
        res["arms"][arm] = {"size_gb": round(Path(g).stat().st_size / 1e9, 2), "rows": rows,
                            "exact": sum(v["exact"] for v in rows.values())}
        if g != f16:
            Path(g).unlink()
        C.save(res)
    lines = ["## VoxCPM2 quantisation ablation", "", f"text `{TEXT}`, seeds {SEEDS[0]}-{SEEDS[-1]}, CPU", "",
             "| arm | size GB | exact | stop steps | non-exact transcripts |", "|---|---|---|---|---|"]
    for arm, d in res["arms"].items():
        bad = "; ".join(f"s{s}: {v['asr']}" for s, v in d["rows"].items() if not v["exact"])
        lines.append(f"| {arm} | {d['size_gb']} | {d['exact']}/{len(SEEDS)} | "
                     f"{' '.join(str(v['stop']) for v in d['rows'].values())} | {bad} |")
    C.summary(lines)
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
