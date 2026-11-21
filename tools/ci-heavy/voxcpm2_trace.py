#!/usr/bin/env python3
"""VoxCPM2 trailing-babble trace (crispasr-diff protocol).

On CPU, q4_k + seed 2 reads back "Hello, this is a short test sentence or
another rule." on main (heavy-cpu run 44853321277). This job answers:

1. Blueprint: does the official voxcpm package (openbmb/VoxCPM2, F32 and its
   default dtype) also add words after the sentence, over several seeds?
2. Stage diff: dump the reference archive for the same sentence and run
   crispasr-diff voxcpm2-tts on the F16 and Q4_K GGUFs. Where does cosine
   drop first, and is F16 clean (precision) or not (runtime)?
3. CLI: F16 vs Q4_K end-to-end over the same seeds, stop step + transcript.

All transcripts come from whisper base.en through the same crispasr build.

  gh workflow run heavy-cpu.yml --ref <branch> -f script=tools/ci-heavy/voxcpm2_trace.py \
     -f pip="huggingface_hub" -f timeout_minutes=300
"""
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sweep_controls_common as C  # noqa: E402

TEXT = "Hello, this is a short test sentence."
SEEDS = [2, 42, 7]
HF_REV = "7936080b41163874f38610f46f438e2c0757f9fd"
GGUF_REPO = "cstr/voxcpm2-GGUF"

BLUEPRINT = r'''
import json, sys, numpy as np, torch, soundfile as sf
from voxcpm.model.voxcpm2 import VoxCPM2Model
model_dir, text, out_dir, mode = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
seeds = [int(s) for s in sys.argv[5].split(",")]
m = VoxCPM2Model.from_local(model_dir, optimize=False, device="cpu")
if mode == "f32":
    m = m.float()
    m.config.dtype = "float32"
    m.base_lm.setup_cache(1, m.config.max_length, "cpu", torch.float32)
    m.residual_lm.setup_cache(1, m.config.max_length, "cpu", torch.float32)
m.eval()
sr = getattr(m, "sample_rate", None) or getattr(m.audio_vae, "sample_rate", 48000)
res = {"dtype": str(next(m.parameters()).dtype), "sr": sr}
for s in seeds:
    with torch.inference_mode():
        torch.manual_seed(s)
        wav = m.generate(target_text=text, max_len=200, inference_timesteps=10, cfg_value=2.0)
    a = wav.squeeze().float().numpy()
    p = f"{out_dir}/py-{mode}-s{s}.wav"
    sf.write(p, a, sr)
    res[str(s)] = {"wav": p, "seconds": round(len(a) / sr, 3)}
print("@@" + json.dumps(res))
'''


def asr(b, wm, wav):
    r = C.cli(b, ["-m", wm, "-f", wav, "-np", "-nt"], C.OUT / f"asr-{Path(wav).stem}.log")
    return r["text"]


def main():
    res = {"text": TEXT}
    bd = C.SCRATCH / "build"
    C.run(["cmake", "-S", C.ROOT, "-B", bd, "-G", "Ninja", "-DCMAKE_BUILD_TYPE=Release", "-DGGML_NATIVE=OFF",
           "-DBUILD_SHARED_LIBS=OFF"], C.OUT / "cmake.log")
    C.run(["cmake", "--build", bd, "-j", str(os.cpu_count() or 4), "--target", "crispasr", "crispasr-diff"],
          C.OUT / "build.log")
    b, diff = bd / "bin" / "crispasr", bd / "bin" / "crispasr-diff"
    wm = C.hf("ggerganov/whisper.cpp", "ggml-base.en.bin", None)
    ggufs = {q: C.hf(GGUF_REPO, f"voxcpm2-{q}.gguf", None) for q in ("f16", "q4_k")}

    # 3. C++ CLI arms (args "ref-only" skips them on a rerun)
    cpp = {}
    for q, g in ([] if "ref-only" in sys.argv[1:] else ggufs.items()):
        for s in SEEDS:
            wav = C.SCRATCH / f"cpp-{q}-s{s}.wav"
            r = C.run([b, "--backend", "voxcpm2", "-m", g, "--tts", TEXT, "--tts-output", wav, "--seed", str(s),
                       "-t", str(os.cpu_count() or 4)], C.OUT / f"cpp-{q}-s{s}.log", check=False, timeout=3600)
            stop = re.findall(r"stopped at step (\d+)", r.stderr)
            cpp[f"{q}-s{s}"] = {"rc": r.returncode, "stop_step": stop[-1] if stop else None,
                                "asr": asr(b, wm, wav) if wav.exists() else None}
            res["cpp"] = cpp
            C.save(res)

    # 1. Blueprint
    from huggingface_hub import snapshot_download
    md = snapshot_download("openbmb/VoxCPM2", revision=HF_REV, cache_dir=str(C.SCRATCH / "hf"))
    py = C.venv("voxcpm", ["voxcpm==2.0.3", "soundfile", "numpy", "gguf", "huggingface_hub"],
                ["--extra-index-url", "https://download.pytorch.org/whl/cpu", "--index-strategy", "unsafe-best-match"])
    (C.SCRATCH / "bp.py").write_text(BLUEPRINT)
    bp = {}
    for mode in (("f32",) if "ref-only" in sys.argv[1:] else ("f32", "native")):
        r = C.run([py, C.SCRATCH / "bp.py", md, TEXT, C.SCRATCH, mode, ",".join(map(str, SEEDS))],
                  C.OUT / f"blueprint-{mode}.log", check=False, timeout=7200)
        m = re.search(r"^@@(.*)$", r.stdout, re.M)
        if not m:
            bp[mode] = {"rc": r.returncode}
            continue
        d = json.loads(m.group(1))
        for s in SEEDS:
            d[str(s)]["asr"] = asr(b, wm, d[str(s)]["wav"])
        bp[mode] = d
        res["blueprint"] = bp
        C.save(res)

    # 2. Stage diff (reference archive in F32, seed 42)
    ref = C.SCRATCH / "voxcpm2-ref.gguf"
    env = dict(os.environ, VOXCPM2_SYN_TEXT=TEXT, VOXCPM2_SEED="42", VOXCPM2_F32="1", VOXCPM2_MAX_STEPS="200",
               PYTHONPATH=str(C.ROOT / "tools"))
    r = C.run([py, C.ROOT / "tools" / "dump_reference.py", "--backend", "voxcpm2-tts", "--model-dir", md,
               "--audio", C.ROOT / "samples" / "jfk.wav", "--output", ref], C.OUT / "dump.log", check=False,
              env=env, timeout=7200)
    res["dump_rc"] = r.returncode
    sd = {}
    if ref.exists():
        for q, g in ggufs.items():
            r = C.run([diff, "voxcpm2-tts", g, ref, C.ROOT / "samples" / "jfk.wav"], C.OUT / f"diff-{q}.log",
                      check=False, timeout=7200)
            sd[q] = {"rc": r.returncode,
                     "rows": [l for l in r.stdout.splitlines() + r.stderr.splitlines()
                              if re.search(r"cos|PASS|FAIL|max_abs", l)][:80]}
    res["stage_diff"] = sd
    C.save(res)

    lines = ["## VoxCPM2 trace", "", f"text: `{TEXT}`", "", "### C++ CLI (CPU)"]
    lines += [f"- {k}: stop {v['stop_step']} asr `{v['asr']}`" for k, v in cpp.items()]
    lines += ["", "### Python blueprint"]
    for mode, d in bp.items():
        lines += [f"- {mode} ({d.get('dtype')}) s{s}: {d[str(s)]['seconds']} s asr `{d[str(s)]['asr']}`"
                  for s in SEEDS if str(s) in d]
    lines += ["", "### crispasr-diff"]
    for q, d in sd.items():
        lines += [f"#### {q} rc={d['rc']}"] + [f"    {l}" for l in d["rows"]]
    C.summary(lines)
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
