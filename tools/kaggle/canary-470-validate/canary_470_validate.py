#!/usr/bin/env python3
"""Validate the rebased #470 branch (pr470-work) with the legacy-layout fix.

  unit        test-canary-layout
  live        test-canary-180m-live "[canary-180m]": 180M Q4 ASR + session auto-detect,
              Q5 EN->DE, 44 s four-repeat long-form + monotonic timings, legacy q4_k,
              legacy F16 (the file the first #470 loader rejected)
  regression  tests/regression/run_one.py canary-1b-v2 (pinned unquantised F16)
  parity      180M F32 CLI, three arms, exact text vs NeMo EncDecMultiTaskModel
              (NeMo's output recorded by chr1s4/crispasr-validate-472-470)
"""
import json, os, re, subprocess, sys, traceback
from pathlib import Path
OUT = Path("/kaggle/working/out"); OUT.mkdir(parents=True, exist_ok=True)
REPO = Path("/tmp/CrispASR"); G = Path("/tmp/g"); G.mkdir(exist_ok=True)
REF = os.environ.get("CRISPASR_REF", "pr470-work")
res = {"errors": []}
NEMO = {"en-en": "And so, my fellow Americans, ask not what your country can do for you. Ask what you can do for your country.",
        "en-en-nopnc": "and so my fellow americans ask not what your country can do for you ask what you can do for your country",
        "en-de": "Und so, meine Landsleute, fragen Sie nicht, was Ihr Land für Sie tun kann. Fragen Sie, was Sie für Ihr Land tun können."}
def save(): (OUT / "result.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
def norm(t): return " ".join(re.sub(r"^\[[^\]]*\]\s*", "", (t or "").strip(), flags=re.M).split())
try:
    subprocess.check_call(["git", "clone", "-b", REF, "https://github.com/CrispStrobe/CrispASR.git", str(REPO)])
    subprocess.run(["git", "submodule", "update", "--init", "--recursive", "--depth", "1"], cwd=str(REPO))
    res["commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(REPO), text=True).strip()
    sys.path.insert(0, str(REPO / "tools" / "kaggle")); import kaggle_harness as kh
    tok = kh.resolve_hf_token()
    if tok: os.environ["HF_TOKEN"] = tok
    import numpy as np, soundfile as sf
    from huggingface_hub import hf_hub_download
    kh.install_build_toolchain()
    kh.sh(f"cmake -S {REPO} -B {REPO}/build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCRISPASR_OPUS=OFF -DCRISPASR_AMR=OFF "
          "-DCRISPASR_BUILD_TESTS=ON " + " ".join(kh.cache_and_link_flags()))
    kh.sh(f"cmake --build {REPO}/build -j$(nproc) --target crispasr-cli crispasr-diff test-canary-layout test-canary-180m-live")
    B = REPO / "build" / "bin"
    r = subprocess.run([str(B / "test-canary-layout")], capture_output=True, text=True)
    res["unit"] = {"rc": r.returncode, "tail": r.stdout[-300:]}; save()

    fl = {q: hf_hub_download("handy-computer/canary-180m-flash-gguf", f"canary-180m-flash-{q}.gguf", cache_dir=str(G))
          for q in ("Q4_K_M", "Q5_K_M", "F32")}
    leg_q4 = hf_hub_download("cstr/canary-1b-v2-GGUF", "canary-1b-v2-q4_k.gguf", cache_dir=str(G))
    leg_f16 = hf_hub_download("cstr/canary-1b-v2-GGUF", "canary-1b-v2.gguf",
                              revision="b3715a517928f8f68833142c90fc5810ad583210", cache_dir=str(G))
    jfk = REPO / "samples" / "jfk.wav"
    a, sr = sf.read(jfk, dtype="float32")
    x4 = G / "jfk_x4.wav"; sf.write(x4, np.concatenate([a] * 4), sr, subtype="PCM_16")
    env = dict(os.environ, CRISPASR_MODEL_CANARY_180M=fl["Q4_K_M"], CRISPASR_MODEL_CANARY_180M_TRANSLATE=fl["Q5_K_M"],
               CRISPASR_AUDIO_CANARY_180M=str(jfk), CRISPASR_AUDIO_CANARY_180M_JFK_X4=str(x4),
               CRISPASR_MODEL_CANARY_LEGACY=leg_q4, CRISPASR_MODEL_CANARY_LEGACY_F16=leg_f16)
    r = subprocess.run([str(B / "test-canary-180m-live"), "[canary-180m]", "--success", "-r", "compact"],
                       capture_output=True, text=True, env=env, timeout=5400)
    res["live"] = {"rc": r.returncode, "tail": (r.stdout + r.stderr)[-3500:]}; save()

    env2 = dict(os.environ, CRISPASR_BIN=str(B / "crispasr"), DIFF_BIN=str(B / "crispasr-diff"), WORK_DIR=str(G / "reg"))
    r = subprocess.run([sys.executable, str(REPO / "tests/regression/run_one.py"), "canary-1b-v2"],
                       capture_output=True, text=True, env=env2, cwd=str(REPO))
    res["regression"] = {"rc": r.returncode, "tail": (r.stdout + r.stderr)[-1500:]}; save()

    res["parity"] = {}
    for key, extra in (("en-en", ["-sl", "en", "-tl", "en"]), ("en-en-nopnc", ["-sl", "en", "-tl", "en", "--no-punctuation"]),
                       ("en-de", ["-sl", "en", "-tl", "de"])):
        r = subprocess.run([str(B / "crispasr"), "--backend", "canary", "-m", fl["F32"], "-f", str(jfk), "-np"] + extra,
                           capture_output=True, text=True)
        out = norm(r.stdout)
        res["parity"][key] = {"rc": r.returncode, "out": out, "nemo": NEMO[key], "exact": out == NEMO[key]}
        save()
except BaseException:
    res["errors"].append(traceback.format_exc())
finally:
    save()
