#!/usr/bin/env python3
"""Were existing crispasr-diff fixtures corrupted by capture aliasing?

For each dumper behind a fixture on cstr/crispasr-regression-fixtures that the
static triage flagged (a .numpy() view of a tensor later passed on, or a hook
storing a tensor reference): run tools/dump_reference.py three times on the
same inputs, in its own venv (system site-packages + that model's package):
  mainA, mainB  — main's tools/ (view captures): mainA vs mainB is the
                  determinism control;
  branch        — fix/dumper-aliasing (owned captures + _safe_capture report).
A stage where mainA == mainB but mainA != branch was corrupted by aliasing in
the old dumper. The branch run's [ALIASING] lines name the capture sites.
"""
import json, os, shutil, subprocess, sys, traceback
from pathlib import Path
WORK = Path("/kaggle/working"); OUT = WORK / "out"; OUT.mkdir(parents=True, exist_ok=True)
REPO = WORK / "CrispASR"; MAIN = Path("/tmp/main_tools"); T = Path("/tmp/audit")
res = {"backends": {}, "errors": []}
def save(): (OUT / "audit.json").write_text(json.dumps(res, indent=1, ensure_ascii=False, default=str))

def load(p):
    import gguf, numpy as np
    r = gguf.GGUFReader(str(p))
    return {t.name: np.array(t.data, dtype=np.float64).reshape(-1) for t in r.tensors}

def compare(a, b):
    import numpy as np
    out = {}
    for k in sorted(set(a) | set(b)):
        if k not in a or k not in b:
            out[k] = "missing in one"; continue
        if a[k].shape != b[k].shape:
            out[k] = f"shape {a[k].shape} vs {b[k].shape}"; continue
        d = float(np.nanmax(np.abs(a[k] - b[k]))) if a[k].size else 0.0
        if d > 0:
            s = float(np.nanmax(np.abs(a[k]))) or 1.0
            out[k] = {"max_abs": d, "rel": d / s}
    return out

def audit(name, backend, model_dir, pip, env=None, audio="jfk.wav", stages=None, timeout=7200):
    R = res["backends"].setdefault(name, {})
    try:
        venv = T / f"venv-{name}"
        if not venv.exists():
            # Kaggle's python has no ensurepip, so the stdlib venv cannot bootstrap;
            # virtualenv carries its own pip.
            subprocess.check_call([sys.executable, "-m", "virtualenv", "-q", "--system-site-packages", str(venv)])
            if pip:
                r = subprocess.run([str(venv / "bin/pip"), "install", "-q"] + pip, capture_output=True, text=True)
                (OUT / f"pip-{name}.log").write_text(r.stdout[-5000:] + r.stderr[-10000:])
        e = dict(os.environ, **(env or {}))
        outs = {}
        for tag, tools in (("mainA", MAIN / "tools"), ("mainB", MAIN / "tools"), ("branch", REPO / "tools")):
            o = T / f"{name}-{tag}.gguf"
            cmd = [str(venv / "bin/python"), str(tools / "dump_reference.py"), "--backend", backend, "--model-dir", str(model_dir),
                   "--audio", str(REPO / "samples" / audio), "--output", str(o)]
            if stages:
                cmd += ["--stages", stages]
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=e, cwd=str(tools))
            (OUT / f"dump-{name}-{tag}.log").write_text(r.stdout[-20000:] + "\n--- stderr ---\n" + r.stderr[-20000:])
            R[f"{tag}_rc"] = r.returncode
            if r.returncode != 0:
                R["error"] = f"{tag}: " + r.stderr[-1500:]; save(); return
            outs[tag] = o
            if tag == "branch":
                R["alias_report"] = [l for l in (r.stdout + r.stderr).splitlines() if "liasing" in l and "]" in l][:20]
        A, B, C = load(outs["mainA"]), load(outs["mainB"]), load(outs["branch"])
        R["determinism_diffs"] = compare(A, B)
        R["main_vs_branch"] = compare(A, C)
        R["n_stages"] = len(A)
        R["verdict"] = ("NONDETERMINISTIC (control differs)" if R["determinism_diffs"]
                        else ("CORRUPTED: " + ", ".join(R["main_vs_branch"])) if R["main_vs_branch"] else "clean")
    except BaseException:
        R["error"] = traceback.format_exc()[-2500:]
    save()

try:
    subprocess.check_call(["git", "clone", "--depth", "1", "-b", "fix/dumper-aliasing", "https://github.com/CrispStrobe/CrispASR.git", str(REPO)])
    subprocess.check_call(["git", "-C", str(REPO), "fetch", "-q", "--depth", "1", "origin", "main"])
    MAIN.mkdir(parents=True, exist_ok=True)
    subprocess.check_call(f"git -C {REPO} archive FETCH_HEAD tools samples | tar -x -C {MAIN}", shell=True)
    res["branch_head"] = subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip()
    res["main_head"] = subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "FETCH_HEAD"], text=True).strip()
    sys.path.insert(0, str(REPO / "tools" / "kaggle"))
    import kaggle_harness as kh
    kh.resolve_hf_token(); os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "gguf", "virtualenv", "pytest"])
    from huggingface_hub import snapshot_download
    T.mkdir(parents=True, exist_ok=True)
    r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-rs", str(REPO / "tests/test_ref_capture_aliasing.py")],
                       capture_output=True, text=True, cwd=str(REPO / "tests"))
    res["unit_tests"] = (r.stdout + r.stderr)[-1500:]; save()
    for rev in ("ctc", "rnnt", "e2e_ctc", "e2e_rnnt"):
        audit(f"gigaam-{rev}", "gigaam", "ai-sage/GigaAM-v3", ["gguf", "sentencepiece", "hydra-core", "omegaconf", "pyannote.audio",
                                                          "transformers==4.57.3"],  # remote code predates 5.x meta init
              env={"GIGAAM_REVISION": rev})
    # liquid_audio.from_pretrained takes a repo id, not a directory
    audit("lfm2-audio", "lfm2-audio", "LiquidAI/LFM2.5-Audio-1.5B", ["gguf", "liquid-audio"])
    md = snapshot_download("Qwen/Qwen3-TTS-12Hz-0.6B-Base", local_dir=str(T / "q3tts"))
    audit("qwen3-tts", "qwen3-tts", md, ["gguf", "qwen-tts"])
    shutil.rmtree(T / "q3tts", ignore_errors=True)
    md = snapshot_download("netease-youdao/Confucius4-R2T2", local_dir=str(T / "r2t2"))
    audit("qwen3-r2t2", "qwen3", md, ["gguf", "qwen-asr"])
    shutil.rmtree(T / "r2t2", ignore_errors=True)
    md = snapshot_download("mistralai/Voxtral-4B-TTS-2603", local_dir=str(T / "vxtts"))
    audit("voxtral-tts", "voxtral-tts", md, ["gguf", "mistral_common", "safetensors"])
    shutil.rmtree(T / "vxtts", ignore_errors=True)
    src = snapshot_download("FireRedTeam/FireRedTTS3", local_dir=str(T / "frt"),
                            allow_patterns=["fireredtts3_base/*", "redae/*", "campp/*", "text_tokenizer/*"])
    subprocess.check_call(["git", "clone", "--depth", "1", "https://github.com/FireRedTeam/FireRedTTS3.git", str(T / "frt-up")])
    audit("fireredtts3", "fireredtts3", src, ["gguf", "safetensors"],
          env={"FIREREDTTS3_UPSTREAM": str(T / "frt-up"), "FIREREDTTS3_SEED": "1234", "OMP_NUM_THREADS": "4"})
except SystemExit:
    pass
except BaseException:
    res["errors"].append(traceback.format_exc())
finally:
    save()
    shutil.rmtree(REPO, ignore_errors=True)
