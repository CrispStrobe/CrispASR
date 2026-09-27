#!/usr/bin/env python3
"""omniASR-CTC split window A/B (fix/omniasr-ctc-chunk).

CRISPASR_OMNIASR_CTC_CHUNK_SEC in {0 (never split), 7, 15, 30} x models
{1B-v2 q4_k (pinned regression file), 300M-v2 q4_k} x audio {jfk 11 s, jfk x2,
jfk x3, multispeaker ~31 s}. --chunk-seconds 120 keeps the CLI's own 30 s
splitter out of it. Ground truth: jfk text for the jfk-based clips, and - when
it installs - the official omnilingual-asr pipeline (unchunked, its limit is
40 s) run in its own venv for every clip.
"""
import json, os, re, subprocess, sys, traceback
from pathlib import Path
OUT = Path("/kaggle/working/out"); OUT.mkdir(parents=True, exist_ok=True)
REPO = Path("/tmp/CrispASR"); G = Path("/tmp/g"); G.mkdir(exist_ok=True)
REF = os.environ.get("CRISPASR_REF", "fix/omniasr-ctc-chunk")
JFK = "and so my fellow americans ask not what your country can do for you ask what you can do for your country"
res = {"errors": [], "runs": {}, "upstream": {}}
def save(): (OUT / "result.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
def norm(t): return " ".join(re.sub(r"[^\w\s']", " ", (t or "").lower()).split())
def wer(ref, hyp):
    r, h = norm(ref).split(), norm(hyp).split()
    d = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        p, d[0] = d[0], i
        for j in range(1, len(h) + 1):
            p, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, p + (r[i - 1] != h[j - 1]))
    return d[len(h)] / max(1, len(r))
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
    arch = kh.detect_cuda_arch()
    kh.sh(f"cmake -S {REPO} -B {REPO}/build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCRISPASR_OPUS=OFF -DCRISPASR_AMR=OFF "
          + " ".join(kh.cuda_build_flags(arch) + kh.cache_and_link_flags()))
    with kh.build_heartbeat("cmake.build"):
        kh.sh(f"cmake --build {REPO}/build -j{kh.safe_build_jobs(gpu=True)} --target crispasr-cli")
    B = REPO / "build" / "bin" / "crispasr"
    a, sr = sf.read(REPO / "samples/jfk.wav", dtype="float32")
    audio = {"jfk": REPO / "samples/jfk.wav", "multispeaker": REPO / "samples/multispeaker.wav"}
    for k in (2, 3):
        p = G / f"jfk_x{k}.wav"; sf.write(p, np.concatenate([a] * k), sr, subtype="PCM_16"); audio[f"jfk_x{k}"] = p
    truth = {"jfk": JFK, "jfk_x2": " ".join([JFK] * 2), "jfk_x3": " ".join([JFK] * 3)}
    models = {"1b-v2": hf_hub_download("cstr/omniASR-CTC-1B-v2-GGUF", "omniasr-ctc-1b-v2-q4_k.gguf",
                                       revision="e05a1d261ca3f0dfe6932ee9d50794860d849ea0", cache_dir=str(G))}
    try:
        from huggingface_hub import HfApi
        fs = [f for f in HfApi().list_repo_files("cstr/omniASR-CTC-300M-v2-GGUF") if f.endswith("q4_k.gguf")]
        models["300m-v2"] = hf_hub_download("cstr/omniASR-CTC-300M-v2-GGUF", fs[0], cache_dir=str(G))
    except Exception as e:
        res["errors"].append(f"300m: {e}")
    save()
    for mk, mp in models.items():
        for ak, ap in audio.items():
            for w in ("0", "7", "15", "30"):
                r = subprocess.run([str(B), "-m", mp, "-f", str(ap), "-np", "--chunk-seconds", "120"], capture_output=True,
                                   text=True, env=dict(os.environ, CRISPASR_OMNIASR_CTC_CHUNK_SEC=w), timeout=900)
                txt = " ".join(l.strip() for l in r.stdout.splitlines() if l.strip())
                ent = {"rc": r.returncode, "text": txt}
                if ak in truth: ent["wer_truth"] = round(wer(truth[ak], txt), 4)
                if r.returncode: ent["err"] = r.stderr[-400:]
                res["runs"][f"{mk}|{ak}|w{w}"] = ent
                print(mk, ak, w, ent.get("wer_truth"), txt[:160], flush=True); save()

    # official pipeline in its own venv (pip-before-import; never touch this interpreter's stack)
    try:
        V = G / "venv"
        subprocess.check_call([sys.executable, "-m", "venv", str(V)])
        py = str(V / "bin" / "python")
        subprocess.check_call([py, "-m", "pip", "install", "-q", "omnilingual-asr"], timeout=1800)
        cards = {"1b-v2": "omniASR_CTC_1B_v2", "300m-v2": "omniASR_CTC_300M_v2"}
        script = G / "up.py"
        script.write_text(
            "import json,sys\nfrom omnilingual_asr.models.inference.pipeline import ASRInferencePipeline\n"
            "card=sys.argv[1]; files=sys.argv[2:]\np=ASRInferencePipeline(model_card=card)\n"
            "print('@@'+json.dumps(dict(zip(files,p.transcribe(files,batch_size=1)))))\n")
        for mk, card in cards.items():
            if mk not in models: continue
            r = subprocess.run([py, str(script), card] + [str(p) for p in audio.values()], capture_output=True, text=True,
                               timeout=3600)
            m = re.search(r"^@@(.*)$", r.stdout, re.M)
            if m:
                up = json.loads(m.group(1)); inv = {str(v): k for k, v in audio.items()}
                res["upstream"][mk] = {inv[f]: t for f, t in up.items()}
            else:
                res["upstream"][mk] = {"error": (r.stdout + r.stderr)[-1500:]}
            save()
        for key, ent in res["runs"].items():
            mk, ak, _ = key.split("|")
            ut = res["upstream"].get(mk, {}).get(ak)
            if isinstance(ut, str): ent["wer_upstream"] = round(wer(ut, ent["text"]), 4)
    except Exception:
        res["errors"].append("upstream: " + traceback.format_exc()[-1500:])
except BaseException:
    res["errors"].append(traceback.format_exc())
finally:
    save()
