#!/usr/bin/env python3
"""Which front-end does FunASR's AutoModel really use for SenseVoiceSmall,
paraformer-zh and Fun-ASR-Nano?

Our dumpers build their own WavFrontend(cmvn_file=None). For each model:
report the AutoModel frontend's CMVN state, compare its features (dither 0)
with the dumper-style cmvn-None features, and record the real pipeline's
transcript (rich tags included) on jfk and the zh clip.
"""
import json, subprocess, sys, traceback
from pathlib import Path
OUT = Path("/kaggle/working/out"); OUT.mkdir(parents=True, exist_ok=True)
res = {"models": {}, "errors": []}
def save(): (OUT / "frontend_truth.json").write_text(json.dumps(res, indent=1, ensure_ascii=False, default=str))
try:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "funasr", "modelscope"])
    import numpy as np, torch, soundfile as sf
    from huggingface_hub import snapshot_download
    from funasr import AutoModel
    from funasr.frontends.wav_frontend import WavFrontend
    for url, n in (("https://github.com/CrispStrobe/CrispASR/raw/main/samples/jfk.wav", "jfk"),
                   ("https://github.com/CrispStrobe/CrispASR/raw/main/samples/paraformer_zh.wav", "zh")):
        subprocess.check_call(["wget", "-q", "-O", f"/tmp/{n}.wav", url])
    for repo in ("FunAudioLLM/SenseVoiceSmall", "funasr/paraformer-zh", "FunAudioLLM/Fun-ASR-Nano-2512"):
        R = res["models"].setdefault(repo, {})
        try:
            md = snapshot_download(repo, local_dir=f"/tmp/{repo.split('/')[1]}")
            if "Fun-ASR-Nano" in repo:
                from funasr.models.fun_asr_nano.model import FunASRNano
                model, kwargs = FunASRNano.from_pretrained(model=md, device="cpu")
                fe = kwargs.get("frontend")
                am = None
            else:
                am = AutoModel(model=md, device="cpu", disable_update=True)
                fe = am.kwargs.get("frontend")
            R["frontend_type"] = type(fe).__name__
            R["frontend_cmvn_file"] = getattr(fe, "cmvn_file", "n/a")
            R["frontend_has_cmvn"] = getattr(fe, "cmvn", None) is not None
            R["frontend_dither"] = getattr(fe, "dither", "n/a")
            R["frontend_conf"] = {k: str(v) for k, v in (am.kwargs.get("frontend_conf", {}) if am else kwargs.get("frontend_conf", {})).items()}
            mine = WavFrontend(cmvn_file=None, fs=16000, window="hamming", n_mels=80, frame_length=25, frame_shift=10,
                               lfr_m=7, lfr_n=6, dither=0.0, upsacle_samples=True, snip_edges=True).eval()
            for c in ("jfk", "zh"):
                a, sr = sf.read(f"/tmp/{c}.wav", dtype="float32")
                sig = torch.from_numpy(a)[None]; ln = torch.tensor([len(a)])
                old = fe.dither; fe.dither = 0.0
                with torch.no_grad():
                    f1, _ = fe(sig, ln); f2, _ = mine(sig, ln)
                fe.dither = old
                f1 = f1[0].numpy(); f2 = f2[0].numpy()
                cs = (f1 * f2).sum(1) / (np.linalg.norm(f1, axis=1) * np.linalg.norm(f2, axis=1) + 1e-9)
                R[f"feat_{c}"] = {"real_vs_dumper_cos_min": float(cs.min()), "max_abs": float(np.abs(f1 - f2).max()),
                                  "real_mean": float(f1.mean()), "dumper_mean": float(f2.mean())}
                if am is not None:
                    kw = {"language": "auto", "use_itn": True} if "SenseVoice" in repo else {}
                    R[f"text_{c}"] = str(am.generate(input=f"/tmp/{c}.wav", **kw)[0].get("text"))
                save()
        except BaseException:
            R["error"] = traceback.format_exc()[-2500:]
        save()
except BaseException:
    res["errors"].append(traceback.format_exc())
finally:
    save()
