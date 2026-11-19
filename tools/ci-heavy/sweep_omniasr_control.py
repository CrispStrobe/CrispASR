#!/usr/bin/env python3
"""Control arm for the omniasr-ctc-300m-v2 extended-sweep finding.

CLI output on samples/jfk.wav stops after "And so my fellow americas ask
not." while omniasr-ctc-1b-v2 on the same runtime passes. Runs the official
omnilingual-asr pipeline (ASRInferencePipeline, fairseq2, its own venv) on
exactly the audio slices the CLI decodes — the whole clip and every split
window for several window lengths — next to the CLI on the published F16 and
Q4_K GGUFs with CRISPASR_OMNIASR_CTC_CHUNK_SEC set to the same window.

  gh workflow run heavy-cpu.yml --ref <branch> -f script=tools/ci-heavy/sweep_omniasr_control.py \
     -f pip="huggingface_hub numpy soundfile" -f timeout_minutes=240
"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sweep_controls_common as C  # noqa: E402

JFK = "And so, my fellow Americans, ask not what your country can do for you, ask what you can do for your country."
WINDOWS = [0, 4, 5, 6, 7]
UP = r'''
import json, sys, torch
from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline
card, files = sys.argv[1], sys.argv[2:]
p = ASRInferencePipeline(model_card=card, device="cpu", dtype=torch.float32)
print("@@" + json.dumps({f: p.transcribe([f], batch_size=1)[0] for f in files}, ensure_ascii=False))
'''


def windows(n, win_s, sr=16000):
    """Mirror examples/cli/crispasr_backend_omniasr.cpp's window split."""
    win = int(win_s * sr)
    if win <= 0 or n <= win:
        return [(0, n)]
    overlap = min(sr, win // 4)
    n_chunks = (n - overlap + (win - overlap) - 1) // (win - overlap)
    step = (n - overlap + n_chunks - 1) // n_chunks
    return [(c * step, n if c == n_chunks - 1 else min(n, c * step + step + overlap)) for c in range(n_chunks)]


def main():
    import numpy as np
    import soundfile as sf
    res = {"windows": {}, "upstream": {}, "cpp": {}}
    a, sr = sf.read(C.ROOT / "samples/jfk.wav", dtype="float32")
    slices = {}
    for w in WINDOWS:
        res["windows"][w] = windows(len(a), w)
        for (s, e) in res["windows"][w]:
            p = C.SCRATCH / f"jfk_{s}_{e}.wav"
            if p.name not in slices:
                sf.write(p, a[s:e], sr, subtype="PCM_16")
                slices[p.name] = str(p)
    upstream_only = "--upstream-only" in sys.argv
    b = None if upstream_only else C.build_crispasr()
    for q, f in (() if upstream_only else (("q4_k", "omniasr-ctc-300m-v2-q4_k.gguf"), ("f16", "omniasr-ctc-300m-v2.gguf"))):
        g = C.hf("cstr/omniASR-CTC-300M-v2-GGUF", f, "fc3e3765175be5aaf0c9e75ed25a7e8843ef04d5")
        for w in WINDOWS:
            r = C.cli(b, ["-m", g, "-f", C.ROOT / "samples/jfk.wav", "--no-punctuation", "-l", "en"],
                      C.OUT / f"cpp-{q}-win{w}.log", env={"CRISPASR_OMNIASR_CTC_CHUNK_SEC": str(w)})
            r["wer"] = C.wer(JFK, r["text"])
            res["cpp"][f"{q}|win{w}"] = r
            C.save(res)
        for name, p in sorted(slices.items()):
            r = C.cli(b, ["-m", g, "-f", p, "--no-punctuation", "-l", "en"], C.OUT / f"cpp-{q}-{name}.log",
                      env={"CRISPASR_OMNIASR_CTC_CHUNK_SEC": "0"})
            res["cpp"][f"{q}|{name}"] = r
            C.save(res)
    C.run(["sudo", "apt-get", "install", "-y", "-qq", "libsndfile1"], check=False)  # fairseq2n needs it
    py = C.venv("omni", ["omnilingual-asr", "numpy", "soundfile", "torch==2.8.0", "torchaudio==2.8.0"],
                ["--extra-index-url", "https://download.pytorch.org/whl/cu128", "--index-strategy", "unsafe-best-match"])
    (C.SCRATCH / "up.py").write_text(UP)
    files = [str(C.ROOT / "samples/jfk.wav")] + sorted(slices.values())
    for card in ("omniASR_CTC_300M_v2", "omniASR_CTC_1B_v2"):
        r = C.run([py, C.SCRATCH / "up.py", card] + files, log=C.OUT / f"upstream-{card}.log", check=False, timeout=7200)
        m = re.search(r"^@@(.*)$", r.stdout, re.M)
        res["upstream"][card] = {Path(k).name: v for k, v in json.loads(m.group(1)).items()} if m else {"rc": r.returncode}
        C.save(res)
    lines = ["## omniasr-ctc-300m-v2 control", ""]
    for k, v in res["cpp"].items():
        lines.append(f"- cpp {k}: `{v['text']}`" + (f" WER {v['wer']}" if "wer" in v else ""))
    for card, d in res["upstream"].items():
        for k, v in d.items():
            lines.append(f"- upstream {card} {k}: `{v}`")
    C.summary(lines)
    print(json.dumps(res, indent=1, ensure_ascii=False))
    return 0 if all("rc" not in d for d in res["upstream"].values()) else 1


if __name__ == "__main__":
    sys.exit(main())
