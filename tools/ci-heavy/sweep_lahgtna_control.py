#!/usr/bin/env python3
"""Control arm for the lahgtna-chatterbox (Arabic) extended-sweep finding.

The regression synthesises 'الطقس جميل جدا اليوم' with the published T3 GGUF
(cstr/lahgtna-chatterbox-v1-GGUF) + base S3Gen and transcribes it with
cohere-transcribe-arabic: 'الان الحال طائي' (WER 1.0). This runs the
upstream model exactly as chatterbox's mtl_tts does
(ChatterboxMultilingualTTS.from_local(<oddadmix snapshot>).generate(text,
language_id="ar"), its own conds.pt, default sampling) for several seeds, the
CrispASR CLI with the regression's arguments for the same seeds, and the CLI
with the upstream conds.pt (baked in the base T3 GGUF) as voice, then scores every WAV with the SAME
cohere-transcribe-arabic GGUF.

  gh workflow run heavy-cpu.yml --ref <branch> -f script=tools/ci-heavy/sweep_lahgtna_control.py \
     -f pip="huggingface_hub numpy soundfile" -f timeout_minutes=300
"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sweep_controls_common as C  # noqa: E402

PHRASE = "الطقس جميل جدا اليوم"
SEEDS = [1234, 1, 2]
UP = r'''
import json, sys, torch, soundfile as sf
import perth
if perth.PerthImplicitWatermarker is None:  # optional native dep missing on CPU runners
    perth.PerthImplicitWatermarker = perth.DummyWatermarker
from huggingface_hub import snapshot_download
from chatterbox.mtl_tts import ChatterboxMultilingualTTS
d = snapshot_download("oddadmix/lahgtna-chatterbox-v1", revision=sys.argv[1])
m = ChatterboxMultilingualTTS.from_local(d, "cpu")
out = {}
for seed in [int(s) for s in sys.argv[4:]]:
    torch.manual_seed(seed)
    wav = m.generate(sys.argv[2], language_id="ar")
    p = f"{sys.argv[3]}/upstream_seed{seed}.wav"
    sf.write(p, wav.squeeze(0).numpy(), m.sr)
    out[seed] = p
print("@@" + json.dumps(out))
'''


def main():
    res = {"phrase": PHRASE, "wavs": {}, "asr": {}}
    b = C.build_crispasr()
    t3 = C.hf("cstr/lahgtna-chatterbox-v1-GGUF", "chatterbox-t3-f16.gguf", "97d045ff08d10dc5686a3bc6a608bf6f8b9584bc")
    s3 = C.hf("cstr/chatterbox-GGUF", "chatterbox-s3gen-q8_0.gguf", "c45504bb8d55473a2213db17ec472ed11b69056a")
    # The base release's T3 GGUF carries the upstream conds.pt verbatim (the
    # same file oddadmix ships); used as a baked voice it replaces the
    # espeak-derived conds the lahgtna GGUF was re-converted with (be25b3e8).
    base_t3 = C.hf("cstr/chatterbox-GGUF", "chatterbox-t3-q4_k.gguf", "c45504bb8d55473a2213db17ec472ed11b69056a")
    asr = C.hf("cstr/cohere-transcribe-arabic-07-2026-GGUF", "cohere-transcribe-arabic-q4_k-imatrix.gguf",
               "dfd6d279cd26c4a8862e5af601976e5c552e709e")
    for seed in SEEDS:
        for tag, extra in (("builtin", []), ("baseconds", ["--voice", base_t3])):
            w = C.SCRATCH / f"cpp_{tag}_seed{seed}.wav"
            r = C.cli(b, ["-m", t3, "--codec-model", s3, "--tts", PHRASE, "--tts-output", w, "--seed", seed,
                          "-l", "ar"] + extra, C.OUT / f"tts-{tag}-{seed}.log",
                      env={"CRISPASR_CHATTERBOX_DEBUG": "1"}, timeout=2400)
            if w.is_file():
                res["wavs"][w.name] = str(w)
            else:
                res["wavs"][w.name] = {"rc": r["rc"], "err": r["stderr_tail"]}
            C.save(res)
    py = C.venv("chatterbox", ["chatterbox-tts", "soundfile", "huggingface_hub"],
                ["--extra-index-url", "https://download.pytorch.org/whl/cpu", "--index-strategy", "unsafe-best-match"])
    (C.SCRATCH / "up.py").write_text(UP)
    r = C.run([py, C.SCRATCH / "up.py", "6b37e50d1952f07306dc9ff3f3d4ff4ddaf32541", PHRASE, C.SCRATCH] +
              [str(s) for s in SEEDS], log=C.OUT / "upstream.log", check=False, timeout=7200)
    m = re.search(r"^@@(.*)$", r.stdout, re.M)
    if m:
        for p in json.loads(m.group(1)).values():
            res["wavs"][Path(p).name] = p
    else:
        res["upstream_error"] = r.stderr[-3000:]
    C.save(res)
    for name, p in sorted(res["wavs"].items()):
        if not isinstance(p, str):
            continue
        a = C.cli(b, ["--backend", "cohere", "-m", asr, "-f", p, "--no-prints"], C.OUT / f"asr-{name}.log")
        a["wer"] = C.wer(PHRASE, a["text"])
        a["cer"] = C.cer(PHRASE, a["text"])
        res["asr"][name] = a
        C.save(res)
        try:
            import shutil
            shutil.copy(p, C.OUT / name)
        except OSError:
            pass
    C.summary(["## lahgtna-chatterbox control", ""] +
              [f"- {k}: WER {v['wer']} CER {v['cer']} `{v['text']}`" for k, v in sorted(res["asr"].items())])
    print(json.dumps(res, indent=1, ensure_ascii=False))
    return 0 if m else 1


if __name__ == "__main__":
    sys.exit(main())
