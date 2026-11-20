#!/usr/bin/env python3
"""Control arm for fastconformer-ctc-ru (FLEURS ru_ru 1554).

After the FireRedPunc fix the CLI keeps word spaces but prints a space before
each punctuation mark ("сказал , что"). The vocabulary has "▁," / "▁." pieces,
so this may be what the model itself emits. This arm runs the upstream NeMo
checkpoint (nvidia/stt_ru_fastconformer_hybrid_large_pc) with the CTC decoder
of the hybrid, next to the CLI on the published F16 and Q4_K GGUFs. If the
CLI F16 text equals NeMo's, the spacing is model output, not a port bug.

  gh workflow run heavy-cpu.yml --ref <branch> -f script=tools/ci-heavy/sweep_ru_nemo_control.py \
     -f pip="huggingface_hub" -f timeout_minutes=240
"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sweep_controls_common as C  # noqa: E402

EXPECTED = "Он сказал, что создал дверной звонок, работающий от WiFi."
UP = r'''
import json, sys
import nemo.collections.asr as na
m = na.models.ASRModel.restore_from(sys.argv[1], map_location="cpu")
m.eval()
res = {}
for dec in ("ctc", "rnnt"):
    m.change_decoding_strategy(decoder_type=dec)
    out = m.transcribe(sys.argv[2:], batch_size=1)
    if isinstance(out, tuple):
        out = out[0]
    res[dec] = [getattr(o, "text", o) for o in out][0]
print("@@" + json.dumps(res, ensure_ascii=False))
'''


def main():
    res = {"expected": EXPECTED}
    wav = C.hf("cstr/crispasr-regression-fixtures", "fleurs-validation/ru_ru/1554.wav",
               "0e0c4dfb541647af315f73c94c24c65394ccdabf")
    res["wav"] = str(wav)
    b = C.build_crispasr()
    for q in ("f16", "q4_k"):
        g = C.hf("cstr/stt-ru-fastconformer-hybrid-ctc-large-GGUF", f"stt-ru-fastconformer-hybrid-ctc-large-{q}.gguf",
                 "d2c81ae7e3af263b6762b4e440ff8a9650082d45")
        r = C.cli(b, ["-m", g, "-f", wav, "-l", "ru"], C.OUT / f"cpp-{q}.log")
        r["wer"] = C.wer(EXPECTED, r["text"])
        res[f"cpp_{q}"] = r
        C.save(res)
    py = C.venv("nemo", ["nemo_toolkit[asr]", "huggingface_hub"],
                ["--extra-index-url", "https://download.pytorch.org/whl/cpu", "--index-strategy", "unsafe-best-match"])
    nemo = C.hf("nvidia/stt_ru_fastconformer_hybrid_large_pc", "stt_ru_fastconformer_hybrid_large_pc.nemo",
                "3508825018a834188b025fc89a9abe9ed77530ef")
    (C.SCRATCH / "up.py").write_text(UP)
    r = C.run([py, C.SCRATCH / "up.py", nemo, wav], log=C.OUT / "nemo.log", check=False, timeout=7200)
    m = re.search(r"^@@(.*)$", r.stdout, re.M)
    up = json.loads(m.group(1)) if m else None
    res["nemo"] = {"rc": r.returncode, "text": up}
    if up:
        res["cpp_f16_equals_nemo_ctc"] = res["cpp_f16"]["text"].strip() == up["ctc"].strip()
    C.save(res)
    C.summary(["## fastconformer-ctc-ru control", "", f"- NeMo upstream: `{up}`"] +
              [f"- {k}: WER {v['wer']} `{v['text']}`" for k, v in res.items() if k.startswith("cpp_") and isinstance(v, dict)])
    print(json.dumps(res, indent=1, ensure_ascii=False))
    return 0 if up else 1


if __name__ == "__main__":
    sys.exit(main())
