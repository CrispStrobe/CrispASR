#!/usr/bin/env python3
"""Control arm for the parakeet-ctc-1.1b-ja extended-sweep finding (CER 0.56).

Runs the upstream NeMo model (grider-transwithai/parakeet-ctc-1.1b-ja,
parakeet-ja-gal.nemo — the only non-corrupt checkpoint, see docs/architecture.md)
exactly as its model card does (ASRModel.restore_from(...).transcribe([...]))
on the regression fixture, next to the CrispASR CLI on the published F16 and
Q8_0 GGUFs. If NeMo produces the same bad transcript the finding is upstream
model quality, not a port bug.

  gh workflow run heavy-cpu.yml --ref <branch> -f script=tools/ci-heavy/sweep_ja_nemo_control.py \
     -f pip="huggingface_hub" -f timeout_minutes=240
"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sweep_controls_common as C  # noqa: E402

EXPECTED = "ピッチャーの山本は1回戦の鳥取城北戦は8回、6対3、3点リードの場面で3人目でマウンドに上がりました。はきょう先発の石川。"
UP = r'''
import json, sys
import nemo.collections.asr as na
m = na.models.ASRModel.restore_from(sys.argv[1], map_location="cpu")
m.eval()
out = m.transcribe(sys.argv[2:], batch_size=1)
if isinstance(out, tuple):
    out = out[0]
print("@@" + json.dumps([getattr(o, "text", o) for o in out], ensure_ascii=False))
'''


def main():
    res = {"expected": EXPECTED}
    wav = C.hf("cstr/crispasr-regression-fixtures", "parakeet-tdt-0.6b-ja/reazon_baseball_14s/audio.wav",
               "70853acda3351e3456f7ed53bd5845a221a6de69")
    res["wav"] = str(wav)
    b = C.build_crispasr()
    for q in ("f16", "q8_0"):
        g = C.hf("cstr/parakeet-ctc-1.1b-ja-GGUF", f"parakeet-ctc-1.1b-ja-{q}.gguf",
                 "033e8bf08d60a582b47a92bba07ca9e392982861")
        for tag, extra in (("default", []), ("no-punc", ["--no-punctuation"])):
            r = C.cli(b, ["-m", g, "-f", wav] + extra, C.OUT / f"cpp-{q}-{tag}.log")
            r["cer"] = C.cer(EXPECTED, r["text"])
            res[f"cpp_{q}_{tag}"] = r
            C.save(res)
        g.unlink(missing_ok=True)
    py = C.venv("nemo", ["nemo_toolkit[asr]", "huggingface_hub"],
                ["--extra-index-url", "https://download.pytorch.org/whl/cpu", "--index-strategy", "unsafe-best-match"])
    nemo = C.hf("grider-transwithai/parakeet-ctc-1.1b-ja", "parakeet-ja-gal.nemo",
                "81b954e9d79da985572028fbe42a7e3c93313dfd")
    (C.SCRATCH / "up.py").write_text(UP)
    r = C.run([py, C.SCRATCH / "up.py", nemo, wav], log=C.OUT / "nemo.log", check=False, timeout=7200)
    m = re.search(r"^@@(.*)$", r.stdout, re.M)
    up = json.loads(m.group(1))[0] if m else None
    res["nemo"] = {"rc": r.returncode, "text": up, "cer": C.cer(EXPECTED, up) if up else None}
    C.save(res)
    C.summary(["## parakeet-ctc-1.1b-ja control", "",
               f"- NeMo upstream: CER {res['nemo']['cer']} `{up}`"] +
              [f"- {k}: CER {v['cer']} `{v['text']}`" for k, v in res.items() if k.startswith("cpp_")])
    print(json.dumps(res, indent=1, ensure_ascii=False))
    return 0 if up else 1


if __name__ == "__main__":
    sys.exit(main())
