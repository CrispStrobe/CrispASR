#!/usr/bin/env python3
"""#610: VibeVoice-ASR answers cut off at the token limit, end to end on CPU.

A small --max-new-tokens cuts the JSON answer inside the first Content, the
same shape as a repetition loop running into the limit. Before #610 the CLI
returned the raw JSON blob as the transcript; now it must return the decoded
part of the Content (no '[{', no '"Content"'). The uncapped run is the
control: it must still give the full JFK sentence.

  gh workflow run heavy-cpu.yml --ref <branch> -f script=tools/ci-heavy/vibevoice_610_truncation.py \
     -f pip="huggingface_hub" -f timeout_minutes=120
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sweep_controls_common as C  # noqa: E402


def main():
    b = C.build_crispasr()
    model = C.hf("cstr/vibevoice-asr-GGUF", "vibevoice-asr-q4_k.gguf", None)
    wav = Path(C.ROOT) / "samples" / "jfk.wav"
    nt = str(os.cpu_count() or 4)
    res = {"runs": {}}
    ok = True
    for cap in [0, 24, 40, 60]:
        args = ["--backend", "vibevoice", "-m", model, "-f", wav, "-t", nt, "-np", "-nt"]
        if cap:
            args += ["-n", str(cap)]
        r = C.run([b] + args, C.OUT / f"cap{cap}.log", check=False, timeout=3000)
        text = r.stdout.strip()
        raw_json = text.startswith("[") or '"Content"' in text or '"Start"' in text
        if cap == 0:
            good = "ask not what your country" in text.lower() and not raw_json
        else:
            good = not raw_json and r.returncode == 0
        ok &= good
        res["runs"][cap] = {"rc": r.returncode, "text": text, "raw_json": raw_json, "ok": good}
        C.save(res)
    res["ok"] = ok
    C.save(res)
    lines = ["## #610 VibeVoice truncation", "", "| max_new_tokens | ok | raw JSON | text |", "|---|---|---|---|"]
    for cap, d in res["runs"].items():
        lines.append(f"| {cap or 'default'} | {d['ok']} | {d['raw_json']} | {d['text'][:160]} |")
    lines.append(f"\noverall: {'PASS' if ok else 'FAIL'}")
    C.summary(lines)
    if not ok:
        raise SystemExit("truncation check failed")


if __name__ == "__main__":
    main()
