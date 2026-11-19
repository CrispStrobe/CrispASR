#!/usr/bin/env python3
"""#595 FluidVoice-Labs/fluid-speech-pro: NeMo stage diff + Q8_0 baseline.

The NeMo fp32 reference (model + checkpoint, ~6 GB RSS) does not fit the
8 GB maintainer VPS, hence heavy-cpu.yml. Steps:

  1. build crispasr, crispasr-diff, crispasr-quantize
  2. download fluid-speech-pro.nemo and, as the baseline, nvidia's
     parakeet-tdt-0.6b-v3.nemo at pinned revisions (sha256 = LFS oid)
  3. convert both to F16 (Fluid also --clean-mode), quantize q8_0 / q4_k
  4. NeMo stage references (tools/dump_reference.py --backend parakeet; the
     Fluid FrameCleaner and, with CRISPASR_REF_FLUID_CLEAN=1, clean mode are
     attached by tools/reference_backends/parakeet.py) on samples/jfk.wav
     and samples/multispeaker.wav
  5. crispasr-diff parakeet + the CLI transcript on every model x quant x clip

Fails when a Fluid F16 stage fails or a F16 transcript differs from NeMo.
The v3 rows are the baseline for how far Q8_0/Q4_K stages move on this
architecture (crispasr-diff thresholds are tuned for F16).

  gh workflow run heavy-cpu.yml -f script=tools/ci-heavy/fluid_speech_validate.py \
     -f pip="torch nemo_toolkit[asr] gguf==0.17.1 soundfile" -f args=--export-gguf --ref main
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(os.environ.get("HEAVY_OUT", ROOT / "fluid-out"))
SCR = Path(os.environ.get("HEAVY_SCRATCH", ROOT / "fluid-scratch"))
MODELS = {  # name: (repo, revision, file, sha256)
    "fluid": ("FluidVoice-Labs/fluid-speech-pro", "46027a3bffc5817ce8f9b29add0a062610134a90",
              "fluid-speech-pro.nemo", "f3e377b792ae5a52a3c071adb7eba5504306f2f4bdcf933ae9da2cbcab6f0449"),
    "v3": ("nvidia/parakeet-tdt-0.6b-v3", "541d1f99c6b0c3cd0b11a95167540bb8edefd82b",
           "parakeet-tdt-0.6b-v3.nemo", "3cbdc85877e668ca7b82d0d56770eb1fac76691f55d6b97545e8d61ca588d10d"),
}
VARIANTS = [("fluid", "fluid", False), ("fluid-clean", "fluid", True), ("v3", "v3", False)]
CLIPS = [("jfk", "samples/jfk.wav"), ("multispeaker", "samples/multispeaker.wav")]
QUANTS = ["f16", "q8_0", "q4_k"]


def run(cmd, log=None, env=None, check=True, merge=True):
    print("+", " ".join(str(c) for c in cmd), flush=True)
    p = subprocess.run([str(c) for c in cmd], stdout=subprocess.PIPE,
                       stderr=subprocess.STDOUT if merge else subprocess.PIPE, text=True,
                       env={**os.environ, **(env or {})})
    if log:
        Path(log).write_text(p.stdout + ("" if merge else "\n--- stderr ---\n" + p.stderr))
    print(p.stdout[-4000:], flush=True)
    if check and p.returncode != 0:
        raise SystemExit(f"command failed ({p.returncode}): {cmd[0]}")
    return p


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def norm_words(s):
    s = unicodedata.normalize("NFKC", s).lower()
    s = re.sub(r"[^\w\s']", " ", s)
    return s.split()


def edit(a, b):
    d = list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(b) + 1):
            cur = min(d[j] + 1, d[j - 1] + 1, prev + (a[i - 1] != b[j - 1]))
            prev, d[j] = d[j], cur
    return d[len(b)]


def wer(hyp, ref):
    r = norm_words(ref)
    return edit(norm_words(hyp), r) / max(1, len(r))


def cer(hyp, ref):
    h = "".join(norm_words(hyp))
    r = "".join(norm_words(ref))
    return edit(list(h), list(r)) / max(1, len(r))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    SCR.mkdir(parents=True, exist_ok=True)
    bd = SCR / "build"
    run(["cmake", "-S", ROOT, "-B", bd, "-G", "Ninja", "-DCMAKE_BUILD_TYPE=Release", "-DGGML_NATIVE=OFF"],
        OUT / "cmake.log")
    run(["cmake", "--build", bd, "-j", str(os.cpu_count() or 4), "--target", "crispasr", "crispasr-diff",
         "crispasr-quantize"], OUT / "build.log")
    binp = lambda n: next(p for p in (bd / "bin" / n, bd / n) if p.exists())

    from huggingface_hub import hf_hub_download
    nemo = {}
    res = {"models": {}, "variants": {}}
    for k, (repo, rev, fn, want) in MODELS.items():
        p = Path(hf_hub_download(repo, fn, revision=rev, local_dir=SCR / k))
        got = sha256(p)
        res["models"][k] = {"repo": repo, "revision": rev, "sha256": got}
        if got != want:
            raise SystemExit(f"{fn}: sha256 {got} != pinned {want}")
        nemo[k] = p

    failures = []
    for var, mk, clean in VARIANTS:
        gg = {"f16": SCR / f"{var}-f16.gguf"}
        cmd = [sys.executable, ROOT / "models/convert-parakeet-to-gguf.py", "--nemo", nemo[mk],
               "--output", gg["f16"], "--extract-dir", SCR / f"ex-{var}"]
        run(cmd + (["--clean-mode"] if clean else []), OUT / f"convert-{var}.log")
        for q in QUANTS[1:]:
            gg[q] = SCR / f"{var}-{q}.gguf"
            run([binp("crispasr-quantize"), gg["f16"], gg[q], q], OUT / f"quantize-{var}-{q}.log")
        v = res["variants"].setdefault(var, {"gguf": {q: {"bytes": p.stat().st_size, "sha256": sha256(p)}
                                                       for q, p in gg.items()}, "clips": {}})
        if "--export-gguf" in sys.argv[1:] and mk == "fluid":
            import shutil
            (OUT / "gguf").mkdir(exist_ok=True)
            for q, p in gg.items():
                name = "fluid-speech-pro-" + ("clean-" if clean else "") + q + ".gguf"
                shutil.copy2(p, OUT / "gguf" / name)
        for clip, sample in CLIPS:
            ref = SCR / f"ref-{var}-{clip}.gguf"
            env = {"CRISPASR_REF_FLUID_CLEAN": "1" if clean else "0", "TMPDIR": str(SCR)}
            run([sys.executable, ROOT / "tools/dump_reference.py", "--backend", "parakeet", "--model-dir", nemo[mk],
                 "--audio", ROOT / sample, "--output", ref], OUT / f"ref-{var}-{clip}.log", env=env)
            tr = run([sys.executable, ROOT / "tools/ci-heavy/fluid_speech_validate.py", "--transcribe", nemo[mk],
                      ROOT / sample], OUT / f"nemo-text-{var}-{clip}.log", env=env, merge=False)
            nemo_text = tr.stdout.strip().splitlines()[-1] if tr.stdout.strip() else ""
            c = v["clips"].setdefault(clip, {"nemo_text": nemo_text})
            for q in QUANTS:
                d = run([binp("crispasr-diff"), "parakeet", gg[q], ref, ROOT / sample],
                        OUT / f"diff-{var}-{clip}-{q}.log", check=False)
                rows = re.findall(r"^\[(PASS|FAIL|ERR )\]\s+(\S+)\s+(.*)$", d.stdout, re.M)
                enc = next((dt for s, st, dt in rows if st == "encoder_output"), "")
                m = re.search(r"cos_min=([0-9.]+)", enc)
                t = run([binp("crispasr"), "-m", gg[q], "-f", ROOT / sample, "-np", "-t", "4"],
                        OUT / f"cli-{var}-{clip}-{q}.log", check=False, merge=False)
                hyp = " ".join(l.strip() for l in t.stdout.splitlines() if l.strip())
                c[q] = {"pass": sum(s == "PASS" for s, _, _ in rows), "fail": sum(s != "PASS" for s, _, _ in rows),
                        "encoder_cos_min": float(m.group(1)) if m else None, "cli_text": hyp,
                        "exact": hyp == nemo_text, "wer_vs_nemo": wer(hyp, nemo_text)}
                if q == "f16" and mk == "fluid":
                    if d.returncode != 0 or c[q]["fail"] or not rows:
                        failures.append(f"{var}/{clip}/f16: diff")
                    if not c[q]["exact"]:
                        failures.append(f"{var}/{clip}/f16: transcript differs from NeMo")
    res["failures"] = failures
    (OUT / "result.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
    lines = ["## Fluid Speech Pro (#595) vs NeMo; parakeet-tdt-0.6b-v3 as quant baseline", "",
             "| model | clip | quant | stages pass/fail | encoder cos_min | text == NeMo | WER vs NeMo |",
             "|---|---|---|---|---|---|---|"]
    for var, v in res["variants"].items():
        for clip, c in v["clips"].items():
            for q in QUANTS:
                r = c[q]
                lines.append(f"| {var} | {clip} | {q} | {r['pass']}/{r['fail']} | {r['encoder_cos_min']} | "
                             f"{r['exact']} | {r['wer_vs_nemo']:.4f} |")
    lines += ["", f"failures: {failures or 'none'}"]
    (OUT / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 1 if failures else 0


def transcribe(nemo_path, wav):
    """NeMo transcript with the same loader the stage dump uses (no dither)."""
    sys.path.insert(0, str(ROOT / "tools"))
    from reference_backends import parakeet as P
    import nemo.collections.asr as nemo_asr
    import torch
    try:
        model = nemo_asr.models.ASRModel.restore_from(nemo_path, map_location="cpu")
    except RuntimeError:
        model = nemo_asr.models.ASRModel.restore_from(nemo_path, map_location="cpu", strict=False)
    P._attach_fluid_speech(model, nemo_path)
    model.eval()
    model.preprocessor.featurizer.dither = 0.0
    model.preprocessor.featurizer.pad_to = 0
    with torch.no_grad():
        h = model.transcribe([wav], batch_size=1)[0]
    print(getattr(h, "text", h))


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "--transcribe":
        transcribe(sys.argv[2], sys.argv[3])
        sys.exit(0)
    sys.exit(main())
