#!/usr/bin/env python3
"""#595 Musci-research/Musci-ASR-qwen-1019: convert -> quantize -> qwen_asr stage diff -> WER.

A GRPO fine-tune of Qwen3-ASR-1.7B with config, tokenizer, chat template and
preprocessor byte-identical to the base (thinker.* layout, tied lm_head), so
it runs on the stock qwen3 backend. The fp32 1.7B reference does not fit the
8 GB maintainer VPS, hence heavy-cpu.yml. Steps:

  1. build crispasr, crispasr-diff, crispasr-quantize (CPU, Release)
  2. download the repo at a pinned revision, verify the safetensors sha256
     against the pinned LFS oid
  3. convert to F16 with models/convert-qwen3-asr-to-gguf.py, then
     crispasr-quantize to q8_0 and q4_k
  4. per clip (samples/jfk.wav English, samples/paraformer_zh.wav Chinese,
     auto-detect): fp32 qwen_asr reference dump (tools/dump_reference.py
     --backend qwen3)
  5. crispasr-diff qwen3 and the crispasr CLI transcript on every quant x clip

Fails (exit 1) when a F16 diff stage fails, a F16 transcript differs from the
reference, or JFK WER exceeds 0.10 on any quant.

  gh workflow run heavy-cpu.yml -f script=tools/ci-heavy/musci_validate.py \
     -f pip="torch numpy soundfile gguf==0.17.1 safetensors qwen-asr huggingface_hub librosa" \
     -f args=--export-gguf -f timeout_minutes=240 --ref main
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
REPO = "Musci-research/Musci-ASR-qwen-1019"
REVISION = os.environ.get("MUSCI_REVISION", "c376e9c4faa9e01a6b54752080448f617847ffe1")
WEIGHTS = "model-00000-of-00001.safetensors"
WEIGHTS_SHA256 = "05f2186e787588b9d699299f34c0fdf173ecea2197753e30439091d6af4f98f1"
NAME = "musci-asr-qwen-1019"
OUT = Path(os.environ.get("HEAVY_OUT", ROOT / "musci-out"))
SCR = Path(os.environ.get("HEAVY_SCRATCH", ROOT / "musci-scratch"))
JFK = "And so, my fellow Americans, ask not what your country can do for you, ask what you can do for your country."
CASES = [("jfk", "samples/jfk.wav"), ("zh", "samples/paraformer_zh.wav")]
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


def ref_text(path):
    import gguf
    r = gguf.GGUFReader(str(path))
    f = r.fields.get("crispasr.ref.generated_text")
    return bytes(f.parts[f.data[0]]).decode("utf-8") if f else ""


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    SCR.mkdir(parents=True, exist_ok=True)
    res = {"repo": REPO, "revision": REVISION, "cases": {}, "quants": {}}

    bd = SCR / "build"
    run(["cmake", "-S", ROOT, "-B", bd, "-G", "Ninja", "-DCMAKE_BUILD_TYPE=Release", "-DGGML_NATIVE=OFF"],
        OUT / "cmake.log")
    run(["cmake", "--build", bd, "-j", str(os.cpu_count() or 4), "--target", "crispasr", "crispasr-diff",
         "crispasr-quantize"], OUT / "build.log")
    binp = lambda n: next(p for p in (bd / "bin" / n, bd / n) if p.exists())

    from huggingface_hub import snapshot_download
    md = Path(snapshot_download(REPO, revision=REVISION, local_dir=SCR / "hf",
                                allow_patterns=["*.json", "*.txt", "*.py", WEIGHTS]))
    got = sha256(md / WEIGHTS)
    res["safetensors_sha256"] = got
    if got != WEIGHTS_SHA256:
        raise SystemExit(f"safetensors sha256 {got} != pinned {WEIGHTS_SHA256}")

    gg = {"f16": SCR / f"{NAME}-f16.gguf"}
    run([sys.executable, ROOT / "models/convert-qwen3-asr-to-gguf.py", "--input", md, "--output", gg["f16"]],
        OUT / "convert.log")
    for q in QUANTS[1:]:
        gg[q] = SCR / f"{NAME}-{q}.gguf"
        run([binp("crispasr-quantize"), gg["f16"], gg[q], q], OUT / f"quantize-{q}.log")
    for q, p in gg.items():
        res["quants"][q] = {"bytes": p.stat().st_size, "sha256": sha256(p)}
    if "--export-gguf" in sys.argv[1:]:
        import shutil
        (OUT / "gguf").mkdir(exist_ok=True)
        for p in gg.values():
            shutil.copy2(p, OUT / "gguf" / p.name)

    failures = []
    for name, sample in CASES:
        c = res["cases"].setdefault(name, {"sample": sample})
        ref = SCR / f"ref-{name}.gguf"
        run([sys.executable, ROOT / "tools/dump_reference.py", "--backend", "qwen3", "--model-dir", md,
             "--audio", ROOT / sample, "--output", ref, "--max-new-tokens", "256"],
            OUT / f"ref-{name}.log", env={"QWEN3_REF_DTYPE": "float32"})
        rt = ref_text(ref)
        c["reference_text"] = rt
        for q in QUANTS:
            r = c.setdefault(q, {})
            d = run([binp("crispasr-diff"), "qwen3", gg[q], ref, ROOT / sample], OUT / f"diff-{name}-{q}.log",
                    env={"CRISPASR_DIFF_NO_GPU": "1"}, check=False)
            rows = re.findall(r"^\[(PASS|FAIL|ERR )\]\s+(\S+)\s+(.*)$", d.stdout, re.M)
            r["diff"] = [{"status": s.strip(), "stage": st, "detail": dt.strip()} for s, st, dt in rows]
            r["diff_rc"] = d.returncode
            if q == "f16" and (d.returncode != 0 or any(s.strip() != "PASS" for s, _, _ in rows) or not rows):
                failures.append(f"{name}/{q}: diff")
            t = run([binp("crispasr"), "--backend", "qwen3", "-m", gg[q], "-f", ROOT / sample, "-nt", "-np"],
                    OUT / f"cli-{name}-{q}.log", check=False, merge=False)
            hyp = " ".join(l.strip() for l in t.stdout.splitlines() if l.strip())
            r["cli_text"] = hyp
            r["cli_rc"] = t.returncode
            r["cer_vs_reference"] = cer(hyp, rt) if rt else None
            if name == "jfk":
                r["wer_vs_human"] = wer(hyp, JFK)
                if r["wer_vs_human"] > 0.10:
                    failures.append(f"{name}/{q}: WER {r['wer_vs_human']:.3f}")
            if q == "f16" and r["cer_vs_reference"] != 0.0:
                failures.append(f"{name}/f16: transcript differs from the qwen_asr reference")
    res["failures"] = failures
    (OUT / "result.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))

    lines = [f"## Musci-ASR-qwen-1019 (#595) @ {REVISION[:12]}", "",
             "| clip | reference (qwen_asr fp32) | quant | CLI | CER vs ref | WER vs human |",
             "|---|---|---|---|---|---|"]
    for name, c in res["cases"].items():
        for q in QUANTS:
            r = c[q]
            w, x = r.get("wer_vs_human"), r["cer_vs_reference"]
            lines.append(f"| {name} | {c['reference_text'][:60]} | {q} | {r['cli_text'][:60]} | "
                         f"{x if x is None else round(x, 4)} | {'' if w is None else round(w, 4)} |")
    lines += ["", "### diff", ""]
    for name, c in res["cases"].items():
        for q in QUANTS:
            for row in c[q].get("diff", []):
                lines.append(f"- {name}/{q} [{row['status']}] {row['stage']} {row['detail']}")
    lines += ["", f"failures: {failures or 'none'}"]
    (OUT / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
