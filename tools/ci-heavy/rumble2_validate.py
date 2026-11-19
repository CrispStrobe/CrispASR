#!/usr/bin/env python3
"""#602 Reson8/Rumble-2-Flash: convert -> quantize -> transformers stage diff -> WER.

Runs on heavy-cpu.yml (the 8 GB maintainer VPS cannot hold the fp32
reference next to a build). Steps:

  1. build crispasr, crispasr-diff, crispasr-quantize (CPU, Release)
  2. download Reson8/Rumble-2-Flash at a pinned revision, verify the
     safetensors sha256 against the HF x-linked-etag
  3. convert to F16 with models/convert-qwen3-asr-to-gguf.py, then
     crispasr-quantize to q8_0 and q4_k
  4. per case (jfk.wav English; ko-369.wav Korean; paraformer_zh.wav Chinese;
     each auto-detect and forced-language): fp32 transformers reference dump
     (tools/dump_reference.py --backend rumble2, the repo's own chat template)
  5. crispasr-diff qwen3 on every quant x case; the crispasr CLI transcript
     on every quant x case; WER (jfk vs the human transcript) and CER vs the
     transformers transcript

Fails (exit 1) when a F16 diff stage fails, a F16 transcript differs from
transformers, or JFK WER exceeds 0.10 on any quant.

  gh workflow run heavy-cpu.yml -f script=tools/ci-heavy/rumble2_validate.py \
     -f pip="torch numpy soundfile gguf==0.17.1 safetensors transformers>=5.5 huggingface_hub librosa" \
     -f timeout_minutes=240 --ref <branch>
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import unicodedata
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REPO = "Reson8/Rumble-2-Flash"
REVISION = os.environ.get("RUMBLE_REVISION", "a0c34287229103b706c670d70791683e34379732")
OUT = Path(os.environ.get("HEAVY_OUT", ROOT / "rumble2-out"))
SCR = Path(os.environ.get("HEAVY_SCRATCH", ROOT / "rumble2-scratch"))
JFK = "And so, my fellow Americans, ask not what your country can do for you, ask what you can do for your country."
CASES = [  # name, sample, language (English name for the template, ISO for the CLI)
    ("jfk-auto", "samples/jfk.wav", None, None),
    ("jfk-en", "samples/jfk.wav", "English", "en"),
    ("ko-auto", "samples/ko-369.wav", None, None),
    ("ko-ko", "samples/ko-369.wav", "Korean", "ko"),
    ("zh-auto", "samples/paraformer_zh.wav", None, None),
    ("zh-zh", "samples/paraformer_zh.wav", "Chinese", "zh"),
]
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
    res = {"repo": REPO, "revision": REVISION, "cases": {}, "quants": {}}

    # 1. build
    bd = SCR / "build"
    run(["cmake", "-S", ROOT, "-B", bd, "-G", "Ninja", "-DCMAKE_BUILD_TYPE=Release", "-DGGML_NATIVE=OFF"],
        OUT / "cmake.log")
    run(["cmake", "--build", bd, "-j", str(os.cpu_count() or 4), "--target", "crispasr", "crispasr-diff",
         "crispasr-quantize"], OUT / "build.log")
    binp = lambda n: next(p for p in (bd / "bin" / n, bd / n) if p.exists())

    # 2. download + verify
    from huggingface_hub import hf_hub_download, snapshot_download
    md = Path(snapshot_download(REPO, revision=REVISION, local_dir=SCR / "hf",
                                allow_patterns=["*.json", "*.txt", "*.jinja", "model.safetensors", "LICENSE"]))
    req = urllib.request.Request(f"https://huggingface.co/{REPO}/resolve/{REVISION}/model.safetensors", method="HEAD")
    etag = None
    try:
        class NoRedirect(urllib.request.HTTPRedirectHandler):
            def redirect_request(self, *a, **k):
                return None
        urllib.request.build_opener(NoRedirect).open(req)
    except urllib.error.HTTPError as e:
        etag = (e.headers.get("x-linked-etag") or "").strip('"')
    got = sha256(md / "model.safetensors")
    res["safetensors_sha256"] = got
    res["safetensors_etag"] = etag
    if etag and etag != got:
        raise SystemExit(f"safetensors sha256 {got} != HF etag {etag}")

    # 3. convert + quantize
    gg = {}
    gg["f16"] = SCR / "rumble-2-flash-f16.gguf"
    run([sys.executable, ROOT / "models/convert-qwen3-asr-to-gguf.py", "--input", md, "--output", gg["f16"]],
        OUT / "convert.log")
    for q in QUANTS[1:]:
        gg[q] = SCR / f"rumble-2-flash-{q}.gguf"
        run([binp("crispasr-quantize"), gg["f16"], gg[q], q], OUT / f"quantize-{q}.log")
    for q, p in gg.items():
        res["quants"][q] = {"bytes": p.stat().st_size, "sha256": sha256(p)}
    # --export-gguf: ship the validated files in the run artifact, so the
    # maintainer uploads exactly the bytes this run checked (sha256 above).
    if "--export-gguf" in sys.argv[1:]:
        import shutil
        (OUT / "gguf").mkdir(exist_ok=True)
        for p in gg.values():
            shutil.copy2(p, OUT / "gguf" / p.name)

    failures = []
    for name, sample, lang_name, lang_iso in CASES:
        c = res["cases"].setdefault(name, {"sample": sample, "language": lang_name})
        ref = SCR / f"ref-{name}.gguf"
        env = {"RUMBLE_REF_LANGUAGE": lang_name or ""}
        p = run([sys.executable, ROOT / "tools/dump_reference.py", "--backend", "rumble2", "--model-dir", md,
                 "--audio", ROOT / sample, "--output", ref, "--max-new-tokens", "256",
                 "--stages", "mel_spectrogram,conv_out,enc_blk00_out,ln_post_out,encoder_output,trace_input_ids,"
                 "trace_audio_pad_pos,trace_inputs_embeds,trace_first_logits,generated_text"],
                OUT / f"ref-{name}.log", env=env)
        m = re.search(r"transcription: '(.*)'\s*$", p.stdout, re.M) or re.search(r'transcription: "(.*)"\s*$',
                                                                                    p.stdout, re.M)
        ref_text = m.group(1) if m else ""
        c["reference_text"] = ref_text
        for q in QUANTS:
            r = c.setdefault(q, {})
            # Stage diff: the auto-detect prompt is what crispasr-diff builds.
            if lang_name is None:
                d = run([binp("crispasr-diff"), "qwen3", gg[q], ref, ROOT / sample], OUT / f"diff-{name}-{q}.log",
                        env={"CRISPASR_DIFF_NO_GPU": "1"}, check=False)
                rows = re.findall(r"^\[(PASS|FAIL|ERR )\]\s+(\S+)\s+(.*)$", d.stdout, re.M)
                r["diff"] = [{"status": s.strip(), "stage": st, "detail": dt.strip()} for s, st, dt in rows]
                r["diff_rc"] = d.returncode
                if q == "f16" and (d.returncode != 0 or any(s.strip() != "PASS" for s, _, _ in rows) or not rows):
                    failures.append(f"{name}/{q}: diff")
            cmd = [binp("crispasr"), "--backend", "qwen3", "-m", gg[q], "-f", ROOT / sample, "-nt", "-np"]
            if lang_iso:
                cmd += ["-l", lang_iso]
            t = run(cmd, OUT / f"cli-{name}-{q}.log", check=False, merge=False)
            hyp = " ".join(l.strip() for l in t.stdout.splitlines() if l.strip())
            r["cli_text"] = hyp
            r["cli_rc"] = t.returncode
            r["cer_vs_transformers"] = cer(hyp, ref_text) if ref_text else None
            if sample.endswith("jfk.wav"):
                r["wer_vs_human"] = wer(hyp, JFK)
                if r["wer_vs_human"] > 0.10:
                    failures.append(f"{name}/{q}: WER {r['wer_vs_human']:.3f}")
            if q == "f16" and r["cer_vs_transformers"] not in (None, 0.0):
                failures.append(f"{name}/f16: transcript differs from transformers")
    res["failures"] = failures
    (OUT / "result.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))

    lines = [f"## Rumble-2-Flash (#602) @ {REVISION[:12]}", "",
             "| case | reference (transformers fp32) | quant | CLI | CER vs ref | WER vs human |", "|---|---|---|---|---|---|"]
    for name, c in res["cases"].items():
        for q in QUANTS:
            r = c[q]
            w = r.get("wer_vs_human")
            lines.append(f"| {name} | {c['reference_text'][:60]} | {q} | {r['cli_text'][:60]} | "
                         f"{r['cer_vs_transformers'] if r['cer_vs_transformers'] is None else round(r['cer_vs_transformers'], 4)} | "
                         f"{'' if w is None else round(w, 4)} |")
    lines += ["", "### diff (auto-detect prompt)", ""]
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
