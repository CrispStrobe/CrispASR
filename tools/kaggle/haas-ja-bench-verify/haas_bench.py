#!/usr/bin/env python3
"""Verify Haas-ja-v1.0's published CER on two asr-ja-bench subsets (#605).

Rebuilds the FLEURS and JSUT subsets of third-intelligence/asr-ja-bench with
its own rebuild.py (which hash-checks every transcript), then scores the
original model with the upstream evaluation script
(Haas-ASR examples/evaluate_asr_ja_bench.py, jacer metrics v1.2). The model
card reports CER strict / flex / kana: FLEURS 3.67 / 2.98 / 2.34,
JSUT 5.32 / 2.43 / 1.36 (bf16, batches of 512; a T4 needs smaller batches,
which can flip a few near-tie tokens).

Outputs in /kaggle/working:
  summary.json, fleurs.jsonl, jsut.jsonl   upstream eval results + hypotheses
  fleurs_bench.zip                         FLEURS benchmark audio (FLAC) and
                                           references, to score the CrispASR
                                           port on the same utterances
  haas_bench.json                          step log
"""
import json
import os
import subprocess
import sys
import time
import traceback
import urllib.request
import zipfile
from pathlib import Path

os.environ["PYTHONUNBUFFERED"] = "1"
os.environ["USE_TF"] = "0"
WORK = Path("/kaggle/working")
TMP = Path("/tmp/haas-bench")  # large inputs stay off the 20 GB output dir
HAAS_ASR_REF = "acc614f"
JACER_REF = "b861959fe0a8f67f47c848d19c1a9cb8a32e1c2e"  # pinned by Haas-ASR's eval group
JSUT_URL = "https://ss-takashi.sakura.ne.jp/corpus/jsut_ver1.1.zip"
BATCH = 16
log = {"steps": {}}
t0 = time.time()


def save():
    (WORK / "haas_bench.json").write_text(json.dumps(log, ensure_ascii=False, indent=1))


def step(name, fn):
    print(f"[step] {name} t={time.time() - t0:.0f}s", flush=True)
    try:
        log["steps"][name] = {"ok": True, "value": fn()}
    except Exception:
        tb = traceback.format_exc()
        print(tb, flush=True)
        log["steps"][name] = {"ok": False, "error": tb[-4000:]}
    save()
    return log["steps"][name]["ok"]


def sh(*cmd, cwd=None):
    print("+", " ".join(map(str, cmd)), flush=True)
    subprocess.check_call(list(map(str, cmd)), cwd=cwd)


def gpu():
    return subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                          capture_output=True, text=True).stdout.strip()


def install():
    sh(sys.executable, "-m", "pip", "install", "-q", "pyarrow",
       f"haas-asr @ git+https://github.com/third-intelligence/Haas-ASR.git@{HAAS_ASR_REF}",
       f"jacer @ git+https://github.com/third-intelligence/jacer.git@{JACER_REF}")
    sh("git", "clone", "-q", "https://github.com/third-intelligence/Haas-ASR.git", TMP / "Haas-ASR")
    sh("git", "checkout", "-q", HAAS_ASR_REF, cwd=TMP / "Haas-ASR")
    return gpu()


def rebuild():
    from huggingface_hub import snapshot_download

    repo = Path(snapshot_download("third-intelligence/asr-ja-bench", repo_type="dataset",
                                  local_dir=str(TMP / "asr-ja-bench")))
    jsut = TMP / "jsut_ver1.1.zip"
    urllib.request.urlretrieve(JSUT_URL, jsut)
    sh(sys.executable, repo / "rebuild.py", "--subset", "fleurs", "--subset", "jsut",
       "--jsut", jsut, "--out", TMP / "bench")
    jsut.unlink()
    return sorted(p.name for p in (TMP / "bench").iterdir())


def evaluate():
    sh(sys.executable, TMP / "Haas-ASR" / "examples" / "evaluate_asr_ja_bench.py",
       "--model", "third-intelligence/Haas-ja-v1.0", "--bench", TMP / "bench",
       "--subset", "fleurs", "--subset", "jsut", "--batch-size", BATCH, "--out", WORK)
    return json.loads((WORK / "summary.json").read_text())


def export_fleurs():
    import soundfile as sf

    bench = TMP / "bench"
    rows = [json.loads(line) for line in open(bench / "fleurs.jsonl", encoding="utf-8") if line.strip()]
    with zipfile.ZipFile(WORK / "fleurs_bench.zip", "w", zipfile.ZIP_STORED) as z:
        z.writestr("fleurs.jsonl", "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
        for r in rows:
            x, sr = sf.read(bench / r["audio"], dtype="float32")
            tmp = TMP / "x.flac"
            sf.write(tmp, x, sr, format="FLAC", subtype="PCM_24")
            z.write(tmp, r["audio"].rsplit(".", 1)[0] + ".flac")
    return {"rows": len(rows), "bytes": (WORK / "fleurs_bench.zip").stat().st_size}


def main():
    TMP.mkdir(parents=True, exist_ok=True)
    if not step("install", install):
        return
    if not step("rebuild", rebuild):
        return
    step("evaluate", evaluate)
    step("export_fleurs", export_fleurs)
    log["total_seconds"] = round(time.time() - t0, 1)
    save()
    print("[done]", json.dumps({k: v["ok"] for k, v in log["steps"].items()}), flush=True)


if __name__ == "__main__":
    main()
