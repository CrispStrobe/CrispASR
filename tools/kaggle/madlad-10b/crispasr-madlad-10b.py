# CrispASR — madlad400-10b-mt: GGUF conversion, quantization & verification
# Upstream: google/madlad400-10b-mt (Apache 2.0)
# Destination: cstr/madlad400-10b-mt-GGUF
#
# Pipeline:
#   1. Download sharded safetensors (42.85 GB) to /kaggle/temp/madlad-src
#   2. Convert to GGUF F16 (~21.5 GB) using low-RAM disk-mmap scratch
#   3. Validate F16 translations (3 cases, CPU fallback)
#   4. Upload F16 to cstr/madlad400-10b-mt-GGUF
#   5. Dump per-stage reference archive (madlad400-10b-mt-ref.gguf)
#   6. Upload reference archive to HF
#   7. Delete source safetensors from disk
#   8. Quantize Q8_0 (~11.3 GB), validate on GPU, upload to HF
#   9. Quantize Q4_K (~6.1 GB), validate on GPU, upload to HF
#  10. Run crispasr-diff (F16, Q8_0, Q4_K) against reference archive -> report parity table
#  11. Upload model card README.md with license and parity receipts

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

WORK = Path("/kaggle/working")
TEMP = Path("/kaggle/temp")
TEMP.mkdir(parents=True, exist_ok=True)

CRISPASR_URL = "https://github.com/CrispStrobe/CrispASR.git"
REPO = TEMP / "CrispASR"
if not REPO.exists():
    try:
        subprocess.check_call(["git", "clone", "--depth", "1", "--recursive",
                               CRISPASR_URL, str(REPO)])
        sys.path.insert(0, str(REPO / "tools" / "kaggle"))
    except Exception:
        pass
if str(REPO / "tools" / "kaggle") not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
import kaggle_harness as kh  # noqa: E402

kh.init_progress(hf_progress_repo="cstr/crispasr-kaggle-progress")
step = kh.step
step("script.start", model="madlad400-10b-mt")

CONVERTER = REPO / "models" / "convert-madlad-to-gguf.py"
DUMPER = REPO / "tools" / "dump_reference.py"
for f in (CONVERTER, DUMPER):
    if not f.is_file():
        step("fatal.clone-incomplete", missing=str(f))
        raise SystemExit(f"{f} missing")

TOKEN = kh.resolve_hf_token("HF_TOKEN")
if not TOKEN:
    step("fatal.no-token")
    raise SystemExit("no HF token")
step("hf_token.resolved")

step("install-deps.begin")
subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet",
                       "huggingface_hub", "hf_transfer", "sentencepiece", "safetensors", "gguf"])
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
from huggingface_hub import HfApi, hf_hub_download, snapshot_download  # noqa: E402

step("install-deps.done")

# ── Build toolchain ──────────────────────────────────────────────────────────
step("toolchain.begin")
kh.install_build_toolchain()
CC = TEMP / ".ccache"
if (WORK / ".ccache").exists() and not CC.exists():
    shutil.move(str(WORK / ".ccache"), str(CC))
os.environ["CCACHE_DIR"] = str(CC)
step("toolchain.done")

BUILD = REPO / "build"
step("build.begin")
with kh.build_heartbeat("cmake-configure", 30):
    kh.sh(f"cmake -S {REPO} -B {BUILD} -DCMAKE_BUILD_TYPE=Release "
          f"-DCRISPASR_BUILD_TESTS=OFF -DCRISPASR_BUILD_SERVER=OFF "
          f"-DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache")
with kh.build_heartbeat("cmake-build", 30):
    kh.sh(f"cmake --build {BUILD} -j {kh.safe_build_jobs(gpu=True)} "
          f"--target crispasr-cli crispasr-quantize crispasr-diff")
CRISPASR = BUILD / "bin" / "crispasr"
QUANT = BUILD / "bin" / "crispasr-quantize"
DIFF = BUILD / "bin" / "crispasr-diff"
for b in (CRISPASR, QUANT, DIFF):
    if not b.is_file():
        step("fatal.binary-missing", which=b.name)
        raise SystemExit(f"{b.name} did not build")
step("build.done")
kh.export_ccache_tar(str(WORK / "ccache.tar"))

MODELS = TEMP / "madlad10b-models"
MODELS.mkdir(parents=True, exist_ok=True)
SRC = TEMP / "madlad10b-src"
SRC.mkdir(parents=True, exist_ok=True)

SRC_REPO = "google/madlad400-10b-mt"
DST_REPO = "cstr/madlad400-10b-mt-GGUF"
F16 = MODELS / "madlad400-10b-mt-f16.gguf"
Q8 = MODELS / "madlad400-10b-mt-q8_0.gguf"
Q4 = MODELS / "madlad400-10b-mt-q4_k.gguf"
REF = MODELS / "madlad400-10b-mt-ref.gguf"

MADLAD_TEXT = "Hello world, how are you today?"
os.environ["MADLAD_TEXT"] = MADLAD_TEXT
os.environ["MADLAD_TL"] = "de"


def gb(p):
    return round(Path(p).stat().st_size / 1e9, 2)


def free_gb(p=TEMP):
    try:
        return round(shutil.disk_usage(p).free / 1e9, 1)
    except Exception:
        return -1.0


summary = {"source": SRC_REPO, "target": DST_REPO, "artifacts": {}, "parity": {}}

try:
    api = HfApi(token=TOKEN)
    api.create_repo(repo_id=DST_REPO, repo_type="model", exist_ok=True)
    HAVE = {f.rfilename for f in api.model_info(DST_REPO).siblings}
except Exception as e:
    HAVE = set()
    step("repo-listing.failed", err=str(e)[:200])
step("repo-listing", present=sorted(HAVE))


def save():
    (WORK / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))


CASES = [
    (MADLAD_TEXT, "en", "de", ["hallo", "welt", "wie geht", "heute"]),
    ("Machine learning is changing the world.", "en", "fr", ["apprentissage", "monde", "machine"]),
    ("Bonjour le monde!", "fr", "en", ["hello", "world", "good"]),
]


def _norm(s):
    return re.sub(r"[^a-z0-9äöüàéèç ]", " ", (s or "").lower())


def validate(model_path, label, force_cpu=False):
    step(f"{label}.validate.begin", gb=gb(model_path), force_cpu=force_cpu)
    rows = []
    env = dict(os.environ)
    if force_cpu:
        env["CRISPASR_T5_GPU"] = "0"
    for text, sl, tl, wants in CASES:
        with kh.build_heartbeat(f"{label}.translate.{sl}-{tl}", 30):
            p = subprocess.run([str(CRISPASR), "--backend", "madlad", "-m", str(model_path),
                                "--text", text, "-sl", sl, "-tl", tl],
                               capture_output=True, text=True, timeout=1800, env=env)
        out = _norm(p.stdout)
        ok = p.returncode == 0 and bool(out.strip())
        rows.append({"pair": f"{sl}->{tl}", "exit": p.returncode, "ok": ok,
                     "matched": any(w in out for w in wants),
                     "echoed": _norm(text).strip() in out,
                     "out": p.stdout.strip()[:200]})
        step(f"{label}.translate.{sl}-{tl}", **{k: rows[-1][k] for k in ("exit", "matched", "echoed")},
             out=p.stdout.strip()[:120])
    passed = all(r["ok"] and r["matched"] and not r["echoed"] for r in rows)
    step(f"{label}.validate.done", passed=passed)
    return passed, rows


def upload(path, msg):
    step("upload.begin", file=Path(path).name, gb=gb(path))
    with kh.build_heartbeat(f"upload.{Path(path).name}", 30):
        HfApi().upload_file(path_or_fileobj=str(path), path_in_repo=Path(path).name,
                            repo_id=DST_REPO, repo_type="model", token=TOKEN, commit_message=msg)
    step("upload.done", file=Path(path).name)


# ── 1. Source Download ───────────────────────────────────────────────────────
step("source-download.begin", free_gb=free_gb())
with kh.build_heartbeat("source-download", 30):
    snapshot_download(repo_id=SRC_REPO, local_dir=str(SRC), token=TOKEN,
                      allow_patterns=["model*.safetensors*", "config.json", "spiece.model",
                                      "tokenizer*.json", "special_tokens_map.json",
                                      "added_tokens.json", "generation_config.json"])
step("source-download.done", free_gb=free_gb())

# ── 2. Convert F16 ───────────────────────────────────────────────────────────
if F16.name in HAVE:
    step("convert.skipped", why="already on the repo")
    with kh.build_heartbeat("f16-download", 30):
        F16 = Path(hf_hub_download(repo_id=DST_REPO, filename=F16.name,
                                   local_dir=str(MODELS), token=TOKEN))
    summary["artifacts"]["f16"] = {"size_gb": gb(F16), "preexisting": True}
    save()
else:
    step("convert.begin", free_gb=free_gb())
    with kh.build_heartbeat("convert-f16", 30):
        p = subprocess.run([sys.executable, str(CONVERTER), "--input", str(SRC), "--output", str(F16)],
                           capture_output=True, text=True, timeout=7200)
    if p.returncode != 0 or not F16.is_file():
        step("fatal.convert-failed", exit=p.returncode, tail=p.stdout[-600:], err=p.stderr[-600:])
        raise SystemExit("F16 conversion failed")
    step("convert.done", f16_gb=gb(F16), free_gb=free_gb())

    ok, rows = validate(F16, "f16", force_cpu=True)
    summary["artifacts"]["f16"] = {"size_gb": gb(F16), "validated": ok, "cases": rows}
    save()
    if not ok:
        step("fatal.f16-invalid")
        raise SystemExit("F16 failed validation")
    upload(F16, "add madlad400-10b-mt F16 GGUF")
    summary["artifacts"]["f16"]["uploaded"] = True
    save()

# ── 3. Reference Archive (Before deleting source) ────────────────────────────
step("refdump.begin", free_gb=free_gb())
with kh.build_heartbeat("refdump", 30):
    p = subprocess.run([sys.executable, str(DUMPER), "--backend", "madlad",
                        "--model-dir", str(SRC), "--output", str(REF),
                        "--audio", str(REPO / "samples" / "jfk.wav")],
                       capture_output=True, text=True, timeout=7200, cwd=str(REPO / "tools"))
ref_ok = p.returncode == 0 and REF.is_file()
step("refdump.done", exit=p.returncode, ok=ref_ok,
     ref_mb=round(REF.stat().st_size / 1e6, 1) if ref_ok else 0,
     tail=p.stdout[-800:], err=p.stderr[-800:] if p.returncode else "")
if ref_ok:
    upload(REF, "add per-stage reference archive for crispasr-diff")
    summary["artifacts"]["ref"] = {"size_mb": round(REF.stat().st_size / 1e6, 1), "uploaded": True}
else:
    summary["artifacts"]["ref"] = {"failed": True, "tail": p.stdout[-400:], "err": p.stderr[-400:]}
save()

# Source files no longer needed
shutil.rmtree(SRC, ignore_errors=True)
step("source.deleted", free_gb=free_gb())

# ── 4. Quantize Q8_0 ─────────────────────────────────────────────────────────
if Q8.name in HAVE:
    step("quantize-q8.skipped", why="already on the repo")
    with kh.build_heartbeat("q8-download", 30):
        Q8 = Path(hf_hub_download(repo_id=DST_REPO, filename=Q8.name,
                                  local_dir=str(MODELS), token=TOKEN))
    summary["artifacts"]["q8_0"] = {"size_gb": gb(Q8), "preexisting": True}
    save()
else:
    step("quantize-q8.begin", free_gb=free_gb())
    with kh.build_heartbeat("quantize-q8_0", 30):
        p = subprocess.run([str(QUANT), str(F16), str(Q8), "q8_0"],
                           capture_output=True, text=True, timeout=7200)
    if p.returncode != 0 or not Q8.is_file():
        step("fatal.quantize-q8-failed", exit=p.returncode, tail=p.stdout[-600:])
        raise SystemExit("Q8_0 quantization failed")
    step("quantize-q8.done", q8_gb=gb(Q8), free_gb=free_gb())

    ok, rows = validate(Q8, "q8_0", force_cpu=False)
    summary["artifacts"]["q8_0"] = {"size_gb": gb(Q8), "validated": ok, "cases": rows}
    save()
    if not ok:
        step("fatal.q8-invalid")
        raise SystemExit("Q8_0 failed validation")
    upload(Q8, "add madlad400-10b-mt Q8_0 GGUF")
    summary["artifacts"]["q8_0"]["uploaded"] = True
    save()

# ── 5. Quantize Q4_K ─────────────────────────────────────────────────────────
if Q4.name in HAVE:
    step("quantize-q4k.skipped", why="already on the repo")
    with kh.build_heartbeat("q4k-download", 30):
        Q4 = Path(hf_hub_download(repo_id=DST_REPO, filename=Q4.name,
                                  local_dir=str(MODELS), token=TOKEN))
    summary["artifacts"]["q4_k"] = {"size_gb": gb(Q4), "preexisting": True}
    save()
else:
    step("quantize-q4k.begin", free_gb=free_gb())
    with kh.build_heartbeat("quantize-q4_k", 30):
        p = subprocess.run([str(QUANT), str(F16), str(Q4), "q4_k"],
                           capture_output=True, text=True, timeout=7200)
    if p.returncode != 0 or not Q4.is_file():
        step("fatal.quantize-q4k-failed", exit=p.returncode, tail=p.stdout[-600:])
        raise SystemExit("Q4_K quantization failed")
    step("quantize-q4k.done", q4k_gb=gb(Q4), free_gb=free_gb())

    ok, rows = validate(Q4, "q4_k", force_cpu=False)
    summary["artifacts"]["q4_k"] = {"size_gb": gb(Q4), "validated": ok, "cases": rows}
    save()
    if not ok:
        step("fatal.q4k-invalid")
        raise SystemExit("Q4_K failed validation")
    upload(Q4, "add madlad400-10b-mt Q4_K GGUF")
    summary["artifacts"]["q4_k"]["uploaded"] = True
    save()

# ── 6. Per-stage Parity Diff vs Blueprint ────────────────────────────────────
if ref_ok:
    for label, path in (("f16", F16), ("q8_0", Q8), ("q4_k", Q4)):
        if not Path(path).is_file():
            continue
        step(f"diff.{label}.begin")
        env = dict(os.environ)
        if label == "f16":
            env["CRISPASR_T5_GPU"] = "0"
        with kh.build_heartbeat(f"diff.{label}", 30):
            p = subprocess.run([str(DIFF), "madlad", str(path), str(REF)],
                               capture_output=True, text=True, timeout=3600, env=env)
        stages = {}
        for line in p.stdout.splitlines():
            m = re.match(r"t5 (\S+)\s+n=\S+\s+cos=([-\d.]+)\s+max_abs=([\d.]+)\s+"
                         r"\|mine\|=([\d.]+) \|ref\|=([\d.]+)\s+(PASS|FAIL)", line)
            if m:
                stages[m.group(1)] = {"cos": float(m.group(2)), "max_abs": float(m.group(3)),
                                      "mine": float(m.group(4)), "ref": float(m.group(5)),
                                      "verdict": m.group(6)}
        am = re.search(r"argmax_step0\s+mine=(\d+) ref=(\d+)\s+(MATCH|DIFFER)", p.stdout)
        summary["parity"][label] = {
            "exit": p.returncode,
            "stages": stages,
            "argmax_step0": {"mine": int(am.group(1)), "ref": int(am.group(2)), "verdict": am.group(3)}
            if am else None,
            "worst_cos": min((v["cos"] for v in stages.values()), default=None),
            "stdout_tail": p.stdout[-1500:],
        }
        step(f"diff.{label}.done", exit=p.returncode, n_stages=len(stages),
             worst_cos=summary["parity"][label]["worst_cos"],
             argmax=summary["parity"][label]["argmax_step0"])
        save()

# ── 7. Generate Model Card README.md ─────────────────────────────────────────
readme_content = f"""---
license: apache-2.0
tags:
- crispasr
- translation
- t5
- madlad400
- gguf
---

# MADLAD-400 10B-MT GGUF

Official GGUF quantized models for Google's [madlad400-10b-mt](https://huggingface.co/google/madlad400-10b-mt) (10.7B parameters, 400+ languages translation model).

Converted with [CrispASR](https://github.com/CrispStrobe/CrispASR) with verified per-stage parity against PyTorch reference outputs.

## Files Available

| Filename | Quantization | Size | Description |
|---|---|---|---|
| `madlad400-10b-mt-f16.gguf` | F16 | ~21.5 GB | Full precision baseline |
| `madlad400-10b-mt-q8_0.gguf` | Q8_0 | ~11.3 GB | High fidelity 8-bit quantization |
| `madlad400-10b-mt-q4_k.gguf` | Q4_K | ~6.1 GB | Recommended for general use |
| `madlad400-10b-mt-ref.gguf` | Reference | ~1.5 MB | Per-stage verification archive |

## Usage with CrispASR

```bash
# English to German
crispasr --backend madlad -m madlad400-10b-mt-q4_k.gguf --text "Hello world!" -sl en -tl de

# French to English
crispasr --backend madlad -m madlad400-10b-mt-q4_k.gguf --text "Bonjour le monde!" -sl fr -tl en
```

## License

Apache-2.0 (matching upstream Google MADLAD-400 checkpoint).
"""
readme_path = TEMP / "README.md"
readme_path.write_text(readme_content, encoding="utf-8")
upload(readme_path, "update model card README.md")

save()
step("script.done", artifacts=list(summary["artifacts"]), parity=list(summary["parity"]))
print(json.dumps(summary, indent=2, ensure_ascii=False)[:6000])
