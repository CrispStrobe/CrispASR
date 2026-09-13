#!/usr/bin/env python3
"""#434 Supertonic-3: convert -> ref dump -> build -> per-stage diff -> TTS->ASR roundtrip.

Verdict gates (results.json "validation"):
  1. crispasr-diff supertonic-tts reports ALL PASS (per-stage cos + norms vs ORT).
  2. CPU-synthesised wav transcribes with word overlap >= 0.8 vs the input text.
  3. CONTROL: the upstream ONNX reference wav passes the SAME ASR at >= 0.8
     (else the ASR arm, not the port, is at fault and the run is inconclusive).
  4. GPU-synthesised wav passes the same transcript gate (CUDA correctness).
SCRIPT_VERSION = v4
"""

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

WORK = Path("/kaggle/working")
TEMP = Path("/kaggle/temp/supertonic-434")
REPO = Path("/kaggle/temp/CrispASR")
MODEL_DIR = TEMP / "supertonic-3"
for p in (WORK, TEMP, MODEL_DIR):
    p.mkdir(parents=True, exist_ok=True)

BRANCH = "feat/434-supertonic"
TEXT = "The quick brown fox jumps over the lazy dog."
LANG, VOICE, STEPS, SEED = "en", "M1", 8, 1234


def run(argv, *, cwd=None, env=None, timeout=7200, capture=False, check=True):
    merged = os.environ.copy()
    if env:
        merged.update({str(k): str(v) for k, v in env.items()})
    print("$ " + " ".join(map(str, argv)), flush=True)
    try:
        return subprocess.run([str(x) for x in argv], cwd=cwd, env=merged, check=check,
                              timeout=timeout, text=True, capture_output=capture)
    except subprocess.CalledProcessError as e:
        # ninja/compilers write errors to stderr — print BOTH streams always.
        if e.stdout:
            print("== stdout ==\n" + e.stdout[-8000:], flush=True)
        if e.stderr:
            print("== stderr ==\n" + e.stderr[-8000:], flush=True)
        raise


if not REPO.exists():
    run(["git", "clone", "--depth", "1", "--branch", BRANCH, "--recursive",
         "https://github.com/CrispStrobe/CrispASR.git", REPO], timeout=2400)
sys.path.insert(0, str(REPO / "tools" / "kaggle"))
import kaggle_harness as kh  # noqa: E402

kh.init_progress()
kh.resolve_hf_token()
commit = subprocess.check_output(["git", "-C", REPO, "rev-parse", "HEAD"], text=True).strip()
print(f"SCRIPT_VERSION=v4 clone={commit}", flush=True)
kh.step("provenance", commit=commit)

kh.step("dependencies")
run([sys.executable, "-m", "pip", "install", "--quiet", "onnx", "onnxruntime", "gguf", "soundfile"])

kh.step("model.download")
run([sys.executable, "-c", (
    "from huggingface_hub import snapshot_download; "
    f"snapshot_download('Supertone/supertonic-3', local_dir={str(MODEL_DIR)!r}, "
    "allow_patterns=['onnx/*','voice_styles/*','LICENSE','README.md'])"
)], timeout=3600)

kh.step("convert.f16")
f16 = WORK / "supertonic3-f16.gguf"
run([sys.executable, REPO / "models/convert-supertonic3-to-gguf.py", "--model", MODEL_DIR,
     "--output", f16, "--dtype", "f16"], timeout=3600)

# checkpoint the artifact the moment it exists
token = os.environ.get("HF_TOKEN")
if token:
    from huggingface_hub import HfApi  # noqa: E402
    api = HfApi(token=token)
    api.create_repo("cstr/supertonic-3-GGUF", repo_type="model", private=False, exist_ok=True)
    kh.step("publish.f16")
    api.upload_file(path_or_fileobj=str(f16), path_in_repo="supertonic3-f16.gguf",
                    repo_id="cstr/supertonic-3-GGUF", repo_type="model",
                    commit_message=f"Supertonic-3 F16 GGUF from {commit[:12]} (pre-validation checkpoint)")

kh.step("reference.dump")
ref = WORK / "supertonic-ref.gguf"
run([sys.executable, REPO / "tools/reference_backends/supertonic_tts.py", "--model-dir", MODEL_DIR,
     "--text", TEXT, "--lang", LANG, "--voice", VOICE, "--steps", str(STEPS), "--seed", str(SEED),
     "--output", ref], timeout=3600)

# extract the reference audio stage as the CONTROL wav
control_wav = WORK / "control_upstream.wav"
run([sys.executable, "-c", f'''
import numpy as np, soundfile as sf
from gguf import GGUFReader
r = GGUFReader({str(ref)!r})
for t in r.tensors:
    if t.name == "audio":
        sf.write({str(control_wav)!r}, np.array(t.data, dtype=np.float32), 44100)
        print("control wav:", t.data.shape)
        break
else:
    raise SystemExit("no audio stage in ref")
'''])

kh.install_build_toolchain()
build = TEMP / "build-cuda"
arch = kh.detect_cuda_arch()
flags = ["-G", "Ninja", "-DCMAKE_BUILD_TYPE=Release", "-DCRISPASR_NO_C2PA_NATIVE=ON",
         "-DGGML_NATIVE=OFF", *kh.cuda_build_flags(arch), *kh.cache_and_link_flags()]
kh.step("build", arch=arch)
run(["cmake", "-S", REPO, "-B", build, *flags], timeout=2400, capture=True)
run(["cmake", "--build", build, "-j", str(kh.safe_build_jobs(gpu=True)), "--target",
     "crispasr-cli", "crispasr-diff"], timeout=7200, capture=True)
cli = build / "bin/crispasr"
diff = build / "bin/crispasr-diff"
if not cli.exists() or not diff.exists():
    raise RuntimeError("build produced no binaries (proof-of-work check)")

results = {"commit": commit, "cuda_arch": arch, "script_version": "v4"}

kh.step("diff.stages")
p = run([diff, "supertonic-tts", f16, ref, REPO / "samples/jfk.wav"], capture=True, check=False)
print(p.stdout, flush=True)
if p.stderr:
    print("== diff stderr ==\n" + p.stderr[-4000:], flush=True)
results["diff_output"] = p.stdout
results["diff_all_pass"] = "ALL PASS" in p.stdout
(WORK / "diff_output.txt").write_text(p.stdout + "\n--- stderr ---\n" + (p.stderr or ""))

def synth(label, extra):
    out = WORK / f"st_{label}.wav"
    t0 = time.perf_counter()
    pr = run([cli, "--backend", "supertonic", "-m", f16, "--tts", TEXT, "-l", LANG,
              "--voice", VOICE, "-o", out, *extra], capture=True, check=False)
    el = time.perf_counter() - t0
    print(pr.stdout[-2000:] if pr.stdout else "", flush=True)
    if pr.stderr:
        print(f"== synth {label} stderr ==\n" + pr.stderr[-4000:], flush=True)
    ok = pr.returncode == 0 and out.exists() and out.stat().st_size > 40000
    return out, ok, el

kh.step("synth.cpu")
wav_cpu, cpu_ok, cpu_s = synth("cpu", ["--no-gpu"])
kh.step("synth.gpu")
wav_gpu, gpu_ok, gpu_s = synth("gpu", [])
results["synth"] = {"cpu_ok": cpu_ok, "cpu_s": cpu_s, "gpu_ok": gpu_ok, "gpu_s": gpu_s}

def asr(wav):
    if not wav.exists():
        return ""
    pr = run([cli, "--backend", "whisper", "-m", "auto", "-f", wav, "-nt", "--no-gpu"],
             capture=True, check=False, timeout=3600)
    txt = (pr.stdout or "") + " " + (pr.stderr or "")
    return re.sub(r"\x1b\[[0-9;]*m", "", txt)

def overlap(full_out, target):
    words = [re.sub(r"[^a-z']", "", w.lower()) for w in target.split()]
    words = [w for w in words if w]
    low = full_out.lower()
    hit = sum(1 for w in words if w in low)
    return hit / max(1, len(words))

kh.step("asr.roundtrip")
tr_control = asr(control_wav)
tr_cpu = asr(wav_cpu) if cpu_ok else ""
tr_gpu = asr(wav_gpu) if gpu_ok else ""
ov_control = overlap(tr_control, TEXT)
ov_cpu = overlap(tr_cpu, TEXT)
ov_gpu = overlap(tr_gpu, TEXT)
results["asr"] = {
    "control_overlap": ov_control, "cpu_overlap": ov_cpu, "gpu_overlap": ov_gpu,
    "control_tail": tr_control[-600:], "cpu_tail": tr_cpu[-600:], "gpu_tail": tr_gpu[-600:],
}

errors = []
if not results["diff_all_pass"]:
    errors.append("per-stage diff not ALL PASS")
if ov_control < 0.8:
    errors.append(f"CONTROL failed ASR ({ov_control:.2f}) — ASR arm broken, run inconclusive")
else:
    if not cpu_ok or ov_cpu < 0.8:
        errors.append(f"CPU synth failed roundtrip (ok={cpu_ok} overlap={ov_cpu:.2f})")
    if not gpu_ok or ov_gpu < 0.8:
        errors.append(f"GPU synth failed roundtrip (ok={gpu_ok} overlap={ov_gpu:.2f})")
results["validation"] = {"passed": not errors, "errors": errors}

(WORK / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n")
print(json.dumps({k: v for k, v in results.items() if k != "diff_output"},
                 ensure_ascii=False, indent=2), flush=True)

if not errors and token:
    kh.step("publish.ref")
    api.upload_file(path_or_fileobj=str(ref), path_in_repo="supertonic-tts/fox-m1/ref.gguf",
                    repo_id="cstr/crispasr-regression-fixtures", repo_type="dataset",
                    commit_message=f"Supertonic-3 reference dump from {commit[:12]}")
    api.upload_file(path_or_fileobj=str(WORK / "results.json"), path_in_repo="validation/results-434.json",
                    repo_id="cstr/supertonic-3-GGUF", repo_type="model",
                    commit_message="Supertonic-3 Kaggle validation results")

kh.step("done", passed=not errors)
if errors:
    raise RuntimeError("; ".join(errors))
