#!/usr/bin/env python3
"""Kaggle kernel: Hojo-ASR-Multi-V1 (#438) — the whole port pipeline in one run.

convert -> upload f16 -> build -> reference dump -> upload ref -> diff(f16)
-> quantize -> upload q4_k -> diff(q4_k) -> end-to-end CLI transcript.

Everything is uploaded the moment it exists, so a late failure never loses an
artifact that cost compute to produce.

Why one CPU kernel and not the usual Kaggle-produces / local-validates split:
the q4_k is ~4.4 GB (the tied 151670x2560 embedding and the whole audio tower
stay F16), and the dev VPS has 8 GB of RAM shared with other agents. There is
no local arm for this model, so the diff loop lives here.

Two readouts that can actually report failure
---------------------------------------------
1. CONTROL ARM. The upstream `hojo-asr` package transcribes the same audio
   first and its text is printed BEFORE any C++ runs. If the reference itself
   comes back wrong, nothing downstream is evidence about the port.
2. CONV TILING A/B. The C++ conv stem is tiled along time with an 8-frame halo,
   which is exact by construction. `CRISPASR_HOJO_ASR_CONV_TILE=0` runs the
   untiled path; the two encoder outputs are compared byte-for-byte AND by
   cosine + magnitude. A tiling bug shows up here even if it happens to leave
   the transcript readable.

Push (chr1s4):
  export KAGGLE_API_TOKEN=<chr1s4 token>
  python -m kaggle kernels push -p tools/kaggle/hojo-asr-438
"""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

SCRIPT_VERSION = "v7"
WORK = Path("/kaggle/working")
REPO = WORK / "CrispASR"
TEMP = Path("/kaggle/temp") if Path("/kaggle/temp").is_dir() else Path("/tmp")
BRANCH = "feat/438-hojo-asr"

SRC_REPO = "HojoAI/Hojo-ASR-Multi-V1"
DEMO_SPACE = "hugging-apps/hojo-asr-multi-v1-demo"
HF_REPO = "cstr/Hojo-ASR-Multi-V1-GGUF"
FIXTURES_REPO = "cstr/crispasr-regression-fixtures"

# German is one of the five languages the card advertises, so the primary
# fixture is the demo Space's german.wav rather than jfk.wav. English is kept
# as a second arm because a model that only works on its headline language is
# a different (and interesting) result from one that works on both.
AUDIO_ARMS = [("german", "examples/german.wav"), ("french", "examples/french.wav")]

print(f"=== hojo-asr-438 {SCRIPT_VERSION} ===", flush=True)

if not REPO.exists():
    subprocess.check_call(["git", "clone", "--depth", "1", "--recursive", "--shallow-submodules",
                           "-b", BRANCH,
                           "https://github.com/CrispStrobe/CrispASR", str(REPO)])
subprocess.check_call(["git", "submodule", "update", "--init", "--recursive", "--depth", "1",
                       "ggml"], cwd=str(REPO))
subprocess.check_call(["git", "log", "--oneline", "-1"], cwd=str(REPO))
sys.path.insert(0, str(REPO / "tools" / "kaggle"))
import kaggle_harness as kh  # noqa: E402

kh.init_progress()


def sec(title):
    print(f"\n{'=' * 78}\n== {title}\n{'=' * 78}", flush=True)


def disk():
    g = kh.free_gb(str(TEMP))
    print(f"  [disk] {g:.1f} GB free on {TEMP}" if g is not None else "  [disk] n/a", flush=True)


# ---------------------------------------------------------------------------
kh.step("install deps")
# --no-deps on hojo-asr: it pins torch==2.8.0 exactly, and reinstalling torch on
# Kaggle costs ~20 minutes and can break torchaudio. Its real requirements are
# transformers (for Qwen3-Omni), omegaconf, and -- non-obviously --
# openai-whisper: hojo_asr/wenet/utils/common.py does
# `from whisper.tokenizer import LANGUAGES`, so the package will not even
# import without it. v1 of this kernel found that in 90 seconds because the
# import check below runs before anything expensive.
os.environ.setdefault("NUMBA_DISABLE_CUDA", "1")  # numba (via openai-whisper) probes CUDA on import
kh.sh_with_progress(
    "pip install -q gguf safetensors huggingface_hub hf_transfer omegaconf "
    "soundfile 'transformers>=4.57.3,<5.0.0' openai-whisper "
    "&& pip install -q --no-deps hojo-asr")
subprocess.run([sys.executable, "-c",
                "import torch, torchaudio, transformers, hojo_asr; "
                "print('torch', torch.__version__); "
                "print('torchaudio', torchaudio.__version__); "
                "print('transformers', transformers.__version__); "
                "print('hojo_asr', hojo_asr.__version__)"], check=True)

kh.step("resolve HF token")
hf_token = kh.resolve_hf_token()
if hf_token:
    os.environ["HF_TOKEN"] = hf_token
    os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

from huggingface_hub import HfApi, hf_hub_download, snapshot_download  # noqa: E402

api = HfApi(token=hf_token)


def upload(local, path_in_repo, repo_id=HF_REPO, repo_type="model"):
    kh.step(f"upload {path_in_repo}")
    api.create_repo(repo_id=repo_id, repo_type=repo_type, exist_ok=True, private=False)
    api.upload_file(path_or_fileobj=str(local), path_in_repo=path_in_repo,
                    repo_id=repo_id, repo_type=repo_type)
    print(f"  -> {repo_id}/{path_in_repo}", flush=True)


# ---------------------------------------------------------------------------
sec("1. download checkpoint + audio fixtures")
src = snapshot_download(repo_id=SRC_REPO, cache_dir=str(TEMP / "hojo-src"), token=hf_token)
print(f"  src: {src}")
subprocess.run(["ls", "-la", src], check=False)
disk()

AUDIO_DIR = TEMP / "audio"
AUDIO_DIR.mkdir(parents=True, exist_ok=True)
arms = []
for name, rel in AUDIO_ARMS:
    try:
        p = hf_hub_download(repo_id=DEMO_SPACE, repo_type="space", filename=rel,
                            cache_dir=str(TEMP / "hojo-demo"), token=hf_token)
        dst = AUDIO_DIR / f"{name}.wav"
        # Normalise to 16 kHz mono float->PCM16 ONCE, so the Python reference
        # and the C++ CLI see byte-identical input. Letting each side do its
        # own resampling is how a "mel mismatch" turns out to be two different
        # resamplers rather than a port bug.
        import numpy as _np
        import soundfile as _sf
        wav_in, sr_in = _sf.read(p, dtype="float32", always_2d=True)
        mono = wav_in.mean(axis=1)
        if sr_in != 16000:
            import torch as _t
            import torchaudio as _ta
            mono = _ta.functional.resample(_t.from_numpy(mono), sr_in, 16000).numpy()
        peak = float(_np.abs(mono).max()) if mono.size else 0.0
        if peak > 1.0:
            mono = mono / peak
        _sf.write(dst, mono, 16000, subtype="PCM_16")
        arms.append((name, dst))
        print(f"  audio[{name}]: {dst}  ({sr_in} Hz -> 16000, {len(mono)/16000:.2f} s)")
    except Exception as e:  # noqa: BLE001
        print(f"  !! audio[{name}] unavailable: {e}")
if not arms:
    jfk = REPO / "samples" / "jfk.wav"
    if jfk.exists():
        arms.append(("jfk", jfk))
        print(f"  falling back to {jfk}")
if not arms:
    raise SystemExit("no audio fixture available — refusing to run a diff with no input")

# ---------------------------------------------------------------------------
sec("2. CONTROL ARM — the upstream package transcribes the audio first")
# If this comes back as garbage, every C++ number afterwards is meaningless,
# so it runs before anything is built and its output is printed verbatim.
control = {}
ctrl_script = TEMP / "control_arm.py"
# bind_lm_dtype comes from tools/reference_backends/hojo_asr.py rather than
# being re-typed here: upstream only runs on CUDA, so on CPU the f32 speech
# embeddings hit the BF16 decoder and torch raises. Two copies of that
# adaptation would drift, and then the control arm stops controlling anything.
ctrl_script.write_text(f'''
import json, sys
sys.path.insert(0, {str(REPO / "tools")!r})
from reference_backends.hojo_asr import bind_lm_dtype
from hojo_asr import HOJO_ASR
import time as _t0
_load0 = _t0.perf_counter()
model = HOJO_ASR.load_model({src!r}, device="cpu")
model.eval()
print(f"  [control] load_model took {{_t0.perf_counter()-_load0:.1f}} s", flush=True)
bind_lm_dtype(model)
out = {{}}
import time as _time
for name, path in {[(n, str(p)) for n, p in arms]!r}:
    t0 = _time.perf_counter()
    with open(path, "rb") as f:
        res = model.run_infer([f.read()], batch_size=1, cuda_enabled=False)
    out[name] = res[0]["text"]
    print(f"  CONTROL[{{name}}] = {{out[name]!r}}  ({{_time.perf_counter()-t0:.1f}} s)", flush=True)
json.dump(out, open({str(TEMP / "control.json")!r}, "w"))
''')
rc = subprocess.run([sys.executable, str(ctrl_script)]).returncode
if rc == 0 and (TEMP / "control.json").exists():
    control = json.load(open(TEMP / "control.json"))
else:
    print("  !! CONTROL ARM FAILED — C++ parity below cannot be trusted", flush=True)
print(f"  control = {control}", flush=True)

# ---------------------------------------------------------------------------
sec("3. convert -> f16 GGUF")
F16 = TEMP / "hojo-asr-multi-v1-f16.gguf"
kh.sh_with_progress(
    f"{sys.executable} {REPO}/models/convert-hojo-asr-to-gguf.py "
    f"--input {src} --output {F16}")
print(f"  f16: {F16.stat().st_size / 1024**3:.2f} GB")
disk()
upload(F16, F16.name)

# ---------------------------------------------------------------------------
sec("4. build crispasr (quantize + diff + cli)")
kh.install_build_toolchain()
flags = kh.cache_and_link_flags()  # already folds in crispasr_cmake_flags()
kh.sh_with_progress(
    f"cmake -G Ninja -B {REPO}/build -S {REPO} -DCMAKE_BUILD_TYPE=Release " + " ".join(flags))
kh.sh_with_progress(
    f"cmake --build {REPO}/build -j{kh.safe_build_jobs(False)} "
    f"--target crispasr-quantize crispasr-diff crispasr-cli")
BIN = REPO / "build" / "bin"
for t in ("crispasr-quantize", "crispasr-diff", "crispasr"):
    print(f"  {t}: {'OK' if (BIN / t).exists() else 'MISSING'}")
    if not (BIN / t).exists():
        raise SystemExit(f"build did not produce {t}")

kh.step("unit tests — the frame schedule guards")
kh.sh_with_progress(
    f"cmake --build {REPO}/build -j{kh.safe_build_jobs(False)} --target test-hojo-asr-frames "
    f"&& {BIN}/test-hojo-asr-frames")

kh.step("wiring audit")
aud = subprocess.run([sys.executable, str(REPO / "tools" / "check-backend-wiring.py"),
                      "--crispasr", str(BIN / "crispasr")], capture_output=True, text=True)
print(aud.stdout[-6000:])
print(f"  wiring audit rc={aud.returncode}")
lb = subprocess.run([str(BIN / "crispasr"), "--list-backends"], capture_output=True, text=True)
hit = [l for l in lb.stdout.splitlines() if "hojo" in l.lower()]
print(f"  --list-backends sees hojo-asr: {hit if hit else 'NO — the backend is invisible'}")

kh.step("regenerate the feature matrix")
# Generated from `crispasr --list-backends-json`, so it can only be produced
# where a binary exists. Copied to /kaggle/working to come back as output.
subprocess.run([sys.executable, str(REPO / "tools" / "gen-feature-matrix.py")],
               cwd=str(REPO), check=False)
fm = REPO / "docs" / "feature-matrix.md"
if fm.exists():
    rows = [l for l in fm.read_text().splitlines() if "hojo" in l.lower()]
    print(f"  feature-matrix hojo rows: {rows if rows else 'NONE — regeneration would DROP it'}")
    # NOT copied out as a committable artifact. This binary is built from a
    # BRANCH based on an older main, so it does not know about backends merged
    # since (breeze-tts-2, voxtral-f16, ...) and the generator would silently
    # delete their rows. The matrix must be regenerated from a build of main
    # AFTER this branch merges. All we want here is the yes/no above.
    print("  (matrix intentionally NOT exported — regenerate from post-merge main)")

# ---------------------------------------------------------------------------
sec("5. reference dump (upstream package via forward hooks)")
refs = {}
ref_greedy = {}
ref_beam = {}
for name, wav in arms:
    ref = TEMP / f"hojo-asr-{name}-ref.gguf"
    rd = subprocess.run([sys.executable, str(REPO / "tools" / "dump_reference.py"),
                         "--backend", "hojo-asr", "--model-dir", src,
                         "--audio", str(wav), "--output", str(ref),
                         "--max-new-tokens", "200"], capture_output=True, text=True)
    print(rd.stdout[-8000:])
    if rd.stderr.strip():
        print("  [stderr]", rd.stderr[-3000:])
    rc = rd.returncode
    for line in rd.stdout.splitlines():
        if "generated_text_greedy:" in line:
            ref_greedy[name] = line.split("generated_text_greedy:", 1)[1].strip()
        elif "generated_text (beams=" in line:
            ref_beam[name] = line.split("):", 1)[1].strip()
    if rc != 0 or not ref.exists():
        print(f"  !! reference dump FAILED for {name}", flush=True)
        continue
    refs[name] = ref
    upload(ref, f"hojo-asr/{name}/ref.gguf", repo_id=FIXTURES_REPO, repo_type="dataset")
disk()

# ---------------------------------------------------------------------------
def run_diff(label, model_gguf, wav, ref, env_extra=None):
    print(f"\n--- crispasr-diff [{label}] ---", flush=True)
    env = dict(os.environ)
    if env_extra:
        env.update(env_extra)
    r = subprocess.run([str(BIN / "crispasr-diff"), "hojo-asr", str(model_gguf),
                        str(ref), str(wav)], env=env, capture_output=True, text=True)
    print(r.stdout)
    if r.stderr.strip():
        print("  [stderr]", r.stderr[-4000:])
    print(f"  rc={r.returncode}", flush=True)
    return r.returncode


sec("6. diff at F16 — parity before quantization noise enters")
for name, wav in arms:
    if name in refs:
        run_diff(f"f16/{name}", F16, wav, refs[name])

# ---------------------------------------------------------------------------
sec("7. CONV TILING A/B — tiled vs untiled encoder, exact-equality check")
# The tiled schedule is an algebraic identity, so anything other than an exact
# match is a bug in the halo arithmetic. Compared by max|delta| AND by norm,
# because cosine alone would not notice a uniform scale.
for name, wav in arms:
    tiled = TEMP / f"enc-tiled-{name}.bin"
    untiled = TEMP / f"enc-untiled-{name}.bin"
    schedules = {}
    for out, tile in ((tiled, "64"), (untiled, "0")):
        env = dict(os.environ)
        env["CRISPASR_HOJO_ASR_ENC_DUMP"] = str(out)
        r = subprocess.run([str(BIN / "crispasr"), "-m", str(F16), "--backend", "hojo-asr",
                            "-f", str(wav), "--max-new-tokens", "1"],
                           env={**env, "CRISPASR_HOJO_ASR_CONV_TILE": tile},
                           capture_output=True, text=True)
        line = [l for l in r.stderr.splitlines() if "conv_schedule=" in l]
        schedules[tile] = line[-1].strip() if line else "(runtime never reported a schedule)"
        print(f"  [{name}] tile={tile}: {schedules[tile]}")
    # The A/B is only evidence if the two arms actually ran DIFFERENT schedules.
    # Identical dumps from two identical runs would read as "tiling is exact"
    # while proving only that the env knob did nothing.
    if schedules.get("64") == schedules.get("0"):
        print(f"  [{name}] !! BOTH ARMS RAN THE SAME SCHEDULE — the A/B is inert, "
              f"not passing ({schedules.get('64')})")
    elif tiled.exists() and untiled.exists():
        import numpy as np
        a = np.fromfile(tiled, dtype=np.float32)
        b = np.fromfile(untiled, dtype=np.float32)
        if a.shape != b.shape:
            print(f"  [{name}] SHAPE MISMATCH {a.shape} vs {b.shape}  <-- tiling bug")
        else:
            d = float(np.max(np.abs(a - b)))
            cos = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))
            print(f"  [{name}] max|tiled-untiled| = {d:.3e}   cos = {cos:.8f}   "
                  f"|tiled| = {np.linalg.norm(a):.4f}   |untiled| = {np.linalg.norm(b):.4f}")
            print(f"  [{name}] VERDICT: {'EXACT' if d == 0.0 else 'NOT EXACT — halo arithmetic is wrong'}")
    else:
        print(f"  [{name}] !! one or both encoder dumps missing — A/B could not run "
              f"(this is NOT a pass)")

# ---------------------------------------------------------------------------
sec("8. quantize -> q4_k")
Q4 = TEMP / "hojo-asr-multi-v1-q4_k.gguf"
rc = subprocess.run([str(BIN / "crispasr-quantize"), str(F16), str(Q4), "q4_k"]).returncode
print(f"  quantize rc={rc}")
if rc == 0 and Q4.exists():
    print(f"  q4_k: {Q4.stat().st_size / 1024**3:.2f} GB")
    upload(Q4, Q4.name)
    disk()
    sec("9. diff at q4_k")
    for name, wav in arms:
        if name in refs:
            run_diff(f"q4_k/{name}", Q4, wav, refs[name])
else:
    print("  !! quantize failed — add the tensor rule instead of forcing it", flush=True)

# ---------------------------------------------------------------------------
sec("10. END-TO-END — the real CLI, read the output (HARD RULE 3c)")
# GREEDY is the roundtrip arm, and it is matched: the reference emits a
# num_beams=1 transcript with every other setting identical, so a difference
# here is the port rather than the search strategy.
#
# The beam arm is CAPPED. core_beam_decode replays each beam's whole suffix
# every step, so beam 4 over 200 tokens is 80,400 token-forwards (~4 h on this
# 4.4 B decoder) versus greedy's 200. An uncapped beam arm here would not
# produce a slow result -- it would produce no result, by blowing the 12 h
# kernel cap. 48 tokens exercises the same code path in ~17 min.
import time as _time

for name, wav in arms:
    for label, gguf in (("f16", F16), ("q4_k", Q4)):
        if not Path(gguf).exists():
            continue
        t0 = _time.perf_counter()
        r = subprocess.run([str(BIN / "crispasr"), "-m", str(gguf), "--backend", "hojo-asr",
                            "-f", str(wav), "-bs", "1"], capture_output=True, text=True)
        dt = _time.perf_counter() - t0
        print(f"\n  CLI[{label}/{name}] greedy rc={r.returncode} ({dt:.1f} s)")
        print(f"    C++ greedy    : {r.stdout.strip()[:400]!r}")
        print(f"    REF greedy    : {ref_greedy.get(name, '(no greedy reference)')}")
        print(f"    REF beam4     : {ref_beam.get(name, '(no beam reference)')}")
        print(f"    CONTROL beam4 : {control.get(name, '(control arm failed)')!r}")
        if r.returncode != 0:
            print(f"    [stderr] {r.stderr[-2000:]}")

    # One capped beam run per arm, on f16 only, purely to exercise + time the
    # beam path. Not a parity arm.
    if Path(F16).exists():
        t0 = _time.perf_counter()
        rb = subprocess.run([str(BIN / "crispasr"), "-m", str(F16), "--backend", "hojo-asr",
                             "-f", str(wav), "-bs", "4", "--max-new-tokens", "48"],
                            capture_output=True, text=True)
        print(f"  CLI[f16/{name}] beam4 capped@48 rc={rb.returncode} "
              f"({_time.perf_counter()-t0:.1f} s)")
        print(f"    C++ beam4(48) : {rb.stdout.strip()[:300]!r}")
        cost = [l for l in rb.stderr.splitlines() if "token-forwards" in l]
        if cost:
            print(f"    {cost[-1].strip()}")

# ---------------------------------------------------------------------------
sec("11. checkpoint ccache + summary")
kh.export_ccache_tar()
print(f"  f16 uploaded : {HF_REPO}/{F16.name}")
print(f"  q4_k uploaded: {HF_REPO}/{Q4.name}" if Q4.exists() else "  q4_k: NOT PRODUCED")
print(f"  refs uploaded: {sorted(refs)}")
print(f"  control      : {control}")
if not control:
    print("  !! THE CONTROL ARM DID NOT RUN — treat every cosine above as unvalidated",
          flush=True)
print("=== done ===", flush=True)
