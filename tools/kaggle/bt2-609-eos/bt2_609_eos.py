#!/usr/bin/env python3
"""
#609: bt2-tts keeps talking after the text (recites the instruction, then babble).

The reporter's outputs say the model reads the text correctly, misses EOS, and
goes on. This kernel measures how often that happens and what the EOS
probability looks like around the end of the text, so a stopping rule can be
chosen from data:

  * reporter payloads (#609 repro package, private dataset; 4 texts, the
    15-line instruction and its reordered variant), voices Piedmont / mixed,
    seeds 87, 42 and 0..N, CUDA (T4), q4_k like the reporter;
  * per run: CRISPASR_BREEZE_EOS_TRACE (frame, p_eos at the sampling
    temperature, rank of EOS), duration, parakeet word timestamps (-ojf);
  * the reporter's CPU case (repro3 000384) once on CPU.

Results: /kaggle/working/bt2_609.json + traces/, rewritten as it goes.
"""
import glob, json, os, re, shutil, subprocess, sys, time, zipfile
from pathlib import Path

WORK = Path("/kaggle/working"); TMP = Path("/kaggle/temp"); TMP.mkdir(parents=True, exist_ok=True)
RES = WORK / "bt2_609.json"; TR = WORK / "traces"; TR.mkdir(exist_ok=True)
OUT = TMP / "wav"; OUT.mkdir(exist_ok=True)
BRANCH = os.environ.get("CRISPASR_BRANCH", "test/609")
N_SEEDS = int(os.environ.get("N_SEEDS", "12"))
CAP = 1500  # frames (120 s) - long enough to see a runaway, bounded
T_END = time.time() + 10.5 * 3600

CR = TMP / "CrispASR"
if not CR.exists():
    subprocess.run(["git", "clone", "--depth", "1", "--branch", BRANCH, "--recurse-submodules", "--shallow-submodules",
                    "https://github.com/CrispStrobe/CrispASR.git", str(CR)], check=True)
sys.path.insert(0, str(CR / "tools" / "kaggle"))
import kaggle_harness as kh
kh.init_progress()
kh.install_build_toolchain()
res = {"started": time.strftime("%F %T"), "branch": BRANCH, "runs": []}
save = lambda: RES.write_text(json.dumps(res, indent=1))


def sh(cmd, timeout=None, env=None):
    return subprocess.run(cmd, shell=isinstance(cmd, str), capture_output=True, text=True, timeout=timeout, env=env)


res["head"] = sh(f"git -C {CR} rev-parse HEAD").stdout.strip()

# ---- data (dataset dirs may arrive as folders or as zips) ----
DS = Path(next(p for p in glob.glob("/kaggle/input/**/crispasr-609-repro*", recursive=True) if os.path.isdir(p)))
DATA = TMP / "data"
if not DATA.exists():
    shutil.copytree(DS, DATA)
    for z in DATA.rglob("*.zip"):
        with zipfile.ZipFile(z) as zf:
            zf.extractall(z.with_suffix(""))
res["data_files"] = sorted(str(p.relative_to(DATA)) for p in DATA.rglob("*") if p.is_file())
save()


def find(pattern):
    hits = sorted(DATA.rglob(pattern))
    return hits[0] if hits else None


# ---- build ----
B = TMP / "b"
flags = ["-G Ninja", "-DCMAKE_BUILD_TYPE=Release", "-DGGML_NATIVE=ON", "-DCRISPASR_OPUS_FETCH=ON"] + \
    kh.cuda_build_flags() + kh.cache_and_link_flags()
r = sh(f"cmake -S {CR} -B {B} " + " ".join(flags), timeout=1800)
if r.returncode:
    res["configure_error"] = r.stderr[-3000:]; save(); raise SystemExit(1)
t0 = time.time()
with kh.build_heartbeat("build", interval_s=120):
    kh.sh_with_progress(f"cmake --build {B} -j{kh.safe_build_jobs(gpu=True)} --target crispasr-cli")
res["build_s"] = round(time.time() - t0); save()
CLI = str(B / "bin" / "crispasr")

# ---- models ----
sh(f"{sys.executable} -m pip install -q huggingface_hub")
from huggingface_hub import hf_hub_download
HF = kh.resolve_hf_token() or None
M = TMP / "models"
BREEZE = hf_hub_download("cstr/breeze-tts-2-GGUF", "breeze-tts-2-q4_k.gguf", local_dir=str(M), token=HF)
CODEC = hf_hub_download("cstr/qwen3-tts-tokenizer-12hz-GGUF", "qwen3-tts-tokenizer-12hz.gguf", local_dir=str(M), token=HF)
ASR = hf_hub_download("cstr/parakeet-tdt-0.6b-v3-GGUF", "parakeet-tdt-0.6b-v3-q4_k.gguf", local_dir=str(M), token=HF)
res["breeze_sha256"] = sh(f"sha256sum {BREEZE}").stdout.split()[0]
save()

COMMON = ["--backend", "bt2-tts", "-m", BREEZE, "--codec-model", CODEC, "--accept-license", "other",
          "--i-have-rights", "--speaker-identity", "synthetic", "--no-watermark",
          "--accept-marking-responsibility", "--no-spoken-disclaimer", "-n", str(CAP)]


def norm(s):
    return re.findall(r"[a-z0-9]+", s.lower())


def asr(wav):
    r = sh([CLI, "-m", ASR, "-f", wav, "-ojf", "-of", wav[:-4], "-np"], timeout=900)
    j = Path(wav[:-4] + ".json")
    words, text = [], ""
    if j.exists():
        try:
            d = json.loads(j.read_text())
            for seg in d.get("transcription", []):
                text += seg.get("text", "")
                for w in seg.get("words", []) or seg.get("tokens", []):
                    t = w.get("timestamps") or {}
                    off = w.get("offsets") or {}
                    st = off.get("from", 0) / 1000.0 if off else None
                    en = off.get("to", 0) / 1000.0 if off else None
                    words.append([w.get("word") or w.get("text"), st, en])
        except Exception as e:
            text = f"<json error {e}>"
    return text.strip(), words, r.stderr[-400:] if r.returncode else ""


def run(tag, payload, voice, seed, gpu=True):
    if time.time() > T_END:
        return None
    p = json.loads(payload.read_text())
    vw = find(f"{voice}.wav"); vt = find(f"{voice}.txt")
    wav = str(OUT / f"{tag}.wav")
    trace = TR / f"{tag}.txt"
    trace.unlink(missing_ok=True)
    cmd = [CLI] + COMMON + ["--tts", p["input"], "--instruct", p["instructions"], "--voice", str(vw),
                            "--ref-text", vt.read_text().strip(), "--seed", str(seed),
                            "--tts-cfg-scale", str(p.get("cfg_scale", 1.0)), "--tts-output", wav]
    if not gpu:
        cmd.append("--no-gpu")
    env = dict(os.environ, CRISPASR_BREEZE_EOS_TRACE=str(trace))
    t0 = time.time()
    r = sh(cmd, timeout=3600, env=env)
    el = time.time() - t0
    frames = [l.split() for l in trace.read_text().splitlines() if l and l != "end"] if trace.exists() else []
    dur = None
    if os.path.exists(wav):
        import wave
        with wave.open(wav) as w:
            dur = w.getnframes() / w.getframerate()
    text, words, aerr = asr(wav) if dur else ("", [], "")
    ref = norm(p["input"])
    hyp = norm(text)
    rec = {"tag": tag, "payload": payload.parent.name + "/" + payload.stem, "voice": voice, "seed": seed,
           "gpu": gpu, "rc": r.returncode, "s": round(el, 1), "dur": dur, "frames": len(frames),
           "hit_cap": len(frames) >= CAP, "ref": p["input"], "asr": text, "words": words,
           "extra_words": max(0, len(hyp) - len(ref)), "err": r.stderr[-800:] if r.returncode else "",
           "asr_err": aerr}
    res["runs"].append(rec)
    save()
    return rec


kh.step("runs")
for pay_dir in ["payloads_r1", "payloads_r4"]:
    pays = sorted((DATA).rglob(f"{pay_dir}/*.json")) or sorted(DATA.rglob(f"{pay_dir}*/**/*.json"))
    seeds = [87, 42] + list(range(N_SEEDS)) if pay_dir == "payloads_r1" else [87, 42]
    for pay in pays:
        for voice in ["Piedmont", "mixed"]:
            for s in seeds:
                run(f"{pay_dir}_{pay.stem}_{voice}_s{s}", pay, voice, s)

kh.step("cpu")
p384 = next((p for p in DATA.rglob("payloads_r1/000384.json")), None) or next(DATA.rglob("**/000384.json"))
run("cpu_000384_mixed_s87", p384, "mixed", 87, gpu=False)
p206 = next(DATA.rglob("**/payloads_r1/000206.json"), None) or next(DATA.rglob("**/000206.json"))
run("cpu_000206_Piedmont_s87", p206, "Piedmont", 87, gpu=False)

res["finished"] = time.strftime("%F %T")
save()
kh.step("done")
bad = [r for r in res["runs"] if r and r["extra_words"] >= 3]
print(len(res["runs"]), "runs,", len(bad), "with >=3 extra words")
