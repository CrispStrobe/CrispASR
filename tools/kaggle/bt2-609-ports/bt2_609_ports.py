#!/usr/bin/env python3
"""
#609, third kernel: do the other Breeze TTS 2 implementations run away on the
reporter's inputs too?

None of them has a stopping rule beyond a sampled EOS and a frame cap (checked
in their sources on 2026-11-24). This runs each one, as shipped, on the same
cases and scores every output with the same parakeet roundtrip as the earlier
#609 kernels (runaway = 4+ words beyond the text).

Cases: payloads_r1 000002 / 000206 / 000214 / 000384 with the reporter's
instruction, voices Piedmont / mixed (reference clip + transcript), seeds
87, 42, 0 (N_SEEDS). CFG is pinned to 1 everywhere (the reporter's setting);
everything else is each port's default.

Arms (each isolated: a failed build or install is recorded and skipped):
  ours       CrispASR main, bt2-tts q4_k, CUDA (length guard on)
  hoppou     HoppouAI/Breeze-TTS-2.cpp, CUDA, q4_k
  audiocpp   0xShug0/audio.cpp, CUDA, Raxdiam q4_k
  qwentts    ljchang/qwentts.cpp@breeze, CUDA, mecha-factory Q5_K_M
  mlxaudio   Blaizzy/mlx-audio, mlx[cuda], mlx-community/Breeze-TTS-2-mlx
  rishikksh  rishikksh20/breeze-tts-mlx, mlx[cuda], INT8
  vanch      vanch007/mlx-breeze-tts2, mlx[cuda], 8-bit
  rust       Liyulingyue/rust-model-inference, CPU (no CUDA backend): one
             smoke case first, the arm only if that is fast enough

Results: /kaggle/working/bt2_609_ports.json, rewritten after every run.
"""
import glob, json, os, re, shutil, subprocess, sys, time, wave, zipfile
from pathlib import Path

WORK = Path("/kaggle/working"); TMP = Path("/kaggle/temp"); TMP.mkdir(parents=True, exist_ok=True)
RES = WORK / "bt2_609_ports.json"
OUT = TMP / "wav"; OUT.mkdir(exist_ok=True)
MV = TMP / "mlxv"; MPY = str(MV / "bin" / "python")   # venv for the MLX ports
SEEDS = [87, 42, 0][: int(os.environ.get("N_SEEDS", "3"))]
T_END = time.time() + 11.3 * 3600
ARM_BUDGET = 100 * 60          # wall clock per arm, build included
CASE_TIMEOUT = 900
ONLY = [a for a in os.environ.get("ARMS", "").split(",") if a]


def sh(cmd, timeout=None, env=None, cwd=None, input=None):
    try:
        return subprocess.run(cmd, shell=isinstance(cmd, str), capture_output=True, text=True,
                              timeout=timeout, env=env, cwd=cwd, input=input)
    except subprocess.TimeoutExpired as e:
        return subprocess.CompletedProcess(cmd, 124, e.stdout or "", f"TIMEOUT after {timeout}s")


def clone(url, dst, branch=None, submodules=False):
    for _ in range(4):
        if dst.exists():
            shutil.rmtree(dst)
        cmd = ["git", "clone", "--depth", "1"] + (["--branch", branch] if branch else []) + \
              (["--recurse-submodules", "--shallow-submodules"] if submodules else []) + [url, str(dst)]
        if subprocess.run(cmd, timeout=1800).returncode == 0:
            return
        time.sleep(15)
    raise RuntimeError("clone failed " + url)


CR = TMP / "CrispASR"
if not CR.exists():
    clone("https://github.com/CrispStrobe/CrispASR.git", CR, os.environ.get("CRISPASR_BRANCH", "main"), True)
sys.path.insert(0, str(CR / "tools" / "kaggle"))
import kaggle_harness as kh
kh.init_progress()
HF = kh.resolve_hf_token() or None   # before the build: the progress mirror needs it
kh.install_build_toolchain()
res = {"started": time.strftime("%F %T"), "seeds": SEEDS, "arms": {}, "runs": []}
save = lambda: RES.write_text(json.dumps(res, indent=1))
res["crispasr_head"] = sh(f"git -C {CR} rev-parse HEAD").stdout.strip()
res["gpu"] = sh("nvidia-smi --query-gpu=name,driver_version --format=csv,noheader").stdout.strip()
save()

DS = Path(next(p for p in glob.glob("/kaggle/input/**/crispasr-609-repro*", recursive=True) if os.path.isdir(p)))
DATA = TMP / "data"
if not DATA.exists():
    shutil.copytree(DS, DATA)
    for z in DATA.rglob("*.zip"):
        with zipfile.ZipFile(z) as zf:
            zf.extractall(z.with_suffix(""))
find = lambda pat: sorted(DATA.rglob(pat))[0]
PAYLOADS = [json.loads(find(f"payloads_r1/{n}.json").read_text()) | {"id": n} for n in ["000002", "000206", "000214", "000384"]]
VOICES = {v: (str(find(f"{v}.wav")), find(f"{v}.txt").read_text().strip()) for v in ["Piedmont", "mixed"]}
for v, (w, t) in VOICES.items():
    (TMP / f"{v}.ref.txt").write_text(t)

# ---- our build: the scorer (parakeet) and the "ours" arm ----
B = TMP / "b"
flags = ["-G Ninja", "-DCMAKE_BUILD_TYPE=Release", "-DGGML_NATIVE=ON", "-DCRISPASR_OPUS_FETCH=ON"] + \
    kh.cuda_build_flags() + kh.cache_and_link_flags()
r = sh(f"cmake -S {CR} -B {B} " + " ".join(flags), timeout=1800)
if r.returncode:
    res["configure_error"] = r.stderr[-3000:]; save(); raise SystemExit(1)
with kh.build_heartbeat("build_ours", interval_s=120):
    kh.sh_with_progress(f"cmake --build {B} -j{kh.safe_build_jobs(gpu=True)} --target crispasr-cli")
CLI = str(B / "bin" / "crispasr")

sh(f"{sys.executable} -m pip install -q huggingface_hub soundfile")
from huggingface_hub import hf_hub_download, snapshot_download
M = TMP / "models"; M.mkdir(exist_ok=True)
ASR = hf_hub_download("cstr/parakeet-tdt-0.6b-v3-GGUF", "parakeet-tdt-0.6b-v3-q4_k.gguf", local_dir=str(M), token=HF)


def norm(s):
    return re.findall(r"[a-z0-9]+", s.lower())


def record(arm, p, voice, seed, wav, secs, extra=None):
    dur = None
    if os.path.exists(wav) and os.path.getsize(wav) > 44:
        try:
            with wave.open(wav) as w:
                dur = w.getnframes() / w.getframerate()
        except Exception as e:
            extra = (extra or {}) | {"wav_error": repr(e)}
    text = ""
    if dur:
        r = sh([CLI, "-m", ASR, "-f", wav, "-nt", "-np"], timeout=900)
        text = " ".join(l.strip() for l in r.stdout.splitlines() if l.strip())
    rec = {"arm": arm, "id": p["id"], "voice": voice, "seed": seed, "s": round(secs, 1), "dur": dur,
           "asr": text, "extra_words": max(0, len(norm(text)) - len(norm(p["input"]))), **(extra or {})}
    res["runs"].append(rec); save()
    print(arm, p["id"], voice, seed, dur, rec["extra_words"], flush=True)
    kh.step("run", arm=arm, id=p["id"], voice=voice, seed=seed, dur=dur, extra=rec["extra_words"], s=rec["s"],
            asr=text[:160], err=(rec.get("err") or "")[-200:])
    return rec


def run_cases(arm, synth, t_arm_end, cases=None):
    """synth(p, voice, wav_path, ref_wav, ref_text, seed) -> CompletedProcess or dict."""
    done = 0
    for p in PAYLOADS:
        for voice, (vw, vt) in VOICES.items():
            for s in SEEDS:
                if time.time() > min(t_arm_end, T_END) or (cases is not None and done >= cases):
                    res["arms"][arm]["truncated"] = True; save()
                    return
                wav = str(OUT / f"{arm}_{p['id']}_{voice}_{s}.wav")
                t0 = time.time()
                try:
                    r = synth(p, voice, wav, vw, vt, s)
                    extra = {}
                    if isinstance(r, subprocess.CompletedProcess) and r.returncode:
                        extra = {"rc": r.returncode, "err": (r.stderr or "")[-600:]}
                except Exception as e:
                    extra = {"err": repr(e)[-600:]}
                record(arm, p, voice, s, wav, time.time() - t0, extra)
                done += 1
                if done % 6 == 0:
                    mirror()


def arm(name):
    def deco(fn):
        if ONLY and name not in ONLY:
            return fn
        if time.time() > T_END:
            res["arms"][name] = {"status": "skipped: out of time"}; save(); return fn
        t_start = time.time()
        res["arms"][name] = {"status": "running", "t0": time.strftime("%T")}; save()
        kh.step("arm", arm=name, free_gb=free_gb())
        t_end = time.time() + ARM_BUDGET
        try:
            info = fn(t_end) or {}
            res["arms"][name].update({"status": "done", **info})
        except Exception as e:
            res["arms"][name].update({"status": "failed", "error": repr(e)[-3000:]})
        res["arms"][name]["minutes"] = round((time.time() - t_start) / 60, 1)
        cleanup(name)
        res["arms"][name]["free_gb_after"] = free_gb()
        summ = summarize().get(name, {})
        save()
        # One line per finished arm on the progress mirror: status plus its scores so far.
        kh.step("arm_done", arm=name, status=res["arms"][name]["status"],
                error=clip(res["arms"][name].get("error", "")), n=summ.get("n", 0), ok=summ.get("ok", 0),
                runaway=summ.get("runaway", 0), any_extra=summ.get("any_extra", 0), max_dur=summ.get("max_dur", 0),
                minutes=res["arms"][name]["minutes"], free_gb=res["arms"][name]["free_gb_after"])
        print("ARM", name, json.dumps(res["arms"][name]), json.dumps(summ), flush=True)
        mirror()
        return fn
    return deco


def clip(e):
    # The head names the failing step and the first error; the tail is often configure chatter.
    return e if len(e) <= 900 else e[:700] + " ... " + e[-200:]


def summarize():
    summ = {}
    for r in res["runs"]:
        a = summ.setdefault(r["arm"], {"n": 0, "ok": 0, "runaway": 0, "any_extra": 0, "max_dur": 0, "by_id": {}})
        a["n"] += 1
        if r.get("dur"):
            a["ok"] += 1
            a["max_dur"] = max(a["max_dur"], round(r["dur"], 1))
        bad = r["extra_words"] >= 4
        a["runaway"] += bad
        a["any_extra"] += r["extra_words"] > 0
        a["by_id"].setdefault(r["id"], [0, 0])
        a["by_id"][r["id"]][0] += bad; a["by_id"][r["id"]][1] += 1
    res["summary"] = summ
    save()
    return summ


def mirror():
    """Force the progress mirror now and put the full results JSON next to it on HF, so a crash or
    a killed session keeps every finished case. Best effort, like the harness's own push."""
    kh._push_progress_to_hf(force=True)
    try:
        from huggingface_hub import HfApi
        if os.environ.get("HF_TOKEN") and kh._HF_PROGRESS_REPO:
            HfApi(token=os.environ["HF_TOKEN"]).upload_file(
                path_or_fileobj=str(RES), path_in_repo=kh._HF_PROGRESS_PATH.replace(".jsonl", "-results.json"),
                repo_id=kh._HF_PROGRESS_REPO, repo_type="dataset", commit_message="609 ports results")
    except Exception as e:
        print("mirror failed:", e, flush=True)


def free_gb():
    st = shutil.disk_usage(str(TMP))
    return round(st.free / 2**30, 1)


def cleanup(name):
    """Everything an arm brought in goes once it is scored: model, source + build tree, wavs, HF cache."""
    for d in [M / name, TMP / f"src_{name}", Path.home() / ".cache" / "huggingface" / "hub",
              Path.home() / ".cargo" / "registry"]:
        shutil.rmtree(d, ignore_errors=True)
    for w in OUT.glob(f"{name}_*.wav"):
        w.unlink(missing_ok=True)
    sh(f"{MPY} -m pip cache purge") if Path(MPY).exists() else None
    sh(f"{sys.executable} -m pip cache purge")


def must(r, what):
    if r.returncode:
        raise RuntimeError(f"{what} failed rc={r.returncode}: {(r.stderr or '')[-2500:]} {(r.stdout or '')[-1000:]}")
    return r


def patch_cuda_driver(src):
    """Vendored ggml copies link CUDA::cuda_driver unconditionally; Kaggle's CUDA 13
    FindCUDAToolkit does not define it, so configure fails (our own ggml carries the
    same fallback, ggml/src/ggml-cuda/CMakeLists.txt). Link libcuda by path, or build
    without the VMM pool when there is none (performance is irrelevant here)."""
    lib = next((p for pat in ["/usr/lib/x86_64-linux-gnu/libcuda.so", "/usr/lib/x86_64-linux-gnu/libcuda.so.1",
                              "/usr/local/cuda/lib64/stubs/libcuda.so",
                              "/usr/local/cuda/targets/x86_64-linux/lib/stubs/libcuda.so",
                              "/usr/local/cuda*/**/libcuda.so"]
                for p in glob.glob(pat, recursive=True)), None)
    hits = [f for f in Path(src).rglob("CMakeLists.txt") if "ggml-cuda" in f.parts and "CUDA::cuda_driver" in f.read_text()]
    for f in hits:
        t = f.read_text()
        f.write_text(t.replace("CUDA::cuda_driver", lib) if lib else "set(GGML_CUDA_NO_VMM ON)\n" + t)
    res.setdefault("cuda_driver_patch", {})[str(src)] = {"lib": lib, "files": [str(f) for f in hits]}; save()


def find_bin(root, name):
    hits = [p for p in Path(root).rglob(name) if p.is_file() and os.access(p, os.X_OK)]
    if not hits:
        raise RuntimeError(f"{name} not found under {root}")
    return str(hits[0])


JOBS = kh.safe_build_jobs(gpu=True)


# ---- ours ----
@arm("ours")
def _(t_end):
    breeze = hf_hub_download("cstr/breeze-tts-2-GGUF", "breeze-tts-2-q4_k.gguf", local_dir=str(M / "ours"), token=HF)
    codec = hf_hub_download("cstr/qwen3-tts-tokenizer-12hz-GGUF", "qwen3-tts-tokenizer-12hz.gguf", local_dir=str(M / "ours"), token=HF)
    common = [CLI, "--backend", "bt2-tts", "-m", breeze, "--codec-model", codec, "--accept-license", "other",
              "--i-have-rights", "--speaker-identity", "synthetic", "--no-watermark",
              "--accept-marking-responsibility", "--no-spoken-disclaimer", "-n", "1500"]
    run_cases("ours", lambda p, v, wav, vw, vt, s: sh(common + [
        "--tts", p["input"], "--instruct", p["instructions"], "--voice", vw, "--ref-text", vt,
        "--seed", str(s), "--tts-cfg-scale", "1", "--tts-output", wav], timeout=CASE_TIMEOUT), t_end)


# ---- HoppouAI/Breeze-TTS-2.cpp ----
@arm("hoppou")
def _(t_end):
    src = TMP / "src_hoppou"
    clone("https://github.com/HoppouAI/Breeze-TTS-2.cpp.git", src, submodules=True)
    patch_cuda_driver(src)
    bd = src / "build"
    must(sh(f"cmake -S {src} -B {bd} -G Ninja -DCMAKE_BUILD_TYPE=Release -DBREEZE_VULKAN=OFF -DBREEZE_CUDA=ON "
            f"-DCMAKE_CUDA_ARCHITECTURES=75 -DBREEZE_BUILD_SERVER=OFF -DBREEZE_MP3=OFF -DBREEZE_BUILD_SHARED=OFF",
            timeout=1800), "configure")
    with kh.build_heartbeat("build_hoppou", interval_s=120):
        must(sh(f"cmake --build {bd} -j{JOBS}", timeout=3600), "build")
    exe = find_bin(bd, "breeze-cli")
    model = hf_hub_download("HoppouAI/Breeze-TTS-2.cpp", "breeze-tts-2-q4_k.gguf", local_dir=str(M / "hoppou"), token=HF)
    run_cases("hoppou", lambda p, v, wav, vw, vt, s: sh([
        exe, model, "--text", p["input"], "--instruction", p["instructions"].strip(), "--ref-audio", vw,
        "--ref-text", vt, "--seed", str(s), "--cfg-scale", "1", "--output", wav], timeout=CASE_TIMEOUT), t_end)
    return {"rev": sh(f"git -C {src} rev-parse HEAD").stdout.strip()}


# ---- 0xShug0/audio.cpp ----
@arm("audiocpp")
def _(t_end):
    src = TMP / "src_audiocpp"
    clone("https://github.com/0xShug0/audio.cpp.git", src, submodules=True)
    patch_cuda_driver(src)
    with kh.build_heartbeat("build_audiocpp", interval_s=120):
        must(sh(f"bash scripts/build_linux.sh --deployment-build --backend cuda --cuda-arch 75 "
                f"--model-set custom --models breeze_tts --target audiocpp_cli", cwd=src, timeout=5400), "build")
    exe = find_bin(src / "build", "audiocpp_cli")
    model = hf_hub_download("Raxdiam/Breeze-TTS-2-GGUF", "breeze-tts-2-q4_k.gguf", local_dir=str(M / "audiocpp"), token=HF)
    run_cases("audiocpp", lambda p, v, wav, vw, vt, s: sh([
        exe, "--task", "clon", "--family", "breeze_tts", "--model", model, "--backend", "cuda",
        "--text", p["input"], "--voice-ref", vw, "--reference-text", vt,
        "--request-option", "instruct=" + p["instructions"].strip(), "--request-option", f"seed={s}",
        "--request-option", "guidance_scale=1", "--out", wav], timeout=CASE_TIMEOUT), t_end)
    return {"rev": sh(f"git -C {src} rev-parse HEAD").stdout.strip()}


# ---- ljchang/qwentts.cpp@breeze ----
@arm("qwentts")
def _(t_end):
    src = TMP / "src_qwentts"
    clone("https://github.com/ljchang/qwentts.cpp.git", src, branch="breeze", submodules=True)
    patch_cuda_driver(src)
    bd = src / "build"
    must(sh(f"cmake -S {src} -B {bd} -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=75",
            timeout=1800), "configure")
    with kh.build_heartbeat("build_qwentts", interval_s=120):
        must(sh(f"cmake --build {bd} -j{JOBS} --target qwen-tts", timeout=3600), "build")
    exe = find_bin(bd, "qwen-tts")
    model = hf_hub_download("mecha-factory/mecha-speech-gguf", "breeze-tts2-Q5_K_M.gguf", local_dir=str(M / "qwentts"), token=HF)
    codec = hf_hub_download("mecha-factory/mecha-speech-gguf", "qwen-tokenizer-12hz-BF16.gguf", local_dir=str(M / "qwentts"), token=HF)
    run_cases("qwentts", lambda p, v, wav, vw, vt, s: sh([
        exe, "--model", model, "--codec", codec, "--instruct", p["instructions"].strip(), "--ref-wav", vw,
        "--ref-text", str(TMP / f"{v}.ref.txt"), "--seed", str(s), "-o", wav],
        input=p["input"], timeout=CASE_TIMEOUT), t_end)
    return {"rev": sh(f"git -C {src} rev-parse HEAD").stdout.strip()}


# ---- MLX ports, on MLX's Linux CUDA backend, in their own venv ----
mlx_state = {}


def mlx_env():
    if "ok" in mlx_state:
        return mlx_state["ok"]
    # Kaggle's python has no ensurepip, so `python -m venv` fails; virtualenv bundles pip.
    shutil.rmtree(MV, ignore_errors=True)
    if sh(f"{sys.executable} -m venv {MV}").returncode:
        shutil.rmtree(MV, ignore_errors=True)
        must(sh(f"{sys.executable} -m pip install -q virtualenv", timeout=600), "virtualenv install")
        must(sh(f"{sys.executable} -m virtualenv -q {MV}", timeout=600), "venv")
    log = []
    for spec in ('"mlx[cuda12]"', '"mlx[cuda]"'):
        r = sh(f"{MPY} -m pip install -q {spec}", timeout=1800)
        log.append(f"{spec}: rc={r.returncode} {r.stderr[-400:]}")
        if r.returncode == 0:
            break
    r = sh([MPY, "-c", "import mlx.core as mx; a=mx.ones((256,256)); print(mx.default_device(), float((a@a).sum()))"], timeout=600)
    log.append(f"probe rc={r.returncode} out={r.stdout[-300:]} err={r.stderr[-800:]}")
    mlx_state["ok"] = r.returncode == 0 and "gpu" in r.stdout.lower()
    res["mlx_probe"] = log; save()
    return mlx_state["ok"]


MLXAUDIO_SNIP = r'''
import sys, json, numpy as np, soundfile as sf
from mlx_audio.tts.utils import load_model
m = load_model(sys.argv[1])
for line in sys.stdin:
    a = json.loads(line)
    try:
        chunks = [np.asarray(r.audio) for r in m.generate(text=a["text"], instruct=a["instr"], ref_audio=a["ref"],
                  ref_text=a["ref_text"], cfg_scale=1.0, seed=a["seed"])]
        sf.write(a["out"], np.concatenate(chunks) if chunks else np.zeros(1), m.sample_rate)
        print(json.dumps({"ok": a["out"]}), flush=True)
    except Exception as e:
        print(json.dumps({"err": repr(e)[-500:]}), flush=True)
'''


class Worker:
    """Keeps one model loaded across cases: JSON line in, JSON line out."""
    def __init__(self, cmd):
        self.p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=open(TMP / "worker.err", "a"),
                                  text=True, bufsize=1)

    def __call__(self, req):
        self.p.stdin.write(json.dumps(req) + "\n"); self.p.stdin.flush()
        line = self.p.stdout.readline()
        if not line:
            raise RuntimeError("worker died: " + (TMP / "worker.err").read_text()[-1500:])
        out = json.loads(line)
        if "err" in out:
            raise RuntimeError(out["err"])
        return out

    def close(self):
        self.p.kill()


@arm("mlxaudio")
def _(t_end):
    if not mlx_env():
        raise RuntimeError("mlx[cuda] unusable on this GPU, see mlx_probe")
    must(sh(f"{MPY} -m pip install -q 'git+https://github.com/Blaizzy/mlx-audio.git' soundfile", timeout=1800), "pip mlx-audio")
    sh(f"{MPY} -m pip install -q 'mlx[cuda12]'", timeout=1800)  # a dependency may have pulled a CPU-only mlx
    path = snapshot_download("mlx-community/Breeze-TTS-2-mlx", local_dir=str(M / "mlxaudio"), token=HF)
    snip = TMP / "mlxaudio_worker.py"; snip.write_text(MLXAUDIO_SNIP)
    w = Worker([MPY, str(snip), path])
    try:
        run_cases("mlxaudio", lambda p, v, wav, vw, vt, s: w(
            {"text": p["input"], "instr": p["instructions"].strip(), "ref": vw, "ref_text": vt, "seed": s, "out": wav}), t_end)
    finally:
        w.close()


@arm("rishikksh")
def _(t_end):
    if not mlx_env():
        raise RuntimeError("mlx[cuda] unusable on this GPU, see mlx_probe")
    src = TMP / "src_rishikksh"
    clone("https://github.com/rishikksh20/breeze-tts-mlx.git", src)
    must(sh(f"{MPY} -m pip install -q -e {src}", timeout=1800), "pip install")
    sh(f"{MPY} -m pip install -q 'mlx[cuda12]'", timeout=1800)
    path = snapshot_download("rishikksh20/Breeze-TTS-2-mlx", local_dir=str(M / "rishikksh"), token=HF)
    run_cases("rishikksh", lambda p, v, wav, vw, vt, s: sh([
        MPY, str(src / "infer.py"), path, "--text", p["input"], "--instruction", p["instructions"].strip(),
        "--ref-audio", vw, "--ref-text", vt, "--seed", str(s), "--cfg-scale", "1", "--output", wav],
        cwd=src, timeout=CASE_TIMEOUT), t_end)
    return {"rev": sh(f"git -C {src} rev-parse HEAD").stdout.strip()}


@arm("vanch")
def _(t_end):
    if not mlx_env():
        raise RuntimeError("mlx[cuda] unusable on this GPU, see mlx_probe")
    src = TMP / "src_vanch"
    clone("https://github.com/vanch007/mlx-breeze-tts2.git", src)
    must(sh(f"{MPY} -m pip install -q -e {src}", timeout=1800), "pip install")
    sh(f"{MPY} -m pip install -q 'mlx[cuda12]'", timeout=1800)
    path = snapshot_download("vanch007/Sirocco-MLX-8bit", local_dir=str(M / "vanch"), token=HF)
    exe = str(MV / "bin" / "mlx-breeze-tts2")
    run_cases("vanch", lambda p, v, wav, vw, vt, s: sh([
        exe, "generate", "--model", path, "--text", p["input"], "--instruction", p["instructions"].strip(),
        "--ref-audio", vw, "--ref-text", vt, "--seed", str(s), "--cfg-scale", "1", "--output", wav],
        timeout=CASE_TIMEOUT), t_end)
    return {"rev": sh(f"git -C {src} rev-parse HEAD").stdout.strip()}


shutil.rmtree(MV, ignore_errors=True)  # the MLX venv (several GB) is not needed past this point

# ---- Liyulingyue/rust-model-inference (CPU only) ----
@arm("rust")
def _(t_end):
    src = TMP / "src_rust"
    clone("https://github.com/Liyulingyue/rust-model-inference.git", src)
    if not shutil.which("cargo"):
        must(sh("curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal", timeout=1200), "rustup")
    env = os.environ | {"PATH": f"{Path.home()}/.cargo/bin:" + os.environ["PATH"]}
    with kh.build_heartbeat("build_rust", interval_s=120):
        must(sh(f"cargo build --release --bin rust-model-inference", cwd=src, env=env, timeout=3600), "cargo build")
    exe = find_bin(src / "target" / "release", "rust-model-inference")
    snap = snapshot_download("BreezeBlue/Breeze-TTS-2", local_dir=str(M / "rust" / "hf"), token=HF)
    out = M / "rust" / "gguf"
    must(sh(f"{sys.executable} {src}/tools/converter/breeze/convert_breeze_plain.py {snap} --out-dir {out}",
            timeout=3600), "convert")
    shutil.rmtree(M / "rust" / "hf", ignore_errors=True)
    model = str(next(out.glob("*BF16.gguf"))); mmproj = str(next(out.glob("*mmproj*.gguf")))
    synth = lambda p, v, wav, vw, vt, s: sh([
        exe, "--model", model, "--mmproj", mmproj, "--tts", "--prompt", p["input"],
        "--instruction", p["instructions"].strip(), "--cfg-scale", "1", "--ref-audio", vw, "--ref-text", vt,
        "--max-tokens", "1500", "--seed", str(s), "--out", wav], env=env, timeout=1800)
    # One smoke case; at CPU speed a full arm only runs if a case takes under 4 minutes.
    t0 = time.time()
    run_cases("rust", synth, t_end, cases=1)
    smoke = time.time() - t0
    info = {"smoke_s": round(smoke, 1), "note": "CPU only; the port's own default cap is --max-tokens 128"}
    if smoke < 240:
        res["arms"]["rust"]["truncated"] = False
        res["runs"] = [r for r in res["runs"] if r["arm"] != "rust"]
        run_cases("rust", synth, t_end)
    return info


# ---- final ----
summ = summarize()
res["finished"] = time.strftime("%F %T")
save()
kh.step("done", free_gb=free_gb())
mirror()
print(json.dumps({"arms": res["arms"], "summary": summ}, indent=1))
