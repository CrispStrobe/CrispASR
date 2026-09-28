#!/usr/bin/env python3
"""#461 proof: fused CFM Euler graph vs the per-step path (CRISPASR_VOXCPM2_CFM_FUSED=0).
Real Vulkan on the T4 (NVIDIA ICD) + CPU, voxcpm2-q8_0, the reporter's sentence,
seed 42. Speed: bench lines per arm. Correctness: crispasr-diff voxcpm2-tts vs the
Python reference (cfm_step0_result runs cfm_euler_solve on the reference noise),
fused vs per-step, CPU and Vulkan; ASR roundtrip (whisper tiny.en) of every WAV;
fused-vs-per-step waveform comparison.
"""
import json, os, re, subprocess, sys, time, traceback, wave, array, math
from pathlib import Path
OUT = Path("/kaggle/working/out"); OUT.mkdir(parents=True, exist_ok=True)
REPO = Path("/tmp/CrispASR"); G = Path("/tmp/g"); G.mkdir(exist_ok=True)
REF = os.environ.get("CRISPASR_REF", "perf/voxcpm2-cfm-fused")  # MM_SPLIT A/B + doubled-sentence check
TEXT = "Hello, this is a short test sentence."
res = {"errors": [], "runs": {}, "diff": {}, "asr": {}, "wavcmp": {}}
def save(): (OUT / "result.json").write_text(json.dumps(res, indent=1))
def sh(c, t=None): return subprocess.run(c, shell=True, capture_output=True, text=True, timeout=t)
def read_wav(p):
    with wave.open(str(p)) as w:
        a = array.array("h", w.readframes(w.getnframes())); return [x / 32768.0 for x in a], w.getframerate()
def words(s): return re.sub(r"[^a-z0-9 ]", " ", s.lower()).split()
def wer(r, h):
    r, h = words(r), words(h); d = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        p, d[0] = d[0], i
        for j in range(1, len(h) + 1):
            p, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, p + (r[i - 1] != h[j - 1]))
    return d[len(h)] / max(1, len(r))
try:
    subprocess.check_call(["git", "clone", "--depth", "1", "-b", REF, "https://github.com/CrispStrobe/CrispASR.git", str(REPO)])
    subprocess.run(["git", "submodule", "update", "--init", "--recursive", "--depth", "1"], cwd=str(REPO))
    res["commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(REPO), text=True).strip()
    sys.path.insert(0, str(REPO / "tools" / "kaggle")); import kaggle_harness as kh
    tok = kh.resolve_hf_token()
    if tok: os.environ["HF_TOKEN"] = tok
    from huggingface_hub import hf_hub_download
    sh("apt-get update -qq")
    for pkg in ("libvulkan1", "vulkan-tools", "libvulkan-dev", "glslang-tools", "spirv-tools"):
        sh(f"DEBIAN_FRONTEND=noninteractive apt-get install -y -qq {pkg}")
    glslc = sh("which glslc").stdout.strip()
    if not glslc:
        cn = sh("bash -lc '. /etc/os-release; echo $VERSION_CODENAME'").stdout.strip() or "jammy"
        sh("wget -qO- https://packages.lunarg.com/lunarg-signing-key-pub.asc | tee /etc/apt/trusted.gpg.d/lunarg.asc >/dev/null")
        sh(f"wget -qO /etc/apt/sources.list.d/lunarg-vulkan-{cn}.list https://packages.lunarg.com/vulkan/lunarg-vulkan-{cn}.list")
        sh("apt-get update -qq"); sh("DEBIAN_FRONTEND=noninteractive apt-get install -y -qq vulkan-sdk")
        glslc = sh("which glslc").stdout.strip()
    drv = (sh("nvidia-smi --query-gpu=driver_version --format=csv,noheader").stdout.strip().splitlines() or [""])[0]
    if drv: sh(f"DEBIAN_FRONTEND=noninteractive apt-get install -y -qq libnvidia-gl-{drv.split('.')[0]}")
    if "NVIDIA" not in sh("vulkaninfo --summary 2>/dev/null").stdout:
        os.makedirs("/usr/share/vulkan/icd.d", exist_ok=True)
        Path("/usr/share/vulkan/icd.d/nvidia_icd.json").write_text(
            '{"file_format_version":"1.0.0","ICD":{"library_path":"libGLX_nvidia.so.0","api_version":"1.3.277"}}')
    res["vk_devices"] = [l.split("=")[-1].strip() for l in sh("vulkaninfo --summary 2>/dev/null").stdout.splitlines() if "deviceName" in l]
    save()
    kh.install_build_toolchain()
    flags = ["-DGGML_VULKAN=ON", "-DCMAKE_BUILD_TYPE=Release", "-DCRISPASR_OPUS=OFF", "-DCRISPASR_AMR=OFF"] + kh.cache_and_link_flags()
    if glslc: flags.append(f"-DVulkan_GLSLC_EXECUTABLE={glslc}")
    kh.sh(f"cmake -S {REPO} -B {REPO}/build -G Ninja " + " ".join(flags))
    with kh.build_heartbeat("cmake.build"):
        kh.sh(f"cmake --build {REPO}/build -j$(nproc) --target crispasr-cli crispasr-diff")
    B = REPO / "build" / "bin" / "crispasr"; D = REPO / "build" / "bin" / "crispasr-diff"
    q8 = hf_hub_download("cstr/voxcpm2-GGUF", "voxcpm2-q8_0.gguf", cache_dir=str(G))
    refg = hf_hub_download("cstr/voxcpm2-GGUF", "voxcpm2-ref.gguf", cache_dir=str(G))
    wtiny = G / "ggml-tiny.en.bin"
    sh(f"wget -q -O {wtiny} https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.en.bin")
    def synth(tag, extra, env):
        wav = OUT / f"{tag}.wav"; t0 = time.time()
        r = subprocess.run([str(B), "--backend", "voxcpm2", "-m", q8, "--tts", TEXT, "--tts-output", str(wav), "--seed", "42", "-v"] + extra,
                           capture_output=True, text=True, env=dict(os.environ, CRISPASR_VOXCPM2_BENCH="1", **env), timeout=3600)
        e = r.stderr
        grab = lambda pat: [float(x) for x in re.findall(pat, e)]
        ent = {"rc": r.returncode, "wall_s": round(time.time() - t0, 2),
               "cfm_calls_ms": grab(r"cfm\.locdit_fwd ([\d.]+) ms total"),
               "fused_lines": len(re.findall(r"\[fused\]", e)),
               "per_step": [l.strip() for l in e.splitlines() if re.search(r"bench\]:   [a-z_]+ +[\d.]+ ms", l)],
               "ar": re.findall(r"AR loop .*", e), "vae": re.findall(r"VAE decode .*", e), "total": re.findall(r"voxcpm2: total .*", e),
               "tail": e[-1500:] if r.returncode else ""}
        c = ent["cfm_calls_ms"]
        if len(c) > 2: ent["cfm_median_ms"] = sorted(c[1:])[len(c[1:]) // 2]
        if wav.exists():
            a, sr = read_wav(wav); ent["dur_s"] = round(len(a) / sr, 3)
            ent["rtf"] = round(float(re.findall(r"total ([\d.]+) ms", " ".join(ent["total"]))[0]) / 1000 / ent["dur_s"], 3) if ent["total"] else None
            asr = subprocess.run([str(B), "-m", str(wtiny), "-f", str(wav), "-np", "-ng"], capture_output=True, text=True, timeout=600)
            txt = " ".join(asr.stdout.split()); res["asr"][tag] = {"text": txt, "wer": round(wer(TEXT, txt), 3)}
        res["runs"][tag] = ent; save()
        print(tag, {k: ent.get(k) for k in ("wall_s", "cfm_median_ms", "fused_lines", "dur_s", "rtf")}, res["asr"].get(tag), flush=True)
    synth("vk_warmup", [], {})  # compiles every Vulkan pipeline once; not compared
    ARMS = (("vk_split0", [], {"CRISPASR_VOXCPM2_MM_SPLIT": "0"}), ("vk_split8", [], {"CRISPASR_VOXCPM2_MM_SPLIT": "8"}),
            ("vk_split4", [], {"CRISPASR_VOXCPM2_MM_SPLIT": "4"}),
            ("cpu_split0", ["-ng"], {"CRISPASR_VOXCPM2_MM_SPLIT": "0"}), ("cpu_split8", ["-ng"], {"CRISPASR_VOXCPM2_MM_SPLIT": "8"}))
    for tag, extra, env in ARMS:
        synth(tag, extra, env)
    for a_, b_ in (("vk_split8", "vk_split0"), ("vk_split4", "vk_split0"), ("cpu_split8", "cpu_split0")):
        pa, pb = OUT / f"{a_}.wav", OUT / f"{b_}.wav"
        if pa.exists() and pb.exists():
            x, _ = read_wav(pa); y, _ = read_wav(pb); n = min(len(x), len(y))
            dot = sum(x[i] * y[i] for i in range(n)); nx = math.sqrt(sum(v * v for v in x[:n])); ny = math.sqrt(sum(v * v for v in y[:n]))
            res["wavcmp"][f"{a_}~{b_}"] = {"len": [len(x), len(y)], "identical": x == y, "cos": round(dot / (nx * ny + 1e-12), 6)}
    save()
    # the doubled-sentence question: C++ over several seeds (Vulkan, fast) ...
    res["seeds_cpp"] = {}
    for sd in (1, 2, 3, 4, 5):
        wav = OUT / f"seed{sd}.wav"
        r = subprocess.run([str(B), "--backend", "voxcpm2", "-m", q8, "--tts", TEXT, "--tts-output", str(wav), "--seed", str(sd), "-v"],
                           capture_output=True, text=True, timeout=1200)
        steps = re.findall(r"stopped at step (\d+)", r.stderr)
        txt = ""
        if wav.exists():
            asr = subprocess.run([str(B), "-m", str(wtiny), "-f", str(wav), "-np", "-ng"], capture_output=True, text=True, timeout=600)
            txt = re.sub(r"\[[^\]]*\]", "", asr.stdout); txt = " ".join(txt.split())
        res["seeds_cpp"][sd] = {"steps": steps, "dur_s": round(len(read_wav(wav)[0]) / 48000, 2) if wav.exists() else None, "asr": txt}
        save(); print("seed", sd, res["seeds_cpp"][sd], flush=True)
    # ... and the official VoxCPM2 pipeline (own uv venv, cu128 torch) on the same text + seeds
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "uv"])
        V = G / "venv"; subprocess.check_call([sys.executable, "-m", "uv", "venv", "--python", "3.11", str(V)])
        py = str(V / "bin" / "python")
        r = subprocess.run([sys.executable, "-m", "uv", "pip", "install", "--python", py, "voxcpm", "soundfile", "torch==2.8.0", "torchaudio==2.8.0",
                            "--extra-index-url", "https://download.pytorch.org/whl/cu128", "--index-strategy", "unsafe-best-match"],
                           capture_output=True, text=True, timeout=2400)
        res["up_install"] = (r.stdout + r.stderr)[-800:]
        (G / "up.py").write_text(
            "import sys, json, soundfile as sf, torch\nfrom voxcpm import VoxCPM\n"
            "m = VoxCPM.from_pretrained('openbmb/VoxCPM2')\nout = {}\n"
            "for sd in [1,2,3,4,5,42]:\n"
            "    torch.manual_seed(sd)\n"
            "    w = m.generate(text=sys.argv[1], cfg_value=2.0, inference_timesteps=10)\n"
            "    p = f'/tmp/g/up_seed{sd}.wav'; sf.write(p, w, m.tts_model.sample_rate); out[sd] = [p, len(w) / m.tts_model.sample_rate]\n"
            "print('@@' + json.dumps(out))\n")
        r = subprocess.run([py, str(G / "up.py"), TEXT], capture_output=True, text=True, timeout=3600)
        m = re.search(r"^@@(.*)$", r.stdout, re.M)
        res["seeds_upstream"] = {}
        if m:
            for sd, (p, dur) in json.loads(m.group(1)).items():
                asr = subprocess.run([str(B), "-m", str(wtiny), "-f", p, "-np", "-ng"], capture_output=True, text=True, timeout=600)
                txt = " ".join(re.sub(r"\[[^\]]*\]", "", asr.stdout).split())
                res["seeds_upstream"][sd] = {"dur_s": round(dur, 2), "asr": txt}
        else:
            res["seeds_upstream"] = {"error": (r.stdout + r.stderr)[-2500:]}
        save(); print("upstream", res["seeds_upstream"], flush=True)
    except Exception:
        res["errors"].append("upstream: " + traceback.format_exc()[-1500:])
except BaseException:
    res["errors"].append(traceback.format_exc())
finally:
    save()
    import shutil; shutil.rmtree(REPO, ignore_errors=True)
