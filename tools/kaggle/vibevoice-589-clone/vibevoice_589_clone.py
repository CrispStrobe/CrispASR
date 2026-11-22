#!/usr/bin/env python3
"""
Issue #589 follow-up: after ac603e69 the reporter (RX 6800, RADV, 7B q8_0)
says ASR works but voice cloning still doesn't. Check base-model cloning on a
real Vulkan driver (NVIDIA T4) built from main.

Arms (all --gpu-backend vulkan, fixed seed, samples/jfk.wav as reference):
  15_C / 15_B         1.5B q4_k, cloned vs no voice (generic)
  7b_C / 7b_B         7B q8_0 (the reporter's file), cloned vs no voice
  7b_C_noidot         7B q8_0 cloned with GGML_VK_DISABLE_INTEGER_DOT_PRODUCT=1
Each output is ASR-roundtripped with whisper base.en (CPU) and scored with
Resemblyzer: cos(C,ref) vs cos(B,ref); delta > 0.05 counts as cloned.
WAVs are left in /kaggle/working for listening.
"""
import os, sys, subprocess, json, re, wave, array, math, shutil
from pathlib import Path

WORK = Path("/kaggle/working")
TMP = Path("/kaggle/temp/i589"); TMP.mkdir(parents=True, exist_ok=True)
MODELS = Path("/kaggle/temp/models"); MODELS.mkdir(parents=True, exist_ok=True)
RESULTS = WORK / "i589_results.json"

HERE = Path(__file__).resolve().parent
CRISPASR_URL = "https://github.com/CrispStrobe/CrispASR.git"
BRANCH = "main"
CLONE = Path("/kaggle/temp/CrispASR")
if not CLONE.exists():
    try:
        subprocess.run(["git", "clone", "--depth", "1", "--branch", BRANCH,
                        "--recurse-submodules", CRISPASR_URL, str(CLONE)],
                       check=True, timeout=1200)
    except Exception as e:
        print(f"clone failed: {e}", flush=True)
sys.path.insert(0, str(CLONE / "tools" / "kaggle") if CLONE.exists() else str(HERE))
import kaggle_harness as kh
kh.init_progress()

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "huggingface_hub", "resemblyzer"], check=False)
HF_TOKEN = kh.resolve_hf_token()
kh.step("hf.token", present=bool(HF_TOKEN))
from huggingface_hub import hf_hub_download

SENT = "The quick brown fox jumps over the lazy dog near the river bank."
SEED = 42
REF = CLONE / "samples" / "jfk.wav"
MODELS_LIST = [
    ("cstr/vibevoice-1.5b-GGUF", "vibevoice-1.5b-tts-q4_k.gguf"),
    ("cstr/VibeVoice-7B-GGUF", "vibevoice-7b-q8_0.gguf"),
    ("ggerganov/whisper.cpp", "ggml-base.en.bin"),
]


def sh(cmd, timeout=None, env=None):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True,
                          timeout=timeout, env=env)


def install_vulkan_sdk():
    """Runtime + dev toolchain for building ggml-vulkan (glslc is LunarG-only)."""
    for pkg in ("libvulkan1", "vulkan-tools", "libvulkan-dev", "glslang-tools", "spirv-tools"):
        sh(f"DEBIAN_FRONTEND=noninteractive apt-get install -y -qq {pkg}")
    glslc = sh("which glslc").stdout.strip()
    if not glslc:
        codename = sh("bash -lc '. /etc/os-release; echo $VERSION_CODENAME'").stdout.strip() or "jammy"
        sh("wget -qO- https://packages.lunarg.com/lunarg-signing-key-pub.asc "
           "| tee /etc/apt/trusted.gpg.d/lunarg.asc >/dev/null")
        sh(f"wget -qO /etc/apt/sources.list.d/lunarg-vulkan-{codename}.list "
           f"https://packages.lunarg.com/vulkan/lunarg-vulkan-{codename}.list")
        sh("apt-get update -qq")
        sh("DEBIAN_FRONTEND=noninteractive apt-get install -y -qq vulkan-sdk")
        glslc = sh("which glslc").stdout.strip()
        if not glslc:
            sh("DEBIAN_FRONTEND=noninteractive apt-get install -y -qq shaderc glslc")
            glslc = sh("which glslc").stdout.strip()
    kh.step("vulkan.glslc", path=glslc)
    if glslc:
        os.environ["GLSLC_PATH"] = glslc
    return glslc


def enable_vulkan():
    kh.step("vulkan.install")
    sh("apt-get update -qq")
    install_vulkan_sdk()
    smi = sh("nvidia-smi --query-gpu=driver_version --format=csv,noheader")
    drv = (smi.stdout.strip().splitlines() or [""])[0]
    major = drv.split(".")[0] if drv else ""
    kh.step("vulkan.nvidia_driver", driver=drv)
    if major:
        sh(f"DEBIAN_FRONTEND=noninteractive apt-get install -y -qq libnvidia-gl-{major}")
    have_glx = bool(sh("ldconfig -p | grep -F libGLX_nvidia.so.0").stdout.strip())
    if have_glx and "NVIDIA" not in sh("vulkaninfo --summary 2>/dev/null").stdout:
        os.makedirs("/usr/share/vulkan/icd.d", exist_ok=True)
        Path("/usr/share/vulkan/icd.d/nvidia_icd.json").write_text(
            '{"file_format_version":"1.0.0","ICD":'
            '{"library_path":"libGLX_nvidia.so.0","api_version":"1.3.277"}}')
    r = sh("vulkaninfo --summary 2>/dev/null")
    devs = [l.split("=")[-1].strip() for l in r.stdout.splitlines() if "deviceName" in l]
    kh.step("vulkan.devices", devices=devs)
    return devs, any("NVIDIA" in d for d in devs)


def build_crispasr():
    kh.install_build_toolchain()
    if not (CLONE / "CMakeLists.txt").exists():
        raise RuntimeError("repo clone missing")
    if not (CLONE / "ggml" / "CMakeLists.txt").exists():
        sh(f"cd {CLONE} && git submodule update --init ggml", timeout=900)
    flags = ["-DCMAKE_BUILD_TYPE=Release", "-DGGML_VULKAN=ON", "-DGGML_CUDA=OFF",
             "-DGGML_NATIVE=OFF", "-DGGML_AVX2=ON", "-DGGML_FMA=ON", "-DGGML_F16C=ON",
             "-DCRISPASR_OPUS_FETCH=ON"] + kh.cache_and_link_flags()
    glslc = os.environ.get("GLSLC_PATH")
    if glslc:
        flags.append(f"-DVulkan_GLSLC_EXECUTABLE={glslc}")
    r = sh(f"cd {CLONE} && cmake -G Ninja -B build-vk " + " ".join(flags), timeout=1200)
    if r.returncode != 0:
        kh.step("build.configure_FAILED", stderr=r.stderr[-2000:])
        raise RuntimeError("cmake configure failed")
    jobs = kh.safe_build_jobs(gpu=False)
    with kh.build_heartbeat("build.ninja", interval_s=30):
        kh.sh_with_progress(f"cmake --build build-vk -j{jobs} --target crispasr-cli", cwd=str(CLONE))
    binp = CLONE / "build-vk" / "bin" / "crispasr"
    if not binp.exists():
        raise RuntimeError("crispasr binary not produced")
    os.environ["LD_LIBRARY_PATH"] = str(binp.parent) + ":" + os.environ.get("LD_LIBRARY_PATH", "")
    kh.step("build.ready", path=str(binp))
    try:
        kh.export_ccache_tar()
    except Exception:
        pass
    return str(binp)


def wav_stats(path):
    try:
        w = wave.open(str(path), "rb")
        sr = w.getframerate()
        a = array.array("h"); a.frombytes(w.readframes(w.getnframes())); w.close()
        if not len(a):
            return dict(dur=0.0, peak=0.0, rms=0.0)
        peak = max(abs(x) for x in a) / 32768.0
        rms = math.sqrt(sum((x / 32768.0) ** 2 for x in a) / len(a))
        return dict(dur=round(len(a) / sr, 2), peak=round(peak, 4), rms=round(rms, 4))
    except Exception as e:
        return dict(err=str(e))


def asr(binp, whisp, wav):
    if not Path(wav).exists():
        return ""
    r = sh(f"{binp} -m {whisp} -f {wav} --language en --no-gpu", timeout=600)
    txt = " ".join(re.sub(r"\[[^\]]*\]", "", l) for l in r.stdout.splitlines()
                   if l.strip().startswith("["))
    return re.sub(r"\s+", " ", txt).strip()


def word_overlap(ref, hyp):
    rw = set(re.findall(r"[a-z']+", ref.lower()))
    hw = set(re.findall(r"[a-z']+", hyp.lower()))
    return round(len(rw & hw) / len(rw), 3) if rw else 0.0


def run_tts(binp, model, tag, voice=True, env_extra=None, timeout=2400):
    out = TMP / f"{tag}.wav"
    env = dict(os.environ); env.update(env_extra or {})
    cmd = [binp, "-m", str(model), "--tts", SENT, "--seed", str(SEED),
           "--gpu-backend", "vulkan", "--tts-output", str(out)]
    if voice:
        cmd += ["--voice", str(REF), "--i-have-rights"]
    rc, err, timed_out = -1, "", False
    with kh.build_heartbeat(f"tts.{tag}", interval_s=30):
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=timeout)
            rc, err = r.returncode, (r.stdout or "") + (r.stderr or "")
        except subprocess.TimeoutExpired as e:
            timed_out = True
    (WORK / f"{tag}.log").write_text(err)
    if out.exists():
        shutil.copy(out, WORK / f"{tag}.wav")
    idot = next((l for l in err.splitlines() if "int dot" in l), "")
    res = dict(tag=tag, rc=rc, timed_out=timed_out, device=idot.strip(),
               audio=wav_stats(out), wav=str(out),
               stderr_tail=err[-1500:] if rc != 0 else "")
    kh.step(f"tts.{tag}", rc=rc, audio=res["audio"])
    return res


def speaker_cos(paths):
    from resemblyzer import VoiceEncoder, preprocess_wav
    import numpy as np
    enc = VoiceEncoder(device="cpu", verbose=False)
    ref = enc.embed_utterance(preprocess_wav(str(REF)))
    out = {}
    for k, p in paths.items():
        try:
            e = enc.embed_utterance(preprocess_wav(p))
            out[k] = round(float(np.dot(e, ref) / (np.linalg.norm(e) * np.linalg.norm(ref))), 3)
        except Exception as ex:
            out[k] = None
    return out


def main():
    devs, nvidia = enable_vulkan()
    binp = build_crispasr()
    for repo, f in MODELS_LIST:
        with kh.build_heartbeat(f"models.{f}", interval_s=30):
            hf_hub_download(repo_id=repo, filename=f, local_dir=str(MODELS), token=HF_TOKEN or None)
    m15 = MODELS / MODELS_LIST[0][1]
    m7 = MODELS / MODELS_LIST[1][1]
    whisp = MODELS / MODELS_LIST[2][1]

    results = {"env": {"vulkan_devices": devs, "real_nvidia": nvidia}, "runs": {}}
    arms = [("15_C", m15, True, None), ("15_B", m15, False, None),
            ("7b_C", m7, True, None), ("7b_B", m7, False, None),
            ("7b_C_noidot", m7, True, {"GGML_VK_DISABLE_INTEGER_DOT_PRODUCT": "1"})]
    for tag, m, voice, env in arms:
        results["runs"][tag] = run_tts(binp, m, tag, voice, env)
        RESULTS.write_text(json.dumps(results, indent=2))

    for tag, run in results["runs"].items():
        hyp = asr(binp, whisp, run["wav"])
        run["asr"] = hyp
        run["overlap"] = word_overlap(SENT, hyp)
        kh.step(f"asr.{tag}", text=hyp[:120], overlap=run["overlap"])

    cos = speaker_cos({t: r["wav"] for t, r in results["runs"].items() if Path(r["wav"]).exists()})
    for t, c in cos.items():
        results["runs"][t]["cos_ref"] = c
    R = results["runs"]
    def delta(c, b):
        cc, bb = R.get(c, {}).get("cos_ref"), R.get(b, {}).get("cos_ref")
        return None if cc is None or bb is None else round(cc - bb, 3)
    results["verdict"] = {
        "15_delta": delta("15_C", "15_B"),
        "7b_delta": delta("7b_C", "7b_B"),
        "7b_noidot_delta": delta("7b_C_noidot", "7b_B"),
    }
    RESULTS.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2), flush=True)
    kh.step("DONE", **results["verdict"])


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        kh.step("FATAL", err=str(e)[:300])
        traceback.print_exc()
        sys.exit(1)
