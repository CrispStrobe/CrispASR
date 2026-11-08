#!/usr/bin/env python3
"""#585: talker + code predictor on CPU, codec on Vulkan (lavapipe).

Builds crispasr with GGML_VULKAN on a CPU worker, runs Qwen3-TTS ICL voice
cloning three ways (split, all-CPU control, VULKAN_CPU=1 alone) and checks:
the split keeps the codec weights on Vulkan0 while the talker is pinned to
CPU, reference encoding works there (the reporter's first crash), and every
output round-trips through parakeet ASR.
"""
import glob, json, os, subprocess, sys, traceback
from pathlib import Path
OUT = Path("/kaggle/working/out"); OUT.mkdir(parents=True, exist_ok=True)
REPO = Path("/tmp/CrispASR"); M = Path("/tmp/m"); M.mkdir(exist_ok=True)
REF = os.environ.get("CRISPASR_REF", "main")
TEXT = "Good evening. I am the voice assistant running on this workstation."
res = {"errors": [], "arms": {}}
def save(): (OUT / "result.json").write_text(json.dumps(res, indent=1))
def sh(cmd, **kw): print("+", cmd, flush=True); subprocess.check_call(cmd, shell=True, **kw)
try:
    sh("apt-get update -qq && apt-get install -y -qq libvulkan-dev spirv-headers glslc mesa-vulkan-drivers vulkan-tools ffmpeg ninja-build >/dev/null")
    icd = sorted(glob.glob("/usr/share/vulkan/icd.d/lvp_icd*.json"))[0]
    os.environ["VK_ICD_FILENAMES"] = icd; os.environ["GGML_VK_VISIBLE_DEVICES"] = "0"
    res["vulkaninfo"] = subprocess.run("vulkaninfo --summary | grep -iE 'deviceName|deviceType'", shell=True,
                                       capture_output=True, text=True).stdout
    subprocess.check_call(["git", "clone", "--depth", "1", "-b", REF, "https://github.com/CrispStrobe/CrispASR.git", str(REPO)])
    subprocess.run(["git", "submodule", "update", "--init", "--recursive", "--depth", "1"], cwd=str(REPO))
    res["commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(REPO), text=True).strip(); save()
    sh(f"cmake -S {REPO} -B {REPO}/build -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_VULKAN=ON -DCRISPASR_BUILD_TESTS=OFF "
       "-DCRISPASR_OPUS=OFF -DCRISPASR_AMR=OFF >/dev/null")
    sh(f"cmake --build {REPO}/build --target crispasr-cli -j$(nproc) >/dev/null")
    base = "https://huggingface.co/cstr"
    for f, u in (("tts.gguf", "qwen3-tts-0.6b-base-GGUF/resolve/main/qwen3-tts-12hz-0.6b-base.gguf"),
                 ("codec.gguf", "qwen3-tts-tokenizer-12hz-GGUF/resolve/main/qwen3-tts-tokenizer-12hz.gguf"),
                 ("asr.gguf", "parakeet-tdt-0.6b-v2-GGUF/resolve/main/parakeet-tdt-0.6b-v2-q4_k.gguf")):
        sh(f"curl -sSfL -o {M/f} {base}/{u}")
    sh(f"ffmpeg -loglevel error -y -i {REPO}/samples/jfk.wav -ar 24000 {M}/jfk24.wav")
    B = str(REPO / "build/bin/crispasr")
    common = [B, "--backend", "qwen3-tts", "-m", str(M / "tts.gguf"), "--codec-model", str(M / "codec.gguf"),
              "--voice", str(M / "jfk24.wav"), "--ref-text",
              "And so my fellow Americans, ask not what your country can do for you, ask what you can do for your country.",
              "--i-have-rights", "--tts", TEXT, "--seed", "42", "-v"]
    arms = {
        "split": (["--gpu-backend", "vulkan"], {"CRISPASR_QWEN3_TTS_VULKAN_CPU": "1", "CRISPASR_QWEN3_TTS_CODEC_GPU": "1"}),
        "split_alias": (["--gpu-backend", "vulkan"], {"CRISPASR_QWEN3_TTS_VULKAN_CPU": "1", "CRISPASR_QWEN3_TTS_CODEC_VULKAN": "1"}),
        "vulkan_cpu_only": (["--gpu-backend", "vulkan"], {"CRISPASR_QWEN3_TTS_VULKAN_CPU": "1"}),
        "cpu": (["-ng"], {}),
    }
    for name, (extra, envx) in arms.items():
        wav = M / f"{name}.wav"
        env = dict(os.environ, CRISPASR_QWEN3_TTS_MAX_FRAMES="200", **envx)
        r = subprocess.run(common + extra + ["--tts-output", str(wav)], capture_output=True, text=True, env=env, timeout=3600)
        log = r.stdout + r.stderr
        (OUT / f"{name}.log").write_text(log)
        a = subprocess.run([B, "--backend", "parakeet", "-m", str(M / "asr.gguf"), "-f", str(wav), "-np", "-nt", "-ng"],
                           capture_output=True, text=True)
        t = a.stdout.strip()
        res["arms"][name] = {
            "rc": r.returncode, "asr": t,
            "roundtrip": "voice assistant" in t.lower() and "evening" in t.lower(),
            "talker_on_cpu": "running the talker on CPU" in log,
            "codec_stays": "codec stays on Vulkan0" in log,
            "codec_weights_on_vulkan": "loading weights onto Vulkan0" in log,
            "codec_lines": [l for l in log.splitlines() if "codec" in l.lower()][:8],
        }
        save()
    a = res["arms"]
    res["pass"] = all([
        all(x["rc"] == 0 and x["roundtrip"] for x in a.values()),
        all(a[k]["talker_on_cpu"] and a[k]["codec_stays"] and a[k]["codec_weights_on_vulkan"] for k in ("split", "split_alias")),
        a["vulkan_cpu_only"]["talker_on_cpu"] and not a["vulkan_cpu_only"]["codec_weights_on_vulkan"],
    ])
except BaseException:
    res["errors"].append(traceback.format_exc())
finally:
    save()
