#!/usr/bin/env python3
"""#461 voxcpm2 on Vulkan: where does a CFM step go? Real Vulkan on the T4 (NVIDIA
ICD), voxcpm2-q8_0, the reporter's sentence, seed 42. Arms: Vulkan bench,
Vulkan + GGML_VK_PERF_LOGGER (per-op GPU time), Vulkan CFG_BATCH=0, CPU bench.
Outputs: per-stage bench lines, per-op totals, WAVs (for ASR roundtrip).
REF env: the branch/commit under test (A/B by pushing twice).
"""
import json, os, re, subprocess, sys, time, traceback
from pathlib import Path
OUT = Path("/kaggle/working/out"); OUT.mkdir(parents=True, exist_ok=True)
REPO = Path("/tmp/CrispASR"); G = Path("/tmp/g"); G.mkdir(exist_ok=True)
REF = os.environ.get("CRISPASR_REF", "main")
TEXT = "Hello, this is a short test sentence."
res = {"errors": [], "runs": {}}
def save(): (OUT / "result.json").write_text(json.dumps(res, indent=1))
def sh(c, t=None): return subprocess.run(c, shell=True, capture_output=True, text=True, timeout=t)
try:
    subprocess.check_call(["git", "clone", "--depth", "1", "-b", REF, "https://github.com/CrispStrobe/CrispASR.git", str(REPO)])
    subprocess.run(["git", "submodule", "update", "--init", "--recursive", "--depth", "1"], cwd=str(REPO))
    res["commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(REPO), text=True).strip()
    sys.path.insert(0, str(REPO / "tools" / "kaggle")); import kaggle_harness as kh
    tok = kh.resolve_hf_token()
    if tok: os.environ["HF_TOKEN"] = tok
    from huggingface_hub import hf_hub_download
    # Vulkan SDK (glslc) + NVIDIA ICD - recipe from tools/kaggle/cv3-vulkan-convtest
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
        kh.sh(f"cmake --build {REPO}/build -j$(nproc) --target crispasr-cli")
    B = REPO / "build" / "bin" / "crispasr"
    model = hf_hub_download("cstr/voxcpm2-GGUF", "voxcpm2-q8_0.gguf", cache_dir=str(G))
    def run(tag, extra, env=None):
        wav = OUT / f"{tag}.wav"
        t0 = time.time()
        r = subprocess.run([str(B), "--backend", "voxcpm2", "-m", model, "--tts", TEXT, "--tts-output", str(wav),
                            "--seed", "42", "-v"] + extra, capture_output=True, text=True,
                           env=dict(os.environ, CRISPASR_VOXCPM2_BENCH="1", **(env or {})), timeout=3600)
        err = r.stderr
        ent = {"rc": r.returncode, "wall_s": round(time.time() - t0, 2),
               "bench": [l for l in err.splitlines() if re.search(r"bench\]:   [a-z_]+ +[\d.]+ ms|AR loop|VAE decode|total \d|TSLM prefill|step 0 |ggml_vulkan: \d|stopped at", l)][:40],
               "tail": err[-1500:] if r.returncode else ""}
        if env and "GGML_VK_PERF_LOGGER" in env:
            ops = {}
            for m in re.finditer(r"^([A-Z_0-9]+)(?:\([^)]*\))?[^\n]*?:\s*(\d+)\s*x\s*([\d.]+)\s*us", err, re.M):
                k = m.group(1); ops[k] = ops.get(k, 0.0) + int(m.group(2)) * float(m.group(3))
            ent["perf_ops_us"] = dict(sorted(ops.items(), key=lambda kv: -kv[1])[:25])
            ent["perf_sample"] = [l for l in err.splitlines() if "us" in l and re.match(r"^[A-Z_]", l)][:60]
        res["runs"][tag] = ent; save()
        print(tag, ent["wall_s"], ent["bench"][-12:], flush=True)
    run("vk", [])
    run("vk_nobatch", [], {"CRISPASR_VOXCPM2_CFG_BATCH": "0"})
    run("vk_perf", [], {"GGML_VK_PERF_LOGGER": "1"})
    run("cpu", ["-ng"])
except BaseException:
    res["errors"].append(traceback.format_exc())
finally:
    save()
    import shutil; shutil.rmtree(REPO, ignore_errors=True)
