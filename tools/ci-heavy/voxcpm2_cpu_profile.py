#!/usr/bin/env python3
"""VoxCPM2 CPU profile: why is F16 ~2.4x slower than Q8_0 on CPU?

Runs the same sentence/seed with F16 and Q8_0 GGUFs and CRISPASR_VOXCPM2_BENCH=1
(per-stage timings), plus `perf record` hot symbols when perf is installable.

  gh workflow run heavy-cpu.yml --ref <branch> -f script=tools/ci-heavy/voxcpm2_cpu_profile.py \
     -f pip="huggingface_hub" -f timeout_minutes=120
"""
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sweep_controls_common as C  # noqa: E402

TEXT = "Hello, this is a short test sentence."


def main():
    bd = C.SCRATCH / "build"
    C.run(["cmake", "-S", C.ROOT, "-B", bd, "-G", "Ninja", "-DCMAKE_BUILD_TYPE=RelWithDebInfo", "-DGGML_NATIVE=OFF",
           "-DBUILD_SHARED_LIBS=OFF"], C.OUT / "cmake.log")
    C.run(["cmake", "--build", bd, "-j", str(os.cpu_count() or 4), "--target", "crispasr"], C.OUT / "build.log")
    b = bd / "bin" / "crispasr"
    subprocess.run("sudo apt-get install -y -q linux-tools-common linux-tools-$(uname -r) >/dev/null 2>&1; "
                   "sudo sysctl -w kernel.perf_event_paranoid=-1 >/dev/null 2>&1", shell=True)
    perf = subprocess.run("command -v perf && perf --version", shell=True, capture_output=True, text=True).returncode == 0
    env = dict(os.environ, CRISPASR_VOXCPM2_BENCH="1")
    nt = str(os.cpu_count() or 4)
    res = {"perf": perf, "arms": {}}
    for q in ["f16", "q8_0"]:
        g = C.hf("cstr/voxcpm2-GGUF", f"voxcpm2-{q}.gguf", None)
        cmd = [b, "--backend", "voxcpm2", "-m", g, "--tts", TEXT, "--tts-output", C.SCRATCH / f"{q}.wav",
               "--seed", "1", "-t", nt]
        r = C.run(cmd, C.OUT / f"bench-{q}.log", check=False, env=env, timeout=1800)
        res["arms"][q] = {"rc": r.returncode}
        if perf:
            data = C.SCRATCH / f"{q}.perf"
            C.run(["perf", "record", "-g", "-o", data, "--"] + cmd, C.OUT / f"perf-rec-{q}.log", check=False,
                  timeout=2400)
            rep = C.run(["perf", "report", "-i", data, "--no-children", "--stdio", "--percent-limit", "1",
                         "-g", "none", "--sort", "symbol"], C.OUT / f"perf-{q}.txt", check=False, timeout=600)
            res["arms"][q]["perf_rc"] = rep.returncode
        Path(g).unlink(missing_ok=True)
        C.save(res)
    lines = ["## VoxCPM2 CPU profile", ""]
    for q in res["arms"]:
        lines += [f"### {q} bench", "```"]
        log = (C.OUT / f"bench-{q}.log").read_text(errors="replace").splitlines()
        lines += [x for x in log if "ms" in x and "voxcpm2" in x][-60:]
        lines += ["```"]
        p = C.OUT / f"perf-{q}.txt"
        if p.exists():
            lines += [f"### {q} perf", "```"]
            lines += [x for x in p.read_text(errors="replace").splitlines() if "%" in x and not x.startswith("#")][:40]
            lines += ["```"]
    C.summary(lines)


if __name__ == "__main__":
    main()
