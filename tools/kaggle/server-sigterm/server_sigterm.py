#!/usr/bin/env python3
"""crispasr-server SIGTERM: control (main's ws_stream / realtime_server) vs fix.

Build crispasr-server with main's two listener files, start it with WS streaming,
SIGTERM it and record whether it exits within 15 s. Then swap in the fix branch's
files, rebuild (incremental) and repeat.
"""
import json, os, shutil, signal, subprocess, sys, time, traceback
from pathlib import Path
WORK = Path("/kaggle/working"); OUT = WORK / "out"; OUT.mkdir(parents=True, exist_ok=True)
REPO = WORK / "CrispASR"; BUILD = REPO / "build"
res = {"errors": []}
def save(): (OUT / "sigterm.json").write_text(json.dumps(res, indent=1))
FILES = ["examples/server/ws_stream.cpp", "examples/server/realtime_server.cpp"]
def trial(tag, port):
    log = open(OUT / f"{tag}.log", "w")
    p = subprocess.Popen([str(BUILD / "bin/crispasr-server"), "--port", str(port), "--ws-port", str(port + 1),
                          "-m", "/tmp/ggml-tiny.bin", "-t", "2"], stdout=log, stderr=subprocess.STDOUT)
    t0 = time.time()
    while time.time() - t0 < 180:
        if "ws://" in (OUT / f"{tag}.log").read_text() or p.poll() is not None:
            break
        time.sleep(0.5)
    ready = "ws://" in (OUT / f"{tag}.log").read_text()
    time.sleep(2)
    p.send_signal(signal.SIGTERM); t1 = time.time()
    try:
        rc = p.wait(timeout=15); dt = time.time() - t1
    except subprocess.TimeoutExpired:
        rc, dt = "ALIVE after 15 s", None
        p.kill(); p.wait()
    res[tag] = {"ready": ready, "rc": rc, "exit_s": dt}; save()
try:
    subprocess.check_call(["git", "clone", "--depth", "2", "--recurse-submodules", "--shallow-submodules", "-b", "fix/server-sigterm",
                           "https://github.com/CrispStrobe/CrispASR.git", str(REPO)])
    sys.path.insert(0, str(REPO / "tools" / "kaggle"))
    import kaggle_harness as kh
    kh.install_build_toolchain()
    flags = kh.cache_and_link_flags()
    subprocess.check_call(["wget", "-q", "-O", "/tmp/ggml-tiny.bin", "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.bin"])
    fixed = {f: (REPO / f).read_text() for f in FILES}
    for f in FILES:  # control: the parent commit (main) versions
        (REPO / f).write_text(subprocess.check_output(["git", "-C", str(REPO), "show", f"HEAD~1:{f}"], text=True))
    kh.sh(f"cmake -S {REPO} -B {BUILD} -G Ninja -DCMAKE_BUILD_TYPE=Release -DCRISPASR_OPUS=OFF -DCRISPASR_AMR=OFF " + " ".join(flags))
    kh.sh(f"cmake --build {BUILD} -j$(nproc) --target crispasr-server")
    trial("control_main", 18480)
    for f in FILES:
        (REPO / f).write_text(fixed[f])
    kh.sh(f"cmake --build {BUILD} -j$(nproc) --target crispasr-server")
    trial("fix", 18490)
except BaseException:
    res["errors"].append(traceback.format_exc())
finally:
    save()
    shutil.rmtree(REPO, ignore_errors=True)
