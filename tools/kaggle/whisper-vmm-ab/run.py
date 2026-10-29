#!/usr/bin/env python3
"""Fail-closed, same-source Whisper CUDA VMM diagnostic. Unsubmitted adapter."""

import ctypes
import hashlib
import json
import os
from pathlib import Path
import re
import resource
import shutil
import statistics
import subprocess
import threading
import time
import urllib.request


SOURCE = "1c0114f74b08f7ff6f3fb7f71fc744ded7e2c3f6"
BRANCH = "validate/whisper-vmm-timing-20261029"
WORK = Path("/kaggle/temp/whisper-vmm-ab")
OUT = Path("/kaggle/working")
REFERENCE = "and so my fellow americans ask not what your country can do for you ask what you can do for your country"
PINS = {
    "model.bin": (
        "https://huggingface.co/ggerganov/whisper.cpp/resolve/5359861c739e955e79d9a303bcbc70fb988958b1/ggml-base.en.bin",
        "a03779c86df3323075f5e796cb2ce5029f00ec8869eee3fdfb897afe36c6d002",
    ),
    "input.wav": (
        "https://raw.githubusercontent.com/CrispStrobe/CrispASR/cc114d2b4a5a9413738f390093e9131576743cb6/samples/jfk.wav",
        "59dfb9a4acb36fe2a2affc14bacbee2920ff435cb13cc314a08c13f66ba7860e",
    ),
}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_checked(command, *, cwd=None, timeout=300, log=None):
    if log is None:
        return subprocess.run(command, cwd=cwd, check=True, timeout=timeout, capture_output=True, text=True).stdout
    with log.open("wb") as stream:
        result = subprocess.run(command, cwd=cwd, stdout=stream, stderr=subprocess.STDOUT, timeout=timeout)
    if result.returncode:
        raise RuntimeError(f"Command failed ({result.returncode}); inspect {log.name}")
    return log.read_text(errors="replace")


def guard():
    mem = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
    state = {
        "available": int(mem["MemAvailable"].split()[0]) * 1024,
        "load": list(os.getloadavg()),
        "cpus": os.cpu_count() or 1,
        "disk": shutil.disk_usage(WORK).free,
    }
    if state["available"] < 3 * 1024**3 or state["load"][0] > 2 * state["cpus"] or state["disk"] < 6 * 1024**3:
        raise RuntimeError("Insufficient memory, CPU or disk headroom")
    return state


def driver_path():
    """Use the active host driver, never a toolkit stub or a copied library."""
    cuda = ctypes.CDLL("libcuda.so.1")
    if cuda.cuInit(0) != 0:
        raise RuntimeError("Host CUDA driver cuInit failed")
    count = ctypes.c_int()
    if cuda.cuDeviceGetCount(ctypes.byref(count)) != 0 or count.value < 1:
        raise RuntimeError("Host CUDA driver has no usable device")
    paths = {
        Path(line.rsplit(" ", 1)[-1]).resolve()
        for line in Path("/proc/self/maps").read_text().splitlines()
        if "libcuda.so" in line and line.rsplit(" ", 1)[-1].startswith("/")
    }
    if len(paths) != 1:
        raise RuntimeError("Could not identify exactly one mapped host libcuda")
    path = paths.pop()
    if "stubs" in path.parts or not path.is_file():
        raise RuntimeError("CUDA runtime resolved to a toolkit stub or missing file")
    dynamic = run_checked(["readelf", "-d", str(path)])
    if not re.search(r"\(SONAME\).*\[libcuda\.so\.1\]", dynamic):
        raise RuntimeError("Host driver lacks the libcuda.so.1 SONAME")
    return path, count.value


def build_arm(source, arm, driver):
    build = WORK / f"build-{arm}"
    vmm_on = arm == "on"
    command = [
        "cmake", "-S", str(source), "-B", str(build), "-DCMAKE_BUILD_TYPE=Release",
        "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON", "-DGGML_CUDA=ON", "-DGGML_NATIVE=OFF",
        "-DCMAKE_CUDA_ARCHITECTURES=75", "-DCRISPASR_BUILD_TESTS=OFF",
        "-DCRISPASR_BUILD_SERVER=OFF", f"-DGGML_CUDA_NO_VMM={'OFF' if vmm_on else 'ON'}",
    ]
    if vmm_on:
        command.append(f"-DGGML_CUDA_DRIVER_LIB={driver}")
    configure = run_checked(command, timeout=300, log=OUT / f"cmake-{arm}.log")
    if vmm_on and ("ggml-cuda: linking CUDA driver" not in configure or "building with GGML_CUDA_NO_VMM" in configure):
        raise RuntimeError("VMM-on configuration silently disabled VMM")
    if not vmm_on and "linking CUDA driver" in configure:
        raise RuntimeError("VMM-off configuration unexpectedly linked the driver")
    guard()
    build_log = run_checked(["cmake", "--build", str(build), "--target", "crispasr-cli", "-j", "2"],
                            timeout=5400, log=OUT / f"build-{arm}.log")
    if "Built target crispasr-cli" not in build_log and "Linking CXX executable" not in build_log:
        raise RuntimeError("Build produced no target/link proof")
    commands = json.loads((build / "compile_commands.json").read_text())
    cuda_commands = [row["command"] for row in commands if row["file"].endswith("ggml-cuda.cu")]
    if len(cuda_commands) != 1:
        raise RuntimeError("Missing unique ggml-cuda compile command")
    no_vmm = "-DGGML_CUDA_NO_VMM" in cuda_commands[0]
    if no_vmm == vmm_on:
        raise RuntimeError("Effective CUDA compile definition disagrees with requested VMM arm")
    candidates = [p for p in build.rglob("crispasr") if p.is_file()]
    if len(candidates) != 1:
        raise RuntimeError("Missing unique rebuilt CrispASR executable")
    cuda_libs = [p for p in build.rglob("libggml-cuda.so*") if p.is_file() and not p.is_symlink()]
    if len(cuda_libs) != 1:
        raise RuntimeError("Missing unique ggml-cuda library")
    library = cuda_libs[0]
    binary_links = run_checked(["ldd", str(candidates[0])])
    cuda_match = re.search(r"libggml-cuda\.so(?:\.\S+)?\s*=>\s*(/\S+)", binary_links)
    if not cuda_match or Path(cuda_match.group(1)).resolve() != library.resolve():
        raise RuntimeError("Executable resolves the wrong ggml-cuda library")
    dynamic = run_checked(["readelf", "-d", str(library)])
    needed = "Shared library: [libcuda.so.1]" in dynamic
    if needed != vmm_on:
        raise RuntimeError("CUDA library driver dependency disagrees with VMM arm")
    if vmm_on:
        linked = run_checked(["ldd", str(library)])
        match = re.search(r"libcuda\.so\.1\s*=>\s*(/\S+)", linked)
        if not match or Path(match.group(1)).resolve() != driver:
            raise RuntimeError("VMM-on library does not resolve to the validated host driver")
    return candidates[0], {
        "binary_sha256": sha(candidates[0]), "cuda_library_sha256": sha(library),
        "driver_needed": needed, "effective_no_vmm": no_vmm,
        "cuda_compile_command_sha256": hashlib.sha256(cuda_commands[0].encode()).hexdigest(),
    }


def wer(hypothesis):
    def words(text):
        return re.findall(r"[a-z0-9]+", text.lower())

    expected, actual = words(REFERENCE), words(hypothesis)
    row = list(range(len(actual) + 1))
    for i, word in enumerate(expected, 1):
        nxt = [i]
        for j, other in enumerate(actual, 1):
            nxt.append(min(nxt[-1] + 1, row[j] + 1, row[j - 1] + (word != other)))
        row = nxt
    return row[-1] / len(expected)


def measure(binary, arm, profile, phase, round_number):
    state = guard()
    name = f"{phase}-{round_number}-{arm}-{'profile' if profile else 'plain'}"
    prefix, log = WORK / name, OUT / f"{name}.log"
    command = [str(binary), "-m", str(WORK / "model.bin"), "-f", str(WORK / "input.wav"),
               "-l", "en", "-t", "2", "-bs", "1", "-otxt", "-of", str(prefix)]
    env = dict(os.environ, LD_LIBRARY_PATH=str(binary.parent) + os.pathsep + os.environ.get("LD_LIBRARY_PATH", ""))
    env.pop("CRISPASR_WHISPER_TIMING_PROFILE", None)
    if profile:
        env["CRISPASR_WHISPER_TIMING_PROFILE"] = "1"
    with log.open("wb") as stream:
        start = time.monotonic()
        proc = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, env=env)
        observed, stop = [False], threading.Event()

        def sample_gpu():
            while not stop.is_set():
                probe = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
                                       capture_output=True, text=True, timeout=10)
                observed[0] |= str(proc.pid) in probe.stdout.split()
                stop.wait(0.05)

        monitor = threading.Thread(target=sample_gpu, daemon=True)
        monitor.start()
        try:
            proc.wait(timeout=180)
            elapsed = time.monotonic() - start
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            raise TimeoutError(name)
        finally:
            stop.set()
            monitor.join(timeout=15)
    contents = log.read_text(errors="replace")
    transcript = prefix.with_suffix(".txt")
    if proc.returncode or not observed[0] or not transcript.is_file():
        raise RuntimeError(f"GPU, exit status or transcript missing: {name}")
    if "CUDA0 total size" not in contents or "using CUDA0 backend" not in contents:
        raise RuntimeError(f"CUDA weight placement or execution missing: {name}")
    device = re.search(r"Device 0: Tesla T4,.*VMM: (yes|no)", contents)
    system = next((line for line in contents.splitlines() if line.startswith("system_info:")), "")
    expected_vmm = "yes" if arm == "on" else "no"
    if not device or device.group(1) != expected_vmm or ("NO_VMM = 1" in system) != (arm == "off"):
        raise RuntimeError(f"Runtime VMM state disagrees with built arm: {name}")
    quality = wer(transcript.read_text())
    decode = re.search(r"decode time\s*=\s*([0-9.]+) ms\s*/\s*([0-9]+) runs", contents)
    load = re.search(r"load time\s*=\s*([0-9.]+) ms", contents)
    if quality != 0 or not decode or not load or int(decode.group(2)) != 27:
        raise RuntimeError(f"Quality or expected 27-call decode gate failed: {name}")
    profile_lines = [line for line in contents.splitlines() if line.startswith("whisper timing profile:")]
    phase_us = {}
    if profile:
        if len(profile_lines) != 3 or not any("decode count=27" in line for line in profile_lines):
            raise RuntimeError(f"Incomplete opt-in phase profile: {name}")
        for line in profile_lines:
            scope = "decode" if "profile: decode " in line else "load"
            phase_us.update({f"{scope}_{key}": int(value)
                             for key, value in re.findall(r"([a-z_]+)=([0-9]+)", line)})
        required = {"load_total_us", "load_alloc_us", "load_read_us", "load_upload_us",
                    "load_prealloc_us", "load_vocab_us", "load_buft_list_us", "load_select_us",
                    "decode_count", "decode_graph_us", "decode_input_us", "decode_compute_us", "decode_read_us"}
        if not required <= phase_us.keys() or phase_us["decode_count"] != 27:
            raise RuntimeError(f"Incomplete phase counters: {name}")
        if abs(phase_us["load_total_us"] / 1000 - float(load.group(1))) > 0.05:
            raise RuntimeError(f"Profile and reported load timers disagree: {name}")
    elif profile_lines:
        raise RuntimeError(f"Profiling leaked into plain control: {name}")
    return {
        "phase": phase, "round": round_number, "arm": arm, "profile": profile,
        "seconds": elapsed, "load_ms": float(load.group(1)), "decode_ms": float(decode.group(1)),
        "decode_calls": int(decode.group(2)), "wer": quality, "cuda_process_observed": observed[0],
        "transcript_sha256": sha(transcript), "resources": state, "phase_us": phase_us,
    }


def main():
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    WORK.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    receipt = {
        "complete": False, "acceptance": False,
        "scope": "Same-source T4 VMM on/off diagnostic; load and process timing include runtime setup. One public JFK clip; no general speed or quality claim.",
        "source": SOURCE, "branch": BRANCH,
        "pins": {name: digest for name, (_, digest) in PINS.items()}, "runs": [],
    }

    def save():
        (OUT / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")

    save()
    guard()
    hardware = run_checked(["nvidia-smi", "--query-gpu=name,compute_cap", "--format=csv,noheader"])
    if not hardware.splitlines() or not hardware.splitlines()[0].startswith("Tesla T4, 7.5"):
        raise RuntimeError("This diagnostic requires the pinned T4 hardware")
    receipt["hardware"] = hardware.strip()
    driver, device_count = driver_path()
    receipt["driver"] = {"soname": "libcuda.so.1", "device_count": device_count,
                         "sha256": sha(driver), "mapped_host_driver": True}
    save()
    for name, (url, digest) in PINS.items():
        destination = WORK / name
        with urllib.request.urlopen(url, timeout=180) as source, destination.open("wb") as target:
            shutil.copyfileobj(source, target, 1024 * 1024)
        if sha(destination) != digest:
            raise RuntimeError(f"Pinned input hash mismatch: {name}")
    guard()
    source = WORK / "crisp"
    run_checked(["git", "clone", "--branch", BRANCH, "--single-branch",
                 "https://github.com/CrispStrobe/CrispASR.git", str(source)], timeout=300,
                log=OUT / "clone.log")
    if run_checked(["git", "rev-parse", "HEAD"], cwd=source).strip() != SOURCE:
        raise RuntimeError("Candidate source pin mismatch")
    run_checked(["git", "submodule", "update", "--init", "--recursive"], cwd=source,
                timeout=300, log=OUT / "submodule.log")
    receipt["ggml_source"] = run_checked(["git", "-C", str(source / "ggml"), "rev-parse", "HEAD"]).strip()
    save()
    binaries = {}
    for arm in ("off", "on"):
        guard()
        binaries[arm], proof = build_arm(source, arm, driver)
        receipt.setdefault("builds", {})[arm] = proof
        save()
    for phase, rounds in (("warmup", 1), ("measured", 4)):
        for index in range(rounds):
            order = (("off", False), ("on", False), ("on", True), ("off", True))
            if index % 2:
                order = tuple(reversed(order))
            for arm, profile in order:
                row = measure(binaries[arm], arm, profile, phase, index)
                receipt["runs"].append(row)
                save()
    if len(receipt["runs"]) != 20 or len({row["transcript_sha256"] for row in receipt["runs"]}) != 1:
        raise RuntimeError("Missing runs or transcript mismatch between VMM and profiler controls")
    receipt["medians"] = {
        f"{arm}-{'profile' if profile else 'plain'}": {
            "process_seconds": statistics.median(row["seconds"] for row in receipt["runs"]
                                                 if row["phase"] == "measured" and row["arm"] == arm
                                                 and row["profile"] == profile),
            "load_ms": statistics.median(row["load_ms"] for row in receipt["runs"]
                                         if row["phase"] == "measured" and row["arm"] == arm
                                         and row["profile"] == profile),
            "decode_ms_per_call": statistics.median(row["decode_ms"] / row["decode_calls"]
                                                    for row in receipt["runs"] if row["phase"] == "measured"
                                                    and row["arm"] == arm and row["profile"] == profile),
        }
        for arm in ("off", "on") for profile in (False, True)
    }
    receipt["phase_medians_us"] = {
        arm: {key: statistics.median(row["phase_us"][key] for row in receipt["runs"]
                                     if row["phase"] == "measured" and row["arm"] == arm and row["profile"])
              for key in next(row["phase_us"] for row in receipt["runs"] if row["phase"] == "measured"
                              and row["arm"] == arm and row["profile"])}
        for arm in ("off", "on")
    }
    receipt["complete"] = True
    save()


if __name__ == "__main__":
    main()
