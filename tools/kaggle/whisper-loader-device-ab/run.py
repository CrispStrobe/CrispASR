#!/usr/bin/env python3
"""Hosted-only, same-toolchain Whisper loader classification-cache A/B."""

import hashlib
import json
import os
from pathlib import Path
import re
import resource
import shlex
import shutil
import statistics
import subprocess
import sys
import threading
import time
import traceback
import urllib.request


VERSION = "whisper-loader-device-ab-v1"
# Fill with immutable, published commits before admission. Both must contain
# the SAME placement tracer; candidate differs only by the loader-local cache.
SOURCES = {"base": "9df3071a933b2c85ceb3e278596c219c5d00c32e",
           "candidate": "0a3a97c1e1c43ba3a1a56f1923ef6a74fb362f89"}
URL = "https://github.com/CrispStrobe/CrispASR.git"
WORK = Path("/kaggle/temp/whisper-loader-device-ab")
OUT = Path("/kaggle/working")
TIMEOUT = 5400
PROFILE_ENV = "CRISPASR_WHISPER_TIMING_PROFILE"
TRACE_ENV = "CRISPASR_WHISPER_PLACEMENT_TRACE"
TRACE_PREFIX = "whisper placement trace: "
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


def checked(command, *, cwd=None, log=None):
    """Persist combined command output, including partial output on timeout."""
    if log is None:
        return subprocess.run(command, cwd=cwd, check=True, timeout=TIMEOUT,
                              capture_output=True, text=True).stdout
    with log.open("wb") as stream:
        result = subprocess.run(command, cwd=cwd, timeout=TIMEOUT,
                                stdout=stream, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f"Command failed ({result.returncode}); inspect {log.name}")
    return log.read_text(errors="replace")


def guard():
    mem = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
    state = {"available_bytes": int(mem["MemAvailable"].split()[0]) * 1024,
             "load": list(os.getloadavg()), "cpus": os.cpu_count() or 1,
             "scratch_free_bytes": shutil.disk_usage(WORK).free,
             "output_free_bytes": shutil.disk_usage(OUT).free}
    if (state["available_bytes"] < 3 * 1024**3 or state["load"][0] > 2 * state["cpus"]
            or state["scratch_free_bytes"] < 8 * 1024**3 or state["output_free_bytes"] < 1024**3):
        raise RuntimeError(f"Insufficient resource headroom: {state}")
    apps = checked(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"])
    if any(line.strip().isdigit() for line in apps.splitlines()):
        raise RuntimeError("GPU has an unrelated compute process before this run")
    return state


def checkout(arm, commit):
    source = WORK / arm
    checked(["git", "init", str(source)], log=OUT / f"git-init-{arm}.log")
    checked(["git", "remote", "add", "origin", URL], cwd=source)
    checked(["git", "fetch", "--depth=1", "origin", commit], cwd=source,
            log=OUT / f"git-fetch-{arm}.log")
    checked(["git", "checkout", "--detach", "FETCH_HEAD"], cwd=source,
            log=OUT / f"git-checkout-{arm}.log")
    if checked(["git", "rev-parse", "HEAD"], cwd=source).strip() != commit:
        raise RuntimeError(f"Source commit mismatch: {arm}")
    checked(["git", "submodule", "update", "--init", "--recursive"], cwd=source,
            log=OUT / f"submodules-{arm}.log")
    shutil.copyfile(source / "src/crispasr.cpp", OUT / f"crispasr-{arm}.cpp")
    return source, {"commit": commit,
                    "tree": checked(["git", "rev-parse", "HEAD^{tree}"], cwd=source).strip(),
                    "loader_sha256": sha(source / "src/crispasr.cpp"),
                    "submodules": checked(["git", "submodule", "status", "--recursive"], cwd=source)}


def build_arm(source, arm, flags, kh):
    build = WORK / f"build-{arm}"
    guard()
    command = ["cmake", "-S", str(source), "-B", str(build), "-G", "Ninja",
               "-DCMAKE_BUILD_TYPE=Release", "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON",
               "-DGGML_NATIVE=OFF", "-DCRISPASR_BUILD_TESTS=OFF", "-DCRISPASR_BUILD_SERVER=OFF"] + flags
    checked(command, log=OUT / f"cmake-{arm}.log")
    guard()
    with kh.build_heartbeat(f"build-{arm}"):
        output = checked(["cmake", "--build", str(build), "--target", "crispasr-cli",
                          "-j", kh.safe_build_jobs(gpu=True)], log=OUT / f"build-{arm}.log")
    if not re.search(r"Linking CXX executable.*crispasr", output):
        raise RuntimeError(f"No fresh executable link proof: {arm}")
    commands = json.loads((build / "compile_commands.json").read_text())
    cuda = [row["command"] for row in commands if row["file"].endswith("ggml-cuda.cu")]
    if len(cuda) != 1 or "-DGGML_CUDA_NO_VMM" not in cuda[0] or "compute_75" not in cuda[0]:
        raise RuntimeError(f"Effective CUDA flags do not prove SM75 / VMM off: {arm}")
    cache = (build / "CMakeCache.txt").read_text()
    if not re.search(r"^GGML_CUDA_NO_VMM:BOOL=ON$", cache, re.M):
        raise RuntimeError(f"VMM configuration missing: {arm}")
    shutil.copyfile(build / "compile_commands.json", OUT / f"compile-commands-{arm}.json")
    shutil.copyfile(build / "CMakeCache.txt", OUT / f"cmake-cache-{arm}.txt")
    binaries = [p for p in build.rglob("crispasr") if p.is_file()]
    libraries = [p for p in build.rglob("libggml-cuda.so*") if p.is_file() and not p.is_symlink()]
    if len(binaries) != 1 or len(libraries) != 1:
        raise RuntimeError(f"Missing unique binary / CUDA library: {arm}")
    binary, library = binaries[0], libraries[0]
    links = checked(["ldd", str(binary)], log=OUT / f"ldd-{arm}.log")
    resolved = re.search(r"libggml-cuda\.so(?:\.\S+)?\s*=>\s*(/\S+)", links)
    if not resolved or Path(resolved.group(1)).resolve() != library.resolve() or "not found" in links:
        raise RuntimeError(f"Binary resolves incorrect libraries: {arm}")
    dynamic = checked(["readelf", "-d", str(library)], log=OUT / f"cuda-dynamic-{arm}.log")
    if "Shared library: [libcuda.so.1]" in dynamic:
        raise RuntimeError(f"VMM-off CUDA library links driver API: {arm}")
    normalized = sorted(normalize_command(row["command"], source, build) for row in commands)
    return binary, {"binary_sha256": sha(binary), "cuda_library_sha256": sha(library),
                    "compile_commands_sha256": sha(build / "compile_commands.json"),
                    "normalized_compile_commands_sha256": hashlib.sha256("\n".join(normalized).encode()).hexdigest(),
                    "effective_sm": 75, "effective_no_vmm": True}


def normalize_command(command, source, build):
    # examples/cli/CMakeLists.txt embeds revision/date/subject plus configure
    # timestamp into the CLI banner. Preserve every other compiler argument.
    metadata = {"CRISPASR_GIT_SHA1", "CRISPASR_GIT_DATE", "CRISPASR_GIT_SUBJECT", "CRISPASR_BUILD_DATE"}
    arguments = []
    for argument in shlex.split(command):
        if argument.startswith("-D") and argument[2:].split("=", 1)[0] in metadata:
            argument = argument.split("=", 1)[0] + "=SOURCE_METADATA"
        arguments.append(argument.replace(str(source), "SOURCE").replace(str(build), "BUILD"))
    return json.dumps(arguments)


def words(text):
    return re.findall(r"[a-z0-9]+", text.lower())


def parse_trace(contents):
    lines = [line for line in contents.splitlines() if line.startswith(TRACE_PREFIX)]
    pattern = re.compile(r"^whisper placement trace: tensor=(\S+) op=\S+ type=\S+ "
                         r"shape=[0-9]+,[0-9]+,[0-9]+,[0-9]+ device=.+ buffer=.+$")
    names = []
    for line in lines:
        match = pattern.fullmatch(line)
        if not match:
            raise RuntimeError("Malformed tensor placement trace")
        names.append(match.group(1))
    if not lines or len(names) != len(set(names)):
        raise RuntimeError("Missing trace or duplicate tensor placement names")
    # Each model load must finish successfully; the loader itself rejects a
    # loaded-tensor count that differs from its complete created-tensor map.
    return lines


def parse_profile(contents, profiled):
    lines = [line for line in contents.splitlines() if line.startswith("whisper timing profile:")]
    if not profiled:
        if lines:
            raise RuntimeError("Profiler leaked into plain control")
        return {}
    phases = {}
    for line in lines:
        scope = "decode" if "profile: decode " in line else "load"
        for key, value in re.findall(r"([a-z_]+)=([0-9]+)", line):
            name = f"{scope}_{key}"
            if name in phases:
                raise RuntimeError(f"Duplicate profile counter: {name}")
            phases[name] = int(value)
    required = {"load_total_us", "load_alloc_us", "load_read_us", "load_upload_us", "load_prealloc_us",
                "load_vocab_us", "load_buft_list_us", "load_select_us", "decode_count", "decode_graph_us",
                "decode_input_us", "decode_compute_us", "decode_read_us"}
    if not required <= phases.keys() or phases["decode_count"] != 27:
        raise RuntimeError("Missing or incorrect timing profile counters")
    return phases


def measure(binary, arm, profile, phase, index, *, trace=False, cpu=False):
    state = guard()
    name = f"{phase}-{index}-{arm}-{'profile' if profile else 'plain'}-{'cpu' if cpu else 'gpu'}"
    log, transcript = OUT / f"{name}.log", OUT / f"{name}.txt"
    command = [str(binary), "-m", str(WORK / "model.bin"), "-f", str(WORK / "input.wav"),
               "-l", "en", "-t", "2", "-bs", "1", "-otxt", "-of", str(transcript.with_suffix(""))]
    command += ["--no-gpu"] if cpu else ["--gpu-backend", "cuda"]
    env = dict(os.environ)
    env.pop(PROFILE_ENV, None)
    env.pop(TRACE_ENV, None)
    if profile:
        env[PROFILE_ENV] = "1"
    if trace:
        env[TRACE_ENV] = "1"
    # Enforce fixed CPU/GPU selection even when the worker offers two T4s.
    env.update(CUDA_VISIBLE_DEVICES="0", OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", MKL_NUM_THREADS="2")
    observed, monitor_errors, stop = [], [], threading.Event()
    with log.open("wb") as stream:
        start = time.monotonic()
        proc = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, env=env, start_new_session=True)

        def sample_gpu():
            while not stop.is_set():
                try:
                    pids = checked(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"])
                    if str(proc.pid) in pids.split():
                        observed.append(proc.pid)
                except Exception as error:
                    monitor_errors.append(str(error))
                    return
                stop.wait(0.05)

        monitor = threading.Thread(target=sample_gpu, daemon=True)
        monitor.start()
        try:
            proc.wait(timeout=TIMEOUT)
            seconds = time.monotonic() - start
        finally:
            if proc.poll() is None:
                os.killpg(proc.pid, 9)
                proc.wait()
            stop.set()
            monitor.join(timeout=TIMEOUT)
    contents = log.read_text(errors="replace")
    if proc.returncode or monitor.is_alive() or monitor_errors or not transcript.is_file():
        raise RuntimeError(f"Process / monitor / output failure: {name}")
    if words(transcript.read_text()) != words(REFERENCE):
        raise RuntimeError(f"Expected zero-WER transcript failed: {name}")
    if not cpu and (not observed or "CUDA0 total size" not in contents or "using CUDA0 backend" not in contents):
        raise RuntimeError(f"GPU PID, weight placement or execution missing: {name}")
    if cpu and (observed or "using CUDA0 backend" in contents or "CUDA0 total size" in contents):
        raise RuntimeError(f"CPU placement control used CUDA: {name}")
    if not cpu and not re.search(r"Device 0: Tesla T4,.*VMM: no", contents):
        raise RuntimeError(f"Runtime did not confirm T4 / VMM off: {name}")
    decode = re.search(r"decode time\s*=\s*([0-9.]+) ms\s*/\s*([0-9]+) runs", contents)
    load = re.search(r"load time\s*=\s*([0-9.]+) ms", contents)
    if not decode or int(decode.group(2)) != 27 or not load:
        raise RuntimeError(f"Missing load timing or expected 27 decoder calls: {name}")
    phases = parse_profile(contents, profile)
    if profile and abs(phases["load_total_us"] / 1000 - float(load.group(1))) > 0.05:
        raise RuntimeError(f"Profile / reported load timer mismatch: {name}")
    traces = parse_trace(contents) if trace else []
    if not trace and TRACE_PREFIX in contents:
        raise RuntimeError(f"Placement tracer leaked into timed call: {name}")
    return {"phase": phase, "round": index, "arm": arm, "profile": profile, "cpu": cpu,
            "process_seconds": seconds, "load_ms": float(load.group(1)), "decode_ms": float(decode.group(1)),
            "decode_calls": int(decode.group(2)), "wer": 0, "gpu_pid_observed": bool(observed),
            "transcript_sha256": sha(transcript), "resources": state, "phase_us": phases,
            "trace": traces, "command": command, "log": log.name, "transcript": transcript.name}


def main(receipt, save):
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    guard()
    if any(not re.fullmatch(r"[0-9a-f]{40}", commit) for commit in SOURCES.values()):
        raise RuntimeError("Fill both immutable source commit pins before submission")
    if SOURCES["base"] == SOURCES["candidate"]:
        raise RuntimeError("A/B source pins must differ")
    hardware = checked(["nvidia-smi", "--query-gpu=index,name,uuid,compute_cap,driver_version", "--format=csv,noheader"])
    receipt["hardware"] = hardware.strip()
    first = hardware.splitlines()[0].split(",") if hardware.splitlines() else []
    if len(first) != 5 or first[1].strip() != "Tesla T4" or first[3].strip() != "7.5":
        raise RuntimeError("Diagnostic requires Tesla T4 SM75 at device 0; no automatic redraw")
    save()
    sources = {}
    for arm, commit in SOURCES.items():
        guard()
        sources[arm], receipt.setdefault("sources", {})[arm] = checkout(arm, commit)
        save()
    if receipt["sources"]["base"]["submodules"] != receipt["sources"]["candidate"]["submodules"]:
        raise RuntimeError("A/B submodule revisions differ")
    checked(["git", "fetch", str(sources["candidate"]), SOURCES["candidate"]], cwd=sources["base"],
            log=OUT / "git-fetch-candidate-for-diff.log")
    changes = checked(["git", "diff", "--name-only", *SOURCES.values()], cwd=sources["base"]).splitlines()
    if changes != ["src/crispasr.cpp"]:
        raise RuntimeError(f"Unexpected candidate source changes: {changes}")
    checked(["git", "diff", *SOURCES.values()], cwd=sources["base"], log=OUT / "base-candidate.patch")
    sys.path.insert(0, str(sources["base"] / "tools/kaggle"))
    import kaggle_harness as kh
    kh.init_progress(hf_progress_repo="")
    kh.install_build_toolchain()
    arch = kh.detect_cuda_arch()
    if arch != "75":
        raise RuntimeError(f"Harness CUDA architecture detection disagrees with SM75: {arch}")
    flags = kh.cuda_build_flags(arch) + kh.cache_and_link_flags()
    # Pin the same resolved compiler executables for BOTH source builds.
    flags += [f"-DCMAKE_C_COMPILER={shutil.which('gcc')}", f"-DCMAKE_CXX_COMPILER={shutil.which('g++')}"]
    receipt["build_flags"] = flags
    receipt["compile_comparison_excludes"] = ["absolute source/build workspace prefixes",
        "CRISPASR_GIT_SHA1", "CRISPASR_GIT_DATE", "CRISPASR_GIT_SUBJECT", "CRISPASR_BUILD_DATE"]
    receipt["toolchain"] = {tool: checked([tool, "--version"]) for tool in ("gcc", "g++", "/usr/local/cuda/bin/nvcc", "cmake", "ninja")}
    save()
    for name, (url, digest) in PINS.items():
        with urllib.request.urlopen(url, timeout=TIMEOUT) as source, (WORK / name).open("wb") as target:
            shutil.copyfileobj(source, target, 1024 * 1024)
        if sha(WORK / name) != digest:
            raise RuntimeError(f"Pinned input hash mismatch: {name}")
    binaries = {}
    for arm in SOURCES:
        binaries[arm], receipt.setdefault("builds", {})[arm] = build_arm(sources[arm], arm, flags, kh)
        save()
    if len({proof["normalized_compile_commands_sha256"] for proof in receipt["builds"].values()}) != 1:
        raise RuntimeError("Normalized compile commands differ between arms")
    # Separate trace controls; their wall times are deliberately excluded.
    for cpu in (False, True):
        pair = []
        for arm in SOURCES:
            row = measure(binaries[arm], arm, False, "placement", 0, trace=True, cpu=cpu)
            receipt["runs"].append(row)
            pair.append(row)
            save()
        if pair[0]["trace"] != pair[1]["trace"] or pair[0]["transcript_sha256"] != pair[1]["transcript_sha256"]:
            raise RuntimeError(f"Ordered per-tensor placement / transcript mismatch (cpu={cpu})")
    receipt["placement_equivalent_for_tested_model_and_configs"] = True
    save()
    for phase, rounds in (("warmup", 1), ("measured", 4)):
        for index in range(rounds):
            order = (("base", False), ("candidate", False), ("candidate", True), ("base", True))
            if index % 2:
                order = tuple(reversed(order))
            for arm, profile in order:
                receipt["runs"].append(measure(binaries[arm], arm, profile, phase, index))
                save()
    timed = [row for row in receipt["runs"] if row["phase"] != "placement"]
    if len(timed) != 20 or len({row["transcript_sha256"] for row in timed}) != 1:
        raise RuntimeError("Incomplete timed controls or transcript bytes differ")
    medians = {}
    for arm in SOURCES:
        for profile in (False, True):
            rows = [row for row in timed if row["phase"] == "measured" and row["arm"] == arm and row["profile"] == profile]
            medians[f"{arm}-{'profile' if profile else 'plain'}"] = {
                "samples": len(rows), "process_seconds": statistics.median(row["process_seconds"] for row in rows),
                "load_ms": statistics.median(row["load_ms"] for row in rows),
                "decode_ms_per_call": statistics.median(row["decode_ms"] / row["decode_calls"] for row in rows),
                "phase_medians_us": {key: statistics.median(row["phase_us"][key] for row in rows)
                                     for key in rows[0]["phase_us"]},
            }
    receipt.update(complete=True, acceptance=True, medians=medians)
    save()


if __name__ == "__main__":
    WORK.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(__file__, OUT / "uploaded-source.py")
    receipt = {"version": VERSION, "script_sha256": sha(__file__), "complete": False, "acceptance": False,
               "runs": [], "pins": {name: value[1] for name, value in PINS.items()},
               "scope": "One base.en/JFK T4 loader-cache A/B; same compiler/CUDA/SM75/VMM-off. Acceptance is bounded to this workload and tested CPU/CUDA placement controls.",
               "boundaries": {"process": "fresh process startup through exit; GPU PID sampling runs concurrently; model loading included; warm OS page cache",
                              "load_ms": "existing whisper model-load timer; excludes backend initialization before loader entry",
                              "phase_medians_us": "existing opt-in host timers; allocation, read/upload and selection are separate; prealloc contains nested vocab/list/select work; do not sum overlapping phases",
                              "decode": "existing cumulative decode timer divided by 27 reported calls; not kernel execution time",
                              "placement": "complete ordered tracer lines for tested model on --no-gpu and --gpu-backend cuda; trace controls excluded from medians"}}

    def save():
        temporary = OUT / "receipt.json.tmp"
        temporary.write_text(json.dumps(receipt, indent=2) + "\n")
        temporary.replace(OUT / "receipt.json")

    save()
    try:
        main(receipt, save)
    except BaseException as error:
        receipt["failure"] = f"{type(error).__name__}: {error}"
        (OUT / "failure.log").write_text(traceback.format_exc())
        save()
        raise
