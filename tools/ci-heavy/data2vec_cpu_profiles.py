#!/usr/bin/env python3
"""Diagnose data2vec CPU-profile sensitivity without changing acceptance gates.

Hosted heavy-cpu.yml only; pip: numpy gguf==0.17.1 huggingface_hub.
One source, immutable inputs, sequential native/AVX2/AVX512 builds. Raw CTC,
explicit-English CLI and the exact nightly invocation are separate controls.
Successful execution means evidence collected, not a newly qualified model.
Use --require-avx512 for a follow-up: unsuitable workers retain preflight
evidence and stop before worker APT/model downloads or compilation.
"""

import argparse
import hashlib
from itertools import groupby
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/regression"))
import run_one as regression  # noqa: E402

MODEL_HASHES = {
    "q4": "93b6ab01f1f83525157d797a385a3e9e014c6761d3e974351363adc452a86f7e",
    "f16": "4df9cc50b7f5340488fa64f5ead34f93660c6ff8785e7e71a859a8860a13254e",
}
AVX512_FLAGS = {"avx512f", "avx512cd", "avx512bw", "avx512dq", "avx512vl"}


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def headroom(scratch):
    memory = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
    state = {"available_bytes": int(memory["MemAvailable"].split()[0]) * 1024,
             "load": list(os.getloadavg()), "cpus": os.cpu_count(),
             "scratch_free_bytes": shutil.disk_usage(scratch).free}
    if (state["available_bytes"] < 6 * 1024**3 or
            state["scratch_free_bytes"] < 8 * 1024**3 or
            state["load"][0] > 2 * state["cpus"]):
        raise RuntimeError(f"insufficient worker resources: {state}")
    return state


def run(argv, out, label, scratch, *, compute=False, timeout=1200):
    if compute:
        (out / f"{label}.resources.json").write_text(json.dumps(headroom(scratch), indent=2) + "\n")
    (out / f"{label}.argv.json").write_text(json.dumps(list(map(str, argv))) + "\n")
    with (out / f"{label}.stdout.log").open("w") as stdout, (out / f"{label}.stderr.log").open("w") as stderr:
        proc = subprocess.run(list(map(str, argv)), cwd=ROOT, stdout=stdout, stderr=stderr,
                              timeout=timeout, check=False)
    if proc.returncode:
        raise RuntimeError(f"{label} failed with {proc.returncode}; see retained logs")
    return (out / f"{label}.stdout.log").read_text()


def decode(values, inverse, blank):
    ids = values.argmax(axis=1).tolist()
    text = "".join(inverse[i] for i, _ in groupby(ids) if i != blank).replace("|", " ").strip()
    return text, ids


def compare_logits(native, reference, inverse, blank):
    import numpy as np

    if native.shape != reference.shape or not np.isfinite(native).all():
        raise ValueError("invalid native CTC shape or non-finite logits")
    text, ids = decode(native, inverse, blank)
    _, ref_ids = decode(reference, inverse, blank)
    divergent = np.flatnonzero(np.asarray(ids) != ref_ids)
    probes = []
    for frame in divergent:
        top = np.argsort(native[frame])[-2:][::-1]
        ref_top = np.argsort(reference[frame])[-2:][::-1]
        probes.append({"frame": int(frame),
                       "native_top2": [[inverse[int(i)], float(native[frame, i])] for i in top],
                       "reference_top2": [[inverse[int(i)], float(reference[frame, i])] for i in ref_top]})
    delta = native.astype(np.float64) - reference
    return {"raw_transcript": text, "argmax_ids": ids, "divergent_frames": probes,
            "relative_l2": float(np.linalg.norm(delta) / np.linalg.norm(reference)),
            "norm_ratio": float(np.linalg.norm(native) / np.linalg.norm(reference)),
            "max_abs": float(np.abs(delta).max())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-avx512", action="store_true")
    args = parser.parse_args()
    out = Path(os.environ["HEAVY_OUT"])
    scratch = Path(os.environ["HEAVY_SCRATCH"]) / "data2vec-cpu-profiles"
    out.mkdir(parents=True, exist_ok=True)
    scratch.mkdir(parents=True, exist_ok=True)
    os.environ["TMPDIR"] = str(scratch)
    manifest = json.loads((ROOT / "tests/regression/manifest.json").read_text())
    entry = next(e for e in manifest["backends"] if e["name"] == "data2vec-base")
    receipt = {"complete": False, "scope": __doc__, "profiles": {},
               "require_avx512": args.require_avx512,
               "source": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
               "pins": {"model": entry["gguf"], "fixtures": manifest["fixtures"],
                        "transcript_reference": entry["transcript_reference"]}}

    def save():
        (out / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")

    save()
    try:
        receipt["initial_resources"] = headroom(scratch)
        for label, argv in [("cpu", ["lscpu"]), ("compiler", ["c++", "--version"]),
                            ("submodules", ["git", "submodule", "status", "--recursive"])]:
            run(argv, out, label, scratch)
        cpuinfo = Path("/proc/cpuinfo").read_text()
        (out / "cpuinfo.txt").write_text(cpuinfo)
        flags = set(next(line.split(":", 1)[1].split() for line in cpuinfo.splitlines() if line.startswith("flags")))
        if not {"avx2", "fma", "f16c"} <= flags:
            raise RuntimeError("worker must support AVX2/FMA/F16C")
        receipt["avx512_unavailable"] = not AVX512_FLAGS <= flags
        if args.require_avx512 and receipt["avx512_unavailable"]:
            receipt["skipped"] = "avx512-unavailable"
            save()
            (out / "summary.md").write_text(
                "AVX512 unavailable: preflight only; no worker APT/model work, builds or numerical acceptance.\n")
            return
        run(["bash", ROOT / "tools/ci-apt.sh", "update"], out, "apt-update", scratch)
        run(["bash", ROOT / "tools/ci-apt.sh", "install", "-y", "ccache", "libopenblas-dev",
             "libespeak-ng-dev", "espeak-ng", "pkg-config"], out, "apt-install", scratch)
        import gguf
        import numpy as np

        get = lambda spec: regression.hf_download(spec["repo"], spec["file"], spec["revision"], scratch)
        models = {"q4": get(entry["gguf"]), "f16": get({**entry["gguf"], **entry["diff_gguf"]})}
        for name, path in models.items():
            if sha256(path) != MODEL_HASHES[name]:
                raise ValueError(f"immutable {name} model hash mismatch")
        ref = regression.hf_download(manifest["fixtures"]["repo"], entry["fixture_ref_path"],
                                     manifest["fixtures"]["revision"], scratch)
        sample = ROOT / entry["sample"]
        receipt["independent_transcript"] = regression.verify_transcript_reference(entry, ref, sample, scratch)
        vocab = json.loads(get(entry["transcript_reference"]["vocab"]).read_text())
        inverse = {value: key for key, value in vocab.items()}
        blank = vocab[entry["transcript_reference"]["blank_token"]]
        reference = next(t.data.copy() for t in gguf.GGUFReader(str(ref)).tensors
                         if t.name == entry["transcript_reference"]["stage"])
        receipt["input_sha256"] = {name: sha256(path) for name, path in
                                   {**models, "reference": ref, "sample": sample}.items()}
        profiles = ["native", "avx2"]
        if AVX512_FLAGS <= flags:
            profiles.append("avx512")
        for profile in profiles:
            case = out / profile
            case.mkdir(exist_ok=True)
            build = scratch / f"build-{profile}"
            opts = ["-DGGML_NATIVE=ON"] if profile == "native" else [
                "-DGGML_NATIVE=OFF", "-DGGML_AVX=ON", "-DGGML_AVX2=ON", "-DGGML_FMA=ON", "-DGGML_F16C=ON",
                f"-DGGML_AVX512={'ON' if profile == 'avx512' else 'OFF'}",
                "-DGGML_AVX_VNNI=OFF", "-DGGML_AVX512_VBMI=OFF", "-DGGML_AVX512_VNNI=OFF",
                "-DGGML_AVX512_BF16=OFF", "-DGGML_AMX_TILE=OFF", "-DGGML_AMX_INT8=OFF", "-DGGML_AMX_BF16=OFF"]
            run(["cmake", "-S", ROOT, "-B", build, "-G", "Ninja", "-DCMAKE_BUILD_TYPE=Release",
                 "-DCMAKE_C_COMPILER_LAUNCHER=ccache", "-DCMAKE_CXX_COMPILER_LAUNCHER=ccache",
                 "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON", "-DCRISPASR_BUILD_TESTS=OFF",
                 "-DCRISPASR_BUILD_EXAMPLES=ON", "-DCRISPASR_BUILD_SERVER=OFF", *opts],
                case, "configure", scratch, compute=True)
            run(["cmake", "--build", build, "--target", "crispasr-cli", "crispasr-diff", "-j2"],
                case, "build", scratch, compute=True, timeout=3600)
            for name in ("compile_commands.json", "CMakeCache.txt"):
                shutil.copyfile(build / name, case / name)
            commands = json.loads((build / "compile_commands.json").read_text())
            cpu_commands = [row["command"] for row in commands if "/ggml-cpu/" in row["file"]]
            if not cpu_commands:
                raise ValueError("missing actual CPU compiler commands")
            tokens = {token for cmd in cpu_commands for token in shlex.split(cmd)}
            if profile == "avx2" and ("-mavx2" not in tokens or any(t.startswith("-mavx512") for t in tokens)):
                raise ValueError("AVX2 actual compiler flags do not match requested profile")
            if profile == "avx512" and "-mavx512f" not in tokens:
                raise ValueError("AVX512 actual compiler flags missing")
            row = receipt["profiles"][profile] = {"cmake_options": opts, "cpu_flags": sorted(
                t for t in tokens if t.startswith(("-m", "-DGGML_"))), "models": {}}
            cli, diff = build / "bin/crispasr", build / "bin/crispasr-diff"
            for name, model in models.items():
                result = row["models"][name] = {"raw_repeats": []}
                for repeat in range(3):
                    label = f"{name}-raw-{repeat}"
                    prefix = case / label
                    text = run([cli, "-m", model, "-f", sample, "-l", "en", "-t", "4",
                                "--no-punctuation", "--return-logits", "-of", prefix],
                               case, label, scratch, compute=True)
                    payload = json.loads(Path(str(prefix) + ".ctc-logits.json").read_text())["ctc_logits"]
                    values = np.asarray(payload["data"], dtype=np.float32).reshape(payload["n_frames"], payload["n_vocab"])
                    observation = compare_logits(values, reference, inverse, blank)
                    observation["cli_transcript"] = text.strip().splitlines()[-1]
                    observation["transcript_gate"] = regression.transcript_gate(entry, observation["raw_transcript"])[0]
                    observation["logits_sha256"] = hashlib.sha256(values.tobytes()).hexdigest()
                    result["raw_repeats"].append(observation)
                    save()
                result["fixed_language_pipeline"] = run([cli, "-m", model, "-f", sample, "-l", "en", "-t", "4"],
                    case, f"{name}-pipeline", scratch, compute=True).strip()
                result["nightly_invocation"] = run([cli, "-m", model, "-f", sample],
                    case, f"{name}-nightly", scratch, compute=True).strip()
                result["repeat_logits_identical"] = len({r["logits_sha256"] for r in result["raw_repeats"]}) == 1
                save()
            raw = run([diff, entry["backend_id"], models["f16"], ref, sample],
                      case, "f16-stage", scratch, compute=True)
            stages = regression.parse_diff_stdout(raw)
            passed, failed, missing, extras = regression.evaluate_stage_thresholds(
                stages, entry["diff_thresholds"], **regression.stage_gate_kwargs(entry))
            row["f16_stage"] = {"values": stages, "passed": passed, "failed": failed,
                                "missing": missing, "extras": extras,
                                "gate": bool(stages and not failed and not missing)}
            if not row["f16_stage"]["gate"]:
                raise ValueError("unchanged F16 stage gate failed")
            save()
            shutil.rmtree(build)
        receipt["complete"] = True
        receipt["acceptance_changed"] = False
        save()
        (out / "summary.md").write_text("Data2vec CPU-profile diagnostic collected. See receipt.json; no acceptance gate changed.\n")
    except Exception as exc:
        receipt["error"] = f"{type(exc).__name__}: {exc}"
        save()
        raise


if __name__ == "__main__":
    main()
