#!/usr/bin/env python3
"""Execute a CPU-qualified, prebuilt CSM correction on an actual Kaggle GPU.

No builds or CPU-model inference. The original operator reference, successful
CPU speech/port receipts and licensed runtime must be supplied explicitly.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

import csm_cpu_profile_ab as profile
import csm_rope_component_check as check


ROOT = Path(__file__).resolve().parents[2]


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("bundle", "cpu-receipt", "port-receipt", "reference", "scratch", "out"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    args = parser.parse_args()
    if not Path("/kaggle/working").is_dir() or os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise RuntimeError("actual Kaggle GPU0 is required; no CPU fallback")
    inherited = sorted(key for key in os.environ if key.startswith(("CRISPASR_", "GGML_")))
    if inherited:
        raise RuntimeError("inherited runtime overrides invalidate controlled CUDA acceptance: " + ", ".join(inherited))
    args.out.mkdir(parents=True, exist_ok=True)
    args.scratch.mkdir(parents=True, exist_ok=True)
    receipt = {"complete": False, "acceptance": False, "scope": __doc__, "commands": {},
               "source": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()}

    def save():
        (args.out / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")

    def run(argv, label, *, resident_model=False, env=None):
        memory = Path("/proc/meminfo").read_text()
        available = int(next(line.split()[1] for line in memory.splitlines() if line.startswith("MemAvailable:")))
        state = {"load": os.getloadavg(), "cpus": os.cpu_count(), "available_kib": available,
                 "gpu": subprocess.check_output(["nvidia-smi", "--id=0", "--query-gpu=memory.free,memory.total",
                                                 "--format=csv,noheader,nounits"], text=True).strip()}
        gpu_free = int(state["gpu"].split(",")[0])
        receipt["commands"][label] = {"argv": list(map(str, argv)), "resources": state}
        save()
        if (available < 4 * 1024**2 or state["load"][0] > 2 * state["cpus"] or
                gpu_free < (3072 if resident_model else 128)):
            raise RuntimeError(f"insufficient GPU/host headroom before {label}")
        peak = 0
        with (args.out / f"{label}.stdout.log").open("wb") as stdout, \
                (args.out / f"{label}.stderr.log").open("wb") as stderr:
            proc = subprocess.Popen(list(map(str, argv)), cwd=args.bundle, env=env, stdout=stdout, stderr=stderr)
            deadline = time.monotonic() + 300
            while proc.poll() is None:
                rows = subprocess.check_output(["nvidia-smi", "--id=0", "--query-compute-apps=pid,used_gpu_memory",
                                                "--format=csv,noheader,nounits"], text=True)
                for row in rows.splitlines():
                    pid, _, memory = row.partition(",")
                    if pid.strip() == str(proc.pid) and memory.strip().isdigit():
                        peak = max(peak, int(memory.strip()))
                if time.monotonic() > deadline:
                    proc.kill()
                    proc.wait()
                    raise TimeoutError(label)
                time.sleep(0.2)
        receipt["commands"][label].update(rc=proc.returncode, peak_gpu_mib=peak)
        save()
        if proc.returncode or (resident_model and peak < 512):
            raise RuntimeError(f"{label}: failed command or missing GPU model residency")

    save()
    try:
        cpu = json.loads(args.cpu_receipt.read_text())
        port = json.loads(args.port_receipt.read_text())
        candidate = cpu.get("original_rope_control", {})
        if (cpu.get("complete") is not True or candidate.get("speech_acceptance") is not True or
                candidate.get("traced_untraced_pcm_identical") is not True or
                len(candidate.get("scores", {})) != 8 or
                not all(cell.get("gate") is True for cell in candidate["scores"].values()) or
                candidate.get("default_vs_explicit_on") is not True or
                port.get("complete") is not True or port.get("f16_original_parity") is not True):
            raise RuntimeError("successful opt-in CPU speech and original F16 port receipts are required")
        from csm_native_prefill import EXPECTED
        from csm_prefill_parity import F16_GATES, MODEL_SHA
        if (port.get("acceptance") is not True or port.get("f16_gates") != F16_GATES or
                port.get("published_q4_payloads_match_fresh_quantization") is not True or
                port.get("models", {}).get("published-q4", {}).get("sha256") != MODEL_SHA or
                cpu.get("model_sha256", {}).get("csm") != MODEL_SHA):
            raise RuntimeError("F16 port gates and published Q4 coefficient lineage must be qualified")
        for attention in ("default", "eager-f32"):
            stages = port.get("comparisons", {}).get(f"f16-original-{attention}", {})
            if set(stages) != set(EXPECTED) or not all(
                    result.get("f16_port_gate") is True and result["rel_l2"] <= F16_GATES["rel_l2"] and
                    abs(result["norm_ratio"] - 1) <= F16_GATES["norm_ratio_delta"] and
                    result["cosine"] >= F16_GATES["cosine_min"] and result["max_abs"] <= F16_GATES["max_abs"]
                    for result in stages.values()):
                raise RuntimeError("all nineteen F16 stage gates must pass under both attention controls")
        if (set(candidate.get("prefill", {})) != {"native", "avx2"} or
                not all(value.get("complete") is True for value in candidate["prefill"].values()) or
                not all(cpu["profiles"][name].get("rope_component", {}).get("pass") is True
                        for name in ("native", "avx2"))):
            raise RuntimeError("both CPU component and sampled/prefill controls must be complete")
        provenance = json.loads((args.bundle / "provenance.json").read_text())
        for path, entry in provenance["files"].items():
            file = args.bundle / path
            if not file.resolve().is_relative_to(args.bundle.resolve()) or sha(file) != entry["sha256"]:
                raise RuntimeError(f"runtime inventory differs: {path}")
        # Tools-only commits may differ. Every compiled native source and the
        # GGML revision must still match the CPU qualification's source tree.
        for qualification, qualified_source in (("speech", cpu["source"]), ("F16 port", port["source"])):
            for path, expected in provenance["runtime_sources"].items():
                content = subprocess.check_output(["git", "show", f"{qualified_source}:{path}"], cwd=ROOT)
                if hashlib.sha256(content).hexdigest() != expected:
                    raise RuntimeError(f"compiled runtime differs from CPU {qualification} qualification: {path}")
            gitlink = subprocess.check_output(["git", "ls-tree", qualified_source, "ggml"], cwd=ROOT, text=True)
            if provenance["ggml"] not in gitlink or provenance.get("host_provided_nvidia_libraries") is not True:
                raise RuntimeError("GGML revision or NVIDIA runtime policy differs")
        if any(cpu["profiles"][name]["rope_component"]["probe_source_sha256"] !=
               provenance["probe_sources"]["csm_rope_probe.cpp"] for name in ("native", "avx2")):
            raise RuntimeError("compiled CUDA operator probe differs from CPU component probe")
        original = json.loads((args.reference / "receipt.json").read_text())
        if sha(args.reference / "receipt.json") != profile.ROPE_REFERENCE_RECEIPT:
            raise RuntimeError("original operator artifact differs from the reviewed producer")
        if not check.compare(args.reference, args.reference)["pass"]:
            raise RuntimeError("original operator fixture fails its locked controls")
        receipt.update(cpu_receipt_sha256=sha(args.cpu_receipt), port_receipt_sha256=sha(args.port_receipt),
                       runtime_provenance_sha256=sha(args.bundle / "provenance.json"),
                       hardware=subprocess.check_output(["nvidia-smi", "--id=0",
                                                          "--query-gpu=name,compute_cap,driver_version",
                                                          "--format=csv,noheader"], text=True).strip())
        save()
        from granite5_cuda_preflight import run_preflight
        preflight = run_preflight(args.bundle, args.out / "runtime-preflight.json")
        if not preflight["runtime_preflight_pass"]:
            raise RuntimeError("actual host CUDA runtime loader preflight failed")
        env = dict(os.environ)
        env.update(CRISPASR_CSM_ROPE_DEVICE="cuda", CRISPASR_CSM_ORIGINAL_ROPE="1",
                   CRISPASR_CORE_ATTN_EAGER_F32="0", LD_LIBRARY_PATH=preflight["ld_library_path"],
                   GGML_BACKEND_PATH=str(args.bundle.resolve()), OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="4")
        env.pop("CRISPASR_CSM_TRACE_PREFIX", None)
        for mode in ("original", "generated", "unscaled"):
            for case in check.CASES:
                cfg = original["cases"][case]["configuration"]
                scaling = cfg["rope_scaling"]
                dest = args.out / mode / case
                argv = [args.bundle / "csm-rope-probe", args.reference / case, dest, cfg["head_dim"],
                        cfg["num_attention_heads"], cfg["num_key_value_heads"],
                        len(original["cases"][case]["positions"]), scaling["original_max_position_embeddings"],
                        cfg["rope_theta"], mode]
                if mode == "generated":
                    argv += [scaling["factor"], scaling["low_freq_factor"], scaling["high_freq_factor"]]
                label = f"component-{mode}-{case}"
                run(argv, label, env=env)
                reported = json.loads((args.out / f"{label}.stdout.log").read_text())
                if not reported.get("backend", "").startswith("CUDA") or reported.get("fallback") is not False:
                    raise RuntimeError("operator did not execute on actual CUDA")
            result = check.compare(args.reference, args.out / mode)
            receipt[f"component_{mode}"] = result
            save()
            if mode == "unscaled":
                from csm_rope_component_run import _unscaled_check
                receipt["component_unscaled_control"] = _unscaled_check(args.reference, args.out / mode, original)
            elif not result["pass"]:
                raise RuntimeError(f"CUDA {mode} component parity failed")
        from huggingface_hub import hf_hub_download
        paths = {}
        for key in ("csm", "asr"):
            spec = cpu["pins"][key]["gguf"]
            paths[key] = Path(hf_hub_download(spec["repo"], spec["file"], revision=spec["revision"],
                                            local_dir=args.scratch / "models"))
            if sha(paths[key]) != cpu["model_sha256"][key]:
                raise RuntimeError(f"physical {key} model differs from CPU qualification")
        cli = args.bundle / "crispasr"
        phrase = cpu["pins"]["csm"]["tts_phrase"]
        if phrase != "The quick brown fox jumps over the lazy dog" or cpu["pins"]["csm"]["wer_max"] != 0.15:
            raise RuntimeError("speech gate differs from original regression")
        receipt["speech"] = {}
        pcm_hashes = []
        for repeat in range(2):
            wav = args.out / f"speech-{repeat}.wav"
            if repeat == 0:
                env.pop("CRISPASR_CSM_ORIGINAL_ROPE", None)
            else:
                env["CRISPASR_CSM_ORIGINAL_ROPE"] = "1"
            run([cli, "--backend", "csm", "-m", paths["csm"], "--gpu-backend", "cuda", "--tts", phrase,
                 "--voice", "default", "--seed", "1234", "--tts-output", wav, "-t", "4"],
                f"synth-{repeat}", resident_model=True, env=env)
            info = profile.pcm_info(wav)
            if (info.get("channels"), info.get("rate"), info.get("sample_width")) != (1, 24000, 2):
                raise RuntimeError("CUDA synthesis produced malformed audio")
            pcm_hashes.append(info["pcm_sha256"])
            run([cli, "--backend", cpu["pins"]["asr"]["backend_id"], "-m", paths["asr"], "-f", wav,
                 "--gpu-backend", "cuda", "--no-prints", "-t", "4"],
                f"judge-{repeat}", resident_model=True, env=env)
            stdout = (args.out / f"judge-{repeat}.stdout.log").read_text()
            text = profile.ab.regression.strip_ai_disclaimer(next(
                (line.strip() for line in reversed(stdout.splitlines()) if line.strip()), ""))
            wer = profile.ab.regression.compute_transcript_metrics(phrase, text)[1]
            receipt["speech"][str(repeat)] = {"wav": info, "transcript": text, "wer": wer,
                                              "wer_max": 0.15, "gate": wer <= 0.15}
            save()
        receipt["identical_repeat_pcm"] = pcm_hashes[0] == pcm_hashes[1]
        receipt["complete"] = True
        receipt["acceptance"] = receipt["identical_repeat_pcm"] and all(
            value["gate"] for value in receipt["speech"].values())
        save()
        return 0 if receipt["acceptance"] else 1
    except Exception as exc:
        receipt["error"] = f"{type(exc).__name__}: {exc}"
        save()
        raise


if __name__ == "__main__":
    sys.exit(main())
