#!/usr/bin/env python3
"""Fresh Granite5 conversion and CPU acceptance against a successful F32 producer.

Candidates remain diagnostic checkpoints. This worker cannot establish CUDA or
publish a production registry entry. Each model runs in a separate process.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
MODEL = "ibm-granite/granite-speech-5.0-2b"
REVISION = "c075cf19e041a2e4d16c5ee24763e5b037d8f479"
STAGING = "cstr/crispasr-granite5-validation-public"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def words(text):
    return re.findall(r"[^\W_]+(?:'[^\W_]+)?", text.casefold())


def session_child(args):
    sys.path.insert(0, str(ROOT / "python"))
    from crispasr import Session
    pcm = np.load(args.pcm)
    with Session(str(args.model), lib_path=str(args.library), use_gpu=False, n_threads=4, verbosity=1) as session:
        assert session.backend == "granite", session.backend
        session.set_ask(args.instruction)
        outputs = {"official": [" ".join(s.text for s in session.transcribe(pcm, language="en")) for _ in range(2)]}
        session.set_ask("")
        outputs["legacy_default"] = " ".join(s.text for s in session.transcribe(pcm))
    args.child_output.write_text(json.dumps(outputs, ensure_ascii=False, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--producer-run", type=int)
    parser.add_argument("--producer-source")
    parser.add_argument("--session-child", action="store_true")
    for name in ("model", "library", "pcm", "child-output"):
        parser.add_argument(f"--{name}", type=Path)
    parser.add_argument("--instruction")
    args = parser.parse_args()
    if args.session_child:
        session_child(args)
        return
    assert args.producer_run and args.producer_source and len(args.producer_source) == 40
    from gguf import GGUFReader
    from huggingface_hub import HfApi, hf_hub_download, snapshot_download

    out = Path(os.environ["HEAVY_OUT"])
    scratch = Path(os.environ["HEAVY_SCRATCH"]) / "granite5-native"
    out.mkdir(parents=True, exist_ok=True)
    scratch.mkdir(parents=True, exist_ok=True)
    os.environ["TMPDIR"] = str(scratch)
    source = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    receipt = {"source": source, "producer_run": args.producer_run, "producer_source": args.producer_source,
               "cpu_acceptance": False, "cuda_acceptance": False, "production_release": False,
               "models": {}, "complete": False}

    def save():
        (out / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")

    def run(command, label, env=None, timeout=7200):
        meminfo = Path("/proc/meminfo").read_text()
        resources = {"load_average": os.getloadavg(), "cpus": os.cpu_count(), "meminfo": meminfo}
        (out / f"{label}-resources.json").write_text(json.dumps(resources, indent=2) + "\n")
        available = int(next(line.split()[1] for line in meminfo.splitlines() if line.startswith("MemAvailable:")))
        if available < 8 * 1024 * 1024 or resources["load_average"][0] > 2 * resources["cpus"]:
            raise RuntimeError(f"Insufficient hosted resource headroom before {label}")
        with (out / f"{label}.log").open("wb") as log:
            result = subprocess.run(list(map(str, command)), cwd=ROOT, stdout=log,
                                    stderr=subprocess.STDOUT, env=env, timeout=timeout)
        print(label, "exit", result.returncode, flush=True)
        if result.returncode:
            print((out / f"{label}.log").read_text(errors="replace")[-5000:], flush=True)
            raise RuntimeError(label)

    save()
    metadata = json.loads(subprocess.check_output([
        "gh", "api", f"repos/CrispStrobe/CrispASR/actions/runs/{args.producer_run}"]))
    assert metadata["status"] == "completed" and metadata["conclusion"] == "success"
    assert metadata["head_sha"] == args.producer_source and metadata["event"] == "workflow_dispatch"
    run(["gh", "run", "download", args.producer_run, "--repo", "CrispStrobe/CrispASR",
         "--name", f"heavy-{args.producer_run}", "--dir", scratch / "producer"], "producer-artifact")
    producer = json.loads((scratch / "producer/receipt.json").read_text())
    assert producer["complete"] and not producer["native_acceptance"]
    assert producer["source"] == args.producer_source and producer["revision"] == REVISION

    def fetch(record, directory):
        path = Path(hf_hub_download(record["repo"], record["path"], repo_type=record["repo_type"],
                                    revision=record["revision"], local_dir=directory))
        assert sha(path) == record["sha256"] and path.stat().st_size == record["bytes"]
        return path

    reference = fetch(producer["outputs"]["jfk-ref.gguf"], scratch / "reference")
    ref = GGUFReader(reference)
    arrays = {t.name: t.data.reshape(-1).copy() for t in ref.tensors}
    expected_text = ref.fields["crispasr.ref.generated_text"].contents()
    assert words(expected_text) and len(words(expected_text)) > 15
    instruction = producer["reference_recipe"]["prompt"].removeprefix("<|audio|>")
    receipt["reference_sha256"] = sha(reference)
    receipt["original_transcript"] = expected_text
    receipt["instruction"] = instruction
    pcm = out / "jfk-original.npy"
    np.save(pcm, arrays["raw_audio"])
    api = HfApi()
    card = api.model_info(STAGING).card_data
    assert card["license"] == "apache-2.0" and card["crispasr_native_acceptance"] is False
    api.upload_file(path_or_fileobj=b"Granite5 native LFS preflight\n" * 64,
                    path_in_repo=f"native/{source}/lfs-storage-preflight.bin", repo_id=STAGING)
    save()
    if not shutil.which("ccache") or not shutil.which("ninja"):
        run(["sudo", "apt-get", "update"], "apt-update")
        run(["sudo", "apt-get", "install", "-y", "ccache", "ninja-build"], "apt-tools")
    build = scratch / "build"
    run(["cmake", "-S", ROOT, "-B", build, "-G", "Ninja", "-DCMAKE_BUILD_TYPE=Release",
         "-DCMAKE_C_COMPILER_LAUNCHER=ccache", "-DCMAKE_CXX_COMPILER_LAUNCHER=ccache",
         "-DGGML_NATIVE=OFF", "-DGGML_BLAS=OFF", "-DCRISPASR_BUILD_SERVER=OFF",
         "-DCRISPASR_BUILD_TESTS=ON"], "configure")
    run(["cmake", "--build", build, "--target", "crispasr-cli", "crispasr-diff", "crispasr-quantize",
         "granite5-stage-probe", "granite5-rope-probe", "test-granitespeech-params", "test-granitenle-params", "-j4"],
        "build")
    for test in ("test-granitespeech-params", "test-granitenle-params"):
        run([build / "bin" / test], test)
    oracle = scratch / "producer/original-yarn.npz"
    assert sha(oracle) == producer["outputs"]["original-yarn.npz"]["sha256"]
    run([sys.executable, ROOT / "tools/ci-heavy/granite5_rope_parity.py", "--probe", build / "bin/granite5-rope-probe",
         "--oracle", oracle, "--out", out / "rope-parity"], "original-rope-parity")
    env = dict(os.environ, LD_LIBRARY_PATH=":".join(sorted({str(p.parent) for p in build.rglob("*.so")})))
    library = next(build.rglob("libcrispasr.so"))
    original = Path(snapshot_download(MODEL, revision=REVISION, ignore_patterns=["out_llm.safetensors"]))
    for path in original.rglob("*"):
        if path.is_file():
            identity = producer["source_files"][str(path.relative_to(original))]
            assert sha(path) == identity["sha256"] and path.stat().st_size == identity["bytes"]
    f16 = scratch / "granite-speech-5.0-2b-f16.gguf"
    run([sys.executable, ROOT / "models/convert-granite-speech-to-gguf.py", "--input", original, "--output", f16],
        "fresh-conversion")

    def checkpoint(path):
        record = {"sha256": sha(path), "bytes": path.stat().st_size, "repo": STAGING, "repo_type": "model",
                  "path": f"native/{source}/{path.name}"}
        try:
            commit = api.upload_file(path_or_fileobj=path, path_in_repo=record["path"], repo_id=STAGING,
                                     commit_message="Checkpoint unaccepted Granite5 native candidate")
        except Exception:
            fallback = out / "failed-transfer"
            fallback.mkdir(exist_ok=True)
            shutil.copyfile(path, fallback / path.name)
            raise
        record["revision"] = commit.oid
        receipt["models"][path.stem] = {"file": record, "accepted": False}
        save()

    checkpoint(f16)
    baseline = fetch(producer["outputs"]["granite-speech-5.0-2b-baseline-f16.gguf"], scratch / "baseline")
    old, fresh = GGUFReader(baseline), GGUFReader(f16)
    old_tensors = {t.name: t for t in old.tensors}
    assert len(old_tensors) == len(fresh.tensors)
    tensor_audit = {}
    for tensor in fresh.tensors:
        before = old_tensors[tensor.name]
        assert tensor.tensor_type == before.tensor_type and np.array_equal(tensor.shape, before.shape)
        digest = hashlib.sha256(tensor.data).hexdigest()
        assert digest == hashlib.sha256(before.data).hexdigest(), tensor.name
        tensor_audit[tensor.name] = digest
    assert fresh.fields["granite_speech.llm.rope_scaling.type"].contents() == "yarn"
    assert fresh.fields["granite_speech.llm.rope_scaling.original_context"].contents() == 4096
    (out / "exact-converted-tensors.json").write_text(json.dumps(tensor_audit, indent=2) + "\n")
    del old, fresh, old_tensors, before, tensor
    baseline.unlink()

    def metrics(expected, actual):
        assert expected.shape == actual.shape and np.isfinite(actual).all()
        a, b = expected.astype(np.float64), actual.astype(np.float64)
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        assert na > 0 and nb > 0
        return {"cosine": float(np.dot(a, b) / (na * nb)), "norm_ratio": float(nb / na),
                "relative_l2": float(np.linalg.norm(a - b) / na), "maxabs": float(np.max(np.abs(a - b)))}

    for quant in ("f16", "q8_0", "q4_k"):
        model = f16 if quant == "f16" else scratch / f"granite-speech-5.0-2b-{quant}.gguf"
        if quant != "f16":
            run([build / "bin/crispasr-quantize", f16, model, quant], f"quantize-{quant}")
            checkpoint(model)
        result = receipt["models"][model.stem]
        case = out / quant
        run([build / "bin/granite5-stage-probe", model, reference, case, instruction, "cpu"], f"stages-{quant}", env)
        result["stages"] = {}
        for name in ("mel_spectrogram", "encoder_out", "projector_out", "llm_inputs_embeds", "llm_prefill_logits"):
            m = metrics(arrays[name], np.fromfile(case / f"{name}.bin", dtype="<f4"))
            result["stages"][name] = m
            save()
            # F16 input equality is the structural gate. The quantizer also
            # quantizes token embeddings; for those, IDs stay exact and the
            # original-input prefill separately isolates decoder arithmetic.
            assert m["cosine"] >= (0.99999 if name == "llm_inputs_embeds" and quant == "f16" else 0.99), (quant, name, m)
            assert 0.99 <= m["norm_ratio"] <= 1.01, (quant, name, m)
        assert np.array_equal(arrays["prompt_ids"], np.fromfile(case / "prompt_ids.bin", dtype="<i4"))
        native_logits = np.fromfile(case / "llm_prefill_logits.bin", dtype="<f4")
        original_pick = int(np.argmax(arrays["llm_prefill_logits"]))
        result["original_first_token_rank"] = int(np.count_nonzero(native_logits > native_logits[original_pick])) + 1
        if quant == "f16":
            assert result["original_first_token_rank"] == 1
        else:
            assert result["original_first_token_rank"] <= 2
            assert float(np.max(native_logits) - native_logits[original_pick]) < 1.0
        assert np.array_equal(np.fromfile(case / "generated_ids.bin", dtype="<i4"),
                              np.fromfile(case / "dynamic_generated_ids.bin", dtype="<i4"))
        text = (case / "generated_text.txt").read_text()
        assert words(text) == words(expected_text), (quant, text, expected_text)
        result["native_text"] = text
        run([build / "bin/crispasr-diff", "granite", model, reference, ROOT / "samples/jfk.wav"], f"diff-{quant}", env)
        alias = scratch / f"renamed-{quant}.gguf"
        os.link(model, alias)
        run([sys.executable, __file__, "--session-child", "--model", alias, "--library", library,
             "--pcm", pcm, "--instruction", instruction, "--child-output", case / "session.json"],
            f"session-{quant}", env)
        sessions = json.loads((case / "session.json").read_text())
        assert all(words(t) == words(expected_text) for t in sessions["official"]), sessions
        assert words(sessions["legacy_default"]) == words(expected_text), sessions
        alias.unlink()
        result["session_texts"] = sessions
        cli_base = case / "cli-official"
        run([build / "bin/crispasr", "-m", model, "-f", ROOT / "samples/jfk.wav", "-l", "en",
             "--ask", instruction, "--max-new-tokens", "96", "--no-gpu", "--no-prints",
             "-otxt", "-of", cli_base], f"cli-{quant}", env)
        cli_text = Path(str(cli_base) + ".txt").read_text()
        assert words(cli_text) == words(expected_text), (quant, cli_text, expected_text)
        result["cli_text"] = cli_text
        result["accepted"] = True
        save()
        if quant != "f16":
            model.unlink()
    receipt["cpu_acceptance"] = True
    receipt["complete"] = True
    save()
    api.upload_folder(folder_path=out, path_in_repo=f"native/{source}/cpu-evidence", repo_id=STAGING)
    (out / "summary.md").write_text("Granite5 original F32 stage/decoder/Session CPU gates passed. CUDA remains pending.\n")


if __name__ == "__main__":
    main()
