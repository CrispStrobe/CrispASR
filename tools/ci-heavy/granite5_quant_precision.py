#!/usr/bin/env python3
"""Hosted diagnostic: isolate head/embedding precision without changing block weights.

Consumes the retained candidate files from the first CPU run, including its
failed Q4 gate. No input candidate is treated as accepted by that job's failure.
Only controlled tensor substitutions are made; no production registry updates.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np

from granite5_native_acceptance import sha, words
from granite5_result_audit import metrics, token_comparison

ROOT = Path(__file__).resolve().parents[2]
INPUT_RUN = 39970747786
INPUT_SOURCE = "c433b9f57347e357922770ad2b8f74c019ab70d0"


def substitute(source, f16, destination, names):
    from gguf import GGUFReader, GGUFWriter
    before, original = GGUFReader(source), GGUFReader(f16)
    originals = {t.name: t for t in original.tensors}
    writer = GGUFWriter(destination, before.fields["general.architecture"].contents(), use_temp_file=True)
    for name, field in before.fields.items():
        if name.startswith("GGUF.") or name == "general.architecture":
            continue
        writer.add_key_value(name, field.contents(), field.types[0], field.types[-1] if len(field.types) > 1 else None)
    expected = {}
    found = set()
    for tensor in before.tensors:
        chosen = originals[tensor.name] if tensor.name in names else tensor
        if tensor.name in names:
            assert np.array_equal(tensor.shape, chosen.shape)
            found.add(tensor.name)
        writer.add_tensor(chosen.name, chosen.data, raw_dtype=chosen.tensor_type)
        expected[tensor.name] = {"sha256": hashlib.sha256(chosen.data).hexdigest(),
                                 "shape": tensor.shape.tolist(), "type": int(chosen.tensor_type)}
    assert found == set(names)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    after = GGUFReader(destination)
    assert len(after.tensors) == len(expected)
    for tensor in after.tensors:
        row = expected[tensor.name]
        assert row["shape"] == tensor.shape.tolist() and row["type"] == int(tensor.tensor_type)
        assert hashlib.sha256(tensor.data).hexdigest() == row["sha256"], tensor.name
    return {"preserved_from_f16": sorted(names), "verified_tensors": len(expected),
            "source_sha256": sha(source), "f16_sha256": sha(f16), "output_sha256": sha(destination)}


def main():
    from gguf import GGUFReader
    from huggingface_hub import HfApi, hf_hub_download
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    out = Path(os.environ["HEAVY_OUT"])
    scratch = Path(os.environ["HEAVY_SCRATCH"]) / "granite5-precision"
    out.mkdir(parents=True, exist_ok=True)
    scratch.mkdir(parents=True, exist_ok=True)
    os.environ["TMPDIR"] = str(scratch)
    source = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    receipt = {"source": source, "input_run": INPUT_RUN, "input_source": INPUT_SOURCE,
               "complete": False, "production_release": False, "cases": {}}

    def save():
        (out / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")

    def run(command, tag, timeout=14400):
        # Hosted resource evidence immediately before compilation/inference.
        resource = {"load_average": os.getloadavg(), "cpus": os.cpu_count(),
                    "meminfo": Path("/proc/meminfo").read_text()}
        (out / f"{tag}-resources.json").write_text(json.dumps(resource, indent=2) + "\n")
        available_kib = int(next(line.split()[1] for line in resource["meminfo"].splitlines()
                                 if line.startswith("MemAvailable:")))
        if available_kib < 5 * 1024 * 1024 or resource["load_average"][0] > 2 * resource["cpus"]:
            raise RuntimeError(f"Insufficient hosted resource headroom before {tag}; snapshot retained")
        with (out / f"{tag}.log").open("wb") as log:
            result = subprocess.run(list(map(str, command)), cwd=ROOT, stdout=log,
                                    stderr=subprocess.STDOUT, timeout=timeout)
        print(tag, result.returncode, flush=True)
        if result.returncode:
            raise RuntimeError(f"{tag} failed; see retained log")

    save()
    record = json.loads(subprocess.check_output(["gh", "api", f"repos/CrispStrobe/CrispASR/actions/runs/{INPUT_RUN}"]))
    assert record["head_sha"] == INPUT_SOURCE and record["status"] == "completed"
    assert record["conclusion"] == "failure" and record["event"] == "workflow_dispatch"
    run(["gh", "run", "download", INPUT_RUN, "--repo", "CrispStrobe/CrispASR", "--name", f"heavy-{INPUT_RUN}",
         "--dir", scratch / "input"], "input-artifact")
    prior = json.loads((scratch / "input/receipt.json").read_text())
    assert prior["source"] == INPUT_SOURCE and not prior["complete"] and not prior["cpu_acceptance"]
    for quant in ("f16", "q8_0"):
        assert prior["models"][f"granite-speech-5.0-2b-{quant}"]["accepted"]
    assert not prior["models"]["granite-speech-5.0-2b-q4_k"]["accepted"]
    # Fetch the independently successful original producer again, not a guessed
    # transcript or an activation dump from the converted model.
    original_run = json.loads(subprocess.check_output(["gh", "api",
        f"repos/CrispStrobe/CrispASR/actions/runs/{prior['producer_run']}"]))
    assert original_run["conclusion"] == "success" and original_run["head_sha"] == prior["producer_source"]
    run(["gh", "run", "download", prior["producer_run"], "--repo", "CrispStrobe/CrispASR",
         "--name", f"heavy-{prior['producer_run']}", "--dir", scratch / "original"], "original-artifact")
    original = json.loads((scratch / "original/receipt.json").read_text())
    assert original["complete"] and original["source"] == prior["producer_source"]

    def fetch(item):
        path = Path(hf_hub_download(item["repo"], item["path"], repo_type=item["repo_type"], revision=item["revision"],
                                    local_dir=scratch / "models"))
        assert sha(path) == item["sha256"] and path.stat().st_size == item["bytes"]
        return path

    reference = fetch(original["outputs"]["jfk-ref.gguf"])
    refs = GGUFReader(reference)
    arrays = {t.name: t.data.reshape(-1).copy() for t in refs.tensors}
    text = refs.fields["crispasr.ref.generated_text"].contents()
    instruction = original["reference_recipe"]["prompt"].removeprefix("<|audio|>")
    expected_ids = arrays["llm_argmax"][len(arrays["prompt_ids"]):]
    f16 = fetch(prior["models"]["granite-speech-5.0-2b-f16"]["file"])
    if not shutil.which("ccache") or not shutil.which("ninja"):
        run(["sudo", "apt-get", "update"], "apt-update")
        run(["sudo", "apt-get", "install", "-y", "ccache", "ninja-build"], "apt-tools")
    build = scratch / "build"
    run(["cmake", "-S", ROOT, "-B", build, "-G", "Ninja", "-DCMAKE_BUILD_TYPE=Release",
         "-DCMAKE_C_COMPILER_LAUNCHER=ccache", "-DCMAKE_CXX_COMPILER_LAUNCHER=ccache", "-DGGML_NATIVE=OFF",
         "-DGGML_BLAS=OFF", "-DCRISPASR_BUILD_SERVER=OFF", "-DCRISPASR_BUILD_TESTS=ON"], "configure")
    run(["cmake", "--build", build, "--target", "granite5-stage-probe", "-j4"], "build")
    api = HfApi()
    for quant in ("q8_0", "q4_k"):
        baseline = fetch(prior["models"][f"granite-speech-5.0-2b-{quant}"]["file"])
        for label, names in [("base", []), ("head", ["output.weight"]), ("embedding", ["token_embd.weight"]),
                             ("both", ["output.weight", "token_embd.weight"])]:
            key = f"{quant}-{label}"
            model = baseline if not names else scratch / f"granite5-{key}.gguf"
            row = {"preserved_from_f16": names, "model_sha256": sha(baseline), "accepted": False}
            if names:
                row["tensor_audit"] = substitute(baseline, f16, model, names)
                row["model_sha256"] = sha(model)
                path = f"precision/{source}/{model.name}"
                commit = api.upload_file(path_or_fileobj=model, path_in_repo=path,
                    repo_id="cstr/crispasr-granite5-validation-public", commit_message="Checkpoint unaccepted precision diagnostic")
                row["checkpoint"] = {"repo": "cstr/crispasr-granite5-validation-public", "repo_type": "model",
                    "path": path, "revision": commit.oid, "sha256": sha(model), "bytes": model.stat().st_size}
            receipt["cases"][key] = row
            save()
            case = out / key
            run([build / "bin/granite5-stage-probe", model, reference, case, instruction, "cpu"], key)
            row["teacher"] = metrics(arrays["llm_prefill_logits"], np.fromfile(case / "llm_prefill_logits.bin", dtype="<f4"))
            row["pipeline"] = metrics(arrays["llm_prefill_logits"], np.fromfile(case / "pipeline_prefill_logits.bin", dtype="<f4"))
            row["tokens"] = token_comparison(expected_ids, np.fromfile(case / "generated_ids.bin", dtype="<i4"))
            actual_text = (case / "generated_text.txt").read_text()
            row["text_exact"] = actual_text.strip() == text.strip()
            row["content_equal"] = words(actual_text) == words(text)
            row["accepted"] = row["teacher"]["pass"] and row["pipeline"]["pass"] and row["tokens"]["exact"] and row["text_exact"]
            save()
            if names:
                model.unlink()
        baseline.unlink()
    receipt["complete"] = True
    receipt["scope"] = "Precision factorial only; full Session/CLI/CUDA acceptance remains separate"
    save()
    (out / "summary.md").write_text("Granite5 head/embedding precision diagnostic complete. See per-arm gates; no automatic promotion.\n")


if __name__ == "__main__":
    main()
