#!/usr/bin/env python3
"""Compare the original decoder on actual CUDA without rebuilding the application."""

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tarfile
import urllib.request

APPLICATION_SOURCE = "01a2887632d343154f2ba447a9a2968c7accb567"
CHECKPOINT_SHA = "9db22c7d50d70bee047746b3c33647d7df28cf066dee9c08164f0c0031dbba9b"
CONDITIONING_SHA = "46badec89720409414dad5c61f3e5cc1c34928a3e0b8391a1cf4f92e4d90dec8"
HELPER_SOURCE = "d84007612167105cf24c7af80014d1f0ea1bd8a3"
HELPER_SHA = "b1fed1e4518ceeb517c749be67ce3eeee9d3243153b5f50043f6edf9c4584f20"
RECIPIENT = "af2f46629cfa40da8c9fffdb9923877d1c4eb312d4ce3b4e36a42aa673f8ba47"
WORK = Path("/kaggle/temp/melotts-decoder")
OUT = Path("/kaggle/working")
ROOT = WORK / "source"
PRIVATE = WORK / "private"


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def fetch(url, path, expected):
    with (
        urllib.request.urlopen(url, timeout=120) as response,
        Path(path).open("wb") as output,
    ):
        shutil.copyfileobj(response, output)
    if sha(path) != expected:
        raise ValueError("Immutable diagnostic input changed")


def main():
    source = os.environ["CRISPASR_VALIDATION_SOURCE"]
    if not re.fullmatch("[0-9a-f]{40}", source):
        raise ValueError("Immutable controller source required")
    WORK.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    PRIVATE.mkdir(exist_ok=True)
    helper_path = WORK / "resources.py"
    fetch(
        "https://raw.githubusercontent.com/CrispStrobe/CrispASR/"
        + HELPER_SOURCE
        + "/tools/kaggle/openvoice-bake/openvoice_bake.py",
        helper_path,
        HELPER_SHA,
    )
    spec = importlib.util.spec_from_file_location("resources", helper_path)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    helper.WORK = WORK
    helper.OUT = OUT
    receipt = dict(
        schema=1,
        complete=False,
        application_acceptance=False,
        source=source,
        application_source=APPLICATION_SOURCE,
        application_checkpoint_sha256=CHECKPOINT_SHA,
        scope="Diagnostic only: original decoder, identical converted weights and captured native inputs",
        worker_sha256=sha(__file__),
        resource_checks=[],
        phases={},
    )

    def save():
        (OUT / "decoder-diagnostic.json").write_text(json.dumps(receipt, indent=2))

    helper.RECEIPT = receipt
    helper.save = save
    helper.guard_resources("decoder-bootstrap")
    if os.environ.get("CRISPASR_DECODER_READY") != "1":
        for cmd in [
            ["git", "init", ROOT],
            [
                "git",
                "-C",
                ROOT,
                "remote",
                "add",
                "origin",
                "https://github.com/CrispStrobe/CrispASR.git",
            ],
            ["git", "-C", ROOT, "fetch", "--depth=1", "origin", source],
            ["git", "-C", ROOT, "checkout", "FETCH_HEAD"],
        ]:
            subprocess.run(list(map(str, cmd)), check=True)
        url = os.environ.pop("CRISPASR_CHECKPOINT_URL", "")
        if not url.startswith("https://"):
            raise ValueError("Scoped checkpoint output URL required")
        checkpoint = WORK / "retained-application.tar.gz"
        fetch(url, checkpoint, CHECKPOINT_SHA)
        with tarfile.open(checkpoint) as archive:
            archive.extractall(WORK, filter="data")
        provenance = json.loads((WORK / "candidate/provenance.json").read_text())
        if provenance["source"] != APPLICATION_SOURCE:
            raise ValueError("Wrong retained application source")
        providers = sorted(
            {
                str(p)
                for base in [Path("/usr/local/lib"), Path("/opt/conda/lib")]
                for p in base.glob("python*/**/nvidia/*/lib")
                if p.is_dir()
            }
        )
        providers += [
            str(p)
            for p in [Path("/usr/local/cuda/lib64"), Path("/usr/local/nvidia/lib64")]
            if p.is_dir()
        ]
        env = dict(
            os.environ,
            CRISPASR_DECODER_READY="1",
            OMP_NUM_THREADS="2",
            OPENBLAS_NUM_THREADS="1",
            LD_LIBRARY_PATH=os.pathsep.join(
                [
                    str(WORK / "candidate"),
                    *providers,
                    os.environ.get("LD_LIBRARY_PATH", ""),
                ]
            ),
        )
        os.execve(sys.executable, [sys.executable, str(Path(__file__).resolve())], env)
    helper.run(
        [sys.executable, "-m", "pip", "install", "-q", "gguf==0.17.1", "cryptography", "huggingface_hub", "tokenizers"],
        "dependencies",
    )
    sys.path.insert(0, str(ROOT / "tools/validation"))
    from private_evidence import seal_directory

    try:
        library = WORK / "candidate/libcrispasr.so"
        helper.run(["ldd", "-r", library], "retained-dependencies")
        log = (OUT / "retained-dependencies.log").read_text()
        if "not found" in log or "undefined symbol:" in log:
            raise RuntimeError("Retained library dependencies unresolved")
        loaded = re.findall(
            r"^\s*(lib(?:ggml\S*|crispasr\S*)\.so\S*) => (\S+)", log, re.M
        )
        if len(loaded) != 4 or any(
            Path(p).resolve().parent != (WORK / "candidate").resolve()
            for _, p in loaded
        ):
            raise RuntimeError("Retained ggml providers displaced")
        receipt["library_sha256"] = sha(library)
        metadata = [
            p
            for p in Path("/kaggle/input").rglob("validation.json")
            if sha(p) == CONDITIONING_SHA
        ]
        if len(metadata) != 1:
            raise ValueError("Accepted conditioning metadata required")
        original = json.loads(metadata[0].read_text())
        data = WORK / "inputs"
        data.mkdir(exist_ok=True)
        melo = data / "melo.gguf"
        bert = data / "bert.gguf"
        for path, name, digest in [
            (melo, "melotts-en-v2-f16.gguf", original["melo"]["model_sha256"]),
            (bert, "bert-base-uncased-q4k.gguf", original["melo"]["bert_sha256"]),
        ]:
            helper.guard_resources("download-" + path.name)
            fetch(
                "https://huggingface.co/cstr/melotts-en-v2-GGUF/resolve/"
                + original["melo"]["revision"]
                + "/"
                + name,
                path,
                digest,
            )

        def run(command, label):
            helper.guard_resources(label)
            with (PRIVATE / (label + ".log")).open("w") as output:
                result = subprocess.run(
                    list(map(str, command)),
                    stdout=output,
                    stderr=subprocess.STDOUT,
                    timeout=1800,
                )
            receipt["phases"][label] = dict(returncode=result.returncode)
            save()
            if result.returncode:
                raise RuntimeError(label + " failed; inspect encrypted evidence")

        captures = PRIVATE / "native-inputs"
        run(
            [
                sys.executable,
                ROOT / "tools/validation/melotts_inputs.py",
                "--library",
                library,
                "--model",
                melo,
                "--bert",
                bert,
                "--output",
                captures,
                "--device",
                "cuda",
            ],
            "native-captures",
        )
        backends = re.findall(
            r"^melotts: final activation backend (\S+)",
            (PRIVATE / "native-captures.log").read_text(),
            re.M,
        )
        if len(backends) != 3 or not all(name.startswith("CUDA") for name in backends):
            raise RuntimeError("Native captures did not use CUDA")
        receipt["native_activation_backends"] = backends
        # Import failures/crashes remain isolated; finally seals every existing capture.
        run(
            [
                sys.executable,
                ROOT / "tools/validation/melotts_decoder.py",
                "--model",
                melo,
                "--captures",
                captures,
                "--output",
                PRIVATE / "original-decoder",
            ],
            "original-generator",
        )
        receipt["decoder_controls"] = json.loads(
            (PRIVATE / "original-decoder/decoder-controls.json").read_text()
        )
        receipt["complete"] = True
    finally:
        controls = PRIVATE / "original-decoder/decoder-controls.json"
        if controls.exists():
            receipt["decoder_controls"] = json.loads(controls.read_text())
        receipt["private_evidence"] = seal_directory(
            PRIVATE, OUT / "private-evidence.bin", RECIPIENT
        )
        save()


if __name__ == "__main__":
    main()
