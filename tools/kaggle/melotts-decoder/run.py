#!/usr/bin/env python3
"""Complete the fixed Q4/F16 speech comparison using the retained CUDA application."""

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

APPLICATION_SOURCE = "9001ff91338e8a7417137ef4582f6805fcdd8e58"
CHECKPOINT_SHA = "5573e9c05f11e0b339660cc82eedf1d4a0ec11475ca0b700d51671b13b13b502"
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
        scope="Diagnostic only: fixed Q4/F16 speech comparison; BERT boundary evidence retained separately in version 3",
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
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "cryptography",
            "faster-whisper",
            "soundfile",
            "scipy",
        ],
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

        bert_f16 = data / "bert-f16.gguf"
        helper.guard_resources("download-f16-control")
        fetch(
            "https://huggingface.co/cstr/melotts-en-v2-GGUF/resolve/"
            + original["melo"]["revision"]
            + "/bert-base-uncased.gguf",
            bert_f16,
            "f7437929246c43d7300cbf6ff39b4715669a6053661117df99a456bdc5e8ecba",
        )
        precision_clips(helper, metadata[0], melo, bert, bert_f16, receipt, save)

        receipt["complete"] = True
    finally:
        receipt["private_evidence"] = seal_directory(
            PRIVATE, OUT / "private-evidence.bin", RECIPIENT
        )
        save()


def precision_clips(helper, metadata_path, melo, q4, f16, receipt, save):
    """Fixed original-precision control, never a replacement ASR or seed sweep."""
    import math
    import numpy as np
    import soundfile as sf
    from scipy.signal import resample_poly
    from faster_whisper import WhisperModel
    from huggingface_hub import snapshot_download

    sys.path.insert(0, str(ROOT / "tools/ci-heavy"))
    from openvoice_interfaces import DISCLOSURE, TEXTS, wer, cuda_vocoder_segments

    metadata = json.loads(metadata_path.read_text())
    reference = metadata_path.parent / "reference.wav"
    converter = metadata_path.parent / "openvoice2-tcc-f32sens.gguf"
    for name, path in [
        ("reference.wav", reference),
        ("openvoice2-tcc-f32sens.gguf", converter),
    ]:
        if sha(path) != metadata["artifacts"][name]["sha256"]:
            raise ValueError("Fixed precision input changed")
    # The actual CLI resolves its converter alongside the Melo model. The
    # version-3 diagnostic validated the mounted file but omitted this staging.
    helper.guard_resources("stage-validated-converter")
    adjacent = melo.parent / converter.name
    shutil.copy2(converter, adjacent)
    if sha(adjacent) != metadata["artifacts"]["openvoice2-tcc-f32sens.gguf"]["sha256"]:
        raise ValueError("Adjacent converter copy differs from the accepted input")
    receipt["converter_sha256"] = sha(adjacent)
    probes = []
    for label, bert in [("q4_k", q4), ("f16-control", f16)]:
        env = dict(
            os.environ,
            CRISPASR_OV2_GPU="1",
            CRISPASR_MELOTTS_BERT=str(bert),
            GGML_SCHED_DEBUG="1",
        )
        common = [
            WORK / "candidate/crispasr",
            "--backend",
            "melotts",
            "-m",
            melo,
            "--codec-model",
            bert,
            "-t",
            "2",
            "--seed",
            "42",
            "--speaker-identity",
            "unknown",
            "--gpu-backend",
            "cuda",
        ]
        cases = [("neutral", DISCLOSURE, [])] + [
            (str(i), text, ["--voice", reference, "--i-have-rights"])
            for i, text in enumerate(TEXTS)
        ]
        for case, text, flags in cases:
            name = label + "-" + case
            helper.guard_resources("precision-" + name)
            path = PRIVATE / (name + ".wav")
            with (PRIVATE / (name + ".log")).open("w") as output:
                subprocess.run(
                    list(
                        map(str, common + flags + ["--tts", text, "--tts-output", path])
                    ),
                    env=env,
                    stdout=output,
                    stderr=subprocess.STDOUT,
                    timeout=900,
                    check=True,
                )
            log = (PRIVATE / (name + ".log")).read_text()
            if case != "neutral" and cuda_vocoder_segments(log) != 1:
                raise RuntimeError("Fixed precision conversion did not use actual CUDA")
            expected = text if case == "neutral" else DISCLOSURE + " " + text
            probes.append((label, case, path, expected))
    helper.guard_resources("load-unchanged-tiny-oracle")
    revision = "0d3d19a32d3338f10357c0889762bd8d64bbdeba"
    decoder = WhisperModel(
        snapshot_download("Systran/faster-whisper-tiny.en", revision=revision),
        device="cpu",
        compute_type="int8",
        cpu_threads=2,
    )
    private = []
    for label, case, path, expected in probes:
        helper.guard_resources("decode-" + label + "-" + case)
        pcm, rate = sf.read(path, dtype="float32")
        divisor = math.gcd(rate, 16000)
        samples = np.ascontiguousarray(
            resample_poly(pcm, 16000 // divisor, rate // divisor), dtype=np.float32
        )
        segments, _ = decoder.transcribe(samples, language="en", beam_size=5)
        transcript = " ".join(segment.text.strip() for segment in segments)
        row = dict(
            profile=label,
            case=case,
            wer=wer(expected, transcript),
            rate=rate,
            samples=len(pcm),
            sha256=sha(path),
        )
        receipt.setdefault("precision_quality", []).append(row)
        private.append(dict(**row, transcript=transcript, expected=expected))
        (PRIVATE / "precision-quality.json").write_text(json.dumps(private, indent=2))
        save()
    receipt["precision_scope"] = (
        "Diagnostic only; no speaker-quality/full-application/default acceptance"
    )


if __name__ == "__main__":
    main()
