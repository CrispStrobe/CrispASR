"""Convert Quds-v4's public NeMo ONNX RNN-T export and prove it natively.

The 8 GB maintainer VPS swaps while importing torch plus the 456 MB ONNX
protobuf. Kaggle supplies enough quiet-box RAM; all builds remain bounded -j2.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path


WORK = Path("/tmp/crispasr-quds-v4")
REPO = WORK / "CrispASR"
SRC = WORK / "onnx"
OUT = WORK / "quds-v4-rnnt-q8_0.gguf"


def run(args: list[str], **kwargs) -> None:
    print("+", " ".join(map(str, args)), flush=True)
    subprocess.run(args, check=True, **kwargs)


def token() -> str:
    for path in (Path("/kaggle/input/crispasr-hf-token/hf_token.txt"),
                 Path("/kaggle/input/datasets/chr1s4/crispasr-hf-token/hf_token.txt")):
        if path.is_file():
            return path.read_text().strip()
    raise RuntimeError("HF token dataset is not mounted")


def main() -> None:
    WORK.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update({"HF_TOKEN": token(), "HF_HOME": str(WORK / "hf"),
                "TMPDIR": str(WORK / "tmp"), "PYTHONUNBUFFERED": "1",
                "OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2", "MKL_NUM_THREADS": "2"})
    Path(env["TMPDIR"]).mkdir(parents=True, exist_ok=True)
    run(["git", "clone", "--depth", "1", "--branch", "fix/387-quds-onnx",
         "https://github.com/CrispStrobe/CrispASR.git", str(REPO)])
    run(["git", "submodule", "update", "--init", "--recursive"], cwd=REPO)
    run(["python", "-m", "pip", "install", "-q", "gguf", "sentencepiece", "pyyaml",
         "onnx", "onnxruntime", "onnx-asr", "huggingface_hub", "datasets", "soundfile", "jiwer"], env=env)

    from huggingface_hub import HfApi, snapshot_download
    snapshot_download("hojreh/Quds-v4-onnx", local_dir=SRC,
                      allow_patterns=["encoder-model.onnx", "decoder_joint-model.onnx", "tokens.txt"],
                      token=env["HF_TOKEN"])
    run(["python", str(REPO / "models/convert-parakeet-to-gguf.py"),
         "--onnx-dir", str(SRC), "--output", str(OUT), "--quant", "q8_0"], env=env)

    run(["cmake", "-G", "Ninja", "-S", str(REPO), "-B", str(REPO / "build"),
         "-DCMAKE_BUILD_TYPE=Release", "-DCRISPASR_BUILD_TESTS=OFF", "-DCRISPASR_BUILD_SERVER=OFF"])
    run(["cmake", "--build", str(REPO / "build"), "--target", "crispasr-cli", "-j2"])
    # JFK is semantically out-of-domain, but a complete native encoder +
    # one-layer RNNT decode is a strong mechanical smoke: it catches every
    # missing tensor/hparam and the former unconditional second-LSTM crash.
    probe = subprocess.run([str(REPO / "build/bin/crispasr"), "--backend", "parakeet",
                            "--gpu-backend", "cpu", "-m", str(OUT),
                            str(REPO / "samples/jfk.wav")],
                           text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    print(probe.stdout, flush=True)
    if probe.returncode != 0 or "vocab=1024" not in probe.stdout or "n_layers=17" not in probe.stdout:
        raise RuntimeError("native Quds model-load/decode proof failed")

    # Accuracy proof on independently labelled Persian speech, plus direct
    # comparison with the publisher's ONNX runtime on the identical samples.
    import jiwer
    import numpy as np
    import onnx_asr
    import soundfile as sf
    from datasets import load_dataset
    reference_model = onnx_asr.load_model(str(SRC))
    dataset = load_dataset("google/fleurs", "fa_ir", split="test", streaming=True)
    for index, sample in enumerate(dataset.take(3)):
        audio = sample["audio"]
        wav = WORK / f"fa-{index}.wav"
        sf.write(wav, np.asarray(audio["array"], dtype=np.float32), audio["sampling_rate"])
        expected = sample["transcription"]
        onnx_text = reference_model.recognize(str(wav))
        prefix = WORK / f"native-{index}"
        run([str(REPO / "build/bin/crispasr"), "--backend", "parakeet", "--gpu-backend", "cpu",
             "-m", str(OUT), "-otxt", "-of", str(prefix), str(wav)])
        native_text = prefix.with_suffix(".txt").read_text()
        wer_onnx = jiwer.wer(expected, onnx_text)
        wer_native = jiwer.wer(expected, native_text)
        agreement = jiwer.wer(onnx_text, native_text)
        print(f"QUDS_PARITY sample={index} onnx_wer={wer_onnx:.3f} "
              f"native_wer={wer_native:.3f} agreement_wer={agreement:.3f}\n"
              f"  expected={expected!r}\n  onnx={onnx_text!r}\n  native={native_text!r}", flush=True)
        if native_wer > max(0.35, wer_onnx + 0.10) or agreement > 0.20:
            raise RuntimeError("native Quds output does not match ONNX/reference quality")

    api = HfApi(token=env["HF_TOKEN"])
    api.create_repo("cstr/quds-v4-rnnt-GGUF", repo_type="model", exist_ok=True)
    api.upload_file(path_or_fileobj=OUT, path_in_repo=OUT.name,
                    repo_id="cstr/quds-v4-rnnt-GGUF", repo_type="model",
                    commit_message="Add Quds-v4 Persian RNN-T Q8_0 converted from ONNX")
    print(f"QUDS_V4_CONVERSION_COMPLETE bytes={OUT.stat().st_size}", flush=True)


if __name__ == "__main__":
    main()
