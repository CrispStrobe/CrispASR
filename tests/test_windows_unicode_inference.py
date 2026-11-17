#!/usr/bin/env python3
"""Windows production C ABI path equivalence; this is not a language-quality test."""
import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import subprocess
import urllib.error
import urllib.request
import wave

MODEL_REVISION = "d9e3ba65a6579796389ea89e5939509ed257f972"
MODEL_NAME = "parakeet-tdt-0.6b-ja-q8_0.gguf"
MODEL_SHA256 = "5a61e6c7d956c3c72a76fafcd798cac0c9ea66d0e29b3910cd04865a1e42cc17"
MODEL_SIZE = 673554880
MODEL_URL = ("https://huggingface.co/cstr/parakeet-tdt-0.6b-ja-GGUF/resolve/"
             f"{MODEL_REVISION}/{MODEL_NAME}")


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dll", required=True, type=Path)
    parser.add_argument("--wav", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if os.name != "nt":
        parser.error("requires a Windows runtime")
    import numpy as np
    from crispasr import Session

    dll = args.dll.resolve(strict=True)
    wav_path = args.wav.resolve(strict=True)
    # Both names must be on the same NTFS volume, regardless of workspace drive.
    profile = Path(os.environ["USERPROFILE"])
    model_dir = profile / "crispasr-592-ascii"
    model_dir.mkdir(exist_ok=True)
    model = model_dir / MODEL_NAME
    if not str(model).isascii():
        raise RuntimeError("the original model path must be ASCII")
    # No companion files: vocabulary and all weights are embedded in this GGUF.
    if not model.exists():
        try:
            response = urllib.request.urlopen(MODEL_URL, timeout=120)
        except urllib.error.HTTPError as error:
            # Public model: authenticate only after an actual rate-limit response.
            token = os.environ.get("HF_TOKEN")
            if error.code != 429 or not token:
                raise
            request = urllib.request.Request(MODEL_URL, headers={"Authorization": f"Bearer {token}"})
            response = urllib.request.urlopen(request, timeout=120)
        with response, model.open("wb") as out:
            while chunk := response.read(1024 * 1024):
                out.write(chunk)
    if model.stat().st_size != MODEL_SIZE or sha256(model) != MODEL_SHA256:
        raise RuntimeError("pinned model size/SHA256 mismatch")
    unicode_dir = profile / "\u6a21\u578b"
    unicode_dir.mkdir(exist_ok=True)
    linked_model = unicode_dir / MODEL_NAME
    os.link(model, linked_model)
    if not os.path.samefile(model, linked_model):
        raise RuntimeError("the two model names must be hard links to the same file")
    if str(linked_model).isascii():
        raise RuntimeError("Unicode path control must contain non-ASCII characters")
    with wave.open(str(wav_path), "rb") as wav:
        if (wav.getnchannels(), wav.getsampwidth(), wav.getframerate()) != (1, 2, 16000):
            raise RuntimeError("WAV must be mono PCM16 at 16 kHz")
        pcm = np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2").astype(np.float32) / 32768.0
    checkout_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    if checkout_sha != os.environ["CHECKOUT_SHA"]:
        raise RuntimeError("checkout does not match the workflow source SHA")
    report = {"pr_head_sha": os.environ["PR_HEAD_SHA"], "checkout_sha": checkout_sha,
              "use_gpu": False, "n_threads": 2,
              "model_revision": MODEL_REVISION, "model_sha256": MODEL_SHA256,
              "model_size": MODEL_SIZE, "wav": str(wav_path), "wav_sha256": sha256(wav_path),
              "sample_count": len(pcm), "scope": "path equivalence, not language quality", "passed": False, "results": []}
    def save_report():
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    save_report()
    print(json.dumps(report, ensure_ascii=False), flush=True)
    # Retain the cookie until both sessions and their DLL dependencies are closed.
    with os.add_dll_directory(str(dll.parent)):
        for label, path in [("ascii", model), ("unicode", linked_model)]:
            with Session(str(path), lib_path=str(dll), backend="parakeet",
                         use_gpu=False, n_threads=2) as session:
                # Confirm which DLL ctypes actually loaded, then hash those bytes.
                get_name = ctypes.windll.kernel32.GetModuleFileNameW
                get_name.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p, ctypes.c_uint32]
                get_name.restype = ctypes.c_uint32
                name = ctypes.create_unicode_buffer(32768)
                length = get_name(session._lib._handle, name, len(name))
                if not length or length >= len(name):
                    raise RuntimeError("cannot resolve loaded production DLL")
                loaded_dll = Path(name.value).resolve(strict=True)
                if not os.path.samefile(loaded_dll, dll):
                    raise RuntimeError("binding loaded an unexpected DLL")
                result = {"label": label, "path": str(path), "backend": session.backend,
                          "session_open_succeeded": True, "inference_succeeded": False,
                          "loaded_dll": str(loaded_dll), "dll_sha256": sha256(loaded_dll)}
                report["results"].append(result)
                save_report()
                segments = session.transcribe(pcm, sample_rate=16000)
                output = [{"text": segment.text, "start": segment.start, "end": segment.end}
                          for segment in segments]
                text = "".join(segment.text for segment in segments).strip()
                if not text:
                    raise RuntimeError(f"{label} path produced empty transcription")
                result.update({"inference_succeeded": True, "text": text, "segments": output})
                save_report()
                print(json.dumps(report["results"][-1], ensure_ascii=False), flush=True)
    first, second = report["results"]
    if first["segments"] != second["segments"] or first["dll_sha256"] != second["dll_sha256"]:
        raise RuntimeError("ASCII/Unicode inference outputs or loaded DLLs differ")
    report["passed"] = True
    save_report()
    print("PASS: production C ABI opened both paths and produced identical non-empty output", flush=True)


if __name__ == "__main__":
    main()
