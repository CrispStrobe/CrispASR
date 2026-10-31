#!/usr/bin/env python3
"""Exercise actual CLI and C ABI Haas dispatch without downloading weights.

The metadata-only GGUF deliberately fails Parakeet's predictor-layout check.
That specific diagnostic proves that dispatch reached the intended loader;
a null result alone would also pass for an unknown backend.
"""

from __future__ import annotations

import argparse
import ctypes
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import wave


ALIASES = ("parakeet", "haas", "haas-ja", "haas-asr", "haas_ja", "haas_asr")
LOADER_DIAGNOSTIC = "no supported transducer predictor or compatible CTC head found"


def write_fixture(directory: Path) -> tuple[Path, Path]:
    def gguf_string(value: str) -> bytes:
        raw = value.encode("utf-8")
        return struct.pack("<Q", len(raw)) + raw

    model = directory / "unhelpful-name.gguf"
    data = struct.pack("<4sIQQ", b"GGUF", 3, 0, 1)
    data += gguf_string("general.architecture") + struct.pack("<I", 8) + gguf_string("haas")
    data += b"\0" * (-len(data) % 32)
    model.write_bytes(data)
    audio = directory / "silence.wav"
    with wave.open(str(audio), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(b"\0\0" * 1600)
    return model, audio


def c_abi_child(library: Path, model: Path, backend: str) -> int:
    class OpenParams(ctypes.Structure):
        _fields_ = [
            ("abi_version", ctypes.c_int), ("n_threads", ctypes.c_int),
            ("use_gpu", ctypes.c_int), ("verbosity", ctypes.c_int),
            ("flash_attn", ctypes.c_int), ("n_gpu_layers", ctypes.c_int),
            ("reserved", ctypes.c_int * 6),
        ]

    lib = ctypes.CDLL(str(library))
    lib.crispasr_backend_caps_abi.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int32]
    lib.crispasr_backend_caps_abi.restype = ctypes.c_int
    expected_caps, actual_caps = ctypes.create_string_buffer(512), ctypes.create_string_buffer(512)
    canonical_rc = lib.crispasr_backend_caps_abi(b"parakeet", expected_caps, len(expected_caps))
    alias_rc = lib.crispasr_backend_caps_abi(backend.encode(), actual_caps, len(actual_caps))
    if backend in ALIASES:
        if canonical_rc <= 0 or alias_rc != canonical_rc or actual_caps.value != expected_caps.value:
            raise AssertionError(f"C ABI capability lookup disagrees for {backend}")
    elif alias_rc != -3:
        raise AssertionError("unknown backend should fail capability lookup")
    lib.crispasr_session_open_with_params.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.POINTER(OpenParams)]
    lib.crispasr_session_open_with_params.restype = ctypes.c_void_p
    params = OpenParams(2, 1, 0, 0, 0, 0)
    session = lib.crispasr_session_open_with_params(str(model).encode(), backend.encode(), ctypes.byref(params))
    if session:
        lib.crispasr_session_close.argtypes = [ctypes.c_void_p]
        lib.crispasr_session_close(session)
        raise AssertionError("metadata-only fixture unexpectedly loaded")
    return 0


def run_check(command: list[str], expect_loader: bool, cli: bool) -> None:
    result = subprocess.run(command, text=True, capture_output=True, timeout=60, check=False)
    output = result.stdout + result.stderr
    if cli and result.returncode == 0:
        raise AssertionError("CLI unexpectedly accepted a metadata-only model\n" + output)
    if not cli and result.returncode != 0:
        raise AssertionError("C ABI subprocess failed\n" + output)
    reached_loader = LOADER_DIAGNOSTIC in output
    if reached_loader != expect_loader:
        raise AssertionError(f"loader dispatch mismatch for {command[-1]!r}\n{output}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cli", type=Path)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--c-abi-child", nargs=2, metavar=("MODEL", "BACKEND"), help=argparse.SUPPRESS)
    args = parser.parse_args()
    library = args.library.resolve(strict=True)
    if args.c_abi_child:
        return c_abi_child(library, Path(args.c_abi_child[0]), args.c_abi_child[1])
    if args.cli is None:
        parser.error("--cli is required")
    cli = args.cli.resolve(strict=True)
    with tempfile.TemporaryDirectory(prefix="crispasr-haas-dispatch-") as temp:
        model, audio = write_fixture(Path(temp))
        for alias in (*ALIASES, "not-a-real-backend"):
            expected = alias in ALIASES
            run_check([str(cli), "-ng", "-t", "1", "-m", str(model), "-f", str(audio), "--backend", alias],
                      expect_loader=expected, cli=True)
            run_check([sys.executable, str(Path(__file__).resolve()), "--library", str(library),
                       "--c-abi-child", str(model), alias], expect_loader=expected, cli=False)
    print(json.dumps({"status": "passed", "aliases": list(ALIASES), "entrypoint_checks": 14, "capability_checks": 7,
                      "scope": "model-free dispatch; no model inference"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
