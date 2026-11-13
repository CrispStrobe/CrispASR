#!/usr/bin/env python3
"""Exercise public GGUF detectors with typed metadata in isolated processes."""

import argparse
import ctypes
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile


def gguf_string(value):
    value = value.encode("utf-8")
    return struct.pack("<Q", len(value)) + value


def fixture(path, metadata):
    raw = b"GGUF" + struct.pack("<IQQ", 3, 0, len(metadata))
    for name, kind, value in metadata:
        raw += gguf_string(name) + struct.pack("<I", kind) + value
    path.write_bytes(raw + bytes((-len(raw)) % 32))


def probe(library, export, path, expected_rc, expected_out):
    # A bad metadata type previously reached GGML_ASSERT. Keep each call in a
    # subprocess so an abort is reported as a failed case, without a core dump.
    if os.name == "posix":
        import resource
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    os.environ["GGML_NO_BACKTRACE"] = "1"
    detector = getattr(ctypes.CDLL(str(library)), export)
    detector.argtypes = (ctypes.c_char_p, ctypes.c_void_p, ctypes.c_int)
    detector.restype = ctypes.c_int
    output = ctypes.create_string_buffer(b"unchanged", 128)
    rc = detector(os.fsencode(path), output, len(output))
    assert rc == expected_rc, (export, path.name, rc, expected_rc)
    assert output.value.decode() == expected_out, (export, path.name, output.value)


def probe_buffer(library, export, path, expected, capacity):
    detector = getattr(ctypes.CDLL(str(library)), export)
    detector.argtypes = (ctypes.c_char_p, ctypes.c_void_p, ctypes.c_int)
    detector.restype = ctypes.c_int
    # Surround the advertised output region with sentinels to detect writes
    # before the buffer or beyond the caller's declared capacity.
    raw = ctypes.create_string_buffer(b"?" * 256, 256)
    output = ctypes.byref(raw, 16)
    before = raw.raw
    assert detector(None, output, capacity) == -1
    assert raw.raw == before
    assert detector(os.fsencode(path), None, capacity) == -1
    assert raw.raw == before
    rc = detector(os.fsencode(path), output, capacity)
    if capacity <= 0:
        wanted_rc, written = -1, b""
    elif len(expected) + 1 > capacity:
        wanted_rc, written = -5, b"\0"
    else:
        wanted_rc, written = len(expected), expected.encode() + b"\0"
    assert rc == wanted_rc, (export, path.name, capacity, rc, wanted_rc)
    assert raw.raw[:16] == before[:16], (export, path.name, capacity, raw.raw)
    assert raw.raw[16 + max(capacity, 0):] == before[16 + max(capacity, 0):], (
        export, path.name, capacity, raw.raw)
    assert raw.raw[16:16 + len(written)] == written, (
        export, path.name, capacity, raw.raw)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--probe", nargs=4, metavar=("EXPORT", "PATH", "RC", "OUT"))
    parser.add_argument("--probe-buffer", nargs=3, metavar=("EXPORT", "PATH", "OUT"))
    args = parser.parse_args()
    library = args.library.resolve(strict=True)
    if args.probe:
        export, path, rc, output = args.probe
        probe(library, export, Path(path), int(rc), output)
        return

    if args.probe_buffer:
        export, path, output = args.probe_buffer
        for capacity in sorted({-1, 0, 1, len(output), len(output) + 1, len(output) + 2}):
            probe_buffer(library, export, Path(path), output, capacity)
        return

    malformed = [(name, kind, struct.pack(fmt, value)) for name, kind, fmt, value in (
        ("uint8", 0, "<B", 1), ("int8", 1, "<b", -1),
        ("uint16", 2, "<H", 1), ("int16", 3, "<h", -1),
        ("uint32", 4, "<I", 1), ("int32", 5, "<i", -1),
        ("float32", 6, "<f", 1), ("bool", 7, "<?", True),
        ("uint64", 10, "<Q", 1), ("int64", 11, "<q", -1),
        ("float64", 12, "<d", 1),
    )]
    malformed += [
        ("string-array", 9, struct.pack("<IQ", 8, 1) + gguf_string("whisper")),
        ("integer-array", 9, struct.pack("<IQI", 4, 1, 1)),
        ("empty-array", 9, struct.pack("<IQ", 8, 0)),
    ]
    cases = [(name, [("general.architecture", kind, value)], -4, "", "")
             for name, kind, value in malformed]
    cases += [
        ("missing", [], -3, "", ""),
        ("unknown", [("general.architecture", 8, gguf_string("unknown-arch"))], 0, "", ""),
        ("empty", [("general.architecture", 8, gguf_string(""))], 0, "", ""),
        ("known", [("general.architecture", 8, gguf_string("granite_speech"))], 7, "granite", "granite"),
        ("alternates", [("general.architecture", 8, gguf_string("voxcpm2"))], None,
         "voxcpm2-tts", "voxcpm2-tts\nvoxcpm2-vae"),
    ]
    failures = []
    count = 0
    with tempfile.TemporaryDirectory(prefix="crispasr-detector-") as folder:
        for name, metadata, rc, single, plural in cases:
            path = Path(folder) / (name + ".gguf")
            fixture(path, metadata)
            for export, expected in (("crispasr_detect_backend_from_gguf", single),
                                     ("crispasr_detect_backends_from_gguf", plural)):
                result = subprocess.run(
                    [sys.executable, __file__, "--library", str(library), "--probe",
                     export, str(path), str(len(expected) if rc is None else rc), expected],
                    capture_output=True, text=True, timeout=30,
                )
                count += 1
                if result.returncode:
                    failures.append(f"{export}/{name}: exit {result.returncode}\n{result.stderr}")
                if name not in ("known", "alternates", "unknown"):
                    continue
                result = subprocess.run(
                    [sys.executable, __file__, "--library", str(library), "--probe-buffer",
                     export, str(path), expected],
                    capture_output=True, text=True, timeout=30,
                )
                count += len({-1, 0, 1, len(expected), len(expected) + 1, len(expected) + 2})
                if result.returncode:
                    failures.append(f"{export}/{name}/capacity: exit {result.returncode}\n{result.stderr}")
    assert not failures, "\n".join(failures)
    print(f"PASS: {count} isolated C ABI metadata and capacity checks")


if __name__ == "__main__":
    main()
