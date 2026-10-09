#!/usr/bin/env python3
"""Exercise real packaged C ABI plugin discovery from an unrelated directory.

Requires a dynamic-backend build with CPU plugins beside its shared library.
No model, compiler, mocked loader, or GPU is required.
"""
import argparse
import ctypes
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


def child(library, use_gpu):
    lib = ctypes.CDLL(str(library))
    lib.ggml_backend_dev_by_type.argtypes = [ctypes.c_int]
    lib.ggml_backend_dev_by_type.restype = ctypes.c_void_p
    assert not lib.ggml_backend_dev_by_type(0), "Control requires an initially empty CPU registry"

    class Params(ctypes.Structure):
        _fields_ = [("abi_version", ctypes.c_int), ("n_threads", ctypes.c_int),
                    ("use_gpu", ctypes.c_int), ("verbosity", ctypes.c_int),
                    ("flash_attn", ctypes.c_int), ("n_gpu_layers", ctypes.c_int),
                    ("reserved", ctypes.c_int * 6)]

    lib.crispasr_session_open_with_params.argtypes = [ctypes.c_char_p, ctypes.c_char_p,
                                                    ctypes.POINTER(Params)]
    lib.crispasr_session_open_with_params.restype = ctypes.c_void_p
    params = Params(2, 1, use_gpu, 0, 0, -1)
    # Deliberately fail model initialization, after the real session loader runs.
    result = lib.crispasr_session_open_with_params(b"absent.gguf", b"nemotron", ctypes.byref(params))
    assert not result, "Missing model must not open"
    device = lib.ggml_backend_dev_by_type(0)
    assert device, "C ABI failed to find CPU plugins adjacent to the loaded library"
    lib.ggml_backend_dev_init.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    lib.ggml_backend_dev_init.restype = ctypes.c_void_p
    lib.ggml_backend_free.argtypes = [ctypes.c_void_p]
    lib.ggml_backend_free.restype = None
    backend = lib.ggml_backend_dev_init(device, None)
    assert backend, "Discovered CPU device must initialize"
    lib.ggml_backend_free(backend)
    # Repeated opens preserve the same registered CPU device.
    assert not lib.crispasr_session_open_with_params(b"absent.gguf", b"nemotron", ctypes.byref(params))
    assert lib.ggml_backend_dev_by_type(0) == device
    print(json.dumps({"use_gpu": use_gpu, "cpu_plugin_initialized": True, "repeat_stable": True}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--child", type=int, choices=(0, 1))
    args = parser.parse_args()
    library = args.library.resolve(strict=True)
    if args.child is not None:
        child(library, args.child)
        return
    environment = dict(os.environ)
    for key in ("GGML_BACKEND_PATH", "LD_LIBRARY_PATH", "DYLD_LIBRARY_PATH"):
        environment.pop(key, None)
    with tempfile.TemporaryDirectory(prefix="crispasr-unrelated-cwd-") as cwd:
        for use_gpu in (0, 1):
            subprocess.run([sys.executable, str(Path(__file__).resolve()), "--library", str(library),
                            "--child", str(use_gpu)], cwd=cwd, env=environment, check=True)


if __name__ == "__main__":
    main()
