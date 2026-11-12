#!/usr/bin/env python3
"""Probe the real shared-library C ABI with an unsupported RE-USE GGUF."""
import argparse
import ctypes
import hashlib
import json
from pathlib import Path
import tempfile


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    import gguf
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--library', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    lib = ctypes.CDLL(str(args.library.resolve()))
    lib.crispasr_detect_backend_from_gguf.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int]
    lib.crispasr_detect_backend_from_gguf.restype = ctypes.c_int
    lib.crispasr_session_open.argtypes = [ctypes.c_char_p, ctypes.c_int]
    lib.crispasr_session_open.restype = ctypes.c_void_p
    for _ in range(3):
        out = ctypes.create_string_buffer(b'nonempty', 64)
        rc = lib.crispasr_detect_backend_from_gguf(str(args.model.resolve()).encode(), out, 64)
        if rc != 0 or out.value or lib.crispasr_session_open(str(args.model.resolve()).encode(), 1):
            raise SystemExit('RE-USE unexpectedly detected or opened as a product backend')
    with tempfile.TemporaryDirectory(prefix='reuse-abi-') as directory:
        path = Path(directory) / 'whisper-control.gguf'
        writer = gguf.GGUFWriter(str(path), arch='whisper')
        writer.add_name('control')
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.close()
        out = ctypes.create_string_buffer(64)
        if lib.crispasr_detect_backend_from_gguf(str(path).encode(), out, 64) != 7 or out.value != b'whisper':
            raise SystemExit('Whisper positive detection control failed')
    if lib.crispasr_detect_backend_from_gguf(None, ctypes.create_string_buffer(64), 64) != -1:
        raise SystemExit('null argument rejection control failed')
    report = {'library_sha256': sha256(args.library), 'model_sha256': sha256(args.model),
              'reuse_detection_and_open_rejections': 3, 'whisper_detection_control': 'pass',
              'null_argument_control': 'pass'}
    if args.output:
        args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, sort_keys=True))


if __name__ == '__main__':
    main()
