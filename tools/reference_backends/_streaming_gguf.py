"""Atomic GGUF reference output with bounded, synchronous activation copies.

The tensor schema is declared before inference. Hooks may arrive in any order;
each writes directly to its final aligned GGUF extent. There is no second data
archive and no list of retained activations. Only a complete, finite archive is
renamed to the requested output path.
"""

from __future__ import annotations

import math
import os
import tempfile
from pathlib import Path

import numpy as np


class StreamingGGUFArchive:
    """Write F32 capture tensors in blocks no larger than ``block_bytes``.

    Schema shapes use NumPy/C order, matching ``write_gguf_archive``. The GGUF
    library writes the versioned header and tensor table; this class writes
    the F32 extents at the offsets declared by that table. NumPy arrays and
    PyTorch tensors are accepted, including noncontiguous or device tensors.
    """

    def __init__(self, output_path, shapes, metadata, *, block_bytes=4 * 1024 * 1024):
        import gguf

        if block_bytes < 4:
            raise ValueError("block_bytes must allow at least one F32 value")
        self.output_path = Path(output_path)
        self.shapes = {name: tuple(int(dim) for dim in shape) for name, shape in sorted(shapes.items())}
        if not self.shapes or any(not shape or any(dim <= 0 for dim in shape)
                                  for shape in self.shapes.values()):
            raise ValueError("reference schema must contain nonempty, positive tensor shapes")
        self.block_elements = block_bytes // 4
        self.written = set()
        self.max_block_bytes = 0
        self.writer = None
        self.file = None
        self.temp_path = None
        self.offsets = {}
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix=f".{self.output_path.name}.", suffix=".partial",
                                         dir=self.output_path.parent)
        os.close(fd)
        self.temp_path = Path(temporary)
        try:
            writer = self.writer = gguf.GGUFWriter(str(self.temp_path), arch="crispasr.reference")
            writer.add_description("CrispASR reference activation dump")
            for key, value in metadata.items():
                key = f"crispasr.ref.{key}"
                if isinstance(value, bool):
                    writer.add_bool(key, value)
                elif isinstance(value, int):
                    writer.add_int32(key, value)
                elif isinstance(value, float):
                    writer.add_float32(key, value)
                elif isinstance(value, str):
                    writer.add_string(key, value)
            for name, shape in self.shapes.items():
                writer.add_tensor_info(name, shape, np.float32, math.prod(shape) * 4)
            writer.write_header_to_file()
            writer.write_kv_data_to_file()
            writer.write_ti_data_to_file()
            if writer.fout is None or len(writer.fout) != 1:
                raise RuntimeError("streaming reference requires a single GGUF archive")
            self.file = writer.fout[0]
            writer.write_padding(self.file, self.file.tell())
            offset = self.file.tell()
            for name, shape in self.shapes.items():
                self.offsets[name] = offset
                offset += writer.ggml_pad(math.prod(shape) * 4, writer.data_alignment)
            self.end_offset = offset
        except BaseException:
            self.abort()
            raise

    def _blocks(self, value):
        """Slice before conversion, so a strided tensor cannot force a huge copy."""
        elements = math.prod(value.shape)
        if elements <= self.block_elements:
            if not isinstance(value, np.ndarray):
                value = value.detach().float().cpu().numpy()
            yield np.ascontiguousarray(value, dtype="<f4")
        elif math.prod(value.shape[1:]) <= self.block_elements:
            step = max(1, self.block_elements // math.prod(value.shape[1:]))
            for start in range(0, value.shape[0], step):
                yield from self._blocks(value[start:start + step])
        else:
            for index in range(value.shape[0]):
                yield from self._blocks(value[index])

    def capture(self, name, value):
        if self.file is None:
            raise RuntimeError("reference archive is closed")
        if name not in self.shapes:
            raise ValueError(f"unexpected reference stage {name!r}")
        if name in self.written:
            raise ValueError(f"duplicate reference stage {name!r}")
        if tuple(value.shape) != self.shapes[name]:
            raise ValueError(f"{name}: expected shape {self.shapes[name]}, got {tuple(value.shape)}")
        self.file.seek(self.offsets[name])
        count = 0
        for block in self._blocks(value):
            if not np.isfinite(block).all():
                raise ValueError(f"{name}: reference stage contains nonfinite values")
            self.max_block_bytes = max(self.max_block_bytes, block.nbytes)
            self.file.write(memoryview(block).cast("B"))
            count += block.nbytes
        if count != math.prod(self.shapes[name]) * 4:
            raise RuntimeError(f"{name}: incomplete reference tensor write")
        self.writer.write_padding(self.file, count)
        self.written.add(name)

    def finish(self):
        missing = set(self.shapes) - self.written
        if missing:
            raise RuntimeError(f"reference archive is missing stages {sorted(missing)}")
        if self.file is None:
            raise RuntimeError("reference archive is closed")
        self.file.truncate(self.end_offset)
        self.file.flush()
        os.fsync(self.file.fileno())
        self.writer.close()
        self.file = None
        os.replace(self.temp_path, self.output_path)
        self.temp_path = None

    def abort(self):
        if self.writer is not None:
            self.writer.close()
        self.file = None
        if self.temp_path is not None:
            self.temp_path.unlink(missing_ok=True)
            self.temp_path = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, _exc, _traceback):
        if exc_type is None:
            try:
                self.finish()
            except BaseException:
                self.abort()
                raise
        else:
            self.abort()
        return False
