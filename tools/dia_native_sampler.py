"""Diagnostic original-Dia adapter to the compiled production sampler.

This is a separately labelled control, never the canonical Torch reference.
The original neural/CFG/mask/delay/DAC code remains unchanged. Each row consumes
one actual native sample in codebook order, preserving the native RNG state.
"""
from contextlib import contextmanager
import ctypes
import re

import numpy as np


def native_sample_rows(log, expected_steps):
    """Read complete pre-override draws; reject missing/reordered/truncated rows."""
    matches = re.findall(r'^DIA_TOK step (\d+):([^\n]*)$', log, re.MULTILINE)
    if len(matches) != expected_steps or expected_steps <= 0:
        raise ValueError('Incomplete native sample trace')
    rows = []
    for expected, (step, values) in enumerate(matches):
        if int(step) != expected:
            raise ValueError('Nonsequential native sample trace')
        tokens = values.split()
        if len(tokens) != 9 or any(not token.isdecimal() or not 0 <= int(token) <= 1024 for token in tokens):
            raise ValueError('Invalid native codebook samples')
        rows.append([int(token) for token in tokens])
    return np.asarray(rows, dtype=np.int64)


class NativeSampler:
    def __init__(self, library, seed=42):
        if not isinstance(seed, int) or not 0 < seed <= 0xffffffff:
            raise ValueError('Diagnostic seed must be a nonzero uint32')
        self.library = ctypes.CDLL(str(library))
        self.library.dia_sampler_create.argtypes = [ctypes.c_uint32]
        self.library.dia_sampler_create.restype = ctypes.c_void_p
        self.library.dia_sampler_free.argtypes = [ctypes.c_void_p]
        self.library.dia_sampler_free.restype = None
        self.library.dia_sampler_rows.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float),
            ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_float, ctypes.c_int,
            ctypes.POINTER(ctypes.c_int64)]
        self.library.dia_sampler_rows.restype = ctypes.c_int
        self.handle = self.library.dia_sampler_create(seed)
        if not self.handle:
            raise MemoryError('Cannot allocate diagnostic native sampler')
        self.samples = []

    def __call__(self, logits_BCxV, temperature, top_p, cfg_filter_top_k=None):
        import torch
        if not self.handle:
            raise RuntimeError('Native sampler is closed')
        if logits_BCxV.device.type != 'cpu' or logits_BCxV.ndim != 2:
            raise ValueError('Native sampler control requires CPU codebook/vocabulary rows')
        logits = logits_BCxV.detach().float().contiguous().numpy()
        rows, vocab = logits.shape
        output = np.empty(rows, dtype=np.int64)
        status = self.library.dia_sampler_rows(self.handle,
            logits.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), rows, vocab,
            temperature, top_p, 0 if cfg_filter_top_k is None else cfg_filter_top_k,
            output.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)))
        if status:
            raise ValueError(f'Native sampler rejected inputs (status={status})')
        # Original generate mutates EOS/PAD entries after sampling. Preserve
        # the actual pre-override draws separately from its returned Tensor.
        self.samples.append(output.copy())
        return torch.from_numpy(output)

    def close(self):
        if self.handle:
            self.library.dia_sampler_free(self.handle)
            self.handle = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


@contextmanager
def install_native_sampler(module, library, seed=42):
    original = module._sample_next_token
    with NativeSampler(library, seed) as sampler:
        module._sample_next_token = sampler
        try:
            yield sampler
        finally:
            module._sample_next_token = original
