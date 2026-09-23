"""Own-storage captures for every reference dumper, plus an aliasing detector.

`tensor.numpy()` on a CPU tensor returns a VIEW of the tensor's storage. When
a dumper captures a stage that way and the model (or the dumper) later changes
that tensor in place — `xs *= sqrt(d)`, `x += residual`, `relu_`,
`masked_fill_` — the reference written to disk is not what the stage produced.
funasr's mel_features reference was corrupted exactly like that (22.6x
scaled, still cos 1.0, so the cosine gate passed it).

install() patches torch.Tensor.numpy (which np.asarray / __array__ go through
too) so that every capture is an owned copy. For every capture it keeps a weak
reference to the source tensor, a snapshot of its values, and the dumper line
that asked for it. report() re-reads each source tensor that is still alive
and lists those whose values changed after capture: the references that the
old view-returning `.numpy()` would have corrupted.

Only captures made from files under tools/reference_backends/ (and
tools/dump_reference.py) are recorded; calls from library code (torch,
transformers, numpy internals) keep the stock behaviour.
"""
from __future__ import annotations

import os
import sys
import traceback
import weakref

_HERE = os.path.dirname(os.path.abspath(__file__))
_TOOLS = os.path.dirname(_HERE)
_installed = False
_records = []  # (weakref to tensor, snapshot ndarray, "file:line")


def _caller_in_dumper():
    """file:line of the nearest frame in a dumper, or None."""
    f = sys._getframe(2)
    while f is not None:
        fn = os.path.abspath(f.f_code.co_filename)
        if fn.startswith(_HERE) and not fn.endswith("_safe_capture.py"):
            return f"{os.path.relpath(fn, _TOOLS)}:{f.f_lineno}"
        if fn == os.path.join(_TOOLS, "dump_reference.py"):
            return f"dump_reference.py:{f.f_lineno}"
        f = f.f_back
    return None


def install():
    """Idempotent. A no-op when torch is not importable."""
    global _installed
    if _installed:
        return
    try:
        import torch
    except Exception:
        return
    orig = torch.Tensor.numpy

    def numpy(self, *args, **kwargs):
        arr = orig(self, *args, **kwargs)
        where = _caller_in_dumper()
        if where is None:
            return arr
        snap = arr.copy()
        try:
            _records.append((weakref.ref(self), snap.copy(), where))
        except TypeError:
            pass
        return snap

    torch.Tensor.numpy = numpy
    torch.Tensor._crispasr_orig_numpy = orig
    _installed = True


def report(stream=None):
    """Print (and return) the captures whose source tensor changed after capture."""
    import numpy as np

    stream = stream or sys.stderr
    try:
        import torch
        orig = getattr(torch.Tensor, "_crispasr_orig_numpy", torch.Tensor.numpy)
    except Exception:
        return []
    hits = {}
    for ref, snap, where in _records:
        t = ref()
        if t is None:
            continue
        try:
            now = orig(t.detach()) if t.requires_grad else orig(t)
        except Exception:
            continue
        if now.shape != snap.shape:
            continue
        same = np.array_equal(now, snap, equal_nan=True) if now.dtype.kind == "f" else np.array_equal(now, snap)
        if not same:
            d = float(np.nanmax(np.abs(now.astype(np.float64) - snap.astype(np.float64)))) if now.size else 0.0
            hits.setdefault(where, [0, 0.0])
            hits[where][0] += 1
            hits[where][1] = max(hits[where][1], d)
    for where, (n, d) in sorted(hits.items()):
        print(f"[ALIASING] {where}: {n} capture(s) changed after capture (max |d| {d:.3g}); "
              f"a view-returning .numpy() would have written the changed values", file=stream)
    if not hits:
        print(f"[aliasing] none of {len(_records)} capture(s) changed after capture", file=stream)
    return sorted(hits.items())


def own(x):
    """Recursively clone tensors (in tuples / lists / dicts) so a hook's capture
    owns its storage: `t.detach().cpu().float()` is the SAME storage as the
    module output for a float32 CPU tensor, and a later in-place op on that
    output would rewrite the stored capture."""
    try:
        import torch
    except Exception:
        return x
    if isinstance(x, torch.Tensor):
        return x.clone()
    if isinstance(x, tuple):
        return type(x)(own(v) for v in x) if not hasattr(x, "_fields") else type(x)(*(own(v) for v in x))
    if isinstance(x, list):
        return [own(v) for v in x]
    if hasattr(x, "to_tuple") and isinstance(x, dict):  # HF ModelOutput: same type, cloned fields
        try:
            return type(x)(**{k: own(v) for k, v in x.items()})
        except Exception:
            return x
    if isinstance(x, dict):
        return {k: own(v) for k, v in x.items()}
    return x


def reset():
    _records.clear()


# Active on import (idempotent), so standalone dumpers that import this module
# for own() get owned .numpy() captures too.
install()
