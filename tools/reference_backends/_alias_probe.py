"""Known-answer probes for _safe_capture (used by tests/test_ref_capture_aliasing.py).

Lives inside reference_backends/ because the detector only records captures
made from dumper files.
"""
import numpy as np


def aliased_capture():
    """Capture a view, then change the tensor in place, as funasr's encoder did."""
    import torch
    x = torch.arange(6, dtype=torch.float32)
    cap = x.numpy()
    x *= 22.6
    return cap, x


def clean_capture():
    """Capture a tensor that nothing touches afterwards."""
    import torch
    x = torch.arange(6, dtype=torch.float32)
    return x.numpy(), x


def via_asarray():
    """np.asarray(tensor) goes through Tensor.__array__ -> numpy()."""
    import torch
    x = torch.ones(3)
    cap = np.asarray(x)
    x.add_(1.0)
    return cap, x
