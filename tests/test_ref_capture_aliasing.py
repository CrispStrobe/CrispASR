"""reference_backends captures must own their storage, and the detector must
name captures whose source tensor changed afterwards (funasr mel_features was
written 22.6x scaled because a .numpy() view saw a later in-place `*=`)."""
import io
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from reference_backends import _alias_probe, _safe_capture  # noqa: E402


def setup_function(_):
    _safe_capture.reset()


def test_capture_survives_in_place_change_and_is_reported():
    cap, x = _alias_probe.aliased_capture()
    assert np.array_equal(cap, np.arange(6, dtype=np.float32))  # not the scaled values
    assert x[1].item() == pytest.approx(22.6)
    buf = io.StringIO()
    hits = _safe_capture.report(buf)
    assert len(hits) == 1 and hits[0][0].endswith("_alias_probe.py:13")
    assert "[ALIASING]" in buf.getvalue()


def test_untouched_capture_is_not_reported():
    cap, x = _alias_probe.clean_capture()
    assert cap.flags["OWNDATA"]
    buf = io.StringIO()
    assert _safe_capture.report(buf) == []
    assert "none of 1 capture(s)" in buf.getvalue()


def test_asarray_path_is_covered():
    cap, x = _alias_probe.via_asarray()
    assert np.array_equal(cap, np.ones(3, dtype=np.float32))
    assert len(_safe_capture.report(io.StringIO())) == 1


def test_calls_outside_dumpers_keep_stock_numpy():
    x = torch.zeros(2)
    arr = x.numpy()  # this test file is not a dumper: a view, as stock torch
    x += 1
    assert arr[0] == 1.0


def test_own_breaks_the_float32_alias_a_hook_would_keep():
    out = torch.ones(4)
    stored = out.detach().cpu().float()  # what 32 hooks stored: same storage
    kept = _safe_capture.own(out.detach().cpu().float())
    out.mul_(3.0)  # a later in-place op on the module output
    assert stored[0].item() == 3.0  # the old capture was rewritten
    assert kept[0].item() == 1.0


def test_own_keeps_containers_and_model_outputs():
    t = torch.ones(2)
    tup = _safe_capture.own((t, torch.tensor([5])))
    assert isinstance(tup, tuple) and tup[0].data_ptr() != t.data_ptr()
    transformers = pytest.importorskip("transformers")
    mo = transformers.modeling_outputs.BaseModelOutput(last_hidden_state=t)
    c = _safe_capture.own(mo)
    assert type(c) is type(mo) and c.last_hidden_state.data_ptr() != t.data_ptr()
