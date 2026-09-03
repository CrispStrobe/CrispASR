"""GGUFWriter.add_tensor(raw_dtype=...) does NOT convert — the caller must.

This is the bug that shipped in `cstr/fastpitch-en-GGUF`'s f16 build:
`models/convert-fastpitch-to-gguf.py` chose F16 for 2-D weights and then handed
`add_tensor` a **float32** array with `raw_dtype=F16`. add_tensor takes the
array's bytes and labels them with the type you named, so the file contained f32
bytes tagged F16. A reader then took the first half of them and reinterpreted
each 4-byte float as two 2-byte ones: half the weights gone, the survivors
garbage — 943,872 NaNs across 138 decoder tensors, values to 6.5e4 where the
real range is +-0.33.

Nobody noticed because the model card recommends q8_0, and because Q8_0 does
NOT have the problem: gguf-py quantizes for a quant raw_dtype, so that path was
correct by accident rather than by symmetry. A converter author reading only the
q8_0 path would conclude add_tensor converts. It does not.

Model-free and dependency-light: writes a few small tensors to a temp dir.
"""

import numpy as np
import pytest

gguf = pytest.importorskip("gguf")
from gguf.constants import GGMLQuantizationType as Q  # noqa: E402


def _roundtrip(tmp_path, arr, qt, name="w"):
    path = str(tmp_path / f"{name}.gguf")
    w = gguf.GGUFWriter(path, "test", use_temp_file=False)
    w.add_name("test")
    w.add_tensor(name, arr, raw_dtype=qt)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    t = gguf.GGUFReader(path).tensors[0]
    return np.asarray(t.data)


# Weight-shaped values: the real fastpitch decoder weights live in +-0.33.
SRC = ((np.arange(256, dtype=np.float32) / 256.0) - 0.5) * 0.6


def test_f32_array_declared_f16_is_corrupted(tmp_path):
    """The bug, pinned. This must stay a demonstration, not a regression."""
    out = _roundtrip(tmp_path, SRC.reshape(16, 16), Q.F16).astype(np.float32)
    # Values explode: f32 bytes reinterpreted as pairs of f16.
    assert np.nanmax(np.abs(out)) > 100, (
        "add_tensor appears to convert now — if gguf-py changed, this test's "
        "premise is gone and the converters can be simplified"
    )


def test_casting_first_round_trips(tmp_path):
    """The fix: cast to the declared dtype before handing it over."""
    out = _roundtrip(tmp_path, SRC.reshape(16, 16).astype(np.float16), Q.F16)
    assert out.dtype == np.float16
    err = np.abs(out.astype(np.float32) - SRC.reshape(16, 16)).max()
    assert err < 1e-3, f"f16 rounding should be tiny, got {err}"
    assert not np.isnan(out.astype(np.float32)).any()


def test_q8_0_is_corrupted_the_same_way(tmp_path):
    """Q8_0 does NOT quantize for you either — I assumed it did, and was wrong.

    The published q8_0/q4_k files are fine, which makes it very easy to conclude
    the quant path converts. It does not: those files were produced by
    crispasr-quantize from a good source, not by this code path. add_tensor
    writes the right BYTE COUNT from the wrong bytes, so the file looks
    structurally valid and dequantizes to noise.
    """
    # Row size must be a multiple of Q8_0's 32-element block, hence 8x32.
    out = _roundtrip(tmp_path, SRC.reshape(8, 32), Q.Q8_0)
    deq = gguf.quants.dequantize(out, Q.Q8_0).astype(np.float32).ravel()[: SRC.size]
    assert np.abs(deq - SRC).max() > 1.0, "if this now round-trips, gguf-py changed"


def test_explicit_quantize_round_trips(tmp_path):
    """The fix for the quant path: quantize before handing the array over."""
    q = gguf.quants.quantize(SRC.reshape(8, 32), Q.Q8_0)
    out = _roundtrip(tmp_path, q, Q.Q8_0)
    deq = gguf.quants.dequantize(out, Q.Q8_0).astype(np.float32).ravel()[: SRC.size]
    assert np.abs(deq - SRC).max() < 0.01
    assert not np.isnan(deq).any()


def test_the_shipped_converter_casts():
    """The fastpitch converter must convert — source-level, no model needed."""
    import ast
    import pathlib

    path = pathlib.Path(__file__).parent.parent / "models" / "convert-fastpitch-to-gguf.py"
    src = path.read_text()
    # PARSE IT FIRST. An earlier version of this test only grepped the text and
    # stayed green while the file had a SyntaxError in its import block — a
    # converter that does not parse is worse than one that does not cast, and
    # the grep could not tell the difference.
    ast.parse(src)
    assert "arr = arr.astype(np.float16)" in src
    assert "gguf.quants.quantize(arr.astype(np.float32), qt)" in src


def test_bananamind_converter_casts_too():
    """e242f9bd fixed the identical latent F16 bug in the bananamind converter
    ("F32 bytes written, F16 type labels stamped on them" — corrupt local f16s
    from before the fix surfaced 2026-09-03). Pin that fix the same way; it
    was present but untested."""
    import ast
    import pathlib

    path = pathlib.Path(__file__).parent.parent / "models" / "convert-bananamind-tts-to-gguf.py"
    src = path.read_text()
    ast.parse(src)
    assert "arr = arr.astype(np.float16)" in src
    # ...and the conversion has to come BEFORE the write, not after it.
    assert src.index("arr = arr.astype(np.float16)") < src.index("writer.add_tensor(name, arr, raw_dtype=qt)")


def test_fastpitch_keeps_position_tables_f32():
    """A matmul WEIGHT may be F16; an ADDEND may not.

    ggml's Metal binary ops assert `src[1]->type == GGML_TYPE_F32`
    (ggml-metal-ops.cpp, ggml_metal_op_bin). fastpitch adds sinusoidal position
    tables straight to the hidden state — `ggml_add(x, pos_slice)` in
    fastpitch_tts.cpp — and `enc.pos_emb` / `dec.pos_emb` are 2-D, so the
    converter's `ndim >= 2 -> F16` rule turned them F16 and aborted every run.

    That abort is why f16 was written off as "the runtime cannot execute an F16
    build". It could: it was two tensors. Guarding the rule so the next person
    widening the F16 selection sees why these are excluded.
    """
    import ast
    import pathlib

    path = pathlib.Path(__file__).parent.parent / "models" / "convert-fastpitch-to-gguf.py"
    src = path.read_text()
    ast.parse(src)
    assert 'name.endswith("pos_emb") or ".pos_emb" in name' in src
    # ...and it must come after the generic F16 choice, or it cannot override it.
    assert src.index("qt = GGMLQuantizationType.F16") < src.index('name.endswith("pos_emb")')
