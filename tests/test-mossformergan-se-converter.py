#!/usr/bin/env python3
"""Model-free checks for models/convert-mossformergan-se-to-gguf.py.

Pins the runtime tensor layouts the native graph relies on (per-tap conv
matrices, dense-conv input parts, ConvTranspose taps, channels-fastest
depthwise kernels, LayerNormalization4DCF transposes) and the precision
policy (matmul weights follow --outtype; norms, slopes, biases, depthwise
kernels and the 3-channel input conv stay F32) on a tiny synthetic
checkpoint. Also checks the reference dumper's verbatim function lifting.
"""

import importlib.util
import os
import shutil
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


CONV = load("mossformergan_se_converter", ROOT / "models" / "convert-mossformergan-se-to-gguf.py")
DUMP = load("mossformergan_se_dumper", ROOT / "tools" / "dump_mossformergan_se_reference.py")

try:
    import torch  # noqa: F401
    import gguf  # noqa: F401

    HAVE_TORCH_GGUF = True
except ImportError:
    HAVE_TORCH_GGUF = False

rng = np.random.default_rng(0)


def r(*shape):
    return rng.standard_normal(shape).astype(np.float32)


class LayoutTests(unittest.TestCase):
    def test_dense_conv_taps(self):
        w = r(64, 192, 2, 3)  # third dense layer: input cat([out1, out0, x])
        a, mm = CONV.transform("dense_encoder.dilated_dense.conv3.weight", w)
        self.assertTrue(mm)
        self.assertEqual(a.shape, (6, 64, 192))
        for kh in range(2):
            for kw in range(3):
                np.testing.assert_array_equal(a[kh * 3 + kw], w[:, :, kh, kw])

    def test_single_part_convs(self):
        w = r(128, 64, 1, 3)
        a, mm = CONV.transform("mask_decoder.sub_pixel.conv.weight", w)
        self.assertTrue(mm)
        self.assertEqual(a.shape, (3, 128, 64))
        np.testing.assert_array_equal(a[2], w[:, :, 0, 2])
        w = r(64, 3, 1, 1)
        a, mm = CONV.transform("dense_encoder.conv_1.0.weight", w)
        self.assertFalse(mm)  # 3 input channels: kept F32
        np.testing.assert_array_equal(a[0], w[:, :, 0, 0])

    def test_conv_transpose_taps(self):
        w = r(128, 64, 2)
        a, mm = CONV.transform("blocks.0.inter_linear.weight", w)
        self.assertTrue(mm)
        self.assertEqual(a.shape, (2, 64, 128))
        np.testing.assert_array_equal(a[1], w[:, :, 1].T)

    def test_depthwise_channels_fastest(self):
        for name, w in (("blocks.0.intra_rnn.0.conv1.weight", r(128, 1, 39, 1)),
                        ("blocks.0.intra_to_u.mdl.3.sequential.1.conv.weight", r(128, 1, 31)),
                        ("mask_decoder.dense_block.fsmn2.fsmn.conv1.weight", r(64, 1, 9, 1))):
            a, mm = CONV.transform(name, w)
            self.assertFalse(mm)
            self.assertEqual(a.shape, (w.shape[2], w.shape[0]))
            np.testing.assert_array_equal(a, w.reshape(w.shape[0], -1).T)

    def test_fconv_and_norms(self):
        w = r(128, 1, 1, 2)
        a, _ = CONV.transform("blocks.3.Fconv.weight", w)
        np.testing.assert_array_equal(a, w[:, 0, 0, :].T)
        g = r(1, 6, 1, 101)
        a, mm = CONV.transform("blocks.0.attn_conv_K_2.2.gamma", g)
        self.assertFalse(mm)
        np.testing.assert_array_equal(a, g[0, :, 0, :].T)
        a, mm = CONV.transform("blocks.0.inter_norm.beta", r(1, 64, 1, 1))
        self.assertEqual(a.shape, (64,))
        a, mm = CONV.transform("blocks.0.attn_conv_V_1.0.weight", r(16, 64, 1, 1))
        self.assertTrue(mm)
        self.assertEqual(a.shape, (16, 64))
        _, mm = CONV.transform("blocks.0.intra_mossformer.qk_offset_scale.gamma", r(4, 128))
        self.assertFalse(mm)
        _, mm = CONV.transform("blocks.0.intra_se.avg_pool_layer.0.weight", r(64, 64))
        self.assertTrue(mm)


def synthetic_state_dict():
    import torch

    freqs = 1.0 / (10000 ** (np.arange(0, 32, 2, dtype=np.float32) / 32.0))
    sd = {
        "dense_encoder.conv_1.0.weight": r(64, 3, 1, 1),
        "dense_encoder.conv_1.1.weight": r(64),
        "dense_encoder.dilated_dense.conv2.weight": r(64, 128, 2, 3),
        "blocks.0.intra_mossformer.rotary_pos_emb.freqs": freqs,
        "blocks.0.intra_to_u.mdl.1.weight": r(128, 128),
        "blocks.0.intra_to_u.mdl.3.sequential.1.conv.weight": r(128, 1, 31),
        "blocks.5.attn_concat_proj.0.weight": r(64, 64, 1, 1),
        "blocks.5.attn_concat_proj.2.gamma": r(1, 64, 1, 101),
    }
    return {"model": {k: torch.from_numpy(v) for k, v in sd.items()}}


@unittest.skipUnless(HAVE_TORCH_GGUF, "torch and gguf are required")
class ConvertTests(unittest.TestCase):
    def convert(self, ckpt, outtype):
        import torch

        tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, tmp, True)
        path = tmp / "last_best_checkpoint.pt"
        torch.save(ckpt, path)
        out = tmp / f"m-{outtype}.gguf"
        CONV.convert(path, out, outtype)
        return out

    def test_metadata_and_precision_policy(self):
        from gguf import GGUFReader

        for outtype, mm_type in (("f32", "F32"), ("f16", "F16"), ("q8_0", "Q8_0")):
            rd = GGUFReader(str(self.convert(synthetic_state_dict(), outtype)))
            kv = {f.name: f for f in rd.fields.values()}
            arch = kv["general.architecture"]
            self.assertEqual(bytes(arch.parts[arch.data[0]]).decode(), "mossformergan-se")
            for key, want in (("n_fft", 400), ("hop_length", 100), ("num_features", 201), ("layout", 3)):
                f = kv[f"mossformergan_se.{key}"]
                self.assertEqual(int(f.parts[f.data[0]][0]), want, key)
            f = kv["mossformergan_se.compress_factor"]
            self.assertAlmostEqual(float(f.parts[f.data[0]][0]), 0.3, places=6)
            types = {t.name: t.tensor_type.name for t in rd.tensors}
            self.assertEqual(types["blocks.0.intra_to_u.mdl.1.weight"], mm_type)
            self.assertEqual(types["dense_encoder.dilated_dense.conv2.weight"], mm_type)
            self.assertEqual(types["blocks.5.attn_concat_proj.0.weight"], mm_type)
            for name in ("dense_encoder.conv_1.0.weight", "dense_encoder.conv_1.1.weight",
                         "blocks.0.intra_to_u.mdl.3.sequential.1.conv.weight", "blocks.5.attn_concat_proj.2.gamma",
                         "blocks.0.intra_mossformer.rotary_pos_emb.freqs"):
                self.assertEqual(types[name], "F32", name)
            shapes = {t.name: list(t.shape) for t in rd.tensors}  # ggml ne order
            self.assertEqual(shapes["dense_encoder.dilated_dense.conv2.weight"], [128, 64, 6])
            self.assertEqual(shapes["blocks.0.intra_to_u.mdl.3.sequential.1.conv.weight"], [128, 31])

    def test_rejects_foreign_checkpoints(self):
        import torch

        with self.assertRaises(SystemExit):
            self.convert({"model": {"encoder.weight": torch.zeros(4, 4)}}, "f16")
        bad = synthetic_state_dict()
        bad["model"]["blocks.0.intra_mossformer.rotary_pos_emb.freqs"] = torch.ones(16)
        with self.assertRaises(SystemExit):
            self.convert(bad, "f16")


class DumperTests(unittest.TestCase):
    def test_lift_functions_executes_only_named_defs(self):
        src = "import not_a_real_module\nX = 1/0\ndef keep(a):\n    return a + 1\ndef other():\n    pass\n"
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
            f.write(src)
        self.addCleanup(os.remove, f.name)
        ns = {}
        DUMP.lift_functions(Path(f.name), ["keep"], ns)
        self.assertEqual(ns["keep"](1), 2)
        self.assertNotIn("other", ns)
        with self.assertRaises(SystemExit):
            DUMP.lift_functions(Path(f.name), ["missing"], {})

    def test_decode_args_match_upstream_yaml(self):
        a = DUMP.make_args()
        self.assertEqual((a.win_type, a.win_len, a.win_inc, a.fft_len), ("hamming", 400, 100, 400))
        self.assertEqual((a.decode_window, a.one_time_decode_length, a.sampling_rate), (10, 10, 16000))


if __name__ == "__main__":
    sys.exit(unittest.main())
