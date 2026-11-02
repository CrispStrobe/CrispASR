#!/usr/bin/env python3
"""Model-free checks for models/convert-mossformer2-ss-to-gguf.py.

A tiny synthetic state dict with the upstream MossFormer2_SS_16K key layout
exercises the name map (GGML_MAX_NAME-safe names), the singleton-axis squeezes,
the F16/F32 precision policy, the shared rotary buffer and a GGUF round trip.
"""
from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("conv_mf2ss", ROOT / "models" / "convert-mossformer2-ss-to-gguf.py")
conv = importlib.util.module_from_spec(spec)
spec.loader.exec_module(conv)

TINY = dict(conv.HPARAMS, encoder_embedding_dim=8, mossformer_sequence_dim=8, num_mossformer_layer=2,
            query_key_dim=4, hidden_dim=16, fsmn_inner_channels=4, fsmn_lorder=3, conv_module_kernel=5,
            rotary_dim=4, encoder_kernel_size=4)


def tiny_state(h=TINY, rng=np.random.default_rng(0)):
    n, d, c, hid, qk = (h["encoder_embedding_dim"], h["mossformer_sequence_dim"], h["fsmn_inner_channels"],
                        h["hidden_dim"], h["query_key_dim"])
    k, lo, ek, spk = h["conv_module_kernel"], 2 * h["fsmn_lorder"] - 1, h["encoder_kernel_size"], h["num_spks"]
    r = lambda *s: rng.standard_normal(s).astype(np.float32)
    sd = {"enc.conv1d.weight": r(n, 1, ek), "dec.weight": r(n, 1, ek)}
    mn = "mask_net."
    sd.update({mn + "norm.weight": r(n), mn + "norm.bias": r(n), mn + "conv1d_encoder.weight": r(d, n, 1),
               mn + "pos_enc.scale": r(1), mn + "pos_enc.inv_freq": r(d // 2),
               mn + "mdl.intra_mdl.norm.weight": r(d), mn + "mdl.intra_mdl.norm.bias": r(d),
               mn + "mdl.intra_norm.weight": r(d), mn + "mdl.intra_norm.bias": r(d), mn + "prelu.weight": r(1),
               mn + "conv1d_out.weight": r(d * spk, d, 1), mn + "conv1d_out.bias": r(d * spk),
               mn + "output.0.weight": r(d, d, 1), mn + "output.0.bias": r(d),
               mn + "output_gate.0.weight": r(d, d, 1), mn + "output_gate.0.bias": r(d),
               mn + "conv1_decoder.weight": r(n, d, 1)})
    freqs = r(h["rotary_dim"] // 2)
    blk = mn + "mdl.intra_mdl.mossformerM."

    def ffconv(prefix, din, dout, scale):
        out = {prefix + ".mdl.0.g": r(1)} if scale else {prefix + ".mdl.0.weight": r(din), prefix + ".mdl.0.bias": r(din)}
        out.update({prefix + ".mdl.1.weight": r(dout, din), prefix + ".mdl.1.bias": r(dout),
                    prefix + ".mdl.3.sequential.1.conv.weight": r(dout, 1, k)})
        return out

    for i in range(h["num_mossformer_layer"]):
        lp, fp = f"{blk}layers.{i}.", f"{blk}fsmn.{i}."
        sd[lp + "rotary_pos_emb.freqs"] = freqs.copy()
        sd.update(ffconv(lp + "to_hidden", d, hid, True))
        sd.update(ffconv(lp + "to_qk", d, qk, True))
        sd.update(ffconv(lp + "to_out", hid // 2, d, True))
        sd.update({lp + "qk_offset_scale.gamma": r(4, qk), lp + "qk_offset_scale.beta": r(4, qk)})
        sd.update({fp + "conv1.0.weight": r(c, d, 1), fp + "conv1.0.bias": r(c), fp + "conv1.1.weight": r(1),
                   fp + "norm1.weight": r(c), fp + "norm1.bias": r(c), fp + "norm2.weight": r(c),
                   fp + "norm2.bias": r(c), fp + "conv2.weight": r(d, c, 1), fp + "conv2.bias": r(d)})
        sd.update(ffconv(fp + "gated_fsmn.to_u", c, c, False))
        sd.update(ffconv(fp + "gated_fsmn.to_v", c, c, False))
        g = fp + "gated_fsmn.fsmn."
        sd.update({g + "linear.weight": r(c, c), g + "linear.bias": r(c), g + "project.weight": r(c, c),
                   g + "conv.conv1.weight": r(c, 1, lo, 1), g + "conv.norm1.weight": r(c),
                   g + "conv.norm1.bias": r(c), g + "conv.prelu1.weight": r(c),
                   g + "conv.conv2.weight": r(c, 2, lo, 1), g + "conv.norm2.weight": r(c),
                   g + "conv.norm2.bias": r(c), g + "conv.prelu2.weight": r(c)})
    return sd


class ConverterTest(unittest.TestCase):
    def test_names_fit_ggml_and_are_unique(self):
        sd = tiny_state()
        self.assertEqual(len(sd), 19 + TINY["num_mossformer_layer"] * 45)  # same layout as the release (1099 @ 24)
        names = [conv.map_name(k) for k in sd]
        kept = [n for n in names if n is not None]
        self.assertEqual(len(kept), len(set(kept)))
        self.assertTrue(all(len(n) < 64 for n in kept))
        self.assertEqual(names.count(None), TINY["num_mossformer_layer"])  # per-layer rotary buffers

    def test_layouts_and_precision_policy(self):
        sd = tiny_state()
        t = conv.build_tensors(sd, "f16", TINY)
        lo, k = 2 * TINY["fsmn_lorder"] - 1, TINY["conv_module_kernel"]
        self.assertEqual(t["blk.1.flash.hidden.dw.weight"].shape, (16, k))
        self.assertEqual(t["blk.0.fsmn.ddn.conv1.weight"].shape, (4, lo))
        self.assertEqual(t["blk.0.fsmn.ddn.conv2.weight"].shape, (4, 2, lo))
        self.assertEqual(t["mn.conv_out.weight"].shape, (16, 8))
        np.testing.assert_array_equal(t["blk.0.fsmn.ddn.conv2.weight"],
                                      sd["mask_net.mdl.intra_mdl.mossformerM.fsmn.0.gated_fsmn.fsmn.conv.conv2.weight"][..., 0])
        f16 = sorted(n for n, a in t.items() if a.dtype == np.float16)
        for n, a in t.items():
            # mul_mat matrices only; ggml_ssm_conv / broadcast operands stay F32.
            self.assertEqual(a.dtype == np.float16, conv.is_matmul(n), n)
        self.assertIn("blk.0.fsmn.conv1.weight", f16)
        self.assertNotIn("blk.0.fsmn.ddn.conv1.weight", f16)
        self.assertNotIn("blk.0.flash.qk_os.gamma", f16)
        self.assertNotIn("blk.0.flash.qk.dw.weight", f16)
        self.assertTrue(all(a.dtype == np.float32 for a in conv.build_tensors(sd, "f32", TINY).values()))

    def test_rejects_inconsistent_checkpoints(self):
        sd = tiny_state()
        sd["mask_net.mdl.intra_mdl.mossformerM.layers.1.rotary_pos_emb.freqs"] += 1
        with self.assertRaises(ValueError):
            conv.build_tensors(sd, "f16", TINY)
        sd = tiny_state()
        sd["mask_net.something_new.weight"] = np.zeros(3, np.float32)
        with self.assertRaises(KeyError):
            conv.build_tensors(sd, "f16", TINY)
        sd = tiny_state()
        del sd["dec.weight"]
        with self.assertRaises(ValueError):
            conv.build_tensors(sd, "f16", TINY)
        sd = tiny_state()
        sd["module.enc.conv1d.weight"] = sd.pop("enc.conv1d.weight")  # DataParallel prefix is accepted
        conv.build_tensors(sd, "f16", TINY)

    def test_gguf_round_trip(self):
        try:
            import gguf
        except ImportError:
            self.skipTest("gguf package not installed")
        t = conv.build_tensors(tiny_state(), "f16", TINY)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tiny.gguf"
            conv.write_gguf(t, path, "f16", "0" * 64, TINY)
            reader = gguf.GGUFReader(str(path))
            field = lambda key: reader.fields[key]
            arch = field("general.architecture")
            self.assertEqual(bytes(arch.parts[arch.data[0]]).decode(), "mossformer2-ss")
            layers = field("mossformer2_ss.num_mossformer_layer")
            self.assertEqual(int(layers.parts[layers.data[0]][0]), 2)
            types = {tensor.name: tensor.tensor_type for tensor in reader.tensors}
            self.assertEqual(len(types), len(t))
            self.assertEqual(types["blk.0.flash.hidden.linear.weight"], gguf.GGMLQuantizationType.F16)
            self.assertEqual(types["blk.0.flash.hidden.dw.weight"], gguf.GGMLQuantizationType.F32)
            self.assertEqual(types["rotary.freqs"], gguf.GGMLQuantizationType.F32)


if __name__ == "__main__":
    unittest.main()
