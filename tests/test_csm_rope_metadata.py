"""Model-free CSM RoPE source-config to GGUF-metadata transport tests."""

import sys
from pathlib import Path
import unittest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "models"))
from csm_rope_config import parse_rope_scaling, write_csm_rope_metadata


class WriterProbe:
    def __init__(self):
        self.fields = {}

    def add_string(self, key, value):
        self.fields[key] = ("string", value)

    def add_float32(self, key, value):
        self.fields[key] = ("f32", value)

    def add_uint32(self, key, value):
        self.fields[key] = ("u32", value)


def source_config():
    return {
        "rope_scaling": {
            "rope_type": "llama3", "factor": 32.0,
            "low_freq_factor": 0.125, "high_freq_factor": 0.5,
            "original_max_position_embeddings": 1024,
        },
        "depth_decoder_config": {
            "rope_theta": 500000.0,
            "rope_scaling": {
                "type": "llama3", "factor": 32.0,
                "low_freq_factor": 0.001953125, "high_freq_factor": 0.0078125,
                "original_max_position_embeddings": 16,
            },
        },
    }


class CsmRopeMetadataTests(unittest.TestCase):
    def test_source_fields_reach_writer_with_required_gguf_types(self):
        writer = WriterProbe()
        bb, dd = write_csm_rope_metadata(writer, source_config(),
                                         bb_max_pos=2048, dd_max_pos=33, bb_theta=500000.0)
        self.assertEqual((bb.kind, dd.kind), ("llama3", "llama3"))
        self.assertEqual(writer.fields, {
            "csm.backbone.rope_scaling_type": ("string", "llama3"),
            "csm.backbone.rope_scaling_factor": ("f32", 32.0),
            "csm.backbone.rope_low_freq_factor": ("f32", 0.125),
            "csm.backbone.rope_high_freq_factor": ("f32", 0.5),
            "csm.backbone.rope_orig_max_pos": ("u32", 1024),
            "csm.depth.rope_scaling_type": ("string", "llama3"),
            "csm.depth.rope_scaling_factor": ("f32", 32.0),
            "csm.depth.rope_low_freq_factor": ("f32", 0.001953125),
            "csm.depth.rope_high_freq_factor": ("f32", 0.0078125),
            "csm.depth.rope_orig_max_pos": ("u32", 16),
            "csm.depth.rope_theta": ("f32", 500000.0),
        })

    def test_unscaled_configs_are_explicit_and_depth_theta_defaults_to_backbone(self):
        for raw in (None, {}, {"rope_type": ""}, {"type": "none"}, {"rope_type": "default"}):
            with self.subTest(raw=raw):
                config = {} if raw is None else {"rope_scaling": raw}
                writer = WriterProbe()
                write_csm_rope_metadata(writer, config, bb_max_pos=2048,
                                        dd_max_pos=33, bb_theta=500000.0)
                for prefix, original in (("csm.backbone", 2048), ("csm.depth", 33)):
                    self.assertEqual(writer.fields[f"{prefix}.rope_scaling_type"], ("string", "none"))
                    self.assertEqual(writer.fields[f"{prefix}.rope_scaling_factor"], ("f32", 1.0))
                    self.assertEqual(writer.fields[f"{prefix}.rope_low_freq_factor"], ("f32", 1.0))
                    self.assertEqual(writer.fields[f"{prefix}.rope_high_freq_factor"], ("f32", 4.0))
                    self.assertEqual(writer.fields[f"{prefix}.rope_orig_max_pos"], ("u32", original))
                self.assertEqual(writer.fields["csm.depth.rope_theta"], ("f32", 500000.0))

    def test_depth_theta_comes_from_depth_config(self):
        config = source_config()
        config["depth_decoder_config"]["rope_theta"] = 250000.0
        writer = WriterProbe()
        write_csm_rope_metadata(writer, config, bb_max_pos=2048,
                                dd_max_pos=33, bb_theta=500000.0)
        self.assertEqual(writer.fields["csm.depth.rope_theta"], ("f32", 250000.0))

    def test_rejects_unsupported_and_incomplete_scaling(self):
        valid = source_config()["rope_scaling"]
        invalid = [
            {"rope_type": "linear"},
            {"rope_type": 0},
            {"rope_type": "llama3", "type": "none"},
            {"factor": 32.0},
            {**valid, "factor": 0.5},
            {**valid, "factor": float("inf")},
            {**valid, "low_freq_factor": 0},
            {**valid, "low_freq_factor": float("nan")},
            {**valid, "high_freq_factor": 0.125},
            {**valid, "original_max_position_embeddings": 0},
            {**valid, "original_max_position_embeddings": 16.0},
            {**valid, "original_max_position_embeddings": True},
        ]
        for missing in ("factor", "low_freq_factor", "high_freq_factor",
                        "original_max_position_embeddings"):
            invalid.append({key: value for key, value in valid.items() if key != missing})
        for raw in invalid:
            with self.subTest(raw=raw):
                with self.assertRaises(ValueError):
                    parse_rope_scaling({"rope_scaling": raw}, default_max_pos=2048, label="backbone")

    def test_bad_depth_rope_config_does_not_partially_write_metadata(self):
        config = source_config()
        config["depth_decoder_config"]["rope_scaling"].pop("high_freq_factor")
        writer = WriterProbe()
        with self.assertRaises(ValueError):
            write_csm_rope_metadata(writer, config, bb_max_pos=2048,
                                    dd_max_pos=33, bb_theta=500000.0)
        self.assertEqual(writer.fields, {})

    def test_non_object_depth_is_not_silently_treated_as_unscaled(self):
        for invalid in (False, 0, [], "invalid"):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                write_csm_rope_metadata(WriterProbe(), {"depth_decoder_config": invalid},
                                        bb_max_pos=2048, dd_max_pos=33, bb_theta=500000.0)


if __name__ == "__main__":
    unittest.main()
