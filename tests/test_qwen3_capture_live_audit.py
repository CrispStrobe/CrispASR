#!/usr/bin/env python3
"""Check that the live audit rejects damaged and misattributed captures."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / "tools/ci-heavy/qwen3_capture_variants_live.py"
spec = importlib.util.spec_from_file_location("qwen_capture_audit", SCRIPT)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


class CaptureAuditTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.sample_dir = self.root / "samples" / "sample"
        self.sample_dir.mkdir(parents=True)
        self.response_dir = self.root / "requests" / "request"
        self.response_dir.mkdir(parents=True)
        self.body = b"audio bytes"
        self.response = {"complete": True, "body_bytes": len(self.body),
                         "body_sha256": hashlib.sha256(self.body).hexdigest()}
        codes = b"\0" * 8
        (self.sample_dir / "codes.bin").write_bytes(codes)
        code_record = {"file": "codes.bin", "shape": [1, 2], "sha256": hashlib.sha256(codes).hexdigest()}
        self.sample = {"request_id": "request", "sample_id": "sample", "capture_complete": True,
                       "generated_codes": dict(code_record, eos_observed=True),
                       "decoder_calls": [dict(code_record, windowed=False, decoded_pcm_sha256="pcm-hash")],
                       "model_input_text": audit.TEXT, "request_source_text": "original source text",
                       "model": {"type": "custom_voice"}, "http_request_manifest": "requests/request/manifest.json",
                       "voice": "Ryan", "reference": {"speaker_embedding_sha256": "embedding-hash"}}
        self.save()

    def save(self):
        (self.response_dir / "manifest.json").write_text(json.dumps(self.response))
        (self.sample_dir / "manifest.json").write_text(json.dumps(self.sample))

    def check(self):
        return audit.assert_manifest(self.root, "request", self.body, "custom_voice", False)

    def test_valid_capture(self):
        self.assertEqual(self.check()["sample_id"], "sample")

    def test_rejects_corrupted_code_file(self):
        (self.sample_dir / "codes.bin").write_bytes(b"\1" * 8)
        with self.assertRaises(AssertionError):
            self.check()

    def test_rejects_wrong_http_body_hash(self):
        self.response["body_sha256"] = "wrong"
        self.save()
        with self.assertRaises(AssertionError):
            self.check()

    def test_rejects_incomplete_or_wrong_variant(self):
        for change in ({"capture_complete": False}, {"model": {"type": "voice_design"}}, {"decoder_calls": []}):
            with self.subTest(change=change):
                original = self.sample.copy()
                self.sample.update(change)
                self.save()
                with self.assertRaises(AssertionError):
                    self.check()
                self.sample = original


if __name__ == "__main__":
    unittest.main()
