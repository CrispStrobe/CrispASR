#!/usr/bin/env python3
"""Model-free controls for offline runtime delivery and cache-log parsing."""
import io
import hashlib
import json
from pathlib import Path
import sys
import tarfile
import tempfile
import unittest
from unittest import mock
from types import SimpleNamespace

import run as task
from run import parse_allocator, prepare_runtime


class CacheTaskTests(unittest.TestCase):
    def test_output_directory_exists_before_cli_launch(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            inputs = root / "inputs"
            inputs.mkdir()
            model = b"model fixture"
            (inputs / "scnet-f16.gguf").write_bytes(model)

            def cli(command, env, stdout, stderr):
                stems = Path(command[command.index("--sep-output-dir") + 1])
                self.assertTrue(stems.is_dir(), "CLI needs the output directory before launch")
                for stem in ("bass", "drums", "other", "vocals"):
                    (stems / (stem + ".wav")).write_bytes(b"mock WAV")
                stderr.write("scnet_bench_allocator: frames=999 metadata_hit=0\n")
                return SimpleNamespace(returncode=0)

            samples = SimpleNamespace(shape=(task.FRAMES, 2))
            numpy = SimpleNamespace(isfinite=lambda x: SimpleNamespace(all=lambda: True),
                                    abs=lambda x: x, max=lambda x: .1)
            soundfile = SimpleNamespace(read=lambda *a, **k: (samples, task.RATE))
            with mock.patch.dict(sys.modules, numpy=numpy, soundfile=soundfile), \
                    mock.patch.object(task, "MODEL_SHA", hashlib.sha256(model).hexdigest()), \
                    mock.patch.object(task, "fixture", side_effect=lambda p: p.write_bytes(b"input")), \
                    mock.patch.object(task, "prepare_runtime", return_value=root / "fake-cli"), \
                    mock.patch.object(task.subprocess, "run", side_effect=cli), \
                    mock.patch("builtins.print"):
                task.run(inputs, root / "output")
            result = json.loads((root / "output" / "result.json").read_text())
            self.assertTrue(result["passed"])
            self.assertEqual(result["observed_frames"], [999])
            self.assertFalse(result["frames_match_source_prediction"])
            self.assertEqual(len(result["stems"]), 4)

    def test_allocator_records_preserve_hits_misses_and_allocation_identity(self):
        text = "\n".join([
            "unrelated output",
            "scnet_bench_allocator: frames=872 capture=0 profile=1 shared_cpu=1 metadata_hit=0 scheduler=0xa buffer=0xb before_bytes=0 after_bytes=4096 scheduler_created=1",
            "scnet_bench_ops: frames=872 nodes=123",
            "scnet_bench_allocator: frames=872 capture=0 profile=1 shared_cpu=1 metadata_hit=1 scheduler=0xa buffer=0xb before_bytes=4096 after_bytes=4096 scheduler_created=0",
            "scnet_bench_allocator: frames=884 capture=0 profile=1 shared_cpu=1 metadata_hit=0 scheduler=0xa buffer=0xb before_bytes=4096 after_bytes=4096 scheduler_created=0",
        ])
        records = parse_allocator(text)
        self.assertEqual([r["metadata_hit"] for r in records], ["0", "1", "0"])
        self.assertEqual([r["frames"] for r in records], ["872", "872", "884"])
        self.assertEqual({r["buffer"] for r in records}, {"0xb"})
        self.assertEqual(parse_allocator("unrelated output"), [])

    def test_runtime_from_archive_or_either_kaggle_expansion_layout(self):
        for layout in ("archive", "direct", "nested"):
            with self.subTest(layout=layout), tempfile.TemporaryDirectory() as name:
                root = Path(name)
                inputs = root / "inputs"
                inputs.mkdir()
                if layout == "archive":
                    with tarfile.open(inputs / "crispasr-linux-x86_64.tar.gz", "w:gz") as archive:
                        for entry, payload in (("crispasr", b"fake executable"), ("libopenblas.so.0", b"fake library")):
                            info = tarfile.TarInfo("crispasr-linux-x86_64/" + entry)
                            info.size = len(payload)
                            archive.addfile(info, io.BytesIO(payload))
                else:
                    directory = inputs / "crispasr-linux-x86_64"
                    if layout == "nested":
                        directory /= "crispasr-linux-x86_64"
                    directory.mkdir(parents=True)
                    (directory / "crispasr").write_bytes(b"fake executable")
                    (directory / "libopenblas.so.0").write_bytes(b"fake library")
                cli = prepare_runtime(inputs, root / "runtime")
                self.assertEqual(cli.read_bytes(), b"fake executable")
                self.assertEqual((cli.parent / "libopenblas.so.0").read_bytes(), b"fake library")
                self.assertTrue(cli.stat().st_mode & 0o111)

    def test_missing_or_ambiguous_extracted_runtime_fails_clearly(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            inputs = root / "inputs"
            inputs.mkdir()
            with self.assertRaisesRegex(FileNotFoundError, "Expected one mounted runtime"):
                prepare_runtime(inputs, root / "empty-output")
            for leaf in ("a", "b"):
                (inputs / leaf).mkdir()
                (inputs / leaf / "crispasr").write_bytes(b"fake")
            with self.assertRaisesRegex(FileNotFoundError, "Expected one mounted runtime"):
                prepare_runtime(inputs, root / "ambiguous-output")


if __name__ == "__main__":
    unittest.main()
