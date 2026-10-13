#!/usr/bin/env python3
"""Weight-free guards: reject stale/unverified inputs before GPU/model work."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('cuda_capture', ROOT / 'tools/ci-heavy/dia_attention_cuda_control.py')
worker = importlib.util.module_from_spec(spec); spec.loader.exec_module(worker)
kernel_spec = importlib.util.spec_from_file_location('cuda_kernel',
    ROOT / 'tools/kaggle/dia-attention-cuda-control/dia_attention_cuda_control.py')
kernel = importlib.util.module_from_spec(kernel_spec); kernel_spec.loader.exec_module(kernel)


class InputGuards(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=os.environ.get('TMPDIR'))
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.bundle = self.root / 'bundle'; self.bundle.mkdir()
        self.source = self.root / 'source'; self.source.mkdir()
        runtime = {}
        for name in worker.RUNTIME_FILES:
            p = self.source / name; p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text('synthetic input guard fixture: ' + name)
            runtime[name] = worker.digest(p)
        payload = self.bundle / 'payload'; payload.write_bytes(b'synthetic; never executable')
        self.receipt = dict(source=worker.PACKAGE_SOURCE, ggml=worker.GGML_SOURCE,
                            attention_capture_source=worker.CAPTURE_SOURCE, build_complete=True,
                            tracked_worktree_dirty=False, architectures=[60, 75], cuda='12.4.1',
                            gpu_executed=False, model_acceptance=False,
                            runtime_source_sha256=runtime,
                            files={'payload': dict(bytes=payload.stat().st_size, sha256=worker.digest(payload))},
                            archive={'name': 'unused-fixture'})
        self.ledger = self.root / 'package-ledger.json'
        self.write_package()

    def write_package(self):
        self.ledger.write_text(json.dumps(dict(receipt=self.receipt)))
        (self.bundle / 'provenance.json').write_text(json.dumps({k: v for k, v in self.receipt.items() if k != 'archive'}))
        self.pin = worker.digest(self.ledger)

    def check_package(self):
        return worker.package_inputs(self.bundle, self.ledger, self.pin, self.source)

    def test_matching_weight_free_fixture_then_stale_source_rejected(self):
        self.assertEqual(self.check_package()['source'], worker.PACKAGE_SOURCE)
        self.receipt['source'] = '2954c5a7c0b120d7aec8ae3d76b3ca283c0ba4f8'
        self.write_package()
        with self.assertRaisesRegex(ValueError, 'package source'):
            self.check_package()

    def test_altered_capture_runtime_rejected(self):
        (self.source / 'src/dia_tts.cpp').write_text('capture forced on')
        with self.assertRaisesRegex(ValueError, 'runtime differs'):
            self.check_package()

    def test_tampered_and_unlisted_package_payload_rejected(self):
        (self.bundle / 'payload').write_bytes(b'tampered')
        with self.assertRaisesRegex(ValueError, 'Input payload differs'):
            self.check_package()
        (self.bundle / 'extra').write_bytes(b'unlisted')
        with self.assertRaisesRegex(ValueError, 'unlisted'):
            self.check_package()

    def test_unpinned_package_ledger_rejected(self):
        self.pin = '0' * 64
        with self.assertRaisesRegex(ValueError, 'Unpinned'):
            self.check_package()

    def test_root_escape_rejected(self):
        outside = self.root / 'outside'; outside.write_bytes(b'synthetic')
        (self.bundle / 'escape').symlink_to(outside)
        facts = dict(bytes=outside.stat().st_size, sha256=worker.digest(outside))
        with self.assertRaisesRegex(ValueError, 'escapes'):
            worker.checked_file(self.bundle, 'escape', facts)

    def test_incomplete_capture_transparency_rejected(self):
        capture = self.root / 'capture'; capture.mkdir()
        receipt = dict(source=worker.CAPTURE_SOURCE, worktree_dirty=False, step=978,
                       original_capture_off_on_off_all_common_arrays_exact=True,
                       native_capture_off_on_off_complete_arrays_exact=False,
                       late_attention_diagnostic_capture_pass=True)
        files = []
        for name in ('receipt.json', 'canonical-ref.gguf', 'original-step978-ref.gguf',
                     'scaled-step978-ref.gguf', 'negative-step-ref.gguf', 'out-of-history-step-ref.gguf'):
            p = capture / name; p.write_text(json.dumps(receipt) if name == 'receipt.json' else 'synthetic')
            files.append(dict(path=name, bytes=p.stat().st_size, sha256=worker.digest(p)))
        ledger = self.root / 'capture-ledger.json'
        ledger.write_text(json.dumps(dict(source=worker.CAPTURE_SOURCE, files=files)))
        with self.assertRaisesRegex(ValueError, 'transparency not proved'):
            worker.capture_inputs(capture, ledger, worker.digest(ledger))
        with self.assertRaisesRegex(ValueError, 'Unpinned'):
            worker.capture_inputs(capture, ledger, '0' * 64)

    def test_unrendered_kernel_stops_before_hardware_or_downloads(self):
        with self.assertRaises(AssertionError):
            kernel.main()

    def test_kernel_archive_paths_reject_escape(self):
        self.assertEqual(kernel.safe_path('bundle/libggml-cuda.so'), Path('bundle/libggml-cuda.so'))
        for name in ('../bundle/secret', '/absolute', 'bundle/../../secret', 'bundle\\secret'):
            with self.subTest(name=name), self.assertRaises(ValueError):
                kernel.safe_path(name)

    @unittest.skipUnless(sys.platform.startswith('linux'), 'ELF loader regression requires Linux')
    def test_actual_elf_loader_needs_inherited_driver_directory(self):
        # Tiny ELF fixtures test dependency lookup, never emulate a GPU.
        driver_dir = self.root / 'host-driver'; driver_dir.mkdir()
        driver_c = self.root / 'driver.c'; driver_c.write_text('int fixture_driver(void) { return 4242; }\n')
        plugin_c = self.root / 'plugin.c'
        plugin_c.write_text('extern int fixture_driver(void); int fixture_plugin(void) { return fixture_driver(); }\n')
        subprocess.run(['cc', '-shared', '-fPIC', str(driver_c), '-Wl,-soname,libfixture-driver.so.1',
                        '-o', str(driver_dir / 'libfixture-driver.so.1')], check=True)
        plugin = self.bundle / 'libfixture-plugin.so'
        subprocess.run(['cc', '-shared', '-fPIC', str(plugin_c), '-L' + str(driver_dir),
                        '-Wl,--no-as-needed', '-l:libfixture-driver.so.1', '-o', str(plugin)], check=True)
        command = [sys.executable, '-c',
                   'import ctypes,sys; lib=ctypes.CDLL(sys.argv[1]); assert lib.fixture_plugin()==4242', str(plugin)]
        parent = dict(os.environ, LD_LIBRARY_PATH=str(driver_dir))
        broken = subprocess.run(command, env=dict(parent, LD_LIBRARY_PATH=str(self.bundle)), capture_output=True)
        self.assertNotEqual(broken.returncode, 0)
        self.assertIn(b'libfixture-driver.so.1', broken.stderr)
        fixed = subprocess.run(command, env=worker.runtime_environment(self.bundle, parent), capture_output=True)
        self.assertEqual(fixed.returncode, 0, fixed.stderr.decode())
        self.assertEqual(parent['LD_LIBRARY_PATH'], str(driver_dir))


if __name__ == '__main__':
    unittest.main()
