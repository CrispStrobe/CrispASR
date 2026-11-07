"""Optional executable controls; set REUSE_MKL_FRONTEND_PROBE to the built tool."""
import math
import os
from pathlib import Path
import struct
import shutil
import subprocess
import tempfile
import unittest
import wave

PROBE = os.environ.get('REUSE_MKL_FRONTEND_PROBE')


@unittest.skipUnless(shutil.which('cmake'), 'cmake is unavailable')
class DependencyControls(unittest.TestCase):
    def configure(self, directory, enabled):
        root = Path(directory)
        module = Path(__file__).resolve().parents[1] / 'cmake/CrispasrReuseMkl.cmake'
        (root / 'CMakeLists.txt').write_text(
            'cmake_minimum_required(VERSION 3.18)\nproject(reuse_dependency_control LANGUAGES NONE)\n'
            f'include("{module.as_posix()}")\n'
            'if(TARGET crispasr-reuse-mkl)\nmessage(FATAL_ERROR "unexpected default MKL target")\nendif()\n')
        return subprocess.run(['cmake', '-S', str(root), '-B', str(root / 'build'),
                               f'-DCRISPASR_REUSE_MKL_FRONTEND={enabled}',
                               f'-DCRISPASR_REUSE_MKL_INCLUDE_DIR={root / "missing"}',
                               f'-DCRISPASR_REUSE_MKL_LIBRARY={root / "missing/libmkl_rt.so"}'],
                              capture_output=True, text=True, timeout=30)

    def test_default_configuration_needs_no_sdk_and_creates_no_mkl_target(self):
        with tempfile.TemporaryDirectory() as directory:
            result = self.configure(directory, 'OFF')
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_requested_missing_sdk_fails_explicitly(self):
        with tempfile.TemporaryDirectory() as directory:
            result = self.configure(directory, 'ON')
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('requires an external oneMKL SDK', result.stdout + result.stderr)


@unittest.skipUnless(PROBE, 'optional oneMKL probe is not configured')
class MklFrontendProbe(unittest.TestCase):
    def invoke(self, *args):
        return subprocess.run([PROBE, *map(str, args)], capture_output=True, text=True, timeout=30)

    def test_independent_endpoint_scale_and_silence_invariants(self):
        result = self.invoke('--self-test')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('exact_endpoint_and_scale_checks=84', result.stdout)
        self.assertIn('silence_bins=6741', result.stdout)

    @staticmethod
    def audio(path, samples=1600, rate=16000, channels=1):
        with wave.open(str(path), 'wb') as stream:
            stream.setnchannels(channels)
            stream.setsampwidth(2)
            stream.setframerate(rate)
            values = [int(7000 * math.sin(i * 0.137) + 3000 * math.cos(i * 0.071))
                      for i in range(samples)]
            stream.writeframes(struct.pack('<' + 'h' * (samples * channels),
                                           *(value for value in values for _ in range(channels))))

    def test_batch_single_layout_finite_and_repeated_runs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.audio(root / 'audio.wav')
            outputs = []
            for index, layout in enumerate(('batch', 'single', 'batch')):
                target = root / f'capture-{index}.f32'
                result = self.invoke(root / 'audio.wav', target, 'ha', layout)
                self.assertEqual(result.returncode, 0, result.stderr)
                payload = target.read_bytes()
                self.assertEqual(len(payload), 4 * (640 + 4 * 21 * 321))
                values = struct.unpack('<' + 'f' * (len(payload) // 4), payload)
                self.assertTrue(all(math.isfinite(value) for value in values))
                outputs.append(payload)
            self.assertEqual(outputs[0], outputs[1])
            self.assertEqual(outputs[0], outputs[2])

    def test_invalid_input_does_not_create_capture(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cases = ({'samples': 320}, {'samples': 401}, {'rate': 8000}, {'channels': 2})
            for index, options in enumerate(cases):
                with self.subTest(options=options):
                    source, target = root / f'bad-{index}.wav', root / f'bad-{index}.f32'
                    self.audio(source, **options)
                    result = self.invoke(source, target)
                    self.assertNotEqual(result.returncode, 0)
                    self.assertFalse(target.exists())
            self.assertNotEqual(self.invoke(root / 'missing.wav', root / 'output.f32', 'invented').returncode, 0)
            self.assertFalse((root / 'output.f32').exists())


if __name__ == '__main__':
    unittest.main()
