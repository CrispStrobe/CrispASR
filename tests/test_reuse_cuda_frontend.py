"""Optional SDK, missing-device, and physical GPU controls for RE-USE cuFFT."""
import math
import os
from pathlib import Path
import shutil
import struct
import subprocess
import tempfile
import unittest
import wave

ROOT = Path(__file__).resolve().parents[1]
PROBE = os.environ.get('REUSE_CUDA_FRONTEND_PROBE')
PROFILE = os.environ.get('REUSE_CUDA_PROFILE_CONTROL')
GPU = os.environ.get('REUSE_CUDA_TEST_GPU') == '1'


@unittest.skipUnless(shutil.which('cmake'), 'cmake is unavailable')
class DependencyControls(unittest.TestCase):
    def configure(self, directory, enabled, mkl=False):
        root = Path(directory)
        (root / 'CMakeLists.txt').write_text(
            'cmake_minimum_required(VERSION 3.18)\nproject(reuse_cuda_dependency LANGUAGES NONE)\n' +
            ('add_library(crispasr-reuse-mkl INTERFACE)\n' if mkl else '') +
            f'include("{(ROOT / "cmake/CrispasrReuseCuda.cmake").as_posix()}")\n' +
            'if(TARGET crispasr-reuse-cuda)\nmessage(FATAL_ERROR "unexpected CUDA target")\nendif()\n')
        command = ['cmake', '-S', str(root), '-B', str(root / 'build'),
                   f'-DCRISPASR_REUSE_CUDA_FRONTEND={enabled}']
        for name in ('CUDA_RUNTIME_INCLUDE_DIR', 'CUFFT_INCLUDE_DIR', 'CUDART_LIBRARY', 'CUFFT_LIBRARY'):
            command.append(f'-DCRISPASR_REUSE_{name}={root / "missing"}')
        return subprocess.run(command, capture_output=True, text=True, timeout=30)

    def test_default_needs_neither_sdk_and_creates_no_cuda_target(self):
        with tempfile.TemporaryDirectory() as root:
            result = self.configure(root, 'OFF')
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_enabled_requires_independent_cpu_hann_dependency(self):
        with tempfile.TemporaryDirectory() as root:
            result = self.configure(root, 'ON')
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('requires CRISPASR_REUSE_MKL_FRONTEND=ON', result.stdout + result.stderr)

    def test_enabled_missing_cuda_sdk_fails_explicitly(self):
        with tempfile.TemporaryDirectory() as root:
            result = self.configure(root, 'ON', mkl=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('requires an external CUDA SDK', result.stdout + result.stderr)


class ProbeFixture:
    @staticmethod
    def audio(path, count=1600, rate=16000, channels=1):
        samples = [int(7000 * math.sin(i * .137) + 3000 * math.cos(i * .071)) for i in range(count)]
        with wave.open(str(path), 'wb') as stream:
            stream.setnchannels(channels)
            stream.setsampwidth(2)
            stream.setframerate(rate)
            stream.writeframes(struct.pack('<' + 'h' * (count * channels),
                                           *(x for x in samples for _ in range(channels))))

    def invoke(self, *args, hidden=False):
        environment = os.environ.copy()
        if hidden:
            environment['CUDA_VISIBLE_DEVICES'] = '-1'
        return subprocess.run([PROBE, *map(str, args)], env=environment,
                              capture_output=True, text=True, timeout=60)


@unittest.skipUnless(PROBE, 'optional CUDA probe is not configured')
class ExecutableControls(ProbeFixture, unittest.TestCase):
    def test_invalid_nonfinite_api_input_preserves_caller_output(self):
        result = self.invoke('--invalid-self-test', hidden=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('REUSE_CUDA_INVALID_INPUT_OWNERSHIP_PASS', result.stdout)

    def test_invalid_inputs_never_create_capture(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for index, options in enumerate(({'count': 320}, {'count': 401}, {'count': 16080},
                                              {'rate': 8000}, {'channels': 2})):
                with self.subTest(options=options):
                    source, output = root / f'{index}.wav', root / f'{index}.f32'
                    self.audio(source, **options)
                    result = self.invoke(source, output, hidden=True)
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn('input', result.stderr)
                    self.assertFalse(output.exists())

    def test_hidden_device_fails_without_creating_capture_or_cpu_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.audio(root / 'input.wav')
            result = self.invoke(root / 'input.wav', root / 'output.f32', hidden=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('CUDA device unavailable', result.stderr)
            self.assertFalse((root / 'output.f32').exists())

    @unittest.skipUnless(PROFILE, 'profile control is not configured')
    def test_hidden_device_profile_rejects_before_loading_a_reference(self):
        environment = os.environ.copy()
        environment['CUDA_VISIBLE_DEVICES'] = '-1'
        result = subprocess.run([PROFILE, '--select-cufft'], env=environment,
                                capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
        self.assertIn('CUDA device unavailable', result.stderr)


@unittest.skipUnless(PROBE and GPU, 'physical GPU qualification was not requested')
class PhysicalGpuControls(ProbeFixture, unittest.TestCase):
    def test_analytic_endpoint_scale_and_silence_controls(self):
        result = self.invoke('--self-test')
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('exact_endpoint_and_scale_checks=84', result.stdout)

    def test_independent_window_full_complex_and_raw_phase_against_torch_cuda(self):
        import numpy as np
        import torch
        self.assertTrue(torch.cuda.is_available(), 'physical GPU qualification requires a CUDA device')
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for count in (400, 1600, 16000):
                with self.subTest(samples=count):
                    source, output = root / 'input.wav', root / 'output.f32'
                    self.audio(source, count=count)
                    with wave.open(str(source), 'rb') as stream:
                        audio = np.frombuffer(stream.readframes(count), '<i2').astype('f4') / 32768
                    result = self.invoke(source, output)
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                    values = np.fromfile(output, '<f4')
                    frames = 1 + count // 80
                    self.assertEqual(values.size, 640 + 4 * frames * 321)
                    self.assertTrue(np.isfinite(values).all())
                    window = torch.hann_window(640)
                    self.assertEqual(values[:640].tobytes(), window.numpy().tobytes())
                    z = torch.stft(torch.from_numpy(audio).cuda(), 640, 80, 640, window.cuda(), return_complex=True)
                    real, imaginary, magnitude, phase = values[640:].reshape(4, frames, 321)
                    self.assertEqual(real.tobytes(), z.real.T.contiguous().cpu().numpy().tobytes())
                    self.assertEqual(imaginary.tobytes(), z.imag.T.contiguous().cpu().numpy().tobytes())
                    reference_phase = torch.angle(z).T.contiguous().cpu().numpy()
                    self.assertLessEqual(float(np.max(np.abs(phase.astype('f8') - reference_phase))), 5e-7)
                    reference_magnitude = torch.log1p(torch.abs(z)).T.contiguous().cpu().numpy()
                    self.assertLessEqual(float(np.max(np.abs(magnitude - reference_magnitude))), 2e-6)
                    previous = output.read_bytes()
                    repeated = self.invoke(source, output)
                    self.assertEqual(repeated.returncode, 0, repeated.stdout + repeated.stderr)
                    self.assertEqual(previous, output.read_bytes())


if __name__ == '__main__':
    unittest.main()
