"""Pure launch controls: mock compiler/runtime and quantizer, no native execution."""
import os
from pathlib import Path
import runpy
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]


class SanitizedPythonLaunch(unittest.TestCase):
    @unittest.skipUnless(os.name == 'posix' and shutil.which('cmake'), 'POSIX CMake controls')
    def test_cmake_runtime_order_scope_and_missing_runtime(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runtime = root / 'mock libasan.so'
            runtime.touch()  # Never load this file.
            compiler = root / 'mock-compiler'
            compiler.write_text('#!/bin/sh\n[ "$1" = -print-file-name=libasan.so ] || exit 97\n'
                                'printf "%s\\n" "$MOCK_ASAN_RUNTIME"\n')
            compiler.chmod(0o755)
            (root / 'CMakeLists.txt').write_text('''cmake_minimum_required(VERSION 3.10)
project(launch_controls NONE)
enable_testing()
set(CMAKE_CXX_COMPILER_ID "${MOCK_COMPILER_ID}")
set(CMAKE_CXX_COMPILER "${MOCK_COMPILER}")
include("${LAUNCH_MODULE}")
add_test(NAME affected COMMAND unused)
add_test(NAME native COMMAND unused)
set_property(TEST affected PROPERTY ENVIRONMENT "KEEP=present")
crispasr_preload_asan_for_python_tests(affected absent)
get_property(affected_env TEST affected PROPERTY ENVIRONMENT)
get_property(native_env TEST native PROPERTY ENVIRONMENT)
file(WRITE "${CMAKE_BINARY_DIR}/environment.txt" "${affected_env}\n${native_env}\n")
''')
            env = dict(os.environ, LD_PRELOAD='', MOCK_ASAN_RUNTIME=str(runtime))
            # Do not preload anything into CMake. The project sets this value
            # after CMake starts, to inspect environment composition only.
            script = (root / 'CMakeLists.txt').read_text().replace(
                'crispasr_preload_asan_for_python_tests(affected absent)',
                'set(ENV{LD_PRELOAD} "existing.so")\n'
                'crispasr_preload_asan_for_python_tests(affected absent)')
            (root / 'CMakeLists.txt').write_text(script)
            for number, (system, compiler_id, enabled, missing) in enumerate([
                ('Linux', 'GNU', 'ON', False), ('Linux', 'GNU', 'OFF', False),
                ('Linux', 'Clang', 'ON', False), ('Darwin', 'GNU', 'ON', False),
                ('Linux', 'GNU', 'ON', True),
            ]):
                with self.subTest(system=system, compiler=compiler_id, address=enabled, missing=missing):
                    build = root / str(number)
                    env['MOCK_ASAN_RUNTIME'] = str(root / 'absent.so') if missing else str(runtime)
                    run = subprocess.run(['cmake', '-S', str(root), '-B', str(build),
                        f'-DCMAKE_SYSTEM_NAME={system}', f'-DMOCK_COMPILER_ID={compiler_id}',
                        f'-DMOCK_COMPILER={compiler}', f'-DCRISPASR_SANITIZE_ADDRESS={enabled}',
                        f'-DLAUNCH_MODULE={ROOT / "cmake/SanitizedPythonTests.cmake"}'],
                        env=env, capture_output=True, text=True, timeout=15)
                    if missing:
                        self.assertNotEqual(run.returncode, 0)
                        self.assertIn("Cannot locate the compiler's ASan runtime", run.stdout + run.stderr)
                    else:
                        self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
                        affected, native = (build / 'environment.txt').read_text().splitlines()
                        expected = (f'KEEP=present;LD_PRELOAD={runtime}:existing.so;ASAN_OPTIONS=detect_leaks=0'
                                    if number == 0 else 'KEEP=present')
                        self.assertEqual(affected, expected)
                        self.assertEqual(native, '')

    def test_quantizer_injector_preserves_runtime_before_fault_library(self):
        with mock.patch.object(sys, 'argv', ['test-quantizer-io.py', '/mock/quantizer', '/mock/faults.so']):
            module = runpy.run_path(str(ROOT / 'tests/test-quantizer-io.py'))
        case = module['QuantizerIO']('test_short_writes_and_finalization_failures')
        case.setUp()
        try:
            for preload in ['', '/mock/libasan.so', '/mock/libasan.so:/mock/other.so']:
                with self.subTest(preload=preload), mock.patch.dict(os.environ, LD_PRELOAD=preload):
                    with mock.patch('subprocess.run', return_value=mock.Mock(returncode=1)) as run:
                        case.run_quant(case.root / 'output.gguf', 'tensor')
                    env = run.call_args.kwargs['env']
                    self.assertEqual(env['LD_PRELOAD'], f'{preload}:/mock/faults.so' if preload else '/mock/faults.so')
                    self.assertEqual(env['QUANTIZER_IO_FAULT'], 'tensor')
        finally:
            case.doCleanups()


if __name__ == '__main__':
    unittest.main()
