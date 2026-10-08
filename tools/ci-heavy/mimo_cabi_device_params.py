#!/usr/bin/env python3
"""Verify MiMo C-ABI open settings with real library calls and an init hook.

No model inference or GPU-performance acceptance is claimed. Removing the two
new forwarding assignments must fail, then the restored build must pass.
"""
import ctypes as C
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]


def worker(library, result_file):
    class OpenParams(C.Structure):
        _fields_ = [(name, C.c_int) for name in
                    ['abi_version', 'n_threads', 'use_gpu', 'verbosity', 'flash_attn', 'n_gpu_layers']] + [
                        ('reserved', C.c_int * 6)]
    lib = C.CDLL(library)
    shim = C.CDLL(os.environ['LD_PRELOAD'])
    shim.mimo_probe_param.argtypes = [C.c_int]
    shim.mimo_probe_param.restype = shim.mimo_probe_calls.restype = C.c_int
    lib.crispasr_session_open_with_params.argtypes = [C.c_char_p, C.c_char_p, C.c_void_p]
    lib.crispasr_session_open_with_params.restype = C.c_void_p
    rows = []
    def check(opts, expected):
        before = shim.mimo_probe_calls()
        handle = lib.crispasr_session_open_with_params(b'/nonexistent/mimo-asr-probe.gguf', b'mimo-asr', opts)
        assert not handle, 'Initializer hook must return a failed open'
        assert shim.mimo_probe_calls() == before + 1, 'Initializer interposition was not reached'
        actual = [shim.mimo_probe_param(i) for i in range(3)]
        rows.append(dict(expected=expected, actual=actual))
        Path(result_file).write_text(json.dumps(rows, indent=2) + '\n')
        assert actual == expected, ('MIMO_CABI_PARAMS_MISMATCH', rows[-1])
    for gpu, verbosity in itertools.product([0, 1], [0, 2]):
        opts = OpenParams(2, 7, gpu, verbosity, -1, -1)
        check(C.byref(opts), [7, gpu, verbosity])
        # Even a failed open must restore TLS defaults for the next caller.
        check(None, [4, 1, 0])
    print('MIMO_CABI_PARAMS_PASS: 4 combinations, 4 default-restoration checks', flush=True)


def main():
    out = Path(os.environ['HEAVY_OUT'])
    scratch = Path(os.environ['HEAVY_SCRATCH']) / 'mimo-cabi-device'
    out.mkdir(parents=True, exist_ok=True)
    scratch.mkdir(parents=True, exist_ok=True)
    os.environ['TMPDIR'] = str(scratch)
    receipt = dict(passed=False, source=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                   scope=__doc__)
    def save():
        (out / 'cabi-params.json').write_text(json.dumps(receipt, indent=2) + '\n')
    def run(args, tag, env=None, should_pass=True):
        with (out / (tag + '.log')).open('wb') as log:
            result = subprocess.run(list(map(str, args)), cwd=ROOT, env=env,
                                    stdout=log, stderr=subprocess.STDOUT, timeout=1800)
        log = (out / (tag + '.log')).read_text()
        print(tag, result.returncode, log[-1500:], flush=True)
        assert (result.returncode == 0) if should_pass else (result.returncode != 0)
        return log
    save()
    build = scratch / 'build'
    run(['cmake', '-S', ROOT, '-B', build, '-G', 'Ninja', '-DCMAKE_BUILD_TYPE=Release',
         '-DBUILD_SHARED_LIBS=ON', '-DGGML_NATIVE=OFF', '-DGGML_BLAS=OFF',
         '-DCRISPASR_BUILD_SERVER=OFF', '-DCRISPASR_BUILD_TESTS=OFF'], 'configure')
    run(['cmake', '--build', build, '--target', 'crispasr-lib', '-j4'], 'build-fixed')
    library = next(build.rglob('libcrispasr.so'))
    shim = scratch / 'mimo-init-probe.so'
    run(['c++', '-std=c++17', '-shared', '-fPIC', '-I' + str(ROOT / 'src'),
         ROOT / 'tools/ci-heavy/mimo_cabi_device_probe.cpp', '-o', shim], 'build-probe')
    env = dict(os.environ, LD_PRELOAD=str(shim))
    command = [sys.executable, __file__, '--worker', library]
    run([*command, out / 'fixed-before.json'], 'fixed-before', env)
    source = ROOT / 'src/crispasr_c_api.cpp'
    original = source.read_text()
    needle = ('mimo_asr_context_params p = mimo_asr_context_default_params();\n'
              '        p.n_threads = s->n_threads;\n'
              '        p.verbosity = g_open_verbosity_tls;\n'
              '        p.use_gpu = g_open_use_gpu_tls;\n')
    assert original.count(needle) == 1
    legacy = needle.replace('        p.verbosity = g_open_verbosity_tls;\n', '').replace(
                           '        p.use_gpu = g_open_use_gpu_tls;\n', '')
    try:
        source.write_text(original.replace(needle, legacy))
        run(['cmake', '--build', build, '--target', 'crispasr-lib', '-j4'], 'build-negative')
        log = run([*command, out / 'negative.json'], 'negative-control', env, should_pass=False)
        assert 'MIMO_CABI_PARAMS_MISMATCH' in log
        receipt['negative_control'] = 'Removing device/verbosity assignments is rejected by actual C-ABI calls'
        save()
    finally:
        source.write_text(original)
    run(['cmake', '--build', build, '--target', 'crispasr-lib', '-j4'], 'build-restored')
    run([*command, out / 'fixed-restored.json'], 'fixed-restored', env)
    receipt.update(passed=True, combinations=4, failed_open_default_restoration=4)
    save()
    (out / 'summary.md').write_text('MiMo C ABI settings PASS: four device/verbosity combinations, four TLS restoration checks; actual-library negative control rejected. No model/GPU inference claim.\n')


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--worker':
        worker(*sys.argv[2:])
    else:
        main()
