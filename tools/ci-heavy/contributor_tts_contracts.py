#!/usr/bin/env python3
"""Build contributor TTS fixes and regenerate/verify actual CLI/C ABI capabilities.

Model-free contract proof only. Original guided-stage and decoded/clone model
acceptance remains separate; no model or quality claims follow from this job.
"""
import ctypes
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(os.environ['HEAVY_OUT'])
SCRATCH = Path(os.environ['HEAVY_SCRATCH']) / 'contributor-tts-contracts'
OUT.mkdir(parents=True, exist_ok=True)
SCRATCH.mkdir(parents=True, exist_ok=True)
os.environ['TMPDIR'] = str(SCRATCH)


def run(command, name):
    with (OUT / (name + '.log')).open('w') as log:
        result = subprocess.run(list(map(str, command)), cwd=ROOT, stdout=log,
                                stderr=subprocess.STDOUT, timeout=4800)
    print(name, result.returncode, (OUT / (name + '.log')).read_text()[-2000:], flush=True)
    assert result.returncode == 0, name


run(['sudo', 'bash', 'tools/ci-apt.sh', 'update'], 'apt-update')
run(['sudo', 'bash', 'tools/ci-apt.sh', 'install', '-y', 'clang-format-18'], 'clang-format')
run([sys.executable, 'tests/test_tts_session_contracts.py', '-v'], 'contracts')
build = SCRATCH / 'build'
run(['cmake', '-S', ROOT, '-B', build, '-G', 'Ninja', '-DCMAKE_BUILD_TYPE=Release',
     '-DBUILD_SHARED_LIBS=ON', '-DGGML_NATIVE=OFF', '-DGGML_BLAS=OFF',
     '-DCRISPASR_BUILD_SERVER=OFF', '-DCRISPASR_BUILD_TESTS=ON'], 'configure')
run(['cmake', '--build', build, '--target', 'crispasr-cli', 'crispasr-lib',
     'test-session-output-rate-parity', '-j4'], 'build')
run(['ctest', '--test-dir', build, '-R', 'every synthesizing backend reports an output sample rate',
     '--output-on-failure'], 'rate-dispatch')
run([sys.executable, 'tools/gen-backend-caps-table.py', '--crispasr', build / 'bin/crispasr'], 'generate-caps')
run([sys.executable, 'tools/gen-feature-matrix.py', '--crispasr', build / 'bin/crispasr'], 'generate-features')
for relative in ('src/core/backend_caps_table.h', 'docs/feature-matrix.md', 'docs/feature-matrix.html'):
    shutil.copy2(ROOT / relative, OUT / Path(relative).name)
run(['cmake', '--build', build, '--target', 'crispasr-lib', '-j4'], 'rebuild-abi')
run([sys.executable, 'tools/gen-backend-caps-table.py', '--check', '--crispasr', build / 'bin/crispasr'], 'caps-drift')
backends = json.loads(subprocess.check_output([str(build / 'bin/crispasr'), '--list-backends-json'], text=True))
if isinstance(backends, dict):
    backends = backends['backends']
(OUT / 'cli-backends.json').write_text(json.dumps(backends, indent=2) + '\n')
lib = ctypes.CDLL(str(next(build.rglob('libcrispasr.so'))))
fn = lib.crispasr_backend_caps_abi
fn.argtypes = [ctypes.c_char_p, ctypes.c_void_p, ctypes.c_int32]
fn.restype = ctypes.c_int
rows = []
for backend in backends:
    name = backend['name']
    buffer = ctypes.create_string_buffer(4096)
    result = fn(name.encode(), buffer, len(buffer))
    expected = ','.join(backend['caps'])
    assert result == len(expected) and buffer.value.decode() == expected, name
    if name.startswith('tada') or name == 'dots-tts':
        assert backend['caps_bitmask'] & 131072 and 'voice-cloning' in backend['caps'], name
        rows.append(dict(name=name, bits=backend['caps_bitmask'], capabilities=expected))
assert len(rows) >= 5
source = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
receipt = dict(source=source, contracts_passed=4, output_rate_dispatch_passed=True,
               cli_cabi_rows_checked=len(backends), changed_capabilities=rows,
               generated_caps_sha256=hashlib.sha256((OUT/'backend_caps_table.h').read_bytes()).hexdigest(),
               model_acceptance=False, original_guidance_parity=False)
(OUT / 'validation.json').write_text(json.dumps(receipt, indent=2) + '\n')
(OUT / 'summary.md').write_text('TTS failure/ownership contracts and actual CLI/C ABI capability parity PASS.\n'
                               'Model-free only; guided/clone numerical acceptance remains pending.\n')
print(json.dumps(receipt), flush=True)
