#!/usr/bin/env python3
"""Check the current runtime head node against a pinned original projection fixture."""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
FIXTURE_SHA = 'b092ed027e26480ae4ad6c59d30d216662c7c5afe949f5300c9afafe86163397'


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['output', 'build', 'fixture']:
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False)
    assert sha(args.fixture) == FIXTURE_SHA
    ggml_pin = subprocess.check_output(['git', 'rev-parse', 'HEAD:ggml'], cwd=ROOT, text=True).strip()
    assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT/'ggml', text=True).strip() == ggml_pin
    cache = (args.build/'CMakeCache.txt').read_text()
    assert 'BUILD_SHARED_LIBS:BOOL=OFF' in cache
    assert any('CMAKE_HOME_DIRECTORY:INTERNAL='+str(home)+'\n' in cache for home in [ROOT, ROOT/'ggml'])
    library_dir = args.build/'src' if (args.build/'src/libggml-cpu.a').is_file() else args.build/'ggml/src'
    libraries = [library_dir/'libggml-cpu.a', library_dir/'libggml-base.a']
    for path in libraries:
        assert path.is_file() and path.stat().st_size > 0
    commands = []
    env = {k: v for k, v in os.environ.items() if not k.startswith(('LD_', 'GGML_', 'CRISPASR_', 'DIA_'))}
    env['TMPDIR'] = str(args.output)

    def run(label, command, expected_success=True):
        command = list(map(str, command)); commands.append(dict(label=label, command=command))
        with (args.output/(label+'.log')).open('xb') as stream:
            result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, env=env)
        assert (result.returncode == 0) == expected_success, label

    def compile_program(label, source, extra_include=None):
        exe = args.output/label
        command = ['c++', '-std=c++17', '-O3', '-march=native']
        if extra_include:
            command += ['-I', extra_include]
        command += ['-I', ROOT/'src', '-I', ROOT/'ggml/include', source,
                    '-Wl,--start-group', *libraries, '-Wl,--end-group', '-fopenmp', '-pthread', '-ldl', '-o', exe]
        run(label+'-compile', command)
        return exe

    unit = compile_program('unit', ROOT/'tests/test-dia-serial-head.cpp')
    run('unit', [unit])
    assert 'DIA_SERIAL_HEAD_UNIT_PASS' in (args.output/'unit.log').read_text()
    # Admission test must detect a deliberately removed GPU guard.
    mutation = args.output/'missing-gpu-guard'; mutation.mkdir()
    header = (ROOT/'src/dia_serial_head.h').read_text()
    assert header.count('|| !cpu ||') == 1
    (mutation/'dia_serial_head.h').write_text(header.replace('|| !cpu ||', '||'))
    bad_unit = compile_program('mutated-unit', ROOT/'tests/test-dia-serial-head.cpp', mutation)
    run('mutated-unit', [bad_unit], False)
    assert 'FAIL: GPU rejects' in (args.output/'mutated-unit.log').read_text()
    probe = compile_program('probe', Path(__file__).with_name('probe.cpp'))
    with zipfile.ZipFile(args.fixture) as archive:
        records = {row['path']: row for row in json.loads(archive.read('manifest.json'))['files']}

        def read(name):
            raw = archive.read(name)
            assert len(raw) == records[name]['bytes'] and hashlib.sha256(raw).hexdigest() == records[name]['sha256']
            return raw

        weight = np.frombuffer(read('dispatch-v1/inputs/head-weight.f32'), dtype='<f4').reshape(2048, 9, 1028)
        native_weight = args.output/'native-weight.f32'
        np.ascontiguousarray(weight.transpose(1, 2, 0)).tofile(native_weight)
        results = []
        for case in ['original-hidden', 'native-hidden']:
            hidden = np.load(io.BytesIO(read('dispatch-v1/inputs/'+case+'.npy')), allow_pickle=False)
            reference = np.load(io.BytesIO(read('order-v3/'+case+'-serial.npy')), allow_pickle=False)
            hidden_path = args.output/(case+'.f32'); hidden.tofile(hidden_path)
            baseline = None
            for threads in [1, 4]:
                for mode in ['off', 'identity', 'serial']:
                    label = case+'-'+mode+'-'+str(threads); dest = args.output/(label+'.f32')
                    run(label, [probe, native_weight, hidden_path, mode, threads, dest])
                    actual = np.fromfile(dest, dtype='<f4').reshape(2, 9, 1028)
                    assert 'DIA_SERIAL_HEAD_GRAPH_PASS' in (args.output/(label+'.log')).read_text()
                    expected = reference if mode == 'serial' else baseline
                    if baseline is None:
                        assert mode == 'off'; baseline = actual.copy()
                    else:
                        assert actual.tobytes() == expected.tobytes(), label
                    results.append(dict(case=case, mode=mode, threads=threads, values=int(actual.size), sha256=sha(dest)))
    run('whole-runtime-syntax', ['c++', '-std=c++17', '-fsyntax-only', '-I', ROOT/'src', '-I', ROOT/'include',
                                '-I', ROOT/'ggml/include', '-I', ROOT/'ggml/src', ROOT/'src/dia_tts.cpp'])
    receipt = dict(source=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                   ggml_source=ggml_pin, fixture_sha256=FIXTURE_SHA, actual_current_graph_pass=True,
                   tiny_analytic_and_admission_unit_pass=True, removed_gpu_guard_rejected=True,
                   native_identity_thread_exact=True, serial_reference_exact=True, whole_runtime_syntax_pass=True,
                   full_generation_acceptance=False, production_defaults_changed=False, results=results, commands=commands,
                   source_files=[dict(path=name, sha256=sha(ROOT/name)) for name in
                                 ['src/dia_tts.cpp', 'src/dia_serial_head.h', 'tests/test-dia-serial-head.cpp',
                                  'tests/CMakeLists.txt', 'tools/ci-heavy/dia-main-head/preflight.py',
                                  'tools/ci-heavy/dia-main-head/probe.cpp']],
                   libraries=[dict(path=str(p), bytes=p.stat().st_size, sha256=sha(p)) for p in libraries],
                   files=[dict(path=p.relative_to(args.output).as_posix(), bytes=p.stat().st_size, sha256=sha(p))
                          for p in sorted(args.output.rglob('*')) if p.is_file()])
    (args.output/'receipt.json').write_text(json.dumps(receipt, indent=2, allow_nan=False)+'\n')
    print('DIA_CURRENT_MAIN_HEAD_PREFLIGHT_PASS: exact captured recurrence, identity/thread controls, actual unit/negative control and runtime syntax')


if __name__ == '__main__':
    main()
