#!/usr/bin/env python3
"""Compile the current Dia library/test target and replay the optional head graph."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
FIXTURE_REV = '81d18136133e7bad61961813f45afa58301fe32e'
FIXTURE_SHA = 'b092ed027e26480ae4ad6c59d30d216662c7c5afe949f5300c9afafe86163397'


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    out = Path(os.environ['HEAVY_OUT']); scratch = Path(os.environ['HEAVY_SCRATCH'])
    out.mkdir(parents=True, exist_ok=True); scratch.mkdir(parents=True, exist_ok=True)
    source = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    assert source == os.environ['GITHUB_SHA']
    assert not subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT)
    ggml = subprocess.check_output(['git', 'rev-parse', 'HEAD:ggml'], cwd=ROOT, text=True).strip()
    assert ggml == subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT/'ggml', text=True).strip()
    env = os.environ.copy(); env.update(TMPDIR=str(scratch), PYTHONDONTWRITEBYTECODE='1')
    commands = []

    def run(label, command):
        command = list(map(str, command)); commands.append(dict(label=label, command=command))
        with (out/(label+'.log')).open('xb') as log:
            subprocess.run(command, cwd=ROOT, env=env, check=True, stdout=log, stderr=subprocess.STDOUT)

    run('apt-install', ['sudo', 'bash', ROOT/'tools/ci-apt.sh', 'install', '-y', 'ninja-build', 'ccache'])
    build = scratch/'build'
    run('configure', ['cmake', '-G', 'Ninja', '-S', ROOT, '-B', build, '-DCMAKE_BUILD_TYPE=Release',
        '-DBUILD_SHARED_LIBS=OFF', '-DCMAKE_C_COMPILER_LAUNCHER=ccache', '-DCMAKE_CXX_COMPILER_LAUNCHER=ccache',
        '-DGGML_NATIVE=ON', '-DGGML_BLAS=OFF', '-DGGML_CUDA=OFF', '-DGGML_VULKAN=OFF',
        '-DCRISPASR_BUILD_TESTS=ON', '-DCRISPASR_BUILD_EXAMPLES=ON', '-DCRISPASR_BUILD_SERVER=OFF',
        '-DCRISPASR_OPUS_FETCH=ON'])
    run('build', ['cmake', '--build', build, '--target', 'dia-tts', 'test-dia-serial-head', '-j2'])
    text = (out/'build.log').read_text()
    assert 'dia_tts.cpp.o' in text and 'Linking' in text and 'test-dia-serial-head' in text
    run('ctest', ['ctest', '--test-dir', build, '-R', '^test-dia-serial-head$', '--output-on-failure', '--verbose'])
    assert 'DIA_SERIAL_HEAD_UNIT_PASS 30 checks' in (out/'ctest.log').read_text()
    fixture = scratch/'fixture.zip'
    url = 'https://huggingface.co/datasets/cstr/crispasr-regression-fixtures/resolve/'+FIXTURE_REV+'/dia-20261016-reduction-cause/complete-reduction-cause-v1.zip'
    with urllib.request.urlopen(url, timeout=300) as response, fixture.open('xb') as output:
        shutil.copyfileobj(response, output)
    assert sha(fixture) == FIXTURE_SHA and fixture.stat().st_size == 170121121
    run('preflight', [sys.executable, ROOT/'tools/ci-heavy/dia-main-head/preflight.py',
                      '--output', out/'preflight', '--build', build, '--fixture', fixture])
    preflight = json.loads((out/'preflight/receipt.json').read_text())
    assert preflight['source'] == source and preflight['actual_current_graph_pass'] and preflight['ggml_source'] == ggml
    method = out/'method'; method.mkdir()
    for name in ['src/dia_tts.cpp', 'src/dia_serial_head.h', 'tests/test-dia-serial-head.cpp', 'tests/CMakeLists.txt',
                 'tools/ci-heavy/dia_main_head_validate.py', 'tools/ci-heavy/dia-main-head/preflight.py',
                 'tools/ci-heavy/dia-main-head/probe.cpp']:
        dest = method/name; dest.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(ROOT/name, dest)
    shutil.copyfile(build/'CMakeCache.txt', out/'CMakeCache.txt')
    shutil.copyfile(build/'src/libdia-tts.a', out/'libdia-tts.a')
    (out/'libs').mkdir()
    for name in ['libggml-cpu.a', 'libggml-base.a']:
        shutil.copyfile(build/'ggml/src'/name, out/'libs'/name)
    shutil.copyfile(ROOT/'LICENSE', out/'LICENSE')
    shutil.copyfile(ROOT/'ggml/LICENSE', out/'GGML-LICENSE')
    import zipfile
    with zipfile.ZipFile(fixture) as archive:
        (out/'Dia-LICENSE').write_bytes(archive.read('Dia-LICENSE'))
    receipt = dict(source=source, ggml_source=ggml, fixture_revision=FIXTURE_REV, fixture_sha256=FIXTURE_SHA,
        actual_library_compiled=True, actual_cmake_unit_target_pass=True, actual_head_graph_pass=True,
        production_defaults_changed=False, full_generation_acceptance=False, commands=commands,
        payloads=[dict(path=p.relative_to(out).as_posix(), bytes=p.stat().st_size, sha256=sha(p))
                  for p in sorted(out.rglob('*')) if p.is_file()])
    (out/'receipt.json').write_text(json.dumps(receipt, indent=2, allow_nan=False)+'\n')
    print('DIA_CURRENT_MAIN_HEAD_BUILD_PASS: real library/CMake target and exact graph recurrence; no speech acceptance')


if __name__ == '__main__':
    main()
