#!/usr/bin/env python3
"""Exercise the public Dia dumper/diff and capture transparency on actual weights.

Bounded 32-step replay validates harness wiring. It is not speech, quantization,
GPU or speed acceptance; current-main free-running speech has a separate worker.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PRISTINE = '9aa8df20ac4ef0b6087c7af1d1f0427eee24c2e1'
CONVERTER = '8d0ad8862d2cad5d305a48a57f2a99b5024bc39a'
ORIGINAL = '4a9e29b1bdbfe1be721353ec034a8c66c9f0a1a8'
MODEL_SHA = '8128132c538e1aef83f2fb5257b4f71d28a67c2a5a0d54d0a1f0a3b0f5d6b73d'
CODEC_SHA = '6db51da2333868918b6d476008d92036a53400bc816ff7e1d2b1e973499e4135'
TEXT = '[S1] We are committed to making it secure.'
sys.dont_write_bytecode = True


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def child(args, out):
    directory = out / args.arm
    directory.mkdir()
    sys.path.insert(0, str(args.runtime / 'python'))
    from crispasr import Session
    os.environ.update(CRISPASR_DIA_MAX_STEPS='32', CRISPASR_DIA_TTS_GPU='0',
        CRISPASR_DIA_DUMP_INPUT_TOKENS=str(directory/'inputs.i32'),
        CRISPASR_DIA_DUMP_STEPLOGITS=str(directory/'logits.f32'))
    if args.arm == 'on-identity':
        os.environ['CRISPASR_DIA_SERIAL_HEAD'] = 'identity'
    if args.arm in ['on', 'on-identity']:
        stages = directory/'stages'; stages.mkdir()
        os.environ.update(CRISPASR_DIA_DUMP_DIR=str(stages), CRISPASR_DIA_DUMP_STEP='5',
            CRISPASR_DIA_DUMP_SELF_ATTN='1', CRISPASR_DIA_DUMP_DECODER_BOUNDARIES='1',
            CRISPASR_DIA_DAC_TRACE_DIR=str(stages))
    with Session(str(args.model), backend='dia', lib_path=str(args.lib), n_threads=4) as session:
        session.set_temperature(1.2, seed=42)
        pcm = session.synthesize(TEXT)
        assert session.output_sample_rate() == 44100
    pcm = np.asarray(pcm, dtype='f4')
    assert pcm.ndim == 1 and len(pcm) and np.isfinite(pcm).all()
    pcm.tofile(directory/'pcm.f32')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--child', action='store_true')
    parser.add_argument('--arm')
    for name in ['runtime', 'model', 'lib']:
        parser.add_argument('--'+name, type=Path)
    args = parser.parse_args()
    out, scratch = Path(os.environ['HEAVY_OUT']), Path(os.environ['HEAVY_SCRATCH'])
    out.mkdir(parents=True, exist_ok=True); scratch.mkdir(parents=True, exist_ok=True)
    os.environ.update(TMPDIR=str(scratch), OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4')
    if args.child:
        child(args, out); return
    source = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    assert source == os.environ['GITHUB_SHA']
    assert not subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT)
    report = dict(source=source, pristine=PRISTINE, converter=CONVERTER, original=ORIGINAL,
        ggml=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT/'ggml', text=True).strip(),
        scope='bounded original-input/DAC-code replay and instrumentation transparency',
        speech_acceptance=False, gpu_acceptance=False, quant_acceptance=False, defaults_changed=False, runs=[])
    def checkpoint():
        (out/'receipt.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    def run(label, command, env=None, expected=0):
        with (out/(label+'.log')).open('w') as log:
            result = subprocess.run(list(map(str, command)), cwd=ROOT, env=env,
                stdout=log, stderr=subprocess.STDOUT, timeout=10800)
        report['runs'].append(dict(label=label, exit=result.returncode, sha256=sha(out/(label+'.log')))); checkpoint()
        print(label, result.returncode, flush=True)
        if expected is not None:
            assert result.returncode == expected, label
        return result.returncode
    checkpoint()
    method=out/'method'; method.mkdir()
    for folder in ['src', 'examples/cli', 'tools', 'tests']:
        for p in (ROOT/folder).rglob('*'):
            if p.is_file() and (p.name in ['dia_tts.cpp', 'dia_sampling.h', 'dia_dump_step.h', 'dia_serial_head.h', 'crispasr_dia_diff.h',
                'crispasr_diff.cpp', 'crispasr_diff.h', 'crispasr_diff_main.cpp', 'dump_reference.py',
                'dia_tts_reference.py', 'dia_attention_capture.py', 'dia_decoder_capture.py', 'dia_reference_wire.py']
                or p.name.startswith('test_dia_') or p.name == 'dia-capture-pristine-main.json'):
                dest=method/p.relative_to(ROOT); dest.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(p, dest)
    for pin in [PRISTINE, CONVERTER]: run('fetch-'+pin[:8], ['git', 'fetch', 'origin', pin])
    pristine, converter = scratch/'pristine', scratch/'converter'
    for path, pin in [(pristine, PRISTINE), (converter, CONVERTER)]:
        run('checkout-'+pin[:8], ['git', 'worktree', 'add', '--detach', path, pin])
    run('pristine-submodules', ['git', '-C', pristine, 'submodule', 'update', '--init', '--recursive'])
    shutil.copyfile(pristine/'src/dia_tts.cpp', method/'pristine-dia_tts.cpp')
    run('apt-update', ['sudo', 'bash', ROOT/'tools/ci-apt.sh', 'update'])
    run('apt-install', ['sudo', 'bash', ROOT/'tools/ci-apt.sh', 'install', '-y', 'ninja-build', 'ccache'])
    def build(runtime, label, targets):
        directory=scratch/(label+'-build')
        run(label+'-configure', ['cmake', '-G', 'Ninja', '-S', runtime, '-B', directory,
            '-DCMAKE_BUILD_TYPE=Release', '-DBUILD_SHARED_LIBS=ON', '-DCMAKE_C_COMPILER_LAUNCHER=ccache',
            '-DCMAKE_CXX_COMPILER_LAUNCHER=ccache', '-DGGML_NATIVE=OFF', '-DGGML_BLAS=OFF', '-DGGML_CUDA=OFF',
            '-DGGML_VULKAN=OFF', '-DCRISPASR_BUILD_TESTS=OFF', '-DCRISPASR_BUILD_SERVER=OFF', '-DCRISPASR_OPUS=OFF', '-DCRISPASR_AMR=OFF'])
        run(label+'-build', ['cmake', '--build', directory, '--target', *targets, '-j4'])
        assert 'dia_tts.cpp.o' in (out/(label+'-build.log')).read_text()
        dest=out/'libs'/label; dest.mkdir(parents=True)
        for p in directory.rglob('*.so*'):
            if p.is_file() and not p.is_symlink(): shutil.copyfile(p, dest/p.name)
        return directory, next(directory.rglob('libcrispasr.so'))
    _, plib = build(pristine, 'pristine', ['crispasr-lib'])
    build_dir, lib = build(ROOT, 'candidate', ['crispasr-lib', 'crispasr-diff'])
    diff = build_dir/'bin/crispasr-diff'
    assert diff.is_file()
    (out/'bin').mkdir()
    shutil.copyfile(diff, out/'bin/crispasr-diff')
    from huggingface_hub import hf_hub_download, snapshot_download
    weights=Path(snapshot_download('nari-labs/Dia-1.6B', revision='257bc72f9b78182ccc6fa07675a9ae4c1a44e2cd', allow_patterns=['config.json', 'model.safetensors']))
    template=hf_hub_download('cstr/dia-1.6b-GGUF', 'dia-1.6b-f16.gguf', revision='3233fbcb32be47761d2e736857b6d1a075b9ba7e')
    models=scratch/'models'; models.mkdir(); model=models/'dia-1.6b-f32.gguf'
    run('neural-export', [sys.executable, converter/'tools/ci-heavy/dia_neural_precision_control.py', '--export', 'f32', '--weights', weights/'model.safetensors', '--template', template, '--output', model])
    assert sha(model) == MODEL_SHA
    ledger=json.loads((out/'neural-f32-physical-audit.json').read_text())
    assert len(ledger['tensors'])==381 and all(r['selected'] and r['exported_type']=='F32' for r in ledger['tensors'])
    ct=hf_hub_download('cstr/dia-1.6b-GGUF', 'dac-44khz.gguf', revision='3233fbcb32be47761d2e736857b6d1a075b9ba7e')
    codec=models/'dac-44khz.gguf'
    run('codec-export', [sys.executable, ROOT/'models/convert-dia-dac-to-gguf.py', '--template', ct, '--output', codec, '--precision', 'output-f32', '--receipt', out/'physical-codec-audit.json'])
    assert sha(codec) == CODEC_SHA
    report.update(model_sha256=sha(model), codec_sha256=sha(codec), library_sha256=sha(lib)); checkpoint()
    blueprint=scratch/'original-dia'
    run('clone-original', ['git', 'clone', '--filter=blob:none', 'https://github.com/nari-labs/dia', blueprint])
    run('pin-original', ['git', '-C', blueprint, 'checkout', '--detach', ORIGINAL])
    shutil.copytree(blueprint/'dia', method/'original-dia', ignore=shutil.ignore_patterns('__pycache__'))
    for name,path in [('CrispASR-LICENSE',ROOT/'LICENSE'),('GGML-LICENSE',ROOT/'ggml/LICENSE'),('Dia-LICENSE',blueprint/'LICENSE')]:
        shutil.copyfile(path,out/name)
    clean={k:v for k,v in os.environ.items() if not k.startswith(('DIA_', 'CRISPASR_DIA_'))}
    tests_env=clean|dict(DIA_SOURCE=str(blueprint), CRISPASR_DIA_DIFF_GGML_BUILD=str(build_dir/'ggml'))
    for test in ['capture_neutrality', 'dump_step', 'diff_device', 'diff_compare', 'reference_dispatch', 'reference_capture', 'output_protocol']:
        run('test-'+test, [sys.executable, ROOT/('tests/test_dia_'+test+'.py'), '-v'], tests_env)
    for arm in ['pristine', 'off-before', 'on', 'off-after', 'on-identity']:
        run('native-'+arm, [sys.executable, __file__, '--child', '--arm', arm, '--runtime', pristine if arm=='pristine' else ROOT,
            '--model', model, '--lib', plib if arm=='pristine' else lib], clean)
    report['transparency_checks']=[dict(arm=arm, file=name, exact=sha(out/'pristine'/name)==sha(out/arm/name))
        for arm in ['off-before', 'on', 'off-after', 'on-identity'] for name in ['inputs.i32', 'logits.f32', 'pcm.f32']]
    report['pristine_off_on_off_exact']=all(row['exact'] for row in report['transparency_checks']); checkpoint()
    archive=out/'ref.gguf'
    env=clean|dict(DIA_SOURCE=str(blueprint), DIA_DEVICE='cpu', DIA_MAX_TOKENS='32', DIA_CAPTURE_STEP='5',
        DIA_CAPTURE_ATTENTION_LAYERS='0,17', DIA_CAPTURE_BOUNDARY_LAYERS='0,17', DIA_TEXT=TEXT, DIA_SEED='42')
    run('public-dump', [sys.executable, ROOT/'tools/dump_reference.py', '--backend', 'dia', '--model-dir', weights, '--output', archive], env)
    from gguf import GGUFReader
    reader=GGUFReader(str(archive))
    expected={t.name for t in reader.tensors}-{'decoder_codes_rev'}
    report['reference_tensors']={t.name:dict(shape=list(map(int,t.shape)), bytes=int(t.data.nbytes)) for t in reader.tensors}
    reader.data._mmap.close()
    env=clean|dict(DIA_CODEC=str(codec), CRISPASR_DIA_DIFF_DUMP_DIR=str(out/'diff-tensors'))
    code=run('public-diff', [diff, 'dia', model, archive], env, expected=None)
    log=(out/'public-diff.log').read_text()
    rows=re.findall(r'^\[(PASS|FAIL)\]\s+(\S+)\s+cos_min=',log,re.M)
    report.update(diff_exit=code, compared_stages=rows, missing_stages=sorted(expected-{name for _,name in rows}),
                  failing_stages=[name for result,name in rows if result=='FAIL'])
    checkpoint()
    # Exercise the independently integrated head node together with captures.
    # Keep original input replay and the exact same cosine/magnitude gates.
    serial_env=clean|dict(DIA_CODEC=str(codec), CRISPASR_DIA_SERIAL_HEAD='serial',
        CRISPASR_DIA_DIFF_DUMP_DIR=str(out/'serial-diff-tensors'))
    serial_code=run('public-serial-diff', [diff, 'dia', model, archive], serial_env, expected=None)
    serial_rows=re.findall(r'^\[(PASS|FAIL)\]\s+(\S+)\s+cos_min=',(out/'public-serial-diff.log').read_text(),re.M)
    report.update(serial_diff_exit=serial_code, serial_compared_stages=serial_rows,
        serial_missing_stages=sorted(expected-{name for _,name in serial_rows}),
        serial_failing_stages=[name for result,name in serial_rows if result=='FAIL'])
    checkpoint()
    # Missing references must fail through the real text-only dispatch.
    rejected=run('missing-reference-control', [diff, 'dia', model, out/'missing.gguf'], clean, expected=None)
    assert rejected != 0 and 'Usage:' not in (out/'missing-reference-control.log').read_text()
    report['missing_reference_rejected']=True
    report['payloads']=[dict(path=p.relative_to(out).as_posix(), bytes=p.stat().st_size, sha256=sha(p))
                        for p in sorted(out.rglob('*')) if p.is_file() and p.name!='receipt.json']
    checkpoint()
    assert report['pristine_off_on_off_exact'], 'Capture changed actual full-model output; raw evidence retained'
    assert not report['missing_stages'] and not report['failing_stages'] and code==0, 'Strict harness parity failed; raw evidence retained'
    assert not report['serial_missing_stages'] and not report['serial_failing_stages'] and serial_code==0, 'Composed serial head/capture parity failed; raw evidence retained'
    report['bounded_harness_pass']=True; checkpoint()
    print('DIA_REFERENCE_WIRE_COMPLETE', len(rows), flush=True)


if __name__ == '__main__':
    main()
