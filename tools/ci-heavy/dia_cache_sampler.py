#!/usr/bin/env python3
"""Factorial Dia cache/sampler diagnosis with unchanged default and speech gates."""
import argparse
import hashlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import wave

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
GRAPH = 'fb4e5924ff00ca7f31d8a4b94a3db6b50f8347b2'
PRISTINE = GRAPH
CONVERTER = '8d0ad8862d2cad5d305a48a57f2a99b5024bc39a'
ORIGINAL = '4a9e29b1bdbfe1be721353ec034a8c66c9f0a1a8'
GGML = '0a862076567afd8468822ccc512c97ac6a69b340'
TEXT = '[S1] We are committed to making it secure.'
MODEL_SHA = '8128132c538e1aef83f2fb5257b4f71d28a67c2a5a0d54d0a1f0a3b0f5d6b73d'
CODEC_SHA = '6db51da2333868918b6d476008d92036a53400bc816ff7e1d2b1e973499e4135'
ASR_SHA = '60a2b074cdf46e81d72bc15d3eb79505dafbb697135f558659d9068f651854a2'
sys.dont_write_bytecode = True

FACTORIAL = ['serial-f16-standard', 'serial-f16-portable', 'serial-f32-standard', 'serial-f32-portable']
ARMS = ['pristine', 'off', 'identity'] + FACTORIAL


def arm_environment(arm):
    mode = 'serial' if arm.startswith('serial-') else 'identity' if arm == 'identity' else 'off'
    return dict(CRISPASR_DIA_SERIAL_HEAD=mode,
                CRISPASR_DIA_PORTABLE_RNG='1' if arm.endswith('-portable') else '0',
                CRISPASR_DIA_CROSS_CACHE_F32='1' if '-f32-' in arm else '0')


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def exact(a, b):
    assert a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes()


def grid_from_inputs(inputs):
    # Current production breaks before incrementing dec_step and discards the
    # last emitted row. The remaining delayed rows are exactly subsequent
    # decoder inputs. Decode-only replay below checks their actual native PCM.
    assert inputs.dtype == np.dtype('<i4') and inputs.ndim == 2 and inputs.shape[1] == 9
    assert 16 < len(inputs) < 3072 and (inputs[0] == 1026).all()
    assert ((inputs >= 0) & (inputs <= 1026)).all()
    raw = inputs[1:]
    aligned = np.stack([raw[np.arange(len(raw)-15)+delay, c]
                        for c, delay in enumerate([0, 8, 9, 10, 11, 12, 13, 14, 15])], axis=1)
    return np.where(aligned < 1024, aligned, 0).astype('<u4')


def score(text):
    expected, actual = 'we are committed to making it secure'.split(), re.findall('[a-z]+', text.lower())
    dp = list(range(len(actual)+1))
    for i, a in enumerate(expected, 1):
        row = [i]
        for j, b in enumerate(actual, 1):
            row.append(min(row[-1]+1, dp[j]+1, dp[j-1]+(a != b)))
        dp = row
    return dp[-1]/len(expected)


def metric(arm, name, actual, expected):
    a = np.asarray(actual, dtype='f8').reshape(-1)
    b = np.asarray(expected, dtype='f8').reshape(-1)
    assert a.shape == b.shape and len(a) and np.isfinite(a).all() and np.isfinite(b).all()
    an, bn = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    assert an > 0 and bn > 0
    row = dict(arm=arm, stage=name, values=len(a), cosine=float(np.dot(a, b)/(an*bn)), mine_norm=an, reference_norm=bn,
               relative_l2=float(np.linalg.norm(a-b)/bn), norm_relative_error=abs(an/bn-1), max_abs=float(np.max(np.abs(a-b))))
    row['pass'] = row['cosine'] >= .99 and row['norm_relative_error'] <= .01 and row['relative_l2'] <= .02
    return row


def child(args, out):
    directory = out/'arms'/args.arm
    directory.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(args.runtime/'python'))
    from crispasr import Session
    if args.child in ['native', 'decode-native', 'auto']:
        if args.arm != 'pristine':
            os.environ.update(arm_environment(args.arm))
        os.environ['CRISPASR_DIA_TTS_GPU'] = '0'
        for suffix in ['FORCE_TOKENS', 'FORCE_OUTPUT_TOKENS', 'GREEDY', 'MAX_STEPS', 'SOURCE_ROPE']:
            assert not any(os.environ.get(p+suffix) for p in ['DIA_', 'CRISPASR_DIA_'])
        if args.child == 'auto':
            with Session(str(args.model), lib_path=str(args.lib), n_threads=4) as session:
                assert session.backend == 'dia'
                (directory/'auto-open.json').write_text(json.dumps(dict(backend=session.backend, model=args.model.name))+'\n')
            return
        if args.child == 'native':
            assert not os.environ.get('CRISPASR_DIA_DECODE_CODES') and not os.environ.get('DIA_DECODE_CODES')
            os.environ.update(CRISPASR_DIA_DUMP_INPUT_TOKENS=str(directory/'inputs.i32'),
                CRISPASR_DIA_DUMP_STEPLOGITS=str(directory/'logits.f32'), CRISPASR_DIA_DUMP_TOKENS='1',
                CRISPASR_DIA_DUMP_DIR=str(directory/'stages'))
            (directory/'stages').mkdir()
        else:
            os.environ['CRISPASR_DIA_DECODE_CODES'] = str(directory/'codes.u32')
        with Session(str(args.model), lib_path=str(args.lib), backend='dia', n_threads=4) as session:
            session.set_temperature(1.2, seed=42)
            pcm = session.synthesize(TEXT)
            assert session.output_sample_rate() == 44100
        pcm = np.asarray(pcm, dtype='f4')
        assert pcm.ndim == 1 and len(pcm) and np.isfinite(pcm).all()
        np.save(directory/('native-pcm.npy' if args.child == 'native' else 'decode-native-pcm.npy'), pcm, allow_pickle=False)
    elif args.child == 'dac':
        import torch
        import dac
        assert torch.__version__ == '2.6.0+cpu'; torch.set_num_threads(4)
        # Use the verified original observer, not main's legacy reference shim.
        sys.path.insert(0, str(args.converter/'tools'))
        from reference_backends import dia_tts_reference as reference
        _, source_hashes = reference.verify_source(args.blueprint)
        sys.path.insert(0, str(args.blueprint))
        from dia.audio import decode
        checkpoint = Path(dac.utils.download())
        assert sha(checkpoint) == 'a88eed82a7024ccc1facdb1e605c4c2f99281c8118c22c9895ffa846d8fb61aa'
        codec = dac.DAC.load(str(checkpoint)).cpu().eval()
        import dac.model.dac as dm
        import dac.nn.quantize as dq
        import dac.nn.layers as dl
        for module, name in [(dm, 'dac-model.py'), (dq, 'dac-quantize.py'), (dl, 'dac-layers.py')]:
            shutil.copyfile(inspect.getsourcefile(module), directory/name)
        (directory/'dac-provenance.json').write_text(json.dumps(dict(source_hashes=source_hashes,
            checkpoint_sha256=sha(checkpoint), torch_version=torch.__version__, threads=4,
            package=importlib.metadata.version('descript-audio-codec')), indent=2)+'\n')
        grid = np.load(out/'original-codes.npy', allow_pickle=False) if args.arm == 'original' else np.fromfile(directory/'codes.u32', dtype='<u4').reshape(-1, 9)
        assert grid.ndim == 2 and grid.shape[1] == 9 and len(grid) and ((grid >= 0) & (grid < 1024)).all()
        with torch.inference_mode():
            pcm = decode(codec, torch.from_numpy(grid.astype('i8').T.copy()).unsqueeze(0)).squeeze().numpy().copy()
        assert pcm.dtype == np.float32 and pcm.ndim == 1 and np.isfinite(pcm).all()
        np.save(directory/'original-dac-pcm.npy', pcm, allow_pickle=False)
    elif args.child == 'asr':
        pcm = np.load(directory/args.audio, allow_pickle=False)
        with Session(str(args.model), lib_path=str(args.lib), backend='nemotron', n_threads=4) as session:
            text = ' '.join(s.text for s in session.transcribe(pcm, sample_rate=44100, language='en')).strip()
        row = dict(transcript=text, wer=score(text), pcm_sha256=sha(directory/args.audio),
            asr_sha256=sha(args.model), library_sha256=sha(args.lib), seconds=len(pcm)/44100,
            rms=float(np.sqrt(np.mean(pcm.astype('f8')**2))))
        (directory/(args.audio+'.asr.json')).write_text(json.dumps(row, indent=2)+'\n')
    print('DIA_MAIN_CHILD_COMPLETE', args.child, args.arm, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--child', choices=['native', 'decode-native', 'auto', 'dac', 'asr'])
    parser.add_argument('--arm', choices=ARMS+['original'])
    parser.add_argument('--audio')
    for name in ['runtime', 'converter', 'blueprint', 'model', 'lib']:
        parser.add_argument('--'+name, type=Path)
    args = parser.parse_args()
    out, scratch = Path(os.environ['HEAVY_OUT']), Path(os.environ['HEAVY_SCRATCH'])
    out.mkdir(parents=True, exist_ok=True); scratch.mkdir(parents=True, exist_ok=True)
    os.environ['TMPDIR'] = str(scratch)
    if args.child:
        child(args, out); return
    source = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    assert source == os.environ['GITHUB_SHA'] and not subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT)
    assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT/'ggml', text=True).strip() == GGML
    report = dict(source=source, pristine_source=PRISTINE, ggml_source=GGML, converter_source=CONVERTER,
        original_source=ORIGINAL, text=TEXT, seed=42, capacity=3072, temperature=1.2, top_p=.95, top_k=45, cfg=3,
        factorial=FACTORIAL, runs=[], production_defaults_changed=False,
        production_speech_acceptance=False, gpu_acceptance=False, q4_acceptance=False, bounded_harness_previously_validated=True)
    def checkpoint():
        (out/'receipt.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    def run(label, command, env=None):
        with (out/(label+'.log')).open('w') as log:
            result = subprocess.run(list(map(str, command)), cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=10800)
        report['runs'].append(dict(label=label, exit=result.returncode, sha256=sha(out/(label+'.log')))); checkpoint()
        print(label, result.returncode, flush=True)
        assert result.returncode == 0, label
    checkpoint()
    method = out/'method'; method.mkdir()
    for name in ['tools/ci-heavy/dia_cache_sampler.py', 'tools/ci-heavy/dia-cache-sampler/selftest.py', 'tools/ci-heavy/dia-main-head/speech_selftest.py', 'src/dia_tts.cpp', 'src/dia_serial_head.h', 'src/dia_sampling.h', 'src/dia_exponential_rng.h', 'src/dia_dump_step.h', 'python/crispasr/_binding.py']:
        dest=method/name; dest.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(ROOT/name, dest)
    run('fetch-graph', ['git', 'fetch', '--depth=1', 'origin', GRAPH])
    for name in ['src/dia_serial_head.h', 'src/crispasr_c_api.cpp',
                 'examples/cli/crispasr_backend_dia.cpp', 'python/crispasr/_binding.py']:
        assert (ROOT/name).read_bytes() == subprocess.check_output(['git', 'show', GRAPH+':'+name], cwd=ROOT)
    changed=subprocess.check_output(['git', 'diff', '--name-only', PRISTINE, source, '--', 'src', 'include', 'ggml', 'examples/cli', 'python'], cwd=ROOT, text=True).splitlines()
    assert changed == ['src/dia_exponential_rng.h', 'src/dia_sampling.h', 'src/dia_tts.cpp'], changed
    pristine = scratch/'pristine'
    run('checkout-pristine', ['git', 'worktree', 'add', '--detach', pristine, PRISTINE])
    run('pristine-submodules', ['git', '-C', pristine, 'submodule', 'update', '--init', '--recursive'])
    run('fetch-converter', ['git', 'fetch', 'origin', CONVERTER])
    converter = scratch/'conversion-tools'
    run('checkout-converter', ['git', 'worktree', 'add', '--detach', converter, CONVERTER])
    for name in ['tools/ci-heavy/dia_neural_precision_control.py', 'tools/dia_model_precision.py', 'tools/dia_dac_export.py',
                 'tools/reference_backends/dia_tts_reference.py', 'tools/reference_backends/_hooks.py', 'tools/reference_backends/__init__.py']:
        dest=method/'converter'/name; dest.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(converter/name, dest)
    shutil.copyfile(pristine/'src/dia_tts.cpp', method/'pristine-dia_tts.cpp')
    run('apt-update', ['sudo', 'bash', ROOT/'tools/ci-apt.sh', 'update'])
    run('apt-install', ['sudo', 'bash', ROOT/'tools/ci-apt.sh', 'install', '-y', 'ninja-build', 'ccache'])
    def build(runtime, label):
        directory = scratch/(label+'-build')
        run(label+'-configure', ['cmake', '-G', 'Ninja', '-S', runtime, '-B', directory,
            '-DCMAKE_BUILD_TYPE=Release', '-DBUILD_SHARED_LIBS=ON', '-DCMAKE_C_COMPILER_LAUNCHER=ccache',
            '-DCMAKE_CXX_COMPILER_LAUNCHER=ccache', '-DGGML_NATIVE=OFF', '-DGGML_BLAS=OFF', '-DGGML_CUDA=OFF',
            '-DGGML_VULKAN=OFF', '-DCRISPASR_BUILD_TESTS=OFF', '-DCRISPASR_BUILD_SERVER=OFF', '-DCRISPASR_OPUS=OFF', '-DCRISPASR_AMR=OFF'])
        run(label+'-build', ['cmake', '--build', directory, '--target', 'crispasr-lib', 'crispasr-cli', '-j4'])
        assert 'dia_tts.cpp.o' in (out/(label+'-build.log')).read_text() and 'Linking' in (out/(label+'-build.log')).read_text()
        dest=out/'libs'/label; dest.mkdir(parents=True)
        for p in directory.rglob('*.so*'):
            if p.is_file() and not p.is_symlink(): shutil.copyfile(p, dest/p.name)
        lib=next(directory.rglob('libcrispasr.so')); cli=directory/'bin/crispasr'
        assert cli.is_file(); run(label+'-ldd', ['ldd', lib])
        return lib, cli
    plib, _ = build(pristine, 'pristine')
    lib, cli = build(ROOT, 'candidate')
    from huggingface_hub import hf_hub_download, snapshot_download
    canonical_dir = scratch/'canonical'
    run('canonical-download', [sys.executable, ROOT/'tools/dia_proof_archives.py', 'canonical-original', canonical_dir])
    canonical=next(canonical_dir.glob('heavy-*'))/'ref.gguf'
    assert sha(canonical) == '3db6f8923abffd612e6c83d7036debebd10ed28b77c3eb9ec3d91fec1ce1a542'
    run('worker-selftest', [sys.executable, ROOT/'tools/ci-heavy/dia-main-head/speech_selftest.py', '--canonical', canonical, '--output', out/'worker-selftest.json'])
    from gguf import GGUFReader
    names={'decoder_codes_rev', 'decoded_audio', 'cross_k0', 'cross_v0', 'decoder_ca0', 'decoder_hidden', 'decoder_step_logits', 'decoder_input_tokens'}
    reference={t.name:np.array(t.data).reshape(tuple(reversed(t.shape))) for t in GGUFReader(str(canonical)).tensors if t.name in names}
    np.save(out/'original-codes.npy', reference['decoder_codes_rev'], allow_pickle=False)
    (out/'reference').mkdir()
    for name, value in reference.items(): np.save(out/'reference'/(name+'.npy'), value, allow_pickle=False)
    original=out/'arms/original'; original.mkdir(parents=True)
    np.save(original/'canonical-pcm.npy', reference['decoded_audio'].reshape(-1), allow_pickle=False)
    weights=Path(snapshot_download('nari-labs/Dia-1.6B', revision='257bc72f9b78182ccc6fa07675a9ae4c1a44e2cd', allow_patterns=['config.json', 'model.safetensors']))
    template=hf_hub_download('cstr/dia-1.6b-GGUF', 'dia-1.6b-f16.gguf', revision='3233fbcb32be47761d2e736857b6d1a075b9ba7e')
    models=scratch/'models'; models.mkdir(); model=models/'dia-1.6b-f32.gguf'
    run('neural-export', [sys.executable, converter/'tools/ci-heavy/dia_neural_precision_control.py', '--export', 'f32', '--weights', weights/'model.safetensors', '--template', template, '--output', model])
    assert sha(model) == MODEL_SHA
    physical=json.loads((out/'neural-f32-physical-audit.json').read_text())
    assert len(physical['tensors'])==381 and all(row['selected'] and row['exported_type']=='F32' for row in physical['tensors'])
    ct=hf_hub_download('cstr/dia-1.6b-GGUF', 'dac-44khz.gguf', revision='3233fbcb32be47761d2e736857b6d1a075b9ba7e')
    codec=models/'dac-44khz.gguf'
    run('codec-export', [sys.executable, ROOT/'models/convert-dia-dac-to-gguf.py', '--template', ct, '--output', codec, '--precision', 'output-f32', '--receipt', out/'physical-codec-audit.json'])
    assert sha(codec) == CODEC_SHA
    asr=Path(hf_hub_download('cstr/nemotron-3.5-asr-streaming-GGUF', 'nemotron-3.5-asr-streaming-0.6b-q4_k.gguf', revision='bbd95a9ca5fa0dfca3312a122dfc45a2b578b9c2'))
    assert sha(asr) == ASR_SHA
    report.update(model_sha256=sha(model), codec_sha256=sha(codec), asr_sha256=sha(asr), library_sha256=sha(lib)); checkpoint()
    blueprint=scratch/'original-dia'
    run('clone-original', ['git', 'clone', '--filter=blob:none', 'https://github.com/nari-labs/dia', blueprint])
    run('pin-original', ['git', '-C', blueprint, 'checkout', '--detach', ORIGINAL])
    shutil.copytree(blueprint/'dia', method/'original-dia', ignore=shutil.ignore_patterns('__pycache__'))
    run('original-tiny-sampling', [sys.executable, ROOT/'tools/ci-heavy/dia-cache-sampler/selftest.py', '--blueprint', blueprint, '--output', out/'original-tiny'])
    for name, path in [('CrispASR-LICENSE', ROOT/'LICENSE'), ('Dia-LICENSE', blueprint/'LICENSE'), ('GGML-LICENSE', ROOT/'ggml/LICENSE')]: shutil.copyfile(path, out/name)
    clean={k:v for k,v in os.environ.items() if not k.startswith(('DIA_', 'CRISPASR_DIA_'))}
    def invoke(kind, arm, model_path=model, audio=None):
        command=[sys.executable, __file__, '--child', kind, '--arm', arm, '--runtime', pristine if arm=='pristine' else ROOT,
                 '--converter', converter, '--blueprint', blueprint, '--model', model_path, '--lib', plib if arm=='pristine' and kind in ['native', 'decode-native', 'auto'] else lib]
        if audio: command+=['--audio', audio]
        run(arm+'-'+kind+('-'+audio if audio else ''), command, clean)
    invoke('dac', 'original')
    old=np.load(original/'canonical-pcm.npy'); new=np.load(original/'original-dac-pcm.npy')
    assert old.shape==new.shape
    error=float(np.linalg.norm(new.astype('f8')-old)/np.linalg.norm(old.astype('f8')))
    report['original_dac_relative_l2']=error; assert error<1e-4
    for audio in ['canonical-pcm.npy', 'original-dac-pcm.npy']:
        invoke('asr', 'original', asr, audio); assert json.loads((original/(audio+'.asr.json')).read_text())['wer']==0
    alias=models/'neutral.gguf'; os.link(model, alias)
    invoke('auto', 'off', alias)
    for arm in ARMS: invoke('native', arm)
    for arm in ['off', 'identity']:
        for name in ['inputs.i32', 'logits.f32', 'native-pcm.npy']:
            assert sha(out/'arms'/arm/name)==sha(out/'arms/pristine'/name), (arm, name)
        files=list((out/'arms/pristine/stages').iterdir())
        assert {p.name for p in files}=={p.name for p in (out/'arms'/arm/'stages').iterdir()}
        for p in files: assert sha(p)==sha(out/'arms'/arm/'stages'/p.name)
    report['pristine_off_identity_exact']=True; checkpoint()
    invoke('asr', 'pristine', asr, 'native-pcm.npy')
    results={'pristine':dict(native_pcm=json.loads((out/'arms/pristine/native-pcm.npy.asr.json').read_text()))}
    for arm in ['off', 'identity']+FACTORIAL:
        directory=out/'arms'/arm
        inputs=np.fromfile(directory/'inputs.i32', dtype='<i4').reshape(-1, 9)
        logits=np.memmap(directory/'logits.f32', mode='r', dtype='<f4').reshape(-1, 2, 9, 1028)
        assert len(inputs)==len(logits) and np.isfinite(logits).all()
        grid=grid_from_inputs(inputs); grid.tofile(directory/'codes.u32')
        assert len(np.load(directory/'native-pcm.npy'))==len(grid)*512
        invoke('decode-native', arm)
        exact(np.load(directory/'native-pcm.npy'), np.load(directory/'decode-native-pcm.npy'))
        invoke('dac', arm)
        expected_inputs=reference['decoder_input_tokens'].astype('<i4')
        common=min(len(inputs),len(expected_inputs))
        different=np.flatnonzero(np.any(inputs[:common]!=expected_inputs[:common],axis=1))
        prefix=int(different[0]) if len(different) else common
        results[arm]=dict(steps=len(inputs), frames=len(grid), native_decode_replay_exact=True,
            original_input_prefix_rows=prefix, original_inputs_exact=bool(np.array_equal(inputs,expected_inputs)),
            original_codes_exact=bool(np.array_equal(grid,reference['decoder_codes_rev'])),
            controls=arm_environment(arm))
        for audio in ['native-pcm.npy', 'original-dac-pcm.npy']:
            invoke('asr', arm, asr, audio); results[arm][audio]=json.loads((directory/(audio+'.asr.json')).read_text())
        if arm in ['off']+FACTORIAL:
            env=clean.copy(); cli_dir=directory/'cli'; cli_dir.mkdir()
            env.update(arm_environment(arm))
            env.update(CRISPASR_DIA_TTS_GPU='0',
                       CRISPASR_DIA_DUMP_INPUT_TOKENS=str(cli_dir/'inputs.i32'), CRISPASR_DIA_DUMP_STEPLOGITS=str(cli_dir/'logits.f32'))
            wav=directory/'cli.wav'
            run(arm+'-cli', [cli, '--no-gpu', '-m', model, '--codec-model', codec, '--tts-output', wav, '-l', 'en',
                '--tts', TEXT, '-t', '4', '--seed', '42', '--temperature', '1.2', '--no-spoken-disclaimer', '--accept-marking-responsibility'], env)
            for name in ['inputs.i32', 'logits.f32']: assert sha(cli_dir/name)==sha(directory/name), (arm, name)
            with wave.open(str(wav), 'rb') as stream:
                assert stream.getnchannels()==1 and stream.getsampwidth()==2 and stream.getframerate()==44100
                pcm=np.frombuffer(stream.readframes(stream.getnframes()), dtype='<i2').astype('f4')/32768
            np.save(directory/'cli-pcm.npy', pcm, allow_pickle=False)
            invoke('asr', arm, asr, 'cli-pcm.npy'); results[arm]['cli-pcm.npy']=json.loads((directory/'cli-pcm.npy.asr.json').read_text())
            results[arm]['cli_session_inputs_logits_exact']=True
        report['results']=results; checkpoint()
    metrics=[]
    for arm in ['off']+FACTORIAL:
        stages=out/'arms'/arm/'stages'
        for name, file, expected in [('cross_k0', 'cpp_cross_k0.f32', reference['cross_k0']),
            ('cross_v0', 'cpp_cross_v0.f32', reference['cross_v0']), ('decoder_ca0', 'cpp_ca0_out.f32', reference['decoder_ca0']),
            ('decoder_hidden', 'cpp_final_hidden.f32', reference['decoder_hidden']),
            ('uncond_logits', 'cpp_step0_uncond.f32', reference['decoder_step_logits'][0,0]),
            ('cond_logits', 'cpp_step0_cond.f32', reference['decoder_step_logits'][0,1])]:
            metrics.append(metric(arm, name, np.fromfile(stages/file, dtype='<f4'), expected))
    report['metrics']=metrics; report['stage_gate_pass']=all(row['pass'] for row in metrics)
    report['speech_pass_by_arm']={arm:all(results[arm][name]['wer']==0 for name in
        ['native-pcm.npy', 'original-dac-pcm.npy', 'cli-pcm.npy']) for arm in FACTORIAL}
    report['f32_portable_both_decoders_and_cli_wer_zero']=report['speech_pass_by_arm']['serial-f32-portable']
    report['complete_diagnostic']=True
    report['payloads']=[dict(path=p.relative_to(out).as_posix(), bytes=p.stat().st_size, sha256=sha(p))
                        for p in sorted(out.rglob('*')) if p.is_file() and p.name!='receipt.json']
    checkpoint()
    print('DIA_CACHE_SAMPLER_FACTORIAL_COMPLETE', json.dumps(report['speech_pass_by_arm']), flush=True)
    assert report['stage_gate_pass'], 'Current-main stage gate failed; complete evidence retained'
    assert report['f32_portable_both_decoders_and_cli_wer_zero'], 'F32 portable speech failed; complete factorial evidence retained'


if __name__ == '__main__':
    main()
