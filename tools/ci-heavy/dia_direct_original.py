#!/usr/bin/env python3
"""Direct original Dia constructor/generate control, without observation hooks."""
import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import subprocess
import sys
import zipfile

import numpy as np
import dia_asr_crosscheck as crossed

ROOT = Path(__file__).resolve().parents[2]
ORIGINAL = '4a9e29b1bdbfe1be721353ec034a8c66c9f0a1a8'
WEIGHTS_REV = '257bc72f9b78182ccc6fa07675a9ae4c1a44e2cd'
WEIGHTS_SHA = 'caba289b60f6d7d1e58fc744f4dc25aae88995fcca46be3d05e220b971486a26'
DAC_SHA = 'a88eed82a7024ccc1facdb1e605c4c2f99281c8118c22c9895ffa846d8fb61aa'
CROSSED_SOURCE = '30e75dcdd8df573ed1f73e9c97b17f192b82d7af'
CROSSED_SHA = 'fdaa4da6e39a86ca52ddca48de50bc8c93de77f8d7bcfd906aab765517fee416'
CASES = {k: dict(text=v, seed=123 if k.endswith('123') else 42)
         for k, v in crossed.TEXTS.items() if k != 'jfk'}
sys.dont_write_bytecode = True


def compare_pcm(actual, captured):
    for value in [actual, captured]:
        if value.dtype != np.float32 or value.ndim != 1 or not value.size or not np.isfinite(value).all():
            raise ValueError('PCM must be a nonempty finite F32 vector')
    same_shape = actual.shape == captured.shape
    return dict(actual_samples=actual.size, captured_samples=captured.size,
                exact=bool(same_shape and actual.tobytes() == captured.tobytes()),
                rms=float(np.sqrt(np.mean(actual.astype('f8') ** 2))),
                max_abs_error=float(np.max(abs(actual.astype('f8') - captured))) if same_shape else None)


def no_hooks(model):
    for module in model.modules():
        if module._forward_hooks or module._forward_pre_hooks or module._backward_hooks:
            raise ValueError('Observer hooks present in direct original control')


def codec_hooks(model):
    """Preserve original DAC WeightNorm pre-hooks; reject added observers."""
    from torch.nn.utils.weight_norm import WeightNorm
    signature = []
    for name, module in model.named_modules():
        if module._forward_hooks or module._backward_hooks:
            raise ValueError('Observer hooks present in original DAC')
        hooks = tuple(module._forward_pre_hooks.items())
        if any(not isinstance(hook, WeightNorm) for _, hook in hooks):
            raise ValueError('Unexpected DAC pre-hook instead of original WeightNorm')
        signature.append((name, hooks))
    return tuple(signature)


def selftest():
    x = np.array([.25, -.5, 0], dtype='f4')
    assert compare_pcm(x, x.copy())['exact']
    assert not compare_pcm(x * 2, x)['exact']
    assert not compare_pcm(x[:1], x)['exact']
    for invalid in [x.astype('f8'), x.reshape(1, -1), x[:0], np.array([np.nan], dtype='f4')]:
        try:
            compare_pcm(invalid, x)
        except ValueError:
            pass
        else:
            raise AssertionError('Invalid PCM accepted')
    assert len(CASES) == 5 and CASES['legacy']['seed'] == 42
    print('DIRECT_ORIGINAL_COMPARISON_CONTROLS_PASS')


def generate(args):
    import torch
    import dac
    from safetensors.torch import load_file
    sys.path.insert(0, str(args.blueprint))
    from dia.config import DiaConfig
    from dia.model import Dia
    assert torch.__version__ == '2.6.0+cpu'
    torch.set_num_threads(4)
    assert crossed.sha(args.weights / 'model.safetensors') == WEIGHTS_SHA
    assert crossed.sha(args.dac) == DAC_SHA
    config = DiaConfig.load(str(args.weights / 'config.json'))
    # Actual original constructor: allocate real F32 parameters and real rotary
    # buffers. No meta construction, __new__, assign=True or buffer replacement.
    dia = Dia(config, compute_dtype='float32', device=torch.device('cpu'))
    state = load_file(str(args.weights / 'model.safetensors'), device='cpu')
    dia.model.load_state_dict(state, strict=True)
    del state
    dia.model.eval()
    dia.dac_model = dac.DAC.load(str(args.dac)).cpu().eval()
    assert dia.compute_dtype == torch.float32
    assert all(p.dtype == torch.float32 for p in dia.model.parameters())
    assert type(dia).generate is Dia.generate and '_generate_output' not in dia.__dict__
    no_hooks(dia.model)
    original_codec_hooks = codec_hooks(dia.dac_model)
    assert any(hooks for _, hooks in original_codec_hooks), "Expected actual original DAC WeightNorm hooks"
    provenance = dict(source=ORIGINAL, torch_version=torch.__version__, weights_sha256=WEIGHTS_SHA,
        dac_sha256=DAC_SHA, config_sha256=crossed.sha(args.weights / 'config.json'),
        methods={name: hashlib.sha256(inspect.getsource(getattr(Dia, name)).encode()).hexdigest()
                 for name in ['__init__', 'generate', '_decoder_step', '_generate_output']},
        environment={k: os.environ.get(k) for k in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'MKL_CBWR', 'ATEN_CPU_CAPABILITY']},
        constructor='actual Dia(config, compute_dtype=float32, device=cpu)',
        observation_hooks=False, original_codec_weightnorm_hooks=sum(len(h) for _, h in original_codec_hooks),
        meta=False, assign=False, buffer_replacement=False, cases=CASES)
    (args.output / 'original-provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    for key, case in CASES.items():
        torch.manual_seed(case['seed'])
        with torch.inference_mode():
            pcm = dia.generate(case['text'], max_tokens=3072, cfg_scale=3., temperature=1.2,
                               top_p=.95, cfg_filter_top_k=45, use_torch_compile=False)
        assert pcm is not None
        pcm = np.asarray(pcm)
        compare_pcm(pcm, pcm)
        no_hooks(dia.model)
        assert codec_hooks(dia.dac_model) == original_codec_hooks, "Original DAC WeightNorm hooks changed"
        np.save(args.output / (key + '.npy'), pcm, allow_pickle=False)
        print('DIRECT_ORIGINAL_SAVED', key, pcm.size, flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--selftest', action='store_true')
    p.add_argument('--generate', action='store_true')
    for name in ['blueprint', 'weights', 'dac', 'output']:
        p.add_argument('--' + name, type=Path)
    args = p.parse_args()
    if args.selftest:
        selftest()
        return
    if args.generate:
        generate(args)
        return
    selftest()
    out, scratch = Path(os.environ['HEAVY_OUT']), Path(os.environ['HEAVY_SCRATCH'])
    out.mkdir(exist_ok=True, parents=True)
    scratch.mkdir(exist_ok=True, parents=True)
    os.environ.update(TMPDIR=str(scratch), PYTHONDONTWRITEBYTECODE='1')
    source = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    assert source == os.environ['GITHUB_SHA'] and not subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT)
    report = dict(source=source, original=ORIGINAL, cases=CASES, runs=[], results={}, complete=False,
                  speech_acceptance=False, gpu_acceptance=False, q4_acceptance=False, production_defaults_changed=False)

    def checkpoint():
        report['payloads'] = [dict(path=f.relative_to(out).as_posix(), bytes=f.stat().st_size, sha256=crossed.sha(f))
                             for f in sorted(out.rglob('*')) if f.is_file() and not f.is_symlink() and f != out / 'receipt.json']
        (out / 'receipt.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')

    def run(label, command, env=None):
        with (out / (label + '.log')).open('xb') as log:
            result = subprocess.run(list(map(str, command)), cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=14400)
        report['runs'].append(dict(label=label, exit=result.returncode))
        checkpoint()
        assert result.returncode == 0, label

    try:
        method = out / 'method'
        method.mkdir()
        for name in ['dia_direct_original.py', 'dia_asr_crosscheck.py']:
            shutil.copyfile(Path(__file__).parent / name, method / name)
        shutil.copytree(Path(__file__).parent / 'dia-direct-original', method / 'dia-direct-original')
        shutil.copyfile(ROOT / 'tests/test_dia_reference_capture.py', method / 'test_dia_reference_capture.py')
        for name in ['dia_tts_reference.py']:
            shutil.copyfile(ROOT / 'tools/reference_backends' / name, method / name)
        for name in ['dia_attention_capture.py', 'dia_decoder_capture.py']:
            shutil.copyfile(ROOT / 'tools' / name, method / name)
        shutil.copyfile(ROOT / 'python/crispasr/_binding.py', method / '_binding.py')
        assert (method / 'dia_asr_crosscheck.py').read_bytes() == subprocess.check_output(
            ['git', 'show', CROSSED_SOURCE + ':tools/ci-heavy/dia_asr_crosscheck.py'], cwd=ROOT)
        bundle = scratch / 'crossed.zip'
        metadata = subprocess.check_output(['gh', 'api', 'repos/CrispStrobe/CrispASR/actions/artifacts/12060029304'])
        (out / 'parent-artifact.json').write_bytes(metadata)
        meta = json.loads(metadata)
        assert meta['digest'] == 'sha256:' + CROSSED_SHA
        assert meta['workflow_run']['id'] == 38930760374 and meta['workflow_run']['head_sha'] == CROSSED_SOURCE
        with bundle.open('xb') as f:
            subprocess.run(['gh', 'api', 'repos/CrispStrobe/CrispASR/actions/artifacts/12060029304/zip'], stdout=f, check=True)
        assert crossed.sha(bundle) == CROSSED_SHA and bundle.stat().st_size == meta['size_in_bytes']
        with zipfile.ZipFile(bundle) as z:
            assert len(z.namelist()) == len(set(z.namelist()))
            for entry in z.infolist():
                part = PurePosixPath(entry.filename)
                assert not part.is_absolute() and '..' not in part.parts and '\\' not in entry.filename
                assert not stat.S_ISLNK(entry.external_attr >> 16)
            z.extractall(out / 'parent')
        parent = json.loads((out / 'parent/receipt.json').read_text())
        assert parent['source'] == CROSSED_SOURCE and parent['complete'] and parent['valid_recognizer_controls']
        for row in parent['payloads']:
            f = out / 'parent' / row['path']
            assert f.stat().st_size == row['bytes'] and crossed.sha(f) == row['sha256']
        blueprint = scratch / 'original-dia'
        run('clone-original', ['git', 'clone', '--filter=blob:none', 'https://github.com/nari-labs/dia', blueprint])
        run('pin-original', ['git', '-C', blueprint, 'checkout', '--detach', ORIGINAL])
        shutil.copytree(blueprint / 'dia', method / 'original-dia', ignore=shutil.ignore_patterns('__pycache__'))
        shutil.copyfile(blueprint / 'LICENSE', out / 'Dia-LICENSE')
        shutil.copyfile(ROOT / 'LICENSE', out / 'CrispASR-LICENSE')
        shutil.copyfile(ROOT / 'ggml/LICENSE', out / 'GGML-LICENSE')
        tiny_env = dict(os.environ, DIA_SOURCE=str(blueprint))
        run('real-tiny-original-control', [sys.executable, ROOT / 'tools/ci-heavy/dia-direct-original/selftest.py'], tiny_env)
        from huggingface_hub import snapshot_download, hf_hub_download
        weights = Path(snapshot_download('nari-labs/Dia-1.6B', revision=WEIGHTS_REV,
            allow_patterns=['config.json', 'model.safetensors']))
        assert crossed.sha(weights / 'model.safetensors') == WEIGHTS_SHA
        import dac
        checkpoint_path = Path(dac.utils.download())
        assert crossed.sha(checkpoint_path) == DAC_SHA
        destination = out / 'direct'
        destination.mkdir()
        clean = {k: v for k, v in os.environ.items() if not k.startswith(('DIA_', 'CRISPASR_DIA_'))}
        clean.update(OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4',
                     MKL_CBWR='COMPATIBLE', ATEN_CPU_CAPABILITY='default')
        run('direct-original', [sys.executable, __file__, '--generate', '--blueprint', blueprint,
            '--weights', weights, '--dac', checkpoint_path, '--output', destination], clean)
        model = Path(hf_hub_download('cstr/nemotron-3.5-asr-streaming-GGUF',
            'nemotron-3.5-asr-streaming-0.6b-q4_k.gguf', revision=crossed.ASR_REV))
        assert crossed.sha(model) == crossed.ASR_SHA
        libdir = out / 'parent/libraries/old'
        for file in list(libdir.iterdir()):
            import re
            dynamic = subprocess.check_output(['readelf', '-d', file], text=True)
            soname = re.search(r'\(SONAME\).*\[([^\]]+)\]', dynamic)
            if soname and not (libdir / soname[1]).exists():
                (libdir / soname[1]).symlink_to(file.name)
        libraries = [f for f in libdir.iterdir() if f.is_file() and not f.is_symlink()
                     and crossed.sha(f) == crossed.PINS['old']['library_sha']]
        assert len(libraries) == 1
        for key, case in CASES.items():
            pcm_path = destination / (key + '.npy')
            row = compare_pcm(np.load(pcm_path, allow_pickle=False), np.load(out / 'parent/audio' / (key + '.npy'), allow_pickle=False))
            asr_out = destination / (key + '-asr.json')
            env = {k: v for k, v in clean.items() if not k.startswith(('LD_PRELOAD', 'LD_LIBRARY_PATH'))}
            env['LD_LIBRARY_PATH'] = str(libdir.resolve())
            run(key + '-asr', [sys.executable, Path(crossed.__file__), '--child', 'old', '--case', key,
                '--library', libraries[0], '--model', model, '--audio', pcm_path, '--output', asr_out], env)
            row['asr'] = json.loads(asr_out.read_text())
            row['captured_asr'] = parent['results']['old'][key]
            report['results'][key] = row
            checkpoint()
        report['complete'] = set(report['results']) == set(CASES)
        report['all_pcm_exact'] = all(r['exact'] for r in report['results'].values())
        report['all_wer_zero'] = all(r['asr']['wer'] == 0 for r in report['results'].values())
        assert report['results']['legacy']['asr']['wer'] == 0, 'Previously accepted direct-original control failed'
        # Direct-original identity is a diagnostic gate, not speech acceptance.
        assert report['all_pcm_exact'], 'Direct constructor/no-hook output differs; retain all controls'
    finally:
        checkpoint()
    print('DIA_DIRECT_ORIGINAL_COMPLETE', report['all_pcm_exact'], report['all_wer_zero'], flush=True)


if __name__ == '__main__':
    main()
