"""Matched original CUDA compute-precision x sampling-recipe diagnostic."""
import hashlib
import importlib.metadata as metadata
import inspect
import io
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import zipfile

VERSION = 'dia-cuda-factorial-v1'
BASE = 'abeee3e8eb5121e0f80bbe8a4a8f6f245fc8fb3f'
ORIGINAL = '4a9e29b1bdbfe1be721353ec034a8c66c9f0a1a8'
WHISPER = '31243bad24cc746f07d4c8bfdd2d974872cb1803'
ASR_SHA = '25a8566e1d0c1e2231d1c762132cd20e0f96a85d16145c3a00adf5d1ac670ead'
DATASET = 'cstr/crispasr-regression-fixtures'
PREFIX = 'dia-20261016-original-cuda/precision-recipe-v1'
SCRATCH = Path('/kaggle/temp/dia-cuda-factorial-v1')
OUT = Path('/kaggle/working/dia-cuda-factorial-v1')
OPTIONS = dict(language='en', task='transcribe', temperature=0.,
               condition_on_previous_text=False, initial_prompt=None, fp16=False, verbose=None)


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def score(text, expected):
    a = re.findall('[a-z]+', re.sub(r'\[S[12]\]', '', expected).lower())
    b = re.findall('[a-z]+', text.lower())
    row = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        new = [i]
        for j, y in enumerate(b, 1):
            new.append(min(new[-1] + 1, row[j] + 1, row[j - 1] + (x != y)))
        row = new
    return row[-1] / len(a)


def validate_install_plan(planned, installed_stack):
    core = {'torch', 'torchaudio', 'torchvision', 'triton', 'pytorch-triton', 'pytorch-triton-rocm'}
    for item in planned:
        name = item['metadata']['name'].lower().replace('_', '-')
        if name in installed_stack or name in core or name.startswith('nvidia-'):
            raise ValueError('Resolver would change installed CUDA/numerical stack: ' + name)


ARMS = {
    'f16-source': dict(precision='float16', temperature=1.3, top_k=35),
    'f16-native-recipe': dict(precision='float16', temperature=1.2, top_k=45),
    'f32-source': dict(precision='float32', temperature=1.3, top_k=35),
    'f32-native-recipe': dict(precision='float32', temperature=1.2, top_k=45),
    'f16-source-anchor': dict(precision='float16', temperature=1.3, top_k=35),
}


def construct_model(Dia, config, state, precision, device):
    dia = Dia(config, compute_dtype=precision, device=device)
    initial = {name: str(p.dtype) for name, p in dia.model.named_parameters()}
    dia.model.load_state_dict(state, strict=True)
    # Normal original constructor and dtype-preserving transfer; never half().
    dia.model.to(dia.device).eval()
    assert initial == {name: str(p.dtype) for name, p in dia.model.named_parameters()}
    assert all(value == ('torch.float32' if precision == 'float32' or 'norm' in name
                         else 'torch.float16') for name, value in initial.items())
    return dia, initial


def generate_case(dia, case, arm):
    return dia.generate(case['text'], max_tokens=3072, use_torch_compile=False,
                        temperature=arm['temperature'], cfg_filter_top_k=arm['top_k'],
                        cfg_scale=3., top_p=.95)


def rng_snapshot(torch):
    return {'cpu': torch.get_rng_state().cpu().numpy().copy(),
            'cuda': torch.cuda.get_rng_state().cpu().numpy().copy()}


def main():
    import torch
    sys.dont_write_bytecode = True
    OUT.mkdir(parents=True, exist_ok=False)
    SCRATCH.mkdir(parents=True, exist_ok=False)
    os.environ.update(HF_HOME=str(SCRATCH / 'hf'), TMPDIR=str(SCRATCH),
                      NUMBA_CACHE_DIR=str(SCRATCH / 'numba'), TIKTOKEN_CACHE_DIR=str(SCRATCH / 'tiktoken'),
                      PYTHONDONTWRITEBYTECODE='1')
    device = torch.cuda.get_device_capability() if torch.cuda.is_available() else (0, 0)
    if device[0] < 7:
        (OUT / 'inconclusive.json').write_text(json.dumps(dict(version=VERSION, conclusive=False,
            reason='No supported CUDA device; no CPU fallback or automatic retry', capability=device)) + '\n')
        print('UNSUPPORTED_CUDA_DRAW_INCONCLUSIVE', device, flush=True)
        return
    # Prove a real installed CUDA kernel works before downloads or installation.
    assert torch.exp(torch.zeros(1, device='cuda')).item() == 1
    torch.cuda.synchronize()
    (OUT / 'gpu-preflight.json').write_text(json.dumps(dict(version=VERSION,
        torch=torch.__version__, cuda=torch.version.cuda, capability=device,
        gpu=torch.cuda.get_device_name(), actual_cuda_probe_passed=True), indent=2) + '\n')
    clone = SCRATCH / 'CrispASR'
    subprocess.run(['git', 'clone', '--filter=blob:none', 'https://github.com/CrispStrobe/CrispASR', clone], check=True)
    subprocess.run(['git', '-C', clone, 'checkout', '--detach', BASE], check=True)
    assert subprocess.check_output(['git', '-C', clone, 'rev-parse', 'HEAD'], text=True).strip() == BASE
    sys.path.insert(0, str(clone / 'tools/kaggle'))
    import kaggle_harness as kh
    kh.init_progress(OUT / 'progress.jsonl')
    kh.resolve_hf_token(require=True)
    kh.provenance(VERSION, clone)
    # Preserve the preinstalled CUDA stack. Abort a resolver proposal that would
    # install Torch, torchaudio, torchvision, Triton or NVIDIA runtime packages.
    constraints = SCRATCH / 'cuda-stack.txt'
    stack = {}
    for name in ['torch', 'torchaudio', 'torchvision', 'triton', 'numpy']:
        try:
            stack[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            pass
    assert 'torch' in stack and 'torchaudio' in stack
    constraints.write_text(''.join(name + '==' + value + '\n' for name, value in stack.items()))
    packages = ['descript-audio-codec==1.0.0', 'openai-whisper==20250625', 'librosa==0.11.0',
                'safetensors', 'pydantic>=2.11.3', 'huggingface_hub', 'soundfile']
    command = [sys.executable, '-m', 'pip', 'install', '-c', str(constraints), *packages]
    proposal = OUT / 'pip-plan.json'
    with (OUT / 'pip-resolve.log').open('xb') as stream:
        subprocess.run(command + ['--dry-run', '--report', str(proposal)], stdout=stream, stderr=subprocess.STDOUT, check=True)
    planned = json.loads(proposal.read_text())['install']
    validate_install_plan(planned, stack)
    with (OUT / 'pip-install.log').open('xb') as stream:
        subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=True)
    assert all(metadata.version(name) == value for name, value in stack.items())
    import numpy as np
    import soundfile as sf
    from huggingface_hub import HfApi, hf_hub_download, snapshot_download
    from safetensors.torch import load_file
    (OUT / 'ffmpeg-version.log').write_bytes(subprocess.check_output(['ffmpeg', '-version']))
    sys.path.insert(0, str(clone / 'tools/ci-heavy'))
    import dia_direct_original as support
    source = SCRATCH / 'original-dia'
    subprocess.run(['git', 'clone', '--filter=blob:none', 'https://github.com/nari-labs/dia', source], check=True)
    subprocess.run(['git', '-C', source, 'checkout', '--detach', ORIGINAL], check=True)
    sys.path.insert(0, str(source))
    from dia.config import DiaConfig
    from dia.model import Dia
    import dac
    method = OUT / 'method'
    method.mkdir()
    shutil.copyfile(__file__, method / 'uploaded-kernel.py')
    shutil.copyfile(clone / 'tools/ci-heavy/dia_direct_original.py', method / 'dia_direct_original.py')
    shutil.copyfile(clone / 'tools/ci-heavy/dia_asr_crosscheck.py', method / 'dia_asr_crosscheck.py')
    shutil.copyfile(clone / 'LICENSE', method / 'CrispASR-LICENSE')
    shutil.copytree(source / 'dia', method / 'original-dia', ignore=shutil.ignore_patterns('__pycache__'))
    for name in ['README.md', 'LICENSE']:
        shutil.copyfile(source / name, method / ('Dia-' + name))
    defaults = {key: value.default for key, value in inspect.signature(Dia.generate).parameters.items()
                if key in ['temperature', 'top_p', 'cfg_filter_top_k', 'cfg_scale']}
    assert defaults == dict(temperature=1.3, top_p=.95, cfg_filter_top_k=35, cfg_scale=3.)
    weights = Path(snapshot_download('nari-labs/Dia-1.6B', revision=support.WEIGHTS_REV,
        allow_patterns=['config.json', 'model.safetensors']))
    assert sha(weights / 'model.safetensors') == support.WEIGHTS_SHA
    shutil.copyfile(weights / 'config.json', method / 'original-config.json')
    config = DiaConfig.load(str(weights / 'config.json'))
    state = load_file(str(weights / 'model.safetensors'), device='cpu')
    checkpoint = Path(dac.utils.download())
    assert sha(checkpoint) == support.DAC_SHA
    codec = dac.DAC.load(str(checkpoint)).to(torch.device('cuda')).eval()
    codec_hooks = support.codec_hooks(codec)
    assert any(hooks for _, hooks in codec_hooks)
    shutil.copytree(Path(dac.__file__).parent, method / 'original-dac',
                    ignore=shutil.ignore_patterns('__pycache__', '*.pt', '*.pth'))
    report = dict(version=VERSION, uploaded_script_sha256=sha(method / 'uploaded-kernel.py'),
        harness_source=BASE, original_source=ORIGINAL, model_revision=support.WEIGHTS_REV,
        weights_sha256=support.WEIGHTS_SHA, config_sha256=sha(weights / 'config.json'),
        dac_sha256=support.DAC_SHA, torch=torch.__version__, cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(), capability=device, installed_cuda_stack=stack,
        arm_plan=ARMS, arms={}, compute_dtype='Per-arm original float16 or float32; F32 norms preserved',
        defaults=defaults, capacity=3072, cases=support.CASES, added_observers=False,
        original_weightnorm_hooks=sum(len(h) for _, h in codec_hooks),
        methods={name: hashlib.sha256(inspect.getsource(getattr(Dia, name)).encode()).hexdigest()
                 for name in ['__init__', 'generate', '_decoder_step', '_generate_output']},
        results={}, controls={}, complete=False, speech_acceptance=False,
        native_acceptance=False, q4_acceptance=False, production_defaults_changed=False)

    def save():
        report['payloads'] = [dict(path=p.relative_to(OUT).as_posix(), bytes=p.stat().st_size, sha256=sha(p))
                              for p in sorted(OUT.rglob('*')) if p.is_file() and p != OUT / 'receipt.json'
                              and p.name != 'progress.jsonl']
        (OUT / 'receipt.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')

    api = HfApi()
    assert not api.repo_info(DATASET, repo_type='dataset').private
    assert not api.file_exists(DATASET, PREFIX + '/receipt.json', repo_type='dataset')

    def checkpoint_proof(stage):
        report['last_completed_stage'] = stage
        save()
        commit = api.upload_folder(repo_id=DATASET, repo_type='dataset', folder_path=str(OUT),
            path_in_repo=PREFIX, ignore_patterns=['progress.jsonl'],
            commit_message='Retain original Dia precision-recipe diagnostic: ' + stage)
        print('CHECKPOINT_PROOF_REVISION', stage, commit.oid, flush=True)
        return commit.oid

    try:
        for arm_name, arm in ARMS.items():
            dia, dtypes = construct_model(Dia, config, state, arm['precision'], torch.device('cuda'))
            assert all(p.is_cuda for p in dia.model.parameters())
            dia.dac_model = codec
            support.no_hooks(dia.model)
            assert support.codec_hooks(codec) == codec_hooks
            report['arms'][arm_name] = dict(parameters=arm, parameter_dtypes=dtypes)
            for key, case in support.CASES.items():
                torch.manual_seed(case['seed'])
                before = rng_snapshot(torch)
                torch.cuda.reset_peak_memory_stats()
                with torch.inference_mode():
                    pcm = generate_case(dia, case, arm)
                torch.cuda.synchronize()
                after = rng_snapshot(torch)
                pcm = np.asarray(pcm)
                support.compare_pcm(pcm, pcm)
                support.no_hooks(dia.model)
                assert support.codec_hooks(codec) == codec_hooks
                label = arm_name + '/' + key
                folder = OUT / label
                folder.mkdir(parents=True)
                np.save(folder / 'pcm.npy', pcm, allow_pickle=False)
                sf.write(folder / 'input.wav', pcm, 44100, subtype='FLOAT')
                for when, states in [('before', before), ('after', after)]:
                    for kind, values in states.items():
                        np.save(folder / ('rng-' + when + '-' + kind + '.npy'), values, allow_pickle=False)
                report['results'][label] = dict(arm=arm_name, case=key, samples=len(pcm),
                    rms=float(np.sqrt(np.mean(pcm.astype('f8') ** 2))),
                    cuda_peak_allocated=torch.cuda.max_memory_allocated(), seed=case['seed'])
                save()
                kh.step('original.cuda.factorial.saved', arm=arm_name, case=key, samples=len(pcm))
            # Publish this arm before the next constructor or inference can fail.
            checkpoint_proof(arm_name)
            del dia
            torch.cuda.empty_cache()
        del codec, state
        torch.cuda.empty_cache()
        # Same sealed recognizer controls as the independent CPU diagnosis.
        parent = Path(hf_hub_download(DATASET, 'dia-20261016-broader-speech/crossed-asr-v1/raw.zip',
            repo_type='dataset', revision='951fa9f714dd2fc5346f8e5e9eadc0fdf9f3fed6'))
        assert sha(parent) == support.CROSSED_SHA
        with zipfile.ZipFile(parent) as archive:
            reference = json.loads(archive.read('receipt.json'))
            index = {row['path']: row for row in reference['payloads']}
            for key in ['legacy', 'jfk']:
                data = archive.read('audio/' + key + '.npy')
                assert hashlib.sha256(data).hexdigest() == index['audio/' + key + '.npy']['sha256']
                pcm = np.load(io.BytesIO(data), allow_pickle=False)
                folder = OUT / ('control-' + key)
                folder.mkdir()
                (folder / 'pcm.npy').write_bytes(data)
                sf.write(folder / 'input.wav', pcm, 16000 if key == 'jfk' else 44100, subtype='FLOAT')
        whisper_source = SCRATCH / 'original-whisper'
        subprocess.run(['git', 'clone', '--filter=blob:none', 'https://github.com/openai/whisper', whisper_source], check=True)
        subprocess.run(['git', '-C', whisper_source, 'checkout', '--detach', WHISPER], check=True)
        sys.path.insert(0, str(whisper_source))
        import whisper
        assert Path(whisper.__file__).resolve().parent == whisper_source / 'whisper'
        shutil.copytree(whisper_source / 'whisper', method / 'original-whisper', ignore=shutil.ignore_patterns('__pycache__'))
        shutil.copyfile(whisper_source / 'LICENSE', method / 'Whisper-LICENSE')
        recognizer = whisper.load_model('base.en', device='cuda', download_root=str(SCRATCH / 'whisper-model')).eval()
        assert sha(SCRATCH / 'whisper-model/base.en.pt') == ASR_SHA and next(recognizer.parameters()).is_cuda
        report['recognizer'] = dict(source=WHISPER, model_sha256=ASR_SHA, options=OPTIONS, device='cuda')
        for label in ['control-legacy', 'control-jfk', *report['results']]:
            folder = OUT / label
            actual = whisper.load_audio(str(folder / 'input.wav'))
            np.save(folder / 'actual-16khz.npy', actual, allow_pickle=False)
            result = recognizer.transcribe(actual, **OPTIONS)
            (folder / 'raw-whisper.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
            key = label.removeprefix('control-') if label.startswith('control-') else report['results'][label]['case']
            row = dict(transcript=result['text'].strip(), wer=score(result['text'], support.crossed.TEXTS[key]))
            if label.startswith('control-'):
                report['controls'][key] = row
            else:
                report['results'][label].update(row)
            save()
            kh.step('recognition.cuda.done', case=label, wer=row['wer'])
        report['complete'] = True
        report['controls_wer_zero'] = all(r['wer'] == 0 for r in report['controls'].values())
        report['all_wer_zero'] = all(r['wer'] == 0 for r in report['results'].values())
        report['anchor_quality_matches'] = all(report['results']['f16-source/' + key]['wer'] ==
            report['results']['f16-source-anchor/' + key]['wer'] for key in support.CASES)
        report['source_fox_controls_pass'] = all(report['results'][arm + '/' + key]['wer'] == 0
            for arm in ['f16-source', 'f16-source-anchor'] for key in ['fox-42', 'fox-123'])
    finally:
        checkpoint_proof('final' if report['complete'] else 'failed-partial')
    assert report['complete'] and report['controls_wer_zero'] and report['anchor_quality_matches'] and report['source_fox_controls_pass'], 'Factorial control gate failed; all proof retained'
    print('MATCHED_DIAGNOSTIC_COMPLETE_SPEECH_ACCEPTANCE_FALSE', report['all_wer_zero'], flush=True)


if __name__ == '__main__':
    main()
