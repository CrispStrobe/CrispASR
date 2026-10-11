#!/usr/bin/env python3
"""Compare pinned original Breeze decoded speech with retained native surfaces.

Original BF16 full-generation oracle with production sampling parameters.
Model and ASR phases run in separate processes; no original-code gate waiver.
Dispatch only after the native speech worker has uploaded its artifact.
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tarfile
import wave

from breeze_guided_reference import UPSTREAM, MODEL, DUMPER_SHA

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(os.environ['HEAVY_OUT'])
SCRATCH = Path(os.environ['HEAVY_SCRATCH']) / 'breeze-original-speech'
NATIVE_SOURCE = 'a2f58338ef9a86dfee114cdd5cd610dfe7b556c6'
PACKAGE_RUN = 38156833241
PACKAGE_SOURCE = '3fdc1838dc1b95f517a3862a9379b24d20d6c0c1'
PACKAGE_SHA = 'fee8a089191385c75840b522fb9709828a84af7c99abc6bcb261d3224c245b94'
ASR_REV = 'bbd95a9ca5fa0dfca3312a122dfc45a2b578b9c2'
ASR_SHA = '14cbc98df495d8ab452c6a4ac367136d94ab5da55e73afaa96ba4f749dd95d09'
TEXT = 'I found the file you asked about and opened it.'
REF_TEXT = 'The quick brown fox jumps over the lazy dog.'
INSTRUCTION = 'Speak clearly and calmly.'
RATE, SEED, LIMIT = 24000, 7, 128
CASES = [('plain', False, False), ('clone', True, False),
         ('design', False, True), ('direction', True, True)]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def run(command, label, timeout=7200):
    with (OUT / (label+'.log')).open('w') as log:
        subprocess.run(list(map(str, command)), cwd=ROOT, stdout=log,
                       stderr=subprocess.STDOUT, check=True, timeout=timeout)


def wer(expected, actual):
    words = lambda value: re.findall('[a-z]+', value.lower())
    target, heard = words(expected), words(actual)
    previous = list(range(len(heard)+1))
    for index, word in enumerate(target, 1):
        row = [index]
        for j, other in enumerate(heard, 1):
            row.append(min(row[-1]+1, previous[j]+1, previous[j-1]+(word != other)))
        previous = row
    return previous[-1]/len(target)


def original():
    import numpy as np
    import torch
    receipt = json.loads((OUT/'validation.json').read_text())
    native = OUT/'native'
    reference = np.load(native/'reference.npy', allow_pickle=False)
    # Reproduce the exact PCM16 reference used by the native clone setter/CLI.
    voice = SCRATCH/'reference.wav'
    with wave.open(str(voice), 'wb') as wav:
        wav.setparams((1, 2, RATE, 0, 'NONE', 'not compressed'))
        wav.writeframes((np.clip(reference, -1, 1)*32767).astype('<i2').tobytes())
    receipt['reference_wav_sha256'] = sha(voice)
    upstream = SCRATCH/'upstream'
    run(['git', 'clone', '--no-checkout', 'https://github.com/breezeblue-ai/breeze-tts.git', upstream], 'clone')
    run(['git', '-C', upstream, 'checkout', '--detach', UPSTREAM], 'checkout')
    assert subprocess.check_output(['git', '-C', upstream, 'rev-parse', 'HEAD'], text=True).strip() == UPSTREAM
    dumper = ROOT/'tools/kaggle/breeze-refdump/breeze_refdump.py'
    assert sha(dumper) == DUMPER_SHA
    code = dumper.read_text().split('# ── checkpoint ─', 1)[1]
    code = code[code.index('from huggingface_hub import snapshot_download'):]
    code = code.split('# ── reference audio', 1)[0]
    needle = 'snapshot_download(HF_MODEL, local_dir=str(CKPT), token=HF_TOKEN or None,'
    assert code.count(needle) == 1
    code = code.replace(needle, needle+'\n                  revision=MODEL_REVISION,')
    (OUT/'executed-original-loading.py').write_text(code)
    torch.set_grad_enabled(False)
    torch.set_num_threads(4)
    env = dict(__name__='__main__', os=os, sys=sys, Path=Path, np=np, torch=torch,
        BREEZE=upstream, CKPT=SCRATCH/'checkpoint', HF_MODEL='BreezeBlue/Breeze-TTS-2',
        HF_TOKEN=os.environ.get('HF_TOKEN', ''), MODEL_REVISION=MODEL, DEVICE='cpu',
        GREEDY=False, MAX_FRAMES=LIMIT,
        step=lambda name, **kw: print(name, json.dumps(kw), flush=True))
    exec(compile(code, str(dumper), 'exec'), env)
    model, codec = env['model'], env['audio_tokenizer']
    assert next(model.parameters()).dtype == torch.bfloat16
    receipt['original_runtime'] = dict(dtype='BF16', attention='eager', device='CPU',
        packages={p: importlib.metadata.version(p) for p in
                  ('torch', 'torchaudio', 'transformers', 'qwen-tts', 'numpy')},
        cpu_capability=torch.backends.cpu.get_cpu_capability(),
        cpuinfo=Path('/proc/cpuinfo').read_text(),
        checkpoint_sha256={str(p.relative_to(SCRATCH/'checkpoint')): sha(p)
                           for p in sorted((SCRATCH/'checkpoint').rglob('*.safetensors'))})
    receipt['original_cases'] = {}
    def save():
        (OUT/'validation.json').write_text(json.dumps(receipt, indent=2)+'\n')
    save()
    for label, ref, ins in CASES:
        request = dict(id=label, text=TEXT, speaker='S0')
        if ref:
            request.update(ref_audio_path=str(voice), ref_text=REF_TEXT)
        if ins:
            request['instruction'] = INSTRUCTION
        template = env['select_template_name'](request)
        scale = 4.0 if ins else 1.0
        inputs = env['prepare_inputs'](env['tokenizer'], codec, model, [request],
            env['get_template'](template), guidance_scale=scale,
            guidance_scale_ref=None, guidance_scale_ins=None)
        for name, value in inputs.items():
            if torch.is_tensor(value):
                np.save(OUT/f'original-{label}-{name}.npy', value.detach().cpu().numpy())
        env['set_all_seeds'](SEED)
        generated = model.generate(**{k: v for k, v in inputs.items() if v is not None},
            output_audio=True, audio_tokenizer=codec, return_dict_in_generate=True,
            max_new_tokens=None, max_length=int(inputs['input_ids'].shape[1])+LIMIT,
            do_sample=True, temperature=.9, top_k=50, top_p=1., repetition_penalty=1.1)
        codes = generated.sequences[0].detach().cpu().numpy().astype(np.int32)
        assert codes.ndim == 2 and codes.shape[1] == 16
        pads = (codes == int(model.config.codebook_pad_token_id)).all(axis=1)
        if pads.any():
            codes = codes[:int(np.argmax(pads))]
        assert 0 < len(codes) <= LIMIT and ((codes >= 0) & (codes < 2048)).all()
        assert generated.audio is not None and len(generated.audio) == 1
        value = generated.audio[0]
        pcm = (value.detach().float().cpu().numpy() if torch.is_tensor(value)
               else np.asarray(value, dtype=np.float32)).reshape(-1)
        assert pcm.size > RATE and np.isfinite(pcm).all()
        rms = float(np.sqrt(np.mean(pcm.astype(np.float64)**2)))
        assert rms > .001, (label, rms)
        np.save(OUT/f'original-{label}-codes.npy', codes)
        np.save(OUT/f'original-{label}.npy', pcm)
        receipt['original_cases'][label] = dict(template=template, cfg_scale=scale,
            frames=len(codes), samples=len(pcm), duration=len(pcm)/RATE, rms=rms,
            pcm_sha256=sha(OUT/f'original-{label}.npy'), codes_sha256=sha(OUT/f'original-{label}-codes.npy'))
        save()
        print(label, receipt['original_cases'][label], flush=True)
    receipt['original_generation_complete'] = True
    save()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--native-run', type=int)
    parser.add_argument('--phase', choices=['original'])
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    SCRATCH.mkdir(parents=True, exist_ok=True)
    os.environ.update(TMPDIR=str(SCRATCH), OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='4')
    if args.phase:
        return original()
    assert args.native_run and args.native_run > 0
    receipt = dict(source=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        native_run=args.native_run, native_source=NATIVE_SOURCE, upstream=UPSTREAM,
        model_revision=MODEL, reviewed_loader_sha256=DUMPER_SHA, complete=False,
        original_generated_code_gate_waived=False, expected=TEXT, reference_text=REF_TEXT,
        instruction=INSTRUCTION, rate=RATE, seed=SEED, max_frames=LIMIT,
        recipe='Original BF16 full generate; production sampling, eager CPU; not fast-streaming timing',
        sampler=dict(backbone_temperature=.9, depth_temperature=.9, top_k=50, top_p=1., repetition_penalty=1.1),
        failures=[])
    def save():
        (OUT/'validation.json').write_text(json.dumps(receipt, indent=2)+'\n')
    save()
    try:
        info = json.loads(subprocess.check_output(['gh', 'api',
            f'repos/CrispStrobe/CrispASR/actions/runs/{args.native_run}'], text=True))
        assert info['head_sha'] == NATIVE_SOURCE and info['status'] == 'completed'
        native = SCRATCH/'native-artifact'
        run(['gh', 'run', 'download', args.native_run, '-R', 'CrispStrobe/CrispASR',
             '-n', f'heavy-{args.native_run}', '-D', native], 'native-download')
        prior = json.loads((native/'validation.json').read_text())
        assert prior['source'] == NATIVE_SOURCE and prior['complete']
        assert (prior['expected'], prior['instruction'], prior['seed'], prior['max_frames']) == (TEXT, INSTRUCTION, SEED, LIMIT)
        assert prior['models']['breeze']['revision'] == '5314fe9ae3ec6eac1a9238c4f6888fdc9fda0e83'
        assert prior['models']['breeze']['sha256'] == '08ea3fbffe9701a3d04a4a63554876841d923fdd21e20add36d28f3ac3447da7'
        assert prior['models']['codec']['sha256'] == '70dc95dbfdd9aa5d9d406236ff771d061bf17b0cda02a72513953355606e719b'
        receipt['native_receipt'] = prior
        (OUT/'native').mkdir()
        for name in ['reference', *['abi-'+label for label, _, _ in CASES]]:
            path = native/(name+'.npy')
            assert sha(path) == prior['cases'][name]['pcm_sha256']
            (OUT/'native'/path.name).write_bytes(path.read_bytes())
        save()
        run([sys.executable, '-u', __file__, '--phase', 'original'], 'original', timeout=18000)
        receipt = json.loads((OUT/'validation.json').read_text())
        assert receipt['original_generation_complete']
        package = SCRATCH/'python-package'
        run(['gh', 'run', 'download', PACKAGE_RUN, '-R', 'CrispStrobe/CrispASR',
             '-n', 'crispasr-python-linux-x86_64', '-D', package], 'package-download')
        archive = package/'crispasr-python-linux-x86_64.tar.gz'
        assert sha(archive) == PACKAGE_SHA
        with tarfile.open(archive) as tar:
            tar.extractall(package, filter='data')
        bundle = package/'crispasr-python-linux-x86_64'
        sys.path.insert(0, str(bundle))
        from crispasr import Session
        from huggingface_hub import hf_hub_download
        import numpy as np
        asr = Path(hf_hub_download('cstr/nemotron-3.5-asr-streaming-GGUF',
            'nemotron-3.5-asr-streaming-0.6b-f16.gguf', revision=ASR_REV))
        assert sha(asr) == ASR_SHA
        receipt['asr'] = dict(revision=ASR_REV, sha256=ASR_SHA, precision='F16',
            package_run=PACKAGE_RUN, package_source=PACKAGE_SOURCE, package_sha256=PACKAGE_SHA)
        with Session(str(asr), lib_path=str(bundle/'crispasr/libcrispasr.so'),
                     backend='nemotron', n_threads=4) as session:
            with wave.open(str(ROOT/'samples/jfk.wav')) as wav:
                assert (wav.getframerate(), wav.getnchannels(), wav.getsampwidth()) == (16000, 1, 2)
                control = np.frombuffer(wav.readframes(wav.getnframes()), dtype='<i2').astype(np.float32)/32768
            expected = 'And so my fellow Americans ask not what your country can do for you ask what you can do for your country'
            transcript = ' '.join(s.text for s in session.transcribe(control, sample_rate=16000, language='en'))
            receipt['asr_control'] = dict(transcript=transcript, wer=wer(expected, transcript), wer_max=.2)
            save()
            assert receipt['asr_control']['wer'] <= .2, 'Independent recognizer control failed'
            receipt['roundtrips'] = {}
            for label, _, _ in CASES:
                for arm, path in [('original', OUT/f'original-{label}.npy'),
                                  ('native', OUT/'native'/f'abi-{label}.npy')]:
                    pcm = np.load(path, allow_pickle=False)
                    assert pcm.ndim == 1 and pcm.size and np.isfinite(pcm).all()
                    transcript = ' '.join(s.text for s in session.transcribe(pcm, sample_rate=RATE, language='en'))
                    key = arm+'-'+label
                    receipt['roundtrips'][key] = dict(transcript=transcript, wer=wer(TEXT, transcript),
                        wer_max=.2, pcm_sha256=sha(path))
                    if receipt['roundtrips'][key]['wer'] > .2:
                        receipt['failures'].append(key+' WER exceeds .2')
                    print(key, receipt['roundtrips'][key], flush=True)
                    save()
        receipt['complete'] = True
        receipt['roundtrip_pass'] = not receipt['failures']
        save()
        (OUT/'summary.md').write_text('Original/native Breeze decoded roundtrips completed. '
            'See validation.json for all eight transcripts and unchanged WER gates. '
            'This diagnostic does not waive the separate exact-code failures or '
            'accept all native surfaces/speaker similarity.\n')
        assert receipt['roundtrip_pass'], receipt['failures']
    except BaseException as error:
        # Preserve the child phase's latest captures/receipt on failure.
        receipt = json.loads((OUT/'validation.json').read_text())
        receipt['error'] = f'{type(error).__name__}: {error}'
        save()
        raise


if __name__ == '__main__':
    main()
