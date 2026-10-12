#!/usr/bin/env python3
"""Exercise the generic Dia dumper/diff with actual original weights and DAC.

The default 128-step reference is a bounded numerical/codec probe, not full
short-speech acceptance. The separate dia_delay_speech worker performs actual
complete CLI/C ABI ASR acceptance. --tokens 3072 requests full original speech.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(os.environ['HEAVY_OUT'])
SCRATCH = Path(os.environ['HEAVY_SCRATCH'])
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--tokens', type=int, default=128)
parser.add_argument('--child-reference', action='store_true')
parser.add_argument('--full-short-speech', action='store_true',
                    help='require 3072-step original capacity and unchanged independent ASR acceptance')
args = parser.parse_args()
if args.full_short_speech:
    assert args.tokens == 3072, 'Full short-speech control must use the original complete capacity'
OUT.mkdir(parents=True, exist_ok=True)
SCRATCH.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT / 'tools'))
from reference_backends import dia_tts_reference as reference

if args.child_reference:
    import numpy as np
    import torch
    import dump_reference
    torch.set_num_threads(4)
    os.environ['DIA_MAX_TOKENS'] = str(args.tokens)
    archives = []
    write_gguf_archive = dump_reference.write_gguf_archive
    def save_archive(captures, metadata, output):
        write_gguf_archive(captures, metadata, output)
        archives.append((captures, metadata))
    dump_reference.write_gguf_archive = save_archive
    # Exercise the real generic CLI/registry/metadata path exactly once.
    sys.argv = ['dump_reference.py', '--backend', 'dia', '--model-dir', 'nari-labs/Dia-1.6B',
                '--audio', str(ROOT / 'samples/jfk.wav'), '--output', str(OUT / 'ref.gguf')]
    dump_reference.main()
    assert len(archives) == 1
    captures, metadata = archives[0]
    metadata['dia_capture_tokens'] = args.tokens
    metadata['dia_recipe'] = 'seed=42,temperature=1.2,top_p=.95,top_k=45,cfg=3,F32,CPU'
    write_gguf_archive(captures, metadata, OUT / 'ref.gguf')
    write_gguf_archive({name: value for name, value in captures.items() if name != 'encoder_output'},
                       metadata, OUT / 'missing-ref.gguf')
    # Negative control: cosine remains one under an incorrect uniform scale.
    # The production diff must reject this archive through magnitude checks.
    captures['encoder_output'] *= 2
    write_gguf_archive(captures, metadata, OUT / 'scaled-ref.gguf')
    np.save(OUT / 'original-pcm.npy', captures['decoded_audio'])
    if args.full_short_speech:
        assert metadata['dia_text'] == '[S1] We are committed to making it secure.'
        pcm = captures['decoded_audio']
        np.save(OUT / 'original-reported.npy', pcm)
        (OUT / 'original-reported.json').write_text(json.dumps({
            'phrase': 'We are committed to making it secure.', 'source_pin': reference.SOURCE_PIN,
            'model_pin': reference.MODEL_PIN, 'seed': 42, 'capacity': args.tokens,
            'audio_seconds': len(pcm) / 44100,
            'rms': float(np.sqrt(np.mean(pcm.astype(np.float64) ** 2))),
            'finite': bool(np.isfinite(pcm).all()), 'scope': 'Actual original generate and real DAC'}, indent=2) + '\n')
    (OUT / 'reference.json').write_text(json.dumps(metadata, indent=2) + '\n')
    sys.exit(0)

receipt = {'source': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
           'official_source': reference.SOURCE_PIN, 'official_model': reference.MODEL_PIN,
           'capture_tokens': args.tokens, 'physical_gpu': False, 'runs': []}

def run(command, name, allow_failure=False):
    with (OUT / (name + '.log')).open('w') as log:
        proc = subprocess.run(list(map(str, command)), cwd=ROOT, stdout=log,
                              stderr=subprocess.STDOUT, timeout=7200)
    data = (OUT / (name + '.log')).read_bytes()
    receipt['runs'].append({'name': name, 'exit': proc.returncode,
                            'log_sha256': hashlib.sha256(data).hexdigest()})
    (OUT / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(name, proc.returncode, data.decode(errors='replace')[-2500:], flush=True)
    if not allow_failure:
        assert proc.returncode == 0, name
    return proc.returncode, data.decode(errors='replace')

source = SCRATCH / 'official-dia'
run(['git', 'clone', '--filter=blob:none', 'https://github.com/nari-labs/dia', source], 'source-clone')
run(['git', '-C', source, 'checkout', '--detach', reference.SOURCE_PIN], 'source-pin')
os.environ['DIA_SOURCE'] = str(source)
os.environ['DIA_DEVICE'] = 'cpu'
run([sys.executable, ROOT / 'tests/test_dia_reference_capture.py', '-v'], 'tiny-original-contracts')
# Child exits before native loading; the 1.6B original never coexists in RAM.
run([sys.executable, __file__, '--child-reference', '--tokens', str(args.tokens)]
    + (['--full-short-speech'] if args.full_short_speech else []), 'original-reference')
build = SCRATCH / 'dia-diff-build'
run(['cmake', '-S', ROOT, '-B', build, '-DCMAKE_BUILD_TYPE=Release', '-DBUILD_SHARED_LIBS=ON',
     '-DGGML_NATIVE=OFF', '-DGGML_CUDA=OFF', '-DGGML_VULKAN=OFF', '-DGGML_BLAS=OFF',
     '-DCRISPASR_BUILD_TESTS=OFF', '-DCRISPASR_BUILD_SERVER=OFF', '-DCRISPASR_OPUS=OFF',
     '-DCRISPASR_AMR=OFF'], 'configure')
run(['cmake', '--build', build, '--target', 'crispasr-diff', '-j4'], 'build-diff')
from huggingface_hub import hf_hub_download
repo, revision = 'cstr/dia-1.6b-GGUF', '3233fbcb32be47761d2e736857b6d1a075b9ba7e'
codec = hf_hub_download(repo, 'dac-44khz.gguf', revision=revision)
os.environ['DIA_CODEC'] = codec
failed = []
if args.full_short_speech:
    asr = hf_hub_download('cstr/nemotron-3.5-asr-streaming-GGUF', 'nemotron-3.5-asr-streaming-0.6b-q4_k.gguf',
                         revision='bbd95a9ca5fa0dfca3312a122dfc45a2b578b9c2')
    lib = next(build.rglob('libcrispasr.so'))
    code, _ = run([sys.executable, ROOT / 'tools/ci-heavy/dia_full_speech.py', '--child', 'recognize',
                   '--model', asr, '--lib', lib, '--phrase', 'reported', '--audio-tag', 'original-reported'],
                  'original-full-short-speech-asr', True)
    receipt['original_full_short_speech_acceptance'] = code == 0
    if code:
        failed.append('original-full-short-speech')
for quant in ('f16', 'q4_k'):
    model = hf_hub_download(repo, f'dia-1.6b-{quant}.gguf', revision=revision)
    code, log = run([build / 'bin/crispasr-diff', 'dia', model, OUT / 'ref.gguf'], quant + '-diff', True)
    if code:
        failed.append(quant)
    # Reject a scale defect even when cosine is unchanged, and reject a missing
    # stage even when every other comparison passes.
    if quant == 'f16' and '[PASS] encoder_output' in log:
        negative_code, negative_log = run([build / 'bin/crispasr-diff', 'dia', model, OUT / 'scaled-ref.gguf'],
                                          'scaled-reference-negative', True)
        assert negative_code != 0 and '[FAIL] encoder_output' in negative_log, negative_log
        missing_code, missing_log = run([build / 'bin/crispasr-diff', 'dia', model, OUT / 'missing-ref.gguf'],
                                        'missing-stage-negative', True)
        assert missing_code != 0 and '[FAIL] encoder_output' in missing_log, missing_log
receipt['numerical_codec_pass'] = not any(name in failed for name in ('f16', 'q4_k'))
receipt['complete_free_speech_acceptance'] = False
receipt['scope'] = 'Pinned original generate and real DAC, exact-history F16/Q4_K numerical/codec comparison. Full native short speech is a separate worker.'
(OUT / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
assert not failed, f'Unchanged numerical/codec gates failed: {failed}'
print('DIA_PINNED_GENERIC_DIFF_PASS', flush=True)
