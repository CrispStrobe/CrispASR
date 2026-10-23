#!/usr/bin/env python3
"""Replay the accepted Granite5 stage, magnitude, token and public API gates.

Consumes published immutable GGUFs and the retained original reference. This
does not rebake a reference, convert weights or claim new GPU qualification.
"""
import argparse
import json
import os
from pathlib import Path
import re

from granite5_execution import validate_case
from granite5_native_acceptance import sha

ROOT = Path(__file__).resolve().parents[2]
INSTRUCTION = 'transcribe the speech with proper punctuation and capitalization.'


def check_pins(pins):
    if not (pins['repo'] == 'cstr/granite-speech-5.0-2b-GGUF' and
            re.fullmatch('[0-9a-f]{40}', pins['revision']) and
            pins['instruction'] == INSTRUCTION and tuple(pins['profiles']) in (('f16',), ('f16', 'q8_0')) and
            set(pins['models']) == set(pins['profiles'])):
        raise ValueError('Explicit immutable qualified profiles and the original instruction are required')
    for quant, record in pins['models'].items():
        if not (record['filename'] == f'granite-speech-5.0-2b-{quant}.gguf' and
                re.fullmatch('[0-9a-f]{64}', record['sha256']) and record['bytes'] > 0):
            raise ValueError('Invalid qualified model content pin')
    reference = pins['reference']
    if not (reference['repo'] == 'cstr/crispasr-regression-fixtures' and
            reference['repo_type'] == 'dataset' and
            reference['revision'] == '0426caae5f4f536d87058f5cb36d77525dd5bcde' and
            reference['path'] == 'granite-speech-5.0-2b/a02a0dd9299e0b759704f520fe3959f50a764f71/jfk-ref.gguf' and
            reference['sha256'] == 'ec91b6de806a1919d41edeb604e0aba8b636f3eadc7a4be487c8bd2ff57d5d84' and
            reference['bytes'] == 19195808):
        raise ValueError('The independent original F32 reference must remain unchanged')


def check_content(path, record):
    if path.stat().st_size != record['bytes'] or sha(path) != record['sha256']:
        raise ValueError('Pinned regression input content differs')


def main():
    from huggingface_hub import HfApi, hf_hub_download
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('pins', 'build-root', 'scratch', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    pins = json.loads(args.pins.read_text())
    check_pins(pins)
    args.out.mkdir(parents=True, exist_ok=True)
    args.scratch.mkdir(parents=True, exist_ok=True)
    receipt = dict(complete=False, cpu_acceptance=False, cuda_acceptance=False,
                   production_release=False, pins_sha256=sha(args.pins), profiles={})
    def save():
        (args.out / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    save()
    card = HfApi().model_info(pins['repo'], revision=pins['revision']).card_data
    if not (card['license'] == 'apache-2.0' and card['crispasr_native_acceptance'] is True):
        raise ValueError('Published model license/qualification card differs')
    bundle = args.scratch / 'runtime'
    bundle.mkdir()
    for name in ('crispasr', 'granite5-stage-probe'):
        target = (args.build_root / 'bin' / name).resolve(strict=True)
        (bundle / name).symlink_to(target)
    libraries = {p.resolve() for p in args.build_root.rglob('libcrispasr.so') if p.is_file()}
    if len(libraries) != 1:
        raise ValueError('Exactly one built C ABI library is required')
    (bundle / 'libcrispasr.so').symlink_to(libraries.pop())
    library_dirs = sorted({str(p.parent.resolve()) for p in args.build_root.rglob('*.so*') if p.is_file()})
    environment = dict(os.environ, LD_LIBRARY_PATH=':'.join(library_dirs),
                       OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='4')
    if any(k.startswith(('CRISPASR_GRANITE_', 'GRANITE_')) or k == 'CRISPASR_N_GPU_LAYERS'
           for k in environment):
        raise ValueError('Inherited Granite overrides invalidate production regression')
    ref = pins['reference']
    reference = Path(hf_hub_download(ref['repo'], ref['path'], repo_type=ref['repo_type'],
                                   revision=ref['revision'], local_dir=args.scratch / 'reference'))
    check_content(reference, ref)
    for quant in pins['profiles']:
        record = pins['models'][quant]
        model = Path(hf_hub_download(pins['repo'], record['filename'], revision=pins['revision'],
                                    local_dir=args.scratch / 'models'))
        check_content(model, record)
        # validate_case fails closed on every missing/numerically wrong stage,
        # mismatched prompt/token, renamed Session or CLI transcript failure.
        result = validate_case(model, reference, pins['instruction'], bundle, ROOT / 'python',
                               ROOT / 'samples/jfk.wav', args.out / quant, 'cpu', environment, quant)
        if not (result['complete'] and result['accepted'] and result['device'] == 'cpu'):
            raise ValueError('Incomplete native regression result')
        receipt['profiles'][quant] = result
        save()
        model.unlink()
    receipt.update(complete=True, cpu_acceptance=True)
    save()
    print('Granite5 CPU regression: all published profiles pass all accepted gates', flush=True)


if __name__ == '__main__':
    main()
