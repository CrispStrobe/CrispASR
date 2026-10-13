#!/usr/bin/env python3
"""Export original F32 critical output tensors into Dia's published DAC layout.

This explicit diagnostic conversion preserves all unrelated published tensor
payloads. It does not promote the result to a production model or change pins.
"""
import argparse
import importlib.metadata
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from dia_dac_export import CHECKPOINT_SHA, digest, export_precision

TEMPLATE_SHA = '6f62c6dbb26fd69f0634db3c7f044464cfbd9a5b8806eb621f8be7034d869a3e'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--template', type=Path, required=True, help='Pinned published Dia dac-44khz.gguf')
    parser.add_argument('--output', type=Path, required=True, help='New diagnostic GGUF output path')
    parser.add_argument('--precision', choices=('output-f32', 'rvq-output-f32', 'f32'), required=True)
    parser.add_argument('--checkpoint', type=Path, help='Optional original 44khz/8kbps/0.0.1 checkpoint')
    parser.add_argument('--receipt', type=Path, required=True, help='Full physical tensor audit JSON')
    args = parser.parse_args()
    if args.receipt.exists() or args.output.exists() or not args.template.is_file():
        parser.error('Input must exist; output and receipt must be new paths')
    if args.receipt.resolve() in (args.output.resolve(), args.template.resolve()):
        parser.error('Receipt, output and template paths must be distinct')
    if digest(args.template) != TEMPLATE_SHA:
        parser.error('Template must be the pinned published Dia dac-44khz.gguf (SHA256 ' + TEMPLATE_SHA + ')')
    if importlib.metadata.version('descript-audio-codec') != '1.0.0':
        raise ValueError('Requires original descript-audio-codec==1.0.0')
    import torch
    import dac
    torch.set_num_threads(4)
    checkpoint = args.checkpoint or Path(dac.utils.download(model_type='44khz', model_bitrate='8kbps', tag='0.0.1'))
    if digest(checkpoint) != CHECKPOINT_SHA:
        raise ValueError('Original DAC checkpoint hash mismatch')
    codec = dac.DAC.load(str(checkpoint)).cpu().eval()
    for module in codec.modules():
        if hasattr(module, 'weight_g'):
            torch.nn.utils.remove_weight_norm(module)
    receipt = export_precision(codec, args.template, args.output, args.precision)
    receipt.update(dac_package='1.0.0')
    args.receipt.write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps({key: value for key, value in receipt.items() if key != 'tensors'}), flush=True)


if __name__ == '__main__':
    main()
