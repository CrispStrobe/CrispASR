#!/usr/bin/env python3
"""Pinned #485 producer: convert, quantize, upload, independent CPU reference.

Experimental artifacts remain private until stage and decoded-output parity pass.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import shutil
import sys
import time

from huggingface_hub import HfApi, snapshot_download

from index_echo_produce_constants import MODELS, LLAMA_REVISION, LICENSE_URL
ROOT = Path(__file__).resolve().parents[2]
parser = argparse.ArgumentParser()
parser.add_argument('--size', choices=['2b', '9b'], default='2b')
parser.add_argument('--reference-memory', help='JSON HF max_memory for placement-only CPU/disk reference offload')
parser.add_argument('--fp32-decoder', action='store_true', help='Fully F32 diagnostic oracle; original blueprint retains nested BF16')
parser.add_argument('--reference-only', action='store_true')
parser.add_argument('--convert-only', action='store_true')
parser.add_argument('--audit-only', action='store_true', help='Audit effective blueprint dtypes without generation')
parser.add_argument('--pipeline-only', action='store_true', help='Released file/VAD/context reference')
parser.add_argument('--quant-only', action='store_true', help='Reuse the validated F16 decoder; skip references')
parser.add_argument('--quants', nargs='+', choices=['q8_0', 'q4_k', 'q4_k_selective'], default=['q8_0', 'q4_k'])
parser.add_argument('--clips', nargs='+', choices=['jfk', 'zh', 'jfk-tail'], default=['jfk', 'zh', 'jfk-tail'])
args = parser.parse_args()
SOURCE, REVISION, DESTINATION = MODELS[args.size]
PREFIX = 'index-echo-' + args.size
if sum([args.reference_only, args.convert_only, args.quant_only, args.pipeline_only, args.audit_only]) > 1:
    parser.error('--reference-only, --convert-only and --quant-only and --pipeline-only are mutually exclusive')
if args.fp32_decoder and not (args.reference_only or args.pipeline_only):
    parser.error('--fp32-decoder requires --reference-only or --pipeline-only')
REFERENCE_DIR = 'reference-f32' if args.fp32_decoder else 'reference'
if args.fp32_decoder: os.environ['INDEX_ECHO_REF_FP32_DECODER'] = '1'
SCRATCH = Path(os.environ['HEAVY_SCRATCH']) / 'index-echo'
OUT = Path(os.environ['HEAVY_OUT'])
SCRATCH.mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)
os.environ['TMPDIR'] = str(SCRATCH)
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['INDEX_ECHO_REF_THREADS'] = '4'
if args.reference_memory:
    if not (args.reference_only or args.pipeline_only):
        parser.error('--reference-memory requires reference-only or pipeline-only')
    memory = json.loads(args.reference_memory)
    if set(memory) != {'cpu'}:
        parser.error('Hosted CPU reference offload accepts only a cpu memory budget')
    os.environ['INDEX_ECHO_REF_DEVICE_MAP'] = 'auto'
    os.environ['INDEX_ECHO_REF_MAX_MEMORY'] = json.dumps(memory)
    os.environ['INDEX_ECHO_REF_OFFLOAD_DIR'] = str(SCRATCH / 'reference-offload')

receipt = dict(source=SOURCE, revision=REVISION, converter_revision=LLAMA_REVISION,
               destination=DESTINATION, validated=False, artifacts=[], events=[],
               reference_memory=json.loads(args.reference_memory) if args.reference_memory else None)


def event(name):
    receipt['events'].append(dict(name=name, utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())))
    (OUT / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(name, flush=True)


def run(*args, cwd=ROOT):
    event('run: ' + ' '.join(map(str, args)))
    subprocess.run(list(map(str, args)), cwd=cwd, check=True)


api = HfApi(token=os.environ['HF_TOKEN'])
# Maintainer creates the private destination once; CI publishing credentials
# may write existing repositories without permission to create new ones.
# Refuse to overwrite a previously published validated model repository.
if not api.repo_info(DESTINATION).private:
    raise RuntimeError('Producer destination must be private until validated')
# Verify publishing rights before downloading or converting multi-GB weights.
api.upload_file(path_or_fileobj=('---\nlicense: apache-2.0\n---\n\n# Index-Echo S2TT ' + args.size.upper() + '\n\n'
                'Private development artifacts for CrispASR issue #485. '
                'Runtime parity and decoded-output validation are pending.\n').encode(),
                path_in_repo='README.md', repo_id=DESTINATION)


def upload(path, remote=None):
    remote = remote or path.name
    digest = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            digest.update(block)
    api.upload_file(path_or_fileobj=path, path_in_repo=remote, repo_id=DESTINATION)
    receipt['artifacts'].append(dict(path=remote, bytes=path.stat().st_size, sha256=digest.hexdigest()))
    event('uploaded: ' + remote)


try:
    if args.pipeline_only and not shutil.which('ffmpeg'):
        run('sudo', 'apt-get', 'update', '-qq')
        run('sudo', 'apt-get', 'install', '-y', '-qq', 'ffmpeg')
    event('download pinned source')
    source = Path(snapshot_download(SOURCE, revision=REVISION, local_dir=SCRATCH / 'source',
        allow_patterns=['audio_config.json', 'audio_tower.safetensors', 'connector.safetensors',
                        'llm/config.json', 'llm/LICENSE'] if args.quant_only else None))
    license_file = source / 'llm' / 'LICENSE'
    if not license_file.exists():
        from urllib.request import urlopen
        license_file = SCRATCH / 'LICENSE'
        with urlopen(LICENSE_URL, timeout=60) as response:
            license_file.write_bytes(response.read())
        receipt['license_source'] = LICENSE_URL
    upload(license_file, 'LICENSE')
    if not args.reference_only and not args.pipeline_only and not args.audit_only:
        audio = SCRATCH / f'{PREFIX}-f16.gguf'
        decoder = SCRATCH / f'{PREFIX}-decoder-f16.gguf'
        if args.quant_only:
            base_revision = api.model_info(DESTINATION).sha
            receipt['f16_base_revision'] = base_revision
            snapshot_download(DESTINATION, revision=base_revision, local_dir=SCRATCH, allow_patterns=[decoder.name])
        else:
            llama = SCRATCH / 'llama-converter'
            run('git', 'init', llama)
            run('git', 'remote', 'add', 'origin', 'https://github.com/ggml-org/llama.cpp.git', cwd=llama)
            run('git', 'fetch', '--depth=1', 'origin', LLAMA_REVISION, cwd=llama)
            run('git', 'checkout', 'FETCH_HEAD', cwd=llama)
            run(sys.executable, ROOT / 'models/convert-index-echo-to-gguf.py',
                '--model', source, '--output', audio, '--decoder-name', decoder.name)
            upload(audio)
            run(sys.executable, llama / 'convert_hf_to_gguf.py', source / 'llm',
                '--outfile', decoder, '--outtype', 'f16', '--no-mtp')
            # Both speech exports omit MTP weights despite their config declaration.
            upload(decoder)
        import gguf
        converted_decoder = gguf.GGUFReader(str(decoder))
        config = json.loads((source / 'llm/config.json').read_text())
        blocks = config.get('text_config', config)['num_hidden_layers']
        assert int(converted_decoder.fields['qwen35.block_count'].contents()) == blocks
        assert not any(t.name.startswith(f'blk.{blocks}.') for t in converted_decoder.tensors)
        receipt['decoder_blocks'] = blocks
        del converted_decoder
        build = SCRATCH / 'build'
        run('cmake', '-S', ROOT, '-B', build, '-G', 'Ninja', '-DCMAKE_BUILD_TYPE=Release',
            '-DCRISPASR_BUILD_TESTS=OFF', '-DCRISPASR_BUILD_SERVER=OFF', '-DGGML_NATIVE=OFF')
        run('cmake', '--build', build, '--target', 'crispasr-quantize', '-j', '4')
        quantizer = build / 'bin/crispasr-quantize'
        for quant in args.quants:
            # Each audio file points to its own quantized decoder companion.
            companion = f'{PREFIX}-decoder-{quant}.gguf'
            run(sys.executable, ROOT / 'models/convert-index-echo-to-gguf.py', '--model', source,
                '--output', audio, '--decoder-name', companion)
            for original, filename in [(audio, f'{PREFIX}-{quant}.gguf'), (decoder, companion)]:
                converted = SCRATCH / filename
                overrides = []
                if quant == 'q4_k_selective':
                    # Existing per-tensor overrides keep this A/B isolated from
                    # other Qwen3.5 users. Never dequantize Q8 into a fake F16 base.
                    rules = ([r'^audio\.conv\.[123]\.weight$=f16',
                              r'^connector\..*\.weight$=f16', r'^audio\..*\.weight$=q8_0']
                             if original == audio else
                             [r'^(token_embd|output)\.weight$=f16',
                              r'\.(ssm_.*|attn_qkv|attn_gate)\.weight$=q8_0'])
                    overrides = [arg for rule in rules for arg in ['--tensor-type', rule]]
                    receipt.setdefault('quant_recipes', {})[filename] = rules
                run(quantizer, original, converted, 'q4_k' if quant == 'q4_k_selective' else quant, *overrides)
                upload(converted)
                converted.unlink()
        audio.unlink()
        decoder.unlink()
        receipt['decoder_no_mtp'] = True
        event('requested conversion cohorts complete')
        upload(OUT / 'receipt.json', 'quant-receipt.json' if args.quant_only else 'conversion-receipt.json')
    for clip in ([] if args.convert_only or args.quant_only or args.pipeline_only or args.audit_only else args.clips):
        event('independent released Python class: CPU F32 ' + clip)
        audio_path = ROOT / 'samples' / ('paraformer_zh.wav' if clip == 'zh' else 'jfk.wav')
        if clip == 'jfk-tail':
            import wave
            with wave.open(str(audio_path), 'rb') as wav:
                params = wav.getparams()
                # Stress non-hop-aligned length and a partial final conv chunk.
                pcm = wav.readframes(wav.getnframes() - 197)
            audio_path = OUT / 'jfk-tail.wav'
            with wave.open(str(audio_path), 'wb') as wav:
                wav.setparams(params); wav.writeframes(pcm)
            upload(audio_path, REFERENCE_DIR + '/jfk-tail.wav')
        ref = OUT / f'{PREFIX}-{clip}-ref.gguf'
        run(sys.executable, ROOT / 'tools/dump_reference.py', '--backend', 'index-echo',
            '--model-dir', source, '--audio', audio_path, '--output', ref)
        upload(ref, f'{REFERENCE_DIR}/{clip}-ref.gguf')
    if args.pipeline_only:
        sys.path.insert(0, str(ROOT / 'tools'))
        from reference_backends.index_echo import dump_pipeline
        pipeline, multi = dump_pipeline(source, OUT, ROOT / 'samples')
        upload(multi, REFERENCE_DIR + '/' + multi.name)
        upload(pipeline, REFERENCE_DIR + '/' + pipeline.name)
    if args.audit_only:
        import importlib.util
        import torch
        torch.set_num_threads(4)
        spec = importlib.util.spec_from_file_location('released_index_echo', source / 'infer.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        model = module.AudioTransModel(str(source), device='cpu', dtype=torch.float32)
        sys.path.insert(0, str(ROOT / 'tools'))
        from reference_backends.index_echo import precision_audit
        audit = precision_audit(model)
        (OUT / 'precision-audit.json').write_text(json.dumps(audit, indent=2) + '\n')
        print(json.dumps(audit, indent=2), flush=True)
    event('producer complete; runtime parity pending')
    (OUT / 'summary.md').write_text(f'Index-Echo {args.size.upper()} pinned conversion and Python reference complete. '
                                   'Artifacts are private and runtime parity is pending.\n')
except Exception:
    event('producer failed; inspect Actions log')
    raise
