#!/usr/bin/env python3
"""GPU-only A/B of retained Q8 candidates; never production acceptance."""
import argparse
import json
import os
from pathlib import Path

import numpy as np

from granite5_cuda_acceptance import qualify_runtime
from granite5_cuda_preflight import run_preflight
from granite5_execution import run
from granite5_native_acceptance import sha
from granite5_result_audit import metrics, token_comparison

ROOT = Path(__file__).resolve().parents[2]
CANDIDATE = dict(repo='cstr/crispasr-granite5-validation-public', repo_type='model',
    path='precision/7287199c319f0c6534e2d64732cf385f1c58fbdb/granite5-q8_0-both.gguf',
    revision='71e6be79b67096a1dbaef1f10c848fd2d1ec3fbd',
    sha256='65010f4e8303e934ff3a72870b960a948eb72b8880c5d289cb8b85761f674102', bytes=4244953792)


def main():
    from gguf import GGUFReader
    from huggingface_hub import hf_hub_download
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('bundle', 'cpu-receipt', 'scratch', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    cpu = json.loads(args.cpu_receipt.read_text())
    if not (cpu['complete'] and cpu['cpu_acceptance'] and cpu['profiles']['q8_0']['accepted']):
        raise ValueError('The immutable controlled CPU baseline is required')
    provenance = qualify_runtime(args.bundle, cpu)
    preflight = run_preflight(args.bundle, args.out / 'runtime-preflight.json')
    if not preflight['runtime_preflight_pass']:
        raise ValueError('Actual CUDA loader preflight failed')
    environment = dict(os.environ, LD_LIBRARY_PATH=preflight['ld_library_path'],
        GGML_BACKEND_PATH=str(args.bundle), CUDA_VISIBLE_DEVICES='0', GGML_SCHED_DEBUG='2',
        OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='4')
    if any(k.startswith(('CRISPASR_GRANITE_', 'GRANITE_')) or k == 'CRISPASR_N_GPU_LAYERS'
           for k in environment):
        raise ValueError('Inherited overrides invalidate this controlled A/B')
    receipt = dict(complete=False, diagnostic_only=True, cuda_acceptance=False, production_release=False,
        cpu_receipt_sha256=sha(args.cpu_receipt), runtime_source=provenance['source'],
        runtime_provenance_sha256=sha(args.bundle / 'provenance.json'), hardware=preflight['gpu0'], cases={})
    def save():
        (args.out / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    save()
    def fetch(record):
        path = Path(hf_hub_download(record['repo'], record['path'], repo_type=record['repo_type'],
                                   revision=record['revision'], local_dir=args.scratch))
        if sha(path) != record['sha256'] or path.stat().st_size != record['bytes']:
            raise ValueError('Controlled input bytes differ')
        return path
    reference = fetch(cpu['reference'])
    ref = GGUFReader(reference)
    arrays = {t.name: t.data.reshape(-1).copy() for t in ref.tensors}
    prompt = arrays['prompt_ids']
    expected = arrays['llm_argmax'][len(prompt):]
    text = ref.fields['crispasr.ref.generated_text'].contents()
    for name, record in [('embedding', cpu['profiles']['q8_0']['file']), ('both', CANDIDATE)]:
        model = fetch(record)
        case = args.out / name
        execution = run([args.bundle / 'granite5-stage-probe', model, reference, case,
                         cpu['instruction'], 'cuda'], 'stages', case, environment, 'cuda')
        result = dict(file=record, execution=execution, stages={}, accepted=False)
        for stage in ('mel_spectrogram', 'encoder_out', 'projector_out', 'llm_inputs_embeds',
                      'llm_prefill_logits', 'pipeline_prefill_logits'):
            key = 'llm_prefill_logits' if stage == 'pipeline_prefill_logits' else stage
            result['stages'][stage] = metrics(arrays[key], np.fromfile(case / (stage + '.bin'), dtype='<f4'))
        result['prompt_exact'] = bool(np.array_equal(prompt, np.fromfile(case / 'prompt_ids.bin', dtype='<i4')))
        for label, path in [('bucketed', 'generated_ids.bin'), ('dynamic', 'dynamic_generated_ids.bin')]:
            result[label] = token_comparison(expected, np.fromfile(case / path, dtype='<i4'))
        result['text_exact'] = (case / 'generated_text.txt').read_text().strip() == text.strip()
        result['component_gates_pass'] = (all(v['pass'] for v in result['stages'].values()) and
            result['prompt_exact'] and result['bucketed']['exact'] and result['dynamic']['exact'] and result['text_exact'])
        receipt['cases'][name] = result
        save()
        model.unlink()
    receipt['complete'] = True
    save()
    print('Controlled CUDA precision diagnostic complete; full model acceptance remains false', flush=True)


if __name__ == '__main__':
    main()
