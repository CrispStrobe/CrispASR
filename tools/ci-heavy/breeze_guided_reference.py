#!/usr/bin/env python3
"""Run the pinned upstream Breeze oracle on GH CPU, preserving all stage captures.

Preserve the reviewed model computation with an explicit matched frame budget.
No Kaggle CPU fallback, automatic fixture overwrite, or native acceptance claim.
"""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time

UPSTREAM = '58ec70ce5fa4cc361bdebf77ec40d1365da00ab2'
MODEL = '3e28c5151381a722f1d8661b4118c298caa77aa4'
DUMPER_SHA = '09c26e31a63ec36d852f2918415929c6148e5d1c750a68f9a5cb8e49819b4761'
SCRIPT_VERSION = '2026-10-11.breeze-guided-gh-cpu.2'
CASES = ('plain', 'clone', 'design-s0', 'design-s1', 'design-s4',
         'direct-s0', 'direct-s1', 'direct-s4')
ROOT = Path(__file__).resolve().parents[2]
OUT = Path(os.environ['HEAVY_OUT'])
SCRATCH = Path(os.environ['HEAVY_SCRATCH'])


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main(*, promote_joint_arithmetic=False, kv_control_installer=None, promote_all_inference=False):
    assert kv_control_installer is None or promote_joint_arithmetic
    assert not promote_all_inference or promote_joint_arithmetic
    import numpy as np
    import soundfile as sf
    import torch
    from huggingface_hub import snapshot_download

    OUT.mkdir(parents=True, exist_ok=True)
    upstream = SCRATCH / 'breeze-upstream'
    subprocess.run(['git', 'clone', '--no-checkout',
                    'https://github.com/breezeblue-ai/breeze-tts.git', str(upstream)], check=True)
    subprocess.run(['git', 'checkout', '--detach', UPSTREAM], cwd=upstream, check=True)
    actual = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=upstream, text=True).strip()
    assert actual == UPSTREAM
    dumper = ROOT / 'tools/kaggle/breeze-refdump/breeze_refdump.py'
    assert digest(dumper) == DUMPER_SHA, 'Reviewed oracle changed; re-audit before running'
    source = dumper.read_text()
    # Reuse its complete stage/guidance computation; replace only deployment.
    code = source.split('# ── checkpoint ─', 1)[1]
    code = code[code.index('from huggingface_hub import snapshot_download'):]
    code = code.split('# ── upload ─', 1)[0]
    needle = 'snapshot_download(HF_MODEL, local_dir=str(CKPT), token=HF_TOKEN or None,'
    assert code.count(needle) == 1
    code = code.replace(needle, needle + '\n                  revision=MODEL_REVISION,')
    # Upstream builds MaxLengthCriteria from the positive prompt, then swaps
    # the shorter negative prompt at scale zero and subtracts that new length.
    # A max_new_tokens=24 request therefore emits 31 frames for our seven-token
    # instruction. Set the stopping length against the actually active prompt.
    needle = 'return_dict_in_generate=True)'
    assert code.count(needle) == 1
    code = code.replace(needle, 'return_dict_in_generate=True, max_new_tokens=None, '
                        'max_length=active_prompt_length(gen_inputs) + MAX_FRAMES)')
    if promote_joint_arithmetic:
        # Original-only control: preserve the loader's BF16-rounded weights
        # and text/codec arithmetic. Change only backbone/depth/LM-head
        # arithmetic, including the backbone input cast used by the isolated
        # precision probe. These outputs never replace canonical fixtures.
        needle = 'update_generation_config_for_breeze(model)'
        assert code.count(needle) == 1
        code = code.replace(needle, '''assert next(model.backbone_model.parameters()).dtype == torch.bfloat16
model.backbone_model.float()
model.depth_decoder.float()
model.lm_head.float()
def promote_backbone_inputs(module, args, kwargs):
    if kwargs.get('inputs_embeds') is not None:
        kwargs['inputs_embeds'] = kwargs['inputs_embeds'].float()
    return args, kwargs
model.backbone_model.register_forward_pre_hook(promote_backbone_inputs, with_kwargs=True)
assert next(model.backbone_model.parameters()).dtype == torch.float32
assert next(model.depth_decoder.parameters()).dtype == torch.float32
if promote_all_inference:
    # The reviewed recipe passes the separate Qwen audio tokenizer. Keep
    # its unused training codec intact while widening every inference module,
    # including text encoding and audio-embedding averaging.
    unused_codec = model.codec_model
    codec_state = [(p.data_ptr(), p.dtype) for p in unused_codec.parameters()]
    before_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    model.codec_model = None
    model.float()
    model.codec_model = unused_codec
    assert codec_state == [(p.data_ptr(), p.dtype) for p in unused_codec.parameters()]
    assert all(p.dtype == torch.float32 for name, p in model.named_parameters()
               if not name.startswith('codec_model.'))
    full_inference_control = dict(active_parameter_dtype='torch.float32',
        unused_training_codec_preserved=True,
        parameter_bytes_before=before_bytes,
        parameter_bytes_after=sum(p.numel() * p.element_size() for p in model.parameters()))
if kv_control_installer is not None:
    kv_control_installer()
''' + needle)
    def active_prompt_length(inputs):
        ids = inputs['input_ids']
        if inputs.get('cfg_scale', 1.0) == 0.0 and inputs.get('cfg_negative_prompt_ids') is not None:
            ids = inputs['cfg_negative_prompt_ids']
        return int(ids.shape[1])
    compile(code, str(dumper), 'exec')
    (OUT / 'reviewed-oracle.py').write_text(source)
    (OUT / 'executed-oracle-section.py').write_text(code)
    torch.set_grad_enabled(False)
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    dumps = OUT / 'breeze-dumps'
    dumps.mkdir()
    def step(name, **values):
        print(f'[{time.strftime("%H:%M:%S")}] {name} ' + json.dumps(values, default=str), flush=True)
    env = dict(__name__='__main__', json=json, os=os, sys=sys, time=time,
               Path=Path, np=np, torch=torch, sf=sf, step=step,
               WORK=OUT, DUMPS=dumps, BREEZE=upstream, CLONE=ROOT,
               CKPT=SCRATCH / 'breeze-checkpoint', HF_MODEL='BreezeBlue/Breeze-TTS-2',
               HF_TOKEN=os.environ.get('HF_TOKEN', ''), MODEL_REVISION=MODEL,
               DEVICE='cpu', RUN_DEVICE='cpu', RUN_GPU=None, SEED=42, GREEDY=True,
               MAX_FRAMES=24, DUMP_TE_LAYERS=True, N_DUMP_FRAMES=5,
               active_prompt_length=active_prompt_length,
               kv_control_installer=kv_control_installer,
               promote_all_inference=promote_all_inference,
               SCRIPT_VERSION=SCRIPT_VERSION,
               SYN_TEXT='The quick brown fox jumps over the lazy dog.',
               REF_TEXT='And so my fellow Americans, ask not what your country can do for you, '
                        'ask what you can do for your country.')
    receipt = dict(script_version=SCRIPT_VERSION, upstream=UPSTREAM, model_revision=MODEL,
                   source=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                   oracle_sha256=DUMPER_SHA, sample_sha256=digest(ROOT / 'samples/jfk.wav'),
                   device='cpu', seed=42, greedy=True, max_frames=24,
                   frame_budget='explicit active prompt length + 24; max_new_tokens=None',
                   unadjusted_reference_run=38121758868,
                   packages={p: importlib.metadata.version(p) for p in
                             ('torch', 'torchaudio', 'transformers', 'qwen-tts', 'numpy')},
                   native_parity_tested=False, full_speech_asr_tested=False,
                   canonical_reference=not promote_joint_arithmetic,
                   arithmetic_control=(
                       'all inference F32, preserved loaded weights; external audio tokenizer and unused training codec unchanged'
                       if promote_all_inference else
                       'joint F32 backbone/depth/LM-head, BF16-rounded weights; unchanged text/codec'
                   ) if promote_joint_arithmetic else None,
                   kv_storage_control=getattr(kv_control_installer, 'description', None),
                   full_inference_f32_control=promote_all_inference,
                   complete=False)
    try:
        exec(compile(code, str(dumper), 'exec'), env)
        assert not env['guided_errors'], env['guided_errors']
        assert set(env['guided_meta']) == set(CASES)
        for case in CASES:
            required = ['prompt_input_ids', 'backbone_logits_frame0', 'codes']
            if case.endswith(('-s0', '-s4')):
                required.append('neg_prompt_input_ids')
            for name in required:
                a = np.load(dumps / 'guided' / case / (name + '.npy'), allow_pickle=False)
                assert a.size and np.isfinite(a).all(), (case, name, a.shape)
                if name == 'codes':
                    assert a.ndim == 2 and a.shape[1] == 16 and a.dtype.kind in 'iu'
                    assert 0 < a.shape[0] <= 24, (case, 'frame budget', a.shape)
                elif name == 'backbone_logits_frame0':
                    assert a.shape == (2052,) and a.dtype.kind == 'f'
                else:
                    assert a.ndim == 1 and a.dtype.kind in 'iu'
        receipt['arrays'] = {}
        for p in sorted(dumps.rglob('*.npy')):
            a = np.load(p, allow_pickle=False)
            assert a.size and np.isfinite(a).all(), str(p)
            receipt['arrays'][str(p.relative_to(dumps))] = dict(
                shape=list(a.shape), dtype=str(a.dtype), sha256=digest(p))
        receipt['guided'] = env['guided_meta']
        receipt['complete'] = True
        print('Original oracle: all eight guidance cases and full finite stage captures COMPLETE.', flush=True)
    except BaseException as exc:
        receipt['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        if promote_all_inference:
            receipt['full_inference_control'] = env.get('full_inference_control')
        if kv_control_installer is not None:
            receipt['kv_storage_witness'] = kv_control_installer.witness
        (OUT / 'validation.json').write_text(json.dumps(receipt, indent=2) + '\n')
        (OUT / 'summary.md').write_text(
            f"Breeze original guidance reference: complete={receipt['complete']}. "
            'CPU oracle only; native parity and full-speech ASR remain separate gates.\n')


if __name__ == '__main__':
    main()
