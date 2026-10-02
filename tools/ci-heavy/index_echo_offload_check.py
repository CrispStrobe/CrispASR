#!/usr/bin/env python3
"""Real HF GDN CPU/disk A/B: prove cached functional weights are materialized.

Tiny random official Qwen3.5 model, no large checkpoint or native oracle.
Compare fully resident math with actual HF dispatch; the unpreloaded arm fails
at the direct cached convolution read, and the preloaded arm must match logits.
"""
import functools
import json
import os
from pathlib import Path
from unittest.mock import patch

import torch
from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig
from transformers.integrations import accelerate as hf_accelerate

OUT = Path(os.environ['HEAVY_OUT'])
TEMP = Path(os.environ['HEAVY_SCRATCH']) / 'gdn-offload-check'
OUT.mkdir(parents=True, exist_ok=True)
TEMP.mkdir(parents=True, exist_ok=True)
torch.set_num_threads(2)
torch.set_grad_enabled(False)
torch.manual_seed(481)
config = Qwen3_5TextConfig(vocab_size=512, hidden_size=64, intermediate_size=128,
    num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=1, head_dim=16,
    linear_num_key_heads=2, linear_num_value_heads=4, linear_key_head_dim=16,
    linear_value_head_dim=16, linear_conv_kernel_dim=4, full_attention_interval=4,
    max_position_embeddings=256, tie_word_embeddings=False,
    rope_parameters=dict(rope_type='default', rope_theta=10000000, partial_rotary_factor=.25,
                         mrope_interleaved=True, mrope_section=[1, 1, 0]))
baseline = Qwen3_5ForCausalLM(config).eval()
checkpoint = TEMP / 'checkpoint'
baseline.save_pretrained(checkpoint)
ids = torch.tensor([[11, 28, 49, 83, 103, 19, 78, 41]])
prompt = baseline(input_ids=ids, use_cache=True, logits_to_keep=1)
steps = [prompt.logits.detach().clone()]
cache = prompt.past_key_values
for token in [42, 18, 73]:
    step = baseline(input_ids=torch.tensor([[token]]), past_key_values=cache, use_cache=True, logits_to_keep=1)
    steps.append(step.logits.detach().clone())
    cache = step.past_key_values
mapping = {'model.embed_tokens': 'cpu', 'model.layers.0': 'cpu', 'model.layers.1': 'disk',
           'model.layers.2': 'cpu', 'model.layers.3': 'disk', 'model.norm': 'cpu',
           'model.rotary_emb': 'cpu', 'lm_head': 'cpu'}
result = dict(torch=torch.__version__, dtype='float32', device_map=mapping, arms={})
for preload in [False, True]:
    dispatch = hf_accelerate.dispatch_model

    @functools.wraps(dispatch)
    def configured_dispatch(*a, **kw):
        if preload:
            kw['preload_module_classes'] = ['Qwen3_5GatedDeltaNet']
        return dispatch(*a, **kw)

    with patch.object(hf_accelerate, 'dispatch_model', new=configured_dispatch):
        model = Qwen3_5ForCausalLM.from_pretrained(checkpoint, dtype=torch.float32,
            device_map=mapping, offload_folder=TEMP / ('preloaded' if preload else 'unpreloaded'),
            offload_buffers=True).eval()
    calls = []
    for layer in model.model.layers:
        if not hasattr(layer, 'linear_attn'):
            continue
        original = layer.linear_attn.causal_conv1d_update

        def checked_update(*a, original=original, **kw):
            weight = a[2]
            calls.append(dict(meta=weight.is_meta, device=str(weight.device)))
            if weight.is_meta:
                raise RuntimeError('Cached functional convolution reached an unmaterialized meta weight')
            return original(*a, **kw)

        layer.linear_attn.causal_conv1d_update = checked_update
    metrics = dict(functional_reads=calls, prefill_max_abs=None, cached_max_abs=[])
    try:
        initial = model(input_ids=ids, use_cache=True, logits_to_keep=1)
        metrics['prefill_max_abs'] = float((initial.logits-steps[0]).abs().max())
        torch.testing.assert_close(initial.logits, steps[0], atol=1e-6, rtol=1e-5)
        cache = initial.past_key_values
        for i, token in enumerate([42, 18, 73], 1):
            current = model(input_ids=torch.tensor([[token]]), past_key_values=cache, use_cache=True, logits_to_keep=1)
            metrics['cached_max_abs'].append(float((current.logits-steps[i]).abs().max()))
            torch.testing.assert_close(current.logits, steps[i], atol=1e-6, rtol=1e-5)
            cache = current.past_key_values
        metrics['matched'] = True
    except RuntimeError as error:
        if preload:
            raise
        assert 'unmaterialized meta weight' in str(error), str(error)
        metrics['matched'] = False
        metrics['error'] = str(error)
    if preload:
        assert metrics['matched'] and calls and not any(c['meta'] for c in calls)
    else:
        assert not metrics['matched'] and any(c['meta'] for c in calls)
    result['arms']['preloaded' if preload else 'unpreloaded'] = metrics
(OUT / 'offload-check.json').write_text(json.dumps(result, indent=2)+'\n')
(OUT / 'summary.md').write_text('Actual CPU/disk GDN: unpreloaded cached convolution has meta weights; '
    'documented parent preload matches fully resident prefill and all three cached steps.\n')
print(json.dumps(result, indent=2), flush=True)
