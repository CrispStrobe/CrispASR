#!/usr/bin/env python3
"""Load pinned upstream Qwen2 math for one unpadded 39-token CPU prefill.

This avoids shipping unrelated Transformers/tokenizer modules to an offline
worker with a 1 MiB script limit. Mathematical class/function bodies come from
the original files. Metadata/deprecation decorators and inactive checkpointing
plumbing are excluded. This is not a general Transformers replacement.
"""
import ast
import hashlib
from pathlib import Path
from types import ModuleType, SimpleNamespace


PINNED = {
    'modeling_qwen2.py': 'a59fa06524227361fb401baf4d177124a27aec146bc01ec931235a9abbab17cb',
    'modeling_rope_utils.py': '55cc0c8cb76f592ab178b4662adcbdf4ad5012bb0b4249c0a96603141f18e9ac',
    'masking_utils.py': '7a963feed8173b8265298dd24350c166222ce6b4719442fce48ab414d0478a6a',
}


def config_from_dict(values):
    required = ('hidden_size', 'intermediate_size', 'num_hidden_layers',
                'num_attention_heads', 'num_key_value_heads', 'vocab_size',
                'hidden_act', 'rms_norm_eps', 'rope_theta', 'max_position_embeddings')
    if any(name not in values for name in required):
        raise ValueError('missing bounded Qwen2 configuration field')
    class Config(SimpleNamespace):
        def to_dict(self):
            return dict(vars(self))
    data = dict(values)
    data.setdefault('attention_dropout', 0.)
    data.setdefault('rope_scaling', None)
    data.setdefault('use_sliding_window', False)
    data.setdefault('layer_types', ['full_attention'] * data['num_hidden_layers'])
    if (data['hidden_act'] != 'silu' or data['rope_scaling'] or data['use_sliding_window']
            or data['attention_dropout'] != 0. or data.get('partial_rotary_factor', 1.) != 1.
            or data['layer_types'] != ['full_attention'] * data['num_hidden_layers']):
        raise ValueError('bounded Qwen2 requires full attention/default RoPE/SwiGLU/no dropout')
    return Config(**data)


def load(source):
    import torch
    root = Path(source)
    module = ModuleType('vibevoice_bounded_upstream_qwen2')
    module.__file__ = str(root / 'modeling_qwen2.py')
    ns = module.__dict__
    ns.update(torch=torch, nn=torch.nn, ACT2FN={'silu': torch.nn.SiLU()},
              GradientCheckpointingLayer=torch.nn.Module, ALL_ATTENTION_FUNCTIONS={})
    removed = {'deprecate_kwarg', 'dynamic_rope_update', 'use_kernel_forward_from_hub'}
    class Decorators(ast.NodeTransformer):
        def visit_ClassDef(self, node):
            self.generic_visit(node)
            node.decorator_list = [d for d in node.decorator_list if decorator_name(d) not in removed]
            return node

        def visit_FunctionDef(self, node):
            self.generic_visit(node)
            node.decorator_list = [d for d in node.decorator_list if decorator_name(d) not in removed]
            return node
    def decorator_name(node):
        if isinstance(node, ast.Call):
            node = node.func
        return node.id if isinstance(node, ast.Name) else ''
    selections = {
        'modeling_rope_utils.py': {'_compute_default_rope_parameters'},
        'masking_utils.py': {'causal_mask_function'},
        'modeling_qwen2.py': {'Qwen2MLP', 'rotate_half', 'apply_rotary_pos_emb', 'repeat_kv',
                             'eager_attention_forward', 'Qwen2Attention', 'Qwen2RMSNorm',
                             'Qwen2DecoderLayer', 'Qwen2RotaryEmbedding'},
    }
    for filename, selected in selections.items():
        data = (root / filename).read_bytes()
        if hashlib.sha256(data).hexdigest() != PINNED[filename]:
            raise ValueError('unqualified upstream source hash: ' + filename)
        nodes = [n for n in ast.parse(data).body
                 if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in selected]
        if {n.name for n in nodes} != selected:
            raise ValueError('missing selected upstream definition')
        future = ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)
        tree = ast.Module(body=[future, *nodes], type_ignores=[])
        tree = ast.fix_missing_locations(Decorators().visit(tree))
        exec(compile(tree, str(root / filename), 'exec'), ns)
        if filename == 'modeling_rope_utils.py':
            ns['ROPE_INIT_FUNCTIONS'] = {'default': ns['_compute_default_rope_parameters']}

    def create_causal_mask(*, config, input_embeds, attention_mask, cache_position,
                           past_key_values, position_ids):
        expected = torch.arange(39, device=input_embeds.device)
        if (tuple(input_embeds.shape[:2]) != (1, 39) or input_embeds.dtype != torch.float32
                or attention_mask is not None or past_key_values is not None
                or not torch.equal(cache_position, expected)
                or not torch.equal(position_ids, expected.unsqueeze(0))
                or config.layer_types != ['full_attention'] * config.num_hidden_layers):
            raise ValueError('bounded mask requires one unpadded39-token initial prefill')
        # Execute the actual upstream causal predicate; use the eager mask's
        # original F32 minimum and shape. Equivalence to the full constructor
        # is checked separately, rather than assumed from dimensions.
        allowed = ns['causal_mask_function'](0, 0, expected[:, None], expected[None, :])
        return torch.where(allowed, 0., torch.finfo(input_embeds.dtype).min)[None, None]

    def rounded_cache(config):
        class Cache:
            used = False

            def update(self, keys, values, layer_idx, cache_kwargs=None):
                if self.used or keys.shape[-2] != 39 or values.shape[-2] != 39:
                    raise ValueError('bounded cache supports one initial prefill update only')
                self.used = True
                keys, values = keys.half(), values.half()
                if not torch.isfinite(keys).all() or not torch.isfinite(values).all():
                    raise ValueError('nonfinite or overflowing F16 K/V')
                return keys.float(), values.float()
        return Cache()

    module.create_causal_mask = create_causal_mask
    module.rounded_cache = rounded_cache
    module.source_hashes = dict(PINNED)
    module.execution_scope = 'extracted upstream math; unpadded39-token/default-RoPE/eager/initial prefill only'
    return module
