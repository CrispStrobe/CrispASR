#!/usr/bin/env python3
"""Bounded 39-token upstream Qwen2 prefill using exact GGUF weight values.

No full model construction: one decoder layer at a time, embeddings/head in row
chunks. This is a prepared diagnostic; running a real 7B replay requires the
corresponding gated native capture. Raw per-block taps are opt-in.
"""
import argparse
import gc
import hashlib
import inspect
import json
import sys
from pathlib import Path

import numpy as np


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def native_key(name):
    for before, after in [('self_attn.', 'attn.'), ('input_layernorm.', 'attn_ln.'),
                          ('post_attention_layernorm.', 'ffn_ln.'), ('mlp.', 'ffn.'),
                          ('gate_proj.', 'gate.'), ('up_proj.', 'up.'), ('down_proj.', 'down.')]:
        name = name.replace(before, after)
    return name


def compare_files(upstream_path, native_path):
    x = np.fromfile(upstream_path, dtype='<f4').astype('float64')
    y = np.fromfile(native_path, dtype='<f4').astype('float64')
    if x.shape != y.shape or not x.size or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('invalid comparison stage shape or values')
    return dict(cosine=float(x@y / max(np.linalg.norm(x)*np.linalg.norm(y), 1e-30)),
                upstream_norm=float(np.linalg.norm(x)), native_norm=float(np.linalg.norm(y)),
                relative_l2=float(np.linalg.norm(x-y) / max(np.linalg.norm(x), 1e-30)),
                max_abs=float(np.max(np.abs(x-y))))


def rounded_kv_cache(config):
    import torch
    from transformers.cache_utils import DynamicCache
    class RoundedKV(DynamicCache):
        def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
            # Match native F16 storage after RoPE for K and before attention for V.
            # Keep upstream attention in F32; this does not reproduce native flash
            # accumulation. A fresh cache is used for each layer's sole prefill.
            key16, value16 = key_states.half(), value_states.half()
            if not torch.isfinite(key16).all() or not torch.isfinite(value16).all():
                raise ValueError('nonfinite or overflowing F16 K/V')
            return super().update(key16.float(), value16.float(), layer_idx, cache_kwargs)
    return RoundedKV(config=config)


def replay(model_path, config, inputs, output, prompt_ids=None, kv_precision='f32'):
    if kv_precision not in ('f32', 'f16'):
        raise ValueError('kv_precision must be f32 or f16')
    import torch
    import gguf
    from gguf.quants import dequantize
    from transformers.models.qwen2 import modeling_qwen2 as upstream
    torch.set_num_threads(1)
    if not (0 < config.hidden_size <= 8192 and 0 < config.num_hidden_layers <= 128
            and 0 < config.vocab_size <= 262144):
        raise ValueError('configuration exceeds bounded base-model dimensions')
    if inputs.shape != (1, 39, config.hidden_size) or not np.isfinite(inputs).all():
        raise ValueError('require finite [1,39,hidden] effective prompt embeddings')
    if config.hidden_act != 'silu' or config.rope_scaling or config.use_sliding_window:
        raise ValueError('only native base full-attention/default-RoPE/SwiGLU configuration is covered')
    if config.num_hidden_layers * inputs.nbytes > 64 << 20:
        raise ValueError('per-block output payload exceeds 64 MiB')
    config._attn_implementation = 'eager'
    reader = gguf.GGUFReader(str(model_path))
    tensors = {t.name: t for t in reader.tensors}
    # Bind this configuration to the actual GGUF dimensions, avoiding an
    # accidental 1.5B configuration on a 7B source.
    expected = {'d_lm': config.hidden_size, 'd_ffn': config.intermediate_size,
                'n_lm_layers': config.num_hidden_layers, 'n_heads': config.num_attention_heads,
                'n_kv_heads': config.num_key_value_heads, 'vocab_size': config.vocab_size}
    for name, value in expected.items():
        field = reader.fields.get('vibevoice.' + name)
        if field is None or int(field.parts[-1].ravel()[0]) != value:
            raise ValueError('GGUF/config mismatch: ' + name)
    rope = reader.fields.get('vibevoice.rope_theta')
    if rope is None or float(rope.parts[-1].ravel()[0]) != config.rope_theta:
        raise ValueError('GGUF/config RoPE theta mismatch')
    layer_sizes = [sum(int(np.prod(t.shape)) * 4 for n, t in tensors.items()
                       if n.startswith(f'lm.layers.{i}.')) for i in range(config.num_hidden_layers)]
    if not all(0 < n <= 1536 << 20 for n in layer_sizes):
        raise ValueError('one decoder layer exceeds 1536 MiB dequantized-weight bound')
    def array(tensor, rows=None):
        data = tensor.data if rows is None else tensor.data[rows]
        result = dequantize(data, tensor.tensor_type)
        shape = tuple(reversed(tensor.shape))
        if rows is not None:
            shape = (-1, shape[-1])
        return torch.from_numpy(np.array(result.reshape(shape), dtype=np.float32, copy=True))
    def load(module, prefix):
        # Assign tensors individually into meta modules, retaining only one
        # layer plus the current tensor's dequantization temporary.
        for name in list(module.state_dict()):
            tensor = tensors[prefix + native_key(name)]
            value = array(tensor)
            parent, _, leaf = name.rpartition('.')
            target = module.get_submodule(parent) if parent else module
            target._parameters[leaf] = torch.nn.Parameter(value, requires_grad=False)
        return module.eval()
    embedding_gate = None
    if prompt_ids is not None:
        if prompt_ids.shape != (39,) or np.any(prompt_ids < 0) or np.any(prompt_ids >= config.vocab_size):
            raise ValueError('require39 valid prompt IDs')
        looked_up = array(tensors['lm.tok_emb.weight'], prompt_ids).numpy().reshape(inputs.shape)
        x, y = looked_up.astype('float64').ravel(), inputs.astype('float64').ravel()
        embedding_gate = dict(max_abs=float(np.max(np.abs(x-y))),
                              relative_l2=float(np.linalg.norm(x-y) / max(np.linalg.norm(x), 1e-30)))
        if embedding_gate['max_abs'] != 0:
            raise ValueError('effective input differs from same-GGUF token embeddings; this neutral-only replay does not cover voice splicing')
    output.mkdir(parents=True, exist_ok=False)
    hidden = torch.from_numpy(inputs.copy())
    positions = torch.arange(39).unsqueeze(0)
    cache_position = positions[0]
    rotary = upstream.Qwen2RotaryEmbedding(config)
    norm = upstream.Qwen2RMSNorm(config.hidden_size, config.rms_norm_eps)
    norm.weight = torch.nn.Parameter(array(tensors['lm.norm.weight']), requires_grad=False)
    # This is the mask constructor used by actual upstream Qwen2Model.forward.
    mask = upstream.create_causal_mask(config=config, input_embeds=hidden,
        attention_mask=None, cache_position=cache_position, past_key_values=None, position_ids=positions)
    rows = []
    with torch.no_grad():
        position_embeddings = rotary(hidden, positions)
        for index in range(config.num_hidden_layers):
            with torch.device('meta'):
                layer = upstream.Qwen2DecoderLayer(config, index)
            load(layer, f'lm.layers.{index}.')
            cache = rounded_kv_cache(config) if kv_precision == 'f16' else None
            hidden = layer(hidden, attention_mask=mask, position_ids=positions,
                           past_key_values=cache, use_cache=cache is not None, cache_position=cache_position,
                           position_embeddings=position_embeddings)
            raw = hidden[0].numpy()
            raw.astype('<f4').tofile(output / f'prefill_block_{index:03}.bin')
            normalized = norm(hidden)[0, -1].numpy()
            normalized.astype('<f4').tofile(output / f'prefill_truncated_norm_{index:03}.bin')
            rows.append(dict(block=index, raw_norm=float(np.linalg.norm(raw.astype('float64'))),
                             last_normalized_norm=float(np.linalg.norm(normalized.astype('float64')))))
            del layer, cache
            gc.collect()
        last = norm(hidden)[0, -1]
        last.numpy().astype('<f4').tofile(output / 'tts_prefill_hidden.bin')
        head = tensors.get('lm_head.weight', tensors['lm.tok_emb.weight'])
        logits = []
        for start in range(0, config.vocab_size, 256):
            chunk = array(head, slice(start, min(start + 256, config.vocab_size)))
            logits.append((chunk @ last).numpy())
            del chunk
        np.concatenate(logits).astype('<f4').tofile(output / 'tts_prefill_logits.bin')
    receipt = dict(scope='actual upstream layer-at-a-time 39-token prefill; no real-model/native parity claim',
        kv_precision=kv_precision,
        kv_rounding=('post-RoPE K and unrotated V rounded through F16, then F32 upstream eager attention'
                     if kv_precision == 'f16' else 'unrounded upstream F32 K/V'),
        weights_sha256=sha(model_path), upstream_qwen2_sha256=sha(inspect.getfile(upstream)),
        config=config.to_dict(), largest_dequantized_layer_bytes=max(layer_sizes),
        head_row_chunk=256, raw_per_block_bytes=inputs.nbytes, blocks=rows, embedding_gate=embedding_gate,
        native_taps='compare captured effective input and gated post-residual raw block outputs; missing files remain explicit')
    (output / 'prefill-replay.json').write_text(json.dumps(receipt, indent=2) + '\n')
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gguf', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True, help='original config.json containing decoder_config')
    parser.add_argument('--capture', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--expected-sha256', required=True)
    parser.add_argument('--kv-precision', choices=('f32','f16'), default='f32',
                        help='f32 default control; f16 rounds K/V to match native base storage, not flash arithmetic')
    args = parser.parse_args()
    print('Prefill replay K/V mode: '+args.kv_precision+'; upstream eager attention does not reproduce native flash accumulation.', file=sys.stderr)
    if sha(args.gguf) != args.expected_sha256:
        parser.error('GGUF weight identity mismatch')
    from transformers import Qwen2Config
    config = Qwen2Config(**json.loads(args.config.read_text())['decoder_config'])
    dimensions = np.fromfile(args.capture / 'tts_prefill_dimensions.bin', dtype='<i4')
    if list(dimensions) != [39, config.vocab_size]:
        parser.error('require exact captured 39-token prefill and config vocabulary')
    path = args.capture / 'tts_prompt_embeds.bin'
    if path.stat().st_size != 39 * config.hidden_size * 4:
        parser.error('invalid effective prompt embedding size')
    inputs = np.fromfile(path, dtype='<f4').reshape(1, 39, config.hidden_size)
    ids_path = args.capture / 'tts_prompt_ids.bin'
    if ids_path.stat().st_size != 39 * 4:
        parser.error('require exact39 prompt IDs')
    prompt_ids = np.fromfile(ids_path, dtype='<i4')
    report = replay(args.gguf, config, inputs, args.output, prompt_ids, args.kv_precision)
    comparisons = []
    for name in [f'prefill_block_{i:03}' for i in range(config.num_hidden_layers)] + ['tts_prefill_hidden', 'tts_prefill_logits']:
        native_name = 'tts_' + name if name.startswith('prefill_block_') else name
        native = args.capture / (native_name + '.bin')
        if not native.exists():
            comparisons.append(dict(stage=native_name, status='MISSING'))
            continue
        comparisons.append(dict(stage=native_name, **compare_files(args.output / (name + '.bin'), native)))
    report.update(native_comparisons=comparisons, config_file_sha256=sha(args.config))
    (args.output / 'prefill-replay.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
