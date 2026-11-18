#!/usr/bin/env python3
"""Bounded block-zero arithmetic controls over retained same-GGUF inputs.

This models the pinned CUDA Q4_K MMQ DS4 activation conversion on CPU, plus
separate Q/probability F16 controls. It does not emulate CUDA instruction/reduction
order, qualify a GPU kernel, or establish a speech-quality cause.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def q8_ds4(values):
    """Pinned quantize_mmq_q8_1 DS4 semantics, including original partial sums.

    Source: ggml 1b972bb, quantize.cu:457-552 and mmq.cuh:60-103.
    roundf ties go away from zero; scale is reciprocal-derived then F16.
    The sum follows the source's four-value threads and XOR 4,2,1 tree.
    CUDA arithmetic instruction selection is outside this CPU control's scope.
    """
    x = np.asarray(values)
    if x.dtype != np.float32 or x.ndim != 2 or x.shape[1] % 256 or not np.isfinite(x).all():
        raise ValueError('require finite F32 rows with 256-value groups')
    groups = x.reshape(x.shape[0], -1, 32)
    maximum = np.abs(groups).max(axis=-1)
    if np.any(maximum == 0):
        raise ValueError('zero-amplitude CUDA conversion is outside this control')
    inverse = np.float32(127) / maximum
    scaled = groups * inverse[..., None]
    # Do the rounding decision in F64 so adding .5 cannot round a F32 value
    # immediately below a half-integer onto the tie before floor.
    q = np.copysign(np.floor(np.abs(scaled).astype(np.float64) + .5), scaled).astype(np.int8)
    scale = (np.float32(1) / inverse).astype(np.float16)
    lanes = groups.reshape(*groups.shape[:-1], 8, 4)
    sums = ((lanes[..., 0] + lanes[..., 1]) + lanes[..., 2]) + lanes[..., 3]
    sums = (((sums[..., 0] + sums[..., 4]) + (sums[..., 2] + sums[..., 6]))
            + ((sums[..., 1] + sums[..., 5]) + (sums[..., 3] + sums[..., 7])))
    partial = sums.astype(np.float16)
    if not np.isfinite(scale).all() or not np.isfinite(partial).all() or np.any(scale == 0):
        raise ValueError('nonfinite/underflowing F16 DS4 metadata')
    return q, scale, partial


def mmq_q4k(tensor, values, torch, turing_subscales=False):
    """Integer dot products in exact-integer-range F32; CPU association afterward.

    Dots of 32 unsigned nibbles with signed Q8 values fit exactly in F32. The
    original unquantized partial sum is used for the Q4_K minimum term. Floating
    accumulation after those integer dots is a CPU algebra control, not a
    bit-exact reproduction of Turing MMA reduction order.
    """
    from gguf.quants import Q4_K
    rows, columns = map(int, reversed(tensor.shape))
    if int(tensor.tensor_type) != 12 or columns % 256:
        raise ValueError('require Q4_K matrix with complete 256-value groups')
    original_shape = values.shape[:-1]
    x = values.detach().float().reshape(-1, columns).numpy()
    qx, dy, sy = q8_ds4(x)
    blocks = np.asarray(tensor.data).reshape(-1, 144)
    dw = blocks[:, :2].copy().view(np.float16).astype(np.float32).reshape(rows, -1)
    dm = blocks[:, 2:4].copy().view(np.float16).astype(np.float32).reshape(rows, -1)
    sc, minimum = Q4_K.get_scale_min(blocks[:, 4:16].copy())
    sc = torch.from_numpy(sc.astype(np.float32).reshape(rows, -1, 8))
    minimum = torch.from_numpy(minimum.astype(np.float32).reshape(rows, -1, 8))
    codes = blocks[:, 16:].reshape(-1, 4, 1, 32) >> np.array([0, 4], np.uint8).reshape(1, 1, 2, 1)
    codes = torch.from_numpy((codes & 15).astype(np.float32).reshape(rows, -1, 32))
    quantized = torch.from_numpy(qx.astype(np.float32))
    dots = torch.bmm(codes.permute(1, 0, 2), quantized.permute(1, 2, 0))
    if turing_subscales:
        # TURING_MMA loader first multiplies d/dmin by six-bit scales in
        # half2 (mmq-load-tiles.cuh:780-784), then converts them back to F32.
        da = (dw[..., None]*sc.numpy()).astype(np.float16).astype(np.float32).reshape(rows, -1)
        db = (-dm[..., None]*minimum.numpy()).astype(np.float16).astype(np.float32).reshape(rows, -1)
        dy32, sy32 = dy.astype(np.float32), sy.astype(np.float32)
        result = torch.zeros(rows, x.shape[0], dtype=torch.float32)
        for block in range(dots.shape[0]):
            factor = torch.from_numpy(da[:, block, None]*dy32[None, :, block])
            # The factor is the exact product of two F16 values. Its F32
            # product with an integer dot fits in F64; round after addition
            # to model the source's contracted FMA without claiming GPU order.
            result = (result.double()+factor.double()*dots[block].double()).float()
            result = result+torch.from_numpy(db[:, block, None]*sy32[None, :, block])
        return result.T.reshape(*original_shape, rows)
    dots = dots.reshape(-1, 8, rows, x.shape[0]).permute(2, 3, 0, 1)
    # Native source applies the integer six-bit scale before the Q8 multiplier.
    positive = dots * sc[:, None] * torch.from_numpy(dy.astype(np.float32)).reshape(1, x.shape[0], -1, 8)
    negative = minimum[:, None] * torch.from_numpy(sy.astype(np.float32)).reshape(1, x.shape[0], -1, 8)
    # Keep this association explicit; the actual MMA instruction order is not claimed.
    a, b = positive[..., 0], negative[..., 0]
    for index in range(1, 8):
        a, b = a + positive[..., index], b + negative[..., index]
    grouped = a * torch.from_numpy(dw)[:, None] - b * torch.from_numpy(dm)[:, None]
    result = grouped[..., 0]
    for index in range(1, grouped.shape[-1]):
        result = result + grouped[..., index]
    return result.T.reshape(*original_shape, rows)


def scalar_mmq_control(torch):
    """Independent scalar nibble/scale decoding versus the batched CPU algebra."""
    import struct
    from types import SimpleNamespace
    rng = np.random.default_rng(589)
    raw = rng.integers(0, 256, (4, 144), dtype=np.uint8)
    for row in raw:
        row[:2] = np.frombuffer(struct.pack('<e', .03173828125), np.uint8)
        row[2:4] = np.frombuffer(struct.pack('<e', .01611328125), np.uint8)
    x = rng.standard_normal((3, 256)).astype(np.float32)
    q, d, partial = q8_ds4(x)
    fake = SimpleNamespace(data=raw, tensor_type=12, shape=(256, 4))
    actual = mmq_q4k(fake, torch.from_numpy(x), torch).numpy()
    turing = mmq_q4k(fake, torch.from_numpy(x), torch, True).numpy()
    expected = np.empty_like(actual)
    expected_turing = np.empty_like(actual)
    for token in range(3):
        for row in range(4):
            scales = raw[row, 4:16].tolist()
            positive = negative = rounded = np.float32(0)
            for sub in range(8):
                if sub < 4:
                    scale, minimum = scales[sub] & 63, scales[sub+4] & 63
                else:
                    scale = (scales[sub+4] & 15) | ((scales[sub-4] >> 6) << 4)
                    minimum = (scales[sub+4] >> 4) | ((scales[sub] >> 6) << 4)
                dot = 0
                for k in range(32):
                    code = (int(raw[row, 16+(sub//2)*32+k]) >> (4*(sub%2))) & 15
                    dot += code * int(q[token, sub, k])
                rounded_d = float(np.float16(.03173828125*scale))
                rounded_m = float(np.float16(-.01611328125*minimum))
                factor = np.float32(rounded_d*float(d[token, sub]))
                rounded = np.float32(float(rounded)+float(factor)*dot)
                rounded = np.float32(rounded+np.float32(rounded_m*float(partial[token, sub])))
                positive = np.float32(positive + np.float32(float(d[token, sub])*int(scale*dot)))
                negative = np.float32(negative + np.float32(float(partial[token, sub])*minimum))
            expected[token, row] = np.float32(np.float32(.03173828125*positive) - np.float32(.01611328125*negative))
            expected_turing[token, row] = rounded
    if not np.array_equal(turing, expected_turing) or np.array_equal(actual, turing):
        raise ValueError('Turing half2 subscale scalar control or rounding counterexample failed')
    if not np.array_equal(actual, expected):
        raise ValueError('independent scalar MMQ control failed: '+str(np.max(np.abs(actual-expected))))
    return dict(scope='tiny CPU algebra only; no CUDA execution', rows=4, tokens=3, columns=256,
                scalar_batched_bit_exact=True, turing_half2_subscale_scalar_bit_exact=True, turing_rounding_changes_result=True, negative_control='original partial sums differ from reconstructed Q8 sums')


def compare(a, b):
    x, y = np.asarray(a, np.float64).ravel(), np.asarray(b, np.float64).ravel()
    if x.shape != y.shape or not x.size or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('invalid comparison operands')
    return dict(max_abs=float(np.max(np.abs(x-y))),
                relative_l2=float(np.linalg.norm(x-y)/max(np.linalg.norm(x), 1e-30)),
                cosine=float(x@y/max(np.linalg.norm(x)*np.linalg.norm(y), 1e-30)))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('gguf', 'config', 'capture', 'qwen2_source', 'output'):
        p.add_argument('--' + name.replace('_', '-'), type=Path, required=True)
    p.add_argument('--expected-sha256', required=True)
    p.add_argument('--expected-baseline-sha256', required=True)
    p.add_argument('--turing-only', action='store_true', help='baseline plus added Turing half2-subscale controls; prior DS4 controls remain reproducible by default')
    args = p.parse_args()
    import torch
    import gguf
    from gguf.quants import dequantize
    import vibevoice_prefill_replay as reference
    import vibevoice_qwen2_bounded_upstream as adapter
    torch.set_num_threads(1)
    scalar_control = scalar_mmq_control(torch)
    if reference.sha(args.gguf) != args.expected_sha256:
        p.error('same-GGUF identity mismatch before open')
    cfg = adapter.config_from_dict(json.loads(args.config.read_text())['decoder_config'])
    cfg._attn_implementation = 'eager'
    if (cfg.hidden_size, cfg.num_hidden_layers, cfg.num_attention_heads, cfg.num_key_value_heads) != (3584, 28, 28, 4):
        p.error('control is restricted to retained actual7B block0')
    source = adapter.load(args.qwen2_source)
    reader = gguf.GGUFReader(str(args.gguf))
    tensors = {t.name: t for t in reader.tensors}
    checks = dict(d_lm=cfg.hidden_size, d_ffn=cfg.intermediate_size, n_lm_layers=cfg.num_hidden_layers,
                  n_heads=cfg.num_attention_heads, n_kv_heads=cfg.num_key_value_heads, vocab_size=cfg.vocab_size)
    for key, value in checks.items():
        if int(reader.fields['vibevoice.' + key].parts[-1].ravel()[0]) != value:
            p.error('configuration binding mismatch: ' + key)
    if float(reader.fields['vibevoice.rope_theta'].parts[-1].ravel()[0]) != cfg.rope_theta:
        p.error('configuration RoPE mismatch')
    layer_bytes = sum(int(np.prod(t.shape))*4 for name,t in tensors.items() if name.startswith('lm.layers.0.'))
    if not 0 < layer_bytes <= 1536 << 20:
        p.error('one layer exceeds unchanged1536MiB reference bound')
    def array(t):
        return np.array(dequantize(t.data, t.tensor_type).reshape(tuple(reversed(t.shape))), np.float32, copy=True)
    ids = np.fromfile(args.capture/'tts_prompt_ids.bin', dtype='<i4')
    inputs = np.fromfile(args.capture/'tts_prompt_embeds.bin', dtype='<f4').reshape(1, 39, cfg.hidden_size)
    embedding = tensors['lm.tok_emb.weight']
    if ids.shape != (39,) or np.any(ids < 0) or np.any(ids >= cfg.vocab_size):
        p.error('require exact39 valid prompt IDs')
    looked_up = dequantize(embedding.data[ids], embedding.tensor_type).reshape(inputs.shape)
    if not np.array_equal(inputs, looked_up):
        p.error('original exact effective-input guard failed')
    native = np.fromfile(args.capture/'tts_prefill_block_000.bin', dtype='<f4').reshape(inputs.shape)
    args.output.mkdir(parents=True, exist_ok=False)
    with torch.device('meta'):
        layer = source.Qwen2DecoderLayer(cfg, 0)
    for name in list(layer.state_dict()):
        parent, _, leaf = name.rpartition('.')
        owner = layer.get_submodule(parent) if parent else layer
        value = array(tensors['lm.layers.0.'+reference.native_key(name)])
        if tuple(value.shape) != tuple(owner._parameters[leaf].shape):
            p.error('parameter shape mismatch: '+name)
        owner._parameters[leaf] = torch.nn.Parameter(torch.from_numpy(value), requires_grad=False)
    layer.eval()
    rotary = source.Qwen2RotaryEmbedding(cfg)
    hidden = torch.from_numpy(inputs.copy())
    positions = torch.arange(39).unsqueeze(0)
    mask = source.create_causal_mask(config=cfg, input_embeds=hidden, attention_mask=None,
                                    cache_position=positions[0], past_key_values=None, position_ids=positions)
    original_rope, original_attention = source.apply_rotary_pos_emb, source.eager_attention_forward
    original_softmax = torch.nn.functional.softmax
    modes = [('baseline_f16kv', False, False, False), ('mmq_ds4', True, False, False),
             ('f16q_probability', False, True, False), ('mmq_ds4_f16q_probability', True, True, False)]
    turing_modes = [('mmq_turing_half2', True, False, True), ('mmq_turing_half2_f16q_probability', True, True, True)]
    modes = modes[:1]+turing_modes if args.turing_only else modes+turing_modes
    report = dict(scope=__doc__, scalar_mmq_control=scalar_control, weights_sha256=args.expected_sha256, upstream_source_hashes=source.source_hashes,
                  cpu_torch=torch.__version__, modes=[], limitation='CPU algebra and explicit F16 rounding; not actual CUDA instruction/reduction-order emulation or a quality cause')
    for label, use_mmq, flash_half, turing_subscales in modes:
        outdir = args.output/label;outdir.mkdir()
        def save(name, value):
            data = value.detach().float().contiguous().numpy()
            if data.nbytes > 64 << 20 or not np.isfinite(data).all():
                raise ValueError('invalid bounded intermediate')
            data.astype('<f4').tofile(outdir/(name+'.bin'))
        for name, module in layer.named_modules():
            if not isinstance(module, torch.nn.Linear):
                continue
            tensor = tensors['lm.layers.0.'+reference.native_key(name+'.weight')]
            if int(tensor.tensor_type) != 12:
                raise ValueError('unexpected non-Q4_K linear: '+name)
            def forward(value, module=module, tensor=tensor, name=name):
                if name.endswith('q_proj'):
                    save('q_proj_input', value)
                if use_mmq:
                    result = mmq_q4k(tensor, value, torch, turing_subscales)
                    if module.bias is not None:
                        result = result + module.bias
                else:
                    result = torch.nn.functional.linear(value, module.weight, module.bias)
                save(name.replace('.', '_')+'_output', result)
                return result
            module.forward = forward
        def rope(*values, **kwargs):
            q, k = original_rope(*values, **kwargs)
            if flash_half:q = q.half().float()
            save('rope_q',q);save('rope_k',k)
            return q,k
        def attention(*values, **kwargs):
            result = original_attention(*values, **kwargs)
            save('attention_output',result[0])
            return result
        def softmax(*values, **kwargs):
            value = original_softmax(*values, **kwargs)
            return value.half().float() if flash_half else value
        source.apply_rotary_pos_emb, source.eager_attention_forward = rope, attention
        torch.nn.functional.softmax = softmax
        try:
            with torch.no_grad():
                actual = layer(hidden,attention_mask=mask,position_ids=positions,cache_position=positions[0],
                               position_embeddings=rotary(hidden,positions),past_key_values=source.rounded_cache(cfg),use_cache=True)
            save('raw_block_000',actual)
        finally:
            source.apply_rotary_pos_emb, source.eager_attention_forward = original_rope, original_attention
            torch.nn.functional.softmax = original_softmax
        digest = reference.sha(outdir/'raw_block_000.bin')
        row = dict(mode=label, output_sha256=digest, native=compare(actual.numpy(),native))
        if not report['modes']:
            if digest != args.expected_baseline_sha256:
                row['baseline_binding']='FAIL: no further modified-arithmetic controls are qualified'
                report['modes'].append(row)
                (args.output/'comparison.json').write_text(json.dumps(report,indent=2)+'\n')
                raise ValueError('baseline block0 is not byte-identical to retained full7B F16-KV replay')
            row['baseline_binding']='bit-exact retained full7B F16-KV upstream block0'
        report['modes'].append(row)
        (args.output/'comparison.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(row),flush=True)
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
