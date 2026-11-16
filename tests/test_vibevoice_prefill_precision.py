#!/usr/bin/env python3
"""Explain retained tiny native prefill boundaries with upstream K/V rounding.

Consumes the existing generated fixture/native outputs. No native rebuild or
native rerun. The original F32 reference failure remains recorded unchanged.
"""
import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixture', type=Path, required=True)
    parser.add_argument('--result', type=Path, required=True)
    args = parser.parse_args()
    if args.result.exists():
        raise FileExistsError(args.result)
    import torch
    import gguf
    from transformers import Qwen2Config, Qwen2Model
    from transformers.cache_utils import DynamicCache
    torch.set_num_threads(1)
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location('prefill', root/'tools/vibevoice_prefill_replay.py')
    replay = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(replay)
    config = Qwen2Config(hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, vocab_size=16,
        max_position_embeddings=64, rope_theta=1000000., rms_norm_eps=1e-6, attention_dropout=0.)
    config._attn_implementation = 'eager'
    model = Qwen2Model(config).eval()
    reader = gguf.GGUFReader(str(args.fixture/'tiny.gguf'))
    tensors = {t.name:t for t in reader.tensors}
    if len(tensors) != 26 or tuple(tensors['lm.tok_emb.weight'].shape) != (32,16):
        raise ValueError('only the retained tiny32/2-layer fixture is covered')
    state = {}
    for name in model.state_dict():
        key = 'lm.tok_emb.weight' if name == 'embed_tokens.weight' else 'lm.'+replay.native_key(name)
        tensor = tensors[key]
        if tensor.tensor_type != gguf.GGMLQuantizationType.F32:
            raise ValueError('tiny control requires original F32 GGUF tensors')
        state[name] = torch.from_numpy(np.array(tensor.data.reshape(tuple(reversed(tensor.shape))),copy=True))
    model.load_state_dict(state)
    class RoundedKV(DynamicCache):
        def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
            # Native writes post-RoPE K and unrotated V to F16 KV, then attention
            # reads those rounded values. No query/activation/weight cast is added.
            return super().update(key_states.half().float(), value_states.half().float(), layer_idx, cache_kwargs)
    inputs = torch.from_numpy(np.fromfile(args.fixture/'inputs.bin',dtype='<f4').reshape(1,39,32))
    natives = {name:np.fromfile(args.fixture/file,dtype='<f4').reshape(2,48)
               for name,file in [('flash_default','native.bin'),('flash_off','native-no-flash.bin')]}
    rows=[]
    raw=[]
    hooks=[layer.register_forward_hook(lambda module,values,output:raw.append(output.detach().clone())) for layer in model.layers]
    with torch.no_grad():
        model(inputs_embeds=inputs,use_cache=True,past_key_values=RoundedKV(config=config))
        for index in range(2):
            rounded = model.norm(raw[index])[0,-1].numpy()
            rounded_logits=(model.embed_tokens.weight @ torch.from_numpy(rounded)).numpy()
            rounded.astype('<f4').tofile(args.fixture/f'rounded-f16-depth{index+1}-hidden.bin')
            rounded_logits.astype('<f4').tofile(args.fixture/f'rounded-f16-depth{index+1}-logits.bin')
            original=np.fromfile(args.fixture/'upstream'/f'prefill_truncated_norm_{index:03}.bin',dtype='<f4')
            original_logits=(model.embed_tokens.weight @ torch.from_numpy(original)).numpy()
            for mode,native in natives.items():
                for precision,hidden,logits in [('original_F32_KV',original,original_logits),('rounded_F16_KV',rounded,rounded_logits)]:
                    for label,reference,mine in [('hidden',hidden,native[index,:32]),('logits',logits,native[index,32:])]:
                        x,y=reference.astype('float64'),mine.astype('float64')
                        relative=float(np.linalg.norm(x-y)/np.linalg.norm(x))
                        cosine=float(x@y/np.linalg.norm(x)/np.linalg.norm(y))
                        rows.append(dict(native_mode=mode,precision=precision,depth=index+1,stage=label,cosine=cosine,relative_l2=relative,
                            upstream_norm=float(np.linalg.norm(x)),native_norm=float(np.linalg.norm(y)),
                            max_abs=float(np.max(np.abs(x-y))),passed=cosine>.999999 and relative<.001))
    for hook in hooks:hook.remove()
    boundary=args.fixture/'upstream/prefill_block_000.bin'
    values=np.fromfile(boundary,dtype='<f4');values[0]+=.25
    perturbed=args.fixture/'perturbed-block.bin';values.tofile(perturbed)
    negative=replay.compare_files(boundary,perturbed)
    assert abs(negative['max_abs']-.25)<1e-6 and negative['relative_l2']>.001
    original_failed=[r for r in rows if r['native_mode']=='flash_default' and r['precision']=='original_F32_KV' and not r['passed']]
    rounded_failed=[r for r in rows if r['native_mode']=='flash_off' and r['precision']=='rounded_F16_KV' and not r['passed']]
    report=dict(scope='tiny39-token precision control only; unchanged native outputs; no7B parity',
        thresholds=dict(cosine_min=.999999,relative_l2_max=.001),
        weights_sha256=replay.sha(args.fixture/'tiny.gguf'),original_f32_failed=original_failed,
        rounded_f16_failed=rounded_failed,boundaries=rows,negative_control=negative)
    args.result.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
    assert original_failed,'the original failing F32 comparison must remain visible'
    assert not rounded_failed,'matching KV precision and nonflash native must pass unchanged gates'
    print('PASS: retained default-flash originalF32 FAIL; roundedF16 versus nonflash passes; negative perturbation detected')


if __name__=='__main__':main()
