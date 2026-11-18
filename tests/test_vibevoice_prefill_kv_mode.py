#!/usr/bin/env python3
"""Retained tiny F16-KV mode exactness/refusal controls; no native inference."""
import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixture',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if (args.fixture/'tiny.gguf').stat().st_size>1<<20:
        raise ValueError('tiny fixture only')
    import torch
    import gguf
    from transformers import Qwen2Config,Qwen2Model
    from transformers.cache_utils import DynamicCache
    torch.set_num_threads(1)
    root=Path(__file__).resolve().parents[1]
    spec=importlib.util.spec_from_file_location('prefill',root/'tools/vibevoice_prefill_replay.py')
    replay=importlib.util.module_from_spec(spec);spec.loader.exec_module(replay)
    config=Qwen2Config(hidden_size=32,intermediate_size=64,num_hidden_layers=2,num_attention_heads=4,
        num_key_value_heads=2,vocab_size=16,max_position_embeddings=64,rope_theta=1000000.,rms_norm_eps=1e-6,attention_dropout=0.)
    config._attn_implementation='eager'
    full=Qwen2Model(config).eval()
    reader=gguf.GGUFReader(str(args.fixture/'tiny.gguf'));tensors={t.name:t for t in reader.tensors}
    state={}
    for name in full.state_dict():
        key='lm.tok_emb.weight' if name=='embed_tokens.weight' else 'lm.'+replay.native_key(name)
        t=tensors[key]
        assert t.tensor_type==gguf.GGMLQuantizationType.F32
        state[name]=torch.from_numpy(np.array(t.data.reshape(tuple(reversed(t.shape))),copy=True))
    full.load_state_dict(state)
    inputs=np.fromfile(args.fixture/'inputs.bin',dtype='<f4').reshape(1,39,32)
    ids=np.arange(39,dtype=np.int64)%16
    args.output.mkdir(parents=True,exist_ok=False)
    # Independent full-model qualified control, retained from precision fixture.
    class QualifiedRoundedKV(DynamicCache):
        def update(self,key_states,value_states,layer_idx,cache_kwargs=None):
            return super().update(key_states.half().float(),value_states.half().float(),layer_idx,cache_kwargs)
    raw=[]
    hooks=[layer.register_forward_hook(lambda module,values,output:raw.append(output.detach().clone())) for layer in full.layers]
    with torch.no_grad():
        full(inputs_embeds=torch.from_numpy(inputs),use_cache=True,past_key_values=QualifiedRoundedKV(config=config))
    for hook in hooks:hook.remove()
    rounded=replay.replay(args.fixture/'tiny.gguf',config,inputs,args.output/'f16',ids,kv_precision='f16')
    default=replay.replay(args.fixture/'tiny.gguf',config,inputs,args.output/'f32-default',ids)
    matches=[]
    for index,value in enumerate(raw):
        path=args.output/'f16'/f'prefill_block_{index:03}.bin'
        actual=np.fromfile(path,dtype='<f4').reshape(1,39,32)
        assert np.array_equal(actual,value.numpy()),'rounded full/streamed raw block mismatch'
        reference=args.fixture/f'rounded-f16-depth{index+1}-hidden.bin'
        hidden=args.output/'f16'/f'prefill_truncated_norm_{index:03}.bin'
        comparison=replay.compare_files(reference,hidden)
        assert comparison['max_abs']==0
        matches.append(dict(stage=f'rounded_depth{index+1}_hidden',**comparison))
        # Default call must retain original upstream F32 blocks bit for bit.
        assert (args.output/'f32-default'/f'prefill_block_{index:03}.bin').read_bytes()==(args.fixture/'upstream'/f'prefill_block_{index:03}.bin').read_bytes()
    logits=replay.compare_files(args.fixture/'rounded-f16-depth2-logits.bin',args.output/'f16/tts_prefill_logits.bin')
    assert logits['max_abs']==0
    matches.append(dict(stage='rounded_depth2_logits',**logits))
    assert rounded['kv_precision']=='f16' and default['kv_precision']=='f32'
    refused=[]
    def refuse(name,call):
        try:call()
        except ValueError:refused.append(name)
        else:raise AssertionError('refusal missing: '+name)
    refuse('invalid_mode',lambda:replay.replay(args.fixture/'tiny.gguf',config,inputs,args.output/'bad-mode',ids,kv_precision='bf16'))
    refuse('wrong_token_count',lambda:replay.replay(args.fixture/'tiny.gguf',config,inputs[:,:38],args.output/'bad-count',ids,kv_precision='f16'))
    refuse('mismatched_effective_input',lambda:replay.replay(args.fixture/'tiny.gguf',config,inputs+1,args.output/'bad-input',ids,kv_precision='f16'))
    refuse('overflow_f16_kv',lambda:replay.rounded_kv_cache(config).update(torch.full((1,2,39,8),70000.),torch.ones(1,2,39,8),0))
    altered=np.fromfile(args.output/'f16/prefill_block_000.bin',dtype='<f4');altered[0]+=.25
    path=args.output/'perturbed.bin';altered.tofile(path)
    negative=replay.compare_files(args.output/'f16/prefill_block_000.bin',path)
    assert abs(negative['max_abs']-.25)<1e-6
    report=dict(scope='tiny upstream F16-KV mode qualification only; no7B/native run',
        raw_full_streamed_blocks_bit_exact=True,f32_default_retained_blocks_bit_exact=True,
        exact_retained_comparisons=matches,refusal_controls=refused,negative_control=negative,
        f16_receipt=rounded,f32_receipt=default)
    (args.output/'comparison.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2));print('PASS: F16-KV full/streamed/retained exact, F32 default unchanged, four refusals, perturbation detected')


if __name__=='__main__':main()
