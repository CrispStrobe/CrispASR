#!/usr/bin/env python3
"""Tiny actual original constructors/generate and literal factorial routing.

Neural steps and output decoding are stubs. No model quality/GPU acceptance.
Requires isolated reference Torch plus --blueprint and the pinned --config.
"""
import argparse
import ast
import importlib.util
import json
from pathlib import Path
import sys
import types


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--blueprint',type=Path,required=True)
    parser.add_argument('--config',type=Path,required=True)
    args=parser.parse_args()
    import torch
    torch.set_num_threads(1)
    sys.modules['dac']=types.ModuleType('dac')
    sys.modules['torchaudio']=types.ModuleType('torchaudio')
    sys.path.insert(0,str(args.blueprint))
    from dia.model import Dia
    from dia.config import DiaConfig
    from dia.state import DecoderOutput
    path=Path(__file__).resolve().parents[1]/'tools/kaggle/dia-cuda-factorial/dia_cuda_factorial.py'
    spec=importlib.util.spec_from_file_location('factorial_worker',path)
    worker=importlib.util.module_from_spec(spec);spec.loader.exec_module(worker)
    assert len(worker.ARMS)==5
    first=list(worker.ARMS.values())[:4]
    assert {(r['precision'],r['temperature'],r['top_k']) for r in first}=={
        (dtype,temp,k) for dtype in ['float16','float32'] for temp,k in [(1.3,35),(1.2,45)]}
    assert list(worker.ARMS.values())[-1]==first[0]
    values=json.loads(args.config.read_text())
    values['model']['encoder'].update(n_layer=1,n_embd=8,n_hidden=16,n_head=2,head_dim=4)
    values['model']['decoder'].update(n_layer=2,n_embd=8,n_hidden=16,gqa_query_heads=2,kv_heads=1,gqa_head_dim=4,cross_query_heads=2,cross_head_dim=4)
    values['data'].update(audio_length=128,text_length=128)
    config=DiaConfig.model_validate(values)
    reference=Dia(config,compute_dtype='float32',device=torch.device('cpu'))
    state={k:torch.full_like(v,.125) for k,v in reference.model.state_dict().items()}
    ledgers={};observed={}
    for precision in ['float16','float32']:
        model,ledger=worker.construct_model(Dia,config,state,precision,torch.device('cpu'))
        assert len(ledger)==47 and all(torch.all(v==.125).item() for v in model.model.state_dict().values())
        assert all(v==('torch.float32' if precision=='float32' or 'norm' in k else 'torch.float16') for k,v in ledger.items())
        ledgers[precision]=ledger
        # Actual original generate and DecoderOutput; only neural computation is stubbed.
        def prepare(text,audio,verbose):
            prefill,start=model._prepare_audio_prompt(audio)
            output=DecoderOutput.new(config,model.device);output.prefill(prefill,start)
            return types.SimpleNamespace(prepare_step=lambda *args:None),output
        model._prepare_generation=prepare
        model._generate_output=lambda codes:codes.cpu().numpy().copy()
        for name,arm in worker.ARMS.items():
            if arm['precision']!=precision:continue
            calls=[]
            def step(tokens,dec_state,cfg,temp,p,k):
                calls.append((cfg,temp,p,k));pred=torch.full((9,),568,dtype=torch.int64)
                if len(calls)==19:pred[0]=1024
                return pred
            model._decoder_step=step
            result=worker.generate_case(model,{'text':'[S1] tiny'},arm)
            assert result.shape==(32,9) and len(calls)==33
            assert all(row==(3.,arm['temperature'],.95,arm['top_k']) for row in calls)
            observed[name]=dict(actual_original_generate_steps=len(calls),parameters=calls[0])
    class Recorder:
        def generate(self,*args,**kwargs):self.args=args;self.kwargs=kwargs
    def check(fn):
        for arm in worker.ARMS.values():
            recorder=Recorder();fn(recorder,{'text':'fixed'},arm)
            assert recorder.args==('fixed',)
            assert recorder.kwargs==dict(max_tokens=3072,use_torch_compile=False,
                temperature=arm['temperature'],cfg_filter_top_k=arm['top_k'],cfg_scale=3.,top_p=.95)
    check(worker.generate_case)
    source=path.read_text();node=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='generate_case');original=ast.get_source_segment(source,node)
    changes=[('max_tokens=3072','max_tokens=3071'),('cfg_scale=3.','cfg_scale=4.'),
        ('top_p=.95','top_p=.9'),("cfg_filter_top_k=arm['top_k']",'cfg_filter_top_k=35'),
        ("temperature=arm['temperature']",'temperature=1.3'),('use_torch_compile=False','use_torch_compile=True')]
    for before,after in changes:
        assert before in original
        ns={};exec(compile(original.replace(before,after),'perturbed-driver','exec'),ns)
        try:check(ns['generate_case'])
        except AssertionError:pass
        else:raise AssertionError('Changed recipe accepted: '+before)
    print(json.dumps(dict(status='PASS',torch=torch.__version__,actual_original_parameters_per_constructor=47,
        precision_ledgers=ledgers,arms=observed,changed_recipe_negative_controls=6,
        neural_steps_and_pcm='stubs',full_model_acceptance=False)))


if __name__=='__main__':main()
