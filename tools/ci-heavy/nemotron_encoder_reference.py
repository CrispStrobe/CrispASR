#!/usr/bin/env python3
"""Capture original NVIDIA Conformer layers before choosing further Q4 guards."""
import hashlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import subprocess
import wave

import numpy as np
from huggingface_hub import snapshot_download
import torch
from transformers import AutoModelForRNNT, AutoProcessor

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(os.environ['HEAVY_OUT']);OUT.mkdir(parents=True,exist_ok=True)
TEMP=Path(os.environ['HEAVY_SCRATCH'])/'nemotron-encoder-reference';TEMP.mkdir(parents=True,exist_ok=True)
os.environ.update(HF_HOME=str(TEMP/'hf'),HF_XET_CACHE=str(TEMP/'xet'))
MODEL='nvidia/nemotron-3.5-asr-streaming-0.6b'
REV='ea30d66debe3740a08b573244286791d423d6b3e'
assert importlib.metadata.version('transformers')=='5.19.0'
torch.set_num_threads(4);torch.set_num_interop_threads(1)
def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024**2),b''):h.update(block)
    return h.hexdigest()
model_dir=snapshot_download(MODEL,revision=REV,allow_patterns=['*.json','*.safetensors'])
assert digest(Path(model_dir)/'model.safetensors')=='9eebdd6590289cb3030f310858f3df93256600a800a3e8200c5993d5f967e174'
processor=AutoProcessor.from_pretrained(model_dir,local_files_only=True)
model=AutoModelForRNNT.from_pretrained(model_dir,local_files_only=True,torch_dtype=torch.float32).eval()
assert next(model.parameters()).device.type=='cpu'
processor.set_num_lookahead_tokens(3)
with wave.open(str(ROOT/'samples/jfk.wav')) as wav:
    assert wav.getframerate()==16000 and wav.getnchannels()==1 and wav.getsampwidth()==2
    signal=np.frombuffer(wav.readframes(wav.getnframes()),dtype='<i2').astype(np.float32)/32768
inputs=processor(signal,sampling_rate=16000,language='en-US',return_tensors='pt').to(model.device,dtype=model.dtype)
# Read and archive the exact driving and block code used by the original run.
for name,entity in [('encoder',type(model.encoder)),('block',type(model.encoder.layers[0])),('generation',model.generate)]:
    (OUT/(name+'-blueprint.py')).write_text(inspect.getsource(entity))
modules={name:type(module).__name__ for name,module in model.named_modules()}
(OUT/'modules.json').write_text(json.dumps(modules,indent=2)+'\n')
assert len(model.encoder.layers)==24
with torch.inference_mode():
    control=model.generate(**inputs,return_dict_in_generate=True,max_new_tokens=512)
    control_ids=control.sequences.detach().clone()
arrays={};counts={};hooks=[]
def capture(name):
    def hook(module,args,output):
        value=getattr(output,'last_hidden_state',None)
        if value is None:value=output[0] if isinstance(output,tuple) else output
        assert isinstance(value,torch.Tensor),name
        # Subsampling is subsequently scaled in place; clone immediately.
        array=value.detach().clone().cpu().float().numpy()
        assert np.isfinite(array).all()
        arrays[name]=array;counts[name]=counts.get(name,0)+1
    return hook
hooks.append(model.encoder.subsampling.register_forward_hook(capture('preencode')))
hooks.append(model.encoder.register_forward_hook(capture('encoder')))
for index,layer in enumerate(model.encoder.layers):hooks.append(layer.register_forward_hook(capture('layer_'+str(index))))
try:
    with torch.inference_mode():output=model.generate(**inputs,return_dict_in_generate=True,max_new_tokens=512)
finally:
    for hook in hooks:hook.remove()
assert torch.equal(control_ids,output.sequences),'Capture changed decoded sequence'
assert len(arrays)==26 and all(count==1 for count in counts.values()),counts
assert np.array_equal(arrays['layer_23'],arrays['encoder']),'Final block and stock encoder differ'
for key,value in inputs.items():
    if isinstance(value,torch.Tensor):arrays['input_'+key]=value.detach().cpu().numpy()
arrays['sequence']=output.sequences.cpu().numpy();arrays['pcm']=signal
archive=OUT/'original-encoder-layers.npz';np.savez_compressed(archive,**arrays)
receipt=dict(diagnostic_complete=True,scope=__doc__,source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
    original_model=MODEL,original_revision=REV,dtype='torch.float32',device='cpu',lookahead_tokens=3,language='en-US',
    transcript=processor.batch_decode(output.sequences,skip_special_tokens=True)[0],capture_sequence_exact=True,
    original_attention_implementation=model.encoder.config._attn_implementation,
    input_scale=model.encoder.input_scale,pcm_sha256=hashlib.sha256(signal.tobytes()).hexdigest(),
    archive_sha256=digest(archive),stages={name:dict(shape=list(array.shape),norm=float(np.linalg.norm(array.astype(np.float64)))) for name,array in arrays.items()},
    packages={name:importlib.metadata.version(name) for name in ['torch','transformers','numpy','huggingface_hub']},
    native_parity_tested=False,quant_promoted=False)
(OUT/'encoder-reference.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt,indent=2),flush=True)
