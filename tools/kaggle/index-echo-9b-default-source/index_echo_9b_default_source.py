#!/usr/bin/env python3
"""Released-default BF16 full-file control, resident on actual GPUs.

Independent source class only. F32 numerical references remain separate.
"""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import torch

SCRIPT_VERSION = '2026-10-02.2'
SOURCE_COMMIT = 'e56353137b967acbc00ffbedeb6b55429c0add4b'
SOURCE_REVISION = 'b8ac6fb7d3dc17cee48a52201bd3d93dc86b0dba'
TEMP = Path('/kaggle/temp/index-echo-default-source')
OUT = Path('/kaggle/working')
ROOT = TEMP / 'repo'
TEMP.mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)
os.environ['TMPDIR'] = str(TEMP)
hardware = subprocess.check_output(['nvidia-smi', '--query-gpu=name,compute_cap,memory.total', '--format=csv,noheader'], text=True).strip()
print('actual GPU:', hardware, 'torch:', torch.__version__, flush=True)
if not torch.cuda.is_available() or sum(torch.cuda.get_device_properties(i).total_memory for i in range(torch.cuda.device_count())) < 24 * 2**30:
    raise RuntimeError('Inconclusive: original resident BF16 source needs 24 GiB aggregate VRAM')
# Check installed BF16 GEMM support on the actual hardware before weight pulls.
a = torch.randn(1024, 1024, device='cuda:0', dtype=torch.bfloat16)
b = a @ a
torch.cuda.synchronize()
del a, b


def run(*args): subprocess.run(list(map(str,args)), check=True)


run('git','init',ROOT)
run('git','-C',ROOT,'remote','add','origin','https://github.com/CrispStrobe/CrispASR.git')
run('git','-C',ROOT,'fetch','--depth=1','origin',SOURCE_COMMIT)
run('git','-C',ROOT,'checkout','FETCH_HEAD')
sys.path.insert(0,str(ROOT/'tools/kaggle'))
import kaggle_harness as kh
kh.init_progress()
kh.provenance(SCRIPT_VERSION,ROOT)
run(sys.executable,'-m','pip','install','-q','transformers==5.6.0','accelerate','gguf','huggingface_hub','librosa>=0.10','soundfile>=0.12','safetensors>=0.5','silero-vad>=5.1','--no-deps')
run(sys.executable,'-m','pip','install','-q','tokenizers','huggingface_hub','psutil','packaging','pyyaml','regex','safetensors','numpy','tqdm','onnxruntime','filelock','requests')
from huggingface_hub import HfApi,snapshot_download
api=HfApi(token=kh.resolve_hf_token(require=True))
os.environ['HF_TOKEN']=api.token
memory={str(i):f'{int(torch.cuda.mem_get_info(i)[0]/2**30)-(2 if i==0 else 1)}GiB' for i in range(torch.cuda.device_count())}
memory['cpu']='0GiB'
os.environ.update(INDEX_ECHO_REF_DEVICE='cuda:0',INDEX_ECHO_REF_DTYPE='bfloat16',INDEX_ECHO_REF_DEVICE_MAP='auto',INDEX_ECHO_REF_MAX_MEMORY=json.dumps(memory),INDEX_ECHO_REF_OFFLOAD_DIR=str(TEMP/'offload'),INDEX_ECHO_REF_THREADS='4')
source=Path(snapshot_download('IndexTeam/Index-Echo-S2TT-9B',revision=SOURCE_REVISION,local_dir=TEMP/'source'))
sys.path.insert(0,str(ROOT/'tools'))
from reference_backends.index_echo import dump_pipeline
def checkpoint(pipeline,multi):
    api.upload_file(path_or_fileobj=pipeline,path_in_repo='index-echo-9b-bf16-default-zh-context/pipeline/partial.json',repo_id='cstr/crispasr-regression-fixtures')
    kh.step('source.case.checkpoint',cases=list(json.loads(pipeline.read_text())['cases']))

with kh.build_heartbeat('released.default-bf16.pipeline',interval_s=30):
    pipeline,multi=dump_pipeline(source,OUT,ROOT/'samples',checkpoint=checkpoint,context_audio='zh-pause')
r=json.loads(pipeline.read_text())
assert len(r['cases'])==5
assert all(case['segments'] and not any(row.get('parse_warn',0) for row in case['rows']) for case in r['cases'].values())
assert isinstance(r['reference_placement'],dict) and not any(v in ['cpu','disk'] for v in r['reference_placement'].values()),'Resident control must not offload'
for module in ['tower','connector','llm']:
    assert list(r['parameter_dtypes'][module]['parameter_elements'])==['torch.bfloat16']
receipt=dict(script_version=SCRIPT_VERSION,source_commit=SOURCE_COMMIT,source_revision=SOURCE_REVISION,hardware=hardware,torch=torch.__version__,memory=memory,validated=False,source_control_passed=True,source_pipeline=r)
for path,remote in [(pipeline,'index-echo-9b-bf16-default-zh-context/pipeline/reference.json'),(multi,'index-echo-9b/pipeline-zh-context/audio.wav')]:
    api.upload_file(path_or_fileobj=path,path_in_repo=remote,repo_id='cstr/crispasr-regression-fixtures')
receipt['fixture_revision']=api.model_info('cstr/crispasr-regression-fixtures').sha
(OUT/'default-source-receipt.json').write_text(json.dumps(receipt,indent=2,ensure_ascii=False)+'\n')
kh.step('released.default-bf16.complete',fixture_revision=receipt['fixture_revision'])
shutil.rmtree(source)
