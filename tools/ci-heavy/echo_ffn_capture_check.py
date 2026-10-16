#!/usr/bin/env python3
"""CPU-only actual-source FFN capture controls; no model weights or quant claims."""
import ast
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.dont_write_bytecode=True
out=Path(os.environ['HEAVY_OUT']);scratch=Path(os.environ['HEAVY_SCRATCH'])
out.mkdir(parents=True,exist_ok=True);scratch.mkdir(parents=True,exist_ok=True)
source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
assert source==os.environ['GITHUB_SHA'] and not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT)
import torch
assert torch.__version__ in ['2.6.0','2.6.0+cpu'] and importlib.metadata.version('transformers')=='5.6.0'
report=dict(source=source,ggml=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT/'ggml',text=True).strip(),
    torch=torch.__version__,transformers=importlib.metadata.version('transformers'),platform=sys.platform,
    scope='Actual original class / ggml CPU tiny capture controls; no full model, Q4, GPU or speech acceptance',runs=[])

def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def checkpoint():
    (out/'receipt.json').write_text(json.dumps(report,indent=2)+'\n')

def run(label,argv):
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(ROOT),PYTEST_DISABLE_PLUGIN_AUTOLOAD='1',
        TMPDIR=str(scratch),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    with (out/(label+'.log')).open('xb') as log:
        result=subprocess.run(list(map(str,argv)),cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=3600)
    report['runs'].append(dict(label=label,argv=list(map(str,argv)),exit=result.returncode,sha256=sha(out/(label+'.log'))));checkpoint()
    print(label,result.returncode,flush=True);assert result.returncode==0,label

checkpoint()
method=out/'method';method.mkdir()
for name in ['src/index_echo.cpp','src/index_echo.h','src/core/index_echo_ffn_capture.h',
    'examples/cli/crispasr_diff_main.cpp','tools/reference_backends/index_echo.py',
    'tools/reference_backends/index_echo_ffn_capture.py','tests/test-index-echo-ffn-capture.cpp',
    'tests/test_index_echo_ffn_capture.py','tests/CMakeLists.txt','tools/ci-heavy/echo_ffn_capture_check.py',
    'tools/ci-heavy/echo_ffn_literal_graph.py','examples/talk-llama/llama-graph.cpp',
    'examples/talk-llama/llama-graph.h','examples/talk-llama/models/qwen35.cpp']:
    dest=method/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,dest)
# Retain the released source, not a handwritten substitute or dev distribution.
dist=importlib.metadata.distribution('transformers')
modeling=Path(dist.locate_file('transformers/models/qwen3_5/modeling_qwen3_5.py'))
shutil.copyfile(modeling,method/'original-modeling-qwen3_5.py')
shutil.copyfile(dist.locate_file('transformers/activations.py'),method/'original-activations.py')
text=modeling.read_text();node=next(n for n in ast.parse(text).body if isinstance(n,ast.ClassDef) and n.name=='Qwen3_5MLP')
report['original_modeling_sha256']=sha(modeling)
report['original_class_sha256']=hashlib.sha256(ast.get_source_segment(text,node).encode()).hexdigest()
for i, file in enumerate(f for f in dist.files if Path(f).name in ['LICENSE','LICENSE.txt','LICENSE.md','NOTICE']):
    shutil.copyfile(dist.locate_file(file),out/f'Transformers-LICENSE-{i}')
checkpoint()
run('original-controls',[sys.executable,'-m','pytest','tests/test_index_echo_ffn_capture.py','-v'])
build=scratch/'build'
run('configure',['cmake','-S',ROOT,'-B',build,'-DCMAKE_BUILD_TYPE=Release','-DGGML_NATIVE=OFF',
    '-DGGML_BLAS=OFF','-DGGML_METAL=OFF','-DGGML_CUDA=OFF','-DGGML_VULKAN=OFF','-DGGML_OPENMP=OFF',
    '-DCRISPASR_BUILD_TESTS=ON','-DCRISPASR_BUILD_SERVER=OFF','-DCRISPASR_OPUS=OFF','-DCRISPASR_AMR=OFF'])
run('build',['cmake','--build',build,'--target','test-index-echo-ffn-capture','-j3'])
run('native-controls',['ctest','--test-dir',build,'-R','^test-index-echo-ffn-capture$','--output-on-failure','--no-tests=error'])
exe=next(p for p in build.rglob('test-index-echo-ffn-capture') if p.is_file())
shutil.copyfile(exe,out/'native-controls-executable')
libs=out/'libs';libs.mkdir()
for pattern in ['*.dylib','*.so*']:
    for file in build.rglob(pattern):
        if file.is_file():
            target=libs/file.name
            if target.exists():assert sha(target)==sha(file)
            else:shutil.copyfile(file,target)
run('fetch-capture-baseline',['git','fetch','--depth=1','origin','d8183cf92abc24eeeb6c969db563e968c5345d46'])
run('literal-builder-controls',[sys.executable,ROOT/'tools/ci-heavy/echo_ffn_literal_graph.py',
    '--libraries',libs,'--output',out/'literal-builder'])
shutil.copyfile(ROOT/'LICENSE',out/'CrispASR-LICENSE');shutil.copyfile(ROOT/'ggml/LICENSE',out/'GGML-LICENSE')
report['passed']=True
report['payloads']=[dict(path=p.relative_to(out).as_posix(),bytes=p.stat().st_size,sha256=sha(p))
    for p in sorted(out.rglob('*')) if p.is_file() and p!=out/'receipt.json']
checkpoint();print('ECHO_FFN_TINY_ACTUAL_SOURCE_PASS',source,flush=True)
