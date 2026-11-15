#!/usr/bin/env python3
"""Build one offline Kaggle code_file; the only dataset file is the real GGUF.

No upload/submission, network access, dependency installer or model read occurs.
Archive source comes from immutable git objects; installed runtime files are
individually hashed and retain distribution metadata/licenses.
"""
import argparse
import base64
import gzip
import hashlib
import importlib.metadata
import importlib.util
import io
import json
from pathlib import Path
import shutil
import subprocess
import tarfile

GGML='1b972bb754aa5f83e2d872ccf37a335717f6658d'
BLUEPRINT='16fb2cb1217c9934a886e1948ffb06120caa2df5'
PACKAGES={'transformers':'4.57.6','gguf':'0.18.0','tokenizers':'0.22.1','huggingface_hub':'0.36.2','safetensors':'0.7.0'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ggml-repo',required=True,type=Path)
    parser.add_argument('--blueprint-repo',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[3]
    commit=subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip()
    if shutil.disk_usage(args.output.parent).free<=12*2**30:raise RuntimeError('strict12GiB disk reserve')
    args.output.mkdir(parents=True,exist_ok=False)
    files={}
    def archive(repo,ref,paths,prefix):
        data=subprocess.check_output(['git','-C',str(repo),'archive','--format=tar',ref,*paths])
        with tarfile.open(fileobj=io.BytesIO(data),mode='r:') as source:
            for member in source:
                if member.isdir():continue
                if not member.isfile():raise RuntimeError('source symlinks are not allowed: '+member.name)
                files[prefix+member.name]=source.extractfile(member).read()
    archive(root,commit,['src','include','tools/vibevoice_prefill_replay.py','tools/vibevoice_first_frame_capture.py',
        'tools/kaggle/kaggle_harness.py','tools/kaggle/issue589-prefill-cpu/capture.cpp','LICENSE'],'source/')
    archive(args.ggml_repo,GGML,[], 'source/ggml/')
    config_path='vibevoice/configs/qwen2.5_7b_32k.json'
    files['blueprint-config.json']=subprocess.check_output(['git','-C',str(args.blueprint_repo),'show',BLUEPRINT+':'+config_path])
    package_versions={}
    for module,version in PACKAGES.items():
        distribution=importlib.metadata.distribution(module.replace('_','-'))
        if distribution.version!=version:raise RuntimeError('runtime version mismatch: '+module)
        location=Path(importlib.util.find_spec(module).origin).parent
        for path in location.rglob('*'):
            if path.is_file() and '__pycache__' not in path.parts and path.suffix not in ('.pyc','.pyo'):
                files['vendor/'+module+'/'+str(path.relative_to(location))]=path.read_bytes()
        metadata=Path(distribution._path)
        for path in metadata.rglob('*'):
            if path.is_file() and path.name not in ('direct_url.json','RECORD','INSTALLER','REQUESTED'):
                files['vendor/'+metadata.name+'/'+str(path.relative_to(metadata))]=path.read_bytes()
        package_versions[module]=version
    manifest=dict(source_commit=commit,ggml_commit=GGML,runtime_versions=package_versions,
        config_provenance=dict(blueprint_commit=BLUEPRINT,path=config_path,
            sha256=hashlib.sha256(files['blueprint-config.json']).hexdigest(),
            limitation='pinned blueprint config; not proven exact original GGUF conversion config'),
        files={name:hashlib.sha256(data).hexdigest() for name,data in files.items()})
    files['bundle-manifest.json']=(json.dumps(manifest,indent=2)+'\n').encode()
    uncompressed=sum(map(len,files.values()))
    if uncompressed>256<<20:raise RuntimeError('uncompressed bundle exceeds256MiB')
    target=io.BytesIO()
    with gzip.GzipFile(fileobj=target,mode='wb',mtime=0,compresslevel=6) as zipped:
        with tarfile.open(fileobj=zipped,mode='w|') as out:
            for name,data in sorted(files.items()):
                entry=tarfile.TarInfo(name);entry.size=len(data);entry.mode=0o644
                out.addfile(entry,io.BytesIO(data))
    payload=target.getvalue();digest=hashlib.sha256(payload).hexdigest()
    template=(Path(__file__).parent/'kernel_template.py').read_text()
    script=template.replace('__BUNDLE_B64__',base64.b64encode(payload).decode()).replace('__BUNDLE_SHA256__',digest).replace('__SOURCE_COMMIT__',commit)
    compile(script,'kernel.py','exec')
    (args.output/'kernel.py').write_text(script)
    (args.output/'bundle.tar.gz').write_bytes(payload)
    metadata=json.loads((Path(__file__).parent/'kernel-metadata.json').read_text())
    (args.output/'kernel-metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    receipt=dict(source_commit=commit,ggml_commit=GGML,embedded_bundle_sha256=digest,embedded_bundle_bytes=len(payload),
        kernel_bytes=len(script.encode()),kernel_sha256=hashlib.sha256(script.encode()).hexdigest(),
        uncompressed_bytes=uncompressed,file_count=len(files),runtime_versions=package_versions,
        config_provenance=manifest['config_provenance'],uploaded=False,submitted=False,
        model_dataset_files_required=['vibevoice-7b-q4_k.gguf'])
    (args.output/'package-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt,indent=2))


if __name__=='__main__':main()
