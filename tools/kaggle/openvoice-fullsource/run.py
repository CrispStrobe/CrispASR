#!/usr/bin/env python3
"""Whole current application on real CUDA, reusing only audited ggml DSOs."""
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tarfile
import urllib.request

SOURCE = os.environ['CRISPASR_VALIDATION_SOURCE']
RECIPIENT = os.environ['CRISPASR_AUDIO_RECIPIENT']
HELPER_SOURCE = 'd84007612167105cf24c7af80014d1f0ea1bd8a3'
HELPER_SHA = 'b1fed1e4518ceeb517c749be67ce3eeee9d3243153b5f50043f6edf9c4584f20'
SDK_NAME = 'libcrispasr-linux-x86_64-cuda'
SDK_SHA = '8fbd68e1822cd16e58055753fc67120c20784451e286d94177f7f7fb78b76cbd'
GGML = 'e2ae679a563014f46c546ce6c6448e8601e4dea6'
WORK = Path('/kaggle/temp/openvoice-full-source')
OUT = Path('/kaggle/working')
ROOT = WORK/'source'
GGML_LIB = WORK/'ggml-runtime'
BUILD = WORK/'build'


def main():
    if not re.fullmatch('[0-9a-f]{40}', SOURCE) or not re.fullmatch('[0-9a-f]{64}', RECIPIENT):
        raise ValueError('Immutable source and operator public key required')
    WORK.mkdir(parents=True, exist_ok=True);OUT.mkdir(parents=True, exist_ok=True)
    path = WORK/'resource-helper.py'
    with urllib.request.urlopen('https://raw.githubusercontent.com/CrispStrobe/CrispASR/'+HELPER_SOURCE+
                                '/tools/kaggle/openvoice-bake/openvoice_bake.py', timeout=60) as response:
        code = response.read()
    import hashlib
    if hashlib.sha256(code).hexdigest() != HELPER_SHA:
        raise ValueError('Resource helper checksum mismatch')
    path.write_bytes(code)
    spec = importlib.util.spec_from_file_location('resources',path)
    helper = importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    helper.WORK=WORK;helper.OUT=OUT
    receipt=dict(schema=1,source=SOURCE,ggml=GGML,sdk_sha256=SDK_SHA,complete=False,
                 scope='Full current CrispASR library/CLI C++; prebuilt matching ggml CUDA; no CUDA kernel rebuild')
    helper.RECEIPT=receipt
    helper.save=lambda:(OUT/'build-provenance.json').write_text(json.dumps(receipt,indent=2))
    receipt['entry_sha256']=helper.sha(Path(__file__))
    receipt['hardware']=subprocess.check_output(['nvidia-smi',
        '--query-gpu=name,compute_cap,memory.total,driver_version','--format=csv,noheader'],
        text=True,env=helper.host_environment()).strip()
    os.environ.update(OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',GGML_SCHED_DEBUG='1')
    helper.guard_resources('full-source-bootstrap')
    if os.environ.get('CRISPASR_FULL_READY')!='1':
        for cmd in [['git','init',ROOT],['git','-C',ROOT,'remote','add','origin','https://github.com/CrispStrobe/CrispASR.git'],
                    ['git','-C',ROOT,'fetch','--depth=1','origin',SOURCE],['git','-C',ROOT,'checkout','FETCH_HEAD'],
                    ['git','-C',ROOT,'submodule','update','--init','--recursive','--depth=1']]:
            subprocess.run(list(map(str,cmd)),check=True)
        actual=subprocess.check_output(['git','-C',ROOT/'ggml','rev-parse','HEAD'],text=True).strip()
        if actual!=GGML:raise ValueError('Current headers do not match the audited provider')
        sdk=WORK/'sdk.tar.gz'
        helper.download('https://github.com/CrispStrobe/CrispASR/releases/download/v0.8.47/'+SDK_NAME+'.tar.gz',sdk,SDK_SHA)
        with tarfile.open(sdk) as archive:archive.extractall(WORK,filter='data')
        GGML_LIB.mkdir(exist_ok=True)
        for lib in (WORK/SDK_NAME/'lib').glob('libggml*.so*'):
            shutil.copy2(lib,GGML_LIB/lib.name,follow_symlinks=False)
        for notice in ['LICENSE','THIRD_PARTY_NOTICES.txt']:
            shutil.copy2(WORK/SDK_NAME/notice,OUT/('SDK-'+notice))
        providers=sorted({str(p) for base in [Path('/usr/local/lib'),Path('/opt/conda/lib')]
                          for p in base.glob('python*/**/nvidia/*/lib') if p.is_dir()})
        providers += [str(p) for p in [Path('/usr/local/cuda/lib64'),Path('/usr/local/nvidia/lib64')] if p.is_dir()]
        host=os.environ.get('LD_LIBRARY_PATH','')
        env=dict(os.environ,CRISPASR_FULL_READY='1',CRISPASR_OV2_HOST_LD_LIBRARY_PATH=host,
                 LD_LIBRARY_PATH=os.pathsep.join([str(GGML_LIB),*providers,host]))
        os.execve(sys.executable,[sys.executable,str(Path(__file__).resolve())],env)
    if subprocess.check_output(['git','-C',ROOT,'rev-parse','HEAD'],text=True).strip()!=SOURCE:
        raise ValueError('Application source changed after bootstrap')
    receipt['ggml_provider_files']={p.name:helper.sha(p) for p in GGML_LIB.iterdir() if p.is_file()}
    helper.save()
    # These dependencies do not install or replace the host's Torch package.
    helper.run([sys.executable,'-m','pip','install','-q','faster-whisper','soundfile','scipy','cryptography','gguf==0.17.1'],
               'quality-dependencies')
    helper.run([sys.executable,'-m','pip','install','-q','--no-deps','resemblyzer','webrtcvad-wheels'],
               'speaker-dependencies')
    hook=WORK/'ggml-import.cmake'
    hook.write_text('''find_package(CUDAToolkit REQUIRED)
if(NOT CUDAToolkit_VERSION MATCHES "^12[.]8[.]")
  message(FATAL_ERROR "Audited provider requires the existing CUDA 12.8 toolkit")
endif()
set(GGML_SCHED_MAX_COPIES 4 CACHE STRING "Match audited provider" FORCE)
set(GGML_CPU ON CACHE BOOL "Match audited provider" FORCE)
set(GGML_AVX2 ON CACHE BOOL "Match audited provider" FORCE)
set(GGML_FMA ON CACHE BOOL "Match audited provider" FORCE)
set(GGML_F16C ON CACHE BOOL "Match audited provider" FORCE)
foreach(name ggml-base ggml-cpu ggml-cuda ggml)
 if(NOT TARGET ${name})
  add_library(${name} SHARED IMPORTED GLOBAL)
  set_target_properties(${name} PROPERTIES
   IMPORTED_LOCATION "'''+str(GGML_LIB)+'''/lib${name}.so"
   INTERFACE_INCLUDE_DIRECTORIES "'''+str(ROOT/'ggml/include')+'''"
   INTERFACE_COMPILE_DEFINITIONS "GGML_SHARED;GGML_BACKEND_SHARED")
 endif()
endforeach()
set_property(TARGET ggml APPEND PROPERTY INTERFACE_COMPILE_DEFINITIONS GGML_USE_CPU GGML_USE_CUDA)
set_property(TARGET ggml PROPERTY INTERFACE_LINK_LIBRARIES "ggml-base;ggml-cpu;ggml-cuda")
set_property(TARGET ggml-cpu PROPERTY INTERFACE_LINK_LIBRARIES ggml-base)
set_property(TARGET ggml-cuda PROPERTY INTERFACE_LINK_LIBRARIES "ggml-base;CUDA::cudart;CUDA::cublas")
''')
    helper.run(['cmake','-S',ROOT,'-B',BUILD,'-DCMAKE_BUILD_TYPE=Release','-DBUILD_SHARED_LIBS=ON',
        '-DCMAKE_PROJECT_crispasr_INCLUDE='+str(hook),'-DGGML_CUDA=ON','-DGGML_BACKEND_DL=OFF',
        '-DGGML_METAL=OFF','-DGGML_VULKAN=OFF','-DGGML_NATIVE=OFF','-DCRISPASR_BUILD_TESTS=OFF',
        '-DCRISPASR_BUILD_SERVER=OFF','-DGGML_BLAS=OFF',
        '-DCMAKE_CUDA_ARCHITECTURES=60-real;61-real;70-real;75-real;80-real;86-real;89-real;90-real;90-virtual;120-real;120-virtual'],
        'configure-current-cuda')
    helper.run(['cmake','--build',BUILD,'--target','crispasr-cli','crispasr-lib','crispasr-diff','-j2'],
               'compile-current-application',timeout=7200)
    libraries=list(BUILD.rglob('libcrispasr.so'))
    if len(libraries)!=1:raise RuntimeError('Missing rebuilt current library')
    native_env=dict(os.environ,LD_LIBRARY_PATH=str(libraries[0].parent)+os.pathsep+os.environ['LD_LIBRARY_PATH'])
    for file in [libraries[0],BUILD/'bin/crispasr']:
        label='dependencies-'+file.name
        helper.run(['ldd','-r',file],label,env=native_env)
        log=(OUT/(label+'.log')).read_text()
        if 'not found' in log or 'undefined symbol:' in log:raise RuntimeError('Unresolved current runtime dependency')
        if str(WORK/SDK_NAME) in log:raise RuntimeError('Old SDK application DSO entered current application')
        loaded={}
        for name,target in re.findall(r'^\s*(libggml(?:-base|-cpu|-cuda)?\.so\S*) => (\S+)',log,re.M):
            actual=Path(target).resolve()
            if actual.parent!=GGML_LIB.resolve():raise RuntimeError('System ggml displaced the audited provider')
            loaded[name]=dict(file=actual.name,sha256=helper.sha(actual))
        if len(loaded)!=4:raise RuntimeError('Expected all four audited ggml providers')
        receipt.setdefault('loaded_ggml',{})[file.name]=loaded
        helper.save()
    helper.run(['git','-C',ROOT,'diff','--exit-code','--','src','include','examples/cli'],
               'unmodified-application-sources')
    # Preserve this exact application before crash-prone inference. This is a
    # private validation checkpoint, not an audited release package.
    helper.guard_resources('checkpoint-current-application')
    candidate=WORK/'candidate';candidate.mkdir(exist_ok=True)
    for file in [BUILD/'bin/crispasr',BUILD/'bin/crispasr-diff']:
        shutil.copy2(file,candidate/file.name)
    for tree in [BUILD,GGML_LIB]:
        for file in tree.rglob('*.so*'):
            if not file.is_file():continue
            actual=file.resolve()
            dest=candidate/actual.name
            if dest.exists():
                if helper.sha(dest)!=helper.sha(actual):raise RuntimeError('Candidate DSO basename collision')
            else:shutil.copy2(actual,dest)
            if file.name!=actual.name:
                alias=candidate/file.name
                if alias.exists() or alias.is_symlink():
                    if helper.sha(alias)!=helper.sha(actual):raise RuntimeError('Candidate alias collision')
                else:alias.symlink_to(actual.name)
    for name in ['LICENSE','THIRD_PARTY_NOTICES.txt']:
        shutil.copy2(ROOT/name,candidate/name)
        shutil.copy2(WORK/SDK_NAME/name,candidate/('ggml-provider-'+name))
    (candidate/'provenance.json').write_text(json.dumps(receipt,indent=2))
    with tarfile.open(OUT/'current-application-checkpoint.tar.gz','w:gz',compresslevel=1) as archive:
        archive.add(candidate,arcname='candidate')
    receipt['application_checkpoint_sha256']=helper.sha(OUT/'current-application-checkpoint.tar.gz')
    helper.save()
    mounted=sorted(p for p in Path('/kaggle/input').rglob('validation.json')
                   if 'crispasr-openvoice-conditioning' in p.parts)
    if not mounted:raise RuntimeError('Accepted real conditioning data missing')
    data=WORK/'inputs';data.mkdir(exist_ok=True)
    metadata=json.loads(mounted[0].read_text())
    shutil.copy2(mounted[0],data/'validation.json')
    for name in metadata['artifacts']:
        if name in ['openvoice2-tcc-f32sens.gguf','reference-0.gguf','reference-1.gguf','reference.wav',
                    'original-0.wav','original-1.wav','reference-ATTRIBUTION.txt','reference-OpenVoice-LICENSE.txt']:
            shutil.copy2(mounted[0].parent/name,data/name)
    native_env.update(HEAVY_OUT=str(OUT),HEAVY_SCRATCH=str(WORK/'validation'))
    helper.run([sys.executable,ROOT/'tools/ci-heavy/openvoice_interfaces.py','--device','cuda',
                '--build-dir',BUILD,'--inputs',data,'--bert-profile','f16','--audio-public-key',RECIPIENT],
               'full-source-cuda-interfaces',env=native_env,timeout=7200)
    accepted=json.loads((OUT/'receipt.json').read_text())
    if not accepted.get('acceptance') or not accepted.get('cli_cuda_observed'):
        raise RuntimeError('Full-source real CUDA acceptance incomplete')
    receipt.update(complete=True,acceptance=True,built_sha256=accepted['built_sha256'])
    helper.save()


if __name__=='__main__':
    main()
