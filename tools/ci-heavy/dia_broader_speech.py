#!/usr/bin/env python3
"""Predeclared Dia CPU speech matrix; no reference injection or default changes."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import wave

import numpy as np
import dia_cache_sampler as support

ROOT = Path(__file__).resolve().parents[2]
RUNTIME = '5e04955e580b798b3cd45d870d407a3dd7600c4e'
CASES = {f'{name}-{seed}': dict(text='[S1] '+text, seed=seed)
         for name,text in [('hello','Hello there, how are you doing today?'),
                           ('fox','The quick brown fox jumps over the lazy dog.')]
         for seed in [42,123]}
ARM = 'serial-f32-portable'


def metrics(actual, expected):
    a,b=np.asarray(actual),np.asarray(expected)
    if a.shape!=b.shape or not a.size or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Missing, misaligned, empty or nonfinite stage arrays')
    width=b.shape[-1] if b.ndim else 1
    x,y=a.astype('f8').reshape(-1,width),b.astype('f8').reshape(-1,width)
    xn,yn=np.linalg.norm(x,axis=1),np.linalg.norm(y,axis=1)
    cosine=np.ones(len(x));mask=(xn>0)&(yn>0)
    cosine[mask]=np.einsum('ij,ij->i',x[mask],y[mask])/(xn[mask]*yn[mask])
    cosine[(xn==0)!=(yn==0)]=0
    ratio=np.ones(len(x));mask=yn>0
    ratio[mask]=xn[mask]/yn[mask];ratio[(~mask)&(xn>0)]=np.inf
    row=dict(values=a.size,cos_min=float(cosine.min()),mine_norm=float(np.linalg.norm(x)),
        reference_norm=float(np.linalg.norm(y)),max_row_norm_error=float(abs(ratio-1).max()),
        relative_l2=float(np.linalg.norm(x-y)/max(np.linalg.norm(y),1e-30)),max_abs=float(abs(x-y).max()))
    row['pass']=bool(row['cos_min']>=.9995 and row['max_row_norm_error']<.02)
    return {k:str(v) if isinstance(v,float) and not np.isfinite(v) else v for k,v in row.items()}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--child',choices=['native','decode-native','auto','asr','dac'])
    parser.add_argument('--case',choices=list(CASES))
    parser.add_argument('--audio')
    for name in ['runtime','converter','blueprint','model','lib']:
        parser.add_argument('--'+name,type=Path)
    args=parser.parse_args()
    out,scratch=Path(os.environ['HEAVY_OUT']),Path(os.environ['HEAVY_SCRATCH'])
    out.mkdir(parents=True,exist_ok=True);scratch.mkdir(parents=True,exist_ok=True)
    os.environ['TMPDIR']=str(scratch);sys.dont_write_bytecode=True
    if args.child:
        case=CASES[args.case];args.seed=case['seed'];args.text=case['text'];args.arm=ARM
        caseout=out/'cases'/args.case
        if args.child=='asr' and args.audio=='canonical-pcm.npy': args.arm='original'
        if args.child=='native':
            os.environ.update(CRISPASR_DIA_DUMP_SELF_ATTN='1',CRISPASR_DIA_DUMP_DECODER_BOUNDARIES='1')
        if args.child=='decode-native':
            folder=caseout/'arms'/ARM/'dac-stages';folder.mkdir()
            os.environ['CRISPASR_DIA_DAC_TRACE_DIR']=str(folder)
        if args.child=='dac':
            # Observe the actual original codec on the native arm's OWN codes.
            import torch
            import dac
            import inspect
            import importlib.metadata
            assert torch.__version__=='2.6.0+cpu';torch.set_num_threads(4)
            sys.path.insert(0,str(ROOT/'tools'))
            from reference_backends import dia_tts_reference as reference
            _,hashes=reference.verify_source(args.blueprint)
            checkpoint=Path(dac.utils.download())
            assert support.sha(checkpoint)=='a88eed82a7024ccc1facdb1e605c4c2f99281c8118c22c9895ffa846d8fb61aa'
            codec=dac.DAC.load(str(checkpoint)).cpu().eval()
            dest=caseout/'arms'/ARM;folder=dest/'original-dac-stages';folder.mkdir()
            grid=np.fromfile(dest/'codes.u32',dtype='<u4').reshape(-1,9)
            assert grid.size and (grid<1024).all()
            captured=reference.capture_codec(codec,torch.from_numpy(grid.astype('i8').T.copy()).unsqueeze(0))
            for name,value in captured.items():np.save(folder/(name+'.npy'),value,allow_pickle=False)
            np.save(dest/'original-dac-pcm.npy',captured['decoded_audio'],allow_pickle=False)
            import dac.model.dac as dm
            import dac.nn.quantize as dq
            import dac.nn.layers as dl
            for module,name in [(dm,'dac-model.py'),(dq,'dac-quantize.py'),(dl,'dac-layers.py')]:
                shutil.copyfile(inspect.getsourcefile(module),dest/name)
            (dest/'dac-provenance.json').write_text(json.dumps(dict(source_hashes=hashes,
                checkpoint_sha256=support.sha(checkpoint),torch_version=torch.__version__,
                package=importlib.metadata.version('descript-audio-codec')),indent=2)+'\n')
        else:support.child(args,caseout)
        return

    source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    assert source==os.environ['GITHUB_SHA'] and not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT)
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT/'ggml',text=True).strip()==support.GGML
    report=dict(source=source,runtime_source=RUNTIME,ggml=support.GGML,cases=CASES,runs=[],results={},
        defaults_changed=False,speech_acceptance=False,gpu_acceptance=False,q4_acceptance=False)
    def checkpoint():
        (out/'receipt.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    def run(label,command,env=None):
        with (out/(label+'.log')).open('xb') as log:
            result=subprocess.run(list(map(str,command)),cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=10800)
        report['runs'].append(dict(label=label,exit=result.returncode,sha256=support.sha(out/(label+'.log'))));checkpoint()
        print(label,result.returncode,flush=True)
        assert result.returncode==0,label
    checkpoint()
    run('fetch-runtime',['git','fetch','--depth=1','origin',RUNTIME])
    changed=subprocess.check_output(['git','diff','--name-only',RUNTIME,'HEAD','--','src','include','ggml','examples','python','bindings'],cwd=ROOT,text=True)
    assert not changed,changed
    method=out/'method';method.mkdir()
    for name in ['tools/ci-heavy/dia_broader_speech.py','tools/ci-heavy/dia_cache_sampler.py',
        'tools/ci-heavy/dia-broader-speech/selftest.py','src/dia_tts.cpp','src/dia_sampling.h',
        'src/dia_exponential_rng.h','src/dia_serial_head.h','src/dia_dump_step.h',
        'tools/dump_reference.py','tools/reference_backends/dia_tts_reference.py',
        'tools/dia_attention_capture.py','tools/dia_decoder_capture.py','python/crispasr/_binding.py']:
        dest=method/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,dest)
    run('apt-install',['sudo','bash',ROOT/'tools/ci-apt.sh','install','-y','ninja-build','ccache'])
    build=scratch/'build'
    run('configure',['cmake','-G','Ninja','-S',ROOT,'-B',build,'-DCMAKE_BUILD_TYPE=Release',
        '-DBUILD_SHARED_LIBS=ON','-DCMAKE_C_COMPILER_LAUNCHER=ccache','-DCMAKE_CXX_COMPILER_LAUNCHER=ccache',
        '-DGGML_NATIVE=OFF','-DGGML_BLAS=OFF','-DGGML_CUDA=OFF','-DGGML_VULKAN=OFF',
        '-DCRISPASR_BUILD_TESTS=OFF','-DCRISPASR_BUILD_SERVER=OFF','-DCRISPASR_OPUS=OFF','-DCRISPASR_AMR=OFF'])
    run('build',['cmake','--build',build,'--target','crispasr-lib','crispasr-cli','-j4'])
    assert 'dia_tts.cpp.o' in (out/'build.log').read_text() and 'Linking' in (out/'build.log').read_text()
    lib=next(build.rglob('libcrispasr.so'));cli=build/'bin/crispasr';assert cli.is_file()
    libs=out/'libs';libs.mkdir()
    for path in build.rglob('*.so*'):
        if path.is_file() and not path.is_symlink():shutil.copyfile(path,libs/path.name)
    shutil.copyfile(build/'CMakeCache.txt',out/'CMakeCache.txt')
    report['library_sha256']=support.sha(lib);checkpoint()
    run('fetch-converter',['git','fetch','origin',support.CONVERTER])
    converter=scratch/'converter'
    run('checkout-converter',['git','worktree','add','--detach',converter,support.CONVERTER])
    for name in ['tools/ci-heavy/dia_neural_precision_control.py','tools/dia_model_precision.py',
        'tools/dia_dac_export.py','models/convert-dia-to-gguf.py']:
        dest=method/'converter'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(converter/name,dest)
    from huggingface_hub import hf_hub_download,snapshot_download
    weights=Path(snapshot_download('nari-labs/Dia-1.6B',revision='257bc72f9b78182ccc6fa07675a9ae4c1a44e2cd',allow_patterns=['config.json','model.safetensors']))
    assert support.sha(weights/'model.safetensors')=='caba289b60f6d7d1e58fc744f4dc25aae88995fcca46be3d05e220b971486a26'
    template=hf_hub_download('cstr/dia-1.6b-GGUF','dia-1.6b-f16.gguf',revision='3233fbcb32be47761d2e736857b6d1a075b9ba7e')
    models=scratch/'models';models.mkdir();model=models/'dia-1.6b-f32.gguf'
    run('neural-export',[sys.executable,converter/'tools/ci-heavy/dia_neural_precision_control.py','--export','f32',
        '--weights',weights/'model.safetensors','--template',template,'--output',model])
    assert support.sha(model)==support.MODEL_SHA
    ledger=json.loads((out/'neural-f32-physical-audit.json').read_text())
    assert len(ledger['tensors'])==381 and all(t['selected'] and t['exported_type']=='F32' for t in ledger['tensors'])
    ct=hf_hub_download('cstr/dia-1.6b-GGUF','dac-44khz.gguf',revision='3233fbcb32be47761d2e736857b6d1a075b9ba7e')
    codec=models/'dac-44khz.gguf'
    run('codec-export',[sys.executable,ROOT/'models/convert-dia-dac-to-gguf.py','--template',ct,'--output',codec,
        '--precision','output-f32','--receipt',out/'physical-codec-audit.json'])
    assert support.sha(codec)==support.CODEC_SHA
    asr=Path(hf_hub_download('cstr/nemotron-3.5-asr-streaming-GGUF','nemotron-3.5-asr-streaming-0.6b-q4_k.gguf',revision='bbd95a9ca5fa0dfca3312a122dfc45a2b578b9c2'))
    assert support.sha(asr)==support.ASR_SHA
    report.update(model_sha256=support.sha(model),codec_sha256=support.sha(codec),asr_sha256=support.sha(asr));checkpoint()
    blueprint=scratch/'original-dia'
    run('clone-original',['git','clone','--filter=blob:none','https://github.com/nari-labs/dia',blueprint])
    run('pin-original',['git','-C',blueprint,'checkout','--detach',support.ORIGINAL])
    shutil.copytree(blueprint/'dia',method/'original-dia',ignore=shutil.ignore_patterns('__pycache__'))
    for name,path in [('CrispASR-LICENSE',ROOT/'LICENSE'),('GGML-LICENSE',ROOT/'ggml/LICENSE'),('Dia-LICENSE',blueprint/'LICENSE')]:
        shutil.copyfile(path,out/name)
    clean={k:v for k,v in os.environ.items() if not k.startswith(('DIA_','CRISPASR_DIA_'))}
    run('worker-selftest',[sys.executable,ROOT/'tools/ci-heavy/dia-broader-speech/selftest.py'],clean)
    from gguf import GGUFReader
    for key,case in CASES.items():
        folder=out/'cases'/key;folder.mkdir(parents=True)
        original=folder/'arms/original';original.mkdir(parents=True)
        original_env=clean|dict(DIA_SOURCE=str(blueprint),DIA_DEVICE='cpu',DIA_TEXT=case['text'],
            DIA_SEED=str(case['seed']),DIA_MAX_TOKENS='3072',DIA_TEMPERATURE='1.2',DIA_TOP_P='.95',DIA_TOP_K='45',DIA_CFG_SCALE='3',
            DIA_CAPTURE_ATTENTION_LAYERS='0,17',DIA_CAPTURE_BOUNDARY_LAYERS='0,17',
            OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',MKL_CBWR='COMPATIBLE',ATEN_CPU_CAPABILITY='default')
        archive=folder/'original-ref.gguf'
        run(key+'-original',[sys.executable,ROOT/'tools/dump_reference.py','--backend','dia','--model-dir',weights,'--output',archive],original_env)
        reader=GGUFReader(str(archive));reference={t.name:np.array(t.data).reshape(tuple(reversed(t.shape))) for t in reader.tensors}
        meta=lambda n:reader.get_field('crispasr.ref.'+n).contents()
        assert meta('dia_text')==case['text'] and meta('dia_source_pin')==support.ORIGINAL and meta('dia_torch_version')=='2.6.0+cpu'
        settings=json.loads(meta('dia_generation_settings'))
        assert settings['seed']==case['seed'] and settings['max_tokens']==3072 and settings['temperature']==1.2
        assert settings['top_p']==.95 and settings['top_k']==45 and settings['cfg_scale']==3.
        assert meta('dia_dac_sha256')=='a88eed82a7024ccc1facdb1e605c4c2f99281c8118c22c9895ffa846d8fb61aa'
        assert meta('dia_attention_capture_layers')=='0,17' and meta('dia_boundary_capture_layers')=='0,17'
        assert meta('dia_weights_sha256')=='caba289b60f6d7d1e58fc744f4dc25aae88995fcca46be3d05e220b971486a26'
        np.save(original/'canonical-pcm.npy',reference['decoded_audio'].reshape(-1),allow_pickle=False)
        reader.data._mmap.close()
        def invoke(kind,audio=None):
            cmd=[sys.executable,__file__,'--child',kind,'--case',key,'--runtime',ROOT,'--converter',converter,
                '--blueprint',blueprint,'--model',asr if kind=='asr' else model,'--lib',lib]
            if audio:cmd+=['--audio',audio]
            run(key+'-'+kind+('-'+audio if audio else ''),cmd,clean)
        invoke('asr','canonical-pcm.npy')
        invoke('native')
        native=folder/'arms'/ARM
        inputs=np.fromfile(native/'inputs.i32',dtype='<i4').reshape(-1,9)
        logits=np.memmap(native/'logits.f32',mode='r',dtype='<f4').reshape(-1,2,9,1028)
        assert len(inputs)==len(logits) and np.isfinite(logits).all()
        grid=support.grid_from_inputs(inputs);grid.tofile(native/'codes.u32')
        assert len(np.load(native/'native-pcm.npy'))==len(grid)*512
        invoke('decode-native');support.exact(np.load(native/'native-pcm.npy'),np.load(native/'decode-native-pcm.npy'))
        invoke('dac')
        for audio in ['native-pcm.npy','original-dac-pcm.npy']:invoke('asr',audio)
        cli_dir=native/'cli';cli_dir.mkdir();wav=native/'cli.wav'
        cli_env=clean|support.arm_environment(ARM)|dict(CRISPASR_DIA_TTS_GPU='0',
            CRISPASR_DIA_DUMP_INPUT_TOKENS=str(cli_dir/'inputs.i32'),CRISPASR_DIA_DUMP_STEPLOGITS=str(cli_dir/'logits.f32'))
        run(key+'-cli',[cli,'--no-gpu','-m',model,'--codec-model',codec,'--tts-output',wav,'-l','en','--tts',case['text'],
            '-t','4','--seed',str(case['seed']),'--temperature','1.2','--no-spoken-disclaimer','--accept-marking-responsibility'],cli_env)
        for name in ['inputs.i32','logits.f32']:assert support.sha(cli_dir/name)==support.sha(native/name)
        with wave.open(str(wav),'rb') as stream:
            assert stream.getnchannels()==1 and stream.getsampwidth()==2 and stream.getframerate()==44100
            pcm=np.frombuffer(stream.readframes(stream.getnframes()),dtype='<i2').astype('f4')/32768
        np.save(native/'cli-pcm.npy',pcm,allow_pickle=False);invoke('asr','cli-pcm.npy')
        stages={}
        for name,value in reference.items():
            if name in ['decoder_codes_rev','decoded_audio','decoder_input_tokens','decoder_step_logits'] or name.startswith('dac_'):continue
            filename={'decoder_ca0':'cpp_ca0_out.f32','decoder_hidden':'cpp_final_hidden.f32'}.get(name,'cpp_'+name+'.f32')
            actual=np.fromfile(native/'stages'/filename,dtype='<f4').reshape(value.shape)
            stages[name]=metrics(actual,value)
            if name=='encoder_tokens':assert np.array_equal(actual,value)
        for batch,name in enumerate(['uncond','cond']):
            actual=np.fromfile(native/'stages'/('cpp_step0_'+name+'.f32'),dtype='<f4').reshape(9,1028)
            stages[name+'_logits']=metrics(actual,reference['decoder_step_logits'][0,batch])
        for path in sorted((native/'original-dac-stages').glob('*.npy')):
            value=np.load(path,allow_pickle=False)
            if path.stem=='decoded_audio':actual=np.load(native/'native-pcm.npy')
            else:actual=np.fromfile(native/'dac-stages'/(path.stem+'.f32'),dtype='<f4').reshape(value.shape)
            stages[path.stem+'_own_codes']=metrics(actual,value)
        assert len(stages)==88,len(stages)
        expected_inputs=reference['decoder_input_tokens'].astype('<i4');common=min(len(inputs),len(expected_inputs))
        different=np.flatnonzero(np.any(inputs[:common]!=expected_inputs[:common],axis=1))
        row=dict(steps=len(inputs),frames=len(grid),original_input_prefix=int(different[0]) if len(different) else common,
            original_inputs_exact=bool(np.array_equal(inputs,expected_inputs)),original_codes_exact=bool(np.array_equal(grid,reference['decoder_codes_rev'])),
            cli_session_inputs_logits_exact=True,native_decode_replay_exact=True,stages=stages,
            original_asr=json.loads((original/'canonical-pcm.npy.asr.json').read_text()),
            native_asr={name:json.loads((native/(name+'.asr.json')).read_text()) for name in ['native-pcm.npy','original-dac-pcm.npy','cli-pcm.npy']})
        row['pass']=row['original_asr']['wer']==0 and all(v['wer']==0 for v in row['native_asr'].values()) and all(v['pass'] for v in stages.values())
        report['results'][key]=row;checkpoint()
        print('DIA_BROADER_CASE_COMPLETE',key,row['pass'],flush=True)
    report['complete_diagnostic']=True
    report['all_four_cases_pass']=len(report['results'])==4 and all(v['pass'] for v in report['results'].values())
    report['payloads']=support.payload_inventory(out)
    checkpoint()
    assert report['all_four_cases_pass'],'Broader speech failed; complete evidence retained'
    print('DIA_BROADER_SPEECH_PASS',list(CASES),flush=True)


if __name__=='__main__':main()
