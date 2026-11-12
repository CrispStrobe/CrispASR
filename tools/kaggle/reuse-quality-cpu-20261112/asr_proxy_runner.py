"""Optional sequential CPU-ASR transcript-agreement research on completed cases.

Never establishes human-reference WER, intelligibility or product acceptance.
The caller must supply and hash-pin an ASR binary and its model.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import time

import numpy as np
import soundfile as sf


def sibling(name):
    spec = importlib.util.spec_from_file_location(name,Path(__file__).with_name(name+'.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


auditor = sibling('audit_saved_waveforms')
wer = sibling('word_error_rate')
SPEAKERS = (121,237,61,260)
CONDITIONS = ('clean','white-0db','white-10db','pink-5db')


class ProxyFailure(RuntimeError):
    pass


def require(condition,message):
    if not condition:
        raise ProxyFailure(message)


def atomic_bytes(path,data):
    auditor.atomic_write(path,lambda temporary:temporary.write_bytes(data))


def inspect_dependencies(binary):
    proc = subprocess.run(['ldd',str(binary)],capture_output=True,text=True,timeout=30)
    require(proc.returncode == 0,'ldd failed; cannot record dynamically resolved runtime')
    result = {}
    for line in proc.stdout.splitlines():
        fields = line.split()
        if '=>' in fields:
            require('not found' not in line,'unresolved runtime dependency')
            path = fields[fields.index('=>')+1]
        elif fields and fields[0].startswith('/'):
            path = fields[0]
        else:
            continue
        if path.startswith('/'):
            result[Path(path).name] = auditor.sha256_file(Path(path))
    return result


def audio_info(path,digest):
    require(auditor.sha256_file(path) == digest,path.name+': WAV file hash mismatch')
    audio,rate = sf.read(path,dtype='float32')
    require(rate == 16000 and audio.shape == (80000,) and np.isfinite(audio).all(),
            path.name+': expected finite mono 5s waveform at 16kHz')
    return {'file':path.name,'file_sha256':digest,'audio_f32_sha256':hashlib.sha256(audio.tobytes()).hexdigest()}


def load_plan(artifacts):
    report = json.loads((artifacts/'quality-report.json').read_text())
    require(not (artifacts/'failure.json').exists(),'upstream failure.json exists')
    require(report.get('status') == 'complete' and report.get('completed_cases') == 16,
            'upstream corpus evaluation is incomplete')
    cases = report['cases']
    ids = {f'speaker-{s}-{c}' for s in SPEAKERS for c in CONDITIONS}
    require(len(cases) == 16 and {x['case_id'] for x in cases} == ids,'expected16 unique predetermined cases')
    by_id = {x['case_id']:x for x in cases}
    plan,clean = [],{}
    for speaker in SPEAKERS:
        for condition in CONDITIONS:
            case_id = f'speaker-{speaker}-{condition}'
            item = by_id[case_id]
            require(item['speaker'] == speaker and item['condition'] == condition,'case identity mismatch')
            path = artifacts/(case_id+'.json')
            receipt = json.loads(path.read_text())
            require({k:v for k,v in receipt.items() if k != 'stage_fingerprints'} == item,'aggregate/case disagreement')
            clean_path = artifacts/f'speaker-{speaker}-clean.wav'
            clean_info = audio_info(clean_path,item['clean_sha256'])
            if speaker in clean:
                require(clean[speaker] == clean_info,'inconsistent clean reference hashes')
            clean[speaker] = clean_info
            entry = {'case_id':case_id,'speaker':speaker,'condition':condition,'clean':clean_info}
            for label in ('input','output'):
                entry[label] = audio_info(artifacts/f'{case_id}-{label}.wav',item[label+'_sha256'])
            if condition == 'clean':
                require(entry['input']['audio_f32_sha256'] == clean_info['audio_f32_sha256'],
                        case_id+': clean input differs from speaker reference audio')
            plan.append(entry)
    return plan,clean,auditor.sha256_file(artifacts/'quality-report.json')


def summarize(cases):
    groups = {}
    for condition in CONDITIONS:
        rows = [x for x in cases if x['condition'] == condition]
        group = {'completed_cases':len(rows)}
        for label in ('input','output'):
            scores = [x[label+'_agreement'] for x in rows]
            words = sum(x['reference_words'] for x in scores)
            errors = sum(x['edits'] for x in scores)
            group[label] = {'edits':errors,'reference_words':words,'micro_wer':errors/words if words else None}
        groups[condition] = group
    return groups


def run_proxy(artifacts,binary,model,output_dir,*,binary_sha256,model_sha256,
              timeout=180,threads=2,executor=subprocess.run,dependency_inspector=inspect_dependencies):
    artifacts,binary,model,output_dir = map(lambda p:Path(p).resolve(),(artifacts,binary,model,output_dir))
    require(timeout > 0 and 1 <= threads <= 4,'invalid timeout or CPU thread count')
    require(not (output_dir/'proxy-report.json').exists(),'output already contains a proxy report; choose a fresh directory')
    output_dir.mkdir(parents=True,exist_ok=True)
    report = {'status':'preflight','scope':'Optional research ASR transcript agreement against the same speaker clean-ASR transcript; not human-reference WER, intelligibility or product acceptance.',
              'invocations':[],'cases':[],'clean_references':{},'timeout_seconds':timeout,'threads':threads,
              'binary_sha256':binary_sha256,'model_sha256':model_sha256}
    aliases = {str(artifacts):'${QUALITY_ARTIFACTS}',str(output_dir):'${ASR_PROXY_OUTPUT}',
               str(binary):'${ASR_BINARY}',str(model):'${ASR_MODEL}'}
    def portable(text):
        for physical,alias in sorted(aliases.items(),key=lambda x:len(x[0]),reverse=True):
            text = text.replace(physical,alias)
        return text
    def checkpoint():
        auditor.write_json(output_dir/'proxy-report.json',report)
    checkpoint()
    cache = {}
    def transcript(record):
        path = output_dir/record['transcript_file']
        require(auditor.sha256_file(path) == record['transcript_sha256'],'completed transcript hash changed')
        return path.read_text()
    def transcribe(info,label,*,fresh=False):
        require(audio_info(artifacts/info['file'],info['file_sha256']) == info,
                info['file']+': source audio changed after preflight')
        key = info['audio_f32_sha256']
        if not fresh and key in cache:
            record = cache[key]
            transcript(record)
            return {'invocation_id':record['invocation_id'],'deduplicated':True,**info},record
        number = len(report['invocations'])
        invocation_id = f'{number:03d}-{label}'
        pending = output_dir/'.pending'
        pending.mkdir(exist_ok=True)
        started = time.monotonic()
        with tempfile.TemporaryDirectory(dir=pending,prefix=invocation_id+'-') as temporary:
            prefix = Path(temporary)/'transcript'
            command = [str(binary),'-m',str(model),'-ng','-t',str(threads),'-l','en','-nt','-np','-otxt','-of',str(prefix),str(artifacts/info['file'])]
            record = {'invocation_id':invocation_id,'audio':info,'command':[portable(x) for x in command],
                      'timeout_seconds':timeout,'status':'running','exit_code':None}
            report['pending_invocation'] = record
            checkpoint()
            try:
                proc = executor(command,capture_output=True,timeout=timeout,check=False)
                stdout,stderr = proc.stdout or b'',proc.stderr or b''
                record.update(status='completed' if proc.returncode == 0 else 'nonzero-exit',exit_code=proc.returncode)
            except subprocess.TimeoutExpired as exc:
                stdout,stderr = exc.output or b'',exc.stderr or b''
                record['status'] = 'timeout'
            except OSError:
                stdout,stderr = b'',b''
                record['status'] = 'spawn-error'
            require(isinstance(stdout,bytes) and isinstance(stderr,bytes),'executor must return raw bytes')
            for stream,data in (('stdout',stdout),('stderr',stderr)):
                raw_file = Path('logs')/(invocation_id+'.'+stream+'.raw')
                clean_file = Path('logs')/(invocation_id+'.'+stream+'.txt')
                atomic_bytes(output_dir/raw_file,data)
                atomic_bytes(output_dir/clean_file,portable(data.decode('utf-8','replace')).encode())
                record[stream+'_raw_sha256'] = hashlib.sha256(data).hexdigest()
                record[stream+'_raw_file'] = str(raw_file)
                record[stream+'_sanitized_file'] = str(clean_file)
            record['elapsed_seconds'] = time.monotonic()-started
            text_path = prefix.with_suffix('.txt')
            if record['status'] == 'completed':
                if not text_path.is_file():
                    record['status'] = 'missing-transcript'
                else:
                    data = text_path.read_bytes()
                    try:
                        data.decode('utf-8')
                    except UnicodeDecodeError:
                        record['status'] = 'invalid-transcript-encoding'
                    destination = Path('transcripts')/(invocation_id+'.txt')
                    atomic_bytes(output_dir/destination,data)
                    record.update(transcript_file=str(destination),transcript_sha256=hashlib.sha256(data).hexdigest())
            auditor.write_json(output_dir/'invocations'/(invocation_id+'.json'),record)
            report['invocations'].append(record)
            report.pop('pending_invocation',None)
            checkpoint()
            require(record['status'] == 'completed',invocation_id+': '+record['status'])
        cache.setdefault(key,record)
        return {'invocation_id':invocation_id,'deduplicated':False,**info},record
    try:
        require(auditor.sha256_file(binary) == binary_sha256,'binary hash mismatch')
        require(auditor.sha256_file(model) == model_sha256,'ASR model hash mismatch')
        libraries = dependency_inspector(binary)
        report['dynamic_libraries_sha256'] = libraries
        plan,clean,input_report_sha = load_plan(artifacts)
        report.update(status='running',upstream_report_sha256=input_report_sha)
        checkpoint()
        references = {}
        for speaker in SPEAKERS:
            mapping,record = transcribe(clean[speaker],f'clean-reference-{speaker}')
            reference = transcript(record)
            require(bool(wer.normalize_words(reference)),f'speaker-{speaker}: empty clean-ASR reference')
            references[speaker] = record
            report['clean_references'][str(speaker)] = mapping
            checkpoint()
        for entry in plan:
            case_id,speaker = entry['case_id'],entry['speaker']
            reference_record = references[speaker]
            reference = transcript(reference_record)
            result = {'case_id':case_id,'speaker':speaker,'condition':entry['condition'],
                      'reference_invocation_id':reference_record['invocation_id']}
            for label in ('input','output'):
                mapping,record = transcribe(entry[label],case_id+'-'+label,
                                           fresh=entry['condition'] == 'clean' and label == 'input')
                hypothesis = transcript(record)
                if entry['condition'] == 'clean' and label == 'input':
                    require(record['transcript_sha256'] == reference_record['transcript_sha256'],
                            case_id+': nondeterministic ASR transcript for identical clean audio')
                    result['clean_input_audio_identity'] = True
                    result['clean_input_transcript_identity'] = True
                    result['clean_input_wav_bytes_equal'] = entry[label]['file_sha256'] == entry['clean']['file_sha256']
                result[label+'_mapping'] = mapping
                result[label+'_agreement'] = wer.compare_transcripts(reference,hypothesis)
            auditor.write_json(output_dir/'cases'/(case_id+'.json'),result)
            report['cases'].append(result)
            report['conditions'] = summarize(report['cases'])
            checkpoint()
        require(auditor.sha256_file(binary) == binary_sha256 and auditor.sha256_file(model) == model_sha256,
                'ASR binary/model changed during execution')
        require(dependency_inspector(binary) == libraries,'runtime dependencies changed during execution')
        report.update(status='complete',completed_cases=16,unique_invocations=len(report['invocations']))
        checkpoint()
        return report
    except Exception as exc:
        report.update(status='failed',error_type=type(exc).__name__,error=portable(str(exc)))
        checkpoint()
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('artifacts','binary','model','output-dir'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--binary-sha256',required=True)
    parser.add_argument('--model-sha256',required=True)
    parser.add_argument('--timeout',type=float,default=180)
    parser.add_argument('--threads',type=int,default=2)
    args = parser.parse_args(argv)
    report = run_proxy(args.artifacts,args.binary,args.model,args.output_dir,
                       binary_sha256=args.binary_sha256,model_sha256=args.model_sha256,
                       timeout=args.timeout,threads=args.threads)
    print(json.dumps({'status':report['status'],'completed_cases':16,'unique_invocations':report['unique_invocations']}))


if __name__ == '__main__':
    main()
