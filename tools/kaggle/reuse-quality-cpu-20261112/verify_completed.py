"""Verify completed offline CPU quality artifacts without model execution."""
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re

import numpy as np
import soundfile as sf

KERNEL_SLUG = 'crispasr-re-use-quality-cpu-20261112'
RUNNING_SCRIPT_SHA256 = 'e5bb069a930745389e180d393dcbfd00d8903c8ec49ec3011505a1936c0e13e9'


def sibling(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(name+'.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


kernel = sibling('quality_cpu')
auditor = sibling('audit_saved_waveforms')


def read_json(path):
    return json.loads(path.read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def close(actual, expected, name):
    if expected is None:
        require(actual is None, name+': expected null')
    else:
        require(isinstance(actual, (int,float)) and not isinstance(actual,bool)
                and math.isfinite(actual) and math.isclose(actual,expected,rel_tol=1e-6,abs_tol=1e-6),
                name+': numerical metric mismatch')


def read_wave(path, digest):
    require(auditor.sha256_file(path) == digest, path.name+': file SHA-256 mismatch')
    data, rate = sf.read(path,dtype='float32')
    require(rate == 16000 and data.shape == (80000,) and np.isfinite(data).all(),
            path.name+': expected finite 5s mono FLOAT waveform at 16kHz')
    require(sf.info(path).subtype == 'FLOAT',path.name+': expected FLOAT WAV')
    return data


def expected_shape(name):
    if name in ('raw_audio','waveform'):
        return [80000]
    if name.startswith('frontend_') or name in ('amplitude','output_phase'):
        return [1001,321]
    if name.startswith('encoder_layer_') and int(name.rsplit('_',1)[1]) < 5:
        return [64,1003,323]
    return [64,251,161]


def validate_fingerprints(stages, raw, wave):
    kernel.validate_stages(stages)
    for name, data in stages.items():
        require(data['shape'] == expected_shape(name),name+': stage shape mismatch')
        require(data['dtype'] == 'float32',name+': stage dtype mismatch')
        require(isinstance(data['sha256'],str) and re.fullmatch('[0-9a-f]{64}',data['sha256']) is not None,
                name+': invalid stage SHA-256')
        norm, peak = data['norm'], data['max_abs']
        require(isinstance(norm,(int,float)) and isinstance(peak,(int,float))
                and math.isfinite(norm) and math.isfinite(peak) and norm >= 0 and peak >= 0,
                name+': invalid stage magnitude')
        require(peak*(1-1e-6) <= norm <= peak*math.sqrt(math.prod(data['shape']))*(1+1e-6),
                name+': inconsistent stage magnitudes')
    for name, array in (('raw_audio',raw),('waveform',wave)):
        data = stages[name]
        require(data['sha256'] == hashlib.sha256(array.tobytes()).hexdigest(),name+': waveform tensor hash mismatch')
        close(data['norm'],float(np.linalg.norm(array.astype(np.float64))),name+' norm')
        close(data['max_abs'],float(np.max(abs(array))),name+' peak')


def independent_metrics(clean, estimate, scorer):
    result = auditor.metrics(clean,estimate,stoi_scorer=scorer)
    result['projection_gain'] = result.pop('gain_projection')
    result['peak_abs'] = float(np.max(abs(estimate.astype(np.float64))))
    result['fraction_abs_gt_one'] = float(np.mean(abs(estimate.astype(np.float64)) > 1))
    return result


def summarize(cases, field):
    groups = {}
    for case in cases:
        groups.setdefault(str(case[field]), []).append(case)
    summary = {}
    for group, rows in groups.items():
        values = {}
        for metric in ('snr_db','si_sdr_db','stoi','estoi','spectral_convergence'):
            pairs = [(row['independent_metrics']['input'].get(metric),row['independent_metrics']['output'].get(metric)) for row in rows]
            pairs = [(a,b) for a,b in pairs if a is not None and b is not None]
            values[metric] = {'valid_pairs':len(pairs),
                              'mean_input':float(np.mean([a for a,b in pairs])) if pairs else None,
                              'mean_output':float(np.mean([b for a,b in pairs])) if pairs else None,
                              'mean_delta':float(np.mean([b-a for a,b in pairs])) if pairs else None}
        summary[group] = {'case_ids':[row['case_id'] for row in rows],'metrics':values}
    return summary


def verify(directory, *, official_corpus=None, expected_script_sha256=RUNNING_SCRIPT_SHA256,
           scorer=None, terminal_receipt=None):
    if terminal_receipt is not None:
        require(terminal_receipt['kernel_slug'].split('/')[-1] == KERNEL_SLUG,'unexpected Kaggle kernel slug')
        require(terminal_receipt['version'] == 1,'unexpected Kaggle kernel version')
        require(terminal_receipt['status'].upper() == 'COMPLETE','Kaggle kernel is not terminal COMPLETE')
        require(terminal_receipt['script_sha256'] == expected_script_sha256,'terminal uploaded script hash mismatch')
    directory = Path(directory)
    report = read_json(directory/'quality-report.json')
    require(not (directory/'failure.json').exists(),'failure.json exists; review terminal failure before acceptance')
    require(report['status'] == 'complete' and report['completed_cases'] == 16,'expected complete 16-case report')
    require(report['script_sha256'] == expected_script_sha256,'unexpected submitted script SHA-256')
    require(report['script_version'] == 'reuse-quality-cpu-20261112-v2','unexpected submitted script version')
    require(report['bundled_source_revision'] == kernel.BUNDLED_SOURCE_REVISION,'bundled source revision mismatch')
    require(report['sample_rate'] == 16000 and report['window_samples'] == 80000,'unexpected window policy')
    require(report['model_files'] == kernel.MODEL_HASHES,'pinned model/source hash map mismatch')
    require(report['bundle'] == {name:item['sha256'] for name,item in kernel.BUNDLE.items()},'bundled source hash map mismatch')
    require(report['official_flac_sha256'] == kernel.CORPUS_FILES,'official corpus hash map mismatch')
    require(report['official_openslr_test_clean_archive_md5'] == kernel.OFFICIAL_ARCHIVE_MD5,'official archive digest mismatch')
    require(report['estoi_normalization_seed'] == 579,'unexpected ESTOI seed')
    selected = report['selected']
    require(len(selected) == 4,'expected four selected recordings')
    by_speaker = {}
    for item in selected:
        speaker = item['speaker']
        require(speaker not in by_speaker,'duplicate selected speaker')
        relative = item['path']
        require(relative.startswith('test-clean/'),'unexpected selected corpus path')
        key = relative[len('test-clean/'):]
        require(key in kernel.CORPUS_FILES and item['source_sha256'] == kernel.CORPUS_FILES[key],
                'selected recording does not match official corpus pin')
        require(str(speaker) == key.split('/')[0],'selected speaker/path mismatch')
        require(item['sex'] == ('F' if speaker in (121,237) else 'M'),'selected speaker metadata mismatch')
        by_speaker[speaker] = item
    conditions = {'clean':None,'white-0db':0,'white-10db':10,'pink-5db':5}
    expected_ids = {f'speaker-{speaker}-{condition}' for speaker in (121,237,61,260) for condition in conditions}
    aggregate = report['cases']
    require(len(aggregate) == 16 and {x['case_id'] for x in aggregate} == expected_ids,'missing, extra or duplicate cases')
    results = []
    for summary in aggregate:
        case_id = summary['case_id']
        receipt = read_json(directory/(case_id+'.json'))
        require({k:v for k,v in receipt.items() if k != 'stage_fingerprints'} == summary,case_id+': aggregate/case disagreement')
        speaker, condition = receipt['speaker'], receipt['condition']
        require(case_id == f'speaker-{speaker}-{condition}' and speaker in by_speaker and condition in conditions,
                case_id+': case identity mismatch')
        require(receipt['sex'] == by_speaker[speaker]['sex'] and receipt['noise_seed'] == 579000+speaker,
                case_id+': case provenance mismatch')
        require(receipt['target_snr_db'] == conditions[condition],case_id+': target SNR mismatch')
        clean = read_wave(directory/f'speaker-{speaker}-clean.wav',receipt['clean_sha256'])
        raw = read_wave(directory/f'{case_id}-input.wav',receipt['input_sha256'])
        wave = read_wave(directory/f'{case_id}-output.wav',receipt['output_sha256'])
        if official_corpus is not None:
            source = Path(official_corpus)/by_speaker[speaker]['path']
            require(auditor.sha256_file(source) == by_speaker[speaker]['source_sha256'],'local official FLAC mismatch')
            official, rate = sf.read(source,frames=80000,dtype='float32')
            require(rate == 16000 and np.array_equal(clean,official),case_id+': clean crop differs from official recording')
        kind = None if condition == 'clean' else 'pink' if condition.startswith('pink') else 'white'
        generated = kernel.make_noisy(clean,kind,conditions[condition],579000+speaker)
        require(np.array_equal(raw,generated),case_id+': noisy input does not reproduce the pinned recipe')
        require(receipt['stage_count'] == 72 and receipt['comparison_stage_fingerprints'] == 71,
                case_id+': stage count mismatch')
        validate_fingerprints(receipt['stage_fingerprints'],raw,wave)
        computed = {}
        for label, estimate in (('input',raw),('output',wave)):
            computed[label] = independent_metrics(clean,estimate,scorer)
            for name, value in receipt[label+'_metrics'].items():
                if name in ('stoi','estoi') and scorer is None:
                    continue
                require(name in computed[label],case_id+': unknown recorded metric '+name)
                close(value,computed[label][name],case_id+' '+label+' '+name)
            required = {'snr_db','si_sdr_db','projection_gain','rms_ratio','spectral_convergence','peak_abs','fraction_abs_gt_one'}
            if report['stoi_available']:
                required |= {'stoi','estoi'}
            require(required <= receipt[label+'_metrics'].keys(),case_id+': incomplete numerical metrics')
        if condition == 'clean':
            require(np.array_equal(clean,raw),case_id+': clean control input changed')
        else:
            close(computed['input']['snr_db'],conditions[condition],case_id+' measured noise SNR')
        results.append({'case_id':case_id,'speaker':speaker,'condition':condition,'independent_metrics':computed})
    return {'conclusive':scorer is not None and terminal_receipt is not None and official_corpus is not None,
            'terminal_receipt_checked':terminal_receipt is not None,
            'speakers':summarize(results,'speaker'),'conditions':summarize(results,'condition'),
            'verified_cases':16,'kernel_slug':KERNEL_SLUG,'kernel_version':1,
            'script_sha256':report['script_sha256'],'official_clean_crop_comparison':official_corpus is not None,
            'local_stoi_available':scorer is not None,'cases':results,
            'scope':'File hashes, exact input recipe, raw/output fingerprints, stage fingerprint structure and independent waveform metrics verified. Intermediate arrays were not retained, so their fingerprints do not establish native parity.'}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifacts',type=Path,required=True)
    parser.add_argument('--official-corpus',type=Path,required=True)
    parser.add_argument('--terminal-receipt',type=Path,required=True,
                        help='API receipt with kernel_slug, version, terminal status and uploaded script_sha256')
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args(argv)
    try:
        from pystoi import stoi
        result = verify(args.artifacts,official_corpus=args.official_corpus,scorer=stoi,
                        terminal_receipt=read_json(args.terminal_receipt))
    except Exception as exc:
        error = 'required artifact access failed' if isinstance(exc,OSError) else str(exc)
        auditor.write_json(args.output,{'conclusive':False,'kernel_slug':KERNEL_SLUG,
                                       'error_type':type(exc).__name__,'error':error})
        raise
    auditor.write_json(args.output,result)
    print(json.dumps({'verified_cases':16,'kernel_slug':KERNEL_SLUG,'script_sha256':result['script_sha256']}))


if __name__ == '__main__':
    main()
