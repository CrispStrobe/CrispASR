"""Offline CPU RE-USE quality evaluation; original graph, no native parity claim."""
import base64
import gzip
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
import time

VERSION = 'reuse-quality-cpu-20261112-v2'
BUNDLED_SOURCE_REVISION = '69b11562aa7b3a174add9a6767098d4eeae680d8'
# Exact bundled reference adapter and progress harness; generated from a frozen checkout.
BUNDLE = {'mamba_ssm.py': {'sha256': '93203e8896d503703ee2026d258f1027bcd4b3132eb394409392ce7016dec835', 'gzip_base64': 'H4sIAAAAAAAC/4VVTY/bOAy9+1cQuaw8Y2s2waKHKXLIZDpAgS1Q7HZOQWBobDnWxpZcfUyT/PpSluM4zm43MGJKIh8fKZKezWbrr6+pkvURvrDmjaVz0LzkmsucQ6k02IpDK6TkBfz1KX39+xM0quA1jaLPTVvzhktrOiWpZFo6g3qG1zy34p2nJmcS+HfHrFDSgCrBWGZxv2U5Nw+N9/ibibp3ZoQHpO0RmByBZB4kQ1IUvlXCAD4IylwhLLCCtZbrBJ3jAkpmbLTnWvIakDp/F0UXB/pdvz6v4I1X7F0oTaPZbBahO6UtNMxWZ9kqnVdRqVUTROj3pRxtUinP+6WTuQ+N1cAMvERRVPByQp24BDBhliWwSuApgXUCzwmc4scI8IdMPsvWYRIveqdH2Lwxm1cJ5BXD3NcJWNHw7Ud4elhfDv1eElK6pT4kD9gtYQmOSv4jO3GtDCGOmoq1fPP7NoGzPEd5Nchx3FkrZ9F2s+0W3f2DkKCZ3PEBZLHtuftf4fU73pvHBPCx2+Hs4Gncbp8ZhmzyQ0sKS5003x3nJ07SeQx3sPJ/QfMeUAFXh/hG68mjW3SwHR3N48HVEd2QgHIH639XjqlxTYd3D8/ezWCNyaCsbbksyBEPXqgRtSOnIaI+Z5pbp2UfDjrL9wQt8TZFs1zEWBR5zYwJ/UWkpF9U4Wre59AXTJYJKWyWEaycEu2yrsW80FFfzj94OVfyfflHApgwbJDlIoG7u/0PpndmdB3GtVyTmA6Ql1z4JtaYDyEtCRgYU+/rouUp0N6xv9kgTY5thhWxx2PfPDTnoiY9EDzA/MMETcis1eofVMfg/xSSM02GGBdIomOWwJtgZvnCasMnAD7yeRHs151MepP+FZKTwE4r15plv9uyohBytwynkMJ8gnu45dWbXoV535HsM/ELmqh/gzcGSq4C/abdFGCV1WoXzL8yzRqOs430XdK09kiGcDsq8cT8+X9NpxZYpv+ZguGGxvEOJYujAUuv6Cu2EgVO2sBqXI01lztbIfqVwjBzLoMCZx5qjauFXGNSixk0rTLYsVg0Mc0rJ/dkEbpsdLF+5vSNOqodcog3lFLs20BpezEoWY55Mmf3oSbIYeowHk28MMaHCWbaWljS4yQwufNxQ12v4kA+HbHvBmmIQJW2rZ0h49LCORjfpGIwXqFhehmql5KiZa2YJfH1XJx8pg63n6lQVWdr/80aAPqZd1VF5HibtJ/Gr+3ZYQgAAA=='}, 'kaggle_harness.py': {'sha256': 'ae2f378ef132fe4d366e0809f94d817a08f5ebc759f62c2ca753e25b12148283', 'gzip_base64': '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'}}
MODEL_HASHES = {'config.json': 'e69986483774f01893d2727503addc49ef201d19e6e1a266b8d4e971e28c478d', 'model.safetensors': '06badaf4029b297f97f7fe48d62990b48d5482608210df577354e0d4e0ed70e1', 'models/codec_module_time_d4.py': '02bd1d478f36a9b38dce7d44368805298f42798ea239304580f1ccffafe766a0', 'models/generator_SEMamba_time_d4.py': 'beebaf2d9ab7b83c101f46246ae7a7a8a2d2f57b6683025d3712c64c1b7d6817', 'models/mamba_block2_SEMamba.py': '97397af2d7e0f5e6530aa8c4a45b4f0f0030b4b56d8c9922a9b0bc25f8b5962e', 'models/stfts.py': 'b82383c1ec7d7e9fe9bca1f93b3c0afe9609e191c38efc16f705e7145198d835'}
CORPUS_FILES = {'121/121726/121-121726-0000.flac': '81378fdb3e648b43efb1cb8c4f87da3fc9cfbe19c5dda97e9e7361c4a2470d2f', '237/126133/237-126133-0000.flac': '25466538df82602d1e597b0cb016e4dd0ce11031cb8994a4289d1c07170878ee', '61/70968/61-70968-0005.flac': '1022f303c108707548ff8738b7d7f512403b8fa443b9664bab94f740ef412183', '260/123286/260-123286-0000.flac': 'd654a43d06f0e8b586b525e8ec83d6e3c0dc83c3f4742f8e25fd291db77248b3'}
OFFICIAL_ARCHIVE_MD5 = '32fa31d27d2e1cad72775fee3f4849a9'
RATE = 16000
LENGTH = 80000
WALL_BUDGET = 7200


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def metrics(clean, estimate, stoi=None):
    import numpy as np
    from scipy.signal import stft
    c = np.asarray(clean, dtype=np.float64)
    y = np.asarray(estimate, dtype=np.float64)
    if c.ndim != 1 or not c.size or c.shape != y.shape or not np.isfinite(c).all() or not np.isfinite(y).all():
        raise ValueError('invalid waveform shape or nonfinite samples')
    cc = c - c.mean()
    yy = y - y.mean()
    energy = float(np.dot(c, c))
    if energy <= 1e-20 or np.dot(cc, cc) <= 1e-20:
        raise ValueError('zero energy reference')
    gain = float(np.dot(cc, yy) / np.dot(cc, cc))
    target = gain * cc
    error = yy - target
    p = float(np.dot(target, target))
    _, _, C = stft(c, RATE, nperseg=512, noverlap=384)
    _, _, Y = stft(y, RATE, nperseg=512, noverlap=384)
    result = dict(snr_db=float(10 * np.log10(energy / max(float(np.dot(c-y, c-y)), 1e-30))),
                  si_sdr_db=None if p <= 1e-30 else float(10 * np.log10(p / max(float(np.dot(error, error)), 1e-30))),
                  projection_gain=gain, rms_ratio=float(np.linalg.norm(y) / np.linalg.norm(c)),
                  spectral_convergence=float(np.linalg.norm(abs(C)-abs(Y)) / np.linalg.norm(abs(C))),
                  peak_abs=float(np.max(abs(y))), fraction_abs_gt_one=float(np.mean(abs(y) > 1)))
    if stoi is not None:
        # pystoi ESTOI adds epsilon-scale random jitter during normalization.
        np.random.seed(579)
        result.update(stoi=float(stoi(c, y, RATE)), estoi=float(stoi(c, y, RATE, extended=True)))
    return result


def metric_controls():
    import numpy as np
    rng = np.random.default_rng(579)
    x = rng.standard_normal(16000)
    assert abs(metrics(x, .5*x)['snr_db'] - 6.020599913279624) < 1e-10
    assert metrics(x, .5*x)['si_sdr_db'] > 250
    assert metrics(x, np.zeros_like(x))['si_sdr_db'] is None
    assert abs(metrics(x, np.zeros_like(x))['snr_db']) < 1e-12
    assert metrics(x, np.roll(x, 1))['si_sdr_db'] < -20


def verify_pinned_files(root, expected):
    actual = {}
    for name, digest in expected.items():
        actual[name] = sha(root/name)
        if actual[name] != digest:
            raise ValueError('pinned file hash mismatch: ' + name)
    return actual


def select_corpus(corpus, expected=CORPUS_FILES):
    import soundfile as sf
    verify_pinned_files(corpus/'test-clean', expected)
    speakers = []
    for line in (corpus/'SPEAKERS.TXT').read_text().splitlines():
        if line.startswith(';') or not line.strip():
            continue
        fields = [x.strip() for x in line.split('|')]
        if len(fields) >= 3 and fields[2] == 'test-clean':
            speakers.append((int(fields[0]), fields[1]))
    selected = []
    for sex in ('F', 'M'):
        for speaker, gender in sorted(speakers):
            if gender != sex:
                continue
            for path in sorted((corpus/'test-clean'/str(speaker)).rglob('*.flac')):
                info = sf.info(path)
                if info.samplerate == RATE and info.channels == 1 and info.frames >= LENGTH:
                    selected.append(dict(speaker=speaker, sex=gender, path=path))
                    break
            if sum(x['sex'] == sex for x in selected) == 2:
                break
    if len(selected) != 4:
        raise ValueError('could not select two female and two male test-clean speakers')
    if {str(x['path'].relative_to(corpus/'test-clean')) for x in selected} != set(expected):
        raise ValueError('deterministic selection differs from pinned official selection')
    return selected


def make_noisy(clean, kind, snr, seed):
    import numpy as np
    clean = np.asarray(clean, dtype=np.float32)
    if clean.ndim != 1 or not len(clean) or not np.isfinite(clean).all() or np.dot(clean, clean) <= 0:
        raise ValueError('noise requires finite nonzero mono clean speech')
    if kind is None:
        return clean.copy()
    if kind not in ('white', 'pink') or snr is None or not np.isfinite(snr):
        raise ValueError('unsupported noise or SNR')
    noise = np.random.default_rng(seed).standard_normal(len(clean))
    if kind == 'pink':
        freq = np.fft.rfftfreq(len(clean), 1/RATE)
        scale = np.zeros_like(freq)
        scale[1:] = 1/np.sqrt(freq[1:])
        noise = np.fft.irfft(np.fft.rfft(noise)*scale, n=len(clean))
    if np.dot(noise, noise) <= 0:
        raise ValueError('noise has zero energy')
    scale = np.sqrt(np.mean(clean.astype(np.float64)**2)/(np.mean(noise**2)*10**(snr/10)))
    return (clean + noise*scale).astype('float32')


def stage_fingerprint(array):
    import numpy as np
    a = np.ascontiguousarray(array, dtype=np.float32)
    if not a.size or not np.isfinite(a).all():
        raise ValueError('empty or nonfinite stage')
    return dict(shape=list(a.shape), dtype=str(a.dtype), sha256=hashlib.sha256(a.tobytes()).hexdigest(),
                norm=float(np.linalg.norm(a.astype(np.float64))), max_abs=float(np.max(abs(a))))


def expected_stages():
    return {'raw_audio','frontend_magnitude','frontend_phase','amplitude','output_phase','waveform'} | {f'encoder_layer_{i}' for i in range(6)} | {f'{prefix}_{i}' for i in range(30) for prefix in ('block_time','block')}


def validate_stages(stages):
    if set(stages) != expected_stages():
        raise ValueError('incomplete original-graph fingerprints')


def validate_waveforms(clean, noisy, enhanced):
    import numpy as np
    arrays = [np.asarray(x) for x in (clean, noisy, enhanced)]
    if arrays[0].ndim != 1 or not len(arrays[0]) or any(x.shape != arrays[0].shape or not np.isfinite(x).all() for x in arrays):
        raise ValueError('clean/noisy/enhanced waveform mismatch')


def checkpoint_case(out, result, report):
    # Write the complete per-case receipt before advancing the aggregate.
    # If aggregate replacement fails, earlier receipts/report remain readable.
    save(out/f"{result['case_id']}.json", result)
    updated = {**report, 'cases': report['cases'] + [{k:v for k,v in result.items() if k != 'stage_fingerprints'}]}
    save(out/'quality-report.json', updated)
    return updated


def record_failure(out, exc):
    out.mkdir(parents=True, exist_ok=True)
    save(out/'failure.json', dict(script_version=VERSION, error_type=type(exc).__name__, error=str(exc)))


def main():
    os.environ.update(OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1',
                      HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', MKL_ENABLE_INSTRUCTIONS='AVX2')
    os.environ.pop('MKL_CBWR', None)
    start = time.monotonic()
    out = Path(os.environ.get('REUSE_QUALITY_OUT', '/kaggle/working'))
    root = Path(os.environ.get('REUSE_QUALITY_TEMP', '/kaggle/temp/reuse-quality-cpu'))
    out.mkdir(parents=True, exist_ok=True)
    root.mkdir(parents=True, exist_ok=True)
    report = dict(script_version=VERSION, script_sha256=sha(__file__), bundled_source_revision=BUNDLED_SOURCE_REVISION, status='preflight', cases=[],
                  scope='Original graph + audited CPU non-fused Mamba adapter. New corpus quality only; stage fingerprints are not native parity or a perceptual acceptance gate.',
                  dataset='ayoubakanoun/librispeechtest', attribution='LibriSpeech test-clean, OpenSLR 12, CC BY 4.0, https://www.openslr.org/12',
                  model_source='nvidia/RE-USE pinned model dataset', sample_rate=RATE, window_samples=LENGTH,
                  wall_budget_seconds=WALL_BUDGET, estoi_normalization_seed=579, bundle={k:v['sha256'] for k,v in BUNDLE.items()})
    save(out/'quality-report.json', report)
    for name, item in BUNDLE.items():
        data = gzip.decompress(base64.b64decode(item['gzip_base64']))
        if hashlib.sha256(data).hexdigest() != item['sha256']:
            raise ValueError('bundled source hash mismatch: ' + name)
        (root/name).write_bytes(data)
    sys.path.insert(0, str(root))
    import kaggle_harness as kh
    kh.init_progress(progress_path=out/'progress.jsonl', hf_progress_repo='')
    kh.step('quality.preflight', version=VERSION)
    required = ('numpy', 'scipy', 'soundfile', 'torch', 'safetensors.torch', 'einops', 'huggingface_hub')
    versions = {}
    for name in required:
        mod = importlib.import_module(name)
        versions[name] = getattr(mod, '__version__', 'not exposed')
    import numpy as np
    import soundfile as sf
    import torch
    from safetensors.torch import load_file
    try:
        from pystoi import stoi
    except ImportError:
        stoi = None
    report.update(dependencies=versions, stoi_available=stoi is not None, cuda_available=torch.cuda.is_available(),
                  adapter_scope='Audited Python non-fused selective scan, CPU only')
    metric_controls()
    if '--preflight-only' in sys.argv:
        report['status'] = 'dependency-and-metric-controls-pass'
        save(out/'quality-report.json', report)
        return
    input_root = Path(os.environ.get('REUSE_QUALITY_INPUT', '/kaggle/input'))
    models = {str(p.resolve()):p for p in input_root.rglob('model.safetensors')
              if (p.parent/'config.json').is_file()}
    if len(models) != 1:
        raise ValueError(f'expected one pinned model, got {len(models)}')
    model_dir = next(iter(models.values())).parent
    report['model_files'] = verify_pinned_files(model_dir, MODEL_HASHES)
    corpus = sorted({str(p.resolve()):p for p in input_root.rglob('SPEAKERS.TXT')}.values())
    if len(corpus) != 1:
        raise ValueError('expected one LibriSpeech metadata root')
    corpus = corpus[0].parent
    report['official_openslr_test_clean_archive_md5'] = OFFICIAL_ARCHIVE_MD5
    report['official_flac_sha256'] = CORPUS_FILES
    report['corpus_metadata'] = {name:sha(corpus/name) for name in ('SPEAKERS.TXT','LICENSE.TXT','README.TXT')}
    report['corpus_license_text'] = (corpus/'LICENSE.TXT').read_text()
    selected = select_corpus(corpus)
    report['selection_policy'] = 'First two female + first two male numeric speaker IDs; first lexicographic utterance at least 5s per speaker; first 5s crop, no padding or normalization.'
    report['selected'] = [{**x,'path':str(x['path'].relative_to(corpus)), 'source_sha256':sha(x['path'])} for x in selected]
    save(out/'quality-report.json', report)
    if '--data-preflight-only' in sys.argv:
        report['status'] = 'inputs-and-metric-controls-pass'
        save(out/'quality-report.json', report)
        return
    sys.path.insert(1, str(model_dir))
    from models.generator_SEMamba_time_d4 import SEMamba
    from models.stfts import mag_phase_stft, mag_phase_istft
    torch.set_num_threads(2)
    torch.set_grad_enabled(False)
    model = SEMamba(json.loads((model_dir/'config.json').read_text())).eval().cpu()
    model.load_state_dict(load_file(str(model_dir/'model.safetensors'), device='cpu'), strict=True)
    stage_stats = {}
    def observe(name, tensor):
        stage_stats[name] = stage_fingerprint(tensor.detach().float().cpu().contiguous().numpy())
    def hook(name):
        return lambda module,args,result: observe(name,result[0])
    for i, layer in enumerate([model.dense_encoder.dense_conv_1,*model.dense_encoder.dense_block.dense_block,model.dense_encoder.dense_conv_2]):
        layer.register_forward_hook(hook(f'encoder_layer_{i}'))
    for i, block in enumerate(model.TSMamba):
        block.freq_mamba.register_forward_pre_hook(lambda module,args,i=i: observe(f'block_time_{i}',args[0].permute(2,0,1)))
        block.register_forward_hook(hook(f'block_{i}'))
    assert len(expected_stages()) == 72  # 71 comparison stages plus raw input.
    report['status'] = 'running'
    conditions = [('clean',None,None),('white-0db','white',0),('white-10db','white',10),('pink-5db','pink',5)]
    for item in selected:
        clean, rate = sf.read(item['path'],frames=LENGTH,dtype='float32')
        clean = np.ascontiguousarray(clean)
        speaker = item['speaker']
        sf.write(out/f'speaker-{speaker}-clean.wav',clean,RATE,subtype='FLOAT')
        for label,kind,snr in conditions:
            if time.monotonic()-start > WALL_BUDGET:
                report['status'] = 'partial-wall-budget'
                save(out/'quality-report.json',report)
                return
            case_id = f'speaker-{speaker}-{label}'
            seed = 579000 + speaker
            noisy = make_noisy(clean, kind, snr, seed)
            sf.write(out/f'{case_id}-input.wav',noisy,RATE,subtype='FLOAT')
            kh.step('quality.case.start',case_id=case_id)
            case_start = time.monotonic()
            stage_stats.clear()
            observe('raw_audio',torch.from_numpy(noisy))
            mag,phase,_ = mag_phase_stft(torch.from_numpy(noisy[None]),640,80,640,'relu_log1p',addeps=False)
            observe('frontend_magnitude',mag[0].T)
            observe('frontend_phase',phase[0].T)
            with kh.build_heartbeat(case_id,interval_s=30):
                amp,out_phase,_ = model(mag,phase)
            observe('amplitude',amp[0].T)
            observe('output_phase',out_phase[0].T)
            zeros = (torch.expm1(torch.relu(amp)) == 0).sum(1)/amp.shape[1]
            suppressed = zeros > .5
            amp[:,:,(suppressed)[0]] = 0
            wave = mag_phase_istft(amp,out_phase,640,80,640,'relu_log1p')[0]
            observe('waveform',wave)
            validate_stages(stage_stats)
            waveform = wave.numpy().copy()
            validate_waveforms(clean, noisy, waveform)
            sf.write(out/f'{case_id}-output.wav',waveform,RATE,subtype='FLOAT')
            result = dict(case_id=case_id,speaker=speaker,sex=item['sex'],condition=label,noise_seed=seed,
                          target_snr_db=snr,input_metrics=metrics(clean,noisy,stoi),output_metrics=metrics(clean,waveform,stoi),
                          elapsed_seconds=time.monotonic()-case_start,suppressed_frame_count=int(suppressed.sum()),
                          stage_count=len(stage_stats),comparison_stage_fingerprints=71,stage_fingerprints=dict(stage_stats),
                          clean_sha256=sha(out/f'speaker-{speaker}-clean.wav'),
                          input_sha256=sha(out/f'{case_id}-input.wav'),output_sha256=sha(out/f'{case_id}-output.wav'))
            report = checkpoint_case(out, result, report)
            kh.step('quality.case.done',case_id=case_id,seconds=result['elapsed_seconds'])
    report['status'] = 'complete'
    report['completed_cases'] = len(report['cases'])
    report['elapsed_seconds'] = time.monotonic()-start
    save(out/'quality-report.json',report)
    kh.step('quality.complete',cases=len(report['cases']))


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        out = Path(os.environ.get('REUSE_QUALITY_OUT', '/kaggle/working'))
        record_failure(out, exc)
        raise
