"""Original-only ten-second RE-USE corpus context/chunking comparison."""
import base64
import gzip
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

VERSION = 'reuse-corpus-ten-second-20261113-v1'
BUNDLE = None  # populated by prepare.py in the submitted copy
SOURCES = {
    61: ('61-70970-0038.flac', 'b91945bea4863b159ab19468d55b238480cd6ee95717a865b63c56a91243e9ff'),
    121: ('121-123859-0000.flac', 'bd507b31ede22afb0a49e9386ce62c3168cf6b2d93e88b6b548ff109d9d070b0'),
    237: ('237-134500-0015.flac', '6ffebcfa98310cfce7cf46ac74fb602237fafc1b2b1915142c3f4c08b8f31fbe'),
    260: ('260-123440-0004.flac', '288d22f60ab2527ac2e4035a16c9095438c71deed8becde769d91a964234d139'),
}
STAGES = {'raw_audio', 'frontend_magnitude', 'frontend_phase', 'encoder_layer_0',
          'block_29', 'amplitude', 'output_phase', 'waveform'}


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            value.update(block)
    return value.hexdigest()


def save(path, value):
    temporary = path.with_suffix('.partial')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def stage_archive(path, captures):
    import numpy as np
    if set(captures) != STAGES:
        raise ValueError('incomplete selected intermediate reference')
    if any(not np.isfinite(x).all() for x in captures.values()):
        raise ValueError('nonfinite selected reference')
    temporary = path.with_suffix('.partial')
    with temporary.open('wb') as stream:
        np.savez_compressed(stream, **captures)
    temporary.replace(path)


def difference(reference, candidate):
    import numpy as np
    a, b = np.asarray(reference, dtype='float64'), np.asarray(candidate, dtype='float64')
    if a.ndim != 1 or not a.size or a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('invalid waveform pair')
    norm = float(np.linalg.norm(a))
    if norm == 0:
        raise ValueError('zero reference waveform')
    other = float(np.linalg.norm(b))
    return dict(rel_rms=float(np.linalg.norm(a-b)/norm), max_abs=float(np.max(abs(a-b))),
                reference_norm=norm, candidate_norm=other, norm_ratio=other/norm,
                cosine=None if other == 0 else float(np.dot(a,b)/(norm*other)))


def main():
    if BUNDLE is None:
        raise ValueError('prepare the single-file submission first')
    os.environ.update(OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1',
                      MKL_ENABLE_INSTRUCTIONS='AVX2', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    os.environ.pop('MKL_CBWR', None)
    root = Path(os.environ.get('REUSE_TEN_TEMP', '/kaggle/temp/reuse-corpus-ten-second'))
    out = Path(os.environ.get('REUSE_TEN_OUT', '/kaggle/working'))
    inputs = Path(os.environ.get('REUSE_TEN_INPUT', '/kaggle/input'))
    root.mkdir(parents=True, exist_ok=True); out.mkdir(parents=True, exist_ok=True)
    for name, value in BUNDLE.items():
        path = root/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(gzip.decompress(base64.b64decode(value)))
    sys.path.insert(0, str(root))
    import quality_cpu as prior
    # Reuse the already exercised audited CPU adapter/progress harness.
    for name, value in prior.BUNDLE.items():
        (root/name).write_bytes(gzip.decompress(base64.b64decode(value['gzip_base64'])))
    import kaggle_harness as kh
    import numpy as np
    import soundfile as sf
    import torch
    from safetensors.torch import load_file
    from chunked import chunked
    try:
        from pystoi import stoi
    except ImportError:
        stoi = None
    kh.init_progress(progress_path=out/'progress.jsonl', hf_progress_repo='')
    models = {str(path.parent.resolve()):path.parent for path in inputs.rglob('model.safetensors')}
    if len(models) != 1:
        raise ValueError('expected the existing attached original model')
    model_dir = next(iter(models.values()))
    prior.verify_pinned_files(model_dir, prior.MODEL_HASHES)
    sys.path.insert(1, str(model_dir))
    from models.generator_SEMamba_time_d4 import SEMamba
    from models.stfts import mag_phase_stft, mag_phase_istft
    torch.set_num_threads(2); torch.set_grad_enabled(False)
    net = SEMamba(json.loads((model_dir/'config.json').read_text())).cpu().eval()
    net.load_state_dict(load_file(str(model_dir/'model.safetensors'), device='cpu'), strict=True)
    captures = {}
    def observe(name, value):
        captures[name] = value.detach().float().cpu().contiguous().numpy().copy()
    def hook(name):
        return lambda module,args,result: observe(name,result[0])
    net.dense_encoder.dense_conv_1.register_forward_hook(hook('encoder_layer_0'))
    net.TSMamba[-1].register_forward_hook(hook('block_29'))
    report = dict(script_version=VERSION, scope='Original-only whole10s versus original5s/2.5s-overlap reference chunking; no native parity or product-quality claim.',
                  attribution='LibriSpeech test-clean, OpenSLR12, CC BY4.0, https://www.openslr.org/12',
                  source_selection='First eligible >=10s utterance encountered per existing speaker in the official test-clean archive; exact files pinned.',
                  sample_rate=16000, samples=160000, noise='white', snr_db=5,
                  window_samples=80000, hop_samples=40000, selected_stages=sorted(STAGES),
                  torch=torch.__version__, stoi_available=stoi is not None, cases=[])
    save(out/'report.json', report)
    started = time.monotonic()
    for speaker, (filename, digest) in SOURCES.items():
        matching = {str(path.resolve()):path for path in inputs.rglob(filename)}
        if len(matching) != 1:
            raise ValueError('expected one attached corpus utterance: '+filename)
        source = next(iter(matching.values()))
        if sha(source) != digest:
            raise ValueError('official corpus source differs: '+filename)
        clean, rate = sf.read(source, frames=160000, dtype='float32')
        if rate != 16000 or clean.shape != (160000,) or not np.isfinite(clean).all():
            raise ValueError('expected genuine ten-second mono speech, without padding/repetition')
        seed = 5791000+speaker
        noisy = prior.make_noisy(clean, 'white', 5, seed)
        prefix = f'speaker-{speaker}'
        for label, audio in (('clean',clean),('input',noisy)):
            sf.write(out/f'{prefix}-{label}.wav', audio, rate, subtype='FLOAT')
        archives = []
        def enhance(block, label):
            captures.clear()
            observe('raw_audio',torch.from_numpy(block))
            mag,phase,_ = mag_phase_stft(torch.from_numpy(block[None]),640,80,640,'relu_log1p',addeps=False)
            observe('frontend_magnitude',mag[0].T); observe('frontend_phase',phase[0].T)
            kh.step('reference.start',speaker=speaker,label=label,samples=len(block))
            with kh.build_heartbeat(prefix+'.'+label,interval_s=30):
                amp,out_phase,_ = net(mag,phase)
            observe('amplitude',amp[0].T); observe('output_phase',out_phase[0].T)
            zero_fraction = (torch.expm1(torch.relu(amp)) == 0).sum(1)/amp.shape[1]
            amp[:,:,(zero_fraction > .5)[0]] = 0
            wave = mag_phase_istft(amp,out_phase,640,80,640,'relu_log1p')[0]
            observe('waveform',wave)
            value = wave.numpy().copy()
            if value.shape != block.shape:
                raise ValueError('original output length differs')
            path = out/f'{prefix}-{label}-selected-reference.npz'
            stage_archive(path,captures)
            captures.clear()
            sf.write(out/f'{prefix}-{label}.wav',value,rate,subtype='FLOAT')
            archives.append(dict(name=path.name,bytes=path.stat().st_size,stages=sorted(STAGES)))
            save(out/f'{prefix}-references.json',archives)
            return value
        whole = enhance(noisy,'whole')
        chunked_wave, starts = chunked(noisy, lambda block,index,start: enhance(block,f'window-{index}'),
                                      window_samples=80000,hop_samples=40000,max_window_samples=80000)
        if starts != [0,40000,80000]:
            raise ValueError('unexpected ten-second chunk plan')
        sf.write(out/f'{prefix}-chunked.wav',chunked_wave,rate,subtype='FLOAT')
        case = dict(speaker=speaker,source=filename,source_sha256=digest,noise_seed=seed,starts=starts,
                    input_metrics=prior.metrics(clean,noisy,stoi),whole_metrics=prior.metrics(clean,whole,stoi),
                    chunked_metrics=prior.metrics(clean,chunked_wave,stoi),whole_vs_chunked=difference(whole,chunked_wave),
                    selected_references=archives)
        save(out/f'{prefix}-result.json',case)
        report['cases'].append(case);save(out/'report.json',report)
        kh.step('case.done',speaker=speaker)
    report.update(status='complete',elapsed_seconds=time.monotonic()-started)
    save(out/'report.json',report)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        out=Path(os.environ.get('REUSE_TEN_OUT','/kaggle/working'));out.mkdir(parents=True,exist_ok=True)
        save(out/'failure.json',dict(error_type=type(error).__name__,error=str(error)))
        raise
