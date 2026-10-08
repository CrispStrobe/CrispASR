#!/usr/bin/env python3
"""Full-clip official RVQ oracle on saved native and official encoder pools.

Same dequantized Q4 weights; not original-checkpoint or full-ASR acceptance.
CPU work only. Download only terminal Kaggle proof or an existing GH artifact.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import urllib.request

os.environ.update(OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4')
import gguf
import numpy as np
from huggingface_hub import hf_hub_download
from pr492_acceptance import PINS, digest, metrics

UPSTREAM = '691ce54144a6844cc641fd96046a6ba20776c8b0'
QUANTIZER_SHA = 'f0e856da2ad7dc52b8aad99d52f217eab44f1ab3620d386a8e706f30725a95c5'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--artifact-run', type=int)
    parser.add_argument('--kaggle', action='store_true')
    parser.add_argument('--expected-source', required=True)
    args = parser.parse_args()
    assert bool(args.artifact_run) != args.kaggle
    out = Path(os.environ['HEAVY_OUT']); out.mkdir(parents=True, exist_ok=True)
    scratch = Path(os.environ['HEAVY_SCRATCH']) / 'rvq-oracle'; scratch.mkdir(parents=True, exist_ok=True)
    receipt = dict(scope=__doc__, upstream=UPSTREAM, quantizer_sha256=QUANTIZER_SHA,
                   expected_source=args.expected_source, arms={}, full_pr_acceptance=False, passed=False)
    def save():
        (out / 'rvq-oracle.json').write_text(json.dumps(receipt, indent=2) + '\n')
    save()
    if args.kaggle:
        from kaggle import KaggleApi
        api = KaggleApi(); api.authenticate()
        ref = os.environ['KAGGLE_ACCOUNT'] + '/crispasr-mimo-pr492-tokenizer-cuda'
        status = json.loads(str(api.kernels_status(ref)))
        assert status['status'] in ['COMPLETE', 'ERROR'], status
        receipt['kaggle_status'] = status
        pattern = r'^mimo-tokenizer/(hardware.json|python-tok_pool_out.npy|tokenizer-diagnosis.json|(q4|promoted)-(flash|eager)/(tok_pool_out|tok_codes).npy)$'
        token = None
        for _ in range(10):
            _, token = api.kernels_output(ref, path=str(scratch), file_pattern=pattern,
                force=True, quiet=True, page_token=token, page_size=100)
            if not token: break
        assert not token
        study_dir = scratch / 'mimo-tokenizer'
    else:
        subprocess.run(['gh', 'run', 'download', str(args.artifact_run), '--repo', 'CrispStrobe/CrispASR',
            '--name', f'heavy-{args.artifact_run}', '--dir', str(scratch / 'artifact')], check=True)
        files = list((scratch / 'artifact').rglob('tokenizer-diagnosis.json')); assert len(files) == 1
        study_dir = files[0].parent
        receipt['artifact_run'] = args.artifact_run
    study = json.loads((study_dir / 'tokenizer-diagnosis.json').read_text())
    assert study['source'] == args.expected_source, study['source']
    receipt['native_source'] = study['source']
    receipt['study_sha256'] = digest(study_dir / 'tokenizer-diagnosis.json')
    repo, revision, filename, sha = PINS['codec']
    codec = Path(hf_hub_download(repo, filename, revision=revision, local_dir=scratch / 'model'))
    assert digest(codec) == sha
    receipt['codec'] = dict(repo=repo, revision=revision, filename=filename, sha256=sha)
    source = scratch / 'quantization.py'
    urllib.request.urlretrieve(f'https://raw.githubusercontent.com/XiaomiMiMo/MiMo-Audio/{UPSTREAM}/src/mimo_audio_tokenizer/quantization.py', source)
    assert digest(source) == QUANTIZER_SHA
    import torch
    torch.set_default_device('cpu'); torch.set_num_threads(4); torch.set_num_interop_threads(1)
    env = {}; exec(compile(source.read_text(), str(source), 'exec'), env)
    reader = gguf.GGUFReader(str(codec)); tensors = {t.name: t for t in reader.tensors}
    books = [np.asarray(tensors[f'encoder.quant.vq.layers.{s}._codebook.embed'].data,
                        dtype=np.float32).reshape(-1, 1280).copy() for s in range(8)]
    quant = env['ResidualVectorQuantizer'](dimension=1280, n_q=8, bins=[len(b) for b in books],
                                         kmeans_init=True).float().eval()
    with torch.inference_mode():
        for layer, book in zip(quant.vq.layers, books):
            layer._codebook.embed.copy_(torch.from_numpy(book))
        def encode(pool):
            x = quant.encode(torch.from_numpy(pool.copy())).numpy()
            assert x.shape == (8, len(pool))
            return x.T.astype(np.int32)
        reference_pool = np.load(study_dir / 'python-tok_pool_out.npy').reshape(-1, 1280)
        assert np.isfinite(reference_pool).all()
        reference_codes = encode(reference_pool)
        np.save(out / 'official-encoder-codes.npy', reference_codes)
        receipt['reference_pool_sha256'] = digest(study_dir / 'python-tok_pool_out.npy')
        for arm in ['q4-flash', 'q4-eager', 'promoted-flash', 'promoted-eager']:
            pool = np.load(study_dir / arm / 'tok_pool_out.npy').reshape(reference_pool.shape)
            native = np.load(study_dir / arm / 'tok_codes.npy').reshape(-1, 8).astype(np.int32)
            assert native.shape == reference_codes.shape and np.isfinite(pool).all()
            official = encode(pool)
            np.save(out / (arm + '-official-own-input-codes.npy'), official)
            def compare(a, b):
                return dict(exact=bool(np.array_equal(a, b)), mismatches=int(np.count_nonzero(a != b)),
                            match_fraction=float(np.mean(a == b)),
                            mismatch_positions=np.argwhere(a != b).tolist())
            receipt['arms'][arm] = dict(pool_reference=metrics(pool, reference_pool),
                native_vs_official_own_input=compare(native, official),
                native_vs_official_encoder=compare(native, reference_codes),
                official_own_vs_official_encoder=compare(official, reference_codes),
                pool_sha256=digest(study_dir / arm / 'tok_pool_out.npy'),
                native_codes_sha256=digest(study_dir / arm / 'tok_codes.npy'))
            save()
    receipt['passed'] = all(receipt['arms'][arm]['native_vs_official_own_input']['exact']
                           for arm in receipt['arms']) and receipt['arms']['promoted-eager']['native_vs_official_encoder']['exact']
    save()
    print(json.dumps(receipt), flush=True)
    assert receipt['passed'], 'Full-clip RVQ oracle gate failed; inspect saved own-input and encoder comparisons'


if __name__ == '__main__':
    main()
