#!/usr/bin/env python3
"""Prepare disjoint CC0 EN/ZH audio on CPU; no model execution or GPU work."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import wave
import zipfile

from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(os.environ['HEAVY_OUT'])
TEMP = Path(os.environ['HEAVY_SCRATCH']) / 'echo-calibration-corpus'
TEMP.mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)
os.environ.update(HF_HOME=str(TEMP / 'hf'), HF_XET_CACHE=str(TEMP / 'xet'))
CV_REPO = 'fsicoli/common_voice_17_0'
CV_REVISION = '8262c16bf297c87a9cd88c51997c4758ed7a8ba2'
ZH_SHARD = 'audio/zh-CN/dev/zh-CN_dev_0.tar'
ZH_SHA256 = 'f41838e75637bd9fcb577cef04df6284e50d17245f7e46afc6142a259190ec49'
EN_REPO = 'cstr/crispasr-imatrix-calib'
EN_REVISION = '1ccafd189b801db25fbb94cacf493b6cfa02a35b'
FIXTURES = 'cstr/crispasr-regression-fixtures'
REFERENCE_REVISION = 'cb678dfd4806778aa55c39fe7b7a710a54cdc153'
AUDIO_REVISION = 'd0a7d7a8be318a5841dfdbe6ad37d3acf75523e3'


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def download(repo, path, revision, dataset=False):
    return Path(hf_hub_download(repo, path, repo_type='dataset' if dataset else 'model',
        revision=revision, local_dir=TEMP / 'downloads' / repo.split('/')[-1]))


def pcm_hash(path):
    with wave.open(str(path), 'rb') as wav:
        assert (wav.getframerate(), wav.getnchannels(), wav.getsampwidth()) == (16000, 1, 2)
        pcm = wav.readframes(wav.getnframes())
        return hashlib.sha256(pcm).hexdigest(), len(pcm) // 2


if not shutil.which('ffmpeg'):
    subprocess.run(['sudo', 'apt-get', 'update', '-qq'], check=True)
    subprocess.run(['sudo', 'apt-get', 'install', '-y', '-qq', 'ffmpeg'], check=True)
# Archive the exact source license and retain both immutable upstream pins.
license_path = download(CV_REPO, 'README.md', CV_REVISION, True)
assert 'license: cc0-1.0' in license_path.read_text()
shutil.copy2(license_path, OUT / 'commonvoice-source-card.md')
manifest = dict(license='CC0-1.0', purpose='independent EN/ZH decoder calibration corpus',
    model_accepted=False, quantization_accepted=False,
    source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
    commonvoice_repo=CV_REPO, commonvoice_revision=CV_REVISION,
    english_repo=EN_REPO, english_revision=EN_REVISION,
    chinese_shard=ZH_SHARD, chinese_shard_sha256=ZH_SHA256,
    selection='24 existing EN dev clips and first 24 MP3 members of pinned zh-CN dev shard; no outcome selection',
    conversion='ffmpeg mono 16 kHz signed PCM16; fixed corpus input, not a reference parity claim',
    ffmpeg=subprocess.check_output(['ffmpeg', '-version'], text=True).splitlines()[0],
    clips=[], heldout_pcm_sha256={})
heldout = [ROOT / 'samples/jfk.wav', ROOT / 'samples/paraformer_zh.wav']
for path in ('index-echo-9b-f32-generation/jfk_tail/audio.wav',
             'index-echo-9b/pipeline-zh-context/audio.wav'):
    heldout.append(download(FIXTURES, path, REFERENCE_REVISION))
roundtrips = json.loads(download(FIXTURES, 'index-echo-9b/roundtrip-piper/roundtrip-audio.json', AUDIO_REVISION).read_text())
for case in roundtrips['cases'].values():
    path = download(FIXTURES, 'index-echo-9b/roundtrip-piper/' + case['audio'], AUDIO_REVISION)
    assert digest(path) == case['sha256']
    heldout.append(path)
for path in heldout:
    manifest['heldout_pcm_sha256'][str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else path.name] = pcm_hash(path)[0]
excluded = set(manifest['heldout_pcm_sha256'].values())
corpus = OUT / 'corpus'
corpus.mkdir(exist_ok=True)


def add_clip(source, language, number, provenance):
    path = corpus / f'{language}_{number:03d}.wav'
    subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-i', str(source),
                    '-ac', '1', '-ar', '16000', '-c:a', 'pcm_s16le', str(path)], check=True)
    pcm_sha, samples = pcm_hash(path)
    assert samples > 0 and pcm_sha not in excluded, 'Empty or held-out acceptance audio in corpus'
    manifest['clips'].append(dict(path=path.name, language=language, source=provenance,
        source_sha256=digest(source), wav_sha256=digest(path), pcm_sha256=pcm_sha,
        samples=samples, sample_rate=16000, duration_s=samples / 16000))


for number in range(24):
    path = f'en/en_{number:03d}.mp3'
    add_clip(download(EN_REPO, path, EN_REVISION, True), 'en', number, path)
shard = download(CV_REPO, ZH_SHARD, CV_REVISION, True)
assert digest(shard) == ZH_SHA256
with tarfile.open(shard) as archive:
    count = 0
    for member in archive:
        if not member.isfile() or not member.name.endswith('.mp3'):
            continue
        source = TEMP / 'current-zh.mp3'
        with archive.extractfile(member) as stream, source.open('wb') as target:
            shutil.copyfileobj(stream, target)
        add_clip(source, 'zh', count, member.name)
        source.unlink()
        count += 1
        if count == 24:
            break
    assert count == 24, 'Pinned shard must contain all 24 clips'
assert len({clip['pcm_sha256'] for clip in manifest['clips']}) == 48, 'Duplicate corpus PCM'
manifest['total_duration_s'] = sum(clip['duration_s'] for clip in manifest['clips'])
manifest['heldout_pcm_disjoint'] = True
manifest['corpus_prepared'] = True
(corpus / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
archive = OUT / 'index-echo-en-zh-calibration.zip'
with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as zipped:
    for path in sorted(corpus.iterdir()):
        zipped.write(path, path.name)
shutil.copy2(corpus / 'manifest.json', OUT / 'corpus-manifest.json')
(OUT / 'corpus-archive.json').write_text(json.dumps(dict(path=archive.name,
    bytes=archive.stat().st_size, sha256=digest(archive)), indent=2) + '\n')
shutil.rmtree(corpus)
(OUT / 'summary.md').write_text('Prepared 24 EN + 24 ZH CC0 clips, PCM-disjoint from all Echo acceptance inputs.\nNo model, imatrix or quantization acceptance is claimed.\n')
