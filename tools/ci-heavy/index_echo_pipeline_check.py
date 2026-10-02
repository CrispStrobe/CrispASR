"""Full-file oracle checks called by index_echo_validate (not a standalone job)."""
import ctypes
import hashlib
import json
from pathlib import Path
import subprocess
import time
import wave


def check_pipeline(root, out, build, library, models, cohort, reference_subdir, model_prefix='index-echo-2b'):
    import numpy as np
    from huggingface_hub import hf_hub_download
    from crispasr import Session
    # Keep the third companion beside the primary, exercising runtime autoload.
    vad_revision = '9ffd54a1e1ee413ddf265af9913beaf518d1639b'
    vad_path = Path(hf_hub_download('ggml-org/whisper-vad', 'ggml-silero-v6.2.0.bin',
                                  revision=vad_revision, local_dir=models / 'vad-companion'))
    oracle = json.loads((models / reference_subdir / 'pipeline.json').read_text())
    required = {'jfk-en': 'en', 'zh-en': 'en', 'zh-ja': 'ja', 'zh-es': 'es', 'multi-en': 'en'}
    if oracle.get('complete') is False or set(oracle['cases']) != set(required):
        raise RuntimeError('Full-file acceptance requires the complete independent five-case source oracle')
    for name, target in required.items():
        case = oracle['cases'][name]
        if case['target'] != target or not case['segments'] or any(row.get('parse_warn', 0) for row in case['rows']):
            raise RuntimeError('Invalid independent source case: ' + name)
    multi_rows = oracle['cases']['multi-en']['rows']
    if len(multi_rows) != 3 or not multi_rows[1].get('has_ctx'):
        raise RuntimeError('Independent source oracle must exercise a second window with prior context')
    failures, decoded = [], {}

    class VADParams(ctypes.Structure):
        _fields_ = [('n_threads', ctypes.c_int), ('use_gpu', ctypes.c_bool), ('gpu_device', ctypes.c_int)]

    lib = ctypes.CDLL(str(library))
    lib.whisper_vad_init_from_file_with_params.argtypes = [ctypes.c_char_p, VADParams]
    lib.whisper_vad_init_from_file_with_params.restype = ctypes.c_void_p
    lib.whisper_vad_detect_speech.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int]
    lib.whisper_vad_detect_speech.restype = ctypes.c_bool
    lib.whisper_vad_n_probs.argtypes = [ctypes.c_void_p]
    lib.whisper_vad_n_probs.restype = ctypes.c_int
    lib.whisper_vad_probs.argtypes = [ctypes.c_void_p]
    lib.whisper_vad_probs.restype = ctypes.POINTER(ctypes.c_float)
    lib.whisper_vad_free.argtypes = [ctypes.c_void_p]
    vad = lib.whisper_vad_init_from_file_with_params(str(vad_path).encode(), VADParams(1, False, 0))
    lib.crispasr_silero_enable_context.argtypes = [ctypes.c_void_p]
    lib.crispasr_silero_enable_context.restype = ctypes.c_bool
    if not vad or not lib.crispasr_silero_enable_context(vad):
        raise RuntimeError('Native Silero companion could not load')
    companion = models / vad_path.name
    assert not companion.exists(), 'Direct-window fixtures must not autoload VAD'
    companion.symlink_to(vad_path)
    try:
        with Session(str(models / f'{model_prefix}-{cohort}.gguf'), lib_path=str(library), n_threads=4) as session:
            for name, expected in oracle['cases'].items():
                audio = (models / reference_subdir if expected['audio'] == 'pipeline-multi.wav' else root / 'samples') / expected['audio']
                with wave.open(str(audio), 'rb') as wav:
                    assert wav.getframerate() == 16000 and wav.getnchannels() == 1 and wav.getsampwidth() == 2
                    pcm = np.frombuffer(wav.readframes(wav.getnframes()), dtype=np.int16).astype(np.float32) / 32768
                assert lib.whisper_vad_detect_speech(vad, pcm.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), len(pcm))
                n = lib.whisper_vad_n_probs(vad)
                probs = np.ctypeslib.as_array(lib.whisper_vad_probs(vad), shape=(n,)).copy()
                reference_probs = np.array(expected['vad_probabilities'], dtype=np.float32)
                if probs.shape != reference_probs.shape:
                    raise RuntimeError('VAD frame count differs from source: ' + name)
                delta = float(np.max(np.abs(probs - reference_probs)))
                cosine = float(np.dot(probs, reference_probs) / max(np.linalg.norm(probs) * np.linalg.norm(reference_probs), 1e-30))
                if delta > .01 or cosine < .999:
                    failures.append(name + ': VAD classifier mismatch')
                session.set_target_language(expected['target'])
                started = time.perf_counter()
                segments = session.transcribe(pcm)
                elapsed = time.perf_counter() - started
                actual = [dict(start=s.start, end=s.end, text=s.text) for s in segments]
                golden = expected['segments']
                if not golden or len(actual) != len(golden) or any(
                        a['text'] != e['text'] or abs(a['start'] - e['start']) > .0051 or
                        abs(a['end'] - e['end']) > .0051 for a, e in zip(actual, golden)):
                    failures.append(name + ': full-pipeline decoded mismatch')
                decoded[name] = dict(segments=actual, reference=golden, elapsed_seconds=elapsed,
                                     vad_cosine=cosine, vad_max_abs=delta, vad_probabilities=probs.tolist())
                print('full pipeline', cohort, name, json.dumps({k:v for k,v in decoded[name].items() if k != 'vad_probabilities'}, ensure_ascii=False), flush=True)
    finally:
        lib.whisper_vad_free(vad)
        companion.unlink()
    # Real CLI default language flow: metadata/caps must avoid unrelated LID.
    prefix = out / f'pipeline-{cohort}-cli'
    companion.symlink_to(vad_path)
    try:
        with (out / f'pipeline-{cohort}-cli.log').open('w') as log:
            result = subprocess.run([str(build / 'bin/crispasr'), '-m', str(models / f'{model_prefix}-{cohort}.gguf'),
                '-f', str(root / 'samples/jfk.wav'), '-l', 'auto', '-osrt', '-of', str(prefix), '-t', '4', '-ng'],
                cwd=root, stdout=log, stderr=subprocess.STDOUT, timeout=3600)
    finally:
        companion.unlink()
    srt = prefix.with_suffix('.srt')
    if result.returncode or not srt.exists():
        failures.append('real CLI failed')
    else:
        text = srt.read_text()
        for segment in oracle['cases']['jfk-en']['segments']:
            if segment['text'] not in text: failures.append('real CLI decoded text mismatch')
    receipt = dict(cohort=cohort, failed=failures, cases=decoded,
                   vad_file=vad_path.name, vad_revision=vad_revision, vad_sha256=hashlib.sha256(vad_path.read_bytes()).hexdigest())
    (out / f'pipeline-{cohort}.json').write_text(json.dumps(receipt, indent=2, ensure_ascii=False) + '\n')
    return failures
