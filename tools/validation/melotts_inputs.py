#!/usr/bin/env python3
"""Compare native inputs and the final activation with original Melo functions.

Boundary checks only: no alternative neural implementation or ASR oracle. The
model's own dictionary/symbol table are the shared data. OOV neural G2P is
deliberately outside these fixed, dictionary-covered English controls.
"""
import argparse
import ast
import ctypes as ct
import hashlib
import json
import os
from pathlib import Path
import re
import resource
import shutil
import subprocess
import types
import urllib.request

UPSTREAM = '209145371cff8fc3bd60d7be902ea69cbdb7965a'
TOKENIZER_REVISION = '86b5e0934494bd15c9632b12f734a8a67f723594'
SOURCES = {
    'LICENSE': '88a50e5a02bbc2a5c2f084dc19da751aa97b1690f5fda76cd8005c8634d1ca70',
    'melo/text/english.py': '28527f574511bfc13fe1f2f6bbb5bfe4072c06e1c7e9b69598138aef81b9905b',
    'melo/text/english_bert.py': '0987c4ce65ff9aff4c1b23b25f5903a580e4c907b3d1814e2e97db6dcb020805',
    'melo/text/japanese.py': '23fa0ac9e10772012978ab747970d143a6b002ef75d0014c61eada7a16a25758',
    'melo/text/symbols.py': '4c1f3839f7e209b5da45daeb269f955b302d0d4c0df6dbf9be88d534b25ceea2',
    'melo/text/__init__.py': 'eca8c17764c4dcd57d3a71185775d1c6e4d53689d6a9a54708ec63577a2ac8ba',
    'melo/models.py': 'f96eef52845224b5797e54008013a5b6a5fafad2035713dc5181a5db8bf8bef4',
    'melo/modules.py': 'de80baee7dc8d89d621c541b899a84193df4bb3ef95f5e6d076cc37b6bb10bf6',
}


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def select_functions(path, names, namespace, *, arpa=False):
    tree = ast.parse(Path(path).read_text())
    selected = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    if arpa:
        selected += [node for node in tree.body if isinstance(node, ast.Assign) and
                     any(isinstance(target, ast.Name) and target.id == 'arpa' for target in node.targets)]
    if len([n for n in selected if isinstance(n, ast.FunctionDef)]) != len(names):
        raise ValueError('Original oracle function selection changed')
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), 'exec'), namespace)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--library', type=Path, required=True)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--bert', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', choices=['cpu', 'cuda'], required=True)
    args = p.parse_args()
    if os.environ.get('GITHUB_ACTIONS') != 'true' and not (args.device == 'cuda' and Path('/kaggle/working').is_dir()):
        raise RuntimeError('Hosted-only model validation')
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    out = args.output; out.mkdir(parents=True, exist_ok=True)
    receipt = dict(complete=False, input_protocol_only=False, full_neural_parity=False,
                   scope='Original input protocol and final activation; not complete Melo neural parity', upstream=UPSTREAM,
                   tokenizer_revision=TOKENIZER_REVISION, source_sha256=SOURCES,
                   cases=[], resource_checks=[])
    def save():
        (out/'controls.json').write_text(json.dumps(receipt, indent=2))
    def guard(label):
        memory = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
        row = dict(phase=label, available_bytes=int(memory['MemAvailable'].split()[0])*1024,
                   load=list(os.getloadavg()), cpus=os.cpu_count() or 1,
                   scratch_free_bytes=shutil.disk_usage(out).free)
        if args.device == 'cuda':
            row['gpu_free_mib'] = [int(x) for x in subprocess.check_output(
                ['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits'], text=True).splitlines()]
            if not row['gpu_free_mib'] or min(row['gpu_free_mib']) < 1024:
                raise RuntimeError('Insufficient actual GPU headroom')
        receipt['resource_checks'].append(row); save()
        if row['available_bytes'] < 3*1024**3 or row['load'][0] > 2*row['cpus'] or row['scratch_free_bytes'] < 2*1024**3:
            raise RuntimeError('Insufficient hosted resources before '+label)
    source = out/'oracle-source'; source.mkdir(exist_ok=True)
    for name, digest in SOURCES.items():
        dest = source/name; dest.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen('https://raw.githubusercontent.com/myshell-ai/MeloTTS/'+UPSTREAM+'/'+name,
                                    timeout=60) as response:
            dest.write_bytes(response.read())
        if sha(dest) != digest:
            raise ValueError('Original Melo source checksum mismatch')
    import numpy as np
    from gguf import GGUFReader
    from huggingface_hub import hf_hub_download
    from tokenizers import Tokenizer
    guard('load-owned-input-data')
    model = GGUFReader(args.model)
    symbols = json.loads(model.fields['melotts.symbols_json'].contents())
    dictionary = json.loads(model.fields['melotts.cmudict_json'].contents())
    del model
    tokenizer_path = hf_hub_download('google-bert/bert-base-uncased', 'tokenizer.json', revision=TOKENIZER_REVISION)
    tokenizer = Tokenizer.from_file(tokenizer_path)
    receipt['tokenizer_sha256'] = sha(tokenizer_path)
    namespace = dict(__name__='original_melo_input_oracle', re=re)
    # This pinned module contains only the original symbol lists/language maps.
    exec(compile((source/'melo/text/symbols.py').read_text(), 'original-melo-symbols', 'exec'), namespace)
    namespace.update(symbols=symbols, eng_dict=dictionary,
                     tokenizer=types.SimpleNamespace(tokenize=lambda text: tokenizer.encode(text, add_special_tokens=False).tokens))
    def punctuation_only(word):
        if len(word) == 1 and word in symbols:
            return [word]
        raise RuntimeError('OOV neural G2P is outside this owned-input control')
    namespace['_g2p'] = punctuation_only
    select_functions(source/'melo/text/japanese.py', {'distribute_phone'}, namespace)
    select_functions(source/'melo/text/english.py', {'post_replace_ph', 'refine_ph', 'refine_syllables', 'g2p'}, namespace, arpa=True)
    select_functions(source/'melo/text/__init__.py', {'cleaned_text_to_sequence'}, namespace)
    guard('load-original-activation-runtime')
    import torch
    generator = next(node for node in ast.parse((source/'melo/models.py').read_text()).body
                     if isinstance(node, ast.ClassDef) and node.name == 'Generator')
    forward = next(node for node in generator.body if isinstance(node, ast.FunctionDef) and node.name == 'forward')
    final_calls = [node.value for node in forward.body if isinstance(node, ast.Assign)
                   and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Attribute)
                   and isinstance(node.value.func.value, ast.Name) and node.value.func.value.id == 'F'
                   and node.value.func.attr == 'leaky_relu']
    if len(final_calls) != 1 or len(final_calls[0].args) != 1 or final_calls[0].keywords:
        raise ValueError('Original final default activation changed')
    final_operator = compile(ast.Expression(body=final_calls[0]), 'original-melo-final-activation', 'eval')
    receipt['torch_version'] = torch.__version__
    class Params(ct.Structure):
        _fields_ = [('n_threads',ct.c_int),('verbosity',ct.c_int),('use_gpu',ct.c_bool),
                    ('noise_scale',ct.c_float),('length_scale',ct.c_float),('noise_w',ct.c_float),
                    ('sdp_ratio',ct.c_float),('speaker_id',ct.c_int),('seed',ct.c_uint32)]
    lib = ct.CDLL(str(args.library))
    lib.melotts_default_params.restype = Params
    lib.melotts_init_from_file.argtypes = [ct.c_char_p, Params]; lib.melotts_init_from_file.restype = ct.c_void_p
    lib.melotts_free.argtypes = [ct.c_void_p]
    lib.melotts_load_bert.argtypes = [ct.c_void_p, ct.c_char_p]; lib.melotts_load_bert.restype = ct.c_bool
    lib.melotts_set_dump_dir.argtypes = [ct.c_void_p, ct.c_char_p]
    lib.melotts_synthesize.argtypes = [ct.c_void_p, ct.c_char_p, ct.POINTER(ct.POINTER(ct.c_float)), ct.POINTER(ct.c_int)]
    lib.melotts_synthesize.restype = ct.c_int; lib.melotts_pcm_free.argtypes = [ct.POINTER(ct.c_float)]
    guard('load-native-model')
    params = lib.melotts_default_params(); params.n_threads=2; params.verbosity=1
    params.use_gpu=args.device=='cuda'; params.seed=42; params.speaker_id=0
    ctx = lib.melotts_init_from_file(os.fsencode(args.model), params)
    if not ctx:
        raise RuntimeError('Native Melo initialization failed')
    try:
        guard('load-native-bert')
        if not lib.melotts_load_bert(ctx, os.fsencode(args.bert)):
            raise RuntimeError('Native BERT initialization failed')
        texts = ['This audio was generated by artificial intelligence.',
                 'The quick brown fox jumps over the lazy dog.',
                 'We enjoy reading books and listening to music in the evening.']
        for index, text in enumerate(texts):
            case = out/('case-'+str(index)); case.mkdir(exist_ok=True)
            lib.melotts_set_dump_dir(ctx, os.fsencode(case))
            guard('native-input-capture-'+str(index))
            pcm = ct.POINTER(ct.c_float)(); rate=ct.c_int()
            count = lib.melotts_synthesize(ctx, text.encode(), ct.byref(pcm), ct.byref(rate))
            try:
                if count <= 0 or not pcm or rate.value != 44100:
                    raise RuntimeError('Native owned-input synthesis failed')
                instrumented = np.ctypeslib.as_array(pcm, shape=(count,)).copy()
            finally:
                lib.melotts_pcm_free(pcm)
            lib.melotts_set_dump_dir(ctx, b'')
            guard('native-uninstrumented-repeat-'+str(index))
            pcm = ct.POINTER(ct.c_float)(); rate=ct.c_int()
            repeated_count = lib.melotts_synthesize(ctx, text.encode(), ct.byref(pcm), ct.byref(rate))
            try:
                if repeated_count > 0 and pcm:
                    plain = np.ctypeslib.as_array(pcm, shape=(repeated_count,)).copy()
                    plain.tofile(case/'audio-uninstrumented.bin')
                if repeated_count != count or not pcm or rate.value != 44100:
                    raise RuntimeError('Instrumentation changed native waveform dimensions')
            finally:
                lib.melotts_pcm_free(pcm)
            relative = float(np.linalg.norm(plain-instrumented)/max(float(np.linalg.norm(plain)), 1e-30))
            peak = float(np.max(np.abs(plain-instrumented)))
            receipt.setdefault('instrumentation_checks', []).append(dict(case=index, relative_l2=relative,
                                                                         peak_absolute_error=peak))
            save()
            if not np.isfinite(plain).all() or relative > 1e-5 or peak > 1e-5:
                raise RuntimeError('Instrumentation changed the native waveform')
            guard('original-final-activation-'+str(index))
            before = np.fromfile(case/'dec_before_post.bin', dtype='<f4')
            after = np.fromfile(case/'dec_post_activation.bin', dtype='<f4')
            negative = int(np.sum(before < 0)); positive = int(np.sum(before > 0))
            if len(before) != len(after):
                receipt.setdefault('activation_checks', []).append(dict(case=index, accepted=False,
                    before_elements=len(before), after_elements=len(after)))
                save()
                raise RuntimeError('Native final-activation tensor dimensions changed')
            reference_activation = eval(final_operator, {'F':torch.nn.functional, 'x':torch.from_numpy(before)}).numpy()
            matches = len(after) == len(reference_activation) and np.allclose(after, reference_activation, rtol=1e-6, atol=1e-7)
            activation = dict(accepted=bool(matches and negative and positive), elements=len(after), negative=negative, positive=positive,
                max_absolute_error=float(np.max(np.abs(after-reference_activation))),
                norm_ratio=float(np.linalg.norm(after)/np.linalg.norm(reference_activation)))
            receipt.setdefault('activation_checks', []).append(dict(case=index, **activation)); save()
            if not activation['accepted']:
                raise RuntimeError('Actual native final activation differs from the original operator')
            # The fixed controls contain no numbers, times or abbreviations;
            # lowercasing is their entire original text_normalize operation.
            phones, tones, word2ph = namespace['g2p'](text.lower())
            phone_ids, tone_ids, lang_ids = namespace['cleaned_text_to_sequence'](
                phones, tones, 'EN', {s:i for i,s in enumerate(symbols)})
            def blanks(values):
                result = np.zeros(2*len(values)+1, dtype='<i4'); result[1::2]=values
                return result
            aligned = np.asarray(word2ph, dtype='<i4')*2; aligned[0] += 1
            expected = dict(phoneme_ids=blanks(phone_ids), tone_ids=blanks(tone_ids), lang_ids=blanks(lang_ids),
                            bert_word2ph=aligned, bert_token_ids=np.asarray(tokenizer.encode(text).ids,dtype='<i4'))
            results = {}
            for name, reference in expected.items():
                reference.tofile(case/(name+'-original.bin'))
                actual = np.fromfile(case/(name+'.bin'),dtype='<i4')
                match = np.array_equal(reference, actual)
                results[name] = dict(exact=bool(match), elements=len(actual), sha256=sha(case/(name+'.bin')))
            row = dict(case=index, accepted=all(v['exact'] for v in results.values()), inputs=results,
                       final_activation=activation,
                       instrumentation=dict(relative_l2=relative, peak_absolute_error=peak))
            receipt['cases'].append(row); save()
            if not row['accepted']:
                raise RuntimeError('Original Melo input boundary differs from the actual native capture')
        receipt['complete'] = True; save()
    finally:
        lib.melotts_free(ctx)


if __name__ == '__main__':
    main()
