#!/usr/bin/env python3
"""Diagnose design-s1 frame zero with identical original code histories on GH CPU.

This does not replace canonical BF16 fixtures or accept generated speech.
Original and native models run in separate processes to release their RAM.
"""
import argparse
import ctypes as C
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(os.environ['HEAVY_OUT'])
SCRATCH = Path(os.environ['HEAVY_SCRATCH']) / 'breeze-depth-probe'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def run(cmd, name, cwd=ROOT):
    with (OUT / (name + '.log')).open('w') as log:
        subprocess.run(list(map(str, cmd)), cwd=cwd, stdout=log,
                       stderr=subprocess.STDOUT, check=True, timeout=7200)


def reference():
    import numpy as np
    import torch
    from huggingface_hub import hf_hub_download
    from breeze_guided_reference import UPSTREAM, MODEL, DUMPER_SHA
    archive = Path(hf_hub_download('cstr/crispasr-regression-fixtures',
        'breeze-guided-20261011/matched-budget-failed-parity-proof.zip', repo_type='dataset',
        revision='27896a1fab69d45c8b246eb659823f92a5e0e458'))
    assert sha(archive) == '13d5eed11955e07f2e5fc2f681df123c1b2b35d2f11998416571dbf962d54f33'
    with zipfile.ZipFile(archive) as z:
        z.extractall(SCRATCH / 'original-proof')
    prior = SCRATCH / 'original-proof/reference'
    fixture = prior / 'breeze-dumps'
    receipt = json.loads((prior / 'validation.json').read_text())
    assert receipt['complete'] and receipt['model_revision'] == MODEL
    for name in ('guided/design-s1/prompt_input_ids.npy',
                 'guided/design-s1/backbone_logits_frame0.npy', 'guided/design-s1/codes.npy'):
        assert sha(fixture / name) == receipt['arrays'][name]['sha256']
    history = np.load(fixture / 'guided/design-s1/codes.npy', allow_pickle=False)[0]
    assert history.shape == (16,) and ((history >= 0) & (history < 2048)).all()
    upstream = SCRATCH / 'upstream'
    run(['git', 'clone', '--no-checkout', 'https://github.com/breezeblue-ai/breeze-tts.git', upstream],
        'upstream-clone')
    run(['git', 'checkout', '--detach', UPSTREAM], 'upstream-checkout', upstream)
    assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=upstream, text=True).strip() == UPSTREAM
    dumper = ROOT / 'tools/kaggle/breeze-refdump/breeze_refdump.py'
    assert sha(dumper) == DUMPER_SHA
    # Execute the reviewed CPU-compatible loading recipe unchanged, with a
    # pinned checkpoint. Stop before any reference audio or generation work.
    code = dumper.read_text().split('# ── checkpoint ─', 1)[1]
    code = code[code.index('from huggingface_hub import snapshot_download'):]
    code = code.split('# ── reference audio → codec codes ─', 1)[0]
    needle = 'snapshot_download(HF_MODEL, local_dir=str(CKPT), token=HF_TOKEN or None,'
    assert code.count(needle) == 1
    code = code.replace(needle, needle + '\n                  revision=MODEL_REVISION,')
    torch.set_grad_enabled(False)
    torch.set_num_threads(4)
    dest = OUT / 'reference'
    dest.mkdir()
    env = dict(__name__='__main__', os=os, sys=sys, Path=Path, np=np, torch=torch,
               BREEZE=upstream, CKPT=SCRATCH / 'checkpoint', HF_MODEL='BreezeBlue/Breeze-TTS-2',
               HF_TOKEN=os.environ.get('HF_TOKEN', ''), MODEL_REVISION=MODEL, DEVICE='cpu',
               GREEDY=True, MAX_FRAMES=24, DUMPS=dest,
               step=lambda name, **kw: print(name, json.dumps(kw), flush=True))
    (dest / 'executed-loading-recipe.py').write_text(code)
    exec(compile(code, str(dumper), 'exec'), env)
    model = env['model']
    assert next(model.parameters()).dtype == torch.bfloat16
    meta = json.loads((fixture / 'meta.json').read_text())
    req = dict(id='design-s1', text=meta['syn_text'], instruction=meta['instruction'], speaker='S0')
    template = env['select_template_name'](req)
    inputs = env['prepare_inputs'](env['tokenizer'], env['audio_tokenizer'], model, [req],
        env['get_template'](template), guidance_scale=1.0,
        guidance_scale_ref=None, guidance_scale_ins=None)
    ids = inputs['input_ids'][0].cpu().numpy().astype(np.int32)
    assert np.array_equal(ids, np.load(fixture / 'guided/design-s1/prompt_input_ids.npy'))
    merged = model._merge_input_ids_with_input_values(
        input_ids=inputs['input_ids'], input_values=inputs['input_values'],
        text_ids_mask=inputs['text_ids_mask'], text_ids_len=inputs['text_ids_len'],
        attention_mask=inputs['attention_mask'])
    embeds = merged['inputs_embeds'] if isinstance(merged, dict) else merged
    bb = model.backbone_model(inputs_embeds=embeds, attention_mask=inputs['attention_mask'],
                              use_cache=False, return_dict=True)
    hidden = bb.last_hidden_state[0, -1]
    logits = model.lm_head(hidden).float().cpu().numpy()
    assert np.array_equal(logits, np.load(fixture / 'guided/design-s1/backbone_logits_frame0.npy'))
    for name, a in dict(prompt_input_ids=ids,
                        prompt_text_mask=inputs['text_ids_mask'][0].cpu().numpy().astype(np.int32),
                        backbone_inputs_embeds=embeds[0].float().cpu().numpy(),
                        backbone_hidden=hidden.float().cpu().numpy(), backbone_logits=logits,
                        history_codes=history).items():
        np.save(dest / (name + '.npy'), a)
    dd = model.depth_decoder
    def full_prefix(label):
        h = hidden.unsqueeze(0).to(next(dd.parameters()).dtype)
        for cb in range(1, 16):
            seq = torch.tensor([[0, *history[:cb]]], dtype=torch.long)
            o = dd(input_ids=seq, backbone_last_hidden_state=h, use_cache=False, return_dict=True)
            np.save(dest / f'{label}_cb{cb}.npy', o.logits[0, -1].float().cpu().numpy())
    full_prefix('bf16_full')
    cache = None
    for cb in range(1, 16):
        seq = [[0, int(history[0])]] if cb == 1 else [[int(history[cb - 1])]]
        o = dd(input_ids=torch.tensor(seq, dtype=torch.long),
               backbone_last_hidden_state=hidden.unsqueeze(0) if cb == 1 else None,
               use_cache=True, past_key_values=cache,
               cache_position=torch.tensor([0, 1] if cb == 1 else [cb]),
               logits_to_keep=1, return_dict=True)
        cache = o.past_key_values
        np.save(dest / f'bf16_cached_cb{cb}.npy', o.logits[0, -1].float().cpu().numpy())
    del cache, o
    # A depth-only arithmetic-precision control using the same BF16-rounded
    # weights and captured BF16 backbone state. This is not a new F32 oracle.
    dd.float()
    full_prefix('promoted_depth_f32_full')
    (dest / 'provenance.json').write_text(json.dumps(dict(upstream=UPSTREAM,
        model_revision=MODEL, reviewed_loader_sha256=DUMPER_SHA, template=template,
        canonical_precision='BF16 eager CPU', history='original design-s1 frame zero',
        promoted_control='depth-only FP32 arithmetic, BF16-rounded weights/hidden',
        prompt_and_backbone_logits_byte_exact=True), indent=2) + '\n')


def native():
    import numpy as np
    from huggingface_hub import hf_hub_download
    run(['sudo', 'bash', 'tools/ci-apt.sh', 'update'], 'apt-update')
    run(['sudo', 'bash', 'tools/ci-apt.sh', 'install', '-y', 'cmake', 'ninja-build',
         'g++', 'libopenblas-dev'], 'apt-install')
    build = SCRATCH / 'build'
    run(['cmake', '-S', ROOT, '-B', build, '-G', 'Ninja', '-DCMAKE_BUILD_TYPE=Release',
         '-DBUILD_SHARED_LIBS=ON', '-DGGML_NATIVE=OFF', '-DGGML_OPENBLAS=ON',
         '-DGGML_CUDA=OFF', '-DGGML_VULKAN=OFF', '-DGGML_METAL=OFF',
         '-DCRISPASR_BUILD_SERVER=OFF', '-DCRISPASR_BUILD_TESTS=OFF'], 'configure')
    run(['cmake', '--build', build, '--target', 'crispasr-lib', '-j4'], 'build')
    libpath = build / 'src/libcrispasr.so'
    lib = C.CDLL(str(libpath))
    class Params(C.Structure):
        _fields_ = [('n_threads', C.c_int), ('verbosity', C.c_int), ('use_gpu', C.c_bool),
                    ('temperature', C.c_float), ('depth_temperature', C.c_float),
                    ('topk', C.c_int), ('topp', C.c_float), ('repetition_penalty', C.c_float),
                    ('seed', C.c_uint64), ('max_new_tokens', C.c_int), ('codec_path', C.c_char_p)]
    F = C.POINTER(C.c_float)
    I = C.POINTER(C.c_int32)
    ptr = lambda a: a.ctypes.data_as(F)
    iptr = lambda a: a.ctypes.data_as(I)
    lib.breeze_tts_2_context_default_params.restype = Params
    lib.breeze_tts_2_init_from_file.argtypes = [C.c_char_p, Params]
    lib.breeze_tts_2_init_from_file.restype = C.c_void_p
    lib.breeze_tts_2_free.argtypes = [C.c_void_p]
    lib.breeze_tts_2_run_depth_history_dump.argtypes = [C.c_void_p, F, I, C.c_int, C.POINTER(F), I]
    lib.breeze_tts_2_run_depth_dump.argtypes = [C.c_void_p, F, C.c_int32, C.POINTER(F), I]
    lib.breeze_tts_2_run_backbone_dump.argtypes = [C.c_void_p, F, C.c_int, F, F, C.POINTER(F), C.c_int]
    lib.breeze_tts_2_run_prefill_embeds_dump.argtypes = [C.c_void_p, I, I, C.c_int, I, C.c_int, F]
    model = Path(hf_hub_download('cstr/breeze-tts-2-GGUF', 'breeze-tts-2-f16.gguf',
        revision='5314fe9ae3ec6eac1a9238c4f6888fdc9fda0e83'))
    assert sha(model) == '08ea3fbffe9701a3d04a4a63554876841d923fdd21e20add36d28f3ac3447da7'
    p = lib.breeze_tts_2_context_default_params()
    p.n_threads, p.use_gpu, p.max_new_tokens = 4, False, 1
    ctx = lib.breeze_tts_2_init_from_file(os.fsencode(model), p)
    assert ctx
    dest = OUT / 'native'
    dest.mkdir()
    ref = OUT / 'reference'
    load = lambda name: np.ascontiguousarray(np.load(ref / (name + '.npy'), allow_pickle=False))
    history, hidden, embeds = load('history_codes'), load('backbone_hidden'), load('backbone_inputs_embeds')
    def depth(label, h):
        lg = np.empty((15, 2051), dtype=np.float32)
        pointers = (F * 15)(*[ptr(row) for row in lg])
        picks = np.empty(16, dtype=np.int32)
        assert lib.breeze_tts_2_run_depth_history_dump(ctx, ptr(h), iptr(history), 16, pointers, iptr(picks)) == 0
        np.save(dest / (label + '_picks.npy'), picks)
        for cb in range(1, 16):
            np.save(dest / f'{label}_cb{cb}.npy', lg[cb - 1])
    try:
        bad = history.copy()
        bad[3] = 2048
        assert lib.breeze_tts_2_run_depth_history_dump(ctx, ptr(hidden), iptr(bad), 16, None, None) == -1
        assert lib.breeze_tts_2_run_depth_history_dump(ctx, ptr(hidden), iptr(history), 15, None, None) == -1
        ordinary = np.empty((15, 2051), np.float32)
        chosen = np.empty(16, np.int32)
        ordinary_ptrs = (F * 15)(*[ptr(row) for row in ordinary])
        assert lib.breeze_tts_2_run_depth_dump(ctx, ptr(hidden), int(history[0]), ordinary_ptrs, iptr(chosen)) == 0
        controlled = np.empty_like(ordinary)
        controlled_ptrs = (F * 15)(*[ptr(row) for row in controlled])
        predicted = np.empty_like(chosen)
        assert lib.breeze_tts_2_run_depth_history_dump(ctx, ptr(hidden), iptr(chosen), 16,
                                                      controlled_ptrs, iptr(predicted)) == 0
        assert np.array_equal(controlled, ordinary) and np.array_equal(predicted, chosen)
        changed = chosen.copy()
        changed[3] = (changed[3] + 1) % 2048
        assert lib.breeze_tts_2_run_depth_history_dump(ctx, ptr(hidden), iptr(changed), 16,
                                                      controlled_ptrs, iptr(predicted)) == 0
        assert np.array_equal(controlled[:3], ordinary[:3])
        assert not np.array_equal(controlled[3], ordinary[3])
        np.save(dest / 'ordinary_depth_logits.npy', ordinary)
        np.save(dest / 'changed_history_logits.npy', controlled)
        depth('oracle_hidden', hidden)
        ids, mask = load('prompt_input_ids'), load('prompt_text_mask')
        assembled = np.empty_like(embeds)
        assert lib.breeze_tts_2_run_prefill_embeds_dump(ctx, iptr(ids), iptr(mask), len(ids),
                                                      None, 0, ptr(assembled)) == len(ids)
        np.save(dest / 'assembled_embeds.npy', assembled)
        for label, e in [('oracle_embeds', embeds), ('native_embeds', assembled)]:
            h, lg = np.empty(2048, np.float32), np.empty(2052, np.float32)
            assert lib.breeze_tts_2_run_backbone_dump(ctx, ptr(e), len(e), ptr(h), ptr(lg), None, 0) == len(e)
            np.save(dest / (label + '_hidden.npy'), h)
            np.save(dest / (label + '_backbone_logits.npy'), lg)
            depth(label, h)
    finally:
        lib.breeze_tts_2_free(ctx)
    (dest / 'provenance.json').write_text(json.dumps(dict(library_sha256=sha(libpath),
        model_sha256=sha(model), device='CPU; GPU backends OFF', invalid_history_controls_passed=2,
        ordinary_history_exact_control_passed=True, changed_history_causal_control_passed=True), indent=2)+'\n')


def summarize():
    import numpy as np
    history = np.load(OUT / 'reference/history_codes.npy')
    rows = []
    for cb in range(1, 16):
        original = np.load(OUT / f'reference/bf16_cached_cb{cb}.npy')
        assert original.shape == (2051,) and np.isfinite(original).all() and np.linalg.norm(original) > 0
        for side, label in [('reference', 'bf16_cached'), ('reference', 'bf16_full'),
                            ('reference', 'promoted_depth_f32_full'),
                            ('native', 'oracle_hidden'), ('native', 'oracle_embeds'), ('native', 'native_embeds')]:
            a = np.load(OUT / side / f'{label}_cb{cb}.npy')
            assert np.isfinite(a).all() and a.shape == (2051,)
            scores = a[:2048]
            pick = int(scores.argmax())
            target = int(history[cb])
            rows.append(dict(codebook=cb, variant=label, original_generated_pick=target,
                pick=pick, original_pick_rank=int(np.count_nonzero(scores > scores[target])),
                gap_to_best=float(scores[pick]-scores[target]),
                cos_to_bf16_cached=float(np.dot(a, original)/np.linalg.norm(a)/np.linalg.norm(original)),
                norm_ratio=float(np.linalg.norm(a)/np.linalg.norm(original))))
    (OUT / 'decisions.json').write_text(json.dumps(rows, indent=2)+'\n')
    (OUT / 'summary.md').write_text('Design-s1 frame-zero depth decision probe completed. '
        'All variants use the original code history. See decisions.json for ranks/gaps and raw arrays; '
        'this is diagnostic evidence, not full generation or speech acceptance.\n')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--phase', choices=['reference', 'native'])
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    SCRATCH.mkdir(parents=True, exist_ok=True)
    if args.phase:
        return globals()[args.phase]()
    receipt = dict(source=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        ggml=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT/'ggml', text=True).strip(),
        complete=False, full_speech_asr_tested=False)
    try:
        for phase in ('reference', 'native'):
            run([sys.executable, '-u', __file__, '--phase', phase], phase)
        summarize()
        receipt['arrays'] = {}
        import numpy as np
        for path in sorted(OUT.rglob('*.npy')):
            a = np.load(path, allow_pickle=False)
            assert a.size and np.isfinite(a).all()
            receipt['arrays'][str(path.relative_to(OUT))] = dict(sha256=sha(path), shape=list(a.shape), dtype=str(a.dtype))
        receipt['complete'] = True
    except BaseException as exc:
        receipt['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        (OUT/'validation.json').write_text(json.dumps(receipt, indent=2)+'\n')


if __name__ == '__main__':
    main()
