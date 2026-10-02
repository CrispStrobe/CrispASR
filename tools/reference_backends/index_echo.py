"""Independent Index-Echo reference using its released inference class.

Pin the source package; do not run a GGUF reconstruction as the oracle.
Capture full audio stages, last prompt-token decoder states, logits and text.
"""
import importlib.util
import functools
import json
import os
import tempfile
from pathlib import Path

import numpy as np

DEFAULT_STAGES = ['mel_spectrogram', 'encoder_input', 'encoder_output', 'connector_output',
                  'prompt_ids', 'llm_logits', 'generated_ids']


def load_blueprint(root):
    """Execute the released constructor; optional Accelerate changes placement only.

    The 9B F32 decoder cannot fit on a standard CPU runner or one T4. Let the
    official HF loader dispatch layers across devices/disk, and bypass only the
    constructor's subsequent homogeneous .to(device). Forward, cache, prompt
    and generation remain the original Python implementations.
    """
    import torch
    from transformers import AutoModelForCausalLM
    from unittest.mock import patch
    root = Path(root)
    spec = importlib.util.spec_from_file_location('index_echo_blueprint', root / 'infer.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    device = os.getenv('INDEX_ECHO_REF_DEVICE', 'cpu')
    dtype = getattr(torch, os.getenv('INDEX_ECHO_REF_DTYPE', 'float32'))
    placement = os.getenv('INDEX_ECHO_REF_DEVICE_MAP')
    original_to = []
    if device.startswith('cuda'):
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    if placement:
        factory = AutoModelForCausalLM.from_pretrained
        memory = json.loads(os.environ['INDEX_ECHO_REF_MAX_MEMORY'])
        memory = {int(k) if k.isdigit() else k: v for k, v in memory.items()}

        def placed_loader(*args, **kwargs):
            # Cached GDN reads conv1d.weight directly instead of calling the
            # child module. Accelerate must preload that parent's children;
            # otherwise the cached functional convolution sees meta weights.
            from transformers.integrations import accelerate as hf_accelerate
            dispatch = hf_accelerate.dispatch_model

            @functools.wraps(dispatch)
            def preload_dispatch(*a, **kw):
                kw['preload_module_classes'] = ['Qwen3_5GatedDeltaNet']
                return dispatch(*a, **kw)

            with patch.object(hf_accelerate, 'dispatch_model', new=preload_dispatch):
                llm = factory(*args, **kwargs,
                              device_map=json.loads(placement) if placement.startswith('{') else placement,
                              max_memory=memory, offload_folder=os.environ['INDEX_ECHO_REF_OFFLOAD_DIR'],
                              offload_buffers=True)
            llm._crispasr_ref_preload_classes = ['Qwen3_5GatedDeltaNet']
            for layer in llm.modules():
                if type(layer).__name__ != 'Qwen3_5GatedDeltaNet':
                    continue
                original_update = layer.causal_conv1d_update

                @functools.wraps(original_update)
                def checked_update(*a, _original=original_update, **kw):
                    if a[2].is_meta:
                        raise RuntimeError('Offloaded cached GDN convolution weight was not materialized')
                    return _original(*a, **kw)

                layer.causal_conv1d_update = checked_update
            original_to.append(llm.to)

            def constructor_to(destination):
                if str(destination) != device:
                    raise RuntimeError('Only the released constructor device move may be bypassed')
                return llm

            llm.to = constructor_to
            return llm

        with patch.object(AutoModelForCausalLM, 'from_pretrained', new=staticmethod(placed_loader)):
            model = module.AudioTransModel(str(root), device=device, dtype=dtype)
        model.llm.to = original_to[0]
    else:
        model = module.AudioTransModel(str(root), device=device, dtype=dtype)
    if os.getenv('INDEX_ECHO_REF_FP32_DECODER') == '1':
        if placement:
            if any(p.dtype != torch.float32 for p in model.llm.parameters()):
                raise RuntimeError('An offloaded F32 diagnostic must load F32 initially')
        else:
            model.llm.to(torch.float32)
    print('reference offload preload:', getattr(model.llm, '_crispasr_ref_preload_classes', []), flush=True)
    print('reference placement:', getattr(model.llm, 'hf_device_map', device), flush=True)
    print('reference parameter dtypes:', json.dumps(precision_audit(model), sort_keys=True), flush=True)
    return module, model


def precision_audit(model):
    """Report effective parameter dtypes, including nested text configuration."""
    result = {}
    for name in ['tower', 'connector', 'llm']:
        module = getattr(model, name)
        counts = {}
        for parameter in module.parameters():
            dtype = str(parameter.dtype)
            counts[dtype] = counts.get(dtype, 0) + parameter.numel()
        result[name] = dict(parameter_elements=counts, module_class=type(module).__name__)
    result['embedding_dtype'] = str(model.llm.get_input_embeddings().weight.dtype)
    return result


def dump(model_dir, audio, stages, **kwargs):
    if os.getenv('INDEX_ECHO_REF_CAPTURE_GENERATION') == '1':
        return dump_generation(model_dir, audio, stages, **kwargs)
    import soundfile as sf
    import torch
    torch.set_num_threads(int(os.getenv('INDEX_ECHO_REF_THREADS', '4')))
    torch.set_grad_enabled(False)
    root = Path(model_dir)
    module, model = load_blueprint(root)
    values = {'parameter_dtypes': json.dumps(precision_audit(model), sort_keys=True)}
    values['reference_preload_classes'] = json.dumps(getattr(model.llm, '_crispasr_ref_preload_classes', []))
    values['reference_placement'] = json.dumps(getattr(model.llm, 'hf_device_map', model.device), sort_keys=True)
    handles = []
    def hook(name, last=False, transform=None):
        def capture(mod, inputs, output):
            v = output[0] if isinstance(output, tuple) else output
            if last:
                v = v[:, -1, :]
            if transform:
                v = transform(v)
            values[name] = v.detach().float().cpu().numpy().copy()
        return capture
    for i, layer in enumerate(model.tower.layers):
        handles.append(layer.register_forward_hook(hook(f'encoder_layer_{i}')))
    def first_input(mod, inputs):
        values['encoder_input'] = inputs[0].detach().float().cpu().numpy().copy()
    handles.append(model.tower.layers[0].register_forward_pre_hook(first_input))
    handles.append(model.tower.ln_post.register_forward_hook(hook('ln_post_out')))
    handles.append(model.tower.proj1.register_forward_hook(hook('proj1_out')))
    for i in range(1, 4):
        handles.append(getattr(model.tower, f'conv2d{i}').register_forward_hook(
            hook(f'conv{i}_out', transform=torch.nn.functional.gelu)))
    handles.append(model.tower.proj2.register_forward_hook(hook('encoder_output')))
    with tempfile.TemporaryDirectory(dir=os.getenv('TMPDIR')) as tmp:
        wav = Path(tmp) / 'input.wav'
        sf.write(wav, audio, 16000, subtype='FLOAT')
        # The exact released frontend with its attention-mask frame count.
        f = model.fe(audio, sampling_rate=16000, return_tensors='pt', return_attention_mask=True)
        frames = int(f.attention_mask.sum(-1)[0])
        values['mel_spectrogram'] = f.input_features[0][:, :frames].float().numpy().copy()
        emb = model.encode_audio(str(wav))
    values['connector_output'] = emb.float().cpu().numpy().copy()
    for h in handles:
        h.remove()
    handles = []
    lang = os.getenv('INDEX_ECHO_TARGET_LANG', 'en')
    content = module.AUDIO_START + module.AUDIO_PAD * emb.shape[0] + module.AUDIO_END + '\n' + module.INSTR[lang]
    prompt = f'<|im_start|>user\n{content}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'
    ids = model.tok(prompt, return_tensors='pt').input_ids.to(model.device)
    x = model.llm.get_input_embeddings()(ids).clone()
    mask = ids == model.pad_id
    assert int(mask.sum()) == emb.shape[0]
    x[mask] = emb.to(x.dtype)
    values['target_lang'] = lang
    values['prompt_ids'] = ids.cpu().numpy().astype(np.int32)
    for i, layer in enumerate(model.llm.model.layers):
        handles.append(layer.register_forward_hook(hook(f'llm_block_{i}', last=True)))
    out = model.llm(inputs_embeds=x, attention_mask=torch.ones_like(ids), use_cache=True, logits_to_keep=1)
    values['llm_logits'] = out.logits[0, -1].float().cpu().numpy().copy()
    for h in handles:
        h.remove()
    trace_cache = out.past_key_values
    del out
    generated = model.llm.generate(inputs_embeds=x, attention_mask=torch.ones_like(ids),
        max_new_tokens=int(os.getenv('INDEX_ECHO_REF_MAX_TOKENS', '2000')), do_sample=False,
        eos_token_id=[model.tok.eos_token_id, model.im_end], pad_token_id=model.tok.eos_token_id)
    values['generated_ids'] = generated.cpu().numpy().astype(np.int32)
    values['generated_text'] = model.tok.decode(generated[0], skip_special_tokens=True).strip()
    # Replay the reference's own greedy IDs through the saved initial cache.
    # This separates recurrent/KV errors from divergent sampling decisions.
    trace = [values['llm_logits']]
    for i in range(min(int(os.getenv('INDEX_ECHO_TRACE_TOKENS', '16')), generated.shape[-1]) - 1):
        step = model.llm(input_ids=generated[:, i:i + 1], past_key_values=trace_cache,
                         use_cache=True, logits_to_keep=1)
        trace_cache = step.past_key_values
        trace.append(step.logits[0, -1].float().cpu().numpy().copy())
    values['teacherforced_logits'] = np.stack(trace)
    return values


def dump_pipeline(model_dir, output_dir, sample_dir, checkpoint=None):
    """Run the released file entry point, including real Silero and context.

    Keep raw rows and probabilities so native windowing and classifier drift
    can be distinguished from decoder drift. All expectations come from the
    released functions, not from the native parser or window helper.
    """
    import json
    import time
    import soundfile as sf
    import torch
    import silero_vad
    torch.set_num_threads(int(os.getenv('INDEX_ECHO_REF_THREADS', '4')))
    torch.set_grad_enabled(False)
    output_dir, sample_dir = Path(output_dir), Path(sample_dir)
    started = time.perf_counter()
    module, model = load_blueprint(model_dir)
    load_seconds = time.perf_counter() - started
    jfk, rate = sf.read(sample_dir / 'jfk.wav', dtype='float32')
    assert rate == 16000
    multi = output_dir / 'pipeline-multi.wav'
    sf.write(multi, np.concatenate([jfk, np.zeros(61 * rate, dtype=np.float32), jfk]), rate, subtype='PCM_16')
    cases = [('jfk-en', sample_dir / 'jfk.wav', 'en'),
             ('zh-en', sample_dir / 'paraformer_zh.wav', 'en'),
             ('zh-ja', sample_dir / 'paraformer_zh.wav', 'ja'),
             ('zh-es', sample_dir / 'paraformer_zh.wav', 'es'),
             ('multi-en', multi, 'en')]
    probabilities = []
    original_loader = silero_vad.load_silero_vad

    class RecordingVAD:
        def __init__(self, wrapped):
            self.wrapped = wrapped

        def __getattr__(self, name):
            return getattr(self.wrapped, name)

        def __call__(self, *args, **kwargs):
            value = self.wrapped(*args, **kwargs)
            probabilities.append(value.item())
            return value

    silero_vad.load_silero_vad = lambda *a, **kw: RecordingVAD(original_loader(*a, **kw))
    result = dict(precision=f'requested {model.device} {model.dtype}', parameter_dtypes=precision_audit(model),
                  reference_placement=getattr(model.llm, 'hf_device_map', model.device),
                  model_load_seconds=load_seconds, complete=False, cases={})
    try:
        for name, audio, lang in cases:
            probabilities.clear()
            started = time.perf_counter()
            rows = list(module.stream_translate(model, str(audio), target_lang=lang))
            elapsed = time.perf_counter() - started
            cues = [dict(start=c['g_st'], end=c['g_et'], text=c['zh'] + '\n' + (c['en'] or ''))
                    for row in rows if '__summary__' not in row for c in row['cues']]
            result['cases'][name] = dict(audio=audio.name, target=lang, rows=rows,
                                        segments=cues, vad_probabilities=list(probabilities), elapsed_seconds=elapsed)
            (output_dir / 'pipeline.json').write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
            if checkpoint:
                checkpoint(output_dir / 'pipeline.json', multi)
            print('released full pipeline', name, elapsed, json.dumps(cues, ensure_ascii=False), flush=True)
    finally:
        silero_vad.load_silero_vad = original_loader
    assert len(result['cases']['multi-en']['rows']) == 3, 'Fixture must exercise two windows plus summary'
    assert result['cases']['multi-en']['rows'][1]['has_ctx'], 'Second window must exercise prior-output context'
    result['complete'] = True
    (output_dir / 'pipeline.json').write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
    if checkpoint:
        checkpoint(output_dir / 'pipeline.json', multi)
    return output_dir / 'pipeline.json', multi


def dump_generation(model_dir, audio, stages, **kwargs):
    """Capture the released generation call itself, without a preceding prefill.

    Hooks observe the original translate_window frontend, prompt, generation
    and cache. The trace is the original generator's raw per-step logits; no
    cache object survives another generation call or is replayed afterwards.
    Keep the historical 2B dump recipe independent and opt into this explicitly.
    """
    import soundfile as sf
    import torch
    torch.set_num_threads(int(os.getenv('INDEX_ECHO_REF_THREADS', '4')))
    torch.set_grad_enabled(False)
    module, model = load_blueprint(model_dir)
    values = dict(parameter_dtypes=json.dumps(precision_audit(model), sort_keys=True),
                  reference_placement=json.dumps(getattr(model.llm, 'hf_device_map', model.device), sort_keys=True),
                  reference_preload_classes=json.dumps(getattr(model.llm, '_crispasr_ref_preload_classes', [])),
                  cache_trace_recipe='actual released translate_window generation forwards; no diagnostic prefill')
    handles, trace = [], []

    def capture(name, last=False, transform=None):
        def hook(_, inputs, output):
            if name in values:
                return
            v = output[0] if isinstance(output, tuple) else output
            if last:
                v = v[:, -1, :]
            if transform:
                v = transform(v)
            values[name] = v.detach().float().cpu().numpy().copy()
        return hook

    for i, layer in enumerate(model.tower.layers):
        handles.append(layer.register_forward_hook(capture(f'encoder_layer_{i}')))

    def first_input(_, inputs):
        values['encoder_input'] = inputs[0].detach().float().cpu().numpy().copy()

    handles.append(model.tower.layers[0].register_forward_pre_hook(first_input))
    for name, target in [('ln_post_out', model.tower.ln_post), ('proj1_out', model.tower.proj1),
                         ('encoder_output', model.tower.proj2), ('connector_output', model.connector)]:
        handles.append(target.register_forward_hook(capture(name)))
    for i in range(1, 4):
        handles.append(getattr(model.tower, f'conv2d{i}').register_forward_hook(
            capture(f'conv{i}_out', transform=torch.nn.functional.gelu)))
    for i, layer in enumerate(model.llm.model.layers):
        handles.append(layer.register_forward_hook(capture(f'llm_block_{i}', last=True)))

    def generation_forward(_, inputs, output):
        logits = output.logits[0, -1].detach().float().cpu().numpy().copy()
        if not trace:
            values['llm_logits'] = logits
        if len(trace) < int(os.getenv('INDEX_ECHO_TRACE_TOKENS', '16')):
            trace.append(logits)

    handles.append(model.llm.register_forward_hook(generation_forward))
    original_fe, original_tok, original_generate = model.fe, model.tok, model.llm.generate

    def frontend(*args, **kw):
        f = original_fe(*args, **kw)
        frames = int(f.attention_mask.sum(-1)[0])
        values['mel_spectrogram'] = f.input_features[0][:, :frames].float().cpu().numpy().copy()
        return f

    class TokenizerCapture:
        def __getattr__(self, name):
            return getattr(original_tok, name)

        def __call__(self, *args, **kw):
            ids = original_tok(*args, **kw)
            values['prompt_ids'] = ids.input_ids.cpu().numpy().astype(np.int32)
            return ids

    def generate(*args, **kw):
        out = original_generate(*args, **kw)
        values['generated_ids'] = out.detach().cpu().numpy().astype(np.int32)
        return out

    model.fe, model.tok, model.llm.generate = frontend, TokenizerCapture(), generate
    try:
        with tempfile.TemporaryDirectory(dir=os.getenv('TMPDIR')) as tmp:
            wav = Path(tmp) / 'input.wav'
            sf.write(wav, audio, 16000, subtype='FLOAT')
            lang = os.getenv('INDEX_ECHO_TARGET_LANG', 'en')
            text, context = model.translate_window(str(wav), [], lang=lang,
                max_new_tokens=int(os.getenv('INDEX_ECHO_REF_MAX_TOKENS', '2000')))
        assert not context
        values['generated_text'], values['target_lang'] = text, lang
        values['teacherforced_logits'] = np.stack(trace)
        ids = values['generated_ids'][0]
        values['raw_greedy_alignment'] = str(np.array_equal(np.argmax(values['teacherforced_logits'], axis=-1), ids[:len(trace)]))
    finally:
        model.fe, model.tok, model.llm.generate = original_fe, original_tok, original_generate
        for handle in handles:
            handle.remove()
    return values
