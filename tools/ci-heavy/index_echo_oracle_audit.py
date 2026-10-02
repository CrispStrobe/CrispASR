"""Instrumentation around the unmodified released translate_window entry point.

No extra decoder prefill and no retained pre-generation cache. This diagnostic
is not an acceptance oracle until valid source output and cache behavior pass.
"""
import json
import os
from pathlib import Path
import sys
import time


def audit(root, source, output, clips):
    import torch
    sys.path.insert(0, str(root / 'tools'))
    from reference_backends.index_echo import load_blueprint, precision_audit
    torch.set_num_threads(4)
    torch.set_grad_enabled(False)
    module, model = load_blueprint(source)
    result = dict(validated=False, torch=torch.__version__, transformers=__import__('transformers').__version__,
                  device=model.device, parameter_dtypes=precision_audit(model),
                  placement=getattr(model.llm, 'hf_device_map', model.device),
                  generation_config=model.llm.generation_config.to_dict(),
                  source_recipe='released translate_window; no diagnostic prefill or cache replay', cases={})
    for clip in clips:
        if clip == 'jfk-tail':
            raise ValueError('Fresh source diagnostic accepts repository audio only')
        audio = root / 'samples' / ('paraformer_zh.wav' if clip == 'zh' else 'jfk.wav')
        logits, generated = [], []

        def forward_hook(_, inputs, out):
            if len(logits) < 16:
                logits.append(out.logits[0, -1].detach().float().cpu().numpy().copy())

        original = model.llm.generate

        def generation_hook(*a, **kw):
            out = original(*a, **kw)
            generated.extend(out[0].detach().cpu().tolist())
            return out

        handle = model.llm.register_forward_hook(forward_hook)
        model.llm.generate = generation_hook
        started = time.perf_counter()
        try:
            text, context = model.translate_window(str(audio), [])
        finally:
            handle.remove()
            model.llm.generate = original
        cues, warnings = module.parse_hyp(text)
        top1 = [int(v.argmax()) for v in logits]
        result['cases'][clip] = dict(text=text, context=context, cues=cues, parse_warnings=warnings,
            valid_subtitles=bool(cues) and warnings == 0 and all(c['en'] for c in cues),
            first_generated_ids=generated[:16], raw_logits_argmax=top1,
            raw_greedy_alignment=top1 == generated[:len(top1)], generated_tokens=len(generated),
            elapsed_seconds=time.perf_counter()-started)
        output.write_text(json.dumps(result, indent=2, ensure_ascii=False)+'\n')
        print('fresh released source', clip, json.dumps(result['cases'][clip], ensure_ascii=False), flush=True)
        # A small immutable diagnostic is available before later clips finish.
        from huggingface_hub import HfApi
        stamp = os.getenv('GITHUB_RUN_ID') or time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())
        remote = 'audits/' + stamp + '-' + str(model.dtype).removeprefix('torch.') + '-' + output.name
        HfApi(token=os.environ['HF_TOKEN']).upload_file(path_or_fileobj=output, path_in_repo=remote,
                                                       repo_id='cstr/index-echo-9b-GGUF')
    return result
