#!/usr/bin/env python3
"""CrispASR #445 — Confucius4-R2T2 STREAMING reference, per chunk.

Runs the upstream driver example.run_streaming (Confucius4-R2T2 @ pinned
commit) unmodified, with its defaults (160 ms step, 160 ms first-chunk
lookahead, unfixed_token_num=1, unfixed_chunk_num=0, adaptive max_new_tokens).

Upstream refuses anything but vLLM, but its only use of vLLM is one call:
llm.generate([{"prompt", "multi_modal_data": {"audio": [...]}}], SamplingParams)
-> text with special tokens skipped. That call is served here by the same
model through transformers, exactly as qwen_asr's own transformers backend
builds its inputs (_infer_asr_transformers), greedy, float32 on CPU. A stub
`vllm` module supplies SamplingParams. Nothing else is patched.

Every model call is logged (prompt suffix after the chat-template base,
max_tokens, audio length, generated text), and so is every
streaming_transcribe return, so the C++ port can be replayed call by call.
"""
import json
import os
import subprocess
import sys
import time
import traceback
import types
from pathlib import Path

os.environ["PYTHONUNBUFFERED"] = "1"
WORK = Path("/kaggle/working")
OUT = WORK / "out"
OUT.mkdir(parents=True, exist_ok=True)
MODEL = "netease-youdao/Confucius4-R2T2"
MODEL_REV = "185ce639118ad1362d049ca0d8ed04b6ec5cd6c9"
R2T2_REPO = "https://github.com/netease-youdao/Confucius4-R2T2.git"
R2T2_COMMIT = "c461192"
CRISPASR = WORK / "CrispASR"

results = {"runs": {}, "errors": []}


def save():
    (OUT / "stream_results.json").write_text(json.dumps(results, indent=1, ensure_ascii=False))


def main():
    t0 = time.time()

    def step(n):
        print(f"[step] {time.time() - t0:7.1f}s {n}", flush=True)

    step("clone")
    subprocess.check_call(["git", "clone", "--depth", "1", "https://github.com/CrispStrobe/CrispASR.git", str(CRISPASR)])
    subprocess.check_call(["git", "clone", R2T2_REPO, str(WORK / "r2t2")])
    subprocess.check_call(["git", "checkout", "-q", R2T2_COMMIT], cwd=str(WORK / "r2t2"))
    results["r2t2_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(WORK / "r2t2"),
                                                     text=True).strip()
    sys.path.insert(0, str(CRISPASR / "tools" / "kaggle"))
    import kaggle_harness as kh
    kh.resolve_hf_token()
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

    step("pip")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "qwen-asr", "librosa"])

    # Stub vllm: upstream imports SamplingParams lazily inside the streaming calls.
    vllm = types.ModuleType("vllm")

    class SamplingParams:
        def __init__(self, **kw):
            self.kw = kw
            self.max_tokens = kw.get("max_tokens")

    vllm.SamplingParams = SamplingParams
    sys.modules["vllm"] = vllm

    step("load")
    from huggingface_hub import snapshot_download
    model_dir = snapshot_download(MODEL, revision=MODEL_REV, local_dir="/tmp/r2t2m")
    import torch
    from qwen_asr import Qwen3ASRModel
    base = Qwen3ASRModel.from_pretrained(model_dir, dtype=torch.float32, device_map="cpu")
    processor, tmodel = base.processor, base.model

    calls = []

    class FakeLLM:
        def __init__(self):
            self.prompt_base = None

        def generate(self, batch, sampling_params=None, use_tqdm=False):
            outs = []
            for inp in batch:
                prompt = inp["prompt"]
                audio = inp["multi_modal_data"]["audio"][0]
                inputs = processor(text=[prompt], audio=[audio], return_tensors="pt", padding=True)
                inputs = inputs.to(tmodel.device).to(tmodel.dtype)
                mx = sampling_params.max_tokens if sampling_params is not None else 512
                with torch.no_grad():
                    ids = tmodel.generate(**inputs, max_new_tokens=mx, do_sample=False)
                seq = ids.sequences if hasattr(ids, "sequences") else ids
                new = seq[:, inputs["input_ids"].shape[1]:]
                text = processor.batch_decode(new, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
                suffix = prompt[len(self.prompt_base):] if self.prompt_base and prompt.startswith(self.prompt_base) \
                    else prompt
                calls.append({"n_audio": int(len(audio)), "max_tokens": mx, "prefix": suffix,
                              "gen_ids": [int(x) for x in new[0].tolist()], "gen_text": text})
                outs.append(types.SimpleNamespace(outputs=[types.SimpleNamespace(text=text)]))
            return outs

    sys.path.insert(0, str(WORK / "r2t2"))
    from r2t2 import R2T2ASRModel
    import example as ex

    llm = FakeLLM()
    asr = R2T2ASRModel(backend="vllm", model=llm, processor=processor,
                       sampling_params=SamplingParams(temperature=0.0, max_tokens=4, skip_special_tokens=True),
                       max_new_tokens=4)
    llm.prompt_base = asr._build_text_prompt(context="", force_language=None)
    results["prompt_base"] = llm.prompt_base

    orig_st = R2T2ASRModel.streaming_transcribe

    def logged_st(self, pcm16k, state, max_new_tokens=None, rollback_punctuation=False):
        n0 = len(calls)
        text, fixed = orig_st(self, pcm16k, state, max_new_tokens, rollback_punctuation)
        for c in calls[n0:]:
            c["after_text"] = state.text
            c["after_fixed"] = fixed
            c["after_raw"] = state._raw_decoded
            c["after_chunk_id"] = state.chunk_id
            c["kind"] = "step"
        return text, fixed

    R2T2ASRModel.streaming_transcribe = logged_st

    import librosa
    clips = {"en_jfk": CRISPASR / "samples" / "jfk.wav", "zh": CRISPASR / "samples" / "paraformer_zh.wav"}
    for name, path in clips.items():
        step(f"stream {name}")
        calls.clear()
        wav, sr = librosa.load(str(path), sr=16000, mono=True)
        tstart = time.time()
        final = ex.run_streaming(asr, wav, step_ms=160, chunk_size_sec=0.16, unfixed_token_num=1,
                                 lookahead_ms=160, language=None, context="")
        for c in calls:
            c.setdefault("kind", "finish")
        results["runs"][name] = {"final_text": final, "n_samples": int(len(wav)), "calls": list(calls),
                                 "seconds": round(time.time() - tstart, 1)}
        print(f"  {name}: {final}", flush=True)
        save()
    step("done")


if __name__ == "__main__":
    try:
        main()
    except BaseException as e:
        results["errors"].append(f"{type(e).__name__}: {e}\n{traceback.format_exc()}")
        save()
        raise
    finally:
        save()
        # Keep /kaggle/working to the results (see r2t2-ref: clones in the
        # output make `kaggle kernels output` hit 429s).
        import shutil
        shutil.rmtree(CRISPASR, ignore_errors=True)
        shutil.rmtree(WORK / "r2t2", ignore_errors=True)
