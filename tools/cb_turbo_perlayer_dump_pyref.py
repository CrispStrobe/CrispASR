"""Per-layer hidden-state dump for the Python chatterbox-turbo reference, to
diff against the C++ GPT-2 graph (issue #94 follow-up).

Pairs with `CRISPASR_CHATTERBOX_DUMP_GPT2_LAYERS=1` on the C++ side:
both implementations write `(D, T)` float32 binary files under `/tmp`
named:

    /tmp/cb_gpt2_step_<n_past>_inputs_embeds.bin           (C++ only)
    /tmp/cb_gpt2_step_<n_past>_L<NN>_post_attn.bin
    /tmp/cb_gpt2_step_<n_past>_L<NN>_post_ffn.bin
    /tmp/py_gpt2_step_<n_past>_L<NN>_post_attn.bin
    /tmp/py_gpt2_step_<n_past>_L<NN>_post_ffn.bin

The companion `tools/cb_turbo_perlayer_diff.py` then walks both sets and
prints per-layer cosine similarity. Run order:

    # 1) C++ side — server with the dump env knob, hit /v1/audio/speech once.
    CRISPASR_GGUF_MMAP=1 CRISPASR_CHATTERBOX_DUMP_GPT2_LAYERS=1 \\
        CRISPASR_CHATTERBOX_T3_SEED=0 \\
        ./build-ninja-compile/bin/crispasr --server --backend chatterbox-turbo \\
        -m /Volumes/backups/ai/crispasr/chatterbox-turbo-t3-q8_0.gguf \\
        --codec-model /Volumes/backups/ai/crispasr/chatterbox-turbo-s3gen-q8_0.gguf \\
        --host 127.0.0.1 --port 51820 --voice-dir /tmp/emptyvoice &
    curl -s -X POST -H "Content-Type: application/json" \\
        -d '{"input":"hello world test"}' \\
        http://127.0.0.1:51820/v1/audio/speech -o /tmp/dump.wav

    # 2) Python side — forces step-0 sample to match C++'s seeded MT19937
    #    output (4024 for "hello world test" at seed 0).
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \\
        python tools/cb_turbo_perlayer_dump_pyref.py

    # 3) Diff
    python tools/cb_turbo_perlayer_diff.py

Override the forced step-0 token via `--force-tok0 <int>` and the prompt
via `--text <str>`.
"""

import argparse
import os
import random
import sys

import numpy as np

os.environ.setdefault("HF_HOME", "/Volumes/backups/ai/huggingface-hub")
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", "/Volumes/backups/ai/huggingface-hub")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch  # noqa: E402

torch.set_num_threads(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model-dir",
        default="/Volumes/backups/ai/huggingface-hub/models--ResembleAI--chatterbox-turbo",
        help="ResembleAI/chatterbox-turbo snapshot directory.",
    )
    ap.add_argument("--text", default="hello world test", help="Prompt text.")
    ap.add_argument("--force-tok0", type=int, default=4024,
                    help="Force the first sampled speech token to this id (match the C++ MT19937 pick).")
    ap.add_argument("--out-dir", default="/tmp", help="Where to write py_gpt2_step_*.bin files.")
    args = ap.parse_args()

    from chatterbox.tts_turbo import ChatterboxTurboTTS

    print("loading model…", flush=True)
    tts = ChatterboxTurboTTS.from_local(args.model_dir, device="cpu")
    print("loaded", flush=True)

    # Force step-0 sample so Python and C++ both start from the same token at the
    # first AR step. Subsequent samples use Python's natural RNG (won't match C++,
    # but we only diff the first AR step where inputs are guaranteed identical).
    orig_multinomial = torch.multinomial
    call_idx = [0]
    force_prefix = [args.force_tok0]

    def patched_multinomial(input, num_samples, replacement=False, generator=None, out=None):
        if call_idx[0] < len(force_prefix):
            forced = force_prefix[call_idx[0]]
            call_idx[0] += 1
            return torch.tensor([[forced]], dtype=torch.long)
        call_idx[0] += 1
        return orig_multinomial(input, num_samples, replacement=replacement, generator=generator)

    torch.multinomial = patched_multinomial

    # Replace GPT2Block.forward so we can intercept post-attn-residual and
    # post-ffn-residual hidden states at the T==1 AR steps.
    import transformers.models.gpt2.modeling_gpt2 as gpt2_mod

    gpt2 = tts.t3.tfmr
    ar_step = [-1]

    # Detect the prefill length on the fly. After prefill, past_kv has size = prefill_len.
    # We need this to name files as cb_gpt2_step_<n_past>_*.bin where n_past = prefill_len.
    prefill_len = [None]

    # HF GPT2Model.forward calls each block positionally with
    #     block(hidden_states, past_key_values, cache_position,
    #           attention_mask, head_mask, encoder_hidden_states, …)
    # The earlier `*_, **kwargs` version silently dropped past_key_values
    # and yielded AR-step hidden states that diverged from a real forward,
    # so per-layer cos numbers were meaningless. Promote the positionals
    # back to kwargs so self.attn receives the KV cache + causal mask
    # exactly as in production.
    _block_pos_keys = (
        "past_key_values",
        "cache_position",
        "attention_mask",
        "head_mask",
        "encoder_hidden_states",
    )

    def patched_block_forward(self, hidden_states, *args_, **kwargs):
        residual = hidden_states
        h = self.ln_1(hidden_states)
        attn_kwargs = {
            k: kwargs[k]
            for k in (
                "past_key_values",
                "cache_position",
                "attention_mask",
                "head_mask",
                "use_cache",
                "output_attentions",
            )
            if k in kwargs
        }
        for i, v in enumerate(args_):
            if i < len(_block_pos_keys) and _block_pos_keys[i] not in attn_kwargs:
                attn_kwargs[_block_pos_keys[i]] = v
        # gpt2 attn doesn't take encoder_hidden_states unless cross-attn is on.
        attn_kwargs.pop("encoder_hidden_states", None)
        attn_out = self.attn(h, **attn_kwargs)
        attn_hidden = attn_out[0]
        rest = attn_out[1:]
        post_attn = attn_hidden + residual

        residual = post_attn
        h = self.ln_2(post_attn)
        ffn_hidden = self.mlp(h)
        post_ffn = residual + ffn_hidden

        il = getattr(self.attn, "layer_idx", None)
        if il is None:
            il = next((i for i, b in enumerate(gpt2.h) if b is self), -1)
        if il >= 0 and ar_step[0] >= 0 and post_attn.size(1) == 1 and prefill_len[0] is not None:
            n_past = prefill_len[0] + ar_step[0]
            pa = post_attn.detach().squeeze(0).float().cpu().numpy().T.astype(np.float32)
            pf = post_ffn.detach().squeeze(0).float().cpu().numpy().T.astype(np.float32)
            with open(f"{args.out_dir}/py_gpt2_step_{n_past}_L{il:02d}_post_attn.bin", "wb") as f:
                pa.tofile(f)
            with open(f"{args.out_dir}/py_gpt2_step_{n_past}_L{il:02d}_post_ffn.bin", "wb") as f:
                pf.tofile(f)

        return (post_ffn,) + rest if rest else (post_ffn,)

    gpt2_mod.GPT2Block.forward = patched_block_forward

    orig_tfmr_forward = gpt2.forward

    def patched_tfmr_forward(*args_, **kwargs_):
        embeds = kwargs_.get("inputs_embeds")
        if embeds is None and args_:
            embeds = args_[0]
        if embeds is not None:
            if embeds.size(1) > 1:
                # Prefill — record length.
                prefill_len[0] = embeds.size(1)
                ar_step[0] = -1
            else:
                ar_step[0] += 1
        return orig_tfmr_forward(*args_, **kwargs_)

    gpt2.forward = patched_tfmr_forward

    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    print(f"generating '{args.text}' (force tok0={args.force_tok0})…", flush=True)
    _ = tts.generate(args.text)
    print(f"done — prefill_len={prefill_len[0]} ar_steps={ar_step[0] + 1}", flush=True)


if __name__ == "__main__":
    main()
