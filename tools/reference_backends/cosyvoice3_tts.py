"""Fun-CosyVoice3-0.5B-2512 LLM reference dump (Phase 2 diff harness).

The CosyVoice3 LLM is a vanilla Qwen2-0.5B body (same config as
`Qwen/Qwen2-0.5B-Instruct`, just retrained / fine-tuned) plus two extra
modules:

    speech_embedding : nn.Embedding(6761, 896)   # speech-token input
    llm_decoder      : nn.Linear(896, 6761)      # speech-token AR head

This script loads `llm.pt` directly into HuggingFace's `Qwen2Model`
class, attaches the two speech-side modules, runs a deterministic
forward, and dumps stage activations as a `.npz` for the diff harness
to compare against our C++ runtime.

Stages dumped:
    text_input_ids       : [T] int32 — the test prompt
    input_embeds         : [T, 896] f32 — token_embd[ids] (input to LM)
    layer_0_out          : [T, 896] f32 — after first Qwen2 block
    layer_23_out         : [T, 896] f32 — after last Qwen2 block
    output_norm_out      : [T, 896] f32 — after final RMSNorm
    step0_logits         : [6761] f32 — speech_lm_head(last_hidden)
    greedy_speech_tokens : [N] int32 — N greedy AR steps from step0
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict

import numpy as np

DEFAULT_PROMPT = "Hello, this is a test."


def _load_qwen2(model_dir: Path):
    """Load CosyVoice3 LLM into a HF Qwen2Model + speech-side modules."""
    import torch
    from transformers import Qwen2Config, Qwen2Model

    cfg_path = model_dir / "CosyVoice-BlankEN" / "config.json"
    cfg = Qwen2Config.from_pretrained(cfg_path.parent)

    state_path = model_dir / "llm.pt"
    sd_raw = torch.load(str(state_path), map_location="cpu", weights_only=False)
    if isinstance(sd_raw, dict) and "state_dict" in sd_raw:
        sd_raw = sd_raw["state_dict"]

    # The .pt prefixes everything with `llm.model.` (a CosyVoice wrapper
    # around the Qwen2Model). Strip it for HF, except for the two speech
    # modules which live at the top level.
    qwen_sd = {}
    extra = {}
    for k, v in sd_raw.items():
        if k == "speech_embedding.weight":
            extra["speech_embedding.weight"] = v
        elif k == "llm_decoder.weight":
            extra["llm_decoder.weight"] = v
        elif k.startswith("llm.model.model."):
            qwen_sd[k[len("llm.model.model."):]] = v
        elif k.startswith("llm.model."):
            # The .lm_head and trailing .norm at the wrapper level
            qwen_sd[k[len("llm.model."):]] = v

    cfg.use_cache = False
    cfg.attn_implementation = "eager"
    model = Qwen2Model(cfg)
    # Filter for keys Qwen2Model actually owns; lm_head lives in
    # Qwen2ForCausalLM, not Qwen2Model — we'd load the embedding+norm
    # only and drop the rest.
    own_keys = set(model.state_dict().keys())
    filtered = {k: v for k, v in qwen_sd.items() if k in own_keys}
    missing, unexpected = model.load_state_dict(filtered, strict=False)
    if missing:
        print(f"  missing qwen2 keys ({len(missing)}): {missing[:5]} ...")
    # `unexpected` from the strict=False call would catch any qwen_sd
    # entries Qwen2Model doesn't own (e.g. lm_head). Those are the
    # CosyVoice wrapper bits — fine to drop.
    model.eval()

    # Build the speech-side modules. The upstream class stores them
    # as bare nn.Module attributes outside Qwen2Model.
    speech_embd = torch.nn.Embedding(
        extra["speech_embedding.weight"].shape[0], extra["speech_embedding.weight"].shape[1])
    speech_embd.weight.data.copy_(extra["speech_embedding.weight"].float())
    speech_embd.eval()

    speech_lm_head = torch.nn.Linear(
        extra["llm_decoder.weight"].shape[1], extra["llm_decoder.weight"].shape[0], bias=False)
    speech_lm_head.weight.data.copy_(extra["llm_decoder.weight"].float())
    speech_lm_head.eval()

    return model, speech_embd, speech_lm_head, cfg


def dump_lm_reference(model_dir: Path, output_npz: Path, prompt: str, n_greedy: int = 32) -> None:
    import torch

    model, speech_embd, speech_lm_head, cfg = _load_qwen2(model_dir)
    print(f"  qwen2 loaded — d={cfg.hidden_size} L={cfg.num_hidden_layers} "
          f"vocab={cfg.vocab_size} kv={cfg.num_key_value_heads}")

    # Tokenize via the bundled gpt2-BPE tokenizer.
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_dir / "CosyVoice-BlankEN")
    ids_pt = tok(prompt, return_tensors="pt").input_ids[0]
    print(f"  prompt {prompt!r} -> {ids_pt.tolist()} ({ids_pt.shape[0]} tokens)")

    # Build inputs_embeds via token_embd (Qwen2Model's embed_tokens).
    with torch.no_grad():
        input_embeds = model.embed_tokens(ids_pt).float()  # [T, d_model]
        T = input_embeds.shape[0]

    # Forward with output_hidden_states so we can capture per-layer outputs.
    # The HF API accepts inputs_embeds=[B, T, d]; add the batch dim.
    out_data: Dict[str, np.ndarray] = {}
    out_data["text_input_ids"] = ids_pt.detach().cpu().numpy().astype(np.int32)
    out_data["input_embeds"] = input_embeds.detach().cpu().numpy().astype(np.float32)

    with torch.no_grad():
        out = model(
            inputs_embeds=input_embeds.unsqueeze(0).float(),
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )
    # output_hidden_states includes the input embedding at index 0 and
    # the output of each transformer block at indices 1..L. So
    # hidden_states[1] = after block 0, hidden_states[-1] = after final
    # block (before the final layer norm).
    hidden_states = out.hidden_states  # tuple of [1, T, d]
    out_data["layer_0_out"] = hidden_states[1][0].detach().cpu().numpy().astype(np.float32)
    out_data["layer_23_out"] = hidden_states[-1][0].detach().cpu().numpy().astype(np.float32)
    # out.last_hidden_state is the post-final-norm output of Qwen2Model.
    out_data["output_norm_out"] = out.last_hidden_state[0].detach().cpu().numpy().astype(np.float32)

    # speech_lm_head on the last position → step0 logits over speech vocab.
    last_hidden = out.last_hidden_state[0, -1, :]  # [d_model]
    with torch.no_grad():
        step0_logits = speech_lm_head(last_hidden.float())  # [6761]
    out_data["step0_logits"] = step0_logits.detach().cpu().numpy().astype(np.float32)
    print(f"  step0 top-5:")
    top = step0_logits.topk(5)
    for i in range(5):
        print(f"    {top.indices[i].item()}: {top.values[i].item():.4f}")

    # Greedy AR loop: pick top speech token (within the codebook range),
    # embed via speech_embedding, append to inputs_embeds, repeat. The
    # max sample range is the codebook (6561); the upper ~200 entries
    # of the head are special markers used by upstream wrappers. To
    # validate matching with our C++ side, we use the same restriction.
    SPEECH_CODEBOOK = 6561
    cur_ids: list[int] = []
    cur_embeds = input_embeds.clone()
    last_logits = step0_logits
    for step in range(n_greedy):
        # Sample within [0, SPEECH_CODEBOOK) — past the codebook are
        # special tokens.
        sub = last_logits[:SPEECH_CODEBOOK]
        nid = int(sub.argmax().item())
        cur_ids.append(nid)
        with torch.no_grad():
            next_e = speech_embd(torch.tensor([nid], dtype=torch.long)).float()  # [1, d]
        cur_embeds = torch.cat([cur_embeds, next_e], dim=0)
        with torch.no_grad():
            out = model(
                inputs_embeds=cur_embeds.unsqueeze(0).float(),
                output_hidden_states=False,
                use_cache=False,
                return_dict=True,
            )
            last_logits = speech_lm_head(out.last_hidden_state[0, -1, :].float())
    out_data["greedy_speech_tokens"] = np.asarray(cur_ids, dtype=np.int32)
    print(f"  greedy first 16: {cur_ids[:16]}")

    output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(str(output_npz), **out_data)
    sizes = {k: v.shape for k, v in out_data.items()}
    print(f"  wrote {output_npz}  ({len(out_data)} stages: {sizes})")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", required=True, help="Local snapshot dir for Fun-CosyVoice3-0.5B-2512")
    ap.add_argument("--output", required=True, help="Output .npz path")
    ap.add_argument("--prompt", default=DEFAULT_PROMPT, help="Text prompt to dump for")
    ap.add_argument("--n-greedy", type=int, default=32, help="Number of greedy AR steps to dump")
    args = ap.parse_args()
    dump_lm_reference(Path(args.model_dir), Path(args.output), args.prompt, args.n_greedy)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
