"""Fun-CosyVoice3-0.5B-2512 reference dump backend.

Two reference-dump entry points:

1. `dump_lm_reference(model_dir, output_npz, prompt, n_greedy)` —
   Phase 2 LLM stages (input_embeds, per-layer hidden states,
   step0_logits, greedy_speech_tokens) written to an .npz. Used by
   the legacy `tools/diff-cosyvoice3-lm.py` harness.

2. `dump(model_dir, audio, stages, **kwargs)` — dump_reference.py
   contract. Returns a dict of {stage_name: ndarray} for any subset of
   the registered stages. Wired into the unified `crispasr-diff` CLI
   via the `cosyvoice3-tts` backend (see crispasr_diff_main.cpp).

   Stages exposed for the Phase 3b DiT single-block diff (blocks 0 + 21):

       flow_dit_blk_<N>_x_in        seeded random input  [T, 1024]
       flow_dit_blk_<N>_t_emb       time_mlp(sin_emb(t=0.5))  [1024]
       flow_dit_blk_<N>_lnx_a       LN(x) pre-modulation
       flow_dit_blk_<N>_h_a         post (1+scale_msa)·LN(x) + shift_msa
       flow_dit_blk_<N>_attn        attention output (pre-residual)
       flow_dit_blk_<N>_xattn       x + gate_msa · attn
       flow_dit_blk_<N>_ff          FFN output (pre-residual)
       flow_dit_blk_<N>_out         final block output (post-FFN residual)

   The C++ extract_stage handler unpacks `embeds_in` as
   [x | t_emb] = (T*dit_dim + dit_dim) floats, runs the per-block
   ggml graph, and returns the requested intermediate. The diff CLI
   pulls x_in / t_emb from the GGUF archive, packs them, and compares
   each named output.

The "audio" arg from `tools/dump_reference.py` is unused (this backend
is text/synth-token-driven, not audio-conditioned for these stages).
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, Iterable, Optional, Set

import numpy as np

DEFAULT_PROMPT = "Hello, this is a test."

# Stages dump_reference.py will request by default. Each is a per-block
# stage (block 0 + block 21); the dump() function honors any subset.
DEFAULT_STAGES = [
    # Block 0
    "flow_dit_blk_0_x_in",
    "flow_dit_blk_0_t_emb",
    "flow_dit_blk_0_lnx_a",
    "flow_dit_blk_0_h_a",
    "flow_dit_blk_0_attn",
    "flow_dit_blk_0_xattn",
    "flow_dit_blk_0_ff",
    "flow_dit_blk_0_out",
    # Block 21 (last)
    "flow_dit_blk_21_x_in",
    "flow_dit_blk_21_t_emb",
    "flow_dit_blk_21_lnx_a",
    "flow_dit_blk_21_h_a",
    "flow_dit_blk_21_attn",
    "flow_dit_blk_21_xattn",
    "flow_dit_blk_21_ff",
    "flow_dit_blk_21_out",
    # Phase 3c — pre-lookahead conv stack (PreLookaheadLayer)
    "flow_pre_la_ids_in",       # speech token ids (T_tok,) int32
    "flow_pre_la_tok_emb",      # input_embedding(ids)         (T_tok, mel_dim)
    "flow_pre_la_c1",           # leaky_relu(conv1(right-pad)) (T_tok, 1024)
    "flow_pre_la_c2",           # conv2(left-pad)              (T_tok, mel_dim)
    "flow_pre_la",              # final + residual             (T_tok, mel_dim)
    # Phase 3c — InputEmbedding (input pipeline)
    "flow_in_pipe_pre_la_in",   # pre_la after repeat_interleave(token_mel_ratio)
    "flow_in_pipe_spk_in",      # raw speaker embedding         (192,)
    "flow_in_pipe_x_in",        # noised mel iterate            (T_mel, mel_dim)
    "flow_in_pipe_cond_in",     # cond prefix                   (T_mel, mel_dim)
    "flow_in_pipe_spk",         # normalize + spk_affine        (spk_dim_out,)
    "flow_in_pipe_cat",         # cat[x, cond, mu, spk]         (T_mel, 320)
    "flow_in_pipe_proj",        # in_proj                       (T_mel, 1024)
    "flow_in_pipe_pos",         # conv_pos_embed(proj)          (T_mel, 1024)
    "flow_in_pipe",             # proj + conv_pos_embed         (T_mel, 1024)
    # Phase 3d-A — full 22-block DiT estimator forward
    "flow_dit_full_x_in",       # seeded post-input-pipeline input    (T_mel, 1024)
    "flow_dit_full_t_emb",      # time_mlp(sin_emb(t=0.5))            (1024,)
    "flow_dit_full_norm",       # after AdaLayerNormZero_Final        (T_mel, 1024)
    "flow_dit_full",            # final mel output (post proj_out)    (T_mel, 80)
    # Phase 3d-B — CFM Euler ODE end-to-end (10-step cosine + CFG)
    "flow_euler_mu_in",         # post pre_la + repeat_interleave    (T_mel, 80)
    "flow_euler_spks_in",       # post normalize + spk_affine        (80,)
    "flow_euler_cond_in",       # prompt-prefix conditioning         (T_mel, 80)
    "flow_euler_x_init",        # seeded torch.manual_seed(0) randn  (T_mel, 80)
    "flow_euler_dphi_step0",    # post-CFG dphi_dt at step 1         (T_mel, 80)
    "flow_euler",               # final mel after 10 Euler steps     (T_mel, 80)
]

# Fixed test-vector parameters for the Phase 3b dumps. Pinned so the
# diff is reproducible across runs and machines.
DIT_T = 8
DIT_DIM = 1024
DIT_SEED = 1234
DIT_TIMESTEP = 0.5

# Phase 3d-A full-stack diff vector. Independent seed, T_mel small but
# >= 2 so RoPE rotates non-trivially.
DIT_FULL_T_MEL = 12
DIT_FULL_SEED = 30303

# Phase 3d-B Euler fixture. T_tok=4 keeps the (T_mel=8) dump fast; the
# spk-emb seed is separate so it deterministically advances the rng
# state past the mu fixture. The init noise comes from upstream's
# `set_all_random_seed(0); rand_noise = torch.randn([1, 80, 50*300])`
# truncated to T_mel — that's the SAME state upstream uses, so the
# diff is bit-equivalent on the noise input.
EULER_T_TOK = 4
EULER_SPK_SEED = 40404
EULER_N_STEPS = 10
EULER_CFG_RATE = 0.7

# Phase 3c test vector — independent seeded fixture. T_tok small enough
# to keep dumps fast; T_mel = 2 · T_tok per token_mel_ratio. Mel/spk dims
# are model-fixed.
PRE_LA_T_TOK = 6
PRE_LA_SEED = 5678
IN_PIPE_SEED = 9012
MEL_DIM = 80
SPK_DIM_IN = 192


# ---------------------------------------------------------------------------
# Phase 2 — LLM dump (unchanged).
# ---------------------------------------------------------------------------

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
            qwen_sd[k[len("llm.model."):]] = v

    cfg.use_cache = False
    cfg.attn_implementation = "eager"
    model = Qwen2Model(cfg)
    own_keys = set(model.state_dict().keys())
    filtered = {k: v for k, v in qwen_sd.items() if k in own_keys}
    missing, unexpected = model.load_state_dict(filtered, strict=False)
    if missing:
        print(f"  missing qwen2 keys ({len(missing)}): {missing[:5]} ...")
    model.eval()

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

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_dir / "CosyVoice-BlankEN")
    ids_pt = tok(prompt, return_tensors="pt").input_ids[0]
    print(f"  prompt {prompt!r} -> {ids_pt.tolist()} ({ids_pt.shape[0]} tokens)")

    with torch.no_grad():
        input_embeds = model.embed_tokens(ids_pt).float()
        T = input_embeds.shape[0]

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
    hidden_states = out.hidden_states
    out_data["layer_0_out"] = hidden_states[1][0].detach().cpu().numpy().astype(np.float32)
    out_data["layer_23_out"] = hidden_states[-1][0].detach().cpu().numpy().astype(np.float32)
    out_data["output_norm_out"] = out.last_hidden_state[0].detach().cpu().numpy().astype(np.float32)

    last_hidden = out.last_hidden_state[0, -1, :]
    with torch.no_grad():
        step0_logits = speech_lm_head(last_hidden.float())
    out_data["step0_logits"] = step0_logits.detach().cpu().numpy().astype(np.float32)
    print(f"  step0 top-5:")
    top = step0_logits.topk(5)
    for i in range(5):
        print(f"    {top.indices[i].item()}: {top.values[i].item():.4f}")

    SPEECH_CODEBOOK = 6561
    cur_ids: list[int] = []
    cur_embeds = input_embeds.clone()
    last_logits = step0_logits
    for step in range(n_greedy):
        sub = last_logits[:SPEECH_CODEBOOK]
        nid = int(sub.argmax().item())
        cur_ids.append(nid)
        with torch.no_grad():
            next_e = speech_embd(torch.tensor([nid], dtype=torch.long)).float()
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


# ---------------------------------------------------------------------------
# Phase 3b — single DiT block dumper (used by both the CLI and dump()).
# ---------------------------------------------------------------------------

def _ensure_upstream_on_path() -> None:
    """Make the upstream FunAudioLLM/CosyVoice clone importable."""
    import sys
    upstream = Path("/Volumes/backups/code/cosyvoice3-stash/CosyVoice-upstream")
    if not upstream.exists():
        raise FileNotFoundError(
            f"Upstream CosyVoice clone not found at {upstream}. "
            "git clone https://github.com/FunAudioLLM/CosyVoice.git into that path.")
    if str(upstream) not in sys.path:
        sys.path.insert(0, str(upstream))


def _capture_dit_block_stages(model_dir: Path, block_idx: int,
                              T: int = DIT_T, seed: int = DIT_SEED,
                              timestep: float = DIT_TIMESTEP):
    """Build one upstream DiTBlock + TimestepEmbedding, run a seeded
    forward, return a dict of {short_name: torch.Tensor} matching the
    C++ side's per-stage outputs."""
    _ensure_upstream_on_path()
    from cosyvoice.flow.DiT.modules import DiTBlock, TimestepEmbedding
    from x_transformers.x_transformers import RotaryEmbedding

    import torch
    state_path = model_dir / "flow.pt"
    sd_raw = torch.load(str(state_path), map_location="cpu", weights_only=False)
    if isinstance(sd_raw, dict) and "state_dict" in sd_raw:
        sd_raw = sd_raw["state_dict"]

    dim, heads, head_dim, ff_mult = DIT_DIM, 16, 64, 2

    block_prefix = f"decoder.estimator.transformer_blocks.{block_idx}."
    time_prefix = "decoder.estimator.time_embed."
    block_sd = {k[len(block_prefix):]: v for k, v in sd_raw.items() if k.startswith(block_prefix)}
    time_sd = {k[len(time_prefix):]: v for k, v in sd_raw.items() if k.startswith(time_prefix)}
    if not block_sd:
        raise RuntimeError(f"no tensors under prefix {block_prefix!r}")

    block = DiTBlock(dim=dim, heads=heads, dim_head=head_dim, ff_mult=ff_mult, dropout=0.0)
    block.load_state_dict(block_sd, strict=False)
    block.eval()

    time_embed = TimestepEmbedding(dim, freq_embed_dim=256)
    time_embed.load_state_dict(time_sd, strict=False)
    time_embed.eval()

    gen = torch.Generator().manual_seed(seed)
    x = torch.randn(1, T, dim, generator=gen, dtype=torch.float32)
    t_scalar = torch.tensor([timestep], dtype=torch.float32)
    with torch.no_grad():
        t_emb = time_embed(t_scalar)  # (1, dim)

    rotary = RotaryEmbedding(head_dim)
    with torch.no_grad():
        rope = rotary.forward_from_seq_len(T)

    # Non-streaming non-masked attention: full-True (1, 1, T, T) mask
    # — same as `add_optional_chunk_mask(..., 0, 0, -1).repeat(1, T, 1).unsqueeze(1)`.
    mask = torch.ones(1, T, T, dtype=torch.bool).unsqueeze(1)

    # Piecewise forward (mirrors DiTBlock.forward) so we can capture
    # the same intermediates the C++ side exposes through extract_stage.
    with torch.no_grad():
        adaln = block.attn_norm
        ln_x = adaln.norm(x)
        emb_a = adaln.linear(adaln.silu(t_emb))
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = torch.chunk(emb_a, 6, dim=1)
        h_a = ln_x * (1 + scale_msa[:, None]) + shift_msa[:, None]
        attn_raw = block.attn(x=h_a, mask=mask, rope=rope)
        x_after_attn = x + gate_msa[:, None] * attn_raw
        ff_norm = block.ff_norm(x_after_attn) * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
        ff_raw = block.ff(ff_norm)
        y = x_after_attn + gate_mlp[:, None] * ff_raw

        # Sanity: confirm the piecewise reconstruction matches the all-in-one
        # block forward — guards against me mis-replicating the upstream block.
        y_baseline = block(x, t_emb, mask=mask, rope=rope)
        drift = (y - y_baseline).abs().max().item()
        assert drift < 1e-4, f"piecewise reconstruction drift {drift}"

    return {
        "x_in": x.squeeze(0),
        "t_emb": t_emb.squeeze(0),
        "lnx_a": ln_x.squeeze(0),
        "h_a": h_a.squeeze(0),
        "attn": attn_raw.squeeze(0),
        "xattn": x_after_attn.squeeze(0),
        "ff": ff_raw.squeeze(0),
        "out": y.squeeze(0),
    }


def _flow_dit_block_arrays(model_dir: Path, block_idx: int,
                           stages_wanted: Optional[Set[str]] = None) -> Dict[str, np.ndarray]:
    """Capture the per-stage activations for one DiT block and return
    them as {full_stage_name: ndarray} ready for the GGUF archive."""
    stages = _capture_dit_block_stages(model_dir, block_idx)
    out: Dict[str, np.ndarray] = {}
    for short, tensor in stages.items():
        name = f"flow_dit_blk_{block_idx}_{short}"
        if stages_wanted is not None and name not in stages_wanted:
            continue
        out[name] = tensor.contiguous().detach().cpu().numpy().astype(np.float32)
    return out


def dump_flow_dit_block_bins(model_dir: Path, out_dir: Path,
                             block_idx: int = 0, T: int = DIT_T,
                             seed: int = DIT_SEED) -> None:
    """Legacy CLI mode — write raw float32 binaries under <out_dir>.
    Kept for ad-hoc debugging during port work; the unified
    crispasr-diff pipeline goes through dump() instead."""
    arrays = _flow_dit_block_arrays(model_dir, block_idx)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, arr in arrays.items():
        path = out_dir / f"{name}.bin"
        path.write_bytes(arr.tobytes())
        print(f"  wrote {path} ({arr.shape})")


# ---------------------------------------------------------------------------
# Phase 3c — pre-lookahead conv + InputEmbedding capture
# ---------------------------------------------------------------------------

def _load_flow_state(model_dir: Path):
    import torch
    state_path = model_dir / "flow.pt"
    sd = torch.load(str(state_path), map_location="cpu", weights_only=False)
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    return sd


def _capture_pre_la_stages(model_dir: Path, T_tok: int = PRE_LA_T_TOK,
                           seed: int = PRE_LA_SEED) -> Dict[str, "torch.Tensor"]:
    """Run the upstream PreLookaheadLayer on seeded random speech-token
    ids and return the named stage activations.

    Stages match the C++ extract_stage names (without the `flow_pre_la_`
    prefix added by the caller).
    """
    _ensure_upstream_on_path()
    import torch
    from cosyvoice.transformer.upsample_encoder import PreLookaheadLayer

    sd = _load_flow_state(model_dir)

    # Instantiate the matching modules (cv3 config: 80→1024 conv1, 1024→80 conv2).
    embed = torch.nn.Embedding(6561, MEL_DIM)
    embed.weight.data.copy_(sd["input_embedding.weight"].float())
    embed.eval()
    pre_la = PreLookaheadLayer(in_channels=MEL_DIM, channels=1024, pre_lookahead_len=3)
    # The flow state-dict prefixes everything under "pre_lookahead_layer."
    pre_la_sd = {k[len("pre_lookahead_layer."):]: v
                 for k, v in sd.items() if k.startswith("pre_lookahead_layer.")}
    pre_la.load_state_dict(pre_la_sd, strict=True)
    pre_la.eval()

    # Seeded random speech-token ids (uniform over [0, 6561))
    rng = torch.Generator().manual_seed(seed)
    ids = torch.randint(0, 6561, (1, T_tok), generator=rng, dtype=torch.long)

    with torch.no_grad():
        tok_emb = embed(ids)  # (1, T_tok, mel_dim)

    # Piecewise rerun of PreLookaheadLayer.forward so we can grab the
    # post-conv1 and post-conv2 intermediates without a forward hook.
    import torch.nn.functional as F
    with torch.no_grad():
        x = tok_emb.transpose(1, 2).contiguous()                      # (1, mel, T_tok)
        x = F.pad(x, (0, pre_la.pre_lookahead_len), value=0.0)        # right-pad 3
        x = F.leaky_relu(pre_la.conv1(x))                             # (1, 1024, T_tok)
        c1_t = x.clone()
        x = F.pad(x, (pre_la.conv2.kernel_size[0] - 1, 0), value=0.0) # left-pad 2
        x = pre_la.conv2(x)                                           # (1, mel, T_tok)
        c2_t = x.clone()
        x = x.transpose(1, 2).contiguous()                            # (1, T_tok, mel)
        y_piece = x + tok_emb
        # Sanity: piecewise matches the monolithic forward.
        y_baseline = pre_la(tok_emb)
        drift = (y_piece - y_baseline).abs().max().item()
        assert drift < 1e-5, f"pre_la piecewise drift {drift}"

    # PyTorch (T, C) row-major IS byte-identical to ggml ne=(C, T)
    # col-major, so we keep the (T, C) shape and let the byte layout
    # implicitly match the C++ side's channel-first ggml output.
    # c1_t / c2_t are stored channel-first by upstream (B, C, T); transpose
    # those to (T, C) for the same convention.
    def tc(t):
        if t.dim() == 3 and t.shape[2] in (MEL_DIM, 1024):
            return t.squeeze(0).contiguous()                       # (T, C)
        if t.dim() == 3:
            return t.squeeze(0).transpose(0, 1).contiguous()        # (B, C, T) -> (T, C)
        raise ValueError(f"unexpected tensor shape {tuple(t.shape)}")

    return {
        "ids_in": ids.squeeze(0).to(torch.int32),
        "tok_emb": tc(tok_emb),
        "c1": tc(c1_t),
        "c2": tc(c2_t),
        "": tc(y_piece),
    }


def _capture_in_pipe_stages(model_dir: Path, T_tok: int = PRE_LA_T_TOK,
                            seed: int = IN_PIPE_SEED) -> Dict[str, "torch.Tensor"]:
    """Run the upstream InputEmbedding on a seeded fixture and return the
    named stage activations. Uses the pre_la dumper's output as the `mu`
    input so we exercise the realistic upstream chain (pre_la -> interleave
    -> InputEmbedding) but each part is independently reproducible.
    """
    _ensure_upstream_on_path()
    import torch
    import torch.nn.functional as F
    from cosyvoice.flow.DiT.dit import InputEmbedding
    from cosyvoice.transformer.upsample_encoder import PreLookaheadLayer

    sd = _load_flow_state(model_dir)

    # Re-run pre_la to get the (T_tok, mel) hidden state.
    embed = torch.nn.Embedding(6561, MEL_DIM)
    embed.weight.data.copy_(sd["input_embedding.weight"].float())
    embed.eval()
    pre_la = PreLookaheadLayer(in_channels=MEL_DIM, channels=1024, pre_lookahead_len=3)
    pre_la_sd = {k[len("pre_lookahead_layer."):]: v
                 for k, v in sd.items() if k.startswith("pre_lookahead_layer.")}
    pre_la.load_state_dict(pre_la_sd, strict=True)
    pre_la.eval()

    rng = torch.Generator().manual_seed(seed)
    ids = torch.randint(0, 6561, (1, T_tok), generator=rng, dtype=torch.long)
    with torch.no_grad():
        h = pre_la(embed(ids))  # (1, T_tok, mel)
    # repeat_interleave by token_mel_ratio = 2 — gives (1, T_mel, mel).
    h_mel = h.repeat_interleave(2, dim=1)
    T_mel = h_mel.shape[1]

    # Seeded random spk_emb, x_noisy, cond. These use the same `rng` so
    # state advances deterministically.
    spk_raw = torch.randn(1, SPK_DIM_IN, generator=rng, dtype=torch.float32)
    x_noisy = torch.randn(1, T_mel, MEL_DIM, generator=rng, dtype=torch.float32)
    cond = torch.randn(1, T_mel, MEL_DIM, generator=rng, dtype=torch.float32)

    # Build the InputEmbedding module from the flow state-dict.
    # The state-dict prefixes InputEmbedding tensors under
    # "decoder.estimator.input_embed.*".
    in_emb = InputEmbedding(mel_dim=MEL_DIM, text_dim=MEL_DIM, out_dim=1024, spk_dim=MEL_DIM)
    in_emb_sd = {k[len("decoder.estimator.input_embed."):]: v
                 for k, v in sd.items() if k.startswith("decoder.estimator.input_embed.")}
    in_emb.load_state_dict(in_emb_sd, strict=True)
    in_emb.eval()

    # Speaker projection: F.normalize over dim=1 -> Linear(192, 80).
    # spk_embed_affine_layer lives at the flow-wrapper level, NOT inside
    # InputEmbedding. The state-dict has "spk_embed_affine_layer.weight/bias".
    spk_affine = torch.nn.Linear(SPK_DIM_IN, MEL_DIM)
    spk_affine.load_state_dict({
        "weight": sd["spk_embed_affine_layer.weight"],
        "bias": sd["spk_embed_affine_layer.bias"],
    }, strict=True)
    spk_affine.eval()

    with torch.no_grad():
        spk_norm = F.normalize(spk_raw, dim=1)
        spk_proj = spk_affine(spk_norm)  # (1, 80)
        # Piecewise reconstruct InputEmbedding.forward.
        spks_bc = spk_proj.unsqueeze(1).expand(1, T_mel, MEL_DIM)         # (1, T_mel, 80)
        catted = torch.cat([x_noisy, cond, h_mel, spks_bc], dim=-1)        # (1, T_mel, 320)
        proj = in_emb.proj(catted)                                         # (1, T_mel, 1024)
        pos = in_emb.conv_pos_embed(proj)                                  # (1, T_mel, 1024)
        out_piece = pos + proj
        # Sanity vs upstream's all-in-one forward.
        out_baseline = in_emb(x_noisy, cond, h_mel, spk_proj)
        drift = (out_piece - out_baseline).abs().max().item()
        assert drift < 1e-5, f"in_pipe piecewise drift {drift}"

    # Match the C++ side's ggml byte layout: keep PyTorch (T, C)
    # row-major shapes, since that's byte-identical to ggml ne=(C, T)
    # col-major. 1D tensors are squeezed of the batch dim only.
    def tc(t):
        if t.dim() == 3:
            return t.squeeze(0).contiguous()                       # (T, C)
        return t.contiguous()

    return {
        "pre_la_in": tc(h_mel),
        "spk_in": spk_raw.squeeze(0).contiguous(),
        "x_in": tc(x_noisy),
        "cond_in": tc(cond),
        "spk": spk_proj.squeeze(0).contiguous(),
        "cat": tc(catted),
        "proj": tc(proj),
        "pos": tc(pos),
        "": tc(out_piece),
    }


# ---------------------------------------------------------------------------
# Phase 3d-A — full 22-block DiT estimator capture
# ---------------------------------------------------------------------------

def _capture_dit_full_stages(model_dir: Path, T_mel: int = DIT_FULL_T_MEL,
                             seed: int = DIT_FULL_SEED,
                             timestep: float = DIT_TIMESTEP) -> Dict[str, "torch.Tensor"]:
    """Run upstream's 22-block transformer_blocks + norm_out + proj_out
    on a seeded post-input-pipeline fixture and return the named stage
    activations.
    """
    _ensure_upstream_on_path()
    import torch
    from cosyvoice.flow.DiT.modules import DiTBlock, TimestepEmbedding, AdaLayerNormZero_Final
    from x_transformers.x_transformers import RotaryEmbedding

    sd = _load_flow_state(model_dir)
    L = 22
    dim, heads, head_dim, ff_mult = DIT_DIM, 16, 64, 2

    # Build all 22 blocks.
    blocks = []
    for i in range(L):
        block = DiTBlock(dim=dim, heads=heads, dim_head=head_dim, ff_mult=ff_mult, dropout=0.0)
        prefix = f"decoder.estimator.transformer_blocks.{i}."
        bsd = {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}
        block.load_state_dict(bsd, strict=False)
        block.eval()
        blocks.append(block)

    # Build TimestepEmbedding for the t_emb input.
    time_embed = TimestepEmbedding(dim, freq_embed_dim=256)
    time_sd = {k[len("decoder.estimator.time_embed."):]: v
               for k, v in sd.items() if k.startswith("decoder.estimator.time_embed.")}
    time_embed.load_state_dict(time_sd, strict=False)
    time_embed.eval()

    # AdaLayerNormZero_Final from the same flow state-dict.
    norm_out = AdaLayerNormZero_Final(dim)
    nsd = {k[len("decoder.estimator.norm_out."):]: v
           for k, v in sd.items() if k.startswith("decoder.estimator.norm_out.")}
    norm_out.load_state_dict(nsd, strict=False)
    norm_out.eval()

    # proj_out: Linear(1024, 80).
    proj_out = torch.nn.Linear(dim, 80)
    proj_out.load_state_dict({
        "weight": sd["decoder.estimator.proj_out.weight"],
        "bias": sd["decoder.estimator.proj_out.bias"],
    }, strict=True)
    proj_out.eval()

    # Seeded inputs.
    gen = torch.Generator().manual_seed(seed)
    x = torch.randn(1, T_mel, dim, generator=gen, dtype=torch.float32)
    t_scalar = torch.tensor([timestep], dtype=torch.float32)
    with torch.no_grad():
        t_emb = time_embed(t_scalar)  # (1, 1024)

    rotary = RotaryEmbedding(head_dim)
    with torch.no_grad():
        rope = rotary.forward_from_seq_len(T_mel)
    # Full-True mask in the (B, 1, T, T) shape upstream uses.
    mask = torch.ones(1, T_mel, T_mel, dtype=torch.bool).unsqueeze(1)

    # Loop all 22 blocks, then norm_out + proj_out.
    with torch.no_grad():
        h = x
        for b in blocks:
            h = b(h, t_emb, mask=mask, rope=rope)
        h_norm = norm_out(h, t_emb)            # (1, T_mel, 1024)
        mel = proj_out(h_norm)                  # (1, T_mel, 80)

    return {
        "x_in": x.squeeze(0).contiguous(),         # (T_mel, 1024)
        "t_emb": t_emb.squeeze(0).contiguous(),    # (1024,)
        "norm": h_norm.squeeze(0).contiguous(),    # (T_mel, 1024)
        "": mel.squeeze(0).contiguous(),           # (T_mel, 80)
    }


# ---------------------------------------------------------------------------
# Phase 3d-B — CausalConditionalCFM Euler ODE capture
# ---------------------------------------------------------------------------

def _capture_euler_stages(model_dir: Path, T_tok: int = EULER_T_TOK,
                          spk_seed: int = EULER_SPK_SEED,
                          n_steps: int = EULER_N_STEPS,
                          cfg_rate: float = EULER_CFG_RATE) -> Dict[str, "torch.Tensor"]:
    """Build a CausalConditionalCFM with the upstream DiT estimator
    (loaded from flow.pt), run a seeded forward + capture intermediates.

    This is the full Euler ODE — 10 steps of CFG-guided cosine schedule
    over the random init noise. Returns the final mel + the post-CFG
    dphi_dt at step 1 (matches the C++ side's `euler_dphi_step0`).
    """
    _ensure_upstream_on_path()
    import sys
    import types
    import torch
    import torch.nn.functional as F
    from cosyvoice.transformer.upsample_encoder import PreLookaheadLayer
    from cosyvoice.flow.DiT.dit import DiT

    # `cosyvoice.flow.flow_matching` imports `matcha.models.components.flow_matching::BASECFM`
    # — a base nn.Module that contributes nothing to the inference path
    # (ConditionalCFM's __init__ stores its args; forward + solve_euler are
    # defined locally). Stub BASECFM so we can import without installing the
    # full matcha package.
    if "matcha" not in sys.modules:
        stub_root = types.ModuleType("matcha")
        stub_models = types.ModuleType("matcha.models")
        stub_comp = types.ModuleType("matcha.models.components")
        stub_fm = types.ModuleType("matcha.models.components.flow_matching")

        class _StubBASECFM(torch.nn.Module):
            def __init__(self, n_feats=None, cfm_params=None, n_spks=1, spk_emb_dim=64, *args, **kwargs):
                super().__init__()
                self.n_feats = n_feats
                self.n_spks = n_spks
                self.spk_emb_dim = spk_emb_dim
                if cfm_params is not None:
                    self.sigma_min = getattr(cfm_params, "sigma_min", 1e-06)
                    self.solver = getattr(cfm_params, "solver", "euler")
                    self.t_scheduler = getattr(cfm_params, "t_scheduler", "cosine")
                    self.training_cfg_rate = getattr(cfm_params, "training_cfg_rate", 0.2)
                    self.inference_cfg_rate = getattr(cfm_params, "inference_cfg_rate", 0.7)

        stub_fm.BASECFM = _StubBASECFM
        sys.modules["matcha"] = stub_root
        sys.modules["matcha.models"] = stub_models
        sys.modules["matcha.models.components"] = stub_comp
        sys.modules["matcha.models.components.flow_matching"] = stub_fm

    from cosyvoice.flow.flow_matching import CausalConditionalCFM
    from omegaconf import DictConfig

    sd = _load_flow_state(model_dir)
    spk_dim_in = SPK_DIM_IN

    # ---- Inputs: speech_tokens, spk_emb, cond=zeros ----
    # Use the same speech-token rng as phase 3c so the pre_la output
    # matches the existing flow_pre_la_* / flow_in_pipe_pre_la_in.
    tok_gen = torch.Generator().manual_seed(PRE_LA_SEED)
    speech_ids = torch.randint(0, 6561, (1, T_tok), generator=tok_gen, dtype=torch.long)
    spk_gen = torch.Generator().manual_seed(spk_seed)
    spks_raw = torch.randn(1, spk_dim_in, generator=spk_gen, dtype=torch.float32)

    # ---- Pre-lookahead → mu ----
    embed = torch.nn.Embedding(6561, MEL_DIM)
    embed.weight.data.copy_(sd["input_embedding.weight"].float())
    embed.eval()
    pre_la = PreLookaheadLayer(in_channels=MEL_DIM, channels=1024, pre_lookahead_len=3)
    pre_la.load_state_dict({k[len("pre_lookahead_layer."):]: v
                            for k, v in sd.items() if k.startswith("pre_lookahead_layer.")}, strict=True)
    pre_la.eval()
    with torch.no_grad():
        h_tok = pre_la(embed(speech_ids))               # (1, T_tok, mel)
    h_mel = h_tok.repeat_interleave(2, dim=1)            # (1, T_mel=2*T_tok, mel)
    T_mel = h_mel.shape[1]
    mu = h_mel.transpose(1, 2).contiguous()              # (1, mel, T_mel) — channel-first for the CFM

    # ---- spk projection ----
    spk_affine = torch.nn.Linear(spk_dim_in, MEL_DIM)
    spk_affine.load_state_dict({
        "weight": sd["spk_embed_affine_layer.weight"],
        "bias": sd["spk_embed_affine_layer.bias"],
    }, strict=True)
    spk_affine.eval()
    with torch.no_grad():
        spks_proj = spk_affine(F.normalize(spks_raw, dim=1))  # (1, mel)

    # cond = zeros (no prompt for the test fixture)
    cond = torch.zeros(1, MEL_DIM, T_mel, dtype=torch.float32)
    mask = torch.ones(1, 1, T_mel, dtype=torch.float32)

    # ---- Build the full DiT estimator from flow.pt ----
    dit = DiT(
        dim=DIT_DIM, depth=22, heads=16, dim_head=64, ff_mult=2,
        mel_dim=MEL_DIM, mu_dim=MEL_DIM, spk_dim=MEL_DIM,
        out_channels=MEL_DIM, dropout=0.0,
    )
    dit_prefix = "decoder.estimator."
    dit_sd = {k[len(dit_prefix):]: v for k, v in sd.items() if k.startswith(dit_prefix)}
    missing, unexpected = dit.load_state_dict(dit_sd, strict=False)
    if missing:
        print(f"  DiT load missing keys ({len(missing)}): {missing[:5]} ...")
    dit.eval()

    # ---- Build CausalConditionalCFM with the cv3 params ----
    cfm_params = DictConfig({
        "sigma_min": 1e-06,
        "solver": "euler",
        "t_scheduler": "cosine",
        "training_cfg_rate": 0.2,
        "inference_cfg_rate": cfg_rate,
        "reg_loss_type": "l1",
    })
    cfm = CausalConditionalCFM(in_channels=240, cfm_params=cfm_params,
                               n_spks=1, spk_emb_dim=MEL_DIM, estimator=dit)
    cfm.eval()
    # CausalConditionalCFM.__init__ sets the seeded rand_noise via
    # set_all_random_seed(0) — that's the SAME noise upstream uses at
    # runtime, so x_init below matches bit-for-bit.

    # The seeded init noise that solve_euler consumes (and that we hand
    # over to the C++ side via the GGUF).
    z = cfm.rand_noise[:, :, :T_mel].clone()             # (1, mel, T_mel)
    x_init = z.clone()                                    # save for the diff

    # ---- Capture dphi_step0 by hooking solve_euler manually ----
    # Replicate the first iteration of solve_euler so we can grab the
    # CFG-combined dphi_dt at step 1 BEFORE the x update.
    with torch.no_grad():
        # Build the t-span the way upstream does.
        n_timesteps = n_steps
        t_span = torch.linspace(0, 1, n_timesteps + 1, dtype=mu.dtype)
        t_span = 1 - torch.cos(t_span * 0.5 * torch.pi)
        t0 = t_span[0].unsqueeze(0)

        # First step's CFG-combined dphi_dt.
        x_in = torch.zeros([2, MEL_DIM, T_mel], dtype=mu.dtype)
        mask_in = torch.zeros([2, 1, T_mel], dtype=mu.dtype)
        mu_in = torch.zeros([2, MEL_DIM, T_mel], dtype=mu.dtype)
        t_in = torch.zeros([2], dtype=mu.dtype)
        spks_in = torch.zeros([2, MEL_DIM], dtype=mu.dtype)
        cond_in = torch.zeros([2, MEL_DIM, T_mel], dtype=mu.dtype)
        x_in[:] = z
        mask_in[:] = mask
        mu_in[0] = mu
        t_in[:] = t0.unsqueeze(0)
        spks_in[0] = spks_proj
        cond_in[0] = cond
        dphi_dt_batched = dit(x_in, mask_in, mu_in, t_in, spks_in, cond_in, streaming=False)
        dphi_cond, dphi_unc = torch.split(dphi_dt_batched, [1, 1], dim=0)
        dphi_step1 = (1.0 + cfg_rate) * dphi_cond - cfg_rate * dphi_unc

        # Full Euler forward (uses the same internal logic — solve_euler).
        mel_final, _ = cfm(mu=mu, mask=mask, n_timesteps=n_timesteps,
                           spks=spks_proj, cond=cond, streaming=False)

    # Convert all to (T_mel, mel_dim) row-major (= ggml ne=(mel, T_mel) col-major byte-identical).
    def tc(t):
        # Inputs are (1, C, T) channel-first. Take .squeeze(0).transpose(0, 1) → (T, C).
        if t.dim() == 3 and t.shape[1] == MEL_DIM:
            return t.squeeze(0).transpose(0, 1).contiguous()
        if t.dim() == 3 and t.shape[2] == MEL_DIM:
            return t.squeeze(0).contiguous()
        raise ValueError(f"unexpected shape {tuple(t.shape)}")

    return {
        "mu_in": tc(mu),
        "spks_in": spks_proj.squeeze(0).contiguous(),
        "cond_in": tc(cond),
        "x_init": tc(x_init),
        "dphi_step0": tc(dphi_step1),
        "": tc(mel_final),
    }


# ---------------------------------------------------------------------------
# dump_reference.py entry point.
# ---------------------------------------------------------------------------

def dump(*, model_dir: Path, audio: np.ndarray, stages: Set[str],
         **kwargs) -> Dict[str, np.ndarray]:  # noqa: ARG001
    """Capture the requested cosyvoice3-tts reference activations.

    For Phase 3b only the flow_dit_blk_<N>_* stages are populated. The
    audio arg is unused (the per-block test vector is seeded random,
    not derived from audio). LM-side stages can be re-added here when
    the unified diff grows past the DiT block.
    """
    requested = set(stages) if stages else set(DEFAULT_STAGES)
    out: Dict[str, np.ndarray] = {}

    # ---- Phase 3b DiT block stages ----
    by_block: Dict[int, Set[str]] = {}
    for s in requested:
        if not s.startswith("flow_dit_blk_"):
            continue
        rest = s[len("flow_dit_blk_"):]
        n_str, _, _ = rest.partition("_")
        if not n_str.isdigit():
            continue
        by_block.setdefault(int(n_str), set()).add(s)
    for block_idx, names in sorted(by_block.items()):
        out.update(_flow_dit_block_arrays(model_dir, block_idx, stages_wanted=names))

    # ---- Phase 3c pre-lookahead stages ----
    if any(s.startswith("flow_pre_la") for s in requested):
        pre_la_stages = _capture_pre_la_stages(model_dir)
        for short, t in pre_la_stages.items():
            name = "flow_pre_la" + (("_" + short) if short else "")
            if name not in requested:
                continue
            arr = t.detach().cpu().numpy()
            if arr.dtype != np.int32 and arr.dtype != np.float32:
                arr = arr.astype(np.float32)
            out[name] = arr

    # ---- Phase 3c InputEmbedding stages ----
    if any(s.startswith("flow_in_pipe") for s in requested):
        in_pipe_stages = _capture_in_pipe_stages(model_dir)
        for short, t in in_pipe_stages.items():
            name = "flow_in_pipe" + (("_" + short) if short else "")
            if name not in requested:
                continue
            out[name] = t.detach().cpu().numpy().astype(np.float32)

    # ---- Phase 3d-A full DiT estimator stages ----
    if any(s.startswith("flow_dit_full") for s in requested):
        full_stages = _capture_dit_full_stages(model_dir)
        for short, t in full_stages.items():
            name = "flow_dit_full" + (("_" + short) if short else "")
            if name not in requested:
                continue
            out[name] = t.detach().cpu().numpy().astype(np.float32)

    # ---- Phase 3d-B CFM Euler ODE end-to-end stages ----
    if any(s.startswith("flow_euler") for s in requested):
        euler_stages = _capture_euler_stages(model_dir)
        for short, t in euler_stages.items():
            name = "flow_euler" + (("_" + short) if short else "")
            if name not in requested:
                continue
            out[name] = t.detach().cpu().numpy().astype(np.float32)

    return out


# ---------------------------------------------------------------------------
# CLI (manual debug runs only — production diff goes through
# tools/dump_reference.py + crispasr-diff).
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", required=True, help="Local snapshot dir for Fun-CosyVoice3-0.5B-2512")
    ap.add_argument("--mode", choices=["lm-ref", "flow-dit-block"], default="lm-ref",
                    help="lm-ref: Phase 2 LM .npz dump. "
                         "flow-dit-block: Phase 3b per-block raw .bin dump (debug).")
    ap.add_argument("--output", help="Output .npz path (lm-ref only)")
    ap.add_argument("--prompt", default=DEFAULT_PROMPT, help="Text prompt (lm-ref only)")
    ap.add_argument("--n-greedy", type=int, default=32, help="Greedy AR steps (lm-ref only)")
    ap.add_argument("--flow-out-dir", help="Output dir for flow-dit-block .bin dumps")
    ap.add_argument("--block-idx", type=int, default=0, help="DiT block index to dump")
    ap.add_argument("--T", type=int, default=DIT_T, help="Sequence length")
    ap.add_argument("--seed", type=int, default=DIT_SEED, help="torch RNG seed for x")
    args = ap.parse_args()
    if args.mode == "lm-ref":
        if not args.output:
            ap.error("--output required for lm-ref mode")
        dump_lm_reference(Path(args.model_dir), Path(args.output), args.prompt, args.n_greedy)
    elif args.mode == "flow-dit-block":
        if not args.flow_out_dir:
            ap.error("--flow-out-dir required for flow-dit-block mode")
        dump_flow_dit_block_bins(Path(args.model_dir), Path(args.flow_out_dir),
                                 block_idx=args.block_idx, T=args.T, seed=args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
