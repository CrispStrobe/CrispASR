#!/usr/bin/env python3
"""
Audio8-TTS (Arktts) reference dumper for the CrispASR diff harness.

Runs Edge0/Audio8-TTS-Realistic-0.6b to generate audio from text, dumping
per-stage tensors (text embedding, transformer hidden states, acoustic
codebooks, and decoded PCM) for parity verification with src/arktts.cpp.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np

try:
    from gguf import GGUFWriter, GGMLQuantizationType
except ImportError:
    sys.exit("pip install gguf")

try:
    import torch
    import torch.nn.functional as F
except ImportError:
    sys.exit("pip install torch")


def write_f32_tensor(writer: GGUFWriter, name: str, data: Any):
    """Write float32 tensor to reference GGUF."""
    if isinstance(data, torch.Tensor):
        data = data.detach().float().cpu().numpy()
    writer.add_tensor(name, data.astype(np.float32))


def main():
    parser = argparse.ArgumentParser(description="Audio8-TTS (Arktts) reference dumper")
    parser.add_argument("--model-dir", type=str, required=True, help="HF model directory")
    parser.add_argument("--text", type=str, default="Welcome to Audio8 TTS.", help="Text to synthesize")
    parser.add_argument("--output", type=str, required=True, help="Output GGUF path")
    parser.add_argument("--max-tokens", type=int, default=20, help="Max generation tokens")
    parser.add_argument("--temperature", type=float, default=0.0, help="Generation temperature (0.0 for greedy)")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    pkg_dir = Path("/tmp/arktts_pkg")
    pkg_dir.mkdir(parents=True, exist_ok=True)
    import shutil
    for py_file in model_dir.glob("*.py"):
        shutil.copy2(py_file, pkg_dir / py_file.name)
    (pkg_dir / "__init__.py").touch()
    sys.path.insert(0, "/tmp")

    print(f"Loading Arktts from {model_dir}...", file=sys.stderr)
    from arktts_pkg.configuration_arktts import ArkttsConfig
    from arktts_pkg.modeling_arktts import ArkttsModel, _precompute_rope
    from arktts_pkg.modeling_arktts_codec import ArkttsCodec
    from arktts_pkg.processing_arktts import ArkttsProcessor
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)
    config = ArkttsConfig.from_pretrained(str(model_dir))
    model = ArkttsModel.from_pretrained(str(model_dir))
    model.freqs_cis = _precompute_rope(config.max_seq_len, config.head_dim, config.rope_base).to(model.device)
    model.fast_freqs_cis = _precompute_rope(config.num_codebooks, config.fast_head_dim, config.rope_base).to(model.device)
    model.eval()

    codec_path = model_dir / "codec.pth"
    codec = None
    if codec_path.exists():
        print(f"Loading Codec from {codec_path}...", file=sys.stderr)
        codec_sd = torch.load(str(codec_path), map_location="cpu")
        if isinstance(codec_sd, dict) and "state_dict" in codec_sd:
            codec_sd = codec_sd["state_dict"]
        codec = ArkttsCodec(config)
        codec.load_state_dict(codec_sd, strict=False)
        codec.eval()

    processor = ArkttsProcessor(tokenizer=tokenizer)

    # 1. Process text prompt
    prompt_text = args.text
    encoded = processor(text=prompt_text, return_tensors="pt")
    input_ids, attention_mask = model._prepare_prompt(**encoded)  # Shape: [1, 11, seq_len]
    print(f"Prompt input_ids shape: {input_ids.shape}", file=sys.stderr)

    writer = GGUFWriter(str(args.output), "arktts-ref")
    writer.add_string("arktts.prompt_text", prompt_text)
    writer.add_uint32("arktts.seq_len", input_ids.shape[-1])

    # Stage 0: text_input_ids (flattened or 2D)
    write_f32_tensor(writer, "text_input_ids", input_ids.squeeze(0))

    # Stage 1: Embeddings
    with torch.no_grad():
        token_embed = model._embed(input_ids)
        write_f32_tensor(writer, "embed_tokens", token_embed.squeeze(0))

        # Stage 2: Slow transformer forward
        seq_len = input_ids.shape[-1]
        attention_mask = torch.ones((1, seq_len), dtype=torch.long)
        position_ids = attention_mask.long().cumsum(-1).sub(1).clamp_min(0)
        rope = model.freqs_cis[position_ids]
        mask = model._causal_mask(attention_mask, torch.arange(seq_len), seq_len)

        hidden = token_embed
        for i, layer in enumerate(model.layers):
            hidden = layer(hidden, rope, mask)
            if i in (0, 11, 23):
                write_f32_tensor(writer, f"slow_layer_{i}_out", hidden.squeeze(0))

        normalized = model.norm(hidden)
        write_f32_tensor(writer, "slow_norm_out", normalized.squeeze(0))

        slow_logits = F.linear(normalized, model.embeddings.weight)
        last_logits = slow_logits[:, -1, :]
        write_f32_tensor(writer, "slow_logits_last", last_logits.squeeze(0))

        # Stage 3: Greedy semantic token selection
        # Filter to semantic range [151678, 155773]
        semantic_scores = last_logits.clone()
        mask_range = torch.full_like(semantic_scores, float("-inf"))
        mask_range[:, config.semantic_begin_id : config.semantic_end_id + 1] = semantic_scores[
            :, config.semantic_begin_id : config.semantic_end_id + 1
        ]
        semantic_id = mask_range.argmax(dim=-1)
        print(f"First generated semantic_id: {semantic_id.item()}", file=sys.stderr)
        write_f32_tensor(writer, "selected_semantic_id", semantic_id)

        # Stage 4: Fast RAS subnetwork for codebooks 1..9
        slow_last_hidden = normalized[:, -1:] if config.norm_fastlayer_input else hidden[:, -1:]
        fast_proj_hidden = model.fast_project_in(slow_last_hidden)
        write_f32_tensor(writer, "fast_proj_hidden", fast_proj_hidden.squeeze(0).squeeze(0))

        # Fast step position 0
        model._setup_generation_caches(1, config.max_seq_len, model.dtype)
        model._fast_step(fast_proj_hidden, 0)

        # Codebook 0 is semantic offset
        c0 = (semantic_id - config.semantic_begin_id).clamp(0, config.codebook_size - 1)
        codebooks = [c0]
        cur_embed = model.fast_embeddings(c0)[:, None]
        for pos in range(1, config.num_codebooks):
            scores = model._fast_step(cur_embed, pos)
            write_f32_tensor(writer, f"fast_logits_pos_{pos}", scores.squeeze(0))
            next_code = scores.argmax(dim=-1)
            codebooks.append(next_code)
            cur_embed = model.fast_embeddings(next_code)[:, None]

        all_codebooks = torch.stack(codebooks, dim=1)  # [1, 10]
        write_f32_tensor(writer, "frame0_codebooks", all_codebooks)
        print(f"Frame 0 acoustic codes (10 codebooks): {all_codebooks.tolist()}", file=sys.stderr)

        # Stage 5: Codec decoding of frame0 (or multi-frame)
        if codec is not None:
            # Replicate or tile codes across 4 frames so vocoder has enough temporal receptive field
            test_codes = all_codebooks.unsqueeze(-1).repeat(1, 1, 4)  # [1, 10, 4]
            write_f32_tensor(writer, "test_codes_4frames", test_codes.squeeze(0))

            indices = test_codes.clone()
            indices[:, 0].clamp_(0, codec.quantizer.semantic_quantizer.codebook_size - 1)
            indices[:, 1:].clamp_(0, codec.quantizer.quantizer.codebook_size - 1)
            sem = codec.quantizer.semantic_quantizer.from_codes(indices[:, :1])
            res = codec.quantizer.quantizer.from_codes(indices[:, 1:])
            quant_sum = sem + res
            write_f32_tensor(writer, "quant_sum_latents", quant_sum.squeeze(0))

            post_out = codec.quantizer.post_module(quant_sum)
            write_f32_tensor(writer, "post_module_out", post_out.squeeze(0))

            upsample_out = codec.quantizer.upsample(post_out)
            write_f32_tensor(writer, "upsample_out", upsample_out.squeeze(0))

            pcm_out = codec.decoder(upsample_out)
            write_f32_tensor(writer, "decoded_pcm", pcm_out.squeeze(0).squeeze(0))
            print(f"Decoded PCM shape: {pcm_out.shape}, samples={pcm_out.numel()}", file=sys.stderr)

    print(f"Writing reference GGUF to {args.output}...", file=sys.stderr)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print("Reference dump completed successfully.", file=sys.stderr)


if __name__ == "__main__":
    main()
