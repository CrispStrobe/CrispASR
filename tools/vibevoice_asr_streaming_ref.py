#!/usr/bin/env python3
"""Dump the official VibeVoice-ASR-Streaming trajectory for C++ parity.

The output names match CRISPASR_VIBEVOICE_DUMP_DIR where practical. Arrays are
raw little-endian float32/int32 plus manifest.json shapes and the final chunk
texts. This deliberately implements the documented streaming loop explicitly,
so prompt, feature, logit, token, delimiter, and persistent-KV boundaries are
all observable rather than hidden behind ``streaming_generate``.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from vibevoice.modular.modeling_vibevoice_asr import VibeVoiceASRForConditionalGeneration
from vibevoice.processor.audio_utils import load_audio_use_ffmpeg
from vibevoice.processor.vibevoice_asr_processor import VibeVoiceASRProcessor


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="microsoft/VibeVoice-ASR-Streaming-1.5B")
    ap.add_argument("--audio", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--context", default="")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--mean-acoustic", action="store_true",
                    help="use posterior mean for deterministic stage-by-stage parity")
    args = ap.parse_args()
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    processor = VibeVoiceASRProcessor.from_pretrained(args.model)
    dtype = torch.float16 if args.device == "cuda" else torch.float32
    model = VibeVoiceASRForConditionalGeneration.from_pretrained(
        args.model, dtype=dtype, attn_implementation="eager"
    ).to(args.device).eval()
    if args.mean_acoustic:
        model.model.acoustic_tokenizer.std_dist_type = "none"
    tokenizer = processor.tokenizer

    # Resolve the authoritative frame settings from the snapshot itself.
    if Path(args.model).is_dir():
        cfg_path = Path(args.model) / "preprocessor_config.json"
    else:
        from huggingface_hub import hf_hub_download
        cfg_path = Path(hf_hub_download(args.model, "preprocessor_config.json"))
    cfg = json.loads(Path(cfg_path).read_text())
    sample_rate = int(cfg["target_sample_rate"])
    frame_samples = int(cfg["speech_tok_compress_ratio"])
    chunk_samples = int(cfg["chunk_frames"]) * frame_samples
    lookahead_samples = int(cfg["lookahead_frames"]) * frame_samples
    window_samples = chunk_samples + lookahead_samples

    audio, _ = load_audio_use_ffmpeg(args.audio, resample=True, target_sr=sample_rate)
    audio = torch.from_numpy(audio).to(args.device, dtype=dtype).reshape(1, -1)
    keys = "speaker, content"
    prompt = (
        "You are a helpful assistant that transcribes audio input into text output. "
        f"Please transcribe the following audios streamingly with these keys: {keys}"
    )
    if args.context.strip():
        prompt += f" and extra info: {args.context.strip()}"
    prompt += "\n"

    manifest = {"sample_rate": sample_rate, "chunk_samples": chunk_samples,
                "lookahead_samples": lookahead_samples, "arrays": {}}

    def dump(name: str, tensor: torch.Tensor, dtype_np) -> None:
        a = tensor.detach().cpu().to(torch.float32 if dtype_np == np.float32 else torch.int32).numpy()
        a = np.asarray(a, dtype=dtype_np)
        a.tofile(out_dir / f"{name}.bin")
        manifest["arrays"][name] = list(a.shape)

    ids = tokenizer.encode(prompt, add_special_tokens=False)
    prompt_ids = torch.tensor([ids], dtype=torch.long, device=args.device)
    dump("prompt_ids", prompt_ids, np.int32)
    embeds = model.get_input_embeddings()
    with torch.no_grad():
        outputs = model(inputs_embeds=embeds(prompt_ids), use_cache=True, return_dict=True)
    state = outputs.past_key_values
    dump("prefill_logits", outputs.logits[:, -1, :], np.float32)

    start_embed = embeds(torch.tensor([[tokenizer.speech_start_id]], device=args.device))
    end_embed = embeds(torch.tensor([[tokenizer.speech_end_id]], device=args.device))
    chunks = []
    total = audio.shape[1]
    for ci, start in enumerate(range(0, total, chunk_samples)):
        segment = audio[:, start:min(start + window_samples, total)]
        if segment.shape[1] < window_samples:
            segment = torch.nn.functional.pad(segment, (0, window_samples - segment.shape[1]))
        with torch.no_grad():
            features = model.encode_speech(segment)
            outputs = model(inputs_embeds=torch.cat([start_embed, features, end_embed], dim=1),
                            past_key_values=state, use_cache=True, return_dict=True)
        state = outputs.past_key_values
        dump(f"chunk_{ci:03d}_speech_features", features, np.float32)
        dump(f"chunk_{ci:03d}_audio_logits", outputs.logits[:, -1, :], np.float32)
        logits = outputs.logits[:, -1, :]
        generated = []
        for step in range(256):
            token = int(torch.argmax(logits, dim=-1).item())
            if token in (tokenizer.text_chunk_end_id, tokenizer.eos_token_id):
                break
            generated.append(token)
            with torch.no_grad():
                outputs = model(inputs_embeds=embeds(torch.tensor([[token]], device=args.device)),
                                past_key_values=state, use_cache=True, return_dict=True)
            state = outputs.past_key_values
            logits = outputs.logits[:, -1, :]
            if step < 4:
                dump(f"chunk_{ci:03d}_step_{step:03d}_logits", logits, np.float32)
        dump(f"chunk_{ci:03d}_generated_ids", torch.tensor(generated), np.int32)
        text = tokenizer.decode(generated, skip_special_tokens=True)
        for special in ("<|text_chunk_end|>", "<|object_ref_start|>", "<|object_ref_end|>",
                        "<|box_start|>", "<|speech_start|>", "<|speech_end|>", "<|speech_pad|>"):
            text = text.replace(special, "")
        chunks.append(text)
        delimiter = torch.tensor([[tokenizer.text_chunk_end_id]], device=args.device)
        with torch.no_grad():
            outputs = model(inputs_embeds=embeds(delimiter), past_key_values=state,
                            use_cache=True, return_dict=True)
        state = outputs.past_key_values
        # A small deterministic KV slice proves that the state grew rather than
        # being reconstructed independently for each audio window.
        key0 = state.layers[0].keys if hasattr(state, "layers") else state.key_cache[0]
        dump(f"chunk_{ci:03d}_kv_key0_tail", key0[:, :, -1, :], np.float32)

    manifest["chunks"] = chunks
    manifest["transcript"] = "".join(chunks)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(manifest["transcript"])


if __name__ == "__main__":
    main()
