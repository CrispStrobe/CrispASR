#!/usr/bin/env python3
"""
Bark stage-by-stage diff test: Python reference vs C++ runtime.

Generates reference dumps from the upstream bark model, then compares
against C++ dumps (produced via BARK_DUMP_DIR env var).

Usage:
  # Step 1: Generate Python reference
  python tools/bark_diff_test.py --generate-ref \
      --text "Hello there" --seed 42 \
      --dump-dir /mnt/storage/bark-tts/diff_ref

  # Step 2: Run C++ with BARK_DUMP_DIR
  BARK_DUMP_DIR=/mnt/storage/bark-tts/diff_cpp \
  ./build/bin/test_bark bark-small-q4_k_v2.gguf --text "Hello there"

  # Step 3: Compare
  python tools/bark_diff_test.py --compare \
      --ref-dir /mnt/storage/bark-tts/diff_ref \
      --cpp-dir /mnt/storage/bark-tts/diff_cpp
"""

import sys, os, types, importlib.util, argparse
import numpy as np
from pathlib import Path


def mock_torchaudio():
    """Mock torchaudio to avoid CUDA dependency."""
    ta = types.ModuleType('torchaudio')
    ta.__spec__ = importlib.util.spec_from_loader('torchaudio', loader=None)
    ta.__version__ = '0.0.0'
    sys.modules['torchaudio'] = ta
    for sub in ['_extension', '_extension.utils', 'functional']:
        m = types.ModuleType(f'torchaudio.{sub}')
        m.__spec__ = importlib.util.spec_from_loader(f'torchaudio.{sub}', loader=None)
        sys.modules[f'torchaudio.{sub}'] = m

    import torch, torch.nn.functional as F
    def fake_resample(wav, orig, target):
        ratio = target / orig
        new_len = int(wav.shape[-1] * ratio)
        return F.interpolate(wav.unsqueeze(0), size=new_len, mode='linear',
                             align_corners=False).squeeze(0)
    sys.modules['torchaudio.functional'].resample = fake_resample
    sys.modules['torchaudio'].functional = sys.modules['torchaudio.functional']


def patch_torch_load():
    """Fix PyTorch 2.6+ weights_only default."""
    import torch
    _orig = torch.load
    def _patched(*a, **kw):
        kw.setdefault('weights_only', False)
        return _orig(*a, **kw)
    torch.load = _patched


def generate_reference(text, seed, dump_dir):
    """Run upstream bark and dump all intermediates."""
    mock_torchaudio()
    patch_torch_load()

    os.environ["SUNO_USE_SMALL_MODELS"] = "True"
    os.environ["XDG_CACHE_HOME"] = "/mnt/storage/huggingface"
    os.environ["HF_HOME"] = "/mnt/storage/huggingface"

    import torch
    torch.manual_seed(seed)
    np.random.seed(seed)

    from bark.generation import (
        generate_text_semantic, generate_coarse, generate_fine,
        codec_decode, preload_models, SAMPLE_RATE, TEXT_ENCODING_OFFSET,
    )
    from transformers import BertTokenizer

    print("Loading models...", file=sys.stderr)
    preload_models(text_use_gpu=False, coarse_use_gpu=False,
                   fine_use_gpu=False, codec_use_gpu=False)

    # Show tokenization
    tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")
    bert_ids = tokenizer.encode(text, add_special_tokens=True)
    offset_ids = [x + TEXT_ENCODING_OFFSET for x in bert_ids]
    print(f"BERT IDs: {bert_ids}", file=sys.stderr)
    print(f"With offset: {offset_ids}", file=sys.stderr)

    dump = Path(dump_dir)
    dump.mkdir(parents=True, exist_ok=True)

    # Save tokenizer output
    np.save(str(dump / "bert_token_ids.npy"), np.array(bert_ids, dtype=np.int32))
    np.save(str(dump / "bert_offset_ids.npy"), np.array(offset_ids, dtype=np.int32))

    # Capture prefill logits from the semantic model (step 0, before sampling).
    # This is deterministic given the same input tokens and validates the
    # GPT-2 forward pass independently of RNG differences.
    try:
        from bark.generation import (
            _tokenize, _normalize_whitespace,
            SEMANTIC_PAD_TOKEN, SEMANTIC_INFER_TOKEN,
            TEXT_PAD_TOKEN,
        )
        import bark.generation as bg

        # Replicate bark's input construction (from generate_text_semantic)
        encoded = tokenizer.encode(text, add_special_tokens=True)
        # Pad/truncate to 256
        encoded = encoded[:256]
        encoded += [TEXT_PAD_TOKEN] * (256 - len(encoded))
        # Offset by TEXT_ENCODING_OFFSET
        encoded = [e + TEXT_ENCODING_OFFSET for e in encoded]
        # Semantic history: all PAD
        sem_hist = [SEMANTIC_PAD_TOKEN] * 256
        # Build input: text(256) + sem_hist(256) + INFER_TOKEN
        input_ids = encoded + sem_hist + [SEMANTIC_INFER_TOKEN]
        x = torch.tensor([input_ids], dtype=torch.long)
        # Get the model
        model = bg.models["text"]["model"]
        model.eval()
        with torch.no_grad():
            # bark text model: merge_context=True sums first 256 + next 256
            # embeddings, then appends the INFER_TOKEN embedding
            logits = model(x, merge_context=True)
        # logits shape: (1, n_out, output_vocab)
        prefill_logits = logits[0, -1, :].cpu().numpy()  # last position
        np.save(str(dump / "semantic_prefill_logits.npy"), prefill_logits.astype(np.float32))
        print(f"  Prefill logits: shape={prefill_logits.shape}, "
              f"argmax={prefill_logits.argmax()}, "
              f"top5={np.argsort(prefill_logits)[-5:][::-1].tolist()}",
              file=sys.stderr)
    except Exception as e:
        print(f"  WARNING: could not capture prefill logits: {e}", file=sys.stderr)

    # Stage 1
    print(f"Stage 1: semantic (text={text!r}, seed={seed})...", file=sys.stderr)
    semantic = generate_text_semantic(text, temp=0.7, use_kv_caching=True)
    print(f"  -> {len(semantic)} tokens, first 10: {semantic[:10].tolist()}", file=sys.stderr)
    np.save(str(dump / "semantic_tokens.npy"), semantic)

    # Stage 2
    print("Stage 2: coarse...", file=sys.stderr)
    coarse = generate_coarse(semantic, temp=0.7, use_kv_caching=True)
    print(f"  -> shape {coarse.shape}", file=sys.stderr)
    np.save(str(dump / "coarse_tokens.npy"), coarse)

    # Stage 3
    print("Stage 3: fine...", file=sys.stderr)
    fine = generate_fine(coarse, temp=0.5)
    print(f"  -> shape {fine.shape}", file=sys.stderr)
    np.save(str(dump / "fine_tokens.npy"), fine)

    # Decode
    print("Decoding...", file=sys.stderr)
    audio = codec_decode(fine)
    print(f"  -> {len(audio)} samples ({len(audio)/SAMPLE_RATE:.2f}s)", file=sys.stderr)
    np.save(str(dump / "audio_pcm.npy"), audio)

    # Also save WAV
    try:
        import soundfile as sf
        sf.write(str(dump / "ref.wav"), audio, SAMPLE_RATE)
    except Exception:
        pass

    print(f"Reference dumped to {dump}", file=sys.stderr)


def compare_stages(ref_dir, cpp_dir):
    """Compare Python reference vs C++ dumps stage by stage."""
    ref = Path(ref_dir)
    cpp = Path(cpp_dir)

    def cosine(a, b):
        a, b = a.flatten().astype(np.float64), b.flatten().astype(np.float64)
        n = min(len(a), len(b))
        a, b = a[:n], b[:n]
        dot = np.dot(a, b)
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na < 1e-30 or nb < 1e-30:
            return 0.0
        return dot / (na * nb)

    def load_cpp_int32(name):
        p = cpp / f"{name}.bin"
        if not p.exists():
            return None
        return np.fromfile(str(p), dtype=np.int32)

    def load_cpp_float(name):
        p = cpp / f"{name}.bin"
        if not p.exists():
            return None
        return np.fromfile(str(p), dtype=np.float32)

    results = []

    # --- Prefill logits (step 0, before any sampling) ---
    ref_logits_path = ref / "semantic_prefill_logits.npy"
    cpp_logits = load_cpp_float("semantic_prefill_logits")
    if ref_logits_path.exists() and cpp_logits is not None:
        ref_logits = np.load(str(ref_logits_path))
        n = min(len(ref_logits), len(cpp_logits))
        cos = cosine(ref_logits[:n], cpp_logits[:n])
        maxabs = np.max(np.abs(ref_logits[:n].astype(np.float64) - cpp_logits[:n].astype(np.float64)))
        ref_top5 = np.argsort(ref_logits)[-5:][::-1]
        cpp_top5 = np.argsort(cpp_logits)[-5:][::-1]
        argmax_match = (ref_logits[:n].argmax() == cpp_logits[:n].argmax())
        print(f"\n[prefill_logits]  ref={len(ref_logits)} cpp={len(cpp_logits)} "
              f"cos={cos:.6f} max_abs={maxabs:.4f} argmax_match={argmax_match}")
        print(f"  ref top5: {ref_top5.tolist()}")
        print(f"  cpp top5: {cpp_top5.tolist()}")
        status = f"cos={cos:.6f}"
        if cos > 0.999:
            status += " PASS"
        elif cos > 0.99:
            status += " CLOSE"
        else:
            status += " DIVERGED"
        results.append(("prefill_logits", status))
    elif cpp_logits is not None:
        print("\n[prefill_logits]  ref not found (re-run --generate-ref)")
        results.append(("prefill_logits", "REF MISSING"))
    else:
        print("\n[prefill_logits]  C++ dump not found (set BARK_DUMP_DIR)")
        results.append(("prefill_logits", "CPP MISSING"))

    # --- Semantic tokens ---
    ref_sem = np.load(str(ref / "semantic_tokens.npy"))
    cpp_sem = load_cpp_int32("semantic_tokens")
    if cpp_sem is not None:
        n = min(len(ref_sem), len(cpp_sem))
        match = np.sum(ref_sem[:n] == cpp_sem[:n])
        print(f"\n[semantic_tokens] ref={len(ref_sem)} cpp={len(cpp_sem)} "
              f"match={match}/{n} ({100*match/n:.1f}%)")
        if match == n and len(ref_sem) == len(cpp_sem):
            results.append(("semantic_tokens", "EXACT"))
        else:
            results.append(("semantic_tokens", f"{100*match/n:.1f}% match"))
    else:
        print("\n[semantic_tokens] C++ dump not found")
        results.append(("semantic_tokens", "MISSING"))

    # --- Coarse tokens ---
    ref_coarse = np.load(str(ref / "coarse_tokens.npy"))
    cpp_coarse = load_cpp_int32("coarse_tokens")
    if cpp_coarse is not None:
        # ref is (2, T), cpp is interleaved (cb0_t0, cb1_t0, cb0_t1, ...)
        ref_interleaved = ref_coarse.T.flatten()  # (T, 2).flatten()
        n = min(len(ref_interleaved), len(cpp_coarse))
        match = np.sum(ref_interleaved[:n] == cpp_coarse[:n])
        print(f"[coarse_tokens]   ref={ref_coarse.shape} cpp={len(cpp_coarse)} "
              f"match={match}/{n} ({100*match/n:.1f}%)")
        results.append(("coarse_tokens", f"{100*match/n:.1f}% match"))
    else:
        print("[coarse_tokens]   C++ dump not found")
        results.append(("coarse_tokens", "MISSING"))

    # --- Fine tokens ---
    ref_fine = np.load(str(ref / "fine_tokens.npy"))
    cpp_fine = load_cpp_int32("fine_codes")
    if cpp_fine is not None:
        # ref is (8, T), cpp is (8*T) flattened row-major
        ref_flat = ref_fine.flatten()
        n = min(len(ref_flat), len(cpp_fine))
        # Compare first 2 codebooks (from coarse — should match if coarse matches)
        T_ref = ref_fine.shape[1]
        T_cpp = len(cpp_fine) // 8 if len(cpp_fine) >= 8 else 0
        T = min(T_ref, T_cpp)
        if T > 0:
            for cb in range(8):
                ref_cb = ref_fine[cb, :T]
                cpp_cb = cpp_fine[cb*T_cpp:(cb*T_cpp)+T]
                m = np.sum(ref_cb == cpp_cb)
                tag = "coarse" if cb < 2 else "fine"
                print(f"[fine cb{cb}]        ref_T={T_ref} cpp_T={T_cpp} "
                      f"match={m}/{T} ({100*m/T:.1f}%) [{tag}]")
        results.append(("fine_codes", f"ref={ref_fine.shape} cpp_T={T_cpp}"))
    else:
        print("[fine_codes]      C++ dump not found")
        results.append(("fine_codes", "MISSING"))

    # --- Audio PCM ---
    ref_pcm = np.load(str(ref / "audio_pcm.npy"))
    cpp_pcm = load_cpp_float("encodec_pcm")
    if cpp_pcm is not None:
        cos = cosine(ref_pcm, cpp_pcm)
        n = min(len(ref_pcm), len(cpp_pcm))
        maxabs = np.max(np.abs(ref_pcm[:n].astype(np.float64) - cpp_pcm[:n].astype(np.float64)))
        print(f"[encodec_pcm]     ref={len(ref_pcm)} cpp={len(cpp_pcm)} "
              f"cos={cos:.6f} max_abs={maxabs:.6f}")
        results.append(("encodec_pcm", f"cos={cos:.6f}"))
    else:
        print("[encodec_pcm]     C++ dump not found")
        results.append(("encodec_pcm", "MISSING"))

    # --- Summary ---
    print(f"\n{'='*50}")
    print("Summary:")
    for name, status in results:
        print(f"  {name:20s}  {status}")
    print(f"{'='*50}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--generate-ref", action="store_true")
    ap.add_argument("--compare", action="store_true")
    ap.add_argument("--text", default="Hello there")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dump-dir", default="/mnt/storage/bark-tts/diff_ref")
    ap.add_argument("--ref-dir", default="/mnt/storage/bark-tts/diff_ref")
    ap.add_argument("--cpp-dir", default="/mnt/storage/bark-tts/diff_cpp")
    args = ap.parse_args()

    if args.generate_ref:
        generate_reference(args.text, args.seed, args.dump_dir)
    elif args.compare:
        compare_stages(args.ref_dir, args.cpp_dir)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
