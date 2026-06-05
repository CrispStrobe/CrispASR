#!/usr/bin/env python3
"""Test FireRedLID on multilingual audio samples."""
import os, subprocess, sys, tempfile
import numpy as np
import soundfile as sf
from datasets import load_dataset

# MLS languages (no English — it's in a separate dataset)
MLS_LANGS = {
    "german": "de", "french": "fr", "spanish": "es", "italian": "it",
    "portuguese": "pt", "polish": "pl", "dutch": "nl",
}

N = 2

SCRATCH_ROOT = os.environ.get("CRISPASR_SCRATCH_DIR") or os.environ.get("CRISP_SCRATCH_DIR") or ".scratch"

def run_lid(model_path, cli, wav_path):
    try:
        out = subprocess.run(
            [cli, "-m", model_path, "-f", wav_path, "-np"],
            capture_output=True, text=True, timeout=180
        )
        return out.stdout.strip()
    except:
        return "ERR"

def main():
    models = sys.argv[1:]
    if not models:
        default_model = os.environ.get("CRISPASR_LID_MODEL")
        if not default_model:
            raise SystemExit("Usage: tools/test_lid_mls.py MODEL.gguf [MODEL2.gguf ...]")
        models = [default_model]
    cli = "./build/bin/crispasr"

    for model_path in models:
        sz = os.path.getsize(model_path) / 1e6
        print(f"\n{'='*60}")
        print(f"Model: {os.path.basename(model_path)} ({sz:.0f} MB)")
        print(f"{'='*60}")
        correct = total = 0

        # English from JFK sample
        for i in range(1):
            predicted = run_lid(model_path, cli, "samples/jfk.wav")
            ok = predicted == "en"
            if ok: correct += 1
            total += 1
            print(f"  {'✓' if ok else '✗'} {'english':12s} exp={'en':3s} got={predicted:12s} (jfk.wav)")

        # MLS languages
        for mls_name, expected in MLS_LANGS.items():
            ds = load_dataset("facebook/multilingual_librispeech", mls_name,
                            split="test", streaming=True)
            for i, row in enumerate(ds):
                if i >= N: break
                audio = row["audio"]
                pcm = np.array(audio["array"], dtype=np.float32)
                sr = audio["sampling_rate"]
                dur = len(pcm) / sr
                os.makedirs(SCRATCH_ROOT, exist_ok=True)
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False, dir=SCRATCH_ROOT) as f:
                    sf.write(f.name, pcm, sr)
                    wav_path = f.name
                predicted = run_lid(model_path, cli, wav_path)
                os.unlink(wav_path)
                ok = predicted == expected
                if ok: correct += 1
                total += 1
                print(f"  {'✓' if ok else '✗'} {mls_name:12s} exp={expected:3s} got={predicted:12s} ({dur:.1f}s)")

        acc = correct / total * 100 if total else 0
        print(f"\nAccuracy: {correct}/{total} ({acc:.1f}%)")

if __name__ == "__main__":
    main()
