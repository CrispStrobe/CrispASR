#!/usr/bin/env python3
"""Benchmark LID models on edge-tts generated multilingual samples.

Generates 3 short TTS samples per language (12 languages), runs LID
inference with the specified model(s), and reports accuracy.

Requirements: pip install edge-tts soundfile
System: ffmpeg (for mp3→wav conversion)

Usage:
  python tools/benchmark_lid.py model1.gguf [model2.gguf ...]

  # Compare Q4_K vs Q2_K:
  python tools/benchmark_lid.py firered-lid-q4_k.gguf firered-lid-q2_k.gguf

The script generates ~36 short wav files under CRISPASR_SCRATCH_DIR on first
run. Subsequent runs reuse cached audio if present.

License: MIT (script only — generated TTS audio is transient and
governed by Microsoft Azure TTS terms of service).
"""

import asyncio
import os
import subprocess
import sys
import time

LANGS = {
    "en": ("en-US-AriaNeural", [
        "The quick brown fox jumps over the lazy dog.",
        "Good morning, how are you doing today?",
        "Technology is changing our world rapidly.",
    ]),
    "de": ("de-DE-KatjaNeural", [
        "Die schnelle braune Katze springt über den faulen Hund.",
        "Guten Morgen, wie geht es Ihnen heute?",
        "Die Technologie verändert unsere Welt rasant.",
    ]),
    "fr": ("fr-FR-DeniseNeural", [
        "Le renard brun rapide saute par-dessus le chien paresseux.",
        "Bonjour, comment allez-vous aujourd'hui?",
        "La technologie change notre monde rapidement.",
    ]),
    "es": ("es-ES-ElviraNeural", [
        "El rápido zorro marrón salta sobre el perro perezoso.",
        "Buenos días, cómo estás hoy?",
        "La tecnología está cambiando nuestro mundo rápidamente.",
    ]),
    "ja": ("ja-JP-NanamiNeural", [
        "今日はとても良い天気ですね。",
        "東京は日本の首都です。",
        "技術は私たちの世界を急速に変えています。",
    ]),
    "zh": ("zh-CN-XiaoxiaoNeural", [
        "今天天气真好啊。",
        "北京是中国的首都。",
        "技术正在迅速改变我们的世界。",
    ]),
    "ko": ("ko-KR-SunHiNeural", [
        "오늘 날씨가 정말 좋습니다.",
        "서울은 한국의 수도입니다.",
        "기술이 우리의 세계를 빠르게 변화시키고 있습니다.",
    ]),
    "ru": ("ru-RU-SvetlanaNeural", [
        "Быстрая коричневая лиса перепрыгивает через ленивую собаку.",
        "Доброе утро, как у вас дела?",
        "Технологии быстро меняют наш мир.",
    ]),
    "ar": ("ar-SA-ZariyahNeural", [
        "صباح الخير، كيف حالك اليوم؟",
        "التكنولوجيا تغير عالمنا بسرعة.",
        "الرياض هي عاصمة المملكة العربية السعودية.",
    ]),
    "hi": ("hi-IN-SwaraNeural", [
        "आज मौसम बहुत अच्छा है।",
        "दिल्ली भारत की राजधानी है।",
        "प्रौद्योगिकी हमारी दुनिया को तेजी से बदल रही है।",
    ]),
    "pt": ("pt-BR-FranciscaNeural", [
        "A rápida raposa marrom pula sobre o cão preguiçoso.",
        "Bom dia, como você está hoje?",
        "A tecnologia está mudando nosso mundo rapidamente.",
    ]),
    "it": ("it-IT-ElsaNeural", [
        "La veloce volpe marrone salta sopra il cane pigro.",
        "Buongiorno, come stai oggi?",
        "La tecnologia sta cambiando rapidamente il nostro mondo.",
    ]),
}

SCRATCH_ROOT = os.environ.get("CRISPASR_SCRATCH_DIR") or os.environ.get("CRISP_SCRATCH_DIR") or ".scratch"
CACHE_DIR = os.path.join(SCRATCH_ROOT, "lid_bench")


async def generate_samples():
    """Generate TTS samples if not cached."""
    import edge_tts

    os.makedirs(CACHE_DIR, exist_ok=True)
    samples = []
    for lang, (voice, texts) in LANGS.items():
        for i, text in enumerate(texts):
            wav_path = os.path.join(CACHE_DIR, f"{lang}_{i}.wav")
            if os.path.exists(wav_path):
                samples.append((lang, wav_path))
                continue
            mp3_path = wav_path.replace(".wav", ".mp3")
            comm = edge_tts.Communicate(text, voice)
            await comm.save(mp3_path)
            subprocess.run(
                ["ffmpeg", "-y", "-i", mp3_path, "-ar", "16000", "-ac", "1", wav_path],
                capture_output=True,
            )
            if os.path.exists(wav_path):
                samples.append((lang, wav_path))
    return samples


def run_benchmark(model_path, samples, cli="./build/bin/crispasr"):
    """Run LID on all samples and return (correct, total, elapsed)."""
    correct = total = 0
    details = []
    t0 = time.time()
    for expected, wav_path in samples:
        try:
            out = subprocess.run(
                [cli, "-m", model_path, "-f", wav_path, "-np"],
                capture_output=True,
                text=True,
                timeout=120,
            )
            predicted = out.stdout.strip()
        except subprocess.TimeoutExpired:
            predicted = "TIMEOUT"
        except Exception:
            predicted = "ERR"
        ok = predicted == expected
        if ok:
            correct += 1
        total += 1
        details.append((expected, predicted, ok))
    elapsed = time.time() - t0
    return correct, total, elapsed, details


def main():
    models = sys.argv[1:]
    if not models:
        print("Usage: python tools/benchmark_lid.py model1.gguf [model2.gguf ...]")
        sys.exit(1)

    cli = "./build/bin/crispasr"
    if not os.path.exists(cli):
        cli = "crispasr"

    print("Generating TTS samples...")
    samples = asyncio.run(generate_samples())
    print(f"Ready: {len(samples)} samples across {len(LANGS)} languages\n")

    for model_path in models:
        sz = os.path.getsize(model_path) / 1e6 if os.path.exists(model_path) else 0
        name = os.path.basename(model_path)
        print(f"{'=' * 60}")
        print(f"  {name} ({sz:.0f} MB)")
        print(f"{'=' * 60}")

        correct, total, elapsed, details = run_benchmark(model_path, samples, cli)

        for expected, predicted, ok in details:
            mark = "\u2713" if ok else "\u2717"
            print(f"  {mark} exp={expected:3s} got={predicted:12s}")

        acc = correct / total * 100 if total else 0
        print(f"\n  Accuracy: {correct}/{total} ({acc:.1f}%) in {elapsed:.0f}s")
        print(f"  Avg per sample: {elapsed / total:.1f}s\n")


if __name__ == "__main__":
    main()
