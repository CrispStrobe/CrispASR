#!/usr/bin/env python3
"""CrispASR — Japanese ASR NeMo reference comparison.

Runs ReazonSpeech NeMo v2 (RNNT) and parakeet-ctc-1.1b-ja (CTC) through
the upstream NeMo pipeline, then builds CrispASR and compares transcripts.
"""
import json, os, re, subprocess, sys, time, traceback
from pathlib import Path

os.environ["PYTHONUNBUFFERED"] = "1"
for s in (sys.stdout, sys.stderr):
    try: s.reconfigure(line_buffering=True)
    except Exception: pass

WORK = Path("/kaggle/working")
CRASH = WORK / "crash.txt"

def main():
    REPO = WORK / "CrispASR"
    BUILD = REPO / "build"
    REF = os.environ.get("CRISPASR_REF", "main")
    URL = "https://github.com/CrispStrobe/CrispASR.git"

    progress = WORK / "progress.jsonl"
    t0 = time.time()
    def step(name, **kv):
        rec = {"t": round(time.time()-t0, 2), "step": name, **kv}
        print(f"[step] {json.dumps(rec)}", flush=True)
        with open(progress, "a") as f: f.write(json.dumps(rec)+"\n")

    step("start")

    # ── clone ─────────────────────────────────────────────────────────
    step("clone")
    if not REPO.exists():
        subprocess.check_call(["git","clone","--depth","1","-b",REF,URL,str(REPO)])
    else:
        subprocess.check_call(["git","fetch","--depth","1","origin",REF], cwd=str(REPO))
        subprocess.check_call(["git","checkout","FETCH_HEAD"], cwd=str(REPO))

    sys.path.insert(0, str(REPO/"tools"/"kaggle"))
    import kaggle_harness as kh
    kh.init_progress()

    # ── generate Japanese test audio ──────────────────────────────────
    kh.step("gen_audio")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "gtts"])
    from gtts import gTTS
    import soundfile as sf
    test_text = "こんにちは、今日はとても良い天気ですね。東京の気温は25度です。"
    wav_path = str(WORK / "test_ja.wav")
    tts = gTTS(test_text, lang='ja')
    tts.save(str(WORK / "test_ja.mp3"))
    subprocess.run(["ffmpeg", "-y", "-i", str(WORK / "test_ja.mp3"),
                    "-ar", "16000", "-ac", "1", wav_path], capture_output=True)
    audio, sr = sf.read(wav_path)
    print(f"Test audio: {len(audio)/sr:.1f}s  text: {test_text}")

    results = {}

    # ── ReazonSpeech NeMo v2 (RNNT) via Python ───────────────────────
    kh.step("nemo_rnnt")
    try:
        import gc, torch
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
            "nemo_toolkit[asr]",
            "git+https://github.com/reazon-research/ReazonSpeech.git#subdirectory=pkg/nemo-asr"])
        from reazonspeech.nemo.asr import load_model, transcribe, audio_from_path
        with kh.build_heartbeat("nemo_rnnt"):
            audio_obj = audio_from_path(wav_path)
            model = load_model()
            ret = transcribe(model, audio_obj)
        results["python_rnnt"] = ret.text
        print(f"Python RNNT: {ret.text}")
        del model; gc.collect()
    except Exception as e:
        print(f"ReazonSpeech RNNT failed: {e}")
        traceback.print_exc()
        results["python_rnnt"] = f"ERROR: {e}"

    # ── Parakeet CTC 1.1b JA via Python ──────────────────────────────
    kh.step("nemo_ctc")
    try:
        import gc, torch
        import nemo.collections.asr as nemo_asr
        from huggingface_hub import hf_hub_download
        with kh.build_heartbeat("nemo_ctc_download"):
            nemo_path = hf_hub_download("grider-transwithai/parakeet-ctc-1.1b-ja", "parakeet-ja-gal.nemo")
        with kh.build_heartbeat("nemo_ctc_inference"):
            ctc_model = nemo_asr.models.EncDecCTCModelBPE.restore_from(nemo_path, map_location="cpu")
            ctc_model.eval()
            with torch.no_grad():
                ctc_result = ctc_model.transcribe([wav_path])
        # NeMo CTC .transcribe() returns a list of Hypothesis objects; extract .text
        raw = ctc_result[0] if isinstance(ctc_result, list) else ctc_result
        results["python_ctc"] = raw.text if hasattr(raw, "text") else str(raw)
        print(f"Python CTC:  {results['python_ctc']}")
        del ctc_model; gc.collect()
    except Exception as e:
        print(f"Parakeet CTC failed: {e}")
        traceback.print_exc()
        results["python_ctc"] = f"ERROR: {e}"

    # ── Build CrispASR ────────────────────────────────────────────────
    kh.step("toolchain")
    kh.install_build_toolchain()

    kh.step("configure")
    flags = kh.cache_and_link_flags()
    with kh.build_heartbeat("cmake.configure"):
        kh.sh(f"cmake -S {REPO} -B {BUILD} -G Ninja -DCMAKE_BUILD_TYPE=Release "
               f"-DCRISPASR_OPUS=OFF -DCRISPASR_AMR=OFF "
               + " ".join(flags))

    kh.step("build")
    with kh.build_heartbeat("cmake.build"):
        kh.sh(f"cmake --build {BUILD} -j$(nproc)")

    # ── Download GGUF models ──────────────────────────────────────────
    kh.step("download_gguf")
    gguf_dir = WORK / "gguf"
    gguf_dir.mkdir(exist_ok=True)
    rnnt_gguf = hf_hub_download("cstr/reazonspeech-nemo-v2-GGUF",
                                 "reazonspeech-nemo-v2-q8_0.gguf",
                                 local_dir=str(gguf_dir))
    ctc_gguf = hf_hub_download("cstr/parakeet-ctc-1.1b-ja-GGUF",
                                "parakeet-ctc-1.1b-ja-q8_0.gguf",
                                local_dir=str(gguf_dir))

    # ── Run CrispASR on same audio ────────────────────────────────────
    cli = str(BUILD / "bin" / "crispasr")

    kh.step("crispasr_rnnt")
    try:
        r = subprocess.run([cli, "--backend", "reazonspeech", "-m", rnnt_gguf,
                           wav_path, "--no-prints"],
                          capture_output=True, text=True, timeout=300)
        cpp_rnnt = r.stdout.strip()
        results["cpp_rnnt"] = cpp_rnnt
        print(f"C++ RNNT:    {cpp_rnnt}")
    except Exception as e:
        results["cpp_rnnt"] = f"ERROR: {e}"

    kh.step("crispasr_ctc")
    try:
        r = subprocess.run([cli, "--backend", "fastconformer-ctc", "-m", ctc_gguf,
                           wav_path, "--no-prints"],
                          capture_output=True, text=True, timeout=300)
        cpp_ctc = r.stdout.strip()
        results["cpp_ctc"] = cpp_ctc
        print(f"C++ CTC:     {cpp_ctc}")
    except Exception as e:
        results["cpp_ctc"] = f"ERROR: {e}"

    # ── Compare ───────────────────────────────────────────────────────
    kh.step("compare")
    print(f"\n{'='*60}")
    print(f"Reference text: {test_text}")
    print(f"{'='*60}")
    for key, val in results.items():
        print(f"  {key:20s}: {val}")

    with open(WORK / "results.json", "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    kh.step("done")

if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        with open(CRASH, "w") as f:
            traceback.print_exc(file=f)
        sys.exit(1)
