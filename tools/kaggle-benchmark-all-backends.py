# ─────────────────────────── cell 0 (markdown) ───────────────────────────
# # CrispASR — Comprehensive Backend Benchmark on Kaggle
#
# Tests ALL CrispASR backends on Kaggle GPU/CPU, collecting:
# - Transcription accuracy (WER against reference)
# - Inference speed (realtime factor)
# - Model sizes (F16, Q4_K, Q8_0)
# - Memory usage
# - Output quality comparison
#
# **Requirements:**
# - Kaggle secret `HF_TOKEN` (read access — models are public)
# - Internet ON
# - Any accelerator (CPU, T4, P100 — benchmark adapts)
# - ~30 GB disk
#
# Results are saved as a GitHub Gist via `GH_GIST_TOKEN` secret (optional).

# ─────────────────────────── cell 1 (code) ───────────────────────────
# ── Configuration ──────────────────────────────────────────────────────────
import os, sys, time, json, subprocess, shutil
from datetime import datetime
from pathlib import Path

WORK = "/kaggle/working"
# Clone and build under /kaggle/temp, NOT /kaggle/working. `kaggle kernels output`
# is page-capped at 500 files and does not auto-continue, so a repo checkout in
# the working dir buries everything that sorts after it — including the kernel
# log, which is the only place a failing backend's stderr appears. That is why
# kugelaudio's failure could not be read back after a 6-hour sweep. /kaggle/working
# should hold only artifacts worth retrieving.
SCRATCH = "/kaggle/temp" if os.path.isdir("/kaggle/temp") else "/tmp"
BUILD_DIR = f"{SCRATCH}/CrispASR/build"
CRISPASR = f"{BUILD_DIR}/bin/crispasr"
QUANTIZE = f"{BUILD_DIR}/bin/crispasr-quantize"
RESULTS_DIR = f"{WORK}/results"
SAMPLE_DIR = f"{WORK}/samples"

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(SAMPLE_DIR, exist_ok=True)

# GH gist token (optional, for publishing results). HF auth is resolved
# post-clone via the shared harness (kh.resolve_hf_token) so it gets the
# 3-tier env → Kaggle Secret(retry) → mounted-dataset fallback.
try:
    from kaggle_secrets import UserSecretsClient
    _secrets = UserSecretsClient()
    GH_GIST_TOKEN = _secrets.get_secret("GH_GIST_TOKEN") if "GH_GIST_TOKEN" in dir(_secrets) else None
except Exception:
    GH_GIST_TOKEN = os.environ.get("GH_GIST_TOKEN", "")

# Reference transcript for jfk.wav
JFK_REF = "and so my fellow americans ask not what your country can do for you ask what you can do for your country"

# All backends to test with their auto-download models
BACKENDS = [
    # (backend, display_name, timeout_seconds, notes)
    ("firered-asr",       "FireRed ASR2 AED",         90, "Q4_K, 900M params"),
    ("whisper",           "Whisper (base)",           60, "ggml-base.bin"),
    ("parakeet",          "Parakeet TDT 0.6B",       60, "Q4_K"),
    ("moonshine",         "Moonshine Tiny",           30, "Q4_K, 27M params"),
    ("moonshine-streaming","Moonshine Streaming Tiny",  30, "Q4_K, 34M params"),
    ("wav2vec2",          "Wav2Vec2 XLSR-EN",         60, "Q4_K, 300M params"),
    ("fastconformer-ctc", "FastConformer CTC Large",  30, "Q4_K, 120M params"),
    ("data2vec",          "Data2Vec Base",             30, "Q4_K, 95M params"),
    ("hubert",            "HuBERT Large",              60, "Q4_K, 300M params"),
    ("canary",            "Canary 1B",                120, "Q4_K, 1B params"),
    ("cohere",            "Cohere Transcribe",        120, "Q4_K, 2B params"),
    ("qwen3",             "Qwen3 ASR 0.6B",           60, "Q4_K"),
    ("omniasr",           "OmniASR CTC 1B v2",       120, "Q4_K, 975M params"),
    ("omniasr-llm",       "OmniASR LLM 300M",        120, "Q4_K, 300M+1.3B params"),
    ("glm-asr",           "GLM ASR Nano",             90, "Q4_K, 1.3B params"),
    ("kyutai-stt",        "Kyutai STT 1B",            90, "Q4_K, 1B params"),
    ("vibevoice",         "VibeVoice ASR",             90, "Q4_K, 4.5B params"),
    ("sensevoice",        "SenseVoice Small",          60, "Q4_K, ~129MB, encoder-only multitask"),
    ("paraformer",        "Paraformer-zh NAR",         60, "Q4_K, ~123MB, zh+en char-level"),
]

# Slow / large backends (only test if BENCHMARK_SLOW=1)
SLOW_BACKENDS = [
    ("voxtral",           "Voxtral Mini 3B",         300, "Q4_K, 3B params"),
    ("voxtral4b",         "Voxtral 4B Realtime",     300, "Q4_K, 4B params"),
    ("granite",           "Granite Speech 1B",       300, "Q4_K, 2.9B params"),
    ("gemma4-e2b",        "Gemma-4-E2B 2.3B",        300, "Q4_K, 2.3B params"),
    ("granite-4.1",       "Granite Speech 4.1 2B",   300, "Q4_K, ~2.94GB, LLM-AR"),
    ("mega-asr",          "Mega-ASR 1.7B",           120, "Q4_K, ~1.3GB, qwen3 backend + robustness LoRA"),
    ("funasr",            "Fun-ASR Nano 2512",       180, "Q8_0 (~1.06GB); F16 hits CUDA F16xF32 !-loop, Q8_0 is GPU-safe"),
    ("fun-asr-mlt-nano",  "Fun-ASR MLT Nano 2512",  240, "F16 (~1.98GB); multilingual multitask"),
    ("granite-4.1-plus",  "Granite Speech 4.1 2B+",  300, "Q4_K, ~2.96GB, LLM-AR plus variant"),
    ("granite-4.1-nar",   "Granite Speech 4.1 NAR",  300, "Q4_K, ~3.2GB, non-autoregressive"),
    ("mimo-asr",          "MiMo-ASR",                420, "Q4_K ~4.2GB; PLAN #115 forces CPU (~297s/11s clip)"),
    # ── previously-uncovered ASR backends (full-sweep coverage) ──
    ("nemotron",          "Nemotron Streaming",      180, "Q4_K, FastConformer streaming encoder"),
    ("moss-audio",        "MOSS Audio",              300, "Q4_K, Whisper enc + Qwen3 LLM"),
    ("lfm2-audio",        "LFM2-Audio 1.5B",         240, "Q5_K, hybrid conv+attn backbone"),
    ("mini-omni2",        "Mini-Omni2",              300, "Q4_K, multi-stream speech+chat"),
    # NOTE: vibevoice-1.5b is a TTS model (vibevoice-1.5b-tts-q4_k.gguf), not ASR
    # — moved to TTS_BACKENDS below (running it as ASR produced an empty transcript).

    # ── 2026-09-07 coverage pass ────────────────────────────────────────────
    # The sweep covered 44 of the 66 canonical backends. These are the missing
    # ASR ones that can resolve through `-m auto`, plus distinct MODELS that
    # share an existing runtime. That distinction matters here and nowhere
    # else: tools/check-backend-wiring.py calls raon/quds-fa "aliases" because
    # they reuse the f5 and parakeet RUNTIMES, which is the right answer for a
    # wiring audit and the wrong one for a model sweep — a different checkpoint
    # is a different thing to transcribe with, and covering f5-tts says nothing
    # about whether Raon's weights work.
    ("canary-qwen",       "Canary-Qwen 2.5B",        300, "Q4_K, canary encoder + Qwen LLM decoder"),
    ("higgs-stt",         "Higgs STT",               240, "Q4_K"),
    ("ark-asr",           "Ark-ASR",                 240, "Q4_K"),
    ("moss-transcribe",   "MOSS Transcribe",         300, "Q4_K, en-only (declared f33c398b)"),
    ("moss-diarize",      "MOSS Diarize",            300, "Q4_K, ASR + speaker turns"),
    # NOT ADDED — and the reason is a correction to my own first pass.
    # gigaam (ru), reazonspeech (ja) and quds-fa (fa) were added here on
    # 2026-09-07 and immediately produced FAIL / EMPTY / CRASH. Those verdicts
    # were meaningless: THIS SWEEP TRANSCRIBES ONE FIXED ENGLISH CLIP
    # (samples/jfk.wav, see JFK_WAV below), so a Russian model scoring WER 0.5
    # on English is the correct behaviour of a healthy model, reported as a
    # failure. A row that is red when the thing works is worse than no row.
    # They come back when the runner accepts a per-backend clip — samples/
    # already carries fleurs_ja_*.wav, paraformer_zh.wav and ko-369.wav, but
    # the loops below unpack a 4-tuple, so that is a real change, not a field.
    # (quds-fa additionally needs CRISPASR_ACCEPT_LICENSE: it is CC-BY-NC-4.0
    # and license_gate_allows_download() refuses before fetching a byte, which
    # is what the 0.27 s CRASH actually was.)
    ("omniasr-300m",      "OmniASR LLM 300M",        120, "Q4_K, smaller omniasr variant"),
    ("omniasr-llm-1b",    "OmniASR LLM 1B",          240, "Q4_K"),
    ("qwen3-1.7b",        "Qwen3 ASR 1.7B",          240, "Q4_K, larger qwen3 variant"),
]

# Backends that CANNOT go through `-m auto`: the registry has no auto-download
# entry for them (verified from `--list-backends-json` caps, not assumed), so a
# sweep row would fail on model resolution and read as a backend failure. They
# need an explicit `-m <path>` and are listed here so the omission is a recorded
# decision rather than an oversight.
NO_AUTODOWNLOAD = ["sidon", "miotts", "bananamind-tts", "omnivoice"]

# NOTE 2026-09-07: two ids here named backends the binary does not expose, so
# those rows could never have succeeded — `cosyvoice3` (real id
# `cosyvoice3-tts`) and `vibevoice-tts` (no such backend; `vibevoice-1.5b` was
# already covered separately, so that slot now points at the uncovered
# `vibevoice-bitnet`). Found by validating every id in these lists against
# `--list-backends-json` before pushing, which is worth doing every time: a
# wrong id fails on the worker and reads exactly like a broken backend.
#
# TTS backends suitable for Kaggle time limits (small/fast models first,
# then larger ones). Each entry downloads via -m auto, synthesises a phrase,
# checks output WAV exists and has >1000 bytes. Models are cleaned up after
# each backend to stay within Kaggle's ~20 GB scratch.
TTS_BACKENDS = [
    # (backend, display_name, timeout_seconds, notes)
    ("piper",             "Piper LessAC Medium",      30, "F16, ~16MB, VITS en_US"),
    ("kokoro",            "Kokoro 82M",               90, "Q8_0, needs espeak-ng + voice GGUF"),
    ("speecht5",          "SpeechT5 TTS",             90, "F16, ~300MB, encoder-decoder + HiFi-GAN"),
    ("fastpitch",         "FastPitch 60M",            30, "F16, ~60MB, non-AR parallel TTS"),
    ("pocket-tts",        "Pocket TTS 100M",          90, "F16, ~381MB, continuous-latent AR"),
    ("bark",              "Bark Small",              120, "Q8_0, ~500MB, 3-stage GPT-2"),
    ("f5-tts",            "F5-TTS v1 Base",          120, "F16, ~953MB, DiT flow-matching"),
    ("csm",               "CSM 1B",                  240, "Q4_K, ~1.1GB, conversational TTS"),
    ("parler-tts",        "Parler TTS Mini v1.1",    180, "Q8_0, ~1GB, T5 + MusicGen decoder"),
    ("dia",               "Dia 1.6B",                240, "Q8_0, ~1.6GB, byte-level + DAC 44.1 kHz"),
    ("orpheus",           "Orpheus 3B-FT",           300, "Q8_0, ~3.5GB, Llama-3.2 + SNAC"),
    # ── previously-uncovered TTS backends (full-sweep coverage) ──
    ("qwen3-tts-customvoice", "Qwen3-TTS CustomVoice", 300, "Q8_0, talker+12Hz codec, built-in speakers"),
    ("vibevoice-bitnet",     "VibeVoice TTS",           300, "Q4_K, diffusion TTS"),
    ("chatterbox",        "Chatterbox",              300, "Q4_K, T3 + S3Gen voice clone"),
    ("cosyvoice3-tts",        "CosyVoice3",              300, "Q4_K, flow-matching + HiFT"),
    ("indextts",          "IndexTTS",                300, "Q4_K, GPT + BigVGAN vocoder"),
    ("zonos",             "Zonos",                   240, "F16, transformer TTS (EOS-sensitive on Q4_K)"),
    ("melotts",           "MeloTTS",                  90, "F16, VITS multilingual"),
    ("outetts",           "OuteTTS",                 180, "Q8_0, Llama + WavTokenizer"),
    ("tada",              "TADA 3B",                 300, "Q4_K, multilingual AR TTS"),
    ("voxcpm2-tts",       "VoxCPM2 TTS",             300, "F16, VAE encoder + LLM"),
    ("vibevoice-1.5b",    "VibeVoice 1.5B TTS",      300, "Q4_K ~1.6GB; was mis-listed as ASR"),
    # The 900 s budget bought the answer: not slow, OOM. `-m auto` used to get
    # the F16 (17.3 GB, more VRAM than any Kaggle GPU has), so no timeout could
    # ever have made it pass. The registry default is Q4_K now, and this entry
    # deliberately does NOT pin the quant — the sweep should exercise the path a
    # user actually gets from `-m auto`.
    ("kugelaudio",        "KugelAudio",              900, "Q4_K ~5.7GB via -m auto (F16 17.3GB needs >16GB VRAM)"),
    # ── 2026-09-07 coverage pass ────────────────────────────────────────────
    # Missing canonical TTS backends that resolve via `-m auto`, plus distinct
    # MODELS on shared runtimes. Every one of these is exercised the same way
    # the rest are: synthesise a phrase, then feed the audio back through
    # parakeet ASR and score the word overlap — a TTS backend that emits a
    # valid-looking WAV of noise fails that, where a file-size check passes it.
    ("moss-tts",          "MOSS TTS",                300, "Q4_K, voice-cloning"),
    ("moss-tts-local",    "MOSS TTS (local voice)",  300, "Q4_K, voice-cloning"),
    ("dots-tts",          "Dots TTS",                240, "Q4_K"),
    ("voxtral-tts",       "Voxtral TTS",             300, "Q4_K"),
    ("qwen3-tts",         "Qwen3-TTS 0.6B",          240, "Q4_K, streaming + voice-cloning"),
    ("raon",              "Raon-OpenTTS 0.3B",       300, "F16 ~959MB, CC-BY-NC; f5 runtime, ggml HiFi-GAN"),
    ("confucius4-tts",    "Confucius4 TTS",          300, "Q4_K, zero-shot"),
    ("chatterbox-nano",   "Chatterbox Nano",         180, "Q4_K, small T3 + Turbo S3Gen"),
    ("chatterbox-turbo",  "Chatterbox Turbo",        240, "Q4_K"),
    ("pocket-tts-de",     "Pocket-TTS German",       120, "Q8_0 ~124MB distilled 6L"),
    ("pocket-tts-fr",     "Pocket-TTS French",       180, "Q8_0 ~365MB undistilled 24L preview"),
    ("tada-1b",           "TADA TTS 1B",             300, "Q4_K"),
    ("cosyvoice3-tts-rl", "CosyVoice3 RL",           300, "Q4_K, RL-tuned variant"),
]

# Text MT backends (translate a sentence; not ASR/TTS but part of the backend
# set). Tested only when BENCHMARK_MT=1 since they need text in/out, not audio.
MT_BACKENDS = [
    ("m2m100",            "M2M-100 418M",            120, "Q8_0, multilingual MT"),
    ("madlad",            "MADLAD-400 3B",           240, "Q4_K, 400-lang MT"),
]

print(f"CrispASR Benchmark — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
print(f"Backends: {len(BACKENDS)} fast + {len(SLOW_BACKENDS)} slow ASR, {len(TTS_BACKENDS)} TTS")

# ─────────────────────────── cell 2 (code) ───────────────────────────
# ── Install dependencies ───────────────────────────────────────────────────
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "huggingface_hub", "hf_transfer", "jiwer"], check=True)
print("✓ Dependencies installed")

# ─────────────────────────── cell 3 (code) ───────────────────────────
# ── Clone and build CrispASR ───────────────────────────────────────────────
CRISPASR_DIR = f"{SCRATCH}/CrispASR"

def run(cmd, timeout=600, stream_stderr=False):
    """Run shell command, return (success, stdout, stderr, elapsed).
    If stream_stderr=True, print stderr lines in real-time (for debugging hangs).
    """
    t0 = time.time()
    if stream_stderr:
        try:
            proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE, text=True)
            stderr_lines = []
            import select
            while True:
                elapsed_so_far = time.time() - t0
                if elapsed_so_far > timeout:
                    proc.kill()
                    return False, "", "TIMEOUT", elapsed_so_far
                # Read stderr line by line in real-time
                ready, _, _ = select.select([proc.stderr], [], [], 1.0)
                if ready:
                    line = proc.stderr.readline()
                    if line:
                        stderr_lines.append(line)
                        print(f"    [live] {line.rstrip()}", flush=True)
                if proc.poll() is not None:
                    # Process finished — drain remaining
                    for line in proc.stderr:
                        stderr_lines.append(line)
                        print(f"    [live] {line.rstrip()}", flush=True)
                    break
            stdout = proc.stdout.read()
            elapsed = time.time() - t0
            return proc.returncode == 0, stdout, "".join(stderr_lines), elapsed
        except Exception:
            return False, "", "TIMEOUT", time.time() - t0
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True,
                           timeout=timeout)
        elapsed = time.time() - t0
        return r.returncode == 0, r.stdout, r.stderr, elapsed
    except subprocess.TimeoutExpired:
        return False, "", "TIMEOUT", time.time() - t0

if os.path.isdir(CRISPASR_DIR):
    # Pull latest to pick up fixes pushed since initial clone
    subprocess.run(f"cd {CRISPASR_DIR} && git fetch --depth 1 origin main "
                   f"&& git reset --hard origin/main "
                   f"&& git submodule update --init --recursive --depth 1",
                   shell=True, check=True, capture_output=True)
    print("✓ CrispASR updated to latest (submodules synced)")
else:
    # --recurse-submodules is not optional: ggml is a submodule, and without it
    # cmake stops at CMakeLists.txt:256 with "The bundled 'ggml' submodule is not
    # initialized" (#298) before a single backend runs.
    subprocess.run(f"git clone --depth 1 --recurse-submodules --shallow-submodules "
                   f"https://github.com/CrispStrobe/CrispASR.git {CRISPASR_DIR}",
                   shell=True, check=True)
    print("✓ CrispASR cloned (with submodules)")

# Shared Kaggle harness (lives in the cloned repo) — build streaming,
# heartbeat+RSS, ccache/mold, CUDA arch auto-detect, 3-tier HF auth.
sys.path.insert(0, os.path.join(CRISPASR_DIR, "tools", "kaggle"))
import kaggle_harness as kh  # noqa: E402
kh.init_progress(progress_path=f"{WORK}/progress.jsonl")
kh.resolve_hf_token()  # env → Kaggle Secret(retry) → mounted dataset
kh.step("clone.done")

# ── Incremental result streaming to an HF dataset (resumable) ────────────────
# After EACH backend we upload its result JSON to an HF dataset so interim
# results survive a kernel crash/timeout, and on a fresh run of the SAME kernel
# we SKIP backends that already have a result file (resume). The run tag is
# stable across restarts by default ("latest"), so re-running the kernel
# continues where it left off; bump CRISPASR_SWEEP_RUN for a clean run.
from huggingface_hub import HfApi as _HfApi
import io as _io

# Target the EXISTING progress dataset (same one kaggle_harness streams step
# progress to), under a sweep-specific prefix so it doesn't collide with the
# harness's runs/*.jsonl.
SWEEP_REPO = os.environ.get("CRISPASR_SWEEP_REPO", "cstr/crispasr-kaggle-progress")
# Bump this per campaign — results are keyed by tag and a matching tag makes the
# kernel SKIP backends that already have a result file. "latest" currently holds
# 60 finished backends from an earlier pin, so reusing it would skip nearly the
# whole sweep and say nothing about the tree under test.
# 2026-09-07: bumped v0.17b -> v0.23. The tree pins ggml 2dd13edd (v0.23.0,
# CrispStrobe/ggml#3), so a run tagged v0.17b would have labelled the results
# with a ggml version that is no longer in the build — and, because a matching
# tag makes the kernel SKIP finished backends, would also have resumed a sweep
# taken before the bump. Both halves of that are wrong: stale label, stale data.
RUN_TAG = os.environ.get("CRISPASR_SWEEP_RUN", "ggml-v0.23")
SWEEP_PREFIX = f"full-backend-sweep/{RUN_TAG}"

# The kernel git-clones CrispASR at runtime, so the C++ under test is always
# fresh from main while THIS SCRIPT is frozen at the last `kaggle kernels push`.
# A run can therefore build the newest code and score it with an outdated
# harness, and nothing in the log looks wrong. Print both halves so the log
# always says which script and which tree produced a verdict.
SWEEP_SCRIPT_VERSION = "2026-09-07-coverage-83-licensed"

# ── Restricted-licence acceptance (owner-authorised 2026-09-07) ────────────
# Ten backends failed the previous sweep in ~0.1 s with 0 bytes written:
# pocket-tts (+ -de/-fr), orpheus, outetts, tada, tada-1b, voxtral-tts, raon
# and quds-fa. None of them are broken. license_gate_allows_download() refuses
# a restricted model BEFORE fetching a byte unless the licence is accepted, so
# every one of those rows was the gate working, recorded as a failure.
#
# WHAT THIS ACCEPTS, stated exactly, because "all" is broader than the
# non-commercial set it is usually described as. license_requires_acceptance_tag()
# gates: cc-by-nc*, cc-by-sa*, llama*, gemma, gemma-terms, qwen-research,
# mistral-ai-research, lfm1.0, lfm-open-1.0, funasr-v1.1, pocket-tts-terms, and
# the catch-all "other". The registry currently carries two distinct NC tags
# (cc-by-nc-4.0 and cc-by-nc-sa-4.0) and license_accepted() matches an EXACT tag
# or the wildcard — so a single specific value cannot cover the set, and "all"
# is the only setting that unblocks the sweep in one variable.
#
# This is an internal benchmark: models are downloaded, synthesised from once,
# scored, and deleted (_cleanup_cache below). Nothing is redistributed. That is
# within CC-BY-NC terms and within the research/terms licences above, and the
# repository already publishes GGUF conversions of these same checkpoints.
# The acceptance is the owner's, given explicitly for this sweep; it is set here
# rather than in any library path so it cannot leak into normal CLI use.
os.environ.setdefault("CRISPASR_ACCEPT_LICENSE", "all")
def _print_provenance():
    import subprocess as _sp
    sha = "unknown"
    try:
        sha = _sp.check_output(["git", "-C", CRISPASR_DIR, "rev-parse", "--short", "HEAD"],
                               text=True, stderr=_sp.DEVNULL).strip()
    except Exception:
        pass
    print(f"[sweep] script_version={SWEEP_SCRIPT_VERSION}  crispasr_clone={sha}  run_tag={RUN_TAG}",
          flush=True)

_print_provenance()
# Optional subset filter: CRISPASR_SWEEP_ONLY="f5-tts,chatterbox,..." runs ONLY
# those backends (skips all others) — for targeted re-tests of a fixed subset.
SWEEP_ONLY = {x.strip() for x in os.environ.get("CRISPASR_SWEEP_ONLY", "").split(",") if x.strip()}


def sweep_skip_only(backend):
    return bool(SWEEP_ONLY) and backend not in SWEEP_ONLY
_hf_token = (os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
             or os.environ.get("HUGGINGFACE_TOKEN"))
_hf_api = _HfApi(token=_hf_token)
_sweep_ok = False
_done_keys = set()


def _sweep_key(category, backend):
    return f"{category}__{backend}".replace("/", "-")


# Up-front WRITE test: actually push a heartbeat file. whoami() / create_repo
# don't reliably exercise write perms on an existing repo, so do a real upload —
# a bad/expired token then fails LOUDLY here, not silently after the ~78-min
# sweep (the 2026-06-20 run wasted a full pass because the attached token 401'd).
try:
    if not _hf_token:
        raise RuntimeError("no HF token in env (HF_TOKEN) — resolve_hf_token must run first")
    _who = _hf_api.whoami().get("name", "?")
    _hf_api.upload_file(path_or_fileobj=_io.BytesIO(b"alive\n"),
                        path_in_repo=f"{SWEEP_PREFIX}/_heartbeat.txt",
                        repo_type="dataset", repo_id=SWEEP_REPO,
                        commit_message="sweep heartbeat (token write-test)")
    for f in _hf_api.list_repo_files(SWEEP_REPO, repo_type="dataset"):
        if f.startswith(f"{SWEEP_PREFIX}/results/") and f.endswith(".json"):
            _done_keys.add(f.split("/")[-1][:-5])
    _sweep_ok = True
    print(f"[sweep] HF streaming OK → {SWEEP_REPO}/{SWEEP_PREFIX} (token user={_who}); "
          f"{len(_done_keys)} backend(s) already done — resume/skip")
except Exception as e:
    print("=" * 72)
    print(f"[sweep] !! HF STREAMING DISABLED — token WRITE-TEST FAILED: {e!r}")
    print(f"[sweep] !! results will stay LOCAL ONLY this run. Fix the HF token in the")
    print(f"[sweep] !! attached dataset: hf_token.txt needs WRITE access to {SWEEP_REPO}.")
    print("=" * 72)


def sweep_done(category, backend):
    """True if this backend already has a streamed result for this run (resume)."""
    return _sweep_key(category, backend) in _done_keys


def sweep_publish(category, result):
    """Upload one backend's result JSON to the HF dataset (interim, resumable)."""
    if not _sweep_ok or not result:
        return
    key = _sweep_key(category, result.get("backend", "unknown"))
    payload = dict(result)
    payload["_category"] = category
    payload["_run"] = RUN_TAG
    data = json.dumps(payload, indent=2, default=str).encode()
    try:
        _hf_api.upload_file(path_or_fileobj=_io.BytesIO(data),
                            path_in_repo=f"{SWEEP_PREFIX}/results/{key}.json",
                            repo_type="dataset", repo_id=SWEEP_REPO,
                            commit_message=f"sweep {RUN_TAG}: {key}")
        _done_keys.add(key)
        print(f"  [sweep] ↑ streamed → {SWEEP_REPO}/{SWEEP_PREFIX}/results/{key}.json")
    except Exception as e:
        print(f"  [sweep] ! upload failed for {key}: {e!r}")

# Detect GPU — try CUDA first, fall back to CPU if cmake fails
has_gpu = os.path.exists("/usr/local/cuda/bin/nvcc")
# Incremental build: keep build dir if cmake config matches, only rebuild changed source.
# Force reconfigure only if cmake flags would change (e.g. GPU detection differs).
os.makedirs(BUILD_DIR, exist_ok=True)
need_reconfigure = not os.path.isfile(f"{BUILD_DIR}/CMakeCache.txt")

# Install ninja + ccache + mold via the harness (primes the persistent
# ccache at /kaggle/working/.ccache so re-run builds are near-free).
_tc = kh.install_build_toolchain()
has_ninja = _tc["ninja"]
generator = ["-G", "Ninja"] if has_ninja else []

# Common cmake flags (match Docker/dev-build.sh patterns)
common_flags = [
    "-DCMAKE_BUILD_TYPE=Release",
    "-DCRISPASR_BUILD_TESTS=OFF",  # skip test binaries — saves ~30% build time
    "-DCMAKE_C_FLAGS=-fopenmp",
    "-DCMAKE_CXX_FLAGS=-fopenmp",
]

cmake_ok = not need_reconfigure  # skip configure if cache exists
if has_gpu and need_reconfigure:
    # CUDA build. kh.cuda_build_flags auto-detects the GPU's compute
    # capability (T4→75, P100→60, A100→80, L4→89) and pins it — without
    # the pin, ggml builds every kernel for its full default arch list,
    # which multiplies nvcc RAM+time and OOM'd the ~16 GB box ~19 kernels
    # into ggml-cuda (2026-05-31). Plus ccache launchers + mold.
    arch = kh.detect_cuda_arch()
    kh.step("cuda.arch", arch=arch)
    print(f"GPU: CUDA detected (sm_{arch}), attempting CUDA build...")
    cuda_flags = kh.cuda_build_flags(arch) + kh.cache_and_link_flags()
    with kh.build_heartbeat("cmake.configure.cuda"):
        r = subprocess.run(
            ["cmake", "-S", CRISPASR_DIR, "-B", BUILD_DIR] + generator + common_flags + cuda_flags,
            capture_output=True, text=True
        )
    if r.returncode == 0:
        cmake_ok = True
        print(f"✓ CUDA cmake configured ({'Ninja' if has_ninja else 'Make'})")
    else:
        print("⚠ CUDA cmake failed, falling back to CPU build")
        print((r.stdout or "")[-2000:]); print((r.stderr or "")[-2000:])
        shutil.rmtree(BUILD_DIR, ignore_errors=True)
        os.makedirs(BUILD_DIR, exist_ok=True)
        has_gpu = False

if not cmake_ok and need_reconfigure:
    print("GPU: CPU-only build")
    with kh.build_heartbeat("cmake.configure.cpu"):
        subprocess.run(
            ["cmake", "-S", CRISPASR_DIR, "-B", BUILD_DIR] + generator + common_flags + [
                "-DGGML_CUDA=OFF",
            ] + kh.cache_and_link_flags(),
            check=True
        )
elif cmake_ok and not need_reconfigure:
    print(f"✓ Using cached cmake config ({'GPU' if has_gpu else 'CPU'})")

# Build only the main binary (not quantize, test tools, etc.). Stream the
# build through the harness (sh_with_progress + heartbeat) so ninja [X/N],
# the current TU, and RSS are visible live — a silent subprocess.run here
# is exactly why the first OOM was undiagnosable. CUDA nvcc is RAM-heavy,
# so cap CUDA at -j2 (still parallel, fits memory); CPU keeps full -j.
build_jobs = kh.safe_build_jobs(gpu=has_gpu)
with kh.build_heartbeat("cmake.build"):
    # Target is `crispasr-cli` — it has OUTPUT_NAME crispasr, so it produces
    # bin/crispasr. Target `crispasr` builds ONLY the library (libcrispasr),
    # leaving bin/crispasr absent — which failed the assert below on every
    # prior run (examples/cli/CMakeLists.txt:12,232).
    kh.sh_with_progress(f"stdbuf -oL -eL cmake --build {BUILD_DIR} "
                        f"--target crispasr-cli -j{build_jobs}")

assert os.path.isfile(CRISPASR), f"Build failed: {CRISPASR} not found"

# Show version + git commit
ok, out, _, _ = run(f"cd {CRISPASR_DIR} && git log --oneline -1")
git_hash = out.strip() if ok else "unknown"
print(f"✓ CrispASR built ({'GPU' if has_gpu else 'CPU'}) — {git_hash}")

# ─────────────────────────── cell 4a (code) ───────────────────────────
# ── Model download helper (handles HF xet storage) ────────────────────────
# CrispASR's built-in curl downloader can't handle HF's xet storage layer.
# This helper downloads one model at a time via huggingface_hub before each test.
from huggingface_hub import hf_hub_download

CACHE_DIR = os.path.expanduser("~/.cache/crispasr")
os.makedirs(CACHE_DIR, exist_ok=True)

# Pre-download shared models (whisper-tiny for LID, silero for VAD) — kept across tests
for local_name, repo, hf_name in [
    ("ggml-tiny.bin", "ggerganov/crispasr", "ggml-tiny.bin"),
    ("ggml-silero-v6.2.0.bin", "ggml-org/whisper-vad", "ggml-silero-v6.2.0.bin"),
]:
    dst = os.path.join(CACHE_DIR, local_name)
    if not os.path.isfile(dst):
        try:
            hf_hub_download(repo, hf_name, local_dir=CACHE_DIR)
            print(f"  ✓ {local_name}")
        except Exception as e:
            print(f"  ✗ {local_name}: {e}")

# Per-backend model registry: backend_name → (local_filename, hf_repo, hf_filename)
MODEL_REGISTRY = {
    "whisper":           ("ggml-base.bin",               "ggerganov/crispasr",                    "ggml-base.bin"),
    "parakeet":          ("parakeet-tdt-0.6b-v3-q4_k.gguf","cstr/parakeet-tdt-0.6b-v3-GGUF",       "parakeet-tdt-0.6b-v3-q4_k.gguf"),
    "moonshine":         ("moonshine-tiny-q4_k.gguf",    "cstr/moonshine-tiny-GGUF",                "moonshine-tiny-q4_k.gguf"),
    "moonshine-streaming":("moonshine-streaming-tiny-q4_k.gguf","cstr/moonshine-streaming-tiny-GGUF","moonshine-streaming-tiny-q4_k.gguf"),
    "wav2vec2":          ("wav2vec2-xlsr-en-q4_k.gguf",  "cstr/wav2vec2-large-xlsr-53-english-GGUF","wav2vec2-xlsr-en-q4_k.gguf"),
    "fastconformer-ctc": ("stt-en-fastconformer-ctc-large-q4_k.gguf","cstr/stt-en-fastconformer-ctc-large-GGUF","stt-en-fastconformer-ctc-large-q4_k.gguf"),
    "data2vec":          ("data2vec-audio-base-960h-q4_k.gguf","cstr/data2vec-audio-960h-GGUF",     "data2vec-audio-base-960h-q4_k.gguf"),
    "hubert":            ("hubert-large-ls960-ft-q4_k.gguf","cstr/hubert-large-ls960-ft-GGUF",      "hubert-large-ls960-ft-q4_k.gguf"),
    "canary":            ("canary-1b-v2-q4_k.gguf",      "cstr/canary-1b-v2-GGUF",                 "canary-1b-v2-q4_k.gguf"),
    "cohere":            ("cohere-transcribe-q4_k.gguf",  "cstr/cohere-transcribe-03-2026-GGUF",    "cohere-transcribe-q4_k.gguf"),
    "qwen3":             ("qwen3-asr-0.6b-q4_k.gguf",    "cstr/qwen3-asr-0.6b-GGUF",              "qwen3-asr-0.6b-q4_k.gguf"),
    "omniasr":           ("omniasr-ctc-1b-v2-q4_k.gguf",  "cstr/omniASR-CTC-1B-v2-GGUF",          "omniasr-ctc-1b-v2-q4_k.gguf"),
    "omniasr-llm":       ("omniasr-llm-300m-v2-q4_k.gguf","cstr/omniasr-llm-300m-v2-GGUF",         "omniasr-llm-300m-v2-q4_k.gguf"),
    "glm-asr":           ("glm-asr-nano-q4_k.gguf",      "cstr/glm-asr-nano-GGUF",                "glm-asr-nano-q4_k.gguf"),
    "firered-asr":       ("firered-asr2-aed-q4_k.gguf",  "cstr/firered-asr2-aed-GGUF",            "firered-asr2-aed-q4_k.gguf"),
    "kyutai-stt":        ("kyutai-stt-1b-q4_k.gguf",     "cstr/kyutai-stt-1b-GGUF",               "kyutai-stt-1b-q4_k.gguf"),
    "vibevoice":         ("vibevoice-asr-q4_k.gguf",     "cstr/vibevoice-asr-GGUF",                "vibevoice-asr-q4_k.gguf"),
    "voxtral":           ("voxtral-mini-3b-2507-q4_k.gguf","cstr/voxtral-mini-3b-2507-GGUF",       "voxtral-mini-3b-2507-q4_k.gguf"),
    "voxtral4b":         ("voxtral-mini-4b-realtime-q4_k.gguf","cstr/voxtral-mini-4b-realtime-GGUF","voxtral-mini-4b-realtime-q4_k.gguf"),
    "granite":           ("granite-speech-4.0-1b-q4_k.gguf","cstr/granite-speech-4.0-1b-GGUF",     "granite-speech-4.0-1b-q4_k.gguf"),
    "gemma4-e2b":        ("gemma4-e2b-it-q4_k.gguf",      "cstr/gemma4-e2b-it-GGUF",              "gemma4-e2b-it-q4_k.gguf"),
    "sensevoice":        ("sensevoice-small-q4_k.gguf",  "cstr/sensevoice-small-GGUF",            "sensevoice-small-q4_k.gguf"),
    "paraformer":        ("paraformer-zh-q4_k.gguf",     "cstr/paraformer-zh-GGUF",               "paraformer-zh-q4_k.gguf"),
    "granite-4.1":       ("granite-speech-4.1-2b-q4_k.gguf","cstr/granite-speech-4.1-2b-GGUF",     "granite-speech-4.1-2b-q4_k.gguf"),
    "mega-asr":          ("mega-asr-1.7b-q4_k.gguf",     "cstr/mega-asr-GGUF",                    "mega-asr-1.7b-q4_k.gguf"),
    "funasr":            ("funasr-nano-2512-q8_0.gguf",  "cstr/funasr-nano-GGUF",                 "funasr-nano-2512-q8_0.gguf"),
    "mimo-asr":          ("mimo-asr-q4_k.gguf",          "cstr/mimo-asr-GGUF",                    "mimo-asr-q4_k.gguf"),
}

# mimo-asr needs a companion tokenizer GGUF alongside the main model; the
# C++ --auto-download path fetches it, but pre-pull it here so a flaky
# companion download doesn't fail the whole backend mid-run.
_mimo_tok = os.path.join(CACHE_DIR, "mimo-tokenizer-q4_k.gguf")
if not os.path.isfile(_mimo_tok):
    try:
        hf_hub_download("cstr/mimo-tokenizer-GGUF", "mimo-tokenizer-q4_k.gguf", local_dir=CACHE_DIR)
    except Exception:
        pass

# Also download moonshine tokenizer
_tok_dst = os.path.join(CACHE_DIR, "tokenizer.bin")
if not os.path.isfile(_tok_dst):
    try:
        hf_hub_download("cstr/moonshine-tiny-GGUF", "tokenizer.bin", local_dir=CACHE_DIR)
    except Exception:
        pass

def ensure_model_downloaded(backend_name):
    """Download model for backend via huggingface_hub (handles xet). Returns download time."""
    if backend_name not in MODEL_REGISTRY:
        return 0
    local_name, repo, hf_name = MODEL_REGISTRY[backend_name]
    dst = os.path.join(CACHE_DIR, local_name)
    if os.path.isfile(dst):
        return 0
    t0 = time.time()
    try:
        hf_hub_download(repo, hf_name, local_dir=CACHE_DIR)
        elapsed = time.time() - t0
        sz = os.path.getsize(dst) / 1024 / 1024 if os.path.isfile(dst) else 0
        print(f"    Downloaded {local_name} ({sz:.0f} MB, {elapsed:.1f}s)")
        return elapsed
    except Exception as e:
        print(f"    ✗ Download failed: {e}")
        return time.time() - t0

# ─────────────────────────── cell 4b (code) ───────────────────────────
# ── Download test audio ────────────────────────────────────────────────────
JFK_WAV = f"{SAMPLE_DIR}/jfk.wav"
if not os.path.isfile(JFK_WAV):
    shutil.copy(f"{CRISPASR_DIR}/samples/jfk.wav", JFK_WAV)
    print(f"✓ jfk.wav copied ({os.path.getsize(JFK_WAV)} bytes)")

# Get audio duration
import wave
with wave.open(JFK_WAV) as wf:
    AUDIO_DURATION = wf.getnframes() / wf.getframerate()
print(f"  Duration: {AUDIO_DURATION:.1f}s")

# ─────────────────────────── cell 5 (code) ───────────────────────────
# ── Helper: compute WER ───────────────────────────────────────────────────
from jiwer import wer as compute_wer
import re

def normalize_text(text):
    """Normalize text for WER: lowercase, remove punctuation, collapse spaces."""
    text = text.lower().strip()
    text = re.sub(r"[^a-z ]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def calc_wer(ref, hyp):
    """Calculate word error rate between reference and hypothesis."""
    ref_norm = normalize_text(ref)
    hyp_norm = normalize_text(hyp)
    if not ref_norm or not hyp_norm:
        return 1.0
    return compute_wer(ref_norm, hyp_norm)

# ─────────────────────────── cell 6 (code) ───────────────────────────
# ── Run benchmark for each backend ─────────────────────────────────────────
import resource

results = []

def _cleanup_cache(backend_name):
    """Remove downloaded model files from cache, keeping whisper-tiny and silero VAD."""
    cache_dir = os.path.expanduser("~/.cache/crispasr")
    if not os.path.isdir(cache_dir):
        return
    freed = 0
    for f in os.listdir(cache_dir):
        fpath = os.path.join(cache_dir, f)
        if not os.path.isfile(fpath):
            continue
        # Keep whisper tiny (LID) and silero VAD — shared across backends
        if "ggml-tiny" in f or "silero" in f or "tokenizer" in f:
            continue
        sz = os.path.getsize(fpath) / 1024 / 1024
        os.remove(fpath)
        freed += sz
    if freed > 0:
        print(f"    Cleaned cache: {freed:.0f} MB freed")

def benchmark_backend(backend, display_name, timeout, notes):
    """Run a single backend benchmark. Returns result dict."""
    print(f"\n{'='*60}")
    print(f"  {display_name} (--backend {backend})")
    print(f"{'='*60}")

    result = {
        "backend": backend,
        "display_name": display_name,
        "notes": notes,
        "status": "UNKNOWN",
        "transcript": "",
        "wer": None,
        "wall_s": None,
        "inference_s": None,
        "elapsed_s": None,
        "realtime_factor": None,
        "model_size_mb": None,
    }

    # Download model via huggingface_hub (handles xet storage) BEFORE timing
    dl_time = ensure_model_downloaded(backend)
    result["download_s"] = round(dl_time, 1)

    # Run inference — model is now cached in ~/.cache/crispasr
    # Stream stderr in real-time for all backends to diagnose hangs
    # Use greedy decoding (--beam 1) for benchmark: beam search is 5x slower
    # and doesn't meaningfully change WER on short test audio.
    cmd = (f"CRISPASR_VERBOSE=1 FIRERED_BENCH=1 {CRISPASR} --backend {backend} -m auto --auto-download "
           f"-f {JFK_WAV} --no-prints -v -bs 1")
    t0 = time.time()
    ok, stdout, stderr, elapsed = run(cmd, timeout=timeout, stream_stderr=True)
    result["wall_s"] = round(elapsed, 2)

    # Parse crispasr's own timing from stderr (excludes download)
    inference_s = elapsed  # fallback: use wall time
    rt_factor = None
    if stderr:
        m_time = re.search(r"transcribed\s+[\d.]+s\s+audio\s+in\s+([\d.]+)s\s+\(([\d.]+)x", stderr)
        if m_time:
            inference_s = float(m_time.group(1))
            rt_factor = float(m_time.group(2))
    result["inference_s"] = round(inference_s, 2)
    result["elapsed_s"] = round(inference_s, 2)
    result["realtime_factor"] = round(rt_factor if rt_factor else AUDIO_DURATION / inference_s, 2) if inference_s > 0 else 0

    if not ok:
        result["status"] = "TIMEOUT" if "TIMEOUT" in stderr else "CRASH"
        if stderr:
            result["stderr_tail"] = stderr[-1500:]
        print(f"  ✗ {result['status']} after {elapsed:.1f}s  (wall)")
        # Show useful stderr lines (skip download progress bars)
        if stderr:
            lines = stderr.strip().split("\n")
            useful = [l for l in lines if any(k in l.lower() for k in
                      ["assert", "error", "fail", "abort", "fatal", "ggml_",
                       "crispasr", "backend", "loaded", "encoder", "decoder",
                       "model", "cache", "kv", "segfault", "signal"])]
            for line in (useful or lines[-5:])[:8]:
                print(f"    stderr: {line[:150]}")
        # Still clean up any downloaded model to free disk
        _cleanup_cache(backend)
        return result

    # Step 2: Parse transcript
    transcript = stdout.strip()
    # Remove timestamp prefixes if present (e.g. [00:00:00.000 --> 00:00:11.000])
    transcript = re.sub(r"\[[\d:.]+\s*-->\s*[\d:.]+\]\s*", "", transcript).strip()
    result["transcript"] = transcript

    if not transcript:
        result["status"] = "EMPTY"
        print(f"  ✗ Empty output after {elapsed:.1f}s")
        _cleanup_cache(backend)
        return result

    # Step 3: Compute WER
    w = calc_wer(JFK_REF, transcript)
    result["wer"] = round(w, 4)
    result["status"] = "PASS" if w < 0.3 else "DEGRADED" if w < 0.5 else "FAIL"

    # Step 4: Find model size before cleanup
    cache_dir = os.path.expanduser("~/.cache/crispasr")
    if os.path.isdir(cache_dir):
        for f in sorted(os.listdir(cache_dir)):
            fpath = os.path.join(cache_dir, f)
            if os.path.isfile(fpath) and (f.endswith(".gguf") or f.endswith(".bin")):
                if "ggml-tiny" not in f and "silero" not in f:
                    result["model_size_mb"] = round(os.path.getsize(fpath) / 1024 / 1024, 1)
                    break

    status_icon = {"PASS": "✓", "DEGRADED": "~", "FAIL": "✗"}.get(result["status"], "?")
    sz_str = f"{result['model_size_mb']:.0f}MB" if result["model_size_mb"] else "?"
    wall_s = result.get("wall_s", elapsed)
    dl_note = f"  (wall={wall_s:.1f}s incl. download)" if abs(wall_s - inference_s) > 1 else ""
    print(f"  {status_icon} WER={w:.1%}  RT={result['realtime_factor']:.1f}x  "
          f"Inference={inference_s:.1f}s  Model={sz_str}{dl_note}")
    print(f"    Output: {transcript[:100]}")
    # Show key diagnostic lines from verbose stderr
    if stderr:
        for line in stderr.strip().split("\n"):
            if any(k in line for k in ["backend", "loaded", "encoder", "decoder",
                                        "GPU", "CUDA", "Metal", "cache", "kv", "transcribed"]):
                print(f"    diag: {line[:130]}")

    # Step 5: Clean up model to free disk for the next backend
    _cleanup_cache(backend)

    return result

# Run all fast backends
for backend, name, timeout, notes in BACKENDS:
    if sweep_skip_only(backend):
        continue
    if sweep_done("asr", backend):
        print(f"⏭ {name} ({backend}): already streamed for run '{RUN_TAG}' — resume skip")
        continue
    r = benchmark_backend(backend, name, timeout, notes)
    results.append(r)
    sweep_publish("asr", r)  # stream interim result immediately (resumable)

# Optionally run slow backends
# Run slow backends if requested (voxtral 3B/4B, granite — need more time)
BENCHMARK_SLOW = os.environ.get("BENCHMARK_SLOW", "1")  # ON by default on Kaggle
if BENCHMARK_SLOW == "1":
    for backend, name, timeout, notes in SLOW_BACKENDS:
        if sweep_skip_only(backend):
            continue
        if sweep_done("asr", backend):
            print(f"⏭ {name} ({backend}): already streamed for run '{RUN_TAG}' — resume skip")
            continue
        r = benchmark_backend(backend, name, timeout, notes)
        results.append(r)
        sweep_publish("asr", r)
else:
    print(f"\n⏭ Skipping {len(SLOW_BACKENDS)} slow backends "
          f"(set BENCHMARK_SLOW=1 to include)")

# TTS smoke benchmark (opt-in, separate from ASR)
BENCHMARK_TTS = os.environ.get("BENCHMARK_TTS", "1")  # full sweep: ASR + TTS by default
if BENCHMARK_TTS == "1":
    # Install espeak-ng for kokoro (piper also benefits from it)
    print("\nInstalling espeak-ng for TTS backends...")
    subprocess.run("sudo apt-get update -qq && sudo apt-get install -y -qq libespeak-ng-dev espeak-ng",
                   shell=True, capture_output=True)

    tts_results = []
    TTS_PHRASE = "The quick brown fox jumps over the lazy dog."

    for backend, name, timeout, notes in TTS_BACKENDS:
        if sweep_skip_only(backend):
            continue
        if sweep_done("tts", backend):
            print(f"⏭ TTS {name} ({backend}): already streamed for run '{RUN_TAG}' — resume skip")
            continue
        print(f"\n{'='*60}")
        print(f"  TTS: {name} (--backend {backend})")
        print(f"{'='*60}")
        outfile = f"/tmp/tts-bench-{backend}.wav"
        cmd = [CRISPASR, "--backend", backend, "-m", "auto", "--auto-download",
               "--tts-output", outfile, "--no-prints"]

        # Per-backend voice/speaker overrides. Several TTS backends REQUIRE a
        # reference voice or speaker and produce 0-byte output without one — the
        # earlier sweep's bare `--tts` made them look like failures when the cause
        # was just missing args. Provide the right voice per backend:
        #   - voice-CLONING (f5/chatterbox/cosyvoice3/vibevoice): a reference wav
        #     (jfk.wav, always cloned with the repo) + the --i-have-rights consent
        #     flag the CLI requires for .wav cloning; cosyvoice3 also needs --ref-text.
        #   - fastpitch: a speaker index (multi-speaker, 5 speakers) via --voice 0.
        #   - orpheus: a built-in voice name.
        REF_WAV = f"{CRISPASR_DIR}/samples/jfk.wav"
        JFK_RT = ("And so my fellow Americans, ask not what your country can do for you, "
                  "ask what you can do for your country.")
        voice_args = {
            "f5-tts":         ["--voice", REF_WAV, "--i-have-rights"],
            "chatterbox":     ["--voice", REF_WAV, "--i-have-rights"],
            "cosyvoice3":     ["--voice", REF_WAV, "--ref-text", JFK_RT, "--i-have-rights"],
            "vibevoice-tts":  ["--voice", REF_WAV, "--i-have-rights"],
            "vibevoice-1.5b": ["--voice", REF_WAV, "--i-have-rights"],
            # 2026-09-07: vibevoice-bitnet was added to TTS_BACKENDS without a
            # voice and failed in 23.5 s with 0 bytes and
            #   "no voice prompt resolved (pass --voice <path.gguf>, ...)"
            # — the exact failure this table's comment was written about. Its
            # two siblings above were already here; the new entry just missed it.
            "vibevoice-bitnet": ["--voice", REF_WAV, "--i-have-rights"],
            "fastpitch":      ["--voice", "0"],
            "orpheus":        ["--voice", "tara"],
        }
        cmd += voice_args.get(backend, [])

        phrase = TTS_PHRASE
        if backend == "dia":
            phrase = "[S1] The quick brown fox jumps over the lazy dog. This is a longer prompt for Dia which needs over one hundred characters to produce good output quality."

        # Per-backend quantisation. `-m auto` takes each registry default, and
        # for kugelaudio that is F16 at ~17.3 GB — which tried to allocate
        # 16483.91 MiB on a 16269 MiB P100 and died with
        #   "failed to allocate 17.28 GB on 'CUDA0' for the model weights"
        # after 289 s. The sweep entry's own note already said "Q4_K ~5.7GB via
        # -m auto (F16 17.3GB needs >16GB VRAM)", so the intent was Q4_K and the
        # resolution silently disagreed with it. Ask for the quant explicitly
        # rather than trusting the default to stay small enough.
        quant_args = {"kugelaudio": ["--model-quant", "q4_k"]}
        cmd += quant_args.get(backend, [])

        cmd += ["--tts", phrase]

        t0 = time.time()
        tts_r = None
        try:
            proc = subprocess.run(cmd, timeout=timeout, capture_output=True, text=True)
            wall = time.time() - t0
            ok = proc.returncode == 0 and os.path.isfile(outfile) and os.path.getsize(outfile) > 1000
            sz = os.path.getsize(outfile) if os.path.isfile(outfile) else 0
            status = "PASS" if ok else "FAIL"
            tts_r = {"backend": backend, "name": name, "wall_s": round(wall, 1),
                     "status": status, "wav_bytes": sz}
            # Carry the failure reason IN the streamed record. The kernel log is
            # the only other place it exists, and that is page-capped and easily
            # unreachable — kugelaudio failed a 6-hour sweep with nothing but
            # "wav_bytes: 0" to show for it.
            if not ok and proc.stderr:
                tts_r["stderr_tail"] = proc.stderr[-1500:]
            print(f"  {status} — {wall:.1f}s, {sz} bytes")
            if not ok and proc.stderr:
                print(f"  stderr: {proc.stderr[-300:]}")
        except subprocess.TimeoutExpired:
            tts_r = {"backend": backend, "name": name, "wall_s": timeout,
                     "status": "TIMEOUT", "wav_bytes": 0}
            print(f"  TIMEOUT after {timeout}s")
        if tts_r:
            tts_results.append(tts_r)
            sweep_publish("tts", tts_r)  # stream interim result immediately (resumable)
        if os.path.isfile(outfile):
            os.remove(outfile)
        _cleanup_cache(backend)

    print(f"\n{'='*60}")
    print("TTS RESULTS")
    print(f"{'='*60}")
    print(f"| Backend | Status | Wall (s) | WAV bytes |")
    print(f"|---|---|---|---|")
    for r in tts_results:
        print(f"| {r['name']} | {r['status']} | {r['wall_s']} | {r['wav_bytes']} |")
    print(f"\n{sum(1 for r in tts_results if r['status']=='PASS')}/{len(tts_results)} passed")
else:
    print(f"\n⏭ Skipping {len(TTS_BACKENDS)} TTS backends "
          f"(set BENCHMARK_TTS=1 to include)")

# ── Text MT backends (m2m100, madlad) — opt-in, translate a sentence ────────
BENCHMARK_MT = os.environ.get("BENCHMARK_MT", "1")
mt_results = []
if BENCHMARK_MT == "1":
    MT_TEXT = "The quick brown fox jumps over the lazy dog."
    for backend, name, timeout, notes in MT_BACKENDS:
        if sweep_skip_only(backend):
            continue
        if sweep_done("mt", backend):
            print(f"⏭ MT {name} ({backend}): already streamed for run '{RUN_TAG}' — resume skip")
            continue
        print(f"\n{'='*60}\n  MT: {name} (--backend {backend})\n{'='*60}")
        cmd = [CRISPASR, "--backend", backend, "-m", "auto", "--auto-download",
               "-sl", "en", "-tl", "de", "--text", MT_TEXT, "--no-prints"]
        t0 = time.time()
        mt_r = None
        try:
            proc = subprocess.run(cmd, timeout=timeout, capture_output=True, text=True)
            wall = time.time() - t0
            out = (proc.stdout or "").strip()
            ok = proc.returncode == 0 and len(out) > 0
            mt_r = {"backend": backend, "name": name, "wall_s": round(wall, 1),
                    "status": "PASS" if ok else "FAIL", "output": out[:200]}
            print(f"  {'PASS' if ok else 'FAIL'} — {wall:.1f}s — {out[:80]!r}")
            if not ok and proc.stderr:
                print(f"  stderr: {proc.stderr[-300:]}")
        except subprocess.TimeoutExpired:
            mt_r = {"backend": backend, "name": name, "wall_s": timeout, "status": "TIMEOUT", "output": ""}
            print(f"  TIMEOUT after {timeout}s")
        if mt_r:
            mt_results.append(mt_r)
            sweep_publish("mt", mt_r)
        _cleanup_cache(backend)

# ── Final combined summary → HF dataset (one index for the whole run) ────────
if _sweep_ok:
    import io as _io
    summary = {"run": RUN_TAG, "asr": results, "tts": tts_results, "mt": mt_results,
               "counts": {"asr": len(results), "tts": len(tts_results), "mt": len(mt_results)}}
    try:
        _hf_api.upload_file(path_or_fileobj=_io.BytesIO(json.dumps(summary, indent=2, default=str).encode()),
                            path_in_repo=f"{SWEEP_PREFIX}/summary.json", repo_type="dataset",
                            repo_id=SWEEP_REPO, commit_message=f"sweep {RUN_TAG}: summary")
        print(f"[sweep] ↑ combined summary → {SWEEP_REPO}/{SWEEP_PREFIX}/summary.json")
    except Exception as e:
        print(f"[sweep] ! summary upload failed: {e!r}")

# ─────────────────────────── cell 7 (code) ───────────────────────────
# ── Format results table ───────────────────────────────────────────────────
import platform

# System info
sys_info = {
    "date": datetime.now().strftime("%Y-%m-%d %H:%M UTC"),
    "platform": platform.platform(),
    "cpu": platform.processor() or "unknown",
    "gpu": "CUDA" if has_gpu else "CPU-only",
    "python": platform.python_version(),
    "audio": f"jfk.wav ({AUDIO_DURATION:.1f}s)",
}

# Build markdown table
md_lines = []
md_lines.append("# CrispASR Backend Benchmark Results\n")
md_lines.append(f"**Date:** {sys_info['date']}  ")
md_lines.append(f"**Platform:** {sys_info['platform']}  ")
md_lines.append(f"**GPU:** {sys_info['gpu']}  ")
md_lines.append(f"**Audio:** {sys_info['audio']}  ")
md_lines.append(f"**Reference:** _{JFK_REF}_\n")

md_lines.append("| # | Backend | Status | WER | RT Factor | Inference (s) | Wall (s) | Model (MB) | Transcript |")
md_lines.append("|---|---|---|---|---|---|---|---|---|")

for i, r in enumerate(results, 1):
    status = {"PASS": "✅", "DEGRADED": "⚠️", "FAIL": "❌",
              "CRASH": "💥", "TIMEOUT": "⏱️", "EMPTY": "🔇"}.get(r["status"], "❓")
    wer_str = f"{r['wer']:.1%}" if r["wer"] is not None else "—"
    rt_str = f"{r['realtime_factor']:.1f}x" if r["realtime_factor"] else "—"
    inf_str = f"{r['inference_s']:.1f}" if r.get("inference_s") else "—"
    wall_str = f"{r['wall_s']:.1f}" if r.get("wall_s") else "—"
    sz_str = f"{r['model_size_mb']:.0f}" if r["model_size_mb"] else "—"
    transcript = r["transcript"][:60] + "..." if len(r["transcript"]) > 60 else r["transcript"]
    transcript = transcript.replace("|", "\\|")

    md_lines.append(
        f"| {i} | **{r['display_name']}** | {status} | {wer_str} | {rt_str} | "
        f"{inf_str} | {wall_str} | {sz_str} | {transcript} |"
    )

# Summary stats
passed = sum(1 for r in results if r["status"] == "PASS")
total = len(results)
md_lines.append(f"\n**Summary:** {passed}/{total} passed, "
                f"{sum(1 for r in results if r['status'] == 'DEGRADED')} degraded, "
                f"{sum(1 for r in results if r['status'] in ('FAIL', 'CRASH', 'TIMEOUT'))} failed\n")

# Speed ranking
speed_results = [r for r in results if r["realtime_factor"] and r["status"] in ("PASS", "DEGRADED")]
if speed_results:
    speed_results.sort(key=lambda r: r["realtime_factor"], reverse=True)
    md_lines.append("## Speed Ranking (fastest first)\n")
    for i, r in enumerate(speed_results, 1):
        md_lines.append(f"{i}. **{r['display_name']}** — {r['realtime_factor']:.1f}x realtime "
                       f"(WER {r['wer']:.1%})")

# Quality ranking
quality_results = [r for r in results if r["wer"] is not None]
if quality_results:
    quality_results.sort(key=lambda r: r["wer"])
    md_lines.append("\n## Quality Ranking (lowest WER first)\n")
    for i, r in enumerate(quality_results, 1):
        md_lines.append(f"{i}. **{r['display_name']}** — WER {r['wer']:.1%} "
                       f"({r['realtime_factor']:.1f}x RT)")

report_md = "\n".join(md_lines)
print(report_md)

# ─────────────────────────── cell 8 (code) ───────────────────────────
# ── Save results ───────────────────────────────────────────────────────────
# Save JSON
json_path = f"{RESULTS_DIR}/benchmark_results.json"
with open(json_path, "w") as f:
    json.dump({"system": sys_info, "results": results}, f, indent=2)
print(f"✓ JSON saved to {json_path}")

# Save Markdown
md_path = f"{RESULTS_DIR}/benchmark_results.md"
with open(md_path, "w") as f:
    f.write(report_md)
print(f"✓ Markdown saved to {md_path}")

# ─────────────────────────── cell 9 (code) ───────────────────────────
# ── Upload results as GitHub Gist (optional) ───────────────────────────────
if GH_GIST_TOKEN:
    import urllib.request

    gist_data = {
        "description": f"CrispASR Benchmark — {sys_info['date']} ({sys_info['gpu']})",
        "public": True,
        "files": {
            "benchmark_results.md": {"content": report_md},
            "benchmark_results.json": {"content": json.dumps(
                {"system": sys_info, "results": results}, indent=2)},
        }
    }

    req = urllib.request.Request(
        "https://api.github.com/gists",
        data=json.dumps(gist_data).encode(),
        headers={
            "Authorization": f"token {GH_GIST_TOKEN}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req) as resp:
            gist_resp = json.loads(resp.read())
            print(f"✓ Gist created: {gist_resp['html_url']}")
    except Exception as e:
        print(f"✗ Gist upload failed: {e}")
else:
    print("⏭ No GH_GIST_TOKEN — skipping gist upload")
    print("  Set a Kaggle secret named GH_GIST_TOKEN to auto-upload results")

# ─────────────────────────── cell 10 (code) ───────────────────────────
# ── Final summary ──────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  CrispASR Benchmark Complete")
print("=" * 60)
print(f"  Backends tested: {len(results)}")
print(f"  Passed: {sum(1 for r in results if r['status'] == 'PASS')}")
print(f"  Fastest: {max((r for r in results if r.get('realtime_factor')), key=lambda r: r['realtime_factor'], default={})}")
print(f"  Best WER: {min((r for r in results if r.get('wer') is not None), key=lambda r: r['wer'], default={})}")
print(f"\n  Results: {RESULTS_DIR}/")
print("=" * 60)
