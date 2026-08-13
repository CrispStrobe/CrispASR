#!/usr/bin/env python3
"""Issue #348 release gate: pinned convert/quant/diff/CUDA/live/roundtrip run.

The kernel deliberately clones both implementations at immutable commits,
recreates the release Q4_K pair from the official checkpoint, and refuses to
continue unless it is byte-identical to the published pair.  All large files
live under /kaggle/temp; /kaggle/working only receives compact evidence.
"""

from __future__ import annotations

import contextlib
import ctypes
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import traceback
import wave
from pathlib import Path

WORK = Path("/kaggle/working")
STAGE = Path("/kaggle/temp/chatterbox-v3-348")
MODELS = STAGE / "models"
SOURCE = STAGE / "official-model"
GENERATED = STAGE / "generated"
REPO = Path("/kaggle/temp/CrispASR")
UPSTREAM = Path("/kaggle/temp/resemble-chatterbox")
RESULTS = WORK / "results"
for directory in (STAGE, MODELS, SOURCE, GENERATED, RESULTS):
    directory.mkdir(parents=True, exist_ok=True)

CRISPASR_URL = "https://github.com/CrispStrobe/CrispASR.git"
CRISPASR_COMMIT = "ea3302edd763715cbf047bb5adae688ac0563a6a"
UPSTREAM_URL = "https://github.com/resemble-ai/chatterbox.git"
UPSTREAM_COMMIT = "5de7a54aa4e5e2baadb0182dde554908b48b85c2"
MODEL_REPO = "ResembleAI/chatterbox"
MODEL_REVISION = "5bb1f6ee58e50c3b8d408bc82a6d3740c2db6e18"
GGUF_REPO = "cstr/chatterbox-GGUF"
FIXTURE_REPO = "cstr/crispasr-regression-fixtures"
T3_SHA256 = "fb63d7c53c4f8bb07286d67709c4d4200fa63b4de3476a4558f37964ce8e5aa6"
S3_SHA256 = "dce74d3df941abcd1bd19f4a96f8325b89d3b58391f4be5c01c216db56a9a725"


def clone_exact(url: str, dest: Path, commit: str, submodules: bool = False) -> None:
    if not (dest / ".git").exists():
        subprocess.run(["git", "clone", "--filter=blob:none", url, str(dest)], check=True, timeout=1200)
    subprocess.run(["git", "-C", str(dest), "fetch", "--depth", "1", "origin", commit], check=True, timeout=1200)
    subprocess.run(["git", "-C", str(dest), "checkout", "--detach", commit], check=True)
    got = subprocess.check_output(["git", "-C", str(dest), "rev-parse", "HEAD"], text=True).strip()
    if got != commit:
        raise RuntimeError(f"checkout mismatch for {dest}: {got} != {commit}")
    if submodules:
        subprocess.run(["git", "-C", str(dest), "submodule", "update", "--init", "--recursive"],
                       check=True, timeout=1800)


# The harness is imported from the exact checkout under test.  A clone failure
# is fatal: silently using a stale bundled helper would invalidate provenance.
clone_exact(CRISPASR_URL, REPO, CRISPASR_COMMIT, submodules=True)
sys.path.insert(0, str(REPO / "tools" / "kaggle"))
import kaggle_harness as kh  # noqa: E402

kh.init_progress()
kh.step("provenance.crispasr", commit=CRISPASR_COMMIT)


def run(argv, *, cwd=None, env=None, timeout=None, log: Path | None = None, check=True):
    argv = [str(x) for x in argv]
    print("$ " + " ".join(argv), flush=True)
    merged = os.environ.copy()
    if env:
        merged.update({str(k): str(v) for k, v in env.items()})
    proc = subprocess.run(argv, cwd=str(cwd) if cwd else None, env=merged, text=True,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=timeout)
    print(proc.stdout, end="", flush=True)
    if log:
        log.write_text(proc.stdout)
    if check and proc.returncode:
        raise subprocess.CalledProcessError(proc.returncode, argv, output=proc.stdout)
    return proc


@contextlib.contextmanager
def heartbeat(label: str):
    with kh.build_heartbeat(label, interval_s=30):
        yield


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def gguf_string(reader, key: str) -> str:
    field = reader.fields[key]
    return bytes(field.parts[field.data[0]]).decode("utf-8")


def compare_gguf_structure(generated: Path, published: Path, architecture: str,
                           checkpoint_key: str, checkpoint: str) -> dict:
    """Compare cross-ISA quants structurally, leaving numerics to crispasr-diff.

    ggml's SIMD quantizers are not promised byte-identical between ARM64 and
    x86_64, so a whole-file SHA is not a portable converter gate.  Tensor
    names/shapes/types and provenance must be exact; the activation harness
    below then gates the numerical consequence stage by stage.
    """
    from gguf import GGUFReader
    left, right = GGUFReader(str(generated)), GGUFReader(str(published))
    left_map = {t.name: (tuple(int(x) for x in t.shape), int(t.tensor_type)) for t in left.tensors}
    right_map = {t.name: (tuple(int(x) for x in t.shape), int(t.tensor_type)) for t in right.tensors}
    if left_map != right_map:
        missing = sorted(set(right_map) - set(left_map))[:10]
        extra = sorted(set(left_map) - set(right_map))[:10]
        changed = sorted(k for k in set(left_map) & set(right_map) if left_map[k] != right_map[k])[:10]
        raise RuntimeError(f"GGUF structure mismatch: missing={missing}, extra={extra}, changed={changed}")
    for reader, label in ((left, "generated"), (right, "published")):
        if gguf_string(reader, "general.architecture") != architecture:
            raise RuntimeError(f"{label} architecture mismatch")
        if gguf_string(reader, "general.source.revision") != MODEL_REVISION:
            raise RuntimeError(f"{label} source revision mismatch")
        if gguf_string(reader, checkpoint_key) != checkpoint:
            raise RuntimeError(f"{label} checkpoint mismatch")
    return {"tensor_count": len(left_map), "structure_equal": True,
            "generated_bytes": generated.stat().st_size, "published_bytes": published.stat().st_size}


def download(repo: str, filename: str, token: str | None, repo_type="model") -> Path:
    from huggingface_hub import hf_hub_download
    label = re.sub(r"[^A-Za-z0-9_.-]", "_", filename)
    with heartbeat(f"download.{label}"):
        path = hf_hub_download(repo_id=repo, filename=filename, repo_type=repo_type,
                               local_dir=str(MODELS), token=token or None)
    return Path(path)


def build() -> tuple[Path, Path]:
    kh.install_build_toolchain()
    # Seed extraction defaults to /kaggle/working/.ccache.  Keep the warm cache
    # out of output storage while preserving it for this build.
    seeded = WORK / ".ccache"
    cache = Path("/kaggle/temp/.ccache")
    if seeded.exists() and not cache.exists():
        shutil.move(str(seeded), str(cache))
    cache.mkdir(parents=True, exist_ok=True)
    os.environ["CCACHE_DIR"] = str(cache)
    arch = kh.detect_cuda_arch()
    flags = [
        "-DCMAKE_BUILD_TYPE=Release", "-DGGML_NATIVE=OFF", "-DCRISPASR_BUILD_TESTS=ON",
        "-DCRISPASR_OPUS_FETCH=ON",
        *kh.cuda_build_flags(arch), *kh.cache_and_link_flags(),
    ]
    build_dir = REPO / "build-cuda"
    kh.step("build.configure", arch=arch)
    with heartbeat("build.configure"):
        run(["cmake", "-S", REPO, "-B", build_dir, "-G", "Ninja", *flags], timeout=1800)
    targets = ["crispasr", "crispasr-diff", "crispasr-quantize", "test-registry",
               "test-chatterbox-text-prep", "test-chatterbox-params", "test-parakeet-strategy",
               "test-titanet", "test-parakeet-longform", "test-parakeet-ja-longform"]
    jobs = kh.safe_build_jobs(gpu=True)
    kh.step("build.compile", jobs=jobs, targets=targets)
    with heartbeat("build.compile"):
        kh.sh_with_progress(
            "cmake --build build-cuda -j" + jobs + " --target " + " ".join(targets), cwd=str(REPO)
        )
    lib_candidates = list(build_dir.glob("src/libcrispasr.so*"))
    lib = next((p for p in lib_candidates if p.name == "libcrispasr.so"), None)
    if lib is None:
        lib = next((p for p in lib_candidates if p.is_file()), None)
    if lib is None:
        raise RuntimeError("CUDA build did not produce libcrispasr.so")
    os.environ["LD_LIBRARY_PATH"] = str(lib.parent) + ":" + os.environ.get("LD_LIBRARY_PATH", "")
    run([build_dir / "bin/crispasr", "--version"], log=RESULTS / "build-version.log")
    return build_dir, lib


def hermetic_tests(build_dir: Path) -> None:
    with heartbeat("tests.hermetic"):
        logs = []
        for binary in ("test-chatterbox-text-prep", "test-chatterbox-params",
                       "test-parakeet-strategy", "test-registry"):
            proc = run([build_dir / "bin" / binary, "--reporter", "compact"],
                       cwd=REPO, timeout=1200)
            logs.append(f"===== {binary} =====\n{proc.stdout}")
        proc = run(["ctest", "--test-dir", build_dir, "--output-on-failure", "-R",
                    "^test-chatterbox-quant-carveout$"], timeout=1200)
        logs.append(f"===== quant carveout =====\n{proc.stdout}")
        (RESULTS / "hermetic-tests.log").write_text("\n".join(logs))
    kh.step("tests.hermetic.pass")


def install_python_oracle() -> None:
    clone_exact(UPSTREAM_URL, UPSTREAM, UPSTREAM_COMMIT)
    kh.step("provenance.upstream", commit=UPSTREAM_COMMIT)
    # Kaggle already supplies CUDA torch/torchaudio; replacing those wheels can
    # consume the run's disk and silently alter the CUDA stack.
    packages = [
        "huggingface_hub>=0.34", "gguf", "librosa==0.11.0", "s3tokenizer",
        "transformers==5.2.0", "diffusers==0.29.0", "conformer==0.3.2",
        "safetensors==0.5.3", "spacy-pkuseg", "pykakasi==2.3.0",
        "pyloudnorm", "omegaconf", "git+https://github.com/resemble-ai/Perth.git@master",
    ]
    with heartbeat("python.dependencies"):
        run([sys.executable, "-m", "pip", "install", "-q", *packages], timeout=2400)
        run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "-e", UPSTREAM], timeout=600)
    run([sys.executable, "-c",
         "import torch,chatterbox; assert torch.cuda.is_available(); print(torch.__version__, torch.cuda.get_device_name())"],
        log=RESULTS / "python-cuda.log")


def fetch_sources(token: str | None) -> None:
    from huggingface_hub import snapshot_download
    allow = ["ve.pt", "ve.safetensors", "s3gen.pt", "s3gen.safetensors",
             "t3_mtl23ls_v3.safetensors", "grapheme_mtl_merged_expanded_v1.json",
             "conds.pt", "Cangjie5_TC.json"]
    with heartbeat("download.official-source"):
        snapshot_download(MODEL_REPO, revision=MODEL_REVISION, allow_patterns=allow,
                          local_dir=str(SOURCE), token=token or None)
    required = ["ve.pt", "s3gen.pt", "s3gen.safetensors", "t3_mtl23ls_v3.safetensors",
                "grapheme_mtl_merged_expanded_v1.json", "conds.pt", "Cangjie5_TC.json"]
    missing = [name for name in required if not (SOURCE / name).is_file()]
    if missing:
        raise RuntimeError(f"pinned official snapshot missing: {missing}")
    source_sha = sha256(SOURCE / "t3_mtl23ls_v3.safetensors")
    expected = "5abca8321ede76f8e61f1cc0d19aea6c946b28871017ce8726f8a69203f05953"
    if source_sha != expected:
        raise RuntimeError(f"official T3 checksum mismatch: {source_sha}")
    kh.step("source.verified", revision=MODEL_REVISION, t3_sha256=source_sha)


def convert_quantize(build_dir: Path, published_t3: Path, published_s3: Path) -> tuple[Path, Path]:
    converter = REPO / "models/convert-chatterbox-to-gguf.py"
    with heartbeat("convert.f16"):
        run([sys.executable, converter, "--input", SOURCE, "--output-dir", GENERATED,
             "--t3-checkpoint", "t3_mtl23ls_v3.safetensors",
             "--s3gen-checkpoint", "s3gen.safetensors", "--revision", MODEL_REVISION,
             "--output-prefix", "chatterbox-v3"], timeout=3600, log=RESULTS / "convert.log")
    quant = build_dir / "bin/crispasr-quantize"
    t3 = GENERATED / "chatterbox-v3-t3-q4_k.gguf"
    s3 = GENERATED / "chatterbox-v3-s3gen-q4_k.gguf"
    with heartbeat("quantize.t3"):
        run([quant, GENERATED / "chatterbox-v3-t3-f16.gguf", t3, "q4_k"],
            timeout=2400, log=RESULTS / "quant-t3.log")
    with heartbeat("quantize.s3"):
        run([quant, GENERATED / "chatterbox-v3-s3gen-f16.gguf", s3, "q4_k"],
            timeout=2400, log=RESULTS / "quant-s3.log")
    checks = {
        "generated_t3": sha256(t3), "published_t3": sha256(published_t3),
        "generated_s3": sha256(s3), "published_s3": sha256(published_s3),
    }
    if checks["published_t3"] != T3_SHA256 or checks["published_s3"] != S3_SHA256:
        raise RuntimeError(f"published artifact checksum mismatch: {checks}")
    structure = {
        "t3": compare_gguf_structure(t3, published_t3, "chatterbox", "chatterbox.t3.checkpoint",
                                     "t3_mtl23ls_v3.safetensors"),
        "s3": compare_gguf_structure(s3, published_s3, "chatterbox-s3gen", "chatterbox.s3gen.checkpoint",
                                     "s3gen.safetensors"),
    }
    evidence = {"sha256": checks, "structure": structure,
                "whole_file_identity_expected_cross_isa": False}
    (RESULTS / "artifact-sha256.json").write_text(json.dumps(evidence, indent=2))
    kh.step("artifacts.structure-equal", generated_t3=checks["generated_t3"],
            generated_s3=checks["generated_s3"], t3_tensors=structure["t3"]["tensor_count"],
            s3_tensors=structure["s3"]["tensor_count"])
    return t3, s3


def run_diff(build_dir: Path, t3: Path, reference: Path, voice: Path, label: str) -> dict[str, int]:
    env = {"CRISPASR_DIFF_USE_GPU": "1"}
    log = RESULTS / f"diff-{label}.log"
    with heartbeat(f"diff.{label}"):
        proc = run([build_dir / "bin/crispasr-diff", "chatterbox", t3, reference, voice],
                   cwd=REPO, env=env, timeout=3600, log=log)
    match = re.search(r"summary:\s+(\d+) pass,\s+(\d+) fail,\s+(\d+) skip", proc.stdout)
    if not match:
        raise RuntimeError(f"diff {label} emitted no parseable summary")
    result = {"pass": int(match.group(1)), "fail": int(match.group(2)), "skip": int(match.group(3))}
    if result["fail"] != 0 or result["pass"] < 30:
        raise RuntimeError(f"diff {label} failed gate: {result}")
    kh.step(f"diff.{label}.pass", **result)
    return result


def dump_cuda_reference(voice: Path) -> Path | None:
    probe = run([sys.executable, "-c",
                 "import torch; print(','.join(torch.cuda.get_arch_list()))"], check=False)
    capability = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"], text=True
    ).strip().replace(".", "")
    wanted = f"sm_{capability}"
    if probe.returncode != 0 or wanted not in probe.stdout.splitlines()[-1].split(","):
        kh.step("reference.cuda.unsupported", gpu_arch=wanted,
                torch_arches=probe.stdout.strip()[-300:],
                reason="Kaggle PyTorch wheel has no kernel image for the assigned GPU")
        return None
    out = RESULTS / "chatterbox-v3-de-jfk-cuda-ref.gguf"
    env = {
        "RESEMBLE_CHATTERBOX_SRC": str(UPSTREAM), "CHATTERBOX_DEVICE": "cuda",
        "CHATTERBOX_LANG": "de", "CHATTERBOX_SEED": "42",
        "CHATTERBOX_SYN_TEXT": "Guten Tag. Dieser Test prüft die deutsche Aussprache und die Stimme.",
    }
    with heartbeat("reference.cuda"):
        run([sys.executable, REPO / "tools/dump_reference.py", "--backend", "chatterbox",
             "--model-dir", SOURCE, "--audio", voice, "--output", out,
             "--max-new-tokens", "120"], cwd=REPO, env=env, timeout=5400,
            log=RESULTS / "reference-cuda.log")
    if out.stat().st_size < 1_000_000:
        raise RuntimeError("CUDA reference archive is unexpectedly small")
    kh.step("reference.cuda.ready", bytes=out.stat().st_size, sha256=sha256(out))
    return out


PHRASES = {
    "ar": "مرحبا، هذا اختبار صوتي قصير.", "da": "Goddag, dette er en kort stemmetest.",
    "de": "Guten Tag, dies ist ein kurzer Stimmtest.", "el": "Γεια σας, αυτή είναι μια σύντομη δοκιμή φωνής.",
    "en": "Hello, this is a short voice test.", "es": "Hola, esta es una breve prueba de voz.",
    "fi": "Hei, tämä on lyhyt äänitesti.", "fr": "Bonjour, ceci est un court test de voix.",
    "he": "שלום, זוהי בדיקת קול קצרה.", "hi": "नमस्ते, यह एक छोटा आवाज़ परीक्षण है।",
    "it": "Ciao, questa è una breve prova vocale.", "ja": "こんにちは、これは短い音声テストです。",
    "ko": "안녕하세요, 이것은 짧은 음성 테스트입니다.", "ms": "Helo, ini ialah ujian suara ringkas.",
    "nl": "Hallo, dit is een korte stemtest.", "no": "Hei, dette er en kort stemmetest.",
    "pl": "Dzień dobry, to krótki test głosu.", "pt": "Olá, este é um breve teste de voz.",
    "ru": "Здравствуйте, это короткая проверка голоса.", "sv": "Hej, detta är ett kort rösttest.",
    "sw": "Habari, hili ni jaribio fupi la sauti.", "tr": "Merhaba, bu kısa bir ses testidir.",
    "zh": "你好，这是一个简短的语音测试。",
}


class ChatterboxABI:
    def __init__(self, library: Path, t3: Path, s3: Path, voice: Path | None):
        self.lib = ctypes.CDLL(str(library))
        L = self.lib
        L.crispasr_set_gpu_backend.argtypes = [ctypes.c_char_p]
        L.crispasr_session_open_explicit.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int]
        L.crispasr_session_open_explicit.restype = ctypes.c_void_p
        for name in ("crispasr_session_set_codec_path", "crispasr_session_set_voice",
                     "crispasr_session_set_source_language", "crispasr_session_set_target_language",
                     "crispasr_session_set_tts_reference_language"):
            getattr(L, name).restype = ctypes.c_int
        L.crispasr_session_set_codec_path.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        L.crispasr_session_set_voice.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p]
        L.crispasr_session_set_source_language.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        L.crispasr_session_set_target_language.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        L.crispasr_session_set_tts_reference_language.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        L.crispasr_session_set_tts_seed.argtypes = [ctypes.c_void_p, ctypes.c_uint64]
        L.crispasr_session_set_tts_seed.restype = ctypes.c_int
        L.crispasr_session_set_tts_steps.argtypes = [ctypes.c_void_p, ctypes.c_int]
        L.crispasr_session_set_tts_steps.restype = ctypes.c_int
        L.crispasr_session_accept_marking_responsibility.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        L.crispasr_session_accept_marking_responsibility.restype = ctypes.c_int
        L.crispasr_session_synthesize_raw.argtypes = [ctypes.c_void_p, ctypes.c_char_p,
                                                       ctypes.POINTER(ctypes.c_int)]
        L.crispasr_session_synthesize_raw.restype = ctypes.POINTER(ctypes.c_float)
        L.crispasr_pcm_free.argtypes = [ctypes.POINTER(ctypes.c_float)]
        L.crispasr_session_close.argtypes = [ctypes.c_void_p]
        L.crispasr_set_gpu_backend(b"cuda")
        self.session = L.crispasr_session_open_explicit(str(t3).encode(), b"chatterbox", 4)
        if not self.session:
            raise RuntimeError("C ABI could not open Chatterbox T3")
        self._ok("codec", L.crispasr_session_set_codec_path(self.session, str(s3).encode()))
        self._ok("marking", L.crispasr_session_accept_marking_responsibility(
            self.session, b"private automated release validation; outputs retained as test evidence"))
        self._ok("seed", L.crispasr_session_set_tts_seed(self.session, 42))
        self._ok("steps", L.crispasr_session_set_tts_steps(self.session, 6))
        if voice:
            self._ok("reference language", L.crispasr_session_set_tts_reference_language(self.session, b"en"))
            self._ok("voice", L.crispasr_session_set_voice(self.session, str(voice).encode(), None))

    @staticmethod
    def _ok(label: str, rc: int) -> None:
        if rc != 0:
            raise RuntimeError(f"C ABI {label} setter failed: {rc}")

    def synth(self, language: str, text: str, out: Path) -> dict:
        self._ok("target language", self.lib.crispasr_session_set_target_language(
            self.session, language.encode()))
        n = ctypes.c_int()
        ptr = self.lib.crispasr_session_synthesize_raw(self.session, text.encode(), ctypes.byref(n))
        if not ptr or n.value <= 0:
            raise RuntimeError(f"C ABI synthesis failed for {language}")
        try:
            samples = [ptr[i] for i in range(n.value)]
        finally:
            self.lib.crispasr_pcm_free(ptr)
        finite = all(math.isfinite(x) for x in samples)
        peak = max(abs(x) for x in samples)
        rms = math.sqrt(sum(x * x for x in samples) / len(samples))
        duration = len(samples) / 24000.0
        if not finite or peak < 1e-4 or rms < 1e-5 or not (0.25 <= duration <= 30.0):
            raise RuntimeError(f"invalid {language} PCM: finite={finite} peak={peak} rms={rms} duration={duration}")
        pcm = bytearray()
        for x in samples:
            value = max(-32768, min(32767, round(x * 32767)))
            pcm.extend(int(value).to_bytes(2, "little", signed=True))
        with wave.open(str(out), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(24000)
            wf.writeframes(pcm)
        return {"samples": n.value, "duration_s": round(duration, 3),
                "peak": round(peak, 6), "rms": round(rms, 6), "sha256": sha256(out)}

    def close(self) -> None:
        if self.session:
            self.lib.crispasr_session_close(self.session)
            self.session = None


def all_language_live(library: Path, t3: Path, s3: Path, voice: Path) -> dict:
    session = ChatterboxABI(library, t3, s3, voice)
    results = {}
    try:
        for language, text in PHRASES.items():
            out = RESULTS / f"live-{language}.wav"
            with heartbeat(f"live.{language}"):
                results[language] = session.synth(language, text, out)
            kh.step(f"live.{language}.pass", **results[language])
    finally:
        session.close()
    if set(results) != set(PHRASES):
        raise RuntimeError("not every supported language completed")
    (RESULTS / "all-language-live.json").write_text(json.dumps(results, ensure_ascii=False, indent=2))
    return results


def generate_independent_reference(build_dir: Path, kokoro: Path, kokoro_voice: Path) -> Path:
    out = RESULTS / "R-kokoro-en.wav"
    text = "The clear morning voice carries gently across the quiet valley."
    with heartbeat("roundtrip.reference-R"):
        run([build_dir / "bin/crispasr", "--gpu-backend", "cuda", "--backend", "kokoro",
             "-m", kokoro, "--voice", kokoro_voice, "--speaker-identity", "synthetic",
             "--tts", text, "--tts-output", out, "--seed", "42"], timeout=1800,
            log=RESULTS / "R-kokoro.log")
    if not out.is_file():
        raise RuntimeError("independent Kokoro reference was not generated")
    return out


def extract_cli_transcript(output: str) -> str:
    diagnostics = ("crispasr ", "crispasr:", "parakeet:", "ggml", "main:", "system_info:")
    candidates = [line.strip() for line in output.splitlines()
                  if line.strip() and not line.strip().lower().startswith(diagnostics)]
    if not candidates:
        return ""
    return candidates[-1]


def clone_baseline_roundtrips(build_dir: Path, library: Path, t3: Path, s3: Path,
                              reference_r: Path, parakeet: Path, titanet: Path) -> dict:
    target = "Guten Tag. Dieser Test prüft die deutsche Aussprache und die Stimme."
    clone = ChatterboxABI(library, t3, s3, reference_r)
    baseline = ChatterboxABI(library, t3, s3, None)
    c_path, b_path = RESULTS / "C-clone-de.wav", RESULTS / "B-default-de.wav"
    try:
        with heartbeat("roundtrip.clone-C"):
            c_stats = clone.synth("de", target, c_path)
        with heartbeat("roundtrip.baseline-B"):
            b_stats = baseline.synth("de", target, b_path)
    finally:
        clone.close()
        baseline.close()
    asr = {}
    for label, path in (("C", c_path), ("B", b_path)):
        log = RESULTS / f"asr-{label}.log"
        with heartbeat(f"roundtrip.asr-{label}"):
            proc = run([build_dir / "bin/crispasr", "--gpu-backend", "cuda", "--backend", "parakeet",
                        "-m", parakeet, "-f", path, "-l", "de"], timeout=1200, log=log)
        transcript = extract_cli_transcript(proc.stdout)
        if (len(re.sub(r"\W", "", transcript, flags=re.UNICODE)) < 12
                or "guten" not in transcript.lower() or "stimme" not in transcript.lower()):
            raise RuntimeError(f"empty/short ASR roundtrip for {label}: {transcript!r}")
        asr[label] = transcript[-1000:]
    speaker_log = RESULTS / "speaker-rcb.log"
    with heartbeat("roundtrip.speaker-RCB"):
        proc = run([build_dir / "bin/test-titanet", titanet, reference_r, c_path, b_path],
                   timeout=1200, log=speaker_log)
    # test-titanet prints one matrix row per input. Parse all decimals and use
    # the first row's off-diagonal entries (R,C) and (R,B).
    matrix_rows = []
    for line in proc.stdout.splitlines():
        values = re.findall(r"[-+]?\d+\.\d+", line)
        if len(values) == 3:
            matrix_rows.append([float(x) for x in values])
    if not matrix_rows:
        raise RuntimeError("could not parse TitaNet similarity matrix")
    rc, rb = matrix_rows[0][1], matrix_rows[0][2]
    if not rc > rb:
        raise RuntimeError(f"speaker clone gate failed: cos(C,R)={rc}, cos(B,R)={rb}")
    result = {"target": target, "clone": c_stats, "baseline": b_stats,
              "asr": asr, "cos_C_R": rc, "cos_B_R": rb, "speaker_gate": True}
    (RESULTS / "roundtrip-rcb.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
    kh.step("roundtrip.RCB.pass", cos_C_R=rc, cos_B_R=rb)
    return result


def asr_subset(build_dir: Path, parakeet: Path) -> dict:
    # Parakeet V3 covers European languages, not the Asian/Arabic part of the
    # Chatterbox matrix.  Those remain guarded by native PCM + CUDA diff; do not
    # mislabel unsupported ASR as a language failure.
    supported = ("en", "de", "fr", "es", "it", "pt", "nl", "pl", "ru")
    result = {}
    for language in supported:
        wav = RESULTS / f"live-{language}.wav"
        with heartbeat(f"asr-live.{language}"):
            proc = run([build_dir / "bin/crispasr", "--gpu-backend", "cuda", "--backend", "parakeet",
                        "-m", parakeet, "-f", wav, "-l", language], timeout=1200,
                       log=RESULTS / f"asr-live-{language}.log")
        text = extract_cli_transcript(proc.stdout)
        if len(re.sub(r"\W", "", text, flags=re.UNICODE)) < 3:
            raise RuntimeError(f"Parakeet roundtrip empty for {language}")
        result[language] = text[-1000:]
        kh.step(f"asr-live.{language}.pass")
    (RESULTS / "asr-subset.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def parakeet_longform(build_dir: Path, parakeet: Path, token: str | None) -> dict:
    # JA TDT is known to loop at Q4_K (the runtime warns and recommends CTC);
    # the release-quality TDT regression artifact is Q8_0.
    ja_model = download("cstr/parakeet-tdt-0.6b-ja-GGUF", "parakeet-tdt-0.6b-ja-q8_0.gguf", token)
    ja_fixture = download(FIXTURE_REPO, "parakeet-tdt-0.6b-ja/reazon_baseball_14s/audio.wav", token)
    cases = [
        ("v3", build_dir / "bin/test-parakeet-longform",
         {"CRISPASR_MODEL_PARAKEET": str(parakeet)}),
        ("ja", build_dir / "bin/test-parakeet-ja-longform",
         {"CRISPASR_MODEL_PARAKEET_JA": str(ja_model),
          "CRISPASR_FIXTURE_PARAKEET_JA": str(ja_fixture)}),
    ]
    result = {}
    for label, binary, env in cases:
        with heartbeat(f"parakeet-longform.{label}"):
            proc = run([binary, "--reporter", "compact"], cwd=REPO, env=env, timeout=5400,
                       log=RESULTS / f"parakeet-longform-{label}.log")
        result[label] = proc.returncode == 0
        kh.step(f"parakeet-longform.{label}.pass")
    return result


def main() -> None:
    with heartbeat("python.bootstrap"):
        run([sys.executable, "-m", "pip", "install", "-q", "huggingface_hub", "hf_transfer", "gguf"],
            timeout=600)
    token = kh.resolve_hf_token(require=True)
    os.environ["HF_TOKEN"] = token
    kh.step("hf.token.ready", present=True)
    build_dir, library = build()
    # Preserve a successful cold CUDA build even if a later model-backed gate
    # fails.  This keeps a corrective rerun from paying the full compile cost.
    with heartbeat("ccache.checkpoint"):
        kh.export_ccache_tar()
    hermetic_tests(build_dir)

    published_t3 = download(GGUF_REPO, "chatterbox-v3-t3-q4_k.gguf", token)
    published_s3 = download(GGUF_REPO, "chatterbox-v3-s3gen-q4_k.gguf", token)
    cpu_ref = download(FIXTURE_REPO, "chatterbox-v3/de-jfk/ref.gguf", token)
    voice = REPO / "samples/jfk.wav"
    parakeet = download("cstr/parakeet-tdt-0.6b-v3-GGUF", "parakeet-tdt-0.6b-v3-q4_k.gguf", token)
    titanet = download("cstr/titanet-large-GGUF", "titanet-large.gguf", token)
    kokoro = download("cstr/kokoro-82m-GGUF", "kokoro-82m-q8_0.gguf", token)
    kokoro_voice = download("cstr/kokoro-voices-GGUF", "kokoro-voice-af_heart.gguf", token)

    install_python_oracle()
    fetch_sources(token)
    t3, s3 = convert_quantize(build_dir, published_t3, published_s3)
    diff_cpu = run_diff(build_dir, t3, cpu_ref, voice, "published-cpu-ref")
    cuda_ref = dump_cuda_reference(voice)
    diff_cuda: dict = (run_diff(build_dir, t3, cuda_ref, voice, "fresh-cuda-ref")
                       if cuda_ref is not None else {"status": "unsupported-worker"})
    reference_r = generate_independent_reference(build_dir, kokoro, kokoro_voice)
    live = all_language_live(library, t3, s3, reference_r)
    rcb = clone_baseline_roundtrips(build_dir, library, t3, s3, reference_r, parakeet, titanet)
    subset = asr_subset(build_dir, parakeet)
    longform = parakeet_longform(build_dir, parakeet, token)

    summary = {
        "status": "pass", "crispasr_commit": CRISPASR_COMMIT,
        "upstream_commit": UPSTREAM_COMMIT, "model_revision": MODEL_REVISION,
        "artifact_sha256": {"t3": T3_SHA256, "s3": S3_SHA256},
        "diff_published_cpu_ref": diff_cpu, "diff_fresh_cuda_ref": diff_cuda,
        "live_languages": sorted(live), "live_language_count": len(live),
        "asr_roundtrip_languages": sorted(subset), "speaker_roundtrip": rcb,
        "parakeet_longform": longform,
    }
    (RESULTS / "SUMMARY.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    kh.step("DONE", status="pass", languages=len(live), diff_cpu=diff_cpu,
            diff_cuda=diff_cuda, parakeet_longform=longform)
    try:
        kh.export_ccache_tar()
    except Exception as exc:
        kh.step("ccache.export.warning", error=str(exc)[:200])
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        kh.step("FATAL", error=str(exc)[:500])
        traceback.print_exc()
        raise
