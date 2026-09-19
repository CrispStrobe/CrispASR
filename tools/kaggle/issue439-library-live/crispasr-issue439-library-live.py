"""Live proof for issue #439's library/session default on the reporter's model.

Runs the exact German sentence through the C session API with the WMT21 x-en
Q4_K checkpoint.  A fresh session (no beam setter) must produce the same text
as a fresh session explicitly set to beam 5.  Beam 1 is retained as a positive
control for the old greedy path; its output is reported but is not required to
loop because quantization/backend changes can alter a degenerate decode.
"""

import ctypes
import os
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

WORK = Path("/kaggle/working")
REPO = WORK / "CrispASR"
BUILD = WORK / "build"
MODELS = WORK / "models"
REF = os.environ.get("CRISPASR_REF", "fix/439-library-default")
TEXT = "Como, tatsächlich schon so ein bisschen im Stile einer Spitzenmannschaft."


def sh(cmd: str) -> None:
    print(f"$ {cmd}", flush=True)
    subprocess.run(cmd, shell=True, check=True)


sh(f"git clone --depth 1 --branch {REF} --recursive https://github.com/CrispStrobe/CrispASR {REPO}")
sys.path.insert(0, str(REPO / "tools" / "kaggle"))
import kaggle_harness as kh  # noqa: E402

kh.init_progress()
kh.resolve_hf_token()
sha = subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip()
kh.step("script.start", ref=REF, sha=sha)

kh.step("build.begin")
kh.install_build_toolchain()
cmake_cmd = (
    f"cmake {REPO} -B{BUILD} -GNinja -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON "
    + " ".join(kh.cuda_build_flags(kh.detect_cuda_arch()))
    + " "
    + " ".join(kh.cache_and_link_flags())
)
with kh.build_heartbeat("cmake-configure"):
    kh.sh_with_progress(cmake_cmd)
with kh.build_heartbeat("cmake-build"):
    kh.sh_with_progress(
        f"stdbuf -oL -eL cmake --build {BUILD} --target crispasr-cli -- -j{kh.safe_build_jobs(gpu=True)}"
    )
kh.step("build.done")

MODELS.mkdir(exist_ok=True)
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
kh.sh_with_progress("pip install -q huggingface_hub hf_transfer")
from huggingface_hub import hf_hub_download  # noqa: E402

model = Path(
    hf_hub_download(
        repo_id="cstr/wmt21-dense-24-wide-x-en-GGUF",
        filename="wmt21-dense-24-wide-x-en-q4_k.gguf",
        local_dir=str(MODELS),
        local_dir_use_symlinks=False,
    )
)
kh.step("download.done", model=model.name, bytes=model.stat().st_size)

lib_candidates = list((BUILD / "src").glob("libcrispasr.so*")) + list((BUILD / "src").glob("libwhisper.so*"))
lib_path = next((p for p in lib_candidates if p.is_file()), None)
assert lib_path, f"shared library absent; candidates={lib_candidates}"
lib = ctypes.CDLL(str(lib_path))
lib.crispasr_session_open_explicit.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int]
lib.crispasr_session_open_explicit.restype = ctypes.c_void_p
lib.crispasr_session_set_beam_size.argtypes = [ctypes.c_void_p, ctypes.c_int]
lib.crispasr_session_set_beam_size.restype = ctypes.c_int
lib.crispasr_session_translate_text.argtypes = [
    ctypes.c_void_p,
    ctypes.c_char_p,
    ctypes.c_char_p,
    ctypes.c_char_p,
    ctypes.c_int,
]
lib.crispasr_session_translate_text.restype = ctypes.c_void_p
lib.crispasr_session_translate_text_free.argtypes = [ctypes.c_void_p]
lib.crispasr_session_close.argtypes = [ctypes.c_void_p]


def translate(beam=None):
    session = lib.crispasr_session_open_explicit(str(model).encode(), b"m2m100", 4)
    assert session, "session open failed"
    try:
        if beam is not None:
            rc = lib.crispasr_session_set_beam_size(session, beam)
            assert rc == 0, f"set_beam_size({beam}) failed: {rc}"
        ptr = lib.crispasr_session_translate_text(session, TEXT.encode(), b"de", b"en", 80)
        assert ptr, f"translation returned null (beam={beam})"
        try:
            return ctypes.string_at(ptr).decode("utf-8")
        finally:
            lib.crispasr_session_translate_text_free(ptr)
    finally:
        lib.crispasr_session_close(session)


def repetition_fraction(s: str) -> float:
    words = re.findall(r"[A-Za-z]+", s.lower())
    return (max(Counter(words).values()) / len(words)) if words else 1.0


kh.step("translate.default.begin")
default = translate()
kh.step("translate.beam5.begin")
beam5 = translate(5)
kh.step("translate.beam1.begin")
beam1 = translate(1)

print("\n=== ISSUE #439 LIVE SESSION RESULTS ===", flush=True)
print(f"commit: {sha}", flush=True)
print(f"default: {default!r}", flush=True)
print(f"beam=5 : {beam5!r}", flush=True)
print(f"beam=1 : {beam1!r}", flush=True)
print(f"default==beam5: {default == beam5}", flush=True)
print(f"default repetition fraction: {repetition_fraction(default):.3f}", flush=True)
print(f"beam1 repetition fraction: {repetition_fraction(beam1):.3f}", flush=True)

passed = bool(default.strip()) and default == beam5 and repetition_fraction(default) < 0.5
kh.step(
    "result",
    passed=passed,
    default_equals_beam5=default == beam5,
    default_repetition=round(repetition_fraction(default), 4),
    beam1_repetition=round(repetition_fraction(beam1), 4),
)
kh.export_ccache_tar()
kh._push_progress_to_hf(force=True)
assert passed, "fresh library session did not behave like explicit beam 5, or still repeated"
kh.step("script.end", passed=True)

