# %% [markdown]
# # CrispASR — F5-TTS reference dump + per-stage validation + HF upload
#
# F5-TTS was never numerically validated (no reference archive, no diff-harness
# branch). This kernel closes that gap on Kaggle's CPU notebook:
#   1. runs the upstream SWivid/F5-TTS PyTorch forward and dumps per-stage
#      ground-truth intermediates -> f5-tts-ref.gguf
#   2. builds test-f5-tts, injects the reference noise (ode_step_0) + ref_mel,
#      runs the C++ synthesis, and prints the per-stage cosine table
#   3. uploads f5-tts-ref.gguf to cstr/f5-tts-GGUF/diff-harness-ref/
#
# CPU only (the reference is deterministic on CPU). Torch is pre-installed.
# Trigger from Kaggle UI ("Save Version -> Run All").

# %% [code]
import os, sys, subprocess, shutil
from pathlib import Path

os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

WORK = Path("/kaggle/working")
REPO = WORK / "CrispASR"
BUILD = Path("/kaggle/temp/build")
BRANCH = os.environ.get("CRISPASR_REF", "f5-ref-dump")
REF_GGUF = WORK / "f5-tts-ref.gguf"
HF_REPO = "cstr/f5-tts-GGUF"
REF_PATH_IN_REPO = "diff-harness-ref/f5-tts-ref.gguf"

# %% [code]
# ── Cell 1: clone repo + harness ──
if REPO.exists():
    shutil.rmtree(REPO)
subprocess.check_call(
    ["git", "clone", "--recurse-submodules", "--shallow-submodules", "--depth", "1",
     "--branch", BRANCH, "https://github.com/CrispStrobe/CrispASR.git", str(REPO)]
)
sys.path.insert(0, str(REPO / "tools" / "kaggle"))
import kaggle_harness as kh  # noqa: E402

kh.init_progress()
token = kh.resolve_hf_token()
kh.step("cloned", branch=BRANCH)

# %% [code]
# ── Cell 2: torch reference env (torch pre-installed; add only the rest) ──
subprocess.check_call([
    sys.executable, "-m", "pip", "install", "--quiet",
    "torchaudio", "safetensors", "gguf", "vocos", "f5-tts",
    "huggingface_hub",
])
kh.step("deps_installed")

# %% [code]
# ── Cell 3: download checkpoint (SWivid/F5-TTS) + Vocos vocoder ──
from huggingface_hub import snapshot_download, hf_hub_download  # noqa: E402

scratch = Path("/tmp/f5-models")
scratch.mkdir(parents=True, exist_ok=True)
# Checkpoint: F5TTS_v1_Base/model_1250000.safetensors (1.35 GB) + vocab (in pkg).
mdir = Path(snapshot_download("SWivid/F5-TTS", cache_dir=str(scratch), token=token,
                              allow_patterns=["F5TTS_v1_Base/*"]))
# Vocos mel-24khz: config.yaml + pytorch_model.bin. Reference reads F5_TTS_VOCOS_DIR.
vdir = Path(snapshot_download("charactr/vocos-mel-24khz", cache_dir=str(scratch), token=token))
os.environ["F5_TTS_VOCOS_DIR"] = str(vdir)
kh.step("models_downloaded", ckpt=str(mdir), vocos=str(vdir))

# %% [code]
# ── Cell 4: run the PyTorch reference dump -> f5-tts-ref.gguf ──
# Fixed, matched params so the C++ side can reproduce the trajectory.
os.environ.update({
    "F5_TTS_SYN_TEXT": "Hello world.",
    "F5_TTS_REF_TEXT": "",
    "F5_TTS_SEED": "42",
    "F5_TTS_STEPS": "32",
    "F5_TTS_CFG": "2.0",
    "F5_TTS_SWAY": "-1.0",
})
subprocess.check_call([
    sys.executable, "tools/dump_reference.py", "--backend", "f5-tts",
    "--model-dir", str(mdir), "--audio", "samples/jfk.wav",
    "--output", str(REF_GGUF),
], cwd=str(REPO))
assert REF_GGUF.exists(), "reference dump did not produce the gguf"
kh.step("refdump_done", size_mib=round(REF_GGUF.stat().st_size / 1024 / 1024, 2))

# %% [code]
# ── Cell 5: build test-f5-tts (the self-contained comparator) ──
kh.install_build_toolchain()
subprocess.check_call(
    ["cmake", "-G", "Ninja", "-S", str(REPO), "-B", str(BUILD),
     "-DCMAKE_BUILD_TYPE=Release", "-DGGML_CUDA=OFF",
     "-DCRISPASR_BUILD_TESTS=ON"] + kh.cache_and_link_flags()
)
kh.sh_with_progress(f"cmake --build {BUILD} --target test-f5-tts -j{kh.safe_build_jobs(gpu=False)}")
testbin = str(next(BUILD.rglob("test-f5-tts")))
model_gguf = hf_hub_download(HF_REPO, "f5-tts-v1-base-f16.gguf", local_dir=str(WORK), token=token)
kh.step("built", testbin=testbin)

# %% [code]
# ── Cell 6: run C++ with injected ref noise+mel, capture the cosine table ──
dump_dir = WORK / "cppdump"
dump_dir.mkdir(exist_ok=True)
proc = subprocess.run(
    [testbin, model_gguf, "Hello world.", str(WORK / "out.wav"),
     "--ref-gguf", str(REF_GGUF), "--dump", str(dump_dir),
     "--ref-text", "", "--seed", "42"],
    cwd=str(REPO), capture_output=True, text=True,
)
print("=== test-f5-tts stdout ===\n" + proc.stdout)
print("=== test-f5-tts stderr (tail) ===\n" + "\n".join(proc.stderr.splitlines()[-40:]))
# Persist the validation table as an artifact next to the ref.
(WORK / "f5-tts-validation.txt").write_text(proc.stdout + "\n---STDERR---\n" + proc.stderr)
kh.step("compared", rc=proc.returncode)

# %% [code]
# ── Cell 7: upload the reference gguf to the diff-harness-ref/ path ──
from huggingface_hub import HfApi  # noqa: E402

api = HfApi(token=token)
api.create_repo(HF_REPO, exist_ok=True, repo_type="model")
api.upload_file(
    path_or_fileobj=str(REF_GGUF),
    path_in_repo=REF_PATH_IN_REPO,
    repo_id=HF_REPO, repo_type="model",
    commit_message="Add F5-TTS diff-harness reference dump (#294)",
)
# Also archive the validation table so the cosines are visible on the repo.
api.upload_file(
    path_or_fileobj=str(WORK / "f5-tts-validation.txt"),
    path_in_repo="diff-harness-ref/f5-tts-validation.txt",
    repo_id=HF_REPO, repo_type="model",
    commit_message="F5-TTS per-stage validation table (#294)",
)
kh.step("uploaded", repo=HF_REPO, path=REF_PATH_IN_REPO)
print(f"DONE: uploaded {REF_PATH_IN_REPO} to {HF_REPO}")
