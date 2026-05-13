# ─────────────────────────── cell 0 (markdown) ───────────────────────────
# # CrispASR — full-suite regression / re-bake (Kaggle)
#
# Two modes, picked by a flag at the top of cell 1:
#
# **`MODE = "validate"`** (default) — same contract as the nightly
# GitHub Actions workflow, but for every backend in one run on faster
# infra. For each `tests/regression/manifest.json` entry:
#   1. Download the GGUF under test at the pinned HF revision.
#   2. Download the reference dump from
#      `cstr/crispasr-regression-fixtures` at the pinned revision.
#   3. Run `crispasr` and assert the transcript matches
#      `expected_transcript` byte-for-byte.
#   4. Run `crispasr-diff` and assert per-stage `cos_min ≥ threshold`.
# Use this to confirm a release branch before tagging, or to validate
# any backend that's too heavy for the nightly GH Actions runner
# (vibevoice 7 GB, voxtral4b 4 GB, …).
#
# **`MODE = "rebake"`** — regenerates the reference dumps from the
# real Python source models (NeMo / transformers / torch) for every
# backend. Stages them locally, optionally uploads to HF. The new
# `cstr/crispasr-regression-fixtures` commit SHA is printed at the
# end; the maintainer pastes it into `manifest.json`'s
# `fixtures.revision`. Run this whenever:
#   - A reference module changes (`tools/reference_backends/...`).
#   - A new backend is added.
#   - Upstream PyTorch / NeMo / transformers bumps with a known
#     numerical effect we want to accept as the new baseline.
# Re-bake is intentional, never silent — the upload step is gated
# on `UPLOAD=True` and prints the manifest patch for review.
#
# Why Kaggle, not GitHub Actions:
#   - Real ML stack (NeMo, transformers, torch) totals 5-10 GB; ten
#     minutes to install per nightly run is unaffordable on GH free.
#   - ~1 Gbit pipe + GPU vs ~250 Mbit on GH Actions makes the long
#     downloads tolerable.
#   - 20 GB scratch fits the heavy backends that don't fit on GH.
#
# Requirements:
# - Kaggle accelerator: any (CPU works; GPU only matters if you
#   want to validate the CUDA path of CrispASR).
# - Internet ON (model downloads + optional HF upload).
# - Optional Kaggle secrets:
#     `HF_TOKEN` — required for `MODE="rebake"` + `UPLOAD=True`.
#                  Read-only token is fine for `MODE="validate"`
#                  (only public HF repos are read).
#     `GH_TOKEN` — to post a summary as a comment on the latest
#                  main commit. Optional.

# ─────────────────────────── cell 1 (code) ───────────────────────────
# ── Configuration ──────────────────────────────────────────────────────────
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# ── Unbuffered I/O + step checkpointing ──────────────────────────────
# Kaggle persists logs only at cell/process end, so a hang past stdout
# buffer fill is invisible until the kernel actually terminates. Force
# line-buffered stdout/stderr and write a per-step JSONL checkpoint to
# /kaggle/working/progress.jsonl. Even when a run hangs, fetching the
# tiny progress.jsonl via `kaggle kernels output --file-pattern
# 'progress.jsonl'` shows the last completed step.
os.environ["PYTHONUNBUFFERED"] = "1"
try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except (AttributeError, ValueError):
    pass

_PROGRESS_PATH = Path("/kaggle/working/progress.jsonl")
_T0 = time.time()


def step(name: str, **extra) -> None:
    """Append one progress checkpoint. Cheap (one JSONL write); call
    liberally — before/after heavy operations, around subprocess.run."""
    rec = {
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "elapsed_s": round(time.time() - _T0, 2),
        "step": name,
        **extra,
    }
    try:
        _PROGRESS_PATH.parent.mkdir(parents=True, exist_ok=True)
        with _PROGRESS_PATH.open("a") as f:
            f.write(json.dumps(rec) + "\n")
    except Exception:
        pass
    print(f"[step {rec['elapsed_s']:>7.1f}s] {name}" +
          (f"  {extra}" if extra else ""), flush=True)


step("script.start")

MODE = os.environ.get("CRISPASR_REGRESSION_MODE", "validate")  # or "rebake"

# Only consulted in rebake mode.
UPLOAD = os.environ.get("CRISPASR_REGRESSION_UPLOAD", "0") == "1"

# Restrict to a subset of backends (comma-separated names). Empty == all.
BACKEND_FILTER = os.environ.get("CRISPASR_REGRESSION_BACKENDS", "").strip()

# Build flags. Default to CPU; flip to "cuda" to test the GPU path.
BUILD_FLAVOUR = os.environ.get("CRISPASR_REGRESSION_BUILD", "cpu")  # cpu | cuda

# CrispASR commit to test. Default to main; pin a SHA to bisect.
CRISPASR_REF = os.environ.get("CRISPASR_REF", "main")

# ── Workspace layout ──────────────────────────────────────────────────────
WORK = Path("/kaggle/working")
REPO = WORK / "CrispASR"
BUILD = WORK / "build"
HF_CACHE = WORK / "hf_cache"
RESULTS = WORK / "results"
REBAKE_STAGE = WORK / "rebake-stage"

for d in (HF_CACHE, RESULTS, REBAKE_STAGE):
    d.mkdir(parents=True, exist_ok=True)

# Funnel all HF downloads into the same cache so validate-after-rebake
# in the same notebook session is a free re-read.
os.environ["HF_HOME"] = str(HF_CACHE)
os.environ["HUGGINGFACE_HUB_CACHE"] = str(HF_CACHE)

print(f"crispasr-regression {datetime.now().strftime('%Y-%m-%d %H:%M')}")
print(f"  MODE             = {MODE}")
print(f"  BUILD_FLAVOUR    = {BUILD_FLAVOUR}")
print(f"  CRISPASR_REF     = {CRISPASR_REF}")
print(f"  BACKEND_FILTER   = {BACKEND_FILTER or '(all)'}")
print(f"  UPLOAD           = {UPLOAD}")

# ─────────────────────────── cell 2 (code) ───────────────────────────
step("cell_2_begin")
# ── Wire HF auth, install Python deps ─────────────────────────────────────
def kaggle_secret(name: str) -> str | None:
    """Pull a Kaggle secret if available, with verbose diagnostics if
    not. The previous silent fall-back to anonymous made a missing
    secret look identical to a missing-attach-toggle, which burned us
    on chr1str/crispasr-auto-rebake-refs.

    Kaggle injects `KAGGLE_USER_SECRETS_TOKEN` (a JWT) into the runtime
    ONLY when at least one secret is attached to the kernel. So if
    that env var is absent, no secret can possibly be read regardless
    of what the account dashboard says — the kernel-side "Add-ons →
    Secrets" Attach toggle is the gate that controls injection.
    Surfacing this distinction up front saves a debugging round.
    """
    try:
        from kaggle_secrets import UserSecretsClient
    except ImportError as exc:
        print(f"kaggle_secret({name!r}): kaggle_secrets module unimportable "
              f"({type(exc).__name__}: {exc}). Are we actually inside a "
              f"Kaggle kernel?", flush=True)
        return None
    has_jwt = bool(os.environ.get("KAGGLE_USER_SECRETS_TOKEN"))
    if not has_jwt:
        print(f"kaggle_secret({name!r}): KAGGLE_USER_SECRETS_TOKEN env var is "
              f"MISSING. Kaggle only injects it when at least one secret is "
              f"ATTACHED to the kernel (Add-ons → Secrets → Attach toggle). "
              f"You probably added the secret on your account settings page "
              f"but didn't open the kernel's per-notebook Secrets pane and "
              f"toggle Attach for {name!r}.", flush=True)
        return None
    try:
        return UserSecretsClient().get_secret(name)
    except Exception as exc:
        # JWT is present but the specific secret named is still missing
        # or unreadable. Could be a label typo, propagation lag, or a
        # different-secret attached.
        print(f"kaggle_secret({name!r}): JWT present but secret read failed "
              f"({type(exc).__name__}: {exc}). Possible causes: label typo "
              f"(case-sensitive), or a different secret is attached.",
              flush=True)
        return None


# Dump Kaggle-injected env keys (no values) so we can see what auth /
# secret machinery the runtime actually has. Helps distinguish a
# missing Attach toggle from a Kaggle-side service flake.
_kaggle_env_keys = sorted(k for k in os.environ if k.startswith("KAGGLE"))
print(f"Kaggle env keys present: {_kaggle_env_keys}", flush=True)

# Source 1: a Kaggle "Variable" (Add-ons → Variables) sets an env var
# directly. Always available in script kernels, no service call needed.
env_hf = os.environ.get("HF_TOKEN")
if env_hf:
    print("HF auth: HF_TOKEN read from env var (Kaggle Variable, env, "
          "or shell). Skipping UserSecretsClient.", flush=True)
    hf_token = env_hf
else:
    # Source 2: a Kaggle "Secret" (Add-ons → Secrets), read via
    # UserSecretsClient. Requires JWT injection (any attached secret),
    # and the service endpoint must be reachable from this runtime.
    # Script-kernel + batch-run combinations sometimes flake on the
    # service call even when the JWT is present — if you see that,
    # add HF_TOKEN as a Variable instead and re-push.
    hf_token = kaggle_secret("HF_TOKEN")
if hf_token:
    os.environ["HF_TOKEN"] = hf_token
    os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token
    print("HF auth: token present (will verify next)", flush=True)
else:
    print("HF auth: anonymous (rebake+upload will fail without HF_TOKEN). "
          "If you've added HF_TOKEN as a Kaggle Secret and the Attach "
          "toggle is on but it still doesn't read, try adding it as a "
          "Kaggle Variable instead (Add-ons → Variables): script-type "
          "kernels in batch mode sometimes can't reach the secrets API "
          "service.", flush=True)

# Need huggingface_hub before we can preflight the token. Pulled here
# (small, ~MB) even if the token check ends up failing — the cost of
# the import is negligible against the fail-fast benefit.
subprocess.check_call([
    sys.executable, "-m", "pip", "install", "--quiet",
    "huggingface_hub",
])

# ── Preflight: prove the token / fixture chain works before we spend
#    10 minutes building + downloading models we'll never use. Better
#    to die loudly in cell 2 than 25 minutes later in cell 7. ────────
def preflight_hf() -> None:
    from huggingface_hub import HfApi
    from huggingface_hub.errors import HfHubHTTPError

    api = HfApi(token=hf_token) if hf_token else HfApi()

    # 1. If a token is present, prove it's valid and tell us whose it is.
    if hf_token:
        try:
            info = api.whoami()
            user = info.get("name") or info.get("fullname") or "?"
            orgs = [o.get("name") for o in info.get("orgs", []) if o.get("name")]
            print(f"HF auth: token OK — user={user!r}  orgs={orgs}")
        except HfHubHTTPError as exc:
            raise SystemExit(
                f"HF auth: token REJECTED by /api/whoami-v2 ({exc}).\n"
                f"  Generate a fresh token at https://huggingface.co/settings/tokens\n"
                f"  and store it as the Kaggle secret HF_TOKEN (read+write for rebake)."
            )

    # 2. For rebake+upload, the token must additionally have *write*
    #    access to the fixtures repo. Probe via repo_info(); HF returns
    #    a `private`/`gated` flag we can sanity-check, and the call
    #    itself 401s if the token can't see private repos when needed.
    #    Writability is harder to introspect cleanly — the cheapest
    #    proof is the upload step itself — but we can at least confirm
    #    the repo exists and the user can see it.
    fixtures_repo = "cstr/crispasr-regression-fixtures"
    try:
        info = api.repo_info(repo_id=fixtures_repo, repo_type="model")
        print(f"HF fixtures: {fixtures_repo} reachable (last_modified={info.last_modified})")
    except HfHubHTTPError as exc:
        msg = (
            f"HF fixtures: cannot reach {fixtures_repo} ({exc}).\n"
            f"  validate mode CAN'T proceed without the fixtures repo;\n"
            f"  rebake+upload CAN'T proceed without write access to it."
        )
        raise SystemExit(msg)

    # The UPLOAD setting is mutated in-place below if needed so the
    # subsequent rebake path skips api.upload_folder() gracefully.
    global UPLOAD
    if MODE == "rebake" and UPLOAD and not hf_token:
        # Don't die — let the rebake stage refs anyway so they can be
        # fetched + uploaded from local (Kaggle Secrets API has been
        # flaky in batch-trigger runs; we don't want a transient
        # secrets-API issue to lose the whole compute cycle). The
        # fetch-and-upload.sh script picks up from /kaggle/working/
        # rebake-stage/ and publishes from the maintainer's box.
        print(
            "WARN: rebake+UPLOAD=1 requested but HF_TOKEN unreadable. "
            "Downgrading to UPLOAD=0; refs will stage to "
            "/kaggle/working/rebake-stage/. Pick them up locally with:\n"
            "  ./tools/kaggle/rebake/fetch-and-upload.sh\n",
            flush=True,
        )
        UPLOAD = False
        # Best-effort write probe: open a no-op preupload (computes
        # remote-cas hash for a 1-byte blob; HF returns 401 if we
        # can't write, OK otherwise). Cheaper than actually committing.
        try:
            api.preupload_lfs_files(
                repo_id=fixtures_repo,
                repo_type="model",
                additions=[],  # zero files — just exercises the auth check
            )
            print(f"HF fixtures: write access to {fixtures_repo} confirmed")
        except HfHubHTTPError as exc:
            raise SystemExit(
                f"HF fixtures: token can READ {fixtures_repo} but write "
                f"probe failed ({exc}). Generate a write-scoped token at "
                f"https://huggingface.co/settings/tokens."
            )
        except Exception as exc:
            # preupload_lfs_files API may shift; fall back to a warning
            # rather than blocking the rebake. The real upload step
            # will surface any actual auth error.
            print(f"HF fixtures: write probe inconclusive ({type(exc).__name__}: "
                  f"{exc}). Proceeding; real upload will catch it.")


preflight_hf()
if MODE == "rebake":
    # The heavy ML stack only matters when re-baking. validate mode
    # never touches NeMo / transformers / torch.
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "--quiet",
        "nemo_toolkit[asr]", "transformers", "torch", "torchaudio",
        "numpy", "gguf",
    ])

# ─────────────────────────── cell 3 (code) ───────────────────────────
step("cell_3_begin")
# ── Clone + build CrispASR ────────────────────────────────────────────────
def sh(cmd: str, cwd: Path | None = None) -> None:
    print(f"$ {cmd}")
    subprocess.check_call(cmd, shell=True, cwd=str(cwd) if cwd else None)


if not REPO.exists():
    sh(f"git clone --recursive https://github.com/CrispStrobe/CrispASR.git {REPO}")
sh(f"git fetch origin && git checkout {CRISPASR_REF}", cwd=REPO)
sh("git submodule update --init --recursive", cwd=REPO)

# ── Fast-build toolchain — match the perf-A/B kernel's pattern.
#    ccache cuts re-build cost from ~5 min to ~30 s on cache-hot runs
#    (cache survives across notebook re-runs because /kaggle/working/
#    persists). mold halves link time vs ld. ninja is the build
#    driver itself. The CCACHE_DIR + CCACHE_MAXSIZE lines are the
#    exact recipe from tools/kaggle-issue81-cuda-ab.py — both kernels
#    share that working dir if the user opens them in the same
#    Kaggle session. ────────────────────────────────────────────
import shutil as _shutil

print("Installing build toolchain (ninja, ccache, mold)…", flush=True)
sh("apt-get update -qq && apt-get install -y --no-install-recommends "
   "cmake ninja-build g++ libopenblas-dev jq ccache mold || true")

HAS_CCACHE = _shutil.which("ccache") is not None
HAS_MOLD = _shutil.which("mold") is not None
HAS_NINJA = _shutil.which("ninja") is not None

# Persist ccache across runs in /kaggle/working/.ccache. Kaggle wipes
# /tmp + /root between runs but keeps /kaggle/working, so this is the
# only sane location for a cache.
CCACHE_DIR = WORK / ".ccache"
CCACHE_DIR.mkdir(exist_ok=True)
os.environ["CCACHE_DIR"] = str(CCACHE_DIR)
os.environ["CCACHE_MAXSIZE"] = "5G"
if HAS_CCACHE:
    subprocess.run("ccache -M 5G && ccache -z",
                   shell=True, capture_output=True)

print(f"  ninja={HAS_NINJA}  ccache={HAS_CCACHE}  mold={HAS_MOLD}  "
      f"CCACHE_DIR={CCACHE_DIR}", flush=True)

build_flags = []
if BUILD_FLAVOUR == "cuda":
    build_flags.append("-DGGML_CUDA=ON")
if HAS_CCACHE:
    build_flags += [
        "-DCMAKE_C_COMPILER_LAUNCHER=ccache",
        "-DCMAKE_CXX_COMPILER_LAUNCHER=ccache",
        "-DCMAKE_CUDA_COMPILER_LAUNCHER=ccache",
    ]
if HAS_MOLD:
    # `-fuse-ld=mold` on the linker line cuts ~30s off our 320-object
    # link. CMake's flag plumbing is split between EXE / SHARED / MODULE
    # link types; set all three so the static archive driver too.
    for kind in ("EXE", "SHARED", "MODULE"):
        build_flags.append(f"-DCMAKE_{kind}_LINKER_FLAGS=-fuse-ld=mold")

sh(
    f"cmake -S {REPO} -B {BUILD} -G Ninja "
    f"-DCMAKE_BUILD_TYPE=Release "
    f"-DCRISPASR_BUILD_TESTS=OFF "
    f"-DCRISPASR_BUILD_EXAMPLES=ON "
    f"-DCRISPASR_BUILD_SERVER=OFF "
    + " ".join(build_flags)
)
# CMake target `crispasr-cli` produces bin/crispasr (target/output names
# intentionally diverge per examples/cli/CMakeLists.txt:12). Asking for
# target `crispasr` here builds only the library, leaving bin/crispasr
# absent — exactly what burned the GH regression workflow on its first
# run (commit 08d1872f) and what just burned this Kaggle one.
sh(f"cmake --build {BUILD} --target crispasr-cli crispasr-diff -j$(nproc)")

if HAS_CCACHE:
    print("ccache stats after build:", flush=True)
    subprocess.run("ccache -s | grep -E 'cache hit|cache miss|files in cache|cache size'",
                   shell=True)

# ─────────────────────────── cell 4 (code) ───────────────────────────
step("cell_4_begin")
# ── Load manifest + backend filter ────────────────────────────────────────
MANIFEST_PATH = REPO / "tests" / "regression" / "manifest.json"
with MANIFEST_PATH.open() as f:
    MANIFEST = json.load(f)

want = set(b.strip() for b in BACKEND_FILTER.split(",") if b.strip())
BACKENDS = [
    b for b in MANIFEST["backends"]
    if not want or b["name"] in want
]
if want:
    missing = want - {b["name"] for b in BACKENDS}
    if missing:
        print(f"WARN: requested backends not in manifest: {sorted(missing)}")

print(f"\nProcessing {len(BACKENDS)} backend(s):")
for b in BACKENDS:
    size_mb = b["gguf"].get("approx_size_mb", "?")
    print(f"  - {b['name']:30s} (gguf ~{size_mb} MB)")


# ─────────────────────────── cell 5 (code) ───────────────────────────
step("cell_5_begin")
# ── VALIDATE mode: download pinned artifacts, run regression ──────────────
def run_validate() -> list[dict]:
    """Per-backend validate. Returns one result dict per backend."""
    sys.path.insert(0, str(REPO / "tests" / "regression"))
    import run_one  # noqa: E402 — added to path above

    results = []
    for entry in BACKENDS:
        name = entry["name"]
        print(f"\n========== validate :: {name} ==========")
        t0 = time.time()
        try:
            from huggingface_hub import hf_hub_download
            gguf_local = Path(hf_hub_download(
                repo_id=entry["gguf"]["repo"],
                filename=entry["gguf"]["file"],
                revision=entry["gguf"]["revision"],
            ))
            ref_local = Path(hf_hub_download(
                repo_id=MANIFEST["fixtures"]["repo"],
                filename=entry["fixture_ref_path"],
                revision=MANIFEST["fixtures"]["revision"],
            ))
            if "fixture_sample_path" in entry:
                sample = Path(hf_hub_download(
                    repo_id=MANIFEST["fixtures"]["repo"],
                    filename=entry["fixture_sample_path"],
                    revision=MANIFEST["fixtures"]["revision"],
                ))
            else:
                sample = REPO / entry["sample"]

            crispasr_bin = BUILD / "bin" / "crispasr"
            diff_bin = BUILD / "bin" / "crispasr-diff"

            actual = run_one.run_transcript(crispasr_bin, gguf_local, sample)
            transcript_ok = (actual == entry["expected_transcript"])
            stages = run_one.run_diff(
                diff_bin, entry["backend_id"], gguf_local, ref_local, sample)
            passes, fails, missing, extras = run_one.evaluate_stage_thresholds(
                stages, entry["diff_thresholds"])
            ok = transcript_ok and not fails and not missing
            results.append({
                "backend": name,
                "mode": "validate",
                "ok": ok,
                "elapsed_s": round(time.time() - t0, 2),
                "transcript_match": transcript_ok,
                "transcript_actual": actual if not transcript_ok else None,
                "stages": {s: stages.get(s) for s in entry["diff_thresholds"]},
                "extras": dict(extras),
                "missing": missing,
            })
            print(f"  -> ok={ok}  transcript={transcript_ok}  "
                  f"passes={len(passes)}  fails={len(fails)}  missing={len(missing)}")
        except Exception as exc:
            results.append({
                "backend": name,
                "mode": "validate",
                "ok": False,
                "elapsed_s": round(time.time() - t0, 2),
                "error": f"{type(exc).__name__}: {exc}",
            })
            print(f"  -> ERROR  {type(exc).__name__}: {exc}")

    return results


# ─────────────────────────── cell 6 (code) ───────────────────────────
step("cell_6_begin")
# ── REBAKE mode: run real Python references, stage new ref.gguf files ────
def run_rebake() -> list[dict]:
    """Per-backend re-bake. Writes new ref.gguf files into REBAKE_STAGE
    at the manifest's `fixture_path`. Does NOT upload — that's a
    separate gated step.
    """
    results = []
    for entry in BACKENDS:
        name = entry["name"]
        print(f"\n========== rebake :: {name} ==========")
        t0 = time.time()
        # `backend_id` is the registered name in tools/dump_reference.py;
        # `fixture_path` is what `manifest.json` says we'll ship.
        out_path = REBAKE_STAGE / entry["fixture_path"]
        out_path.parent.mkdir(parents=True, exist_ok=True)

        # Source-model spec. Per-backend modules know how to interpret
        # `--model-dir` (HF id or local path). The manifest carries
        # `source_model` exactly for re-bake — fall back is a hard
        # error rather than a guess, because guessing a wrong NeMo
        # checkpoint silently changes what we baseline against.
        source = entry.get("source_model")
        if not source:
            results.append({
                "backend": name,
                "mode": "rebake",
                "ok": False,
                "elapsed_s": 0.0,
                "error": "manifest entry has no `source_model`; "
                         "add it before re-baking",
            })
            print(f"  -> SKIP (no source_model)")
            continue
        # For re-bake, the sample WAV must be on disk so the Python
        # reference can read it. Pull from the fixtures HF repo if the
        # manifest points there; otherwise expect it in-tree.
        if "fixture_sample_path" in entry:
            from huggingface_hub import hf_hub_download
            sample = Path(hf_hub_download(
                repo_id=MANIFEST["fixtures"]["repo"],
                filename=entry["fixture_sample_path"],
                revision=MANIFEST["fixtures"]["revision"],
            ))
        else:
            sample = REPO / entry["sample"]

        cmd = [
            sys.executable, "-u", str(REPO / "tools" / "dump_reference.py"),
            "--backend", entry["backend_id"],
            "--model-dir", source,
            "--audio", str(sample),
            "--output", str(out_path),
        ]
        try:
            subprocess.check_call(cmd, cwd=str(REPO))
            results.append({
                "backend": name,
                "mode": "rebake",
                "ok": True,
                "elapsed_s": round(time.time() - t0, 2),
                "out_path": str(out_path),
                "out_size_b": out_path.stat().st_size,
            })
            print(f"  -> wrote {out_path} ({out_path.stat().st_size / 1024:.1f} KiB)")
        except subprocess.CalledProcessError as exc:
            results.append({
                "backend": name,
                "mode": "rebake",
                "ok": False,
                "elapsed_s": round(time.time() - t0, 2),
                "error": f"dump_reference exit={exc.returncode}",
            })

    return results


# ─────────────────────────── cell 7 (code) ───────────────────────────
step("cell_7_begin")
# ── Dispatch + upload ─────────────────────────────────────────────────────
if MODE == "validate":
    RESULTS_DATA = run_validate()
elif MODE == "rebake":
    RESULTS_DATA = run_rebake()
else:
    raise SystemExit(f"unknown MODE={MODE!r}; want 'validate' or 'rebake'")

# Write a single JSON Lines artifact for downstream diffing.
results_jsonl = RESULTS / f"results-{MODE}-{datetime.now().strftime('%Y%m%dT%H%M%S')}.jsonl"
with results_jsonl.open("w") as f:
    for r in RESULTS_DATA:
        f.write(json.dumps(r) + "\n")
print(f"\nResults: {results_jsonl}")

# Summary line for stdout (so a Kaggle screenshot is self-contained).
n_ok = sum(1 for r in RESULTS_DATA if r.get("ok"))
n_fail = sum(1 for r in RESULTS_DATA if not r.get("ok"))
print(f"\nSUMMARY  mode={MODE}  ok={n_ok}/{len(RESULTS_DATA)}  fail={n_fail}")
for r in RESULTS_DATA:
    flag = "✓" if r.get("ok") else "✗"
    print(f"  {flag} {r['backend']:30s} {r['elapsed_s']:6.1f}s  {r.get('error', '')}")

if MODE == "rebake" and UPLOAD:
    if any(not r.get("ok") for r in RESULTS_DATA):
        raise SystemExit(
            "rebake had failures; refusing to upload. Inspect failures, "
            "fix, re-run rebake with the same workspace until all backends "
            "pass, then re-run with UPLOAD=1.")
    from huggingface_hub import HfApi
    api = HfApi()
    print(f"\nUploading {REBAKE_STAGE}/ → cstr/crispasr-regression-fixtures")
    # Use upload_folder so the structure mirrors the staging dir
    # exactly. delete_patterns kept empty: never silently delete a
    # ref.gguf that's still in the manifest.
    commit_info = api.upload_folder(
        repo_id="cstr/crispasr-regression-fixtures",
        repo_type="model",
        folder_path=str(REBAKE_STAGE),
        commit_message=f"rebake {len(RESULTS_DATA)} backend(s) — "
                       f"crispasr ref {CRISPASR_REF}",
    )
    print(f"\nNew fixtures commit: {commit_info.oid}")
    print(f"  → bump manifest.json's fixtures.revision to {commit_info.oid}")
    print(f"  → https://huggingface.co/cstr/crispasr-regression-fixtures/commit/{commit_info.oid}")

# Exit non-zero so a Kaggle scheduled run shows up as failed when
# anything regressed. Without this, Kaggle treats any successful
# notebook execution as "green" regardless of cell-internal state.
sys.exit(n_fail)
