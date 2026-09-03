"""CrispASR Kaggle kernel harness — the shared "gold standard" helpers.

No single kernel had all of these; this module is the union, distilled
from the best parts of `crispasr-regression.py` (logging, heartbeat,
ccache/mold, 3-tier auth) and `fusion-ab` / `kaggle-issue81-cuda-ab`
(CUDA arch auto-detect). Every kernel clones the CrispASR repo early,
so each can import this right after the clone:

    import sys
    sys.path.insert(0, f"{REPO}/tools/kaggle")
    import kaggle_harness as kh
    kh.init_progress()                       # line-buffered I/O + JSONL
    kh.install_build_toolchain()             # ninja + ccache + mold
    arch = kh.detect_cuda_arch()             # "75" on T4, "80" on A100…
    flags = kh.cuda_build_flags(arch) + kh.cache_and_link_flags()
    with kh.build_heartbeat("cmake.build"):
        kh.sh_with_progress(f"stdbuf -oL -eL cmake --build {BUILD} "
                            f"--target {target} -j{kh.safe_build_jobs(gpu=True)}")
    token = kh.resolve_hf_token()            # env → Secret(retry) → dataset
    # Uploading kernels: kh.resolve_hf_token(require=True) FIRST, before any
    # build/compute — a missing token then aborts in seconds instead of the
    # finished run losing every artifact to upload 401s.

Design rules (so a kernel never dies *because of* the harness):
  * pure stdlib at import time — huggingface_hub / kaggle_secrets are
    imported lazily inside the functions that need them;
  * every external probe (nvidia-smi, apt, ccache, Secrets API, /proc)
    is wrapped with a safe fallback;
  * imports cleanly off-Kaggle (macOS dev box) so kernels can be
    `python -c 'import kaggle_harness'` smoke-tested before push.
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

# ─────────────────────────── logging / progress ────────────────────────────

_T0 = time.time()
_PROGRESS_PATH = Path("/kaggle/working/progress.jsonl")

# Live progress mirror. `kaggle kernels output` gates /kaggle/working
# access until the run terminates, so a hung/OOM'd run is otherwise only
# visible in the browser UI. Rolling progress.jsonl up to a public HF
# dataset gives programmatic mid-run visibility (poll the dataset).
_HF_PROGRESS_REPO = "cstr/crispasr-kaggle-progress"
_HF_PROGRESS_PATH = (
    f"runs/{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    f"-{os.environ.get('KAGGLE_KERNEL_RUN_TYPE', 'local').lower()}"
    f"-{os.environ.get('KAGGLE_KERNEL_REF', 'unknown').split('/')[-1] or 'unknown'}"
    f".jsonl"
)
_HF_PUSH_INTERVAL_S = 30.0
_HF_LAST_PUSH = 0.0


def init_progress(progress_path: str | os.PathLike | None = None,
                  hf_progress_repo: str | None = None) -> None:
    """Force line-buffered stdout/stderr (Kaggle buffers parent stdout
    heavily — a hang past the buffer fill is invisible until the kernel
    terminates) and optionally override where the JSONL checkpoint and
    HF mirror live. Call once, as early as possible after import."""
    global _PROGRESS_PATH, _HF_PROGRESS_REPO
    os.environ["PYTHONUNBUFFERED"] = "1"
    try:
        sys.stdout.reconfigure(line_buffering=True)
        sys.stderr.reconfigure(line_buffering=True)
    except (AttributeError, ValueError):
        pass
    if progress_path is not None:
        _PROGRESS_PATH = Path(progress_path)
    if hf_progress_repo is not None:
        _HF_PROGRESS_REPO = hf_progress_repo


def _push_progress_to_hf(force: bool = False) -> None:
    """Best-effort upload of progress.jsonl to HF. Silent on any failure —
    progress reporting must never crash the run."""
    global _HF_LAST_PUSH
    now = time.time()
    if not force and (now - _HF_LAST_PUSH) < _HF_PUSH_INTERVAL_S:
        return
    if not os.environ.get("HF_TOKEN") or not _HF_PROGRESS_REPO:
        return
    if not _PROGRESS_PATH.exists():
        return
    try:
        from huggingface_hub import HfApi
        HfApi(token=os.environ["HF_TOKEN"]).upload_file(
            path_or_fileobj=str(_PROGRESS_PATH),
            path_in_repo=_HF_PROGRESS_PATH,
            repo_id=_HF_PROGRESS_REPO,
            repo_type="dataset",
            commit_message=f"progress @ {now - _T0:.0f}s",
        )
        _HF_LAST_PUSH = now
    except Exception:
        pass


def provenance(script_version: str, clone_dir: "str | os.PathLike | None" = None) -> dict:
    """Log BOTH halves of what a run actually exercised, because they drift
    apart silently.

    Kernels `git clone` CrispASR at runtime, so the C++ UNDER TEST is always
    fresh from main — but the HARNESS (arms, env gating, capture, pass/fail
    predicate) freezes at the last `kaggle kernels push`. `git push` does NOT
    update a Kaggle kernel. So a run can build and exercise the newest code
    while scoring it with an outdated verdict, and nothing in the log looks
    wrong (2026-09-03: a verdict fix sat correct in main while the kernel kept
    running the old script). Print the clone's SHA and the script's own version
    constant so the two halves are visible and comparable in the log.

    `script_version` should be a literal in the kernel script — bump it when
    you change the arms, the capture, or the predicate.
    """
    sha = "unknown"
    root = str(clone_dir) if clone_dir else os.environ.get("CRISPASR_CLONE", "")
    if root:
        try:
            r = subprocess.run(["git", "-C", root, "rev-parse", "--short", "HEAD"],
                               capture_output=True, text=True, timeout=60)
            if r.returncode == 0:
                sha = (r.stdout or "").strip() or "unknown"
        except Exception:
            pass
    info = {"script_version": script_version, "clone_sha": sha}
    step("provenance", **info)
    return info


def step(name: str, **extra) -> None:
    """Append one checkpoint to the local JSONL, print it (flushed), and
    roll the file up to HF (rate-limited, best-effort)."""
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
    _push_progress_to_hf()


# ─────────────────────────── shell + build streaming ───────────────────────

_BUILD_PROGRESS: dict = {"last_ninja": None, "last_tu": None, "lines": 0}
_NINJA_RE = re.compile(r"^\[(\d+)/(\d+)\]")
_TU_RE = re.compile(r"(\S+\.(?:cpp|cc|cxx|c|cu))(?::|\s|$)")


def sh(cmd: str, cwd: str | os.PathLike | None = None, check: bool = True) -> int:
    """Run a shell command, inheriting stdout/stderr. Raises on non-zero
    when check=True."""
    print(f"$ {cmd}", flush=True)
    rc = subprocess.run(cmd, shell=True, cwd=str(cwd) if cwd else None).returncode
    if check and rc != 0:
        raise subprocess.CalledProcessError(rc, cmd)
    return rc


def sh_with_progress(cmd: str, cwd: str | os.PathLike | None = None) -> None:
    """Popen the command, pipe stdout+stderr, forward every line to the
    parent stdout with explicit flush (so Kaggle's log capture sees the
    build in near-real-time), and track ninja [X/N] + last TU into
    _BUILD_PROGRESS for the heartbeat ticker to report."""
    print(f"$ {cmd}", flush=True)
    proc = subprocess.Popen(
        cmd, shell=True, cwd=str(cwd) if cwd else None,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        bufsize=1, text=True,
    )
    assert proc.stdout is not None
    try:
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            _BUILD_PROGRESS["lines"] += 1
            m = _NINJA_RE.match(line)
            if m:
                _BUILD_PROGRESS["last_ninja"] = f"{m.group(1)}/{m.group(2)}"
            m = _TU_RE.search(line)
            if m:
                _BUILD_PROGRESS["last_tu"] = m.group(1).rsplit("/", 1)[-1]
    finally:
        rc = proc.wait()
    if rc != 0:
        raise subprocess.CalledProcessError(rc, cmd)


@contextlib.contextmanager
def build_heartbeat(label: str, interval_s: float = 30.0, rss: bool = True):
    """Emit step("<label>.heartbeat") every interval_s until the block
    exits. When sh_with_progress() runs inside, each tick carries ninja's
    [X/N] + last TU; with rss=True it also reports VmRSS (MiB) and free
    disk (GB) — a climbing RSS toward the box ceiling is the signature of
    the CUDA-compile OOM, visible *before* the kill."""
    t_start = time.time()
    stop_event = threading.Event()

    def _ticker():
        while not stop_event.wait(interval_s):
            extra: dict = {"elapsed_in_block_s": round(time.time() - t_start, 1)}
            if _BUILD_PROGRESS["last_ninja"]:
                extra["ninja"] = _BUILD_PROGRESS["last_ninja"]
                extra["tu"] = _BUILD_PROGRESS["last_tu"]
                extra["lines"] = _BUILD_PROGRESS["lines"]
            if rss:
                r = rss_mib()
                if r is not None:
                    extra["rss_mib"] = r
                g = free_gb("/kaggle/working")
                if g is not None:
                    extra["free_gb"] = g
            step(f"{label}.heartbeat", **extra)

    thread = threading.Thread(target=_ticker, daemon=True, name=f"hb-{label}")
    thread.start()
    try:
        yield
    finally:
        stop_event.set()
        thread.join(timeout=1.0)


# ─────────────────────────── build toolchain / flags ───────────────────────

_HAS_CCACHE = False
_HAS_MOLD = False
_HAS_NINJA = False


def _warm_ccache_from_dataset(ccache_dir: Path) -> None:
    """Copy a ccache seed from an attached Kaggle dataset into the build
    ccache directory. Looks for `crispasr-ccache/ccache.tar` (a tar of the
    ccache dir) or a bare `crispasr-ccache/.ccache/` tree at the standard
    Kaggle mount paths. Silently no-ops if no dataset is attached.

    To create/update the dataset, call export_ccache_tar() at the end of a
    kernel and upload the single /kaggle/working/ccache.tar it writes. Attach
    via kernel-metadata.json (SAME account as the kernel — cross-account
    attach is rejected):
        "dataset_sources": ["chr1str/crispasr-ccache", ...]
    Shaves ~15 min off incremental CUDA builds.

    Kaggle AUTO-EXTRACTS archives uploaded to a dataset, so the ccache.tar you
    upload arrives mounted as a bare `.ccache/` tree. That is the normal case
    and it is handled correctly; the tar branch below only fires for a tar that
    somehow survives unextracted. What signals a broken seed is the file COUNT,
    not the layout — see the warning."""
    import tarfile
    # Owner-agnostic scan. The hard-coded owner list this replaced is the same
    # bug already fixed in kaggle_token_from_dataset(): it silently missed any
    # account not named in the list, and a missing ccache reads as a slow build
    # rather than an error. Probe for the artifact, don't guess the path.
    search_paths = [Path("/kaggle/input/crispasr-ccache")]
    inp = Path("/kaggle/input")
    if inp.exists():
        for sub in inp.iterdir():
            if not sub.is_dir():
                continue
            if sub.name == "datasets":
                for owner in sub.iterdir():  # /kaggle/input/datasets/<owner>/<slug>
                    if owner.is_dir():
                        search_paths.extend(s for s in owner.iterdir() if s.is_dir())
            else:
                search_paths.append(sub)  # /kaggle/input/<slug>
    for base in search_paths:
        tar_path = base / "ccache.tar"
        if tar_path.exists():
            try:
                with tarfile.open(tar_path, "r") as tf:
                    # The tar roots everything at ".ccache/" (both the documented
                    # `tar cf ccache.tar .ccache/` recipe and export_ccache_tar).
                    # ccache_dir is ITSELF the .ccache directory, so a plain
                    # extractall(ccache_dir) lands the objects at
                    # <ccache_dir>/.ccache/... — one level too deep, and ccache
                    # then sees an empty cache. This is why the tar path never
                    # worked and why both seed datasets ended up as loose trees
                    # (the fallback that then hit the 500-file page cap).
                    # Strip the leading component; a tar without it passes through.
                    members = []
                    for m in tf.getmembers():
                        if m.name in (".ccache", "./.ccache"):
                            continue
                        for pre in (".ccache/", "./.ccache/"):
                            if m.name.startswith(pre):
                                m.name = m.name[len(pre):]
                                break
                        if m.name and m.name != ".":
                            members.append(m)
                    tf.extractall(str(ccache_dir), members=members)
                n = sum(1 for _ in ccache_dir.rglob("*") if _.is_file())
                print(f"  ccache: warmed from {tar_path} ({n} files)", flush=True)
                if n == 0:
                    print("  ccache: ⚠ WARNING — tar extracted to 0 files; the seed "
                          "is empty or wrongly rooted (expect a cold build).",
                          flush=True)
                return
            except Exception as e:
                print(f"  ccache: failed to extract {tar_path}: {e}", flush=True)
        bare_dir = base / ".ccache"
        if bare_dir.is_dir():
            try:
                shutil.copytree(str(bare_dir), str(ccache_dir), dirs_exist_ok=True)
                n = sum(1 for _ in ccache_dir.rglob("*") if _.is_file())
                print(f"  ccache: warmed from {bare_dir} ({n} files)", flush=True)
                # A bare tree is NORMAL, not a defect: Kaggle auto-extracts any
                # archive uploaded to a dataset, so a ccache.tar you upload
                # arrives here already unpacked to .ccache/. copytree lands it
                # at the right depth, so this branch is the usual path.
                #
                # What IS diagnostic is the file COUNT. `kaggle kernels output`
                # pages at 500 files with no auto-continue, so a dataset built
                # by downloading a kernel's loose /kaggle/working tree gets
                # truncated at exactly that boundary. Both account copies were
                # found stuck at ~500 objects on 2026-07-20, silently costing
                # every CUDA build a near-cold start. A healthy CrispASR cache
                # is a few thousand objects.
                if n <= 501:
                    print(f"  ccache: ⚠ WARNING — only {n} files, at/below the "
                          "500-file kernels-output page cap. This seed is almost "
                          "certainly TRUNCATED; expect a near-cold build. Rebuild "
                          "it with tools/kaggle/ccache-refresh.", flush=True)
                return
            except Exception as e:
                print(f"  ccache: failed to copy {bare_dir}: {e}", flush=True)
    print("  ccache: no seed dataset found (cold build)", flush=True)


def export_ccache_tar(dest: str | os.PathLike = "/kaggle/working/ccache.tar") -> str | None:
    """Tar CCACHE_DIR into a SINGLE file for refreshing the ccache dataset.

    Call at the end of a kernel that did a real build, then upload `dest` as
    chr1str/crispasr-ccache (and the chr1s4 copy — cross-account attach is
    blocked, so each account needs its own).

    One file is the whole point: `kaggle kernels output` stops at 500 files
    with no auto-continue, so anything that writes a loose tree into
    /kaggle/working makes the artifact you actually want unreachable."""
    import tarfile
    src = Path(os.environ.get("CCACHE_DIR", "/kaggle/temp/.ccache"))
    if not src.is_dir():
        print(f"  ccache: nothing to export, {src} missing", flush=True)
        return None
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    n = sum(1 for _ in src.rglob("*") if _.is_file())
    try:
        with tarfile.open(dest, "w") as tf:
            tf.add(str(src), arcname=".ccache")
    except Exception as e:
        print(f"  ccache: export failed: {e}", flush=True)
        return None
    mb = dest.stat().st_size / 1e6
    print(f"  ccache: exported {n} files -> {dest} ({mb:.1f} MB)", flush=True)
    if n <= 501:
        print("  ccache: ⚠ only %d files — did the build actually run?" % n, flush=True)
    return str(dest)


def install_build_toolchain() -> dict:
    """apt-install ninja + ccache + mold (best effort) and prime the ccache.

    CCACHE_DIR goes on the EPHEMERAL layer (/kaggle/temp), never inside
    /kaggle/working. ccache writes thousands of loose files, and
    /kaggle/working is the kernel-output mount, which `kaggle kernels output`
    serves 500 files at a time with no auto-continue. Keeping the cache there
    was self-perpetuating breakage: the loose tree buried ccache.tar past the
    page cap, so refreshing the dataset downloaded a truncated tree, which was
    then uploaded as the seed, so the next build warmed from a truncated tree
    and ran near-cold — and so on. Both account copies of crispasr-ccache were
    found broken exactly this way (2026-07-20).

    Nothing is lost by moving it: a script kernel starts fresh every run, so
    /kaggle/working's persistence never helped the cache anyway — the seed
    comes from the attached dataset. Call export_ccache_tar() at the end to
    write ONE /kaggle/working/ccache.tar for refreshing that dataset.

    Returns {'ninja','ccache','mold': bool}."""
    global _HAS_CCACHE, _HAS_MOLD, _HAS_NINJA
    sh("apt-get update -qq && apt-get install -y --no-install-recommends "
       "cmake ninja-build g++ ccache mold || true", check=False)
    if not shutil.which("ninja"):
        subprocess.run("pip install -q ninja", shell=True, capture_output=True)
    _HAS_NINJA = shutil.which("ninja") is not None
    _HAS_CCACHE = shutil.which("ccache") is not None
    _HAS_MOLD = shutil.which("mold") is not None
    _scratch = Path("/kaggle/temp") if Path("/kaggle/temp").is_dir() else Path("/tmp")
    ccache_dir = Path(os.environ.get("CRISPASR_CCACHE_DIR", str(_scratch / ".ccache")))
    try:
        ccache_dir.mkdir(parents=True, exist_ok=True)
        os.environ["CCACHE_DIR"] = str(ccache_dir)
        os.environ["CCACHE_MAXSIZE"] = "5G"
        # Make the cache RELOCATABLE. ccache hashes the working directory by
        # default, so a build tree at a different path misses every object even
        # with a perfectly good cache — moving the sweep's build from
        # /kaggle/working to /kaggle/temp turned a ~3 min warm build into a full
        # 734-object cold one. NOHASHDIR drops the cwd from the hash; BASEDIR
        # rewrites absolute paths beneath it to relative, so -I flags baked into
        # the command line do not bust the hash either. Same pairing that makes
        # the cache shared across local git worktrees.
        os.environ["CCACHE_NOHASHDIR"] = "1"
        os.environ["CCACHE_BASEDIR"] = str(_scratch)
        # Warm ccache from attached dataset (chr1s4/crispasr-ccache or
        # chr1str/crispasr-ccache). Shaves ~15 min off incremental builds.
        _warm_ccache_from_dataset(ccache_dir)
        if _HAS_CCACHE:
            subprocess.run("ccache -M 5G && ccache -z", shell=True,
                           capture_output=True)
    except Exception:
        pass
    print(f"  toolchain: ninja={_HAS_NINJA} ccache={_HAS_CCACHE} "
          f"mold={_HAS_MOLD} CCACHE_DIR={ccache_dir}", flush=True)
    return {"ninja": _HAS_NINJA, "ccache": _HAS_CCACHE, "mold": _HAS_MOLD}


def crispasr_cmake_flags() -> list[str]:
    """CrispASR-specific cmake flags every Kaggle kernel wants.

    `-DCRISPASR_NO_C2PA_NATIVE=ON`: src/CMakeLists.txt builds the native C2PA
    signer from the `third_party/c2pa-audio` **git submodule**. Most kernels
    clone with `--depth 1` and init only `ggml` (or no submodule at all), so
    cmake generate dies with:

        Cannot find source file: .../third_party/c2pa-audio/src/c2pa_native.cpp
        No SOURCES given to target: crispasr_c2pa_native

    C2PA provenance signing is irrelevant to a benchmark / conversion / A-B
    kernel, and disabling it skips the target entirely — cheaper than fetching
    another submodule. Kernels that DO want C2PA should clone `--recursive`
    instead. (Confirmed fix on chr1str/crispasr-issue81-onnx-bench, 2026-07-18.)
    """
    return ["-DCRISPASR_NO_C2PA_NATIVE=ON"]


def cache_and_link_flags() -> list[str]:
    """ccache compiler-launcher flags + mold linker flags, for whatever
    install_build_toolchain() detected. Safe to call even if it wasn't —
    returns [] for anything unavailable.

    Also folds in crispasr_cmake_flags() so the ~20 existing kernels that
    already call this get the c2pa-submodule fix without an edit each. New
    kernels should call crispasr_cmake_flags() explicitly for clarity.
    """
    flags: list[str] = list(crispasr_cmake_flags())
    if _HAS_CCACHE:
        flags += [
            "-DCMAKE_C_COMPILER_LAUNCHER=ccache",
            "-DCMAKE_CXX_COMPILER_LAUNCHER=ccache",
            "-DCMAKE_CUDA_COMPILER_LAUNCHER=ccache",
        ]
    if _HAS_MOLD:
        for kind in ("EXE", "SHARED", "MODULE"):
            flags.append(f"-DCMAKE_{kind}_LINKER_FLAGS=-fuse-ld=mold")
    return flags


def detect_cuda_arch(default: str = "75") -> str:
    """Query the GPU's compute capability via nvidia-smi and return it as
    a CMAKE_CUDA_ARCHITECTURES value (T4→'75', P100→'60', A100→'80',
    L4→'89'). Falls back to `default` (T4) if nvidia-smi is absent or
    silent. Pinning one arch instead of ggml's full fat-binary list cuts
    each nvcc TU to a single SASS target — ~5× less compile RAM + time,
    which is what keeps the ggml-cuda sweep from OOMing the ~16 GB box."""
    try:
        r = subprocess.run(
            "nvidia-smi --query-gpu=compute_cap --format=csv,noheader",
            shell=True, capture_output=True, text=True, timeout=30)
        caps = [l.strip() for l in r.stdout.splitlines() if l.strip()]
        if caps:
            return sorted(caps)[0].replace(".", "")
    except Exception:
        pass
    return default


def cuda_build_flags(arch: str | None = None) -> list[str]:
    """Standard CrispASR CUDA cmake flags: GGML_CUDA on, no-VMM fallback,
    explicit nvcc, CUDA stubs on LIBRARY_PATH, and a pinned arch. Pass an
    arch or let it auto-detect."""
    if arch is None:
        arch = detect_cuda_arch()
    stubs = "/usr/local/cuda/lib64/stubs"
    if os.path.isdir(stubs):
        os.environ["LIBRARY_PATH"] = f"{stubs}:{os.environ.get('LIBRARY_PATH', '')}"
    flags = ["-DGGML_CUDA=ON", "-DGGML_CUDA_NO_VMM=ON",
             f"-DCMAKE_CUDA_ARCHITECTURES={arch}"]
    nvcc = "/usr/local/cuda/bin/nvcc"
    if os.path.isfile(nvcc):
        flags.append(f"-DCMAKE_CUDA_COMPILER={nvcc}")
    return flags


def safe_build_jobs(gpu: bool) -> str:
    """nvcc TUs are RAM-heavy; -j$(nproc)=4 on the ~16 GB Kaggle box OOMs
    mid-ggml-cuda. Cap CUDA builds at -j2 (still parallel, fits memory);
    CPU builds keep full parallelism."""
    return "2" if gpu else "$(nproc)"


# ─────────────────────────── runtime probes ────────────────────────────────

def rss_mib() -> float | None:
    """Resident set size of this process in MiB (Linux /proc only)."""
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return round(int(line.split()[1]) / 1024.0, 1)
    except Exception:
        pass
    return None


def free_gb(path: str = "/kaggle/working") -> float | None:
    """Free space at `path` in GB."""
    try:
        st = os.statvfs(path)
        return round(st.f_bavail * st.f_frsize / 1e9, 1)
    except Exception:
        return None


# ─────────────────────────── HF auth (3-tier) ──────────────────────────────

def kaggle_secret(name: str, retries: int = 3, backoff_s: float = 5.0) -> str | None:
    """Pull a Kaggle secret with diagnostics + retry. Kaggle injects
    KAGGLE_USER_SECRETS_TOKEN only when a secret is *attached* to the
    kernel, so its absence means no secret can be read regardless of the
    account dashboard. The Secrets service also flakes intermittently, so
    retry with backoff before falling through to the dataset path."""
    try:
        from kaggle_secrets import UserSecretsClient
    except Exception as exc:
        print(f"kaggle_secret({name!r}): kaggle_secrets unimportable "
              f"({type(exc).__name__}). Not inside a Kaggle kernel?", flush=True)
        return None
    if not os.environ.get("KAGGLE_USER_SECRETS_TOKEN"):
        print(f"kaggle_secret({name!r}): KAGGLE_USER_SECRETS_TOKEN missing — "
              f"no secret is ATTACHED (Add-ons → Secrets → Attach).", flush=True)
        return None
    last = None
    for attempt in range(1, retries + 1):
        try:
            return UserSecretsClient().get_secret(name)
        except Exception as exc:
            last = exc
            if attempt < retries:
                print(f"kaggle_secret({name!r}): attempt {attempt}/{retries} "
                      f"failed ({type(exc).__name__}); retry in {backoff_s}s",
                      flush=True)
                time.sleep(backoff_s)
    print(f"kaggle_secret({name!r}): all {retries} attempts failed "
          f"({type(last).__name__ if last else '?'}); using dataset fallback.",
          flush=True)
    return None


def _kaggle_input_root() -> Path:
    """Root of Kaggle dataset mounts. Overridable via KAGGLE_INPUT_ROOT so the
    token-resolution logic is unit-testable off-Kaggle against a fake tree
    (tests/test_kaggle_harness_token.py). On a real worker this is always
    /kaggle/input."""
    return Path(os.environ.get("KAGGLE_INPUT_ROOT", "/kaggle/input"))


def kaggle_token_from_dataset(filename: str = "hf_token.txt") -> str | None:
    """Read an HF token from a private Kaggle Dataset mounted via
    kernel-metadata.json `dataset_sources` (e.g. chr1str/crispasr-hf-token
    → /kaggle/input/crispasr-hf-token/hf_token.txt). Bypasses the flaky
    Secrets API; datasets are filesystem-mounted before the script runs.

    Kaggle mounts datasets at several locations depending on the
    environment version:
      /kaggle/input/<slug>/           — classic ("short") path
      /kaggle/input/datasets/<owner>/<slug>/  — newer ("long") environments
    We scan BOTH depths, short first, each sorted. Some workers mount
    attached datasets ONLY under the long path: CrispEmbed's T19-E3 run 1
    (2026-08) completed a full 21-minute pipeline on such a worker
    ("/kaggle/input contains 1 entries: ['datasets']") and then lost every
    artifact to upload 401s, because its (stale, vendored) copy of this
    resolver never descended into datasets/<owner>/<slug>. Keep both depths
    covered here, and keep this file the single source of truth.

    kaggle_usage.md gotcha #26 also applies to how kernels REACH this code:
    a Kaggle *script* kernel ships ONLY its `code_file` — sibling files
    (this harness, corpora, reference audio) are NOT present at runtime.
    Kernels must `git clone` the repo in-kernel and import the harness FROM
    THE CLONE; never read data via Path(__file__).parent, and never rely on
    a bundled copy (bundled = local-dev fallback only)."""
    root = _kaggle_input_root()
    candidates: list[Path] = [
        root / "crispasr-hf-token" / filename,
    ]
    # Owner-agnostic scan: probe <filename> in EVERY mounted dataset dir, at
    # both the classic depth (<root>/<slug>/) and the newer nested depth
    # (<root>/datasets/<owner>/<slug>/). The old code only matched owner
    # names containing "hf-token" and hard-coded chr1str, so a chr1s4 kernel on
    # the newer mount path (/kaggle/input/datasets/chr1s4/crispasr-hf-token/)
    # never had its token file scanned → token silently unresolved (the
    # 2026-06-20 v2/v3 full-sweep runs). Don't filter by dir name — probe the file.
    # Deterministic precedence (matches the t19 driver's sorted short-then-long
    # glob order): classic dirs first, then nested dirs, each depth sorted.
    classic_dirs: list[Path] = []
    nested_dirs: list[Path] = []
    if root.exists():
        for sub in sorted(root.iterdir()):
            if not sub.is_dir():
                continue
            if sub.name == "datasets":
                for owner in sorted(sub.iterdir()):  # nested <owner>/<slug>
                    if owner.is_dir():
                        nested_dirs.extend(
                            s for s in sorted(owner.iterdir()) if s.is_dir())
            else:
                classic_dirs.append(sub)  # classic <root>/<slug>
    for d in classic_dirs + nested_dirs:
        for fn in (filename, "hf_token.txt", "token", "access_token"):
            p = d / fn
            if p not in candidates:
                candidates.append(p)
    # Also try the flat file variants
    candidates.append(root / "crispasr-hf-token" / "token")
    candidates.append(root / "crispasr-hf-token" / "access_token")

    # Debug: list what we're scanning
    for p in candidates:
        try:
            exists = p.exists()
            if exists:
                tok = p.read_text().strip()
                if tok and len(tok) > 8:
                    print(f"HF auth: token from {p} (dataset fallback).",
                          flush=True)
                    return tok
                else:
                    print(f"HF auth: {p} exists but empty/short", flush=True)
            # Don't log every non-existent path — too noisy
        except Exception as e:
            print(f"HF auth: {p} error: {e}", flush=True)

    # Last resort: dump what's actually under the input root for debugging
    if root.exists():
        dirs = sorted(root.iterdir())
        print(f"HF auth: {root} contains {len(dirs)} entries: "
              f"{[d.name for d in dirs[:10]]}", flush=True)
        # Also check one level deeper
        for d in dirs:
            if d.is_dir():
                try:
                    children = sorted(d.iterdir())
                    if children:
                        print(f"HF auth:   {d.name}/ -> "
                              f"{[c.name for c in children[:5]]}", flush=True)
                except PermissionError:
                    pass
    return None


def resolve_hf_token(secret_name: str = "HF_TOKEN",
                     require: bool = False) -> str | None:
    """3-tier HF auth: env HF_TOKEN → Kaggle Secret (with retry) → mounted
    Kaggle Dataset file. Exports HF_TOKEN + HUGGING_FACE_HUB_TOKEN +
    HF_HUB_ENABLE_HF_TRANSFER on success and returns the token (or None).

    Any kernel that UPLOADS must call this with require=True at the very
    top, BEFORE building or computing anything: with require=True a missing
    token raises SystemExit immediately, instead of the run completing and
    then losing every artifact to 401s at upload time (CrispEmbed T19-E3
    run 1 burned a full 21-minute pipeline exactly this way — the token
    dataset was mounted only under the long path and the resolver missed
    it; `hf_token_ok: False` was in the log all along). Download-only
    kernels can keep the default require=False: public cstr/* repos work
    unauthenticated."""
    tok = (os.environ.get("HF_TOKEN")
           or kaggle_secret(secret_name)
           or kaggle_token_from_dataset())
    if tok:
        os.environ["HF_TOKEN"] = tok
        os.environ["HUGGING_FACE_HUB_TOKEN"] = tok
        # Only opt into the fast downloader if it is actually importable.
        # huggingface_hub raises outright ("enabled but not available") rather
        # than falling back, so setting this unconditionally turned every
        # kernel that resolves a token WITHOUT pip-installing hf_transfer into
        # a hard download failure — it killed the breeze-refdump run at its
        # first hf_hub_download, minutes into a GPU session (2026-09-03).
        try:
            import hf_transfer  # noqa: F401

            os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
        except ImportError:
            os.environ.pop("HF_HUB_ENABLE_HF_TRANSFER", None)
    elif require:
        raise SystemExit(
            "FATAL: no HF token from env/secret/dataset — uploads would 401 "
            "after the compute finishes; aborting up front. Attach the "
            "hf-token dataset (kernel-metadata.json dataset_sources) or the "
            "HF_TOKEN secret.")
    else:
        print("HF auth: no token from env/secret/dataset — public-only "
              "downloads (fine for cstr/* public repos).", flush=True)
    return tok
