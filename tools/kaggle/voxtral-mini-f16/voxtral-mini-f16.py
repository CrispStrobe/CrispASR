#!/usr/bin/env python3
"""Kaggle kernel (#437): publish the F16 GGUFs for the two Voxtral Mini models.

Issue #437: `cstr/voxtral-mini-3b-2507-GGUF` and
`cstr/voxtral-mini-4b-realtime-GGUF` only ever shipped q4_k and q8_0. Both
converters emit F16 natively — the F16 existed at conversion time, it was just
never uploaded. So this is a convert-and-upload job that REUSES the existing
converters unchanged:

    models/convert-voxtral-to-gguf.py     → mistralai/Voxtral-Mini-3B-2507
    models/convert-voxtral4b-to-gguf.py   → mistralai/Voxtral-Mini-4B-Realtime-2602

⚠ The verification here deliberately does not trust this script's own log.
"uploaded OK" is a statement about the HTTP call, and HuggingFace has returned
200 on incomplete uploads. Three independent checks run per model, each of
which can FAIL on its own:

  1. local sha256 + byte size of the converted file, computed before upload;
  2. the hub's own `lfs.oid` for the published blob, which IS the sha256 of the
     content it stores — compared against (1). A truncated or corrupted upload
     cannot match;
  3. tools/verify-remote-gguf.py, which range-reads the PUBLISHED file's GGUF
     header and re-derives the file size the header implies, then asserts the
     dominant tensor dtype really is F16.

Each model is attempted independently: one failing does not prevent the other
from shipping, and the final summary names which is which.

Push (under chr1s4 — its crispasr-hf-token clone is the one attached):
    export KAGGLE_API_TOKEN=<chr1s4 token>
    python -m kaggle kernels push -p tools/kaggle/voxtral-mini-f16
"""

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

# Bump on every push. Gotcha #24: the kernel script is frozen at the last push
# while the cloned C++/converters are always fresh from the branch, so the log
# has to show BOTH halves or a stale harness silently scores a fresh run.
SCRIPT_VERSION = "437-f16-v1"

BRANCH = "feat/437-voxtral-f16"
REPO_URL = "https://github.com/CrispStrobe/CrispASR"

WORK = Path("/kaggle/working")
REPO = WORK / "CrispASR"
TEMP = Path("/kaggle/temp") if Path("/kaggle/temp").is_dir() else WORK

MODELS = [
    {
        "name": "voxtral-mini-3b-2507",
        "src_repo": "mistralai/Voxtral-Mini-3B-2507",
        "hf_repo": "cstr/voxtral-mini-3b-2507-GGUF",
        "converter": "models/convert-voxtral-to-gguf.py",
        # consolidated.safetensors is a 9.35 GB duplicate of the sharded
        # weights in Mistral's own naming; the converter globs `model-*` and
        # would ignore it, so fetching it would burn 9 GB of disk for nothing.
        "allow": ["model-*.safetensors", "config.json", "tekken.json"],
        # The published q4_k has 765 tensors; F16 must have the same count.
        "expect_tensors": 765,
    },
    {
        "name": "voxtral-mini-4b-realtime",
        "src_repo": "mistralai/Voxtral-Mini-4B-Realtime-2602",
        "hf_repo": "cstr/voxtral-mini-4b-realtime-GGUF",
        "converter": "models/convert-voxtral4b-to-gguf.py",
        "allow": ["model.safetensors", "config.json", "tekken.json"],
        # Published q4_k has 714 tensors.
        "expect_tensors": 714,
    },
]


def log(msg: str) -> None:
    print(msg, flush=True)


def df(path: Path) -> str:
    try:
        st = os.statvfs(str(path))
        return f"{st.f_bavail * st.f_frsize / 2**30:.1f} GiB free on {path}"
    except OSError:
        return f"(statvfs failed on {path})"


# ── Phase 0: environment + fail-fast internet probe ──────────────────────────
log(f"=== script {SCRIPT_VERSION} ===")
log(f"  {df(WORK)}")
log(f"  {df(TEMP)}")
try:
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith(("MemTotal", "MemAvailable")):
                log("  " + line.strip())
except OSError:
    pass

# Gotcha #3: a worker can come up with no internet at all. Find that out in
# seconds rather than after a failed 9 GB download.
try:
    with urllib.request.urlopen("https://huggingface.co/api/models/mistralai/Voxtral-Mini-3B-2507",
                                timeout=30) as r:
        r.read(64)
    log("  internet: OK")
except Exception as e:  # noqa: BLE001
    log(f"  NO_INTERNET_RETRY: {e}")
    raise SystemExit(0)

# ── Phase 1: clone the branch under test ─────────────────────────────────────
log("=== clone repo ===")
cloned_branch = None
for ref in (BRANCH, "main"):
    if REPO.exists():
        shutil.rmtree(REPO, ignore_errors=True)
    try:
        subprocess.check_call(["git", "clone", "--depth", "1", "-b", ref, REPO_URL, str(REPO)])
        cloned_branch = ref
        break
    except Exception as e:  # noqa: BLE001
        log(f"  clone {ref} failed: {e}")
if cloned_branch is None:
    log("CLONE_FAILED")
    raise SystemExit(1)
sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(REPO)).decode().strip()
log(f"  branch={cloned_branch} sha={sha}")

sys.path.insert(0, str(REPO / "tools" / "kaggle"))
import kaggle_harness as kh  # noqa: E402

kh.init_progress()

VERIFIER = REPO / "tools" / "verify-remote-gguf.py"
log(f"  verifier present: {VERIFIER.exists()}")

# ── Phase 2: deps ────────────────────────────────────────────────────────────
kh.step("install deps")
kh.sh_with_progress("pip install -q safetensors gguf huggingface_hub hf_transfer")
# The converters open safetensors with framework='pt' (they need torch's
# bfloat16 to downcast), so torch is REQUIRED even though nothing runs on a GPU.
try:
    import torch  # noqa: F401
    log("  torch already present")
except ImportError:
    kh.sh_with_progress("pip install -q torch --index-url https://download.pytorch.org/whl/cpu")

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
os.environ["TMPDIR"] = str(TEMP)

# ── Phase 3: token ───────────────────────────────────────────────────────────
kh.step("resolve HF token")
hf_token = kh.resolve_hf_token()
if not hf_token:
    log("NO_HF_TOKEN — cannot upload; aborting before spending the download")
    raise SystemExit(1)
os.environ["HF_TOKEN"] = hf_token
os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token
log("  HF_TOKEN resolved")

from huggingface_hub import HfApi, snapshot_download  # noqa: E402

api = HfApi(token=hf_token)
# A token that authenticates but has no write scope would otherwise surface as
# a failure 40 minutes in, after the conversion. Find out now.
try:
    who = api.whoami()
    log(f"  hf user={who.get('name')} type={who.get('type')}")
except Exception as e:  # noqa: BLE001
    log(f"  HF_TOKEN_INVALID: {e}")
    raise SystemExit(1)


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(16 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def hub_blob_facts(repo: str, filename: str, tries: int = 8) -> dict | None:
    """Ask the hub what it thinks it stores: size + lfs.oid (= content sha256)."""
    url = f"https://huggingface.co/api/models/{repo}/tree/main"
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {hf_token}"})
    for attempt in range(tries):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                tree = json.loads(r.read())
            for ent in tree:
                if ent.get("path") == filename:
                    return ent
        except Exception as e:  # noqa: BLE001
            log(f"    tree fetch attempt {attempt + 1}: {e}")
        time.sleep(15)
    return None


results = {}

for m in MODELS:
    name = m["name"]
    out_name = f"{name}-f16.gguf"
    url = f"https://huggingface.co/{m['hf_repo']}/resolve/main/{out_name}"
    r = {"uploaded": False, "sha_match": False, "header_ok": False, "error": None}
    results[name] = r
    log(f"\n################ {name} ################")
    src_dir = None
    f16 = TEMP / out_name
    try:
        kh.step(f"{name}: download source")
        scratch = TEMP / f"{name}-src"
        scratch.mkdir(parents=True, exist_ok=True)
        src_dir = snapshot_download(
            repo_id=m["src_repo"], cache_dir=str(scratch), token=hf_token,
            allow_patterns=m["allow"],
        )
        log(f"  src: {src_dir}")
        log(f"  {df(TEMP)}")

        kh.step(f"{name}: convert F16")
        kh.sh_with_progress(
            f"python {m['converter']} --input {src_dir} --output {f16}", cwd=str(REPO)
        )
        if not f16.exists():
            raise RuntimeError("converter produced no output file")
        size = f16.stat().st_size
        log(f"  F16: {f16} — {size:,} bytes ({size / 2**30:.2f} GiB)")

        kh.step(f"{name}: sha256 (pre-upload)")
        local_sha = sha256_of(f16)
        log(f"  local sha256 = {local_sha}")
        log(f"  local size   = {size}")

        # Free the source before the upload so peak disk is
        # max(src + gguf, gguf) rather than src + gguf + upload staging.
        shutil.rmtree(scratch, ignore_errors=True)
        src_dir = None
        log(f"  after source cleanup: {df(TEMP)}")

        kh.step(f"{name}: upload")
        try:
            api.create_repo(repo_id=m["hf_repo"], repo_type="model", exist_ok=True)
        except Exception as e:  # noqa: BLE001
            log(f"  create_repo: {e}")
        api.upload_file(
            path_or_fileobj=str(f16), path_in_repo=out_name,
            repo_id=m["hf_repo"], repo_type="model",
            commit_message=f"Add F16 GGUF ({name}) — #437",
        )
        r["uploaded"] = True
        log("  upload call returned")

        # ── check 2: the hub's own content hash ──────────────────────────────
        kh.step(f"{name}: verify hub blob")
        ent = hub_blob_facts(m["hf_repo"], out_name)
        if ent is None:
            log(f"  FAIL  {out_name} does not appear in the repo tree at all")
        else:
            lfs = ent.get("lfs") or {}
            hub_size = lfs.get("size", ent.get("size"))
            hub_oid = lfs.get("oid")
            log(f"  hub size   = {hub_size}")
            log(f"  hub lfs.oid= {hub_oid}")
            if hub_size != size:
                log(f"  FAIL  size mismatch: local {size} vs hub {hub_size}")
            elif hub_oid is None:
                log("  FAIL  hub reports no lfs.oid — cannot confirm content")
            elif hub_oid != local_sha:
                log(f"  FAIL  sha256 mismatch: local {local_sha} vs hub {hub_oid}")
            else:
                r["sha_match"] = True
                log("  ok    hub content sha256 == local sha256")

        # ── check 3: read the published header back ──────────────────────────
        kh.step(f"{name}: verify published header")
        if VERIFIER.exists():
            rc = subprocess.call([
                sys.executable, str(VERIFIER), url,
                "--self-test",
                "--expect-dominant", "F16",
                "--min-tensors", str(m["expect_tensors"]),
                "--forbid-dtype", "Q4_K", "--forbid-dtype", "Q8_0",
            ])
            r["header_ok"] = rc == 0
            log(f"  verify-remote-gguf exit={rc}")
        else:
            log("  SKIP  verifier not in the clone (old branch?) — header unverified")

    except Exception as e:  # noqa: BLE001
        import traceback
        r["error"] = f"{type(e).__name__}: {e}"
        log(f"  ERROR {r['error']}")
        traceback.print_exc()
    finally:
        if src_dir:
            shutil.rmtree(TEMP / f"{name}-src", ignore_errors=True)
        f16.unlink(missing_ok=True)
        log(f"  cleanup done: {df(TEMP)}")

# ── Summary ──────────────────────────────────────────────────────────────────
log("\n================ SUMMARY ================")
all_ok = True
for name, r in results.items():
    ok = r["uploaded"] and r["sha_match"] and r["header_ok"]
    all_ok &= ok
    log(f"  {name:30s} {'PASS' if ok else 'FAIL'}  "
        f"uploaded={r['uploaded']} sha_match={r['sha_match']} header_ok={r['header_ok']}"
        + (f"  err={r['error']}" if r["error"] else ""))
log(f"  VERDICT: {'ALL_MODELS_PUBLISHED' if all_ok else 'INCOMPLETE'}")
(WORK / "verdict.json").write_text(json.dumps(
    {"script_version": SCRIPT_VERSION, "clone_sha": sha, "all_ok": all_ok, "models": results},
    indent=2))
raise SystemExit(0 if all_ok else 1)
