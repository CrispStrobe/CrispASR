import os
import json
import re
import shlex
import subprocess
import time
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = ROOT.parent
CRISPLENS = CODE_DIR / "CrispLens"
CLOUD_BACKUP = CODE_DIR / "cloud-backup"
ENV_FILE = CODE_DIR / ".env"


def run(cmd, cwd=ROOT, timeout=30, check=True):
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        proc = subprocess.CompletedProcess(
            cmd,
            124,
            stdout=(exc.stdout or ""),
            stderr=(exc.stderr or "") + f"\ncommand timed out after {timeout} seconds",
        )
    if check and proc.returncode != 0:
        raise AssertionError(
            f"command failed ({proc.returncode}): {cmd!r}\n"
            f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        )
    return proc


def env_keys():
    if not ENV_FILE.exists():
        return {}
    out = {}
    for line in ENV_FILE.read_text(errors="ignore").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        out[k.strip()] = v.strip().strip("'\"")
    return out


def required_env_path(key):
    val = env_keys().get(key, "").rstrip("/")
    assert val, f"{key} must be set in {ENV_FILE}"
    return val


def ssh_cmd(remote_script, timeout=30, check=True, attempts=3):
    env = env_keys()
    host = env.get("VPS_IP") or "168.119.190.252"
    cmd = [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=10",
        f"root@{host}",
        remote_script,
    ]
    proc = None
    for attempt in range(attempts):
        proc = run(cmd, timeout=timeout, check=False)
        if proc.returncode == 0 or not _is_transient_ssh_failure(proc):
            break
        if attempt + 1 < attempts:
            time.sleep(2 * (attempt + 1))
    if proc is not None and _is_transient_ssh_failure(proc):
        pytest.skip(f"VPS SSH unavailable for live test: {proc.stderr.strip()}")
    if check and proc is not None and proc.returncode != 0:
        raise AssertionError(
            f"command failed ({proc.returncode}): {cmd!r}\n"
            f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        )
    return proc


def _is_transient_ssh_failure(proc):
    stderr = proc.stderr.lower()
    return proc.returncode in (124, 255) and any(
        marker in stderr
        for marker in [
            "command timed out",
            "operation timed out",
            "connection timed out",
            "connection refused",
            "no route to host",
            "network is unreachable",
        ]
    )


def test_expected_repositories_exist():
    assert CRISPLENS.is_dir(), CRISPLENS
    assert CLOUD_BACKUP.is_dir(), CLOUD_BACKUP
    assert (ROOT / "docs" / "architecture.md").is_file()
    assert (CLOUD_BACKUP / "api" / "app.py").is_file()
    assert (CRISPLENS / "fastapi_app.py").is_file()


def test_parent_env_contains_required_connection_keys_without_printing_values():
    keys = env_keys()
    for key in [
        "CRISPLENS_URL",
        "CRISPLENS_LOGIN",
        "CRISPLENS_PW",
        "VPS_IP",
        "DB_PASSWORD",
        "ARCHIVE_PASSWORD",
        "VPS_STORAGE_ROOT",
        "VPS_SCRATCH_DIR",
        "CLOUD_BACKUP_SCRATCH_DIR",
        "CRISPLENS_SCRATCH_DIR",
        "CRISPASR_SCRATCH_DIR",
        "CB_API_STORAGE_ROOT",
        "CB_API_LANCE_ROOT",
    ]:
        assert key in keys
        assert keys[key]


def test_cloud_backup_api_routes_are_defined():
    app = (CLOUD_BACKUP / "api" / "app.py").read_text(errors="ignore")
    for route in [
        "/api/health",
        "/api/manifest/push",
        "/api/manifest/pull",
        "/api/index/push-embeddings",
        "/api/index/by-embedding",
        "/api/search",
        "/api/files/by-hash/{sha256}",
        "/api/index/embed-models",
        "/api/index/embed-query",
        "/api/v2/index/search",
    ]:
        assert route in app


def test_cloud_backup_blob_store_uses_same_filesystem_partial_writes():
    files_py = (CLOUD_BACKUP / "api" / "files.py").read_text(errors="ignore")
    assert "tempfile.mkstemp" in files_py
    assert "dir=final.parent" in files_py
    assert "os.replace(partial, final)" in files_py


def test_cloud_backup_requires_env_for_large_blob_and_lance_roots():
    db_py = (CLOUD_BACKUP / "api" / "db.py").read_text(errors="ignore")
    lance_py = (CLOUD_BACKUP / "api" / "lance.py").read_text(errors="ignore")
    assert "CB_API_STORAGE_ROOT must be configured" in db_py
    assert "DEFAULT_STORAGE_ROOT" not in db_py
    assert "DEFAULT_LANCE_ROOT" not in lance_py
    assert "CB_API_LANCE_ROOT" in lance_py


def test_crisplens_v4_crispasr_wrapper_and_transcript_search_are_wired():
    v4 = CRISPLENS / "electron-app-v4"
    renderer = v4 / "renderer"
    wrapper = (v4 / "renderer" / "src" / "lib" / "CrispAsrWrapper.js").read_text(errors="ignore")
    preload = (v4 / "preload.js").read_text(errors="ignore")
    main = (v4 / "electron-main.js").read_text(errors="ignore")
    process_view = (v4 / "renderer" / "src" / "lib" / "ProcessView.svelte").read_text(errors="ignore")
    settings_view = (v4 / "renderer" / "src" / "lib" / "SettingsView.svelte").read_text(errors="ignore")
    local_db = (v4 / "renderer" / "src" / "lib" / "LocalDB.js").read_text(errors="ignore")
    local_adapter = (v4 / "renderer" / "src" / "lib" / "LocalAdapter.js").read_text(errors="ignore")
    server_db = (v4 / "server" / "db.js").read_text(errors="ignore")
    server_images = (v4 / "server" / "routes" / "images.js").read_text(errors="ignore")
    server_ingest = (v4 / "server" / "routes" / "ingest.js").read_text(errors="ignore")
    server_cloud_drives = (v4 / "server" / "routes" / "cloud-drives.js").read_text(errors="ignore")

    assert "registerPlugin('CrispASR')" in wrapper
    assert "electronAPI?.crispasrTranscribe" in wrapper
    assert "Capacitor.isNativePlatform" in wrapper
    assert "crisplens.crispasr.language" in wrapper
    assert "crispasrTranscribe" in preload
    assert "ipcMain.handle('crispasr-transcribe'" in main
    assert "--output-json-full" in main
    assert "CRISPASR_SCRATCH_DIR" in main
    assert "isVideoName" in process_view
    assert "transcribeMedia" in process_view
    assert 'accept="image/*,video/*"' in process_view
    for text in [local_db, local_adapter, server_db, server_images, server_ingest]:
        assert "media_type" in text
        assert "transcript" in text
    assert "i.transcript LIKE ?" in local_adapter
    assert "LOWER(i.transcript) LIKE LOWER(?)" in server_images
    assert "transcript=COALESCE" in server_ingest
    assert "media_type, duration_sec, fps, frame_count, transcript" in server_ingest
    assert "function scratchRoot()" in server_cloud_drives
    assert "process.env.CRISPLENS_SCRATCH_DIR" in server_cloud_drives
    assert "os.tmpdir()" not in server_cloud_drives
    assert "crispasr_video_transcripts" in settings_view
    assert "hasCrispAsrWrapper" in settings_view
    assert "crisplens.crispasr.model" in settings_view
    assert "crisplens.crispasr.extraArgs" in settings_view
    assert "CrispASR Video Transcripts" in (renderer / "src" / "stores.js").read_text(errors="ignore")

    package_json = json.loads((renderer / "package.json").read_text(errors="ignore"))
    assert package_json["dependencies"]["crisplens-crispasr-capacitor"] == "file:plugins/crispasr-capacitor"
    assert package_json["devDependencies"]["@capacitor/ios"].startswith("^7.")
    assert package_json["devDependencies"]["vite"].startswith("^8.")

    plugin = renderer / "plugins" / "crispasr-capacitor"
    podspec = (plugin / "CrisplensCrispasrCapacitor.podspec").read_text(errors="ignore")
    swift = (plugin / "ios" / "Sources" / "CrispASRPlugin" / "CrispASRPlugin.swift").read_text(errors="ignore")
    android = (plugin / "android" / "src" / "main" / "java" / "com" / "crisplens" / "crispasr" / "CrispASRPlugin.kt")
    assert android.is_file()
    assert (plugin / "ios" / "Vendor" / "crispasr.xcframework").exists()
    assert "vendored_frameworks" in podspec
    assert "crispasr.xcframework" in podspec
    assert "EXCLUDED_ARCHS[sdk=iphonesimulator*]" in podspec
    assert "CAPBridgedPlugin" in swift
    assert 'jsName = "CrispASR"' in swift
    assert ".applicationSupportDirectory" in swift
    assert "AVAssetExportSession" in swift
    assert "whisper_full" in swift


def test_cb_api_env_template_documents_storage_roots_and_shard_warning():
    template = (CLOUD_BACKUP / "deploy" / "etc" / "cb-api.env.example").read_text(errors="ignore")
    for key in [
        "CB_API_DB_PATH",
        "CB_API_STORAGE_ROOT",
        "CB_API_SHARD_ROOT",
        "CB_API_LANCE_ROOT",
        "XDG_CACHE_HOME",
        "CRISP_CB_SHARED_OWNERS",
    ]:
        assert key in template
    assert "# CB_API_SHARD_ROOT=" in template
    assert "CIFS" in template
    assert "SMB" in template
    assert "ext4" in template or "XFS" in template


def test_no_hardcoded_tmp_or_default_tempfile_in_core_cross_stack_paths():
    paths = [
        CRISPLENS / "transcription_providers.py",
        CRISPLENS / "face_recognition_core.py",
        CRISPLENS / "routers" / "processing.py",
        CRISPLENS / "cloud_drive_manager.py",
        CRISPLENS / "drive_mount.py",
        CRISPLENS / "face_rec_ui.py",
        CRISPLENS / "electron-app-v4" / "server" / "routes" / "cloud-drives.js",
        CRISPLENS / "deploy-v2.sh",
        CRISPLENS / "fix_db.sh",
        CRISPLENS / "routers" / "cloud_drives.py",
        CLOUD_BACKUP / "api" / "app.py",
        CLOUD_BACKUP / "api" / "files.py",
        CLOUD_BACKUP / "deploy" / "usr" / "local" / "sbin" / "recover_archive_listings.py",
        CLOUD_BACKUP / "prune.py",
        CLOUD_BACKUP / "retrieve.py",
        CLOUD_BACKUP / "search_engine.py",
        CLOUD_BACKUP / "controller.py",
        ROOT / "src" / "crispasr_cache.cpp",
        ROOT / "examples" / "cli" / "crispasr_server.cpp",
        ROOT / "examples" / "cli" / "crispasr_lid_cli.cpp",
        ROOT / "examples" / "cli" / "crispasr_diarize_cli.cpp",
        ROOT / "tests" / "test-crispasr-cache.cpp",
        ROOT / "tests" / "test-server-tts.sh",
        ROOT / "tests" / "test-translators.sh",
        ROOT / "tests" / "test-vibevoice-base-clone.sh",
        ROOT / "tools" / "benchmark_lid.py",
        ROOT / "tools" / "test_lid_mls.py",
    ]
    offenders = []
    for path in paths:
        text = path.read_text(errors="ignore")
        tempfile_call = re.search(r"tempfile\.(TemporaryDirectory|mkdtemp|mkstemp|NamedTemporaryFile)\(", text)
        if "/tmp" in text or (tempfile_call and "dir=" not in text):
            offenders.append(str(path.relative_to(CODE_DIR)))
    assert not offenders, offenders


@pytest.mark.live
def test_vps_ssh_services_are_active():
    proc = ssh_cmd(
        "hostname; "
        "systemctl is-active vps-worker.service; "
        "systemctl is-active cb-api.service; "
        "systemctl is-active internxt-relogin.timer",
        timeout=25,
    )
    lines = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    assert lines[0]
    assert lines[1:] == ["active", "active", "active"]


@pytest.mark.live
def test_vps_internal_health_endpoints_and_expected_listener_shape():
    script = r"""
set -eu
curl -fsS --max-time 5 http://127.0.0.1:7869/api/health
printf '\n---\n'
curl -fsS --max-time 5 http://127.0.0.1:7865/api/health
printf '\n---\n'
if ss -ltn | grep -q ':7861 '; then echo '7861-listening'; else echo '7861-not-listening'; fi
"""
    proc = ssh_cmd(script, timeout=25)
    out = proc.stdout
    assert '"backend":"cloud-backup-api"' in out
    assert '"lance_enabled":true' in out
    assert '"fastembed_enabled":true' in out
    assert '"model_ready":true' in out
    assert "7861-not-listening" in out


@pytest.mark.live
def test_vps_scratch_roots_are_on_attached_storage():
    script = r"""
set -eu
worker_scratch="$(awk -F= '$1=="VPS_SCRATCH_DIR" {print $2}' /etc/vps-worker.env | tail -1)"
face_scratch="$(systemctl cat face-rec.service | sed -n 's/.*CRISPLENS_SCRATCH_DIR=\([^ "]*\).*/\1/p' | tail -1)"
cb_scratch="$(awk -F= '$1=="CB_API_SCRATCH_DIR" {print $2}' /etc/cb-api.env | tail -1)"
test -n "$worker_scratch"
test -n "$face_scratch"
test -n "$cb_scratch"
case "$worker_scratch:$face_scratch:$cb_scratch" in
  */tmp*|/tmp*:*) exit 20 ;;
esac
test -d "$worker_scratch"
test -d "$face_scratch"
test -d "$cb_scratch"
printf '%s\n%s\n%s\n' "$worker_scratch" "$face_scratch" "$cb_scratch"
"""
    proc = ssh_cmd(script, timeout=25)
    for line in proc.stdout.splitlines():
        if line.strip():
            assert not line.strip().startswith("/tmp")


@pytest.mark.live
def test_vps_service_cache_envs_avoid_tmp():
    script = r"""
set -eu
for service in cb-api.service vps-worker.service face-rec.service; do
  pid="$(systemctl show -p MainPID --value "$service")"
  test -n "$pid"
  test "$pid" != "0"
  tr '\0' '\n' <"/proc/$pid/environ" | grep -E '^(TMPDIR|HF_HOME|HF_HUB_CACHE|TRANSFORMERS_CACHE|XDG_CACHE_HOME|CB_API_SCRATCH_DIR|CRISPLENS_SCRATCH_DIR)=' || true
done
"""
    proc = ssh_cmd(script, timeout=25)
    required = [
        "TMPDIR=/mnt/storage/cb_api_scratch/tmp",
        "CB_API_SCRATCH_DIR=/mnt/storage/cb_api_scratch",
        "HF_HOME=/mnt/storage/hf-cache",
        "TMPDIR=/mnt/storage/cloudworker/scratch/tmp",
        "HF_HOME=/mnt/storage/cloudworker/scratch/hf-cache",
        "TMPDIR=/mnt/storage/crisplens/scratch/tmp",
        "HF_HOME=/mnt/storage/crisplens/scratch/hf-cache",
    ]
    for item in required:
        assert item in proc.stdout
    for line in proc.stdout.splitlines():
        if line.strip():
            assert "=/tmp" not in line


@pytest.mark.live
def test_vps_storage_roots_and_catalog_are_within_expected_current_bounds():
    storage_root = shlex.quote(required_env_path("VPS_STORAGE_ROOT"))
    script = r"""
set -eu
storage_root=__STORAGE_ROOT__
df -Pk / "$storage_root" | awk 'NR>1 {print "df_free_kib",$4}'
python3 - <<'PY'
import os, sqlite3
checks = [
    ("/root/cloudworker_state/master_catalog.db", ["files", "file_references", "batches", "api_keys", "chunk_embeddings"]),
    ("/root/cloudworker_state/worker_state.db", ["jobs"]),
]
for path, tables in checks:
    print("db", path, os.path.getsize(path))
    con = sqlite3.connect(path)
    cur = con.cursor()
    for table in tables:
        cur.execute(f"select count(*) from {table}")
        print(table, cur.fetchone()[0])
    con.close()
PY
""".replace("__STORAGE_ROOT__", storage_root)
    proc = ssh_cmd(script, timeout=25)
    out = proc.stdout
    # Root should not be close to completely full; Storage Box should have TB-scale headroom.
    free_values = []
    for line in out.splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[0] == "df_free_kib":
            free_values.append(int(parts[1]))
    assert free_values[0] > 10 * 1024 * 1024  # >10 GiB in KiB
    assert free_values[1] > 1024 * 1024 * 1024  # >1 TiB in KiB
    counts = {}
    for line in out.splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[1].isdigit():
            counts[parts[0]] = int(parts[1])
    assert counts["files"] >= 66
    assert counts["file_references"] >= counts["files"]
    assert counts["chunk_embeddings"] >= 0
    assert counts["jobs"] >= 2


@pytest.mark.live
def test_vps_apache_exposes_crisplens_but_not_cb_api_prefix():
    script = r"""
set -eu
grep -RIn "7865\|7869\|/cb" /etc/apache2/sites-enabled 2>/dev/null || true
"""
    proc = ssh_cmd(script, timeout=25)
    out = proc.stdout
    assert "127.0.0.1:7865" in out
    assert "127.0.0.1:7869" not in out


@pytest.mark.live
def test_vps_internxt_cli_auth_and_worker_queues():
    storage_root = shlex.quote(required_env_path("VPS_STORAGE_ROOT"))
    script = r"""
set -eu
storage_root=__STORAGE_ROOT__
/root/internxt-python/venv/bin/python /root/internxt-python/cli.py whoami >/dev/null
find "$storage_root/incoming" -maxdepth 1 -type f | wc -l
find "$storage_root/processing" -maxdepth 1 -type f | wc -l
find "$storage_root/incoming/.manifests" -maxdepth 1 -type f 2>/dev/null | wc -l
"""
    script = script.replace("__STORAGE_ROOT__", storage_root)
    proc = ssh_cmd(script, timeout=30)
    counts = [int(x.strip()) for x in proc.stdout.splitlines() if x.strip().isdigit()]
    assert counts[0] == 0  # incoming archive queue
    assert counts[1] == 0  # processing archive queue
    assert counts[2] >= 1  # receipts/manifests are present for sync/reconciliation


@pytest.mark.live
def test_deployed_worker_uses_streaming_hash_and_no_extract_glob_upload():
    proc = ssh_cmd("grep -n \"f.read\\|extract_dir}/*\" /root/internxt-python/vps_worker.py || true", timeout=20)
    assert "f.read()" not in proc.stdout
    assert "extract_dir}/*" not in proc.stdout
