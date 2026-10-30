"""Hosted readback of release assets and original wheel publication inputs.

Optional repair adds missing project notices only to the four known legacy
archive families. Every original member must remain byte-identical. No builds,
model loads, dependency downloads or numerical changes occur here.
"""

import argparse
import base64
import copy
from email.parser import BytesParser
from functools import lru_cache
import hashlib
from html.parser import HTMLParser
import json
import os
import posixpath
import re
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.parse
import urllib.request
import zipfile


REPO = "CrispStrobe/CrispASR"
ROOT = Path(__file__).resolve().parents[2]
OUT = Path(os.environ["HEAVY_OUT"])
SCRATCH = Path(os.environ["HEAVY_SCRATCH"]) / "publication-audit"
OUT.mkdir(parents=True, exist_ok=True)
SCRATCH.mkdir(parents=True, exist_ok=True)
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))
REPAIRABLE = {"crispasr-python-linux-x86_64.tar.gz", "crispasr-python-linux-arm64.tar.gz",
              "crispasr-wasm.tar.gz"}


def api(path):
    return json.loads(subprocess.check_output(["gh", "api", f"repos/{REPO}/{path}"], text=True))


def sha_stream(stream):
    digest = hashlib.sha256()
    count = 0
    for block in iter(lambda: stream.read(1024 * 1024), b""):
        count += len(block)
        digest.update(block)
    return dict(bytes=count, sha256=digest.hexdigest())


def sha_file(path):
    with path.open("rb") as stream:
        return sha_stream(stream)


def download(url, path):
    # No token, cookies or netrc are used for public-byte readback.
    request = urllib.request.Request(url, headers={"User-Agent": "CrispASR-publication-audit"})
    with OPENER.open(request, timeout=180) as response, path.open("wb") as target:
        shutil.copyfileobj(response, target, 1024 * 1024)
    return sha_file(path)


def safe_name(name):
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or "\\" in name:
        raise ValueError(f"Unsafe archive member: {name}")
    return str(path)


def inventory(path):
    members = {}
    total = 0

    def add(name, stream, link=None, mode=None, link_type=None):
        nonlocal total
        name = safe_name(name)
        if name in members:
            raise ValueError(f"Duplicate archive member: {name}")
        record = dict(link=link) if link is not None else sha_stream(stream)
        record["mode"] = mode
        if link is not None:
            record["link_type"] = link_type
        total += record.get("bytes", 0)
        if total > 32 * 1024**3 or len(members) > 100000:
            raise ValueError("Archive exceeds publication audit bounds")
        members[name] = record

    if path.name.endswith((".zip", ".whl")):
        with zipfile.ZipFile(path) as archive:
            for member in archive.infolist():
                if not member.is_dir():
                    with archive.open(member) as stream:
                        add(member.filename, stream, mode=member.external_attr)
    else:
        # Single-pass streaming avoids quadratic gzip seeks on large bundles.
        with tarfile.open(path, "r|gz") as archive:
            for member in archive:
                if member.isfile():
                    with archive.extractfile(member) as stream:
                        add(member.name, stream, mode=member.mode)
                elif member.issym() or member.islnk():
                    add(member.name, None, link=member.linkname, mode=member.mode,
                        link_type="symlink" if member.issym() else "hardlink")
                elif not member.isdir():
                    raise ValueError(f"Unexpected archive member type: {member.name}")
    return members


@lru_cache(maxsize=64)
def tag_file(tag, name):
    item = api(f"contents/{name}?ref={tag}")
    return base64.b64decode(item["content"])


def runtime_smoke(path, version):
    """Exercise the downloaded portable CLI/C ABI without loading any model."""
    import tempfile
    if os.getloadavg()[0] > 2 * (os.cpu_count() or 1):
        raise RuntimeError("Hosted runner too loaded for the bounded runtime smoke")
    available = next(int(line.split()[1]) for line in Path("/proc/meminfo").read_text().splitlines()
                     if line.startswith("MemAvailable:"))
    if available < 512 * 1024:
        raise RuntimeError("Hosted runner has less than 512 MiB available")
    env = {"PATH": os.environ["PATH"], "LANG": "C.UTF-8",
           "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}
    with tempfile.TemporaryDirectory(dir=SCRATCH) as folder:
        destination = Path(folder)
        with tarfile.open(path, "r:gz") as archive:
            archive.extractall(destination, filter="data")
        if path.name == "crispasr-linux-x86_64.tar.gz":
            executable = destination / "crispasr-linux-x86_64/crispasr"
            command = [str(executable), "--version"]
        else:
            lib = destination / "libcrispasr-linux-x86_64/lib"
            candidates = list(lib.glob("libcrispasr.so*")) or list(lib.glob("libwhisper.so*"))
            assert candidates, "Downloaded C ABI bundle contains no loadable library"
            library = sorted(candidates, key=lambda p: len(p.name))[0]
            command = [sys.executable, "-c",
                       "import ctypes,sys; lib=ctypes.CDLL(sys.argv[1]); "
                       "names=['crispasr_session_open','crispasr_session_transcribe',"
                       "'crispasr_session_synthesize']; "
                       "assert all(getattr(lib,n) for n in names); "
                       "print('Packaged C ABI loads and exports all three Session entry points')",
                       str(library)]
        process = subprocess.run(command, env=env, capture_output=True, text=True, timeout=30)
        (OUT / (path.name + ".smoke.log")).write_text(process.stdout + process.stderr)
        assert process.returncode == 0, (path.name, process.stdout, process.stderr)
        if path.name == "crispasr-linux-x86_64.tar.gz":
            assert version in process.stdout
        return dict(exit_code=process.returncode, stdout=process.stdout.strip(), no_model_loaded=True)


def archive_root(members):
    roots = {PurePosixPath(name).parts[0] for name in members}
    if len(roots) != 1:
        raise ValueError("Expected one project archive root")
    return next(iter(roots))


def notice_digests(content):
    canonical = content.replace(b"\r\n", b"\n")
    return {hashlib.sha256(value).hexdigest()
            for value in (content, canonical, canonical.replace(b"\n", b"\r\n"))}


def root_notices_present(members, notices):
    root = archive_root(members)
    return all(members.get(f"{root}/{name}", {}).get("sha256") in notice_digests(content)
               for name, content in notices.items())


def repair_notices(path, members, notices):
    root = archive_root(members)
    missing = {str(PurePosixPath(root) / name): content for name, content in notices.items()
               if members.get(f"{root}/{name}", {}).get("sha256") not in notice_digests(content)}
    if set(missing) & members.keys():
        raise ValueError("Refusing to replace an existing notice with different bytes")
    if not missing:
        return None
    repaired = path.parent / "repaired" / path.name
    repaired.parent.mkdir(exist_ok=True)
    if path.name.endswith(".zip"):
        with zipfile.ZipFile(path) as source, zipfile.ZipFile(repaired, "w") as target:
            for item in source.infolist():
                if item.is_dir():
                    target.writestr(copy.copy(item), b"")
                else:
                    with source.open(item) as src, target.open(copy.copy(item), "w", force_zip64=True) as dst:
                        shutil.copyfileobj(src, dst, 1024 * 1024)
            for name, content in missing.items():
                target.writestr(name, content, compress_type=zipfile.ZIP_DEFLATED)
    else:
        import io
        with tarfile.open(path, "r|gz") as source, tarfile.open(repaired, "w:gz") as target:
            for item in source:
                target.addfile(copy.copy(item), source.extractfile(item) if item.isfile() else None)
            for name, content in missing.items():
                item = tarfile.TarInfo(name)
                item.mode, item.size = 0o644, len(content)
                target.addfile(item, io.BytesIO(content))
    after = inventory(repaired)
    assert all(after.get(name) == value for name, value in members.items())
    assert set(after) - set(members) == set(missing)
    return repaired, after, sorted(missing)


def rename_asset(asset_id, name):
    return json.loads(subprocess.check_output(
        ["gh", "api", f"repos/{REPO}/releases/assets/{asset_id}",
         "--method", "PATCH", "-f", "name=" + name], text=True))


def delete_asset(asset_id):
    subprocess.run(["gh", "api", f"repos/{REPO}/releases/assets/{asset_id}",
                    "--method", "DELETE"], check=True)


def rename_asset_verified(asset_id, name):
    last_error = None
    for attempt in range(4):
        try:
            current = api(f"releases/assets/{asset_id}")
            if current["name"] == name:
                return current
            rename_asset(asset_id, name)
            current = api(f"releases/assets/{asset_id}")
            if current["name"] != name:
                raise RuntimeError("Asset rename did not reach its requested state")
            return current
        except Exception as error:
            last_error = error
            if attempt < 3:
                time.sleep(2**attempt)
    raise RuntimeError(f"Could not confirm asset {asset_id} named {name}") from last_error


def publish_repaired(args, asset, repaired, destination, proof, proof_path):
    """Verify staged bytes first, retain the old asset through promotion, and roll back."""
    name = asset["name"]
    suffix = ".tar.gz" if name.endswith(".tar.gz") else ".zip"
    staged_name = name[:-len(suffix)] + ".notice-staged-" + proof["repaired"]["sha256"][:16] + suffix
    backup_name = name + ".notice-backup-" + proof["original"]["sha256"][:16]
    staged_path = repaired.parent / staged_name
    shutil.copyfile(repaired, staged_path)
    subprocess.run(["gh", "release", "upload", args.tag, str(staged_path), "--repo", REPO], check=True)
    staged = {a["name"]: a for a in api(f"releases/tags/{args.tag}")["assets"]}[staged_name]
    verification = repaired.parent / "public-readback"
    assert download(staged["browser_download_url"] + f"?verify={time.time_ns()}", verification) == proof["repaired"]
    assert staged["digest"] == "sha256:" + proof["repaired"]["sha256"]
    proof.update(staged_asset_id=staged["id"], staged_bytes_verified=True,
                 original_asset_id=asset["id"], backup_name=backup_name)
    proof_path.write_text(json.dumps(proof, indent=2) + "\n")
    try:
        rename_asset_verified(asset["id"], backup_name)
        fresh = rename_asset_verified(staged["id"], name)
        readback = download(fresh["browser_download_url"] + f"?verify={time.time_ns()}", verification)
        assert readback == proof["repaired"]
        assert fresh["digest"] == "sha256:" + readback["sha256"]
        shutil.copyfile(verification, destination)
        proof.update(published=True, final_asset_id=fresh["id"], anonymous_readback=readback)
        proof_path.write_text(json.dumps(proof, indent=2) + "\n")
    except Exception as error:
        rollback_errors = []
        # Both renames are idempotent. Attempt them even if a prior API request
        # committed but its response was lost, so flags cannot hide a partial swap.
        try:
            rename_asset_verified(staged["id"], staged_name)
        except Exception as rollback_error:
            rollback_errors.append(str(rollback_error))
        try:
            rename_asset_verified(asset["id"], name)
            proof["original_restored"] = True
        except Exception as rollback_error:
            rollback_errors.append(str(rollback_error))
        state = {}
        for key, asset_id in [("original", asset["id"]), ("replacement", staged["id"])]:
            try:
                current = api(f"releases/assets/{asset_id}")
                state[key] = dict(id=asset_id, name=current["name"])
            except Exception as state_error:
                state[key] = dict(id=asset_id, error=str(state_error))
        proof.update(published=False, error=str(error), rollback_errors=rollback_errors,
                     recovery_asset_state=state)
        proof_path.write_text(json.dumps(proof, indent=2) + "\n")
        raise
    # Cleanup is separate from promotion: if deletion fails, the verified new
    # canonical asset stays available and the old backup remains recoverable.
    try:
        delete_asset(asset["id"])
    except Exception as error:
        proof["cleanup_error"] = str(error)
        proof_path.write_text(json.dumps(proof, indent=2) + "\n")
        raise
    proof["original_backup_removed_after_verification"] = True
    proof_path.write_text(json.dumps(proof, indent=2) + "\n")
    staged_path.unlink()
    verification.unlink()
    return fresh, readback


def parse_runtime_checksums(text, flavor):
    if flavor not in {"cuda", "cuda126", "cuda13"}:
        raise ValueError("Unreviewed CUDA runtime flavor")
    major = "13" if flavor == "cuda13" else "12"
    expected = {f"cudart64_{major}.dll", f"cublas64_{major}.dll", f"cublasLt64_{major}.dll"}
    result = {}
    for line in text.splitlines():
        if not line.strip():
            continue
        fields = line.split(maxsplit=1)
        if len(fields) != 2 or not re.fullmatch(r"[0-9a-fA-F]{64}", fields[0]):
            raise ValueError("Malformed CUDA runtime checksum")
        name = fields[1].strip().lstrip("*")
        if name in result:
            raise ValueError("Duplicate CUDA runtime checksum")
        result[name] = fields[0].lower()
    if set(result) != expected:
        raise ValueError("CUDA runtime checksum set must contain the exact three DLLs")
    return result



def validate_cuda_payloads(records):
    for flavor in ("cuda", "cuda126", "cuda13"):
        text = records[f"crispasr-windows-x86_64-{flavor}-runtime-sha256.txt"]["runtime_checksums"]
        checksums = parse_runtime_checksums(text, flavor)
        full = records[f"crispasr-windows-x86_64-{flavor}.zip"]["members"]
        runtime_entries = [(PurePosixPath(n).name, value) for n, value in full.items()
                           if PurePosixPath(n).name.lower().startswith(("cublas", "cudart"))]
        assert len(runtime_entries) == 3
        runtime_files = dict(runtime_entries)
        assert set(runtime_files) == set(checksums)
        for filename, digest in checksums.items():
            assert runtime_files[filename]["sha256"] == digest
        if flavor in {"cuda", "cuda126"}:
            runtime = records[f"{flavor}-runtime.zip"]["members"]
            assert len(runtime) == 3
            assert {PurePosixPath(n).name for n in runtime} == set(checksums)
            for n, value in runtime.items():
                expected = runtime_files[PurePosixPath(n).name]
                assert all(value[k] == expected[k] for k in ("bytes", "sha256"))
        if flavor in {"cuda", "cuda13"}:
            for filename, digest in checksums.items():
                assert records[filename]["sha256"] == digest
    for name, record in records.items():
        if "-non-cuda" not in name:
            continue
        full = records[name.replace("-non-cuda", "")]["members"]
        def binaries(members):
            return {str(PurePosixPath(n).relative_to(PurePosixPath(n).parts[0])): value
                    for n, value in members.items()
                    if n.lower().endswith((".dll", ".exe", ".lib", ".a"))
                    and not PurePosixPath(n).name.lower().startswith(("cublas", "cudart"))}
        original, split = binaries(full), binaries(record["members"])
        assert original.keys() == split.keys(), name
        for relative, value in split.items():
            assert all(original[relative].get(k) == value.get(k)
                       for k in ("bytes", "sha256", "link")), (name, relative)


def require_native_asset_set(baseline_names, actual_names, version):
    expected = {name.replace("0.8.44", version) for name in baseline_names if not name.endswith(".whl")}
    if linux_gpu_wheels_expected(version):
        expected.update({"libcrispasr-linux-x86_64-vulkan.tar.gz",
                         "libcrispasr-linux-arm64-cuda.tar.gz"})
    actual = {name for name in actual_names if not name.endswith(".whl")}
    if actual != expected:
        raise ValueError(f"Native asset set mismatch: missing={sorted(expected - actual)}, "
                         f"unexpected={sorted(actual - expected)}")


def native(args, result):
    run = api(f"actions/runs/{args.release_run}")
    assert run["status"] == "completed" and run["conclusion"] == "success", run
    assert run["head_sha"] == args.source and run["path"] == ".github/workflows/release.yml"
    assert run["event"] == "push" and run["head_branch"] == args.tag
    release = api(f"releases/tags/{args.tag}")
    baseline = api("releases/tags/v0.8.44")
    # Wheels are produced by a later workflow and audited separately.
    assets = {a["name"]: a for a in release["assets"] if not a["name"].endswith(".whl")}
    if any(".notice-staged-" in n or ".notice-backup-" in n for n in assets):
        raise ValueError("Unfinished notice repair: recover using the retained repair receipt before rerunning")
    require_native_asset_set([a["name"] for a in baseline["assets"]], assets, args.tag.removeprefix("v"))
    notices = {name: tag_file(args.tag, name) for name in ("LICENSE", "THIRD_PARTY_NOTICES.txt")}
    records = {}
    for name, asset in sorted(assets.items()):
        path = SCRATCH / name
        readback = download(asset["browser_download_url"] + f"?verify={time.time_ns()}", path)
        assert readback["bytes"] == asset["size"]
        assert asset["digest"] == "sha256:" + readback["sha256"]
        record = dict(asset_id=asset["id"], url=asset["browser_download_url"], **readback)
        if name.endswith((".tar.gz", ".zip")):
            members = inventory(path)
            project = name.startswith(("crispasr-", "libcrispasr-"))
            notices_present = root_notices_present(members, notices) if project else False
            if project and not notices_present:
                allowed = name in REPAIRABLE or name == f"crispasr-{args.tag}-xcframework.zip"
                if not args.repair_notices or not allowed:
                    raise ValueError(f"Native notices missing from {name}")
                repaired, after, added = repair_notices(path, members, notices)
                record["notice_repair"] = dict(original=readback, added=added,
                                                original_members_preserved=len(members))
                repair_dir = OUT / "notice-repairs"
                repair_dir.mkdir(exist_ok=True)
                proof = dict(asset=name, original=readback, repaired=sha_file(repaired),
                             added=added, all_original_members_byte_identical=True,
                             original_members=members, published=False)
                proof_path = repair_dir / (name + ".json")
                proof_path.write_text(json.dumps(proof, indent=2) + "\n")
                # If an upload/readback fails, retain the exact repaired payload
                # as a workflow artifact rather than losing a produced file.
                checkpoint = repair_dir / name
                shutil.copyfile(repaired, checkpoint)
                fresh, new_readback = publish_repaired(args, asset, repaired, path, proof, proof_path)
                record.update(new_readback, asset_id=fresh["id"])
                checkpoint.unlink()
                members = after
                repaired.unlink()
            record["members"] = members
            if name in {"crispasr-linux-x86_64.tar.gz", "libcrispasr-linux-x86_64.tar.gz"}:
                record["runtime_smoke"] = runtime_smoke(path, args.tag.removeprefix("v"))
            if "-non-cuda" in name:
                assert not any(PurePosixPath(n).name.lower().startswith(("cublas", "cudart"))
                               for n in members), name
        elif name.endswith("runtime-sha256.txt"):
            record["runtime_checksums"] = path.read_text(encoding="utf-8-sig")
        records[name] = record
        (OUT / "native-assets.json").write_text(json.dumps(records, indent=2) + "\n")
        path.unlink()
        print("Verified native asset:", name, flush=True)
    validate_cuda_payloads(records)
    result["native"] = dict(complete=True, release_run=args.release_run,
                            assets=len(records), all_full_anonymous_readbacks=True,
                            asset_records="native-assets.json")


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.hrefs = set()

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            self.hrefs.update(urllib.parse.unquote(v) for k, v in attrs if k == "href")


def load_native_inventory(args, result):
    if args.release_run:
        return json.loads((OUT / "native-assets.json").read_text())
    if not args.native_audit_run:
        raise ValueError("Wheel verification requires a successful native inventory producer")
    run = api(f"actions/runs/{args.native_audit_run}")
    assert run["status"] == "completed" and run["conclusion"] == "success"
    assert run["path"] == ".github/workflows/verify-release-publication.yml"
    artifacts = api(f"actions/runs/{args.native_audit_run}/artifacts?per_page=100")["artifacts"]
    selected = [a for a in artifacts if a["name"] == f"release-publication-{args.native_audit_run}"]
    assert len(selected) == 1 and not selected[0]["expired"]
    path = SCRATCH / "native-inventory.zip"
    with path.open("wb") as target:
        subprocess.run(["gh", "api", f"repos/{REPO}/actions/artifacts/{selected[0]['id']}/zip"],
                       stdout=target, check=True)
    with zipfile.ZipFile(path) as archive:
        for name in ("receipt.json", "native-assets.json"):
            assert archive.getinfo(name).file_size <= 64 * 1024**2
        receipt = json.loads(archive.read("receipt.json"))
        assert receipt["complete"] and receipt["acceptance"] and receipt["native"]["complete"]
        assert receipt["source"] == args.source and receipt["tag"] == args.tag
        assert receipt["audit_source"] == run["head_sha"]
        records = json.loads(archive.read("native-assets.json"))
    path.unlink()
    result["native_inventory_producer"] = dict(run=args.native_audit_run, source=run["head_sha"],
                                               artifact_id=selected[0]["id"])
    return records


def is_library(name):
    return name.endswith((".dll", ".dylib", ".so")) or ".so." in name


OPTIONAL_HELPERS = {"libcrispasr_helpers.so", "libcrispasr_helpers.dylib", "crispasr_helpers.dll"}


def bundle_libraries(members):
    root = archive_root(members)

    def resolve(name):
        seen = set()
        while "link" in members[name]:
            if name in seen or len(seen) >= 32:
                raise ValueError("Cyclic native library alias")
            seen.add(name)
            row = members[name]
            target = row["link"] if row["link_type"] == "hardlink" else posixpath.join(posixpath.dirname(name), row["link"])
            name = safe_name(posixpath.normpath(target))
            if not name.startswith(root + "/") or name not in members:
                raise ValueError("Native alias leaves its bundle or has no payload")
        return members[name]

    for sub in ("lib", "bin", "src", ""):
        folder = PurePosixPath(root) / sub
        names = [n for n in members if PurePosixPath(n).parent == folder and is_library(PurePosixPath(n).name)]
        if names:
            result = {PurePosixPath(n).name: resolve(n) for n in names
                      if PurePosixPath(n).name not in OPTIONAL_HELPERS}
            assert result and any("crispasr" in n or "whisper" in n for n in result)
            return result
    raise ValueError("Native bundle contains no principal libraries")


def verify_native_wheel(archive, expected, manifest):
    packaged = {PurePosixPath(n).name: n for n in archive.namelist()
                if PurePosixPath(n).parent == PurePosixPath("crispasr") and is_library(PurePosixPath(n).name)}
    assert expected.keys() <= packaged.keys(), sorted(expected.keys() - packaged.keys())
    for name, original in expected.items():
        with archive.open(packaged[name]) as stream:
            assert sha_stream(stream) == {k: original[k] for k in ("bytes", "sha256")}, name
    for name in packaged.keys() - expected.keys() - OPTIONAL_HELPERS:
        row = manifest["native_libraries"][name]
        assert row["kind"] == "installed-runtime" and row["copyright_package"]
        with archive.open(packaged[name]) as stream:
            assert sha_stream(stream)["sha256"] == row["staged_sha256"]
        copyright_name = "packages/" + row["copyright_package"].replace(":", "_") + "/copyright"
        assert copyright_name in manifest["files"]
    return len(expected)


# #557: Linux x86_64 +vulkan/+hip and aarch64 +cuda wheels exist from 0.8.47 on. Older tags
# keep their original exact distribution set.
LINUX_GPU_WHEELS_SINCE = (0, 8, 47)
LINUX_GPU_WHEEL_ASSETS = {
    "gpu-wheel-linux-x86_64-vulkan": "libcrispasr-linux-x86_64-vulkan.tar.gz",
    "gpu-wheel-linux-x86_64-hip": "libcrispasr-linux-x86_64-hip.tar.gz",
    "gpu-wheel-linux-arm64-cuda": "libcrispasr-linux-arm64-cuda.tar.gz",
}
GPU_INDEX_FLAVORS = ("cuda", "vulkan", "hip")


def linux_gpu_wheels_expected(version):
    return tuple(int(part) for part in version.split(".")[:3]) >= LINUX_GPU_WHEELS_SINCE


def gpu_index_flavors(version):
    # whl/hip first exists with the 0.8.47 Linux +hip wheel.
    return GPU_INDEX_FLAVORS if linux_gpu_wheels_expected(version) else ("cuda", "vulkan")


def wheel_local_suffix(artifact_name):
    if not artifact_name.startswith("gpu-"):
        return ""
    for flavor in GPU_INDEX_FLAVORS:
        if artifact_name.endswith("-" + flavor):
            return "+" + flavor
    raise ValueError(f"Unknown GPU wheel flavor: {artifact_name}")


def wheels(args, result):
    run = api(f"actions/runs/{args.wheel_run}")
    assert run["status"] == "completed" and run["conclusion"] == "success", run
    assert run["path"] == ".github/workflows/release-python-wheels.yml"
    assert run["event"] in {"workflow_run", "workflow_dispatch"}
    version = args.tag.removeprefix("v")
    native = load_native_inventory(args, result)
    native_assets = {
        "cpu-wheel-linux-x86_64": "libcrispasr-linux-x86_64.tar.gz",
        "cpu-wheel-linux-arm64": "libcrispasr-linux-arm64.tar.gz",
        "cpu-wheel-macos-arm64": "libcrispasr-macos-arm64.tar.gz",
        "cpu-wheel-windows-x86_64": "libcrispasr-windows-x86_64.tar.gz",
        "gpu-wheel-linux-x86_64-cuda": "libcrispasr-linux-x86_64-cuda.tar.gz",
        "gpu-wheel-windows-x86_64-cuda": "libcrispasr-windows-x86_64-cuda.tar.gz",
        "gpu-wheel-windows-x86_64-vulkan": "libcrispasr-windows-x86_64-vulkan.tar.gz",
    }
    if linux_gpu_wheels_expected(version):
        native_assets.update(LINUX_GPU_WHEEL_ASSETS)
    source_files = {item["name"] for item in api(f"contents/python/crispasr?ref={args.tag}")
                    if item["type"] == "file" and (item["name"].endswith(".py") or item["name"] == "_helpers.c")}
    assert source_files
    artifacts = api(f"actions/runs/{args.wheel_run}/artifacts?per_page=100")["artifacts"]
    wanted = {"sdist", *native_assets}
    selected = {a["name"]: a for a in artifacts if a["name"] in wanted}
    assert selected.keys() == wanted
    cpu = SCRATCH / "cpu"
    cpu.mkdir(exist_ok=True)
    distributions = {}
    for name, artifact in sorted(selected.items()):
        assert not artifact["expired"]
        archive_path = SCRATCH / "artifact.zip"
        with archive_path.open("wb") as stream:
            subprocess.run(["gh", "api", f"repos/{REPO}/actions/artifacts/{artifact['id']}/zip"],
                           stdout=stream, check=True)
        with zipfile.ZipFile(archive_path) as archive:
            files = [m for m in archive.infolist() if not m.is_dir()]
            assert len(files) == 1
            member = files[0]
            filename = PurePosixPath(safe_name(member.filename)).name
            assert filename.endswith((".whl", ".tar.gz"))
            destination = (cpu if name.startswith("cpu-") or name == "sdist" else SCRATCH) / filename
            with archive.open(member) as source, destination.open("wb") as target:
                shutil.copyfileobj(source, target, 1024 * 1024)
        record = dict(artifact_id=artifact["id"], **sha_file(destination))
        if filename.endswith(".whl"):
            with zipfile.ZipFile(destination) as archive:
                names = archive.namelist()
                metadata = [n for n in names if n.endswith(".dist-info/METADATA")]
                assert len(metadata) == 1
                fields = BytesParser().parsebytes(archive.read(metadata[0]))
                expected_version = version + wheel_local_suffix(name)
                assert fields["Name"] == "crispasr" and fields["Version"] == expected_version
                packaged_sources = {PurePosixPath(n).name for n in names
                                    if PurePosixPath(n).parent == PurePosixPath("crispasr")
                                    and (n.endswith(".py") or PurePosixPath(n).name == "_helpers.c")}
                assert packaged_sources == source_files
                for source in source_files:
                    actual = archive.read("crispasr/" + source)
                    expected = tag_file(args.tag, "python/crispasr/" + source)
                    # Git checkout may use CRLF on Windows. No other source
                    # rewriting is permitted by this packaging-only workflow.
                    assert actual.replace(b"\r\n", b"\n") == expected.replace(b"\r\n", b"\n")
                record["tag_package_sources_verified"] = len(source_files)
                notice = "crispasr/licenses/THIRD_PARTY_NOTICES.txt"
                assert archive.read(notice).replace(b"\r\n", b"\n") == tag_file(args.tag, "THIRD_PARTY_NOTICES.txt").replace(b"\r\n", b"\n")
                assert archive.read("crispasr/licenses/LICENSE").replace(b"\r\n", b"\n") == tag_file(args.tag, "LICENSE").replace(b"\r\n", b"\n")
                manifest = json.loads(archive.read("crispasr/licenses/manifest.json"))
                license_files = {n.removeprefix("crispasr/licenses/") for n in names
                                 if n.startswith("crispasr/licenses/") and not n.endswith("/")}
                assert license_files == set(manifest["files"]) | {"manifest.json"}
                for member_name, digest in manifest["files"].items():
                    with archive.open("crispasr/licenses/" + member_name) as stream:
                        assert sha_stream(stream)["sha256"] == digest
                record["native_notices_verified"] = len(manifest["files"])
                expected_native = bundle_libraries(native[native_assets[name]]["members"])
                record["qualified_native_libraries_verified"] = verify_native_wheel(archive, expected_native, manifest)
        distributions[filename] = record
        archive_path.unlink()
    sys.path.insert(0, str(ROOT / "python/tools"))
    from check_pypi_distributions import validate
    original = validate(cpu, version)
    with OPENER.open(f"https://pypi.org/pypi/crispasr/{version}/json", timeout=120) as response:
        pypi = json.load(response)
    public = {f["filename"]: f for f in pypi["urls"]}
    assert public.keys() == {d["filename"] for d in original}
    readbacks = []
    for record in original:
        item = public[record["filename"]]
        path = SCRATCH / "readback"
        downloaded = download(item["url"], path)
        assert downloaded == {k: record[k] for k in ("bytes", "sha256")}
        assert item["digests"]["sha256"] == downloaded["sha256"]
        readbacks.append(dict(filename=record["filename"], url=item["url"], **downloaded))
        path.unlink()
    assets = {a["name"]: a for a in api(f"releases/tags/{args.tag}")["assets"]}
    for flavor in gpu_index_flavors(version):
        url = f"https://crispstrobe.github.io/CrispASR/whl/{flavor}/crispasr/"
        with OPENER.open(url, timeout=120) as response:
            parser = Links()
            parser.feed(response.read().decode())
        for name, record in distributions.items():
            if f"+{flavor}" in name:
                asset = assets[name]
                assert urllib.parse.unquote(asset["browser_download_url"]) in parser.hrefs
                path = SCRATCH / "gpu-readback"
                assert download(asset["browser_download_url"] + f"?verify={time.time_ns()}", path) == {
                    k: record[k] for k in ("bytes", "sha256")}
                path.unlink()
    result["wheels"] = dict(complete=True, producer_run=args.wheel_run, producer_source=run["head_sha"],
                            original_distributions=distributions, pypi_anonymous_readbacks=readbacks,
                            gpu_index_and_full_readbacks_verified=True)


REVIEWED_TAGS = {"v0.8.45", "v0.8.46", "v0.8.47", "v0.8.48", "v0.8.49"}
NOTICE_REPAIR_TAGS = {"v0.8.45"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--release-run", type=int)
    parser.add_argument("--wheel-run", type=int)
    parser.add_argument("--native-audit-run", type=int)
    parser.add_argument("--repair-notices", action="store_true")
    args = parser.parse_args()
    # Reviewed releases only; add a tag here deliberately, never by pattern.
    assert args.tag in REVIEWED_TAGS, args.tag
    # Notice repair was reviewed for the v0.8.45 archive families only.
    assert not args.repair_notices or args.tag in NOTICE_REPAIR_TAGS, args.tag
    tag = api(f"git/ref/tags/{args.tag}")["object"]
    assert tag["type"] == "tag"
    assert api(f"git/tags/{tag['sha']}")["object"]["sha"] == args.source
    assert tag_file(args.tag, "VERSION").decode().strip() == args.tag[1:]
    assert args.release_run or args.wheel_run
    receipt = dict(tag=args.tag, source=args.source, audit_source=os.environ.get("GITHUB_SHA"),
                   complete=False, acceptance=False)
    try:
        if args.release_run:
            native(args, receipt)
        if args.wheel_run:
            wheels(args, receipt)
        receipt.update(complete=True, acceptance=True)
    finally:
        (OUT / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    (OUT / "summary.md").write_text(
        f"{args.tag}: full public-byte readback and original producer/notice checks PASS.\n")


if __name__ == "__main__":
    main()
