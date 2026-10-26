"""Hosted readback of release assets and original wheel publication inputs.

Optional repair adds missing project notices only to the four known legacy
archive families. Every original member must remain byte-identical. No builds,
model loads, dependency downloads or numerical changes occur here.
"""

import argparse
import base64
import copy
from email.parser import BytesParser
import hashlib
from html.parser import HTMLParser
import json
import os
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

    def add(name, stream, link=None, mode=None):
        nonlocal total
        name = safe_name(name)
        if name in members:
            raise ValueError(f"Duplicate archive member: {name}")
        record = dict(link=link) if link is not None else sha_stream(stream)
        record["mode"] = mode
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
                    add(member.name, None, link=member.linkname, mode=member.mode)
                elif not member.isdir():
                    raise ValueError(f"Unexpected archive member type: {member.name}")
    return members


def tag_file(tag, name):
    item = api(f"contents/{name}?ref={tag}")
    return base64.b64decode(item["content"])


def repair_notices(path, members, notices):
    roots = {PurePosixPath(name).parts[0] for name in members}
    if len(roots) != 1:
        raise ValueError(f"Expected one archive root for notice repair: {path.name}")
    root = next(iter(roots))
    missing = {str(PurePosixPath(root) / name): content for name, content in notices.items()
               if not any(PurePosixPath(n).name == name and
                          value.get("sha256") == hashlib.sha256(content).hexdigest()
                          for n, value in members.items())}
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


def native(args, result):
    run = api(f"actions/runs/{args.release_run}")
    assert run["status"] == "completed" and run["conclusion"] == "success", run
    assert run["head_sha"] == args.source and run["path"] == ".github/workflows/release.yml"
    release = api(f"releases/tags/{args.tag}")
    baseline = api("releases/tags/v0.8.44")
    expected = {a["name"].replace("0.8.44", args.tag.removeprefix("v")) for a in baseline["assets"]}
    # Wheels are produced by a later workflow and audited separately.
    expected = {name for name in expected if not name.endswith(".whl")}
    assets = {a["name"]: a for a in release["assets"] if not a["name"].endswith(".whl")}
    assert expected <= assets.keys(), sorted(expected - assets.keys())
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
            notices_present = all(any(PurePosixPath(n).name == wanted and
                                      value.get("sha256") == hashlib.sha256(content).hexdigest()
                                      for n, value in members.items())
                                  for wanted, content in notices.items())
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
                subprocess.run(["gh", "release", "upload", args.tag, str(repaired),
                                "--repo", REPO, "--clobber"], check=True)
                fresh = {a["name"]: a for a in api(f"releases/tags/{args.tag}")["assets"]}[name]
                new_readback = download(fresh["browser_download_url"] + f"?verify={time.time_ns()}", path)
                assert new_readback == sha_file(repaired)
                assert fresh["digest"] == "sha256:" + new_readback["sha256"]
                record.update(new_readback, asset_id=fresh["id"])
                proof.update(published=True, final_asset_id=fresh["id"], anonymous_readback=new_readback)
                proof_path.write_text(json.dumps(proof, indent=2) + "\n")
                checkpoint.unlink()
                members = after
                repaired.unlink()
            record["members"] = members
            if "-non-cuda" in name:
                assert not any(PurePosixPath(n).name.lower().startswith(("cublas", "cudart"))
                               for n in members), name
        elif name.endswith("runtime-sha256.txt"):
            record["runtime_checksums"] = path.read_text(encoding="utf-8-sig")
        records[name] = record
        (OUT / "native-assets.json").write_text(json.dumps(records, indent=2) + "\n")
        path.unlink()
        print("Verified native asset:", name, flush=True)
    for name, record in records.items():
        if name.endswith("runtime-sha256.txt"):
            flavor = name.removeprefix("crispasr-windows-x86_64-").removesuffix("-runtime-sha256.txt")
            full = records[f"crispasr-windows-x86_64-{flavor}.zip"]["members"]
            for line in record["runtime_checksums"].splitlines():
                digest, filename = line.split(maxsplit=1)
                filename = filename.strip().lstrip("*")
                found = [v for n, v in full.items() if PurePosixPath(n).name == filename]
                assert len(found) == 1 and found[0]["sha256"] == digest.lower()
                if flavor in {"cuda", "cuda126"}:
                    runtime = records[f"{flavor}-runtime.zip"]["members"]
                    paired = [v for n, v in runtime.items() if PurePosixPath(n).name == filename]
                    assert len(paired) == 1
                    assert all(paired[0][k] == found[0][k] for k in ("bytes", "sha256"))
                else:
                    assert records[filename]["sha256"] == digest.lower()
    for name, record in records.items():
        if "-non-cuda" in name:
            full = records[name.replace("-non-cuda", "")]["members"]
            by_relative_path = {str(PurePosixPath(n).relative_to(PurePosixPath(n).parts[0])): value
                                for n, value in full.items()}
            for n, value in record["members"].items():
                if n.lower().endswith((".dll", ".exe", ".lib", ".a")):
                    relative = str(PurePosixPath(n).relative_to(PurePosixPath(n).parts[0]))
                    original = by_relative_path.get(relative)
                    assert original and all(original.get(k) == value.get(k)
                                            for k in ("bytes", "sha256", "link")), (name, relative)
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


def wheels(args, result):
    run = api(f"actions/runs/{args.wheel_run}")
    assert run["status"] == "completed" and run["conclusion"] == "success", run
    assert run["path"] == ".github/workflows/release-python-wheels.yml"
    version = args.tag.removeprefix("v")
    artifacts = api(f"actions/runs/{args.wheel_run}/artifacts?per_page=100")["artifacts"]
    wanted = {"sdist", "cpu-wheel-linux-x86_64", "cpu-wheel-linux-arm64", "cpu-wheel-macos-arm64",
              "cpu-wheel-windows-x86_64", "gpu-wheel-linux-x86_64-cuda",
              "gpu-wheel-windows-x86_64-cuda", "gpu-wheel-windows-x86_64-vulkan"}
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
                expected_version = version + ("+vulkan" if "vulkan" in name else "+cuda" if name.startswith("gpu-") else "")
                assert fields["Name"] == "crispasr" and fields["Version"] == expected_version
                notice = "crispasr/licenses/THIRD_PARTY_NOTICES.txt"
                assert archive.read(notice) == tag_file(args.tag, "THIRD_PARTY_NOTICES.txt")
                manifest = json.loads(archive.read("crispasr/licenses/manifest.json"))
                for member_name, digest in manifest["files"].items():
                    with archive.open("crispasr/licenses/" + member_name) as stream:
                        assert sha_stream(stream)["sha256"] == digest
                record["native_notices_verified"] = len(manifest["files"])
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
    for flavor in ("cuda", "vulkan"):
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--release-run", type=int)
    parser.add_argument("--wheel-run", type=int)
    parser.add_argument("--repair-notices", action="store_true")
    args = parser.parse_args()
    assert args.tag == "v0.8.45"  # the reviewed release; do not silently generalize policy
    tag = api(f"git/ref/tags/{args.tag}")["object"]
    assert tag["type"] == "tag"
    assert api(f"git/tags/{tag['sha']}")["object"]["sha"] == args.source
    assert tag_file(args.tag, "VERSION").decode().strip() == args.tag[1:]
    assert args.release_run or args.wheel_run
    receipt = dict(tag=args.tag, source=args.source, complete=False, acceptance=False)
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
