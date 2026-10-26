"""Carry the native bundle's license notices into a platform wheel.

This is packaging QA: runtime/package sources still come from the release tag.
No license is replaced and no new redistribution permission is inferred.
"""

import argparse
import hashlib
import json
import platform
import shutil
import subprocess
import zipfile
from pathlib import Path


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def stage(bundle, package, pyproject):
    bundle, package, pyproject = map(Path, (bundle, package, pyproject))
    required = [bundle / "LICENSE", bundle / "THIRD_PARTY_NOTICES.txt"]
    for source in required:
        if not source.is_file() or not source.read_bytes().strip():
            raise ValueError(f"Missing or empty native bundle notice: {source}")
    if not (package / "_binding.py").is_file():
        raise ValueError("Not a crispasr package directory")
    text = pyproject.read_text()
    section = "[tool.setuptools.package-data]"
    if text.count(section) != 1:
        raise ValueError("Expected one explicit package-data section")
    before, data = text.split(section)
    anchor = 'crispasr = [\n'
    if data.count(anchor) != 1:
        raise ValueError("Expected the crispasr package-data list")
    # A tag predating this helper does not include text/data in its wheel.
    # Change only that packaging list; version and runtime files stay pinned.
    if '"licenses/**/*"' not in data:
        data = data.replace(anchor, anchor + '    "licenses/**/*",\n', 1)

    destination = package / "licenses"
    if destination.exists():
        raise ValueError("Refusing to merge a stale wheel notice directory")
    destination.mkdir()
    for source in required:
        shutil.copyfile(source, destination / source.name)
    if (bundle / "licenses").is_dir():
        shutil.copytree(bundle / "licenses", destination / "native")

    packages = []
    if platform.system() == "Linux":
        # Preserve the distribution's full license texts and package copyright
        # files for any external libraries staged from the runner. libcrispasr
        # and its own build targets are covered by the native bundle notices.
        common = Path("/usr/share/common-licenses")
        if not common.is_dir():
            raise ValueError("Linux runner has no common license texts")
        shutil.copytree(common, destination / "common", symlinks=False)
        listing = subprocess.check_output(["ldconfig", "-p"], text=True)
        paths = {}
        for line in listing.splitlines():
            if " => " in line:
                name = line.split()[0]
                paths.setdefault(name, []).append(Path(line.split(" => ", 1)[1]))
        seen = set()
        for library in package.iterdir():
            if not library.is_file() or library.name not in paths:
                continue
            for installed in paths[library.name]:
                candidates = [installed, installed.resolve()]
                candidates += [Path(str(p).replace("/usr/lib/", "/lib/", 1))
                               for p in list(candidates) if str(p).startswith("/usr/lib/")]
                owner = None
                for candidate in candidates:
                    result = subprocess.run(["dpkg-query", "-S", str(candidate)],
                                            capture_output=True, text=True)
                    if result.returncode == 0:
                        owner = result.stdout.splitlines()[0].rsplit(": ", 1)[0]
                        break
                if owner is None:
                    raise ValueError(f"Cannot identify installed runtime: {library.name}")
                if owner in seen:
                    continue
                seen.add(owner)
                copyright_file = Path("/usr/share/doc") / owner.split(":")[0] / "copyright"
                if not copyright_file.is_file() or not copyright_file.read_bytes().strip():
                    raise ValueError(f"Missing package copyright: {owner}")
                target = destination / "packages" / owner.replace(":", "_")
                target.mkdir(parents=True)
                shutil.copyfile(copyright_file, target / "copyright")
                identity = subprocess.check_output(
                    ["dpkg-query", "-W", "-f",
                     "${binary:Package}\t${Version}\t${source:Package}\t${source:Version}\n",
                     owner], text=True).strip()
                packages.append(dict(binary_package=owner, identity=identity))

    manifest = {
        "schema": "crispasr-wheel-notices-v1",
        "native_bundle_notices_preserved": True,
        "runtime_packages": packages,
        "files": {str(p.relative_to(destination)): digest(p)
                  for p in sorted(destination.rglob("*")) if p.is_file()},
        "scope": "Notice preservation and installed package identity; no new license grant.",
    }
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    pyproject.write_text(before + section + data)
    print(json.dumps({"notice_files": len(manifest["files"]),
                      "runtime_packages": len(packages)}))
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--package", required=True, type=Path)
    parser.add_argument("--pyproject", type=Path)
    parser.add_argument("--verify-wheel", type=Path)
    args = parser.parse_args()
    if args.verify_wheel:
        verify_wheel(args.verify_wheel, args.package)
    else:
        if not args.bundle or not args.pyproject:
            parser.error("staging requires --bundle and --pyproject")
        stage(args.bundle, args.package, args.pyproject)


def verify_wheel(wheel, package):
    directory = Path(package) / "licenses"
    expected = {"crispasr/licenses/" + str(p.relative_to(directory)): digest(p)
                for p in directory.rglob("*") if p.is_file()}
    if not expected or "crispasr/licenses/THIRD_PARTY_NOTICES.txt" not in expected:
        raise ValueError("No staged native notices to verify")
    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError("Duplicate wheel members")
        for name, wanted in expected.items():
            if name not in names:
                raise ValueError(f"Wheel omitted native notice: {name}")
            h = hashlib.sha256()
            with archive.open(name) as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    h.update(block)
            if h.hexdigest() != wanted:
                raise ValueError(f"Wheel changed native notice: {name}")
    print(json.dumps({"verified_wheel": Path(wheel).name,
                      "native_notice_files_verified": len(expected)}))


if __name__ == "__main__":
    main()
