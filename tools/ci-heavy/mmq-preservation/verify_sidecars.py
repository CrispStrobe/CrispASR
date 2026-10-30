"""Independent small-sidecar audit; never download preserved native ZIP payloads."""

import hashlib
import io
import json
import subprocess
import sys
import zipfile
from pathlib import Path, PurePosixPath
import requests

REPO = "CrispStrobe/CrispASR"
SOURCE = "04ded753e909da019a2cd391f651f51f5299b6f5"
PINS = {
    "OFF": (
        11532751916,
        520620096,
        "ddaf82ddf80992c9d5b8f86263cc23317e5b92f6377dcf9c365926fbb00fe244",
    ),
    "ON": (
        11532004516,
        520659350,
        "84a6b98729aba3412642959f09e60f41b029f66fe545375293b252b39768bfa1",
    ),
}


def api(path):
    return json.loads(subprocess.check_output(["gh", "api", f"repos/{REPO}/" + path]))


def sha(data):
    return hashlib.sha256(data).hexdigest()


def private_url(asset_id):
    token = subprocess.check_output(["gh", "auth", "token"], text=True).strip()
    try:
        response = requests.get(
            f"https://api.github.com/repos/{REPO}/releases/assets/{asset_id}",
            headers={
                "Authorization": "Bearer " + token,
                "Accept": "application/octet-stream",
            },
            stream=True,
            allow_redirects=False,
            timeout=30,
        )
        try:
            assert response.status_code in (
                301,
                302,
                303,
                307,
                308,
            ), "Expected signed asset redirect"
            location = response.headers["Location"]
            assert location.startswith("https://")
            return location
        finally:
            response.close()
    except requests.RequestException:
        raise RuntimeError("Asset redirect failed; credentials/URL omitted") from None


class Ranges(io.RawIOBase):
    def __init__(self, url, total):
        self.url = url
        self.total = total
        self.position = 0
        self.transferred = 0
        self.calls = 0
        self.session = requests.Session()

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        position = (
            offset
            if whence == 0
            else self.position + offset if whence == 1 else self.total + offset
        )
        assert 0 <= position <= self.total
        self.position = position
        return position

    def read(self, size=-1):
        size = (
            self.total - self.position
            if size < 0
            else min(size, self.total - self.position)
        )
        assert (
            size <= 1024**2 and self.transferred + size <= 4 * 1024**2
        ), "Refuse large local payload read"
        if not size:
            return b""
        start = self.position
        end = start + size - 1
        try:
            response = self.session.get(
                self.url,
                headers={
                    "Range": f"bytes={start}-{end}",
                    "Accept-Encoding": "identity",
                },
                stream=True,
                timeout=30,
            )
        except requests.RequestException:
            raise RuntimeError("Range request failed; signed URL omitted") from None
        try:
            assert (
                response.status_code == 206
            ), "Range unsupported; full body rejected before read"
            assert (
                response.headers["Content-Range"] == f"bytes {start}-{end}/{self.total}"
            )
            data = response.raw.read(size + 1)
            assert len(data) == size
        finally:
            response.close()
        self.position += size
        self.transferred += size
        self.calls += 1
        return data

    def close(self):
        self.session.close()
        super().close()


def main():
    folder = Path(sys.argv[1])
    out = Path(sys.argv[2])
    repo = Path(sys.argv[3]).resolve()
    receipt = json.loads((folder / "receipt.json").read_text())
    pin = json.loads((repo / "docs/historical/cuda126-mmq-experiment-2026-10-08.json").read_text())
    assert (
        receipt["complete"] is True
        and receipt["published"] is False
        and receipt["draft"] is True
    )
    assert (
        receipt["original_source"] == SOURCE and receipt["producer_run"] == 37733919241
    )
    assert receipt["gpu_executed"] is False and receipt["rebuilt"] is False
    assert set(receipt["modes"]) == {"OFF", "ON"}
    release = api(f'releases/{receipt["draft_release_id"]}')
    assert (
        release["draft"]
        and release["prerelease"]
        and release["target_commitish"] == SOURCE
    )
    assert (
        release["tag_name"] == "archive-issue483-mmq-20261008"
        and "producer 37733919241" in release["body"]
    )
    assets = api(f'releases/{release["id"]}/assets?per_page=100')
    result = {
        "draft_release_id": release["id"],
        "modes": {},
        "full_payload_downloaded_locally": False,
    }
    for mode, (artifact_id, nbytes, digest) in PINS.items():
        row = receipt["modes"][mode]
        assert (
            row["producer_artifact_id"] == artifact_id
            and row["original_bytes"] == nbytes
            and row["original_sha256"] == digest
        )
        assert (
            row["full_draft_asset_readback"] is True
            and row["manifest"] == pin["manifests"][mode]
        )
        assert row["gpu_executed"] is False and row["rebuilt"] is False
        matches = [a for a in assets if a["id"] == row["draft_asset_id"]]
        assert len(matches) == 1
        asset = matches[0]
        assert (
            asset["name"] == row["preservation_file"]
            and asset["size"] == row["preservation_bytes"]
        )
        if asset.get("digest"):
            assert asset["digest"] == "sha256:" + row["preservation_sha256"]
        source = Ranges(private_url(asset["id"]), asset["size"])
        small = {}
        with zipfile.ZipFile(source) as archive:
            entries = archive.infolist()
            names = [m.filename for m in entries]
            assert len(names) < 200 and len(set(names)) == len(names)
            assert all(
                not PurePosixPath(n).is_absolute()
                and ".." not in PurePosixPath(n).parts
                and "\\" not in n
                for n in names
            )
            original = f"original-artifact-{artifact_id}.zip"
            large = archive.getinfo(original)
            assert (
                large.file_size == large.compress_size == nbytes
                and large.compress_type == zipfile.ZIP_STORED
            )
            assert [n for n in names if n.endswith(".zip")] == [original]
            local_notices = folder / mode / "original-notices"
            expected = {
                original,
                "inventory.json",
                "notices/NVIDIA-CUDA-12.6.3-LICENSE.txt",
                "notices/nvidia-origin.json",
                "notices/ADDITIONAL-CUDA-NOTICE.txt",
            }
            expected |= {
                "notices/original/" + str(p.relative_to(local_notices))
                for p in local_notices.rglob("*")
                if p.is_file()
            }
            assert set(names) == expected
            normalized = {
                k.replace("\\", "/"): v for k, v in row["package_members"].items()
            }
            for member in entries:
                if member.filename == original:
                    continue
                assert member.file_size <= 1024**2
                data = archive.read(member)
                small[member.filename] = {"bytes": len(data), "sha256": sha(data)}
                if member.filename.startswith("notices/original/"):
                    relative = member.filename.removeprefix("notices/original/")
                    assert (
                        sha(data)
                        == normalized[relative]["sha256"]
                        == sha((local_notices / relative).read_bytes())
                    )
                elif member.filename in (
                    "notices/NVIDIA-CUDA-12.6.3-LICENSE.txt",
                    "notices/nvidia-origin.json",
                ):
                    assert (
                        data
                        == (
                            repo
                            / "tools/ci-heavy/mmq-preservation"
                            / PurePosixPath(member.filename).name
                        ).read_bytes()
                    )
                elif member.filename == "inventory.json":
                    excluded = {
                        "preservation_file",
                        "preservation_bytes",
                        "preservation_sha256",
                        "draft_asset_id",
                        "full_draft_asset_readback",
                    }
                    assert json.loads(data) == {
                        k: v for k, v in row.items() if k not in excluded
                    }
                else:
                    assert all(
                        x in data
                        for x in (
                            b"cudart64_12.dll",
                            b"cublas64_12.dll",
                            b"cublasLt64_12.dll",
                            b"does not grant standalone SDK redistribution rights",
                        )
                    )
        result["modes"][mode] = {
            "draft_asset_id": asset["id"],
            "wrapper_sha256": row["preservation_sha256"],
            "original_nested_bytes": large.file_size,
            "original_nested_crc32": large.CRC,
            "small_sidecars": small,
            "http_range_requests": source.calls,
            "transferred_bytes": source.transferred,
            "original_native_zip_payload_read_locally": False,
        }
        source.close()
    final = api(f'releases/{release["id"]}')
    assert (
        final["draft"] and final["prerelease"] and final["target_commitish"] == SOURCE
    )
    result["complete"] = True
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(
        "MMQ independent sidecar ranges verified:",
        sum(r["transferred_bytes"] for r in result["modes"].values()),
        "bytes; no native ZIP payload downloaded",
    )


if __name__ == "__main__":
    main()
