#!/usr/bin/env python3
"""Recover one Kaggle output without buffering it in memory (see README.md)."""
import argparse
from contextlib import redirect_stderr, redirect_stdout
import hashlib
import os
from pathlib import Path
import re
import sys
import tempfile
from urllib.parse import urlsplit


class RecoveryError(Exception):
    """A safe diagnostic that contains no remote URLs or credentials."""


def find_artifact(api, request_factory, owner, kernel, artifact, version=None):
    """List all pages and require one exact file_name, never a basename/regex match."""
    matches = []
    token = ""
    seen = set()
    try:
        # SDK retry diagnostics can include private endpoint/account details.
        with open(os.devnull, "w") as quiet, redirect_stdout(quiet), redirect_stderr(quiet), \
                api.build_kaggle_client() as client:
            while True:
                request = request_factory()
                request.user_name = owner
                request.kernel_slug = kernel
                request.page_size = 100
                request.page_token = token
                if version:
                    request.version_label = version
                response = client.kernels.kernels_api_client.list_kernel_session_output(request)
                matches.extend(item.url for item in response.files or [] if item.file_name == artifact)
                token = response.next_page_token
                if not token:
                    break
                if token in seen:
                    raise RecoveryError("output listing repeated a page token")
                seen.add(token)
    except RecoveryError:
        raise
    except Exception:
        raise RecoveryError("output listing failed; check authentication, access and kernel/version") from None
    if len(matches) != 1:
        raise RecoveryError(f"expected one exact artifact match; found {len(matches)}")
    return matches[0]


def stream_artifact(url, destination, expected_bytes, expected_sha256, *, get=None):
    """Verify a unique sibling temporary file, then atomically replace destination.

    Memory is bounded by a 1 MiB chunk. All failures preserve the destination;
    interrupted/failed transfers are discarded rather than resumed or trusted.
    """
    if not isinstance(expected_bytes, int) or expected_bytes < 0:
        raise RecoveryError("expected bytes must be a nonnegative integer")
    if not re.fullmatch(r"[0-9a-fA-F]{64}", expected_sha256):
        raise RecoveryError("expected SHA-256 must contain 64 hexadecimal digits")
    try:
        parsed = urlsplit(url)
        if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password or parsed.fragment:
            raise ValueError
    except Exception:
        raise RecoveryError("artifact URL must be HTTPS without credentials or fragment") from None
    if get is None:
        try:
            import requests
        except ImportError:
            raise RecoveryError("requests is required for streamed recovery") from None
        get = requests.get
    destination = Path(destination)
    temporary = None
    stage = "download"
    try:
        with get(url, stream=True, timeout=(30, 120), allow_redirects=False,
                 headers={"Accept-Encoding": "identity"}) as response:
            if response.status_code != 200:
                raise RecoveryError(f"artifact download requires HTTP 200; received {response.status_code}")
            if response.headers.get("Content-Encoding", "identity").lower() != "identity":
                raise RecoveryError("unexpected HTTP content encoding")
            length = response.headers.get("Content-Length")
            if length is not None:
                if not re.fullmatch(r"[0-9]+", length) or int(length) != expected_bytes:
                    raise RecoveryError("HTTP Content-Length does not match expected bytes")
            stage = "temporary file creation"
            with tempfile.NamedTemporaryFile(mode="wb", prefix=f".{destination.name}.",
                                             suffix=".part", dir=destination.parent, delete=False) as output:
                temporary = Path(output.name)
                digest = hashlib.sha256()
                count = 0
                stage = "stream transfer or temporary file write"
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if not chunk:
                        continue
                    count += len(chunk)
                    if count > expected_bytes:
                        raise RecoveryError("download exceeds expected bytes")
                    output.write(chunk)
                    digest.update(chunk)
                if count != expected_bytes:
                    raise RecoveryError(f"download byte count mismatch: expected {expected_bytes}, received {count}")
                if digest.hexdigest() != expected_sha256.lower():
                    raise RecoveryError("download SHA-256 mismatch")
                stage = "temporary file flush"
                output.flush()
                os.fsync(output.fileno())
        stage = "atomic publication"
        os.replace(temporary, destination)
        temporary = None
        return count, digest.hexdigest()
    except RecoveryError:
        raise
    except Exception:
        # Requests/SDK/OS messages can contain signed URLs, owner names or paths.
        raise RecoveryError(f"{stage} failed; destination was not replaced") from None
    finally:
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                raise RecoveryError("temporary file cleanup failed; remove the leftover .part file") from None


def authenticated_api():
    """Use Kaggle's normal private configuration; map the repository token alias."""
    if os.environ.get("KAGGLE_TOKEN") and not os.environ.get("KAGGLE_API_TOKEN"):
        os.environ["KAGGLE_API_TOKEN"] = os.environ["KAGGLE_TOKEN"]
    try:
        # Some SDK versions print during import/auth. Never expose those messages.
        with open(os.devnull, "w") as quiet, redirect_stdout(quiet), redirect_stderr(quiet):
            from kaggle.api.kaggle_api_extended import KaggleApi
            from kagglesdk.kernels.types.kernels_api_service import ApiListKernelSessionOutputRequest
            api = KaggleApi()
            api.authenticate()
        return api, ApiListKernelSessionOutputRequest
    except Exception:
        raise RecoveryError("Kaggle authentication failed; check private environment/configuration and SDK installation") from None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kernel", help="kernel slug only; account comes from KAGGLE_ACCOUNT/KAGGLE_USERNAME")
    parser.add_argument("--version", help="optional Kaggle version label; default is latest session")
    parser.add_argument("--artifact", required=True, help="exact output file_name, including any directories")
    parser.add_argument("--destination", required=True, type=Path)
    parser.add_argument("--expected-bytes", required=True, type=int)
    parser.add_argument("--sha256", required=True)
    args = parser.parse_args(argv)
    try:
        owner = os.environ.get("KAGGLE_ACCOUNT") or os.environ.get("KAGGLE_USERNAME")
        if not owner or not re.fullmatch(r"[A-Za-z0-9_-]+", owner):
            raise RecoveryError("set KAGGLE_ACCOUNT or KAGGLE_USERNAME in the private environment")
        if not re.fullmatch(r"[A-Za-z0-9_-]+", args.kernel):
            raise RecoveryError("kernel must be a slug without an account or URL")
        api, request_factory = authenticated_api()
        url = find_artifact(api, request_factory, owner, args.kernel, args.artifact, args.version)
        count, digest = stream_artifact(url, args.destination, args.expected_bytes, args.sha256)
        print(f"Verified and published {count} bytes; SHA-256 {digest}")
        return 0
    except RecoveryError as error:
        print(f"recover-output: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
