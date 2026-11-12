#!/usr/bin/env python3
"""One-shot, fail-closed launcher for the frozen RE-USE GPU diagnostic.

This program never schedules itself. See README.md before preparing an at job.
"""
import argparse
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timedelta, timezone
import hashlib
from importlib.metadata import version as package_version
import json
import math
import os
from pathlib import Path
import re
import subprocess
import tempfile

from recover_output import authenticated_api

CANDIDATE = "8c2503d21e2f983254a75e85dbc1e8d0e0e79da0"
CANDIDATE_DIR = "tools/kaggle/reuse-quality-followup-20261112"
SCRIPT_SHA = "5264912f1d2851b123849c4451e2b62d9029c9f98eef9de8b152cbb81e81452d"
METADATA_SHA = "701e0ec82af6ff9770892d75f5c69507b664c44daaa9f8491b96cde5d21a086b"
SOURCE = "529e073f8c9c99b688be5823e1be78443d50763e"
TARGET = "crispasr-reuse-quality-followup-20261112"
PRODUCER = "crispasr-reuse-ten-second-20261112"
NOT_BEFORE = datetime(2026, 11, 14, tzinfo=timezone.utc)
EXPIRES = datetime(2026, 11, 15, tzinfo=timezone.utc)
TIMEOUT_SECONDS = 3600
REQUIRED_SECONDS = TIMEOUT_SECONDS + 900
EVIDENCE = {
    "window-2-reference.gguf": "25da6651db2ea3ceb7910daec5ff7968db83aa29b56732f1f0f7114b2763cb02",
    "window-2-diff.log": "cd3f56d4d782c0a667cf62364866349613e983f7962cc0b2949bd9a1651e1d50",
    "reference-manifest.json": "2bd8acf4d3758a15834350c990b2bc4ebb4f63e09ab7146eb80a8d61589f15e1",
}


class Refusal(Exception):
    """Only fixed public diagnostics may be included in this exception."""


def require(condition, message):
    if not condition:
        raise Refusal(message)


def file_sha(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git(repo, *args):
    return subprocess.check_output(["git", "-C", str(repo), *args], stderr=subprocess.DEVNULL, timeout=30)


def pinned_candidate(repo, evidence):
    """Read immutable Git objects, never the current worktree or moving branch."""
    script = git(repo, "show", f"{CANDIDATE}:{CANDIDATE_DIR}/qualification.py")
    metadata = git(repo, "show", f"{CANDIDATE}:{CANDIDATE_DIR}/kernel-metadata.json")
    require(hashlib.sha256(script).hexdigest() == SCRIPT_SHA, "candidate script hash mismatch")
    require(hashlib.sha256(metadata).hexdigest() == METADATA_SHA, "candidate metadata hash mismatch")
    require(f"SOURCE_REVISION = '{SOURCE}'".encode() in script, "candidate runtime source mismatch")
    meta = json.loads(metadata)
    require(meta["id"] == "${KAGGLE_ACCOUNT}/" + TARGET and meta["title"] == TARGET,
            "candidate target mismatch")
    require(meta["kernel_sources"] == ["${KAGGLE_ACCOUNT}/" + PRODUCER], "producer attachment mismatch")
    require(meta["code_file"] == "qualification.py" and meta["enable_gpu"] == "true",
            "candidate resource metadata mismatch")
    for filename, expected in EVIDENCE.items():
        require(file_sha(evidence / filename) == expected, "retained producer evidence hash mismatch")
    return script, metadata


def quota_guard(quota, now):
    """Use structured seconds, including timedelta days and active reservations."""
    require(set(quota) == {"allowed", "used", "reserved", "refresh"}, "incomplete quota response")
    values = [quota[key] for key in ("allowed", "used", "reserved")]
    require(all(isinstance(v, (float, int)) and not isinstance(v, bool) and math.isfinite(v) and v >= 0
                for v in values), "invalid quota values")
    require(isinstance(quota["refresh"], datetime) and quota["refresh"].tzinfo is not None,
            "uncertain quota refresh time")
    require(NOT_BEFORE <= quota["refresh"] - timedelta(days=7) <= now < quota["refresh"],
            "quota window has not refreshed or is uncertain")
    require(quota["allowed"] - quota["used"] - quota["reserved"] >= REQUIRED_SECONDS,
            "insufficient unreserved GPU quota")
    return {key: quota[key] for key in ("allowed", "used", "reserved")}


def snapshot_guard(snapshot):
    require(snapshot["target_exists"] is False, "target kernel already exists; no duplicate submission")
    producer = snapshot["producer"]
    require(producer["version"] == 1 and producer["status"] == "COMPLETE" and producer["id"] > 0,
            "producer is not the expected completed version")
    require(bool(producer["last_run_time"]), "producer timestamp is uncertain")
    require(all(snapshot["files"].count(name) == 1 for name in EVIDENCE),
            "producer archive, native log or manifest is missing or duplicated")


def write_event(stream, state, **values):
    stream.write(json.dumps({"state": state, **values}, sort_keys=True) + "\n")
    stream.flush()
    os.fsync(stream.fileno())


def launch(repo, evidence, receipt, remote, owner, now):
    """Consume one persistent Git-common-dir claim before any remote operation."""
    require(NOT_BEFORE <= now < EXPIRES, "outside the authorized UTC launch window")
    require(bool(re.fullmatch(r"[A-Za-z0-9_-]+", owner or "")), "private account environment is missing")
    script, metadata = pinned_candidate(repo, evidence)
    common = Path(os.fsdecode(git(repo, "rev-parse", "--path-format=absolute", "--git-common-dir")).strip())
    state_dir = common / "one-shot-kaggle"
    state_dir.mkdir(mode=0o700, exist_ok=True)
    claim = state_dir / (TARGET + ".claim")
    try:
        fd = os.open(claim, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        raise Refusal("one-shot claim already exists; inspect the prior receipt, never retry automatically") from None
    with os.fdopen(fd, "w") as claimed:
        write_event(claimed, "claimed", candidate=CANDIDATE, script_sha256=SCRIPT_SHA,
                    utc=now.isoformat())
    # Failure or ambiguous submission leaves this claim intact, including after a crash.
    fd = os.open(receipt, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as events:
        write_event(events, "attempt", candidate=CANDIDATE, script_sha256=SCRIPT_SHA,
                    metadata_sha256=METADATA_SHA, source=SOURCE, utc=now.isoformat())
        stage = "account identity"
        try:
            require(remote.identity() == owner, "authenticated account differs from the private configured account")
            stage = "producer and duplicate check"
            before = remote.snapshot(owner)
            snapshot_guard(before)
            stage = "quota checks"
            quota = quota_guard(remote.quota(), now)
            write_event(events, "quota-accepted", seconds=quota, required_seconds=REQUIRED_SECONDS)
            stage = "producer stability check"
            after = remote.snapshot(owner)
            snapshot_guard(after)
            require(before == after, "producer metadata changed during preflight")
            rendered = metadata.decode().replace("${KAGGLE_ACCOUNT}", owner)
            require("${KAGGLE_ACCOUNT}" not in rendered, "unrendered metadata placeholder")
            with tempfile.TemporaryDirectory(prefix="reuse-one-shot-", dir=receipt.parent) as folder:
                folder = Path(folder)
                (folder / "qualification.py").write_bytes(script)
                (folder / "kernel-metadata.json").write_text(rendered)
                stage = "submission"
                write_event(events, "submit-intent", requested_timeout_seconds=TIMEOUT_SECONDS,
                            target=TARGET, producer=PRODUCER, producer_version=1)
                result = remote.submit(folder, TIMEOUT_SECONDS)
            require(result is not None and result > 0, "submission response did not confirm a version")
            write_event(events, "submitted", version=result)
        except Exception as error:
            # Never expose SDK errors, signed URLs, account names, credentials or paths.
            reason = str(error) if isinstance(error, Refusal) else "operation failed or outcome is uncertain"
            write_event(events, "refused-or-uncertain", stage=stage, reason=reason)
            raise Refusal(f"{stage}: {reason}; one-shot claim retained") from None


class KaggleRemote:
    """Installed official SDK adapter; no polling, retries or accelerator redraw."""
    def identity(self):
        require(package_version("kaggle") == "2.2.2" and package_version("kagglesdk") == "0.1.37",
                "SDK changed since no-retry submission behavior was reviewed")
        self.api, self.request_factory = authenticated_api()
        self.owner = self.api.config_values.get(self.api.CONFIG_NAME_USER)
        return self.owner

    def quota(self):
        response = self.api.quota_view()
        gpu = response.gpu_quota
        values = [gpu.total_time_allowed, gpu.time_used, gpu.time_reserved]
        require(all(isinstance(value, timedelta) for value in values), "unexpected GPU quota schema")
        return dict(zip(("allowed", "used", "reserved"), (value.total_seconds() for value in values)),
                    refresh=response.quota_refresh_time)

    def snapshot(self, owner):
        from kagglesdk.kernels.types.kernels_api_service import ApiGetKernelRequest
        # Exact nonexistent lookups return 403 on this backend. Use the official
        # authenticated mine/search listing, and refuse a full/truncated page.
        kernels = self.api.kernels_list(mine=True, search=TARGET, page=1, page_size=100)
        require(isinstance(kernels, list) and len(kernels) < 100,
                "target search is unavailable or truncated")
        require(all(isinstance(item.ref, str) and item.ref.startswith(owner + "/") for item in kernels),
                "target search returned an unexpected account")
        target_exists = any(item.ref == owner + "/" + TARGET for item in kernels)
        with self.api.build_kaggle_client() as client:
            request = ApiGetKernelRequest()
            request.user_name = owner
            request.kernel_slug = PRODUCER
            item = client.kernels.kernels_api_client.get_kernel(request).metadata
        require(item.ref == owner + "/" + PRODUCER, "producer identity is uncertain")
        status = self.api.kernels_status(owner + "/" + PRODUCER).status
        require(getattr(status, "name", None) == "COMPLETE", "producer is not complete")
        files = []
        token = ""
        tokens = set()
        with self.api.build_kaggle_client() as client:
            while True:
                request = self.request_factory()
                request.user_name = owner
                request.kernel_slug = PRODUCER
                request.page_size = 100
                request.page_token = token
                response = client.kernels.kernels_api_client.list_kernel_session_output(request)
                files.extend(entry.file_name for entry in response.files or [])
                token = response.next_page_token
                if not token:
                    break
                require(token not in tokens, "output listing repeated a page")
                tokens.add(token)
        return {"target_exists": target_exists, "files": sorted(files), "producer": {
            "id": item.id, "version": item.current_version_number, "status": status.name,
            "last_run_time": item.last_run_time.isoformat() if item.last_run_time else None}}

    def submit(self, folder, timeout):
        response = self.api.kernels_push(str(folder), timeout=str(timeout))
        require(not response.error and not response.invalid_dataset_sources and
                not response.invalid_kernel_sources and not response.invalid_model_sources,
                "Kaggle rejected the submission or an attached source")
        require(response.ref == self.owner + "/" + TARGET and response.kernel_id > 0 and
                response.version_number == 1, "submission identity or version was unexpected")
        return response.version_number


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--evidence-dir", required=True, type=Path)
    parser.add_argument("--receipt", required=True, type=Path)
    parser.add_argument("--execute-once", action="store_true", help="submit once if all guards pass")
    args = parser.parse_args(argv)
    try:
        require(args.execute_once, "reviewed execution requires --execute-once; this tool never schedules itself")
        # SDK diagnostics can include private usernames and signed URLs. Receipt events are explicit and safe.
        with open(os.devnull, "w") as quiet, redirect_stdout(quiet), redirect_stderr(quiet):
            launch(args.repo, args.evidence_dir, args.receipt, KaggleRemote(),
                   os.environ.get("KAGGLE_ACCOUNT") or os.environ.get("KAGGLE_USERNAME"),
                   datetime.now(timezone.utc))
        print("One-shot submission accepted; inspect the private local receipt and terminal run later.")
        return 0
    except Exception as error:
        reason = str(error) if isinstance(error, Refusal) else "local prerequisite failed; inspect private setup"
        print("reuse-one-shot: " + reason)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
