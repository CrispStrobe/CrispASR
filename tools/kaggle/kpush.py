#!/usr/bin/env python3
"""Push a Kaggle kernel directory without hardcoding the Kaggle account.

Kernel dirs under tools/kaggle/ carry the placeholder ${KAGGLE_ACCOUNT} in
kernel-metadata.json (id, dataset_sources, kernel_sources, model_sources) and
may use it in the kernel's code file. This renders a temporary copy with the
account from the environment and pushes that, authenticated with the token
from the environment - nothing account- or token-specific is ever committed.

    KAGGLE_ACCOUNT=<username> KAGGLE_TOKEN=<api token> \\
        python3 tools/kaggle/kpush.py tools/kaggle/<kernel-dir> [extra kaggle push args]

In GitHub Actions both come from the repository secrets of the same names.
Locally, export them from a private env file (never from a command line).
One Kaggle account per person (Kaggle Terms of Use): do not point this at a
second account to get around quotas or session limits.
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

PLACEHOLDER = "${KAGGLE_ACCOUNT}"


def render(text: str, account: str) -> str:
    return text.replace(PLACEHOLDER, account)


def main(argv):
    if len(argv) < 2:
        sys.exit(__doc__)
    src = Path(argv[1]).resolve()
    account = (os.environ.get("KAGGLE_ACCOUNT") or os.environ.get("KAGGLE_USERNAME") or "").strip()
    token = (os.environ.get("KAGGLE_TOKEN") or os.environ.get("KAGGLE_KEY") or os.environ.get("KAGGLE_API_TOKEN") or "").strip()
    if not account or not token:
        sys.exit("kpush: set KAGGLE_ACCOUNT and KAGGLE_TOKEN (GitHub secrets / private env file)")
    meta_path = src / "kernel-metadata.json"
    if not meta_path.is_file():
        sys.exit(f"kpush: no kernel-metadata.json in {src}")
    with tempfile.TemporaryDirectory(prefix="kpush-") as tmp:
        dst = Path(tmp) / src.name
        shutil.copytree(src, dst)
        meta = json.loads(render(meta_path.read_text(), account))
        code = meta.get("code_file")
        if code and (dst / code).is_file():
            (dst / code).write_text(render((dst / code).read_text(), account))
        (dst / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
        leftover = [p.name for p in (dst / "kernel-metadata.json", dst / (code or "")) if p.is_file() and PLACEHOLDER in p.read_text()]
        if leftover:
            sys.exit(f"kpush: unrendered placeholder in {leftover}")
        env = dict(os.environ, KAGGLE_API_TOKEN=token, KAGGLE_USERNAME=account, KAGGLE_KEY=token)
        return subprocess.call([sys.executable, "-c", "from kaggle.cli import main; main()", "kernels", "push", "-p", str(dst)] + argv[2:], env=env)


if __name__ == "__main__":
    sys.exit(main(sys.argv))
