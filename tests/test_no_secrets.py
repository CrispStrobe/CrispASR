"""Tracked files must not contain credentials.

Three Kaggle API tokens sat in public tools/kaggle/* scripts from 2026-07 to
2026-09 (export KAGGLE_API_TOKEN=KGAT_...) and had to be revoked and scrubbed
from history. This scans every tracked text file for token formats we use, so
the next one fails CI instead of shipping.
"""
import re
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PATTERNS = {
    "kaggle api token": re.compile(r"KGAT_[0-9a-f]{20,}"),
    "huggingface token": re.compile(r"\bhf_[A-Za-z0-9]{30,}"),
    "github token": re.compile(r"\b(?:ghp|gho|ghs|ghu)_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,}"),
    "openai-style key": re.compile(r"\bsk-[A-Za-z0-9]{32,}"),
    "aws access key": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    "private key block": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |)PRIVATE KEY-----"),
}


# Deliberately public, documented material. Keep this list tiny and justified.
ALLOW = {
    # Fixed self-signed C2PA "AI-generated" marker key, public by design (see the
    # header comment in src/core/crispasr_c2pa_default_cert.h): not a trust anchor.
    ("assets/c2pa/crispasr-default-c2pa.key", "private key block"),
    ("src/core/crispasr_c2pa_default_cert.h", "private key block"),
}


# Public LibriSpeech test-clean corpus mirror (OpenSLR 12, CC BY 4.0).
# This exact ref is allowed only in dataset_sources, never as a kernel/model ID.
PUBLIC_KAGGLE_DATASETS = frozenset({'ayoubakanoun/librispeechtest'})


def literal_kaggle_metadata_refs(metadata):
    bad = []
    for field in ('id', 'dataset_sources', 'kernel_sources', 'model_sources'):
        refs = [metadata.get(field, '')] if field == 'id' else metadata.get(field) or []
        for ref in refs:
            if not ref or ref.startswith('${KAGGLE_ACCOUNT}/'):
                continue
            if field == 'dataset_sources' and ref in PUBLIC_KAGGLE_DATASETS:
                continue
            bad.append(ref)
    return bad


def scan(root: Path):
    files = subprocess.run(["git", "ls-files", "-z"], cwd=root, capture_output=True, check=True).stdout.split(b"\0")
    hits = []
    for rel in filter(None, files):
        p = root / rel.decode()
        try:
            data = p.read_bytes()
        except OSError:
            continue
        if b"\0" in data[:8192]:  # binary
            continue
        text = data.decode("utf-8", "replace")
        for name, rx in PATTERNS.items():
            if (rel.decode(), name) in ALLOW:
                continue
            for m in rx.finditer(text):
                line = text.count("\n", 0, m.start()) + 1
                hits.append(f"{rel.decode()}:{line}: {name} ({m.group(0)[:8]}...)")
    return hits


class NoSecretsTest(unittest.TestCase):
    def test_private_maintainer_locations_not_in_docs(self):
        # Public evidence uses portable aliases; physical locations belong in
        # ignored maintainer notes. Hosted CI's /mnt/heavy is a public runner path.
        pattern = re.compile(r"/mnt/(?:volume1|storage)\b|/home/(?:claude|codex)[^/\s]*/")
        self.assertIsNotNone(pattern.search("/mnt/" + "volume1/private-checkout"))
        self.assertIsNone(pattern.search("${CRISPASR_STORAGE}/fixtures"))
        files = subprocess.check_output(["git", "ls-files", "-z", "--", "*.md", "*.json"], cwd=ROOT)
        hits = []
        for name in filter(None, files.decode().split("\0")):
            path = ROOT / name
            if path.is_file() and pattern.search(path.read_text(errors="replace")):
                hits.append(name)
        self.assertEqual(hits, [], "private maintainer locations in tracked documentation:\n" + "\n".join(hits))

    def test_no_credentials_in_tracked_files(self):
        hits = scan(ROOT)
        self.assertEqual(hits, [], "credentials in tracked files:\n" + "\n".join(hits))

    def test_kaggle_metadata_uses_account_placeholder(self):
        # The Kaggle account comes from the KAGGLE_ACCOUNT secret, rendered at push
        # time by tools/kaggle/kpush.py - never a literal owner in the repo.
        import json
        public = 'ayoubakanoun/librispeechtest'
        self.assertEqual(literal_kaggle_metadata_refs({'dataset_sources': [public]}), [])
        self.assertEqual(literal_kaggle_metadata_refs({
            'id': '${KAGGLE_ACCOUNT}/quality',
            'dataset_sources': ['${KAGGLE_ACCOUNT}/pinned-model', public],
            'kernel_sources': ['${KAGGLE_ACCOUNT}/reference'],
            'model_sources': ['${KAGGLE_ACCOUNT}/model'],
        }), [])
        for ref in (public + '/extra', public + '-other', 'other-owner/librispeechtest',
                    'ayoubakanoun/other-dataset', 'literal-private-owner/pinned-model'):
            with self.subTest(rejected_dataset=ref):
                self.assertEqual(literal_kaggle_metadata_refs({'dataset_sources': [ref]}), [ref])
        for field in ('id', 'kernel_sources', 'model_sources'):
            with self.subTest(rejected_field=field):
                value = public if field == 'id' else [public]
                self.assertEqual(literal_kaggle_metadata_refs({field: value}), [public])
        bad = []
        for f in sorted(ROOT.glob("tools/kaggle/**/kernel-metadata.json")):
            m = json.loads(f.read_text())
            bad += [f"{f.relative_to(ROOT)}: {r}" for r in literal_kaggle_metadata_refs(m)]
        self.assertEqual(bad, [], "literal Kaggle owners in kernel metadata:\n" + "\n".join(bad))

    def test_kaggle_account_name_not_committed(self):
        # CI passes the KAGGLE_ACCOUNT secret; the name itself must not appear in
        # any tracked file (skipped where the secret is unavailable, e.g. forks).
        import os
        name = os.environ.get("KAGGLE_ACCOUNT", "").strip()
        if not name:
            self.skipTest("KAGGLE_ACCOUNT not set")
        rx = re.compile(re.escape(name), re.I)
        files = subprocess.run(["git", "ls-files", "-z"], cwd=ROOT, capture_output=True, check=True).stdout.split(b"\0")
        hits = []
        for rel in filter(None, files):
            try:
                data = (ROOT / rel.decode()).read_bytes()
            except OSError:
                continue
            if b"\0" not in data[:8192] and rx.search(data.decode("utf-8", "replace")):
                hits.append(rel.decode())
        self.assertEqual(hits, [], "Kaggle account name committed in:\n" + "\n".join(hits))

    def test_patterns_fire(self):
        # Positive control: each pattern must match a synthetic sample.
        samples = {
            "kaggle api token": "KGAT_" + "0" * 32,
            "huggingface token": "hf_" + "a" * 34,
            "github token": "ghp_" + "b" * 36,
            "openai-style key": "sk-" + "c" * 40,
            "aws access key": "AKIA" + "D" * 16,
            "private key block": "-----BEGIN RSA " + "PRIVATE KEY-----",  # split: this file is scanned too
        }
        for name, s in samples.items():
            self.assertTrue(PATTERNS[name].search(s), name)


if __name__ == "__main__":
    sys.exit(unittest.main())
