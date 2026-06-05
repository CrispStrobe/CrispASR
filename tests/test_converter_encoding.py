"""Guard test: all convert-*-to-gguf.py text-mode open() calls must specify encoding="utf-8".

Issue #139 — on Windows with non-UTF-8 system locale, bare open() decodes UTF-8
bytes as the system ANSI code page (e.g. GBK), corrupting tokenizer vocab in the GGUF.
"""

import pathlib
import re
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"


def _text_opens_without_encoding(path: pathlib.Path) -> list[tuple[int, str]]:
    """Return (line_number, line) for text-mode open() calls missing encoding=."""
    hits: list[tuple[int, str]] = []
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        # Skip comments
        stripped = line.lstrip()
        if stripped.startswith("#"):
            continue
        # Must contain open(
        if "open(" not in line:
            continue
        # Skip non-builtin opens (safe_open, tarfile.open, zipfile.open, etc.)
        if re.search(r'(safe_open|tarfile\.open|zipfile\.open|gzip\.open)\s*\(', line):
            continue
        # Skip binary mode — look for "rb"/"wb"/"r+b" etc. anywhere after open(
        if re.search(r"""['"]r?[+]?b['"]""", line):
            continue
        if re.search(r"""['"]wb['"]""", line):
            continue
        # At this point it's a text-mode builtin open() — must have encoding=
        if "encoding=" not in line:
            hits.append((i, line.strip()))
    return hits


class TestConverterEncoding(unittest.TestCase):
    def test_all_text_opens_specify_utf8(self) -> None:
        scripts = sorted(MODELS_DIR.glob("convert-*-to-gguf.py"))
        self.assertGreater(len(scripts), 0, "no converter scripts found")

        violations: list[str] = []
        for script in scripts:
            for lineno, line in _text_opens_without_encoding(script):
                violations.append(f"  {script.name}:{lineno}: {line}")

        self.assertEqual(
            violations,
            [],
            "Text-mode open() without encoding='utf-8' — will break on "
            f"Windows with non-UTF-8 locale:\n" + "\n".join(violations),
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
