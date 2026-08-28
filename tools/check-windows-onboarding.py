#!/usr/bin/env python3
"""Keep the documented Windows first run identical to the packaged E2E."""

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent.parent
COMMANDS = (
    r".\crispasr.exe --version",
    r'.\crispasr.exe --backend kokoro -m auto --tts "The quick brown fox jumps over the lazy dog." --tts-output hello.wav',
    r".\crispasr.exe --backend parakeet -m auto -f .\hello.wav -l en",
)


def main() -> int:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    release = (ROOT / ".github/workflows/release.yml").read_text(encoding="utf-8")
    failures: list[str] = []
    for command in COMMANDS:
        if command not in readme:
            failures.append(f"README.md is missing: {command}")
        if command not in release:
            failures.append(f"release.yml E2E is missing: {command}")
    if failures:
        for failure in failures:
            print(f"FAIL: {failure}", file=sys.stderr)
        return 1
    print("RESULT: PASS — Windows README commands match the packaged-archive E2E")
    return 0


if __name__ == "__main__":
    sys.exit(main())
