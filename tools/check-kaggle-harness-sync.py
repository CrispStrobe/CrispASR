#!/usr/bin/env python3
"""Assert every bundled kaggle_harness.py matches the canonical one.

Each tools/kaggle/<kernel>/ directory ships its own copy of
tools/kaggle/kaggle_harness.py, intended as the fallback used when the
in-kernel `git clone` fails (CPU workers get no internet at all).

⚠ THAT FALLBACK DOES NOT WORK — PROVEN IN PRODUCTION 2026-09-03. A
sidon-quant-cuda draw landed on a worker with no internet; the clone failed
("could not read Username for 'https://github.com'"), the script took the
fallback branch, and the run died with:

    ModuleNotFoundError: No module named 'kaggle_harness'

That is the exact no-internet scenario these copies exist for, happening for
real, with the protection absent — so `kaggle kernels push` on a script kernel
uploads ONLY `code_file` and the bundled copy never reaches the worker.

(Note for anyone re-deriving this: `kaggle kernels pull` returning a single .py
does NOT establish it — pull is selective by design, its `--metadata` flag
*generates* kernel-metadata.json rather than fetching one, so it says nothing
about what push uploaded. The ModuleNotFoundError is the evidence; the pull
observation is not.)

Consequences: keeping the copies byte-identical is cheap hygiene and nothing
more — a green run of this check is NOT evidence that the no-internet path
works, because that path is broken for every script kernel in the tree. The
only delivery route that survives no internet is publishing the harness as a
Kaggle DATASET listed in each kernel's `dataset_sources` (the mechanism the
hf-token dataset already uses), after which these copies and this check can be
retired. That is a cross-kernel policy change spanning several owners, so it
belongs to the maintainer rather than to whoever reads this next.

Note also that 7 of the 61 bundled copies are untracked (a `.gitignore` glob
added after 54 had already been committed), so they cannot propagate through a
commit at all — harmless while the copies are local-testing conveniences,
load-bearing if anyone revives the fallback without fixing the delivery route.

Nothing kept those copies in sync. On 2026-07-20 there were **four** distinct
versions across 53 files, and the canonical one was used by exactly one kernel.
That is a latent, silent failure: the drift only bites on the clone-failure
path, which is precisely the path nobody exercises until a worker has no
network — and then the kernel runs an arbitrarily old harness (missing token
mount paths, missing ccache fixes) and fails in a way that looks like a Kaggle
problem rather than a stale bundle.

Fix drift with:
    for f in $(find tools/kaggle -name kaggle_harness.py \\
                 -not -path tools/kaggle/kaggle_harness.py); do
        cp tools/kaggle/kaggle_harness.py "$f"
    done
"""

import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CANON = ROOT / "tools" / "kaggle" / "kaggle_harness.py"


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> int:
    if not CANON.is_file():
        print(f"ERROR: canonical harness missing at {CANON}", file=sys.stderr)
        return 1
    want = sha(CANON)
    copies = sorted(p for p in (ROOT / "tools" / "kaggle").rglob("kaggle_harness.py")
                    if p.resolve() != CANON.resolve())
    drifted = [p for p in copies if sha(p) != want]

    print(f"canonical: {CANON.relative_to(ROOT)}  sha256={want[:16]}")
    print(f"bundled copies: {len(copies)}   drifted: {len(drifted)}")
    if drifted:
        print("\nERROR: bundled kaggle_harness.py copies differ from the canonical one:",
              file=sys.stderr)
        for p in drifted:
            print(f"  {p.relative_to(ROOT)}  sha256={sha(p)[:16]}", file=sys.stderr)
        print("\nRe-copy tools/kaggle/kaggle_harness.py over each (see this file's "
              "docstring).", file=sys.stderr)
        return 1
    print("OK: every bundled harness matches the canonical copy.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
