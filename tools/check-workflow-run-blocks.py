#!/usr/bin/env python
"""Lint the shell inside GitHub Actions `run:` blocks.

    python tools/check-workflow-run-blocks.py

Why this exists
---------------
Issue #339: all six Linux binary tarballs of v0.8.26 failed to build, and two
of the causes were plain shell mistakes sitting in `release.yml`:

  1. A literal `\\n` where a line continuation was meant:

         python3 scripts/check-bundled-deps.py release/... \\n   --allow '...'

     Inside a YAML `run: |` block those two characters reach bash verbatim.
     Bash reads `\\n` as an escaped `n`, so a stray positional argument `n`
     lands on the command — `error: unrecognized arguments: n`, exit 2. It hit
     four jobs (cuda, cuda13, hip, vulkan) and nothing caught it, because
     release.yml only ever executes on a tag push: the first time this shell
     runs is the release you were trying to cut.

  2. An assertion ordered before the step that satisfies it (not detectable
     here — that one needs a human reading the step).

`release.yml` is the least-tested code in the repo and the most expensive to
get wrong, so the cheap static checks belong in CI rather than in a tag push.
"""

import re
import os
import subprocess
import sys
from pathlib import Path

import yaml

WORKFLOWS = Path(".github/workflows")
# `<<EOF`, `<<'PY'`, `<< -"X"` … an inline script whose body is not shell.
HEREDOC = re.compile(r"<<-?\s*(?P<tag>'[^']+'|\"[^\"]+\"|[A-Za-z_][A-Za-z0-9_]*)")
# Steps that do not run under bash: skip the syntax check for them.
NON_BASH = {"pwsh", "powershell", "cmd", "python", "node"}


def steps_of(doc):
    for jname, job in (doc.get("jobs") or {}).items():
        if not isinstance(job, dict):
            continue
        runs_on = str(job.get("runs-on", ""))
        jshell = ((job.get("defaults") or {}).get("run") or {}).get("shell", "")
        for step in job.get("steps") or []:
            if isinstance(step, dict) and "run" in step:
                yield jname, step.get("name", "<unnamed>"), step["run"], \
                    step.get("shell", jshell), runs_on


def main() -> int:
    problems = 0
    checked = 0
    for wf in sorted(WORKFLOWS.glob("*.y*ml")):
        try:
            doc = yaml.safe_load(wf.read_text())
        except yaml.YAMLError as e:
            print(f"{wf}: YAML does not parse: {e}")
            problems += 1
            continue
        if not isinstance(doc, dict):
            continue

        for jname, sname, body, shell, runs_on in steps_of(doc):
            where = f"{wf.name} / {jname} / {sname}"
            # A custom shell is a command template ("python3 {0}", "bash -e {0}"):
            # judge it by its interpreter, so python3/python3.12 count as python.
            interp = os.path.basename(shell.split()[0]) if shell.strip() else ""
            interp = "python" if interp.startswith("python") else interp
            if interp in ("python", "node"):
                continue  # not shell at all: none of the checks below apply

            # (1) a literal backslash-n is never intentional in SHELL, but it is
            # entirely normal inside an embedded heredoc (a python/node script
            # written inline), so track heredoc regions and skip them.
            heredoc_end = None
            for i, line in enumerate(body.splitlines(), 1):
                if heredoc_end is not None:
                    if line.strip() == heredoc_end:
                        heredoc_end = None
                    continue
                m = HEREDOC.search(line)
                if m:
                    heredoc_end = m.group("tag").strip("'\"")
                    continue
                if "\\n" not in line or line.lstrip().startswith("#"):
                    continue
                # printf '...\n', echo -e, sed/awk scripts: all legitimate
                if any(t in line for t in ("printf", "echo -e", "sed", "awk", "tr ", "perl")):
                    continue
                print(f"{where}: line {i}: literal '\\n' in shell — "
                      f"did you mean a line continuation?\n    {line.strip()[:110]}")
                problems += 1

            # (2) PowerShell here-string delimiters must start their line.
            #
            # `@'` and `'@` are only recognised at column 0; indent either and
            # PowerShell fails to parse the whole script. That is easy to get
            # wrong here because the indentation you see in the YAML is not what
            # PowerShell sees — the block scalar strips the common prefix first,
            # so a delimiter looks indented in the file and is fine, or lines up
            # with its neighbours and is not. The same #339 argument applies:
            # release.yml runs only on a tag push, so its first execution is the
            # release, and a parse error there costs the assets that job builds.
            # Nothing else covers this — check (3) skips every non-bash shell.
            if shell in ("pwsh", "powershell"):
                depth = 0
                for i, line in enumerate(body.splitlines(), 1):
                    s = line.strip()
                    # A closing delimiter may be followed by a pipeline on the
                    # same line (`'@ | Set-Content x`); an opening one may not.
                    opens = s in ("@'", '@"')
                    closes = s.startswith("'@") or s.startswith('"@')
                    if not (opens or closes):
                        continue
                    if line[:1] not in ("@", "'", '"'):
                        print(f"{where}: line {i}: PowerShell here-string delimiter is "
                              f"indented; it must start the line or the script will not parse"
                              f"\n    {s[:110]}")
                        problems += 1
                    depth += opens - closes
                if depth != 0:
                    print(f"{where}: unbalanced PowerShell here-string (depth {depth})")
                    problems += 1

            # (3) the shell must at least parse
            if interp in NON_BASH or (not shell and "windows" in runs_on):
                continue
            checked += 1
            r = subprocess.run(["bash", "-n"], input=body, capture_output=True, text=True)
            if r.returncode != 0:
                print(f"{where}: bash syntax error:\n    {r.stderr.strip().splitlines()[0][:110]}")
                problems += 1

    if problems:
        print(f"\n{problems} problem(s) in workflow run blocks")
        return 1
    print(f"workflow run blocks OK ({checked} bash steps parsed)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
