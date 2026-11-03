#!/usr/bin/env python3
"""Which kinds of files did this push / pull request change?

The `changes` job of a workflow runs this and gates its expensive jobs
(macOS, Windows, iOS, Android, sanitizer and GPU-toolchain builds) on the
answer, so a Python-only or docs-only change does not queue a dozen runners
on the free-tier pool every CrispStrobe repository shares.

Filters come from the FILTERS environment variable, written as a small YAML
subset so they read naturally inside the workflow file:

    FILTERS: |
      native:
        - 'src/**'
        - '**.cpp'
        - '!docs/**'

Patterns use GitHub's `on.<event>.paths` glob syntax, and like GitHub the LAST
pattern that matches a file decides (`!` excludes). A filter is true when any
changed file ends up included. Prints one `<name>=true|false` line per filter
for $GITHUB_OUTPUT.

Fail-open, always: anything that is not an ordinary push or pull_request
(workflow_dispatch, schedule, tags, workflow_call), a missing or all-zero base
(new branch), a base that is not in the history (force push) or any error
answers TRUE for every filter. This gate may skip only work it can prove is
unrelated; it never hides work.

Environment:
    EVENT    github.event_name
    BASE     PR base SHA, or `github.event.before` for a push
    HEAD     defaults to HEAD
    REF      github.ref (tag pushes always run everything)
    FILTERS  as above

Local use (the before/after table in docs/contributing.md was made this way):
    git diff --name-only A B | FILTERS="$(cat f.yml)" ci_changes.py --files -
"""
import os
import re
import subprocess
import sys


def glob_to_regex(pattern: str) -> re.Pattern:
    """GitHub path-filter glob -> regex: `**` any depth, `*` within a segment, `?` one char."""
    out, i = [], 0
    while i < len(pattern):
        c = pattern[i]
        if pattern.startswith('**/', i):
            out.append('(?:.*/)?'); i += 3
        elif pattern.startswith('**', i):
            out.append('.*'); i += 2
        elif c == '*':
            out.append('[^/]*'); i += 1
        elif c == '?':
            out.append('[^/]'); i += 1
        else:
            out.append(re.escape(c)); i += 1
    return re.compile('^' + ''.join(out) + '$')


def parse_filters(text: str) -> dict:
    filters, current = {}, None
    for raw in text.splitlines():
        line = raw.split(' #', 1)[0].rstrip()
        if not line.strip() or line.strip().startswith('#'):
            continue
        s = line.strip()
        if s.startswith('- '):
            if current is None:
                raise ValueError(f'pattern before any filter name: {raw!r}')
            filters[current].append(s[2:].strip().strip('\'"'))
        elif s.endswith(':'):
            current = s[:-1].strip()
            filters.setdefault(current, [])
        else:
            raise ValueError(f'cannot parse filter line: {raw!r}')
    return filters


def matches(path: str, patterns: list) -> bool:
    hit = False
    for p in patterns:
        neg = p.startswith('!')
        if glob_to_regex(p[1:] if neg else p).match(path):
            hit = not neg
    return hit


def changed_files():
    """Changed paths, or None when the answer must be 'run everything'."""
    event = os.environ.get('EVENT', '')
    ref = os.environ.get('REF', '')
    base = os.environ.get('BASE', '').strip()
    head = os.environ.get('HEAD', '').strip() or 'HEAD'
    if event not in ('push', 'pull_request') or ref.startswith('refs/tags/'):
        print(f'  event={event!r} ref={ref!r}: running everything', file=sys.stderr)
        return None
    if not base or set(base) == {'0'}:
        print('  no base commit: running everything', file=sys.stderr)
        return None
    try:
        subprocess.check_call(['git', 'cat-file', '-e', base + '^{commit}'])
        # PR: three-dot (what the branch changed since it forked), like GitHub's
        # own paths filter. Push: two-dot over every commit in the push.
        if event == 'pull_request':
            rng = [base + '...' + head]
        else:
            if subprocess.call(['git', 'merge-base', '--is-ancestor', base, head]) != 0:
                print('  base is not an ancestor (force push): running everything', file=sys.stderr)
                return None
            rng = [base, head]
        out = subprocess.check_output(['git', 'diff', '--name-only', '--no-renames', *rng], text=True)
    except (subprocess.CalledProcessError, OSError) as e:
        print(f'  cannot diff ({e!r}): running everything', file=sys.stderr)
        return None
    return [f for f in out.splitlines() if f]


def main() -> int:
    filters = parse_filters(os.environ.get('FILTERS', ''))
    if not filters:
        raise ValueError('FILTERS is empty')
    if len(sys.argv) > 1 and sys.argv[1] == '--files':
        src = sys.stdin if sys.argv[2:] in ([], ['-']) else open(sys.argv[2])
        files = [l.strip() for l in src if l.strip()]
    else:
        files = changed_files()
    if files is not None:
        print(f'  {len(files)} changed file(s)', file=sys.stderr)
    for name, patterns in filters.items():
        if files is None:
            print(f'{name}=true')
            continue
        hit = [f for f in files if matches(f, patterns)]
        for f in hit[:10]:
            print(f'  {name}: {f}', file=sys.stderr)
        print(f'{name}={"true" if hit else "false"}')
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except Exception as e:  # a gate that crashes must not skip the gated jobs
        # Gated jobs test `!= 'false'`, so printing nothing also runs them.
        print(f'::warning::ci_changes.py failed ({e!r}); running everything', file=sys.stderr)
        try:
            for name in parse_filters(os.environ.get('FILTERS', '')):
                print(f'{name}=true')
        except Exception:
            pass
        sys.exit(0)
