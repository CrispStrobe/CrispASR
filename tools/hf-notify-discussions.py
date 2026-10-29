#!/usr/bin/env python3
"""Poll HuggingFace Hub for new / recently-updated discussions across
every repo owned by a given user/org.

Why this exists: HF doesn't email maintainers when someone opens a
discussion or community-post on a repo — but those threads are how
external users actually report bugs (the suneetk glm-asr-nano issue
sat for 23 days before we noticed). A nightly poller closes that
gap.

Usage:

    # one-off, prints to stdout
    python tools/hf-notify-discussions.py --user cstr

    # with a state file so we only see new/updated threads since last run
    python tools/hf-notify-discussions.py --user cstr \\
        --state ~/.cache/crispasr/hf-notify-seen.json

    # only the last 7 days (no state file)
    python tools/hf-notify-discussions.py --user cstr --since-days 7

    # emit JSON for programmatic chaining (slack/email/etc)
    python tools/hf-notify-discussions.py --user cstr --json

Auth: anonymous works for public repos (we own no private ones).
Set HF_TOKEN if you ever change that.

Exit code: number of new threads (0 if nothing new). Lets cron-style
runners gate on "is there anything to look at".

Schedule via macOS launchd (sample plist at the bottom of this
docstring), system cron, GitHub Actions, or as a fourth Kaggle
notebook (`kernel_type: script`, weekly schedule, posts results to
the same channel as the regression suite).

Sample launchd plist (drop in ~/Library/LaunchAgents/):

    <?xml version="1.0" encoding="UTF-8"?>
    <!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
                            "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
    <plist version="1.0">
    <dict>
      <key>Label</key><string>com.cstr.hf-notify-discussions</string>
      <key>ProgramArguments</key>
      <array>
        <string>/Users/christianstrobele/miniconda3/bin/python</string>
        <string>/Users/christianstrobele/code/CrispASR/tools/hf-notify-discussions.py</string>
        <string>--user</string><string>cstr</string>
        <string>--state</string>
        <string>/Users/christianstrobele/.cache/crispasr/hf-notify-seen.json</string>
        <string>--out</string>
        <string>/Users/christianstrobele/.cache/crispasr/hf-notify-latest.md</string>
      </array>
      <key>StartCalendarInterval</key>
      <dict>
        <key>Hour</key><integer>9</integer>
        <key>Minute</key><integer>0</integer>
      </dict>
      <key>StandardOutPath</key>
      <string>/Users/christianstrobele/.cache/crispasr/hf-notify.log</string>
      <key>StandardErrorPath</key>
      <string>/Users/christianstrobele/.cache/crispasr/hf-notify.err</string>
    </dict>
    </plist>

Then: `launchctl load ~/Library/LaunchAgents/com.cstr.hf-notify-discussions.plist`.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path


@dataclass
class Thread:
    repo_id: str
    repo_type: str          # model | dataset | space
    num: int                # discussion number on HF
    title: str
    author: str
    created_at: datetime
    last_at: datetime       # last activity (new comment or status change)
    is_pull_request: bool
    status: str             # "open" / "closed"
    # Filled in from get_discussion_details() when --details is on. Without
    # it, HF's discussion listing has no event history, so last_at would just
    # equal created_at — useless for "when did someone last reply?".
    last_comment_at: datetime | None = None
    last_comment_author: str | None = None
    comment_count: int = 0
    has_details: bool = False

    @property
    def url(self) -> str:
        # HF Hub serves both discussions and PRs under the same /discussions/N
        # path; the distinction is in the thread metadata, not the URL slug.
        prefix = "" if self.repo_type == "model" else f"{self.repo_type}s/"
        return f"https://huggingface.co/{prefix}{self.repo_id}/discussions/{self.num}"

    @property
    def key(self) -> str:
        return f"{self.repo_type}:{self.repo_id}:{self.num}"


def collect_repos(api, user: str) -> list[tuple[str, str]]:
    """Return [(repo_id, repo_type), …] for everything owned by `user`."""
    repos: list[tuple[str, str]] = []
    # `list_models`, `list_datasets`, `list_spaces` all accept `author=`.
    for fn, kind in [
        (api.list_models, "model"),
        (api.list_datasets, "dataset"),
        (api.list_spaces, "space"),
    ]:
        try:
            for r in fn(author=user, full=False):
                # The id attribute is `cstr/foo` for owned repos.
                repo_id = getattr(r, "id", None) or getattr(r, "modelId", None)
                if repo_id:
                    repos.append((repo_id, kind))
        except Exception as exc:
            print(f"# warn: failed to list {kind}s for {user}: {exc}",
                  file=sys.stderr)
    return repos


def _to_utc(dt: datetime) -> datetime:
    """Normalise to tz-aware UTC. Some huggingface_hub versions return naive
    datetimes that crash when compared against the script's tz-aware `since`.
    """
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _parse_iso(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return _to_utc(datetime.fromisoformat(value))
    except ValueError:
        return None


def collect_threads(api, repos: list[tuple[str, str]],
                    include_prs: bool = False) -> tuple[list[Thread], set[str]]:
    """Fetch all discussion threads across `repos`. Anonymous OK for public.

    `include_prs=False` passes `discussion_type='discussion'` to the HF API
    so PRs are skipped server-side — saves one fetch per PR-heavy repo and
    keeps the state file free of PR entries when the caller doesn't care
    about them (which would otherwise hide them on a later opt-in).

    Returns (threads, fetched_keys) where fetched_keys are the
    "type:repo_id" keys whose listing actually succeeded — rate-limited or
    broken repos are excluded so the repo cache doesn't mark them as clean.
    """
    threads: list[Thread] = []
    fetched: set[str] = set()
    for repo_id, repo_type in repos:
        key = f"{repo_type}:{repo_id}"
        # Tolerate transient HF rate-limit / network blips: one quick retry
        # before giving up on the repo. The cron rerun will catch anything
        # that still fails.
        attempts = 0
        while True:
            attempts += 1
            try:
                kwargs = {"repo_id": repo_id, "repo_type": repo_type}
                if not include_prs:
                    kwargs["discussion_type"] = "discussion"
                discs = list(api.get_repo_discussions(**kwargs))
                fetched.add(key)
                break
            except TypeError:
                # Older huggingface_hub doesn't accept discussion_type. Retry
                # without the filter — we'll post-filter in main().
                if "discussion_type" in kwargs:
                    del kwargs["discussion_type"]
                    try:
                        discs = list(api.get_repo_discussions(**kwargs))
                        fetched.add(key)
                        break
                    except Exception as exc:
                        print(f"# warn: discussions for {repo_id} ({repo_type}): {exc}",
                              file=sys.stderr)
                        discs = []
                        break
                else:
                    raise
            except Exception as exc:
                if attempts < 2:
                    print(f"# warn: retrying {repo_id} ({repo_type}) after: {exc}",
                          file=sys.stderr)
                    continue
                # Repos with discussions disabled throw; so do rate limits.
                print(f"# warn: discussions for {repo_id} ({repo_type}): {exc}",
                      file=sys.stderr)
                discs = []
                break

        for d in discs:
            # `d` is a DiscussionWithDetails; HF's API exposes:
            #   num, title, status, author, created_at, is_pull_request
            last = d.created_at
            # Some HF clients expose `events` with timestamps; if the
            # discussion has any new events, take the max. Otherwise
            # fall back to created_at.
            events = getattr(d, "events", None)
            if events:
                try:
                    last = max(getattr(ev, "created_at", d.created_at)
                               for ev in events)
                except Exception:
                    pass
            threads.append(Thread(
                repo_id=repo_id,
                repo_type=repo_type,
                num=d.num,
                title=d.title,
                # `author` may be missing (older HF API) or explicitly None
                # (system events). Coerce both to "?" so markdown doesn't
                # render `by \`None\``.
                author=getattr(d, "author", None) or "?",
                created_at=_to_utc(d.created_at),
                last_at=_to_utc(last),
                is_pull_request=getattr(d, "is_pull_request", False),
                status=getattr(d, "status", "open"),
            ))
    return threads, fetched


def enrich_with_details(api, threads: list[Thread]) -> None:
    """Fill in real last-activity / last-comment data per thread.

    `get_repo_discussions()` only returns the thread header (created_at,
    status, author) — no event history. So without this pass `last_at` is
    always `created_at` and the digest cannot answer "when did someone last
    comment on this?". `get_discussion_details()` needs one extra API call
    per thread; failures are tolerated (the header data is still usable).
    """
    for t in threads:
        try:
            d = api.get_discussion_details(
                repo_id=t.repo_id, discussion_num=t.num, repo_type=t.repo_type)
        except Exception as exc:
            print(f"# warn: details for {t.repo_id} #{t.num}: {exc}",
                  file=sys.stderr)
            continue

        t.status = getattr(d, "status", None) or t.status
        t.has_details = True
        events = [e for e in (getattr(d, "events", None) or [])
                  if getattr(e, "created_at", None)]
        if not events:
            continue

        t.last_at = max(_to_utc(e.created_at) for e in events)
        comments = [e for e in events if getattr(e, "type", "") == "comment"]
        t.comment_count = len(comments)
        if comments:
            last = max(comments, key=lambda e: _to_utc(e.created_at))
            t.last_comment_at = _to_utc(last.created_at)
            t.last_comment_author = getattr(last, "author", None) or "?"


def load_repo_cache(path: Path | None) -> dict[str, dict]:
    """{ "model:cstr/foo": {"checked_at": iso, "thread_count": N} }"""
    if not path or not path.exists():
        return {}
    try:
        data = json.loads(path.read_text())
    except Exception:
        return {}
    return data.get("repos", {}) if isinstance(data, dict) else {}


def save_repo_cache(path: Path | None, cache: dict[str, dict]) -> None:
    if not path:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"version": 1, "repos": cache},
                               indent=1, sort_keys=True) + "\n")


def select_repos(repos: list[tuple[str, str]], cache: dict[str, dict],
                 now: datetime, recheck_days: int,
                 max_checks: int) -> tuple[list[tuple[str, str]], int]:
    """Bound the number of per-repo API calls.

    HF allows 1000 api requests / 5 min, and we own ~650 repos — a full sweep
    burns the whole quota and gets 429'd on shared CI IPs. So:

      - repos with known discussions are always polled (they are the inbox),
      - never-seen repos are polled first,
      - discussion-free repos are re-checked once they are `recheck_days` old,
        oldest first, capped at `max_checks` per run (0 = unlimited).

    Skipping a discussion-free repo cannot hide a thread: it only becomes
    visible after one of these checks, and then it is polled every run.
    """
    always: list[tuple[str, str]] = []
    due: list[tuple[float, str, str]] = []
    for repo_id, repo_type in repos:
        key = f"{repo_type}:{repo_id}"
        entry = cache.get(key)
        if entry and entry.get("thread_count"):
            always.append((repo_id, repo_type))
            continue
        checked = _parse_iso(entry.get("checked_at")) if entry else None
        if checked is None:
            due.append((-1e9, repo_id, repo_type))  # never checked → first
        else:
            age_days = (now - checked).total_seconds() / 86400
            if age_days >= recheck_days:
                due.append((-age_days, repo_id, repo_type))
    due.sort(key=lambda x: x[0])
    batch = due if max_checks <= 0 else due[:max_checks]
    return always + [(r, t) for _, r, t in batch], len(due)


def load_state(state_path: Path | None) -> dict[str, str]:
    """{thread.key: iso last_at} of what we've already reported."""
    if not state_path or not state_path.exists():
        return {}
    try:
        return json.loads(state_path.read_text())
    except Exception:
        return {}


def save_state(state_path: Path | None, threads: list[Thread]) -> None:
    if not state_path:
        return
    state_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {t.key: t.last_at.isoformat() for t in threads}
    state_path.write_text(json.dumps(payload, indent=1))


def is_new(t: Thread, state: dict[str, str], since: datetime | None) -> bool:
    """A thread is 'new' if either:
      - it's missing from state, OR
      - state has an older last_at than what HF reports now.
    If `since` is given, also drop threads older than that as a coarse
    filter (useful for first-run without state).
    """
    if since and t.last_at < since:
        return False
    prev = state.get(t.key)
    if prev is None:
        return True
    try:
        prev_dt = datetime.fromisoformat(prev)
    except Exception:
        return True
    return t.last_at > prev_dt


def render_markdown(new_threads: list[Thread]) -> str:
    if not new_threads:
        return "_no new HuggingFace discussions since last check._\n"
    out = [f"# HuggingFace discussions — {len(new_threads)} new / updated\n"]
    new_threads.sort(key=lambda t: t.last_at, reverse=True)
    for t in new_threads:
        emoji = "🟢" if t.status == "open" else "✅"
        kind = "PR" if t.is_pull_request else "discussion"
        if t.last_comment_at:
            engagement = (f"last comment {t.last_comment_at:%Y-%m-%d %H:%M} UTC "
                          f"by `{t.last_comment_author}` "
                          f"({t.comment_count} comment"
                          f"{'' if t.comment_count == 1 else 's'})")
        elif t.has_details:
            engagement = "no comments yet"
        else:
            engagement = "comment history unavailable (API error)"
        out.append(
            f"## {emoji} [{t.repo_id} #{t.num} — {t.title}]({t.url})\n"
            f"  - {kind}, status: **{t.status}**, by `{t.author}`\n"
            f"  - opened {t.created_at:%Y-%m-%d %H:%M} UTC\n"
            f"  - {engagement}\n"
            f"  - last activity {t.last_at:%Y-%m-%d %H:%M} UTC\n"
        )
    return "\n".join(out)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Poll HF Hub for new discussions across cstr/*")
    parser.add_argument("--user", default="cstr",
                        help="HF user/org whose repos to poll (default: cstr)")
    parser.add_argument("--state",
                        help="JSON file storing what we've already reported")
    parser.add_argument("--since-days", type=int,
                        help="Only report threads updated in the last N days "
                             "(applied even when --state is given, as a "
                             "safety bound)")
    parser.add_argument("--out",
                        help="Write the markdown summary here in addition "
                             "to stdout")
    parser.add_argument("--json", action="store_true",
                        help="Emit JSON instead of markdown")
    parser.add_argument("--include-prs", action="store_true",
                        help="Also include pull requests (default: "
                             "discussions only)")
    parser.add_argument("--details", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="Fetch per-thread event history for real "
                             "last-activity / last-comment dates (default: on; "
                             "one extra API call per thread)")
    parser.add_argument("--repo-cache",
                        help="JSON file tracking when each repo was last "
                             "checked for discussions (enables incremental "
                             "polling instead of a full sweep every run)")
    parser.add_argument("--recheck-days", type=int, default=7,
                        help="Re-check discussion-free repos after this many "
                             "days (default: 7)")
    parser.add_argument("--max-checks", type=int, default=120,
                        help="Max discussion-free repos to check per run "
                             "(0 = unlimited; default: 120)")
    parser.add_argument("--full", action="store_true",
                        help="Ignore the repo cache and poll every repo")
    args = parser.parse_args()

    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("ERROR: huggingface_hub not installed. "
              "Run: pip install huggingface_hub", file=sys.stderr)
        return 2

    api = HfApi(token=os.environ.get("HF_TOKEN") or None)

    print(f"# polling HF for discussions on {args.user}/*", file=sys.stderr)
    repo_cache_path = (Path(args.repo_cache).expanduser()
                       if args.repo_cache else None)
    repo_cache = load_repo_cache(repo_cache_path)
    repos = collect_repos(api, args.user)
    # HF pagination can silently return a short list (transient 429/5xx), which
    # would drop whole repos — and their threads — from this run. Anything the
    # cache knows to have discussions is re-added so the inbox never loses one.
    known = {f"{t}:{r}" for r, t in repos}
    missing = [(key.split(":", 1)[1], key.split(":", 1)[0])
               for key, entry in repo_cache.items()
               if entry.get("thread_count") and key not in known]
    if missing:
        print(f"# warn: repo listing returned {len(repos)} but the cache knows "
              f"{len(missing)} more with discussions; re-adding them",
              file=sys.stderr)
        repos.extend(missing)
    print(f"# {len(repos)} repos found", file=sys.stderr)

    now = datetime.now(timezone.utc)
    if args.full or not repo_cache:
        selected, due_count = repos, 0
    else:
        selected, due_count = select_repos(repos, repo_cache, now,
                                           args.recheck_days, args.max_checks)
        print(f"# polling {len(selected)} of {len(repos)} repos "
              f"({due_count} due for a re-check; {args.recheck_days}d cycle)",
              file=sys.stderr)

    threads, fetched = collect_threads(api, selected,
                                       include_prs=args.include_prs)
    if not args.include_prs:
        # Defence-in-depth: older HfApi versions don't honour discussion_type,
        # so post-filter here too. collect_threads already drops the kwarg on
        # TypeError, so this catches the same case without an extra branch.
        threads = [t for t in threads if not t.is_pull_request]
    print(f"# {len(threads)} total discussions across them", file=sys.stderr)

    if repo_cache_path:
        stamp = now.isoformat(timespec="seconds")
        counts: dict[str, int] = {}
        for t in threads:
            k = f"{t.repo_type}:{t.repo_id}"
            counts[k] = counts.get(k, 0) + 1
        for repo_id, repo_type in selected:
            key = f"{repo_type}:{repo_id}"
            # Only record repos we actually got an answer for; a rate-limited
            # repo must stay 'due' instead of being marked discussion-free.
            if key in fetched:
                repo_cache[key] = {"checked_at": stamp,
                                   "thread_count": counts.get(key, 0)}
        save_repo_cache(repo_cache_path, repo_cache)
        print(f"# repo cache updated ({len(fetched)} repos checked)",
              file=sys.stderr)

    if args.details and threads:
        print(f"# fetching event history for {len(threads)} threads...",
              file=sys.stderr)
        enrich_with_details(api, threads)

    state_path = Path(args.state).expanduser() if args.state else None
    state = load_state(state_path)

    since = None
    if args.since_days:
        since = datetime.now(timezone.utc) - timedelta(days=args.since_days)

    new_threads = [t for t in threads if is_new(t, state, since)]

    if args.json:
        payload = [{
            "repo_id": t.repo_id,
            "repo_type": t.repo_type,
            "num": t.num,
            "title": t.title,
            "author": t.author,
            "status": t.status,
            "created_at": t.created_at.isoformat(),
            "last_at": t.last_at.isoformat(),
            "last_comment_at": (t.last_comment_at.isoformat()
                                if t.last_comment_at else None),
            "last_comment_author": t.last_comment_author,
            "comment_count": t.comment_count,
            "has_details": t.has_details,
            "url": t.url,
        } for t in new_threads]
        rendered = json.dumps(payload, indent=2)
    else:
        rendered = render_markdown(new_threads)

    print(rendered)
    if args.out:
        Path(args.out).expanduser().parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).expanduser().write_text(rendered)

    # Only persist what we'd surface under the SAME filter (`--include-prs`)
    # on the next run. Saving PRs here when --include-prs was off would mark
    # them seen forever, hiding them from a future opt-in.
    save_state(state_path, threads)

    # POSIX exit codes are 8-bit unsigned. Clamp so a flood of ≥256 new
    # threads doesn't wrap to a falsy value and trick cron-style runners
    # that gate on "is there anything new" via non-zero exit.
    return min(len(new_threads), 255)


if __name__ == "__main__":
    sys.exit(main())
