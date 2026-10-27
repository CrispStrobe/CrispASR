# Repository Instructions

- When asked to check a GitHub issue, read the whole issue before acting: the description, all comments, linked comments, and recent owner/reporter follow-ups. Treat issue comments as part of the requirements, not optional context.

## Read first, if you have it

Fresh agents should use `docs/maintenance-handoff.md` for the current completed
work, evidence and bounded next actions, then claim a task in `PLAN.md`.
`PLAN-ARCHIVE.md` and dated handoffs are historical snapshots, not active queues.

Local operational notes, when present, are in gitignored `CLAUDE.md`. Read them
for storage/worktree/job recovery; keep actual machine paths and private account
details there. Public plans and receipts use portable aliases and public run IDs.

Maintainer checkouts keep a development guide **beside** this repository, at
`../crispasr-crispembed-dev.md` — a sibling of the repo root, not a file inside
it, and deliberately not tracked here (it spans CrispASR *and* CrispEmbed).

**If that file exists, read it in full before writing any code.** It is the
method, and it is not summarised anywhere: the hard rules, the
convert → quantize → dump-reference → diff → parity port pipeline, the ggml and
GPU-portability gotchas, the A/B discipline, and the storage layout. Where it
and this file disagree, it wins.

If it is not there you are on an ordinary clone — nothing is missing from the
build or the tests. Use the map below, `README.md` and `docs/`.

## Where things live

Read this before opening any of them: together they are >2 MB, and only the
development guide above is meant to be read end to end.

| Need | Read |
|---|---|
| A specific past lesson ("has this bitten us?") | `docs/LEARNINGS-INDEX.md` first — 272 lessons by topic and by model — then grep the heading in `LEARNINGS.md` |
| Is this backend fast / which quant ships | `PERFORMANCE.md` |
| What is in flight right now | `PLAN.md` — and claim your task there before starting |
| What already shipped | `HISTORY.md` — archive; consult to confirm a claim, never as a plan |
| Adding a backend | `docs/contributing.md` (12-point checklist) |
| Build, test, lint commands | `README.md`, and the development guide above |

Two habits that repeatedly cost time here:

- **`PLAN.md` and `HISTORY.md` prose goes stale.** Items marked OPEN are often
  already shipped. Audit against the CODE, never the note.
- **Auto-detection working in the CLI proves nothing about the bindings.** The
  CLI has a filename pass that short-circuits the GGUF-architecture table the C
  ABI depends on; test through the C ABI. (Issue #335.)

## Regenerated files — do not hand-edit

- `docs/LEARNINGS-INDEX.md` → `python tools/gen-learnings-index.py` (CI gates it
  with `--check`; adding a `##` section to `LEARNINGS.md` shifts every line
  number it cites).
- `bindings/go/whisper.go` cgo `LDFLAGS` → `python tools/sync_go_cgo_ldflags.py`.

- Ruby Session/Stream/Chat context guard declarations and registrations →
  `python tools/sync_ruby_context_guards.py` (`--check` runs in the Ruby ABI probe).
