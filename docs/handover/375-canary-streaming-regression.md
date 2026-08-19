# Handover — #375 canary streaming regression

**State: NOT reproduced. Do not ship a fix until it is.** PR #376 is open and on
hold; my review of it is on the PR, including a correction of my own first pass.

## The report

`cdoepmann`, 2026-08-19. Upgraded CrispASR, canary recognition degraded:
"sentences are interrupted and words/phrases are repeated a few times before the
recognition continues". Present on **all quantizations**, and on **CUDA and CPU
alike**. Bisected to `282e5d0b` (good) .. `c08f7a52` (bad) — 41 commits, and they
could not narrow further because some commits in the range do not compile.

## What has been established (measured, not argued)

Built both endpoints, ran `canary-1b-v2-q4_k` on CPU:

| audio | `282e5d0b` vs `main` |
|---|---|
| `jfk_x12.wav` 132 s, 16 seams | byte-identical, md5 `7cce22b79c7cae62f05591caa512fe6c` |
| `de/fleurs_600s.wav` 594 s, ~74 seams | byte-identical, md5 `58346dcdf82aa6c1eda898cc8f4f75b7` |

Command: `crispasr --backend canary -m <gguf> -ng -t 8 [-l de] -f <wav> -nt`
Model: `/Volumes/backups/ai/crispasr-gguf/canary-dl/canary-1b-v2-q4_k.gguf`
Good worktree: `.claude/worktrees/bisect-375-good` (already built).

**Ruled out:**
* **ggml** — submodule pointer is `0714117daca2471b00e09554c7eaa74a06b0b2c5` at
  BOTH endpoints. Nothing in that family can be involved. (Verified with
  `git ls-tree <rev> ggml`, not with a log filter.)
* `73bb9b2f` (the encoder-graph UAF fix, and the only commit in the window
  touching canary sources) changes nothing on either probe.
* `core/generation_health.h` (`d9845cbe`) — test-only, never referenced from src/.

**NOT ruled out, and never examined:** the window contains six commits touching
the audio input path — `f3d82d30` (routes CLI + stereo decode through glint),
`6ad72199`, `bf249d09`, `6b087cdf`, `4053ca11`, `3739e70d` (clamps/bounds-checks
file parsers). A changed decode path shifts sample counts and therefore chunk
boundaries. Nobody has tested these. **If the reporter's audio is a compressed
format (mp3/m4a/opus/webm) rather than WAV, start here, not in canary.cpp.**

## The trap that cost the first two attempts

Both PR #376's diagnosis and my first review were built on a file-path-scoped
search ("the only canary change in the window"), which by construction cannot see
cross-cutting changes. `73bb9b2f` was **selected, not bisected**.

Then I "reproduced" the symptom on `main` and reasoned from it — without checking
the artifacts are absent on the good commit. They are not. Anything that looks
like the report on `main` must be diffed against `282e5d0b` on the SAME file
before it is treated as the regression. That check is two runs and it invalidated
a whole review.

## Pre-existing defect found on the way (real, but a different issue)

Canary streams everything in 8 s chunks / 2 s overlap and merges seams by
token-id LCS. When the AED rewords the overlap the LCS misses and text is
duplicated. On `jfk_x12.wav`, on BOTH endpoints:

* `"ask not Ask not what your country…"` — duplication (capitalisation rewording)
* `"ask what you can do. for your country."` — sentence broken at a seam
* `"for yourself. your country."` — misrecognition at a seam

`CRISPASR_CANARY_SEAM_DEDUP=1` (the #365 fuzzy matcher, already in the tree,
gated) removes the duplication cleanly. It is gated because on a 600 s clip it
also dropped a leading "Many" that could not be confirmed as duplicate. **Whether
to default it on is a corpus question** — run both settings over the regression
audio and count insertions vs deletions. Do not decide it from one clip. This
deserves its own issue and should not be folded into #375.

## What to do next, in order

1. **Get a reproduction.** Asked on #375: a failing audio sample, the exact GGUF
   + quant, the exact command, and ideally both transcripts (good build vs bad).
   Without at least the audio, nothing here can be validated.
2. When audio arrives, first run
   `CRISPASR_CANARY_STREAM_THRESHOLD_S=99999` (forces single-pass, no chunking, no
   seam merge). If the symptom vanishes it is the seam merge; if it persists the
   whole seam-dedup line of attack — including PR #376 — is the wrong tree.
3. If it is the seam merge, compare `CRISPASR_CANARY_SEAM_DEDUP=1` against the
   PR's time-floor heuristic ON THAT AUDIO before choosing.
4. If it is not the seam merge, bisect the window **by building**, starting with
   the six audio-input commits. Note the reporter's warning that parts of the
   range do not compile; `282e5d0b` and `c08f7a52` both build fine with
   `-DGGML_NATIVE=OFF -DCRISPASR_BUILD_TESTS=OFF` after
   `git submodule update --init --recursive ggml`.

## Why PR #376 is on hold

It adds a timestamp-based prefix trim to the streamed seam. Three concerns, in
order of weight:

1. **Unvalidated against the regression** — as is any fix right now.
2. It reintroduces exactly what the code above it says was measured as harmful:
   *"Deliberately NOT a plain time floor. Trimming the overlap by timestamp alone
   was measured removing real speech."* Silent deletion is a worse failure than
   duplication, because users can see duplication.
3. On the one end-to-end file it was tested against it turns `ask not Ask not`
   into `ask not not` — it removes one token of a two-token duplicate. And
   `leading_covered_multi`'s `skip >= 2 && skip < n` returns 0 for a fully covered
   chunk, i.e. it declines the clearest duplicate case while acting on ambiguous
   ones.

HARD RULE 3 applies: the decoded-output roundtrip is the only acceptance test,
and unit tests of the helper are not it.
