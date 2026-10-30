# Superseded secret-scan queue repair

Secret scan checks the complete tracked tree on every push, pull request and
manual dispatch. Previously, checkpoint pushes left obsolete scans queued even
when a newer scan already covered the resulting tree.

The workflow now groups runs by workflow name and PR number or ref, with
`cancel-in-progress: true`. A newer run replaces queued or running checks of
the same context. Different branches, tags and PRs remain independent.
All triggers, read-only permissions, the five-minute timeout, credential
patterns, account-name handling and the four existing tests are unchanged.

Local checks pass: all four secret guards unskipped, and the existing workflow
linter parses all 437 bash steps. A parsed-YAML comparison confirms that only
the concurrency metadata changes. Two committed branch checkpoints exercise
actual hosted replacement. [First run 38744186281](https://github.com/CrispStrobe/CrispASR/actions/runs/38744186281)
at 0bb6c9ab was queued; after the second committed push, GitHub automatically
completed it as cancelled and left
[new run 38744331967](https://github.com/CrispStrobe/CrispASR/actions/runs/38744331967)
at 3afaf4ae queued. No manual cancellation was called on the first run.
This proves queue supersession without requiring an available runner. A queued
latest scan is not a hosted test pass.

Fast-forward integration 3afaf4ae787baae3deb37c55781154393db993d9 preserves the
tested workflow bytes. The JSON receipt records parsed workflow equality
apart from concurrency and both official run identities. Main scan38745123181 at51891e04 subsequently passes the full hosted guard.
The local four-test guard also passed unskipped; future tips need their own scan.
