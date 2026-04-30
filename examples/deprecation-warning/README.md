# Migration notice for binary filenames

> [!IMPORTANT]
[2026 Apr 30] User-facing example binaries in this fork were renamed to the `crispasr-` prefix for consistency. The Whisper backend itself remains available via `--backend whisper`.

This migration was important, but it is a breaking change that may not always be immediately obvious to users.

Please update all scripts and workflows to use the new binary names.

| Old Filename | New Filename |
| ---- | ---- |
| main | crispasr |
| bench | crispasr-bench |
| stream | crispasr-stream |
| command | crispasr-command |
| server | crispasr-server |
| talk-llama | crispasr-talk-llama |
