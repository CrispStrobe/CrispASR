# CrispASR / CrispLens / cloud-backup Audit

Date: 2026-05-14

Scope: read the markdown and code in this repo plus `../CrispLens` and
`../cloud-backup`; understand the intended interaction; check local live
endpoints; run available local tests. Later passes added audit tests and the
minimum code/docs wiring needed for storage-root safety and CrispLens video
transcript search.

Update: after the first pass, `../.env` was identified as the place holding
remote URLs and credentials. I used it for additional live checks and kept all
secret values out of this file.

Update 2: the VPS is reachable over SSH as `ssh root@168.119.190.252`. I used
that path for read-only service checks.

## Executive Summary

The architecture makes sense as a three-part system:

- `CrispASR` is the local/native ASR runtime and can run as an
  OpenAI-compatible persistent HTTP transcription server.
- `CrispLens` is the media/face/video indexing application. It has partial
  video ingest support: representative frame extraction, face recognition on
  frames, optional transcript storage, and VLM enrichment using the transcript.
  It currently uses `faster_whisper`, OpenAI Whisper, or Gladia for video
  transcription; CrispASR would be a sensible additional provider, not a
  required existing dependency. Its stored video transcripts are now included
  in browse/path search and semantic text embeddings.
- `cloud-backup` is the archive, manifest, blob, and VPS index layer. It can
  move TB-scale originals to cloud storage through a VPS, and `cb-api` provides
  manifest, full-text, vector, and file-by-hash APIs for index clients.

The basic idea can scale to TB of cloud data with less than 1 TB of VPS index
and less than 50 GB local working indexes, but only if the boundaries are kept
strict:

- Do not store original file bytes in the VPS index layer unless that storage
  is explicitly a Hetzner Storage Box/blob tier, not the VPS index budget.
- Do not push unlimited `full_text` or high-granularity chunk embeddings for
  every TB of source data into SQLite-only indexes.
- Keep local clients as manifest/vector/thumbnail caches, not full-data mirrors.
- Move all large scratch, model caches, and crash-relevant temporary files off
  `/tmp` and off the Mac boot disk.

Right now it is not all correctly wired for production. The main issues are
listed below.

## Verified Codebases

- `CrispASR`: `/Users/christianstrobele/code/CrispASR`
- `CrispLens`: `/Users/christianstrobele/code/CrispLens`
- `cloud-backup`: `/Users/christianstrobele/code/cloud-backup`

Relevant docs read:

- `CrispASR`: `README.md`, `ARCHITECTURE.md`, `docs/architecture.md`,
  `docs/server.md`, `docs/install.md`, `PLAN.md` and related project docs.
- `CrispLens`: `README.md`, `electron-app-v4/README.md`,
  `electron-app-v2/README.md`, `PLAN.md`, `config.example.yaml`.
- `cloud-backup`: `readme.md`, `deploy/README.md`, deployment env/unit files.

## How The Pieces Interact

### Intended Flow

1. `cloud-backup/controller.py` scans a source folder, hashes files, records
   manifest state, creates encrypted 7z batches, and rsyncs them to the VPS.
2. `cloud-backup/vps_worker.py` receives archives, verifies them, uploads the
   encrypted archive to cloud (`/backups/zips`), optionally extracts it, and
   uploads individual files to cloud root for browsability.
3. `cloud-backup/api/app.py` (`cb-api`, default `127.0.0.1:7869`) exposes
   manifest pull/push, text search, vector search, blob upload/download, and
   embedding routes.
4. `CrispLens` indexes images/videos through FastAPI v2 or Node/Electron v4,
   using SQLite for image/person/face state and ONNX/InsightFace/dlib for
   inference.
5. `CrispLens` can already detect video files in `process_image()`, route them
   to `process_video()`, extract still frames, detect/recognize faces on those
   frames, and optionally store a transcript on the parent `images` row.
   CrispASR is not currently one of the transcription providers.

### Current Gap

There is no direct CrispLens -> CrispASR integration found. That is not
necessarily wrong, because your current architecture puts CrispASR
transcription in `vps_worker` on the VPS and locally in
CrispSorter/crisp-index-server. If CrispLens is intended to search videos by
their own transcript, however, CrispASR fits cleanly as an optional additional
provider because CrispASR already exposes an OpenAI-compatible
`/v1/audio/transcriptions` endpoint.

There is also no direct CrispLens -> `cb-api` integration found in the current
CrispLens code. `cb-api` comments refer to CrispSorter clients, not CrispLens.
CrispLens does have direct Internxt/Filen browse/download code and proxy routes.

## Live Checks

Local endpoint checks:

- `http://127.0.0.1:7861/api/health`: connection refused.
- `http://127.0.0.1:7865/api/health`: connection refused.
- `http://127.0.0.1:7869/api/health`: TCP connection accepted but no response
  within 5 seconds.

This means the local apps are not currently live in a healthy, testable state.

Remote checks using `../.env`:

- `CRISPLENS_URL` is stored without a URL scheme. Treating it as HTTPS and HTTP,
  `/api/health` returned HTTP 200 with `ok=true` and `model_ready=true`.
- The remote CrispLens root returned HTTP 200 HTML over HTTPS and HTTP.
- Direct VPS port checks to `:7869`, `:7861`, and `:7865` timed out. That is
  consistent with services being reverse-proxied only, firewalled, down, or not
  exposed on public interfaces.
- The expected `/cb/...` reverse-proxy prefix for `cb-api` returned 404 on the
  CrispLens host. I did not find a working public `cb-api` route from the env
  values.
- Authenticated CrispLens smoke test was not successful: one credential pair
  returned HTTP 401; the other attempt timed out during TLS handshake. No
  credential values were logged.

SSH checks on `root@168.119.190.252`:

- Host reachable: `ubuntu-2gb-fsn1-1`, uptime about 44 days.
- `vps-worker.service`: active/running.
- `cb-api.service`: active/running.
- `internxt-relogin.timer`: active.
- Internal `cb-api` health works:
  `http://127.0.0.1:7869/api/health` returned `ok=true`,
  `lance_enabled=true`, and `fastembed_enabled=true`.
- Internal CrispLens v2 health works:
  `http://127.0.0.1:7865/api/health` returned `ok=true` and
  `model_ready=true`.
- No service is listening on internal port `7861`; the VPS is running
  CrispLens v2/FastAPI on `7865`, not v4/Node on `7861`.
- Public Apache listens on `80` and `443` and proxies `img.akademie-rs.de`
  to `127.0.0.1:7865`. I found no Apache `/cb` proxy to
  `127.0.0.1:7869`.
- Direct public `:7869`, `:7865`, and `:7861` timeouts are expected exposure
  behavior; those services are localhost-bound and operationally accessed over
  SSH.
- Storage: root disk is 75 GB with 17 GB free; the Hetzner Storage Box mount
  configured as `VPS_STORAGE_ROOT` in `../.env` is 5.0 TB with 3.6 TB free.
- Current sizes: `/root/cloudworker_state` about 508 KB,
  `VPS_STORAGE_ROOT` about 1.8 GB, `CB_API_STORAGE_ROOT` about 4 KB, and
  `CB_API_LANCE_ROOT` about 3.8 MB.
- Current queues: `incoming` 0 files, `processing` 0 files,
  `incoming/.manifests` 21 files.
- Internxt CLI auth is valid on the VPS (`whoami` succeeds; email redacted).
- The deployed paths are not git checkouts: `/root/cloud-backup` and
  `/opt/crisp-lens` have no `.git` metadata. `vps-worker.service` runs
  `/root/internxt-python/vps_worker.py`, not `/root/cloud-backup/vps_worker.py`.
- Current catalog size is very small: `master_catalog.db` is about 500 KB with
  66 `files`, 2,116 `file_references`, 87 `batches`, one `api_keys` row, and
  zero `chunk_embeddings`.
- `worker_state.db` contains two jobs, both `PROCESSED` and `cloud_backed_up=1`.
- `/etc/vps-worker.env` now contains `VPS_STORAGE_ROOT` and
  `LOCAL_BACKUP_DIR`; the latter points under the attached Storage Box backup
  tree, not the VPS root volume. The running service stayed active after the
  env-file update.
- `/etc/vps-worker.env` now also contains `VPS_SCRATCH_DIR` on the Storage
  Box. `face-rec.service` has a systemd drop-in setting
  `CRISPLENS_SCRATCH_DIR` on the Storage Box for the next CrispLens service
  load.
- CrispLens is managed by `face-rec.service`, running one Uvicorn worker from
  `/opt/crisp-lens` on `127.0.0.1:7865`.
- Live CrispLens DB is about 1.4 MB with 72 `images`, 29 `faces`, 9 `people`,
  48 `description_embeddings`, one `cloud_drives` row, and 4 `users`.

Tests run:

- `../cloud-backup`: `python -m pytest -q` -> `57 passed in 101.28s`.
- `../CrispLens`: `python -m pytest -q` -> originally `27 passed`; after
  transcript-search tests and wiring, `30 passed in 6.02s`.
- `CrispASR`: attempted a pytest target, but no Python tests were collected
  from the selected files.

Additional audit tests added and run after SSH access was confirmed:

- Added `tests/test_audit_cross_stack.py` in this repo. It covers:
  - sibling repo presence and expected route/config files;
  - `../.env` key presence without printing secret values;
  - `cloud-backup` API route definitions;
  - blob-store same-filesystem partial-write behavior;
  - explicit env configuration for `CB_API_STORAGE_ROOT` / LanceDB instead of
    hardcoded storage-path defaults;
  - enforced invariants for configured scratch roots and deployed worker
    streaming-hash / recursive directory upload behavior;
  - live SSH checks for `vps-worker`, `cb-api`, and relogin timer;
  - live internal health checks for `cb-api` and CrispLens;
  - live storage free-space and catalog-count checks;
  - live Apache proxy shape check;
  - live Internxt CLI auth and worker queue checks.
- Added `pytest.ini` to register the custom `live` marker.
- `CrispASR`: `python -m pytest -q tests/test_audit_cross_stack.py -rsx`
  -> now `14 passed in 14.23s` after restarting `vps-worker.service`.
- `CrispASR`: `python -m pytest -q tests/test_backend_config.py
  tests/regression/test_driver_smoke.py tests/test_audit_cross_stack.py -rsx`
  -> `31 passed, 4 xfailed in 20.18s`.
- `../CrispLens`: `python -m pytest -q` -> `30 passed in 6.86s`.
- `../cloud-backup`: `python -m pytest -q` ->
  `57 passed in 85.58s`.
- `CrispASR`: `cmake --build build --target crispasr-lib`,
  `cmake --build build --target crispasr-cli crispasr-server -j2` -> passed
  with existing compiler warnings only.

## Findings

### Fixed: Crash-Relevant Scratch Paths Are Now Configurable

The stated rule is that `/tmp` is not used because crash recovery must not lose
state. The audited production paths now satisfy that invariant.

Changes made:

- `../CrispLens/scratch_paths.py` centralizes `CRISPLENS_SCRATCH_DIR` /
  `CRISP_SCRATCH_DIR` / `FACE_REC_SCRATCH_DIR` handling.
- CrispLens transcription audio extraction, video frame extraction,
  upload-byte processing, cloud-drive downloads, and SMB/SFTP test mounts now
  use the configured scratch root.
- `../cloud-backup/retrieve.py` now derives remote extraction scratch from
  `VPS_SCRATCH_DIR` or `VPS_STORAGE_ROOT`.
- `../cloud-backup/controller.py` now stages generated remote scripts and
  service files through configured scratch roots.
- CrispASR cache and CLI/server scratch files now use `CRISPASR_SCRATCH_DIR`
  or cache-local scratch, not system temp.

The live VPS confirmed this was not just theoretical before the fix: deployed
CrispLens logs showed images processed from system-temp upload paths, followed
by repeated “file not found on disk” errors for those same paths. That is the
failure mode the configured-scratch rule is meant to prevent after the patched
code is deployed/restarted.

The local `../.env` now contains `CRISPLENS_SCRATCH_DIR`,
`CLOUD_BACKUP_SCRATCH_DIR`, `CRISPASR_SCRATCH_DIR`, and `VPS_SCRATCH_DIR`.

### Critical: `/Volumes/backups` Is Effectively Full

`df -h /Volumes/backups` shows a 1.9 TiB volume with about 13 GiB available and
capacity reported as 100%. Current large observed directories include:

- `/Volumes/backups/ai/crispasr`: 166 GB
- `/Volumes/backups/ai/huggingface-hub`: 280 GB
- `/Volumes/backups/texte`: 39 GB

Using the attached SSD for large files is correct, but the current free-space
margin is too small for 2 GB archives, model conversion, temporary extraction,
or crash-safe partial writes. A crash-safe setup needs enough headroom for at
least the largest in-flight archive, extracted working set, and atomic partials.

### High: `cb-api` Storage Env Is Now Explicit, With SQLite Shards Disabled On CIFS

`../cloud-backup/api/db.py` supports:

- `CB_API_STORAGE_ROOT`
- `CB_API_SHARD_ROOT`
- `CB_API_LANCE_ROOT`

`../cloud-backup/api/embed.py` also expects production to set
`XDG_CACHE_HOME` to keep fastembed model downloads off the root disk.

The live VPS `/etc/cb-api.env` now explicitly sets:

- `CB_API_STORAGE_ROOT=/mnt/akademie_storage/cb_api_blobs`
- `CB_API_LANCE_ROOT=/mnt/akademie_storage/cb_api_lance`
- `XDG_CACHE_HOME=/mnt/akademie_storage/.fastembed_cache`

`CB_API_SHARD_ROOT` is intentionally unset. The attached Hetzner Storage Box is
CIFS-mounted, and direct SQLite WAL testing on the VPS showed `database is
locked` on `/mnt/akademie_storage` while the same test works on local ext4.
Enabling SQLite shards on CIFS would be unsafe and would also route the API
away from the current single `master_catalog.db` without a migration.

I removed the code fallback defaults for these large storage roots during this
audit pass. Missing `CB_API_STORAGE_ROOT` should now fail explicitly instead of
silently choosing a deployment-specific path. The env template now documents
the large roots as required and `CB_API_SHARD_ROOT` as optional only for
SQLite-safe ext4/XFS block storage.

### High: `cb-api` Is Healthy Over SSH, But Not Publicly Exposed

The local `7869` health check hung for 5 seconds. Since `/api/health` is meant
to be unauthenticated and cheap, this should be treated as a wiring or runtime
health issue until explained.

Possible causes include import-time optional dependency stalls, DB open/migrate
against a problematic path, or an already-running but stuck process. The test
suite passes, so this looks more like local service state/config than route
logic.

Using SSH, `cb-api` is healthy on the VPS at `127.0.0.1:7869` and reports both
LanceDB and fastembed enabled. Direct public `VPS_IP:7869` times out because it
is localhost-bound/firewalled. Apache does not expose a `/cb/` reverse proxy.
So the setup is valid for SSH-admin access, but not for HTTP clients unless
they use an SSH tunnel or a proxy is added.

### High: Standard-Mode VPS Upload Has Scaling Failure Modes

The normal path extracts an archive and uploads `extract_dir/*` recursively via
the cloud CLI. For TB-scale and high file counts:

- Shell expansion of `extract_dir/*` can hit argument-list limits if many files
  are directly under the extraction root.
- `vps_worker.py` reads an entire archive into memory during checksum
  verification (`hashlib.sha256(f.read())`). With 2 GB batches this is already
  uncomfortable; with larger batches it risks OOM on a small VPS.
- `CMD_TIMEOUT = 600` seconds can be too short for large cloud uploads,
  extraction, or provider throttling.
- One worker loop means throughput depends heavily on one VPS process and one
  provider CLI session.

The design can scale, but the implementation should stream hashes, avoid shell
globs for extracted content, and support resumable/queued per-file uploads.

The deployed worker has the same risk points: it uses `hashlib.sha256(f.read())`
for archive verification and uploads extracted contents via
`upload {extract_dir}/* -t / -r`.

### High: Index Budget Depends On Strict Data Policy

The requested budgets are plausible only if the index layer stores metadata and
derived vectors, not the source bytes.

`cb-api` includes `/api/files/by-hash`, which stores content-addressed blobs
under `CB_API_STORAGE_ROOT`. That is a file cache/blob tier, not an index. If it
is enabled for all source files and placed on the VPS, the VPS will exceed the
`<1 TB index` constraint as source data grows.

Full-text bodies and embeddings also need caps:

- A 1024-d float32 embedding is roughly 4 KB per chunk before DB/index overhead.
- 10 million chunks is roughly 40 GB raw vectors; 100 million chunks is roughly
  400 GB raw vectors before overhead.
- Storing `full_text` copies for many TB of documents can become hundreds of GB
  or more by itself.

Recommendation: treat `full_text` and chunk embeddings as opt-in or capped by
collection, chunk budget, and retention tier. Store summaries/metadata locally;
keep complete text in cloud or Storage Box where possible.

The live VPS has not yet exercised this at scale: the catalog currently has
only 66 file rows, 2,116 references, and zero chunk embeddings.

The live CrispLens DB is also small: 72 images and 48 description embeddings.
So neither the VPS catalog nor the media index has yet validated the target
TB-scale behavior.

### Medium: CrispLens Has Partial Video Search Support, But Not CrispASR ASR

CrispLens does already have video support in code:

- `video_processing.py` extracts representative JPEG frames using ffmpeg
  scene-change detection plus evenly spaced fallback sampling, then pHash
  dedupes near-identical frames.
- `face_recognition_core.process_image()` routes video extensions to
  `process_video()`.
- `process_video()` inserts one parent `images` row with `media_type='video'`,
  duration/fps/frame count, one `video_frames` row per extracted frame, and
  links detected faces back to frame timestamps via `faces.frame_id`.
- `schema_complete.sql` includes `images.transcript` and the `video_frames`
  table.
- If `transcription.enabled` is true, `process_video()` writes transcript text
  to `images.transcript`.
- If VLM enrichment runs, the transcript is appended to the midpoint-frame VLM
  prompt, so video descriptions/tags can indirectly include audio context.

Current transcription providers are:

- `faster_whisper`
- `openai_whisper`
- `gladia`

There is no CrispASR provider despite CrispASR already exposing an
OpenAI-compatible transcription endpoint. That is not inherently a bug if
CrispASR transcription remains owned by `vps_worker` and
CrispSorter/crisp-index-server. It is a reasonable future enhancement for
CrispLens if you want video transcript search inside the image/video gallery.

The video support is also not fully surfaced:

- `PLAN.md` does not mention video or transcript search; the roadmap is
  about albums, shortcuts, face clusters, events, context menus, etc.
- `README.md` does not document video ingest/transcription.
- `image_ops.get_image_record()` and `browse_images_filtered()` now include
  `media_type`, `duration_sec`, `fps`, `frame_count`, and `transcript` in their
  returned JSON dictionaries.
- `browse_images_filtered()` path/text filtering now searches
  `images.transcript` in addition to filepath, `local_path`, `ai_description`,
  and `ai_tags`.
- `text_embeddings.backfill()` now embeds the combined
  description/transcript text, so transcript-only video rows participate in
  `/api/search/semantic`.
- The live VPS DB has zero `video_frames`, so this path is present in code but
  not exercised in production data.

On the live VPS, CrispLens is running as FastAPI/v2 on localhost port `7865`.
There is no Node/v4 listener on `7861`.

### Medium: CrispLens Local/Standalone Indexes Can Stay Under 50 GB Only With Limits

CrispLens v4 standalone stores SQLite/WASM data, face embeddings, thumbnails,
and local cache in browser storage. The docs expose max items and max MB
settings, which is good. But TB-scale media libraries cannot be fully mirrored
locally in browser IndexedDB or a Mac working DB if thumbnails/original blobs are
not aggressively capped.

Recommended policy:

- local Mac working set: metadata, hashes, face embeddings, low-res thumbnails,
  recent/selected items only;
- original media: cloud/Storage Box only;
- local indexes: collection-scoped and evictable.

### Medium: Path Defaults Are Inconsistent Across `cloud-backup`

Docs and deploy files should treat the storage mount as configuration, not as
a path literal. During this audit pass I removed the remaining hardcoded
`/mnt/...` runtime defaults from `cloud-backup`; `/tmp/cloud_outgoing` remains
in retrieval/prune paths and still needs scratch-dir configuration.

Examples:

- `controller.py` now derives `vps_incoming` and `vps_logs` from
  `VPS_STORAGE_ROOT` or explicit `VPS_INCOMING` / `VPS_LOGS`.
- `../.env` now carries `VPS_STORAGE_ROOT` and the cb-api storage/index roots
  for local and live tests.
- `retrieve.py` checks `/tmp/cloud_outgoing`.

This is easy to compensate for with env vars, but risky for unattended deploys.

### Medium: Cloud Provider Abstraction Is Uneven

CrispLens has direct Internxt and Filen client/proxy logic. `cloud-backup` is
documented as Internxt-default and has a generic “cloud provider CLI” concept,
but the worker path is strongly tied to the Internxt Python CLI shape.

Hetzner Storage Box is used as mounted storage in the architecture, not as a
first-class cloud provider target in the same sense as Internxt/Filen.

For “Internxt oder Filen oder Hetzner Storage Box” as interchangeable backends,
the provider boundary needs to be formalized: upload, list, verify, download,
conflict behavior, auth refresh, rate limits, and error normalization.

### Medium: SQLite-Only Search Will Not Be Enough For Large Vector Workloads

`cb-api` has optional LanceDB support and sharding, which is the right direction.
But unless `CB_API_LANCE_ROOT` is explicitly set and the optional dependencies
are installed, the system falls back to SQLite-only storage and fan-out queries.

SQLite with WAL is fine for manifests and modest full-text, but it should not be
the only plan for TB-scale vector search.

The live VPS has LanceDB enabled, so vector search is not SQLite-only there.
SQLite sharding is intentionally unset because the available attached storage
is CIFS-mounted. Current data volume is too small to validate fan-out behavior.

### Low: Local Databases Are Not Representative

`../cloud-backup/index_manifest.db` is SQLCipher-encrypted or otherwise not a
plain SQLite DB in this environment (`sqlite3` reports “file is not a
database”). That is expected per docs when using `sqlcipher3`, but it means I
could not inspect local manifest contents with stock sqlite.

`../CrispLens/face_recognition.db` is currently an empty 0-byte file in this
checkout, so it does not prove real production data shape or size.

## Scalability Assessment

### TB In Cloud

Feasible for archive storage. The controller/VPS/cloud pipeline has the right
macro shape: incremental scan, encrypted batches, manifests, cloud upload
verification, and optional per-file browsability.

Not yet robust enough for unattended TB-scale operation without the fixes above:
stream hashing, no shell globs, provider abstraction, larger/resumable timeouts,
and scratch-space enforcement.

### Less Than 1 TB VPS Index

Feasible if “index” means manifest rows, metadata, FTS snippets, and bounded
vectors. Not feasible if `/api/files/by-hash` stores all originals on the VPS
under the index budget.

For a Hetzner VPS, put only DB/index files on the VPS root or attached index
volume. Put blobs, LanceDB shards, fastembed caches, and worker scratch on
Storage Box or another large attached mount.

### Less Than 50 GB Local Mac Working Index

Feasible if local stores are cache-like:

- no full originals;
- capped thumbnails;
- collection-scoped vector indexes;
- LRU eviction;
- model caches on `/Volumes/backups/ai`, not boot disk.

Not feasible if CrispLens standalone tries to keep thumbnails/embeddings for an
entire TB-scale corpus in browser IndexedDB without aggressive caps.

## Recommended Next Steps

1. Continue expanding the scratch-root pattern to lower-priority examples and
   one-off tools as they become production-relevant.
2. Decide whether `cb-api` should remain SSH-only or be exposed through a
   controlled reverse proxy. It is healthy on the VPS over localhost, but not
   reachable as public HTTP.
3. Keep the live `/etc/cb-api.env` and local `../.env` aligned on
   `CB_API_STORAGE_ROOT`, `CB_API_LANCE_ROOT`, and `XDG_CACHE_HOME`; keep
   `CB_API_SHARD_ROOT` unset while the target storage is CIFS/SMB.
4. CrispLens now has CrispASR transcription entry points in all relevant
   surfaces: v2/FastAPI can use either CrispASR's OpenAI-compatible HTTP
   endpoint or the local `crispasr` CLI wrapper, and v4/Electron standalone can
   call the CLI wrapper through IPC. v4 local and server image schemas now store
   `media_type`, duration fields, and `transcript`, and local/server search
   include transcript text. Mobile/Capacitor now has native wrappers: iOS wires
   the `CrispASR` plugin name, audio extraction, and the vendored framework
   entry point; Android decodes media with `MediaCodec`, resamples to 16 kHz
   mono, calls CrispASR through JNI, and returns transcript segments.
5. Decide and document the index data policy: which clients may push
   `full_text`, chunk embeddings, and blobs; define per-collection caps.
6. Continue hardening `vps_worker.py` for provider-specific retry/resume and
   large-operation timeouts; the deployed worker no longer uses full-archive
   reads for checksum verification or shell-glob expansion for extracted
   uploads.
7. Free substantial space on `/Volumes/backups` before running more large tests.

## Current Verdict

Conceptually sound and now correctly wired for the current SSH-only deployment
shape: cb-api and CrispLens are healthy over VPS localhost, cb-api uses attached
storage for blobs/LanceDB/cache, `CB_API_SHARD_ROOT` is intentionally unset on
CIFS, configured scratch roots avoid system temp in audited production paths,
and CrispLens can create/search video transcripts through CrispASR in v2 and
v4/Electron standalone. The v4 Capacitor path now has a native iOS plugin
skeleton and JavaScript wrapper; Android still needs a packaged runtime before
offline mobile transcription can work there. The worker/index paths also still
need TB-scale exercise.

## 2026-05-16 Follow-Up

With `/Volumes/backups` absent, large local model/live tests remain postponed.
Safe work completed without writing large local files:

- VPS service envs now force scratch/cache paths away from `/tmp`:
  `cb-api.service` uses `/mnt/storage/cb_api_scratch` and
  `/mnt/storage/hf-cache`; `vps-worker.service` uses
  `/mnt/storage/cloudworker/scratch/...`; `face-rec.service` uses
  `/mnt/storage/crisplens/scratch/...`.
- Config backups were written to
  `/mnt/storage/config-backups/20260516T103345Z`.
- Existing large VPS `/tmp` artifacts were moved to storage, not left on the
  root volume. Symlink-heavy trees were stored as verified tar archives because
  the CIFS Storage Box mount does not reliably recreate Unix symlinks as
  directories. Relevant migration locations:
  `/mnt/storage/tmp-migrated/20260516T103212Z`,
  `/mnt/storage/tmp-migrated/20260516T103441Z`, and
  `/mnt/storage/tmp-migrated/20260516T111528Z`.
- `/tmp` on the VPS was reduced to small residual files only, with the large
  regenerated `hf-cache`, `jina-nano-regen`, and build trees archived under
  `/mnt/storage/tmp-migrated/20260516T111528Z`.
- CrispLens v4 cloud-drive downloads no longer use `os.tmpdir()`; they use
  `CRISPLENS_SCRATCH_DIR`, `CRISP_SCRATCH_DIR`, or app-local scratch.

Verification after this follow-up:

- `python -m pytest -q tests/test_audit_cross_stack.py` passed
  (`16 passed`).
- `node -c server/routes/cloud-drives.js` passed.
- `CRISPLENS_LIVE_TEST_ROOT=/Users/christianstrobele/code/.scratch/crisplens-live-tests node tests/live_transcript_import.js`
  passed using a small synthetic transcript DB.
- VPS `cb-api.service`, `vps-worker.service`, `face-rec.service`, and
  `internxt-relogin.timer` were active; `127.0.0.1:7869/api/health` and
  `127.0.0.1:7865/api/health` returned healthy JSON.

After `/Volumes/backups` was mounted again, a real local live test was run
without downloading new models:

- `/Volumes/backups` was writable but nearly full (`~1.3 GiB` free), so only
  existing models/caches were used.
- CrispASR transcribed `samples/jfk.mp3` with existing
  `/Volumes/backups/ai/crispasr/moonshine-tiny-q4_k.gguf` and
  `/Volumes/backups/ai/crispasr/tokenizer.bin`.
- The JSON transcript was written to
  `/Volumes/backups/crispasr-scratch/live-transcript/jfk-moonshine-20260516T132159.json`.
- CrispLens imported that JSON into
  `/Volumes/backups/crisplens-live-tests/transcript-import-1778930531476.db`
  and transcript search for `ask not what your country` returned the imported
  media row.
- Follow-up verification passed:
  `python -m pytest -q tests/test_audit_cross_stack.py` (`16 passed`),
  `npm run build` in `../CrispLens/electron-app-v4/renderer`,
  `npm audit --audit-level=moderate` in both v4 package roots, and
  `CRISPASR_SCRATCH_DIR=/Volumes/backups/crispasr-scratch build/bin/test-crispasr-cache`.
- Vite warning cleanup was completed in the v4 renderer: an explicit
  `svelte.config.js` was added, ineffective mixed static/dynamic imports were
  made explicit static imports, and the `jeep-sqlite` Node `crypto` fallback was
  routed to a browser `crypto.getRandomValues` shim. `npm run build` now
  completes without Vite warnings.

After `/Volumes/backups` was cleaned up (`~145 GiB` free), the postponed live
checks were expanded:

- CrispASR transcribed `samples/jfk.mp3` with existing local models:
  moonshine, fastconformer-ctc, and data2vec.
- A real video test used
  `/Users/christianstrobele/Downloads/PURplus-Was_darf_ein_Detektiv.mp4`.
  A 70 second audio crop was written under
  `/Volumes/backups/crispasr-scratch/live-video/`, transcribed with existing
  `/Volumes/backups/ai/crispasr/canary-1b-v2-q4_k.gguf`, imported into
  CrispLens v4, and found by transcript search for `Detektivarbeit`.
- cloud-backup live API checks ran via SSH against VPS localhost `cb-api`.
  The master catalog was backed up first under
  `/mnt/storage/cloudworker/backups/master_catalog/live-tests/`. Manifest
  push/pull, FTS search, embedding push/query, and v2 search succeeded.
- Production renderer preview smoke served `/`, `manifest.webmanifest`,
  `sw.js`, `assets/index.js`, and `assets/jeep-sqlite.entry.js` from the built
  v4 app. The bundled SQLite component includes the browser crypto shim path.
- cloud-backup deploy env examples and the shared local `.env` were reconciled
  with the live scratch/cache variables.
- Remaining change classification was completed:
  - CrispASR source/docs/tests: `AUDIT.md`, `PLAN.md`, `HISTORY.md`,
    scratch-root hardening in ASR test scripts/tools, `pytest.ini`, and
    `tests/test_audit_cross_stack.py`.
  - CrispLens source/docs/tests: `PLAN.md`, `HISTORY.md`,
    `electron-app-v4/renderer/vite.config.js`,
    `electron-app-v4/renderer/src/lib/nodeCryptoBrowserShim.js`, and
    `electron-app-v4/tests/live_transcript_import.js`.
  - CrispLens generated artifacts: tracked renderer bundles under
    `electron-app-v4/renderer/dist/assets/`.
  - cloud-backup source/docs/tests: scratch/cache hardening in API/controller
    paths, deploy env examples, local/live tests, `readme.md`, `PLAN.md`, and
    `HISTORY.md`.
  - Unrelated or pre-existing local artifacts still need owner review before
    any commit: CrispASR `.ccache/`, `docs/prompts/`,
    `tools/benchmark_asr_engines.results.json`, and
    `tools/kaggle-issue81-cuda-ab.py`.
- iOS Capacitor smoke was completed locally. CocoaPods was run under the
  existing Ruby 3.1.3 toolchain, `npx cap sync ios` detected the CrispASR
  plugin, and `xcodebuild` built the iOS simulator app with derived data under
  `/Volumes/backups/crisplens-xcode-deriveddata`. The built app installed and
  launched on the iPhone 17 simulator as `com.crisplens.app`.
- Android Capacitor packaging was completed for the v4 app. `@capacitor/android`
  was added, the generated Android project was created, and the CrispASR
  plugin now builds a native JNI bridge against the sibling CrispASR checkout.
  The full debug APK builds with JDK 21 and Gradle cache on `/Volumes/backups`;
  the verified APK was copied to `/Volumes/backups/crisplens-android-builds/`
  before generated build outputs were cleaned from the repo volume.
- VPS audit tooling was improved by installing `ripgrep 14.1.0` via apt
  (`~5.3 MB` installed size).
- The TB-scale confidence test remains intentionally blocked until a real-world
  dataset is specified.
