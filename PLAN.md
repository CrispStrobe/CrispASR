# CrispASR — Pending work

## Start here

Live work only. Completed threads move to `HISTORY.md`; technical deep-dives to
`LEARNINGS.md`.

**Before you pick something up:** re-read this section on `origin/main`, add a
`## CLAIMED <date> — <what>` block naming your worktree, and **push that claim
to main before you start**. Several agents run here at once; a claim that lands
with the work is a claim that did nothing. Delete it when the work lands, or if
it goes stale for more than a day.

## Release preparation checkpoint 2026-10-09

Comprehensive `RELEASE_NOTES_v0.8.43.md` is committed before any version bump.
Merged scope: PRs #519, #520, #522, #523, #524, #525. No new version/tag or
production quantization pin. Full no-publication Release dry run
[GH 37958100129](https://github.com/CrispStrobe/CrispASR/actions/runs/37958100129)
has workflow event head `23b401b3d20a2ef794a88ade6a261369e899359d`; the old checkout selected the moving branch at each job start. CPU/Vulkan/shared artifacts actually built `3b814f780277286ece56e8cdee6bdc435e07ae4d`; Python built `be13df2d3b0796b3b0bc23c5df714382a2acca28`. Observed changes since the event head are documentation only, but this is not an immutable matrix proof. Checkout repair is claimed above. All 35 build/validation jobs pass, and the full no-publication workflow including its bookkeeping job is now SUCCESS. Both CUDA 12.8/12.6 Windows CLI split and library artifacts are present. Actual archive SHA pairing plus mixed-12.8/12.6 rejection control passes on GH CPU 37980464852, source `202ac02932908e698fcc254eb16a5c077fb22333`; multi-gigabyte archive work stays on GH. Actual CUDA package C ABI discovery failure is tracked in PR #528; this old package is not production acceptance. Linux x86_64 CPU/Vulkan
packages have been downloaded and physically inspected: both CLI/quantizer
start, all loader dependencies resolve, both executables use $ORIGIN without
build-host paths, and the needed OpenBLAS library is bundled. The shared-library package also loads through ctypes, returns 0.8.42,
exports the actual session open/close and additive preencode symbols, and
successfully enumerates Nemotron/VoxCPM2 through the C ABI. This is package
startup/dependency validation, not Vulkan hardware/model acceptance. Full
receipt: `docs/release-package-dry-run-2026-10-09.json`. Main integration CI
for the final checkpoint still needs to settle. Staged WASM initializes under
Node and passes nine export checks, four unopened-session getter contracts and
three existing piano assertions. Linux CUDA CLI/quantizer also start with loader
overrides cleared; baseline x64 CPU fallback is bundled, both executable loader
closures resolve, and the dependency contract passes. CUDA backend libraries
correctly retain external driver/runtime/BLAS dependencies (ABI 12); they are
absent on this CPU host, so this is not GPU loading or inference acceptance.
Actual CUDA artifact source is `3b814f780`, recorded from the CLI. Latest
full no-secrets scan passes all four checks without skips (57.145 seconds).
Q4 encoder isolation `37957419289` and original OmniVoice generation
`37957953495` completed successfully as diagnostics; quality gates remain
rejected. Attention-source Q4 encoder gate is 0.891277 at 636,775,328 bytes;
FFN-source Q4 is 0.993778 at 1,034,709,920 bytes. Capture OFF/ON/OFF controls
pass, all 24 F16 layers still pass original, and both candidates retain actual
Q4 payload with physical protected/unrelated byte audits. Original OmniVoice
F32 seed-42/32-step cloning also fails at 2/9 WER in both unprocessed and full
cleanup arms; capture OFF/ON PCM is exact and original-ASR JFK control passes.
The upstream failure does not waive native acceptance. Full public proofs
are pinned and remote-SHA verified in the new original-clone and encoder-group
diagnostics. Issue #518 and PR #521 now record the terminal original result. Documentation anchors pass (227 across 82 files). The current control
checkpoint passes all four no-secrets tests with the account environment set
(no skips, 56.378 seconds). The output-lifetime lesson and generated index are
updated; CPU capture evidence is explicitly not GPU capture acceptance.
Tag only through `scripts/bump-version.sh` after green chosen tip; verify
release packages before publication. The pending quant/clone experiments are
excluded from this release scope. Issue #485 has the terminal calibrated-Q4
rejection update; draft PR #521 records the completed, rejected original control.

## CLAIMED 2026-10-09 — Nemotron Q4 RNNT precision repair

Owner: this maintainer session. Worktree
`/mnt/storage/crispasr/salvage-20261008/nemotron-quant`, branch
`fix/nemotron-rnnt-quant`. Original NVIDIA/F16 fixed-audio proof is complete:
all 22 VoxCPM2 controls read correctly; published native Q4 reads only 8/22.
No fresh/reused state differences. Isolate joint/predictor/prompt critical
precision using existing quantizer overrides, compare Parakeet RNNT/GigaAM
rules, physically audit source types and size, retain encoder weight equality,
and run original stage/norm gates plus full decoded controls. GH CPU only;
no model publication/default change until acceptance. Worker `64972cd34`,
GH `37907322941`, compares F16, published/plain Q4, joint+embed and full
RNNT+prompt guards on all 26 original controls in fresh/reversed-reused
sessions. Raw unchanged strict stage/norm logs and per-tensor byte/type
audits are mandatory; decoded recovery alone cannot promote a failed stage
gate. Completed successfully as a diagnostic: full RNNT+prompt preservation
recovers original words on 26/26 controls and all 22 VoxCPM2 controls at
440,576,928 bytes, versus 479,337,248 published Q4 bytes. Fresh/reused outputs
agree, but encoder cosine minimum 0.947463 fails the unchanged 0.999 gate
(F16 0.999982). No promotion. Next expose existing preencode diagnostics and
compare the earliest reference boundary before further encoder guards. Worker
`c9aff16e7`, GH `37915072437`, FAILED its F16-first control: mel cosine minimum
0.993376 and preencode 0.943016, while encoder remains 0.999982 PASS.
No further Q4 guards ran. Terminal logs/artifacts archived. Worker `85c04e5ff`,
GH `37916931975`, FAILED the same unchanged gate and retained full arrays.
Only mel frame 1100 and subsampling frame 138 have max error >0.1. The
reference terminal mel frame is zero; native is log-mel. Upstream current
NeMo uses floor(n_samples/hop) and masks invalid frames to zero. Worker
`464bc0438` implements that length rule while retaining shape/thresholds;
GH `37938926545` completed the F16-first three-stage control and guarded
Q4 comparison: F16 mel/preencode cosine minima round to 1.000000; encoder
0.999991. All three unchanged gates PASS. Q4 first diverges at preencode
output projection (cosine minimum 0.991164). Retaining its actual source F32
restores preencode parity at 455,895,968 bytes, but encoder still FAILS
0.938963. No Q4 promotion. Next compare original/native encoder layer outputs. GH `37939212612` completed: F16 matches original words on 26/26 controls,
all 22 VoxCPM2 controls, and fresh/reversed-reused outputs agree. All three
strict stage gates PASS. RNNT+prompt Q4 still recovers 26/26 words but fails
preencode/encoder parity; no Q4 promotion. The isolated frontend and additive boundary diagnostic are merged in
[PR #525](https://github.com/CrispStrobe/CrispASR/pull/525), commit
`03c22e3f37e3b824d080c925ea067807418c3016`. All 44 applicable PR checks
passed, with one intentional skip. Exact source `1977bd047` passed the CPU
controls and actual dual-T4 stage/transcript/streaming validation described
below. The integration claim is complete; encoder Q4 investigation remains
active in its separate experimental worktree.
Terminal 26-control proof: `docs/nemotron-frontend-controls-2026-10-09.json`. GPU-only worker draft
`3f1ef083b` lives in the claimed `nemotron-quant` worktree. It requires
immutable successful bundle pins before launch, actual SM75 hardware/native
CUDA allocation, full F16 arrays/unchanged 0.999 stage gates, original 26-case
fresh/reused transcripts, and long-turn packet/reset/flush checks on presets
0/2/3. No GPU build/quantization, experimental incremental frontend, or claim
of original streaming parity. The original 24-layer encoder capture worker
`3ca187201`, GH `37946465069`, is running on CPU with pinned original NVIDIA
weights and Transformers 5.19.0. Capture must preserve original generation
IDs exactly; clone subsampling outputs before their in-place input scaling.
Next compare these original layers to native F16/Q4 before choosing more guards. Native diagnostic worker
`4818ccefe`, GH `37947134922`, now captures all 24 full-sequence outputs
with explicit ggml output lifetimes. It requires capture OFF/ON/OFF final
encoder bytes to agree for F16, plain Q4 and RNNT+prompt+source-F32-preout
Q4. F16 must pass existing original three-stage gates first. Model/tensor
hashes and complete arrays are archived; no threshold, quant default or
production capture change. Compare against original layer reference only
after both workers complete; failed capture controls invalidate the probe. Both layer jobs completed. Original
capture preserves generation IDs exactly and retains all 24 layers. Initial
native final output OFF/ON/OFF is byte-exact, but intermediate F16 arrays
reuse two buffers and are INVALID. No Q4 guard may be chosen from them. The explicit-reserve rerun
GH `37953858590` SUCCESS validates every F16 layer against original (all
minima >0.999) and all capture OFF/ON/OFF controls. Plain Q4 first fails
layer 0; RNNT+prompt+source-preout Q4 first fails layer 1, cosine minimum
0.99676585 and relative-L2 0.06854713. Public raw complete arrays and
per-layer comparison are pinned in the layer-probe diagnostic. Next this
session claims a predeclared two-group precision isolation in the same
`nemotron-quant` worktree: retain existing RNNT/prompt/preout guards and
compare attention-only source preservation versus FFN-only preservation.
Physically audit source types/bytes, unrelated Q4 identity and actual sizes;
retain unchanged F16-first stage gates/capture controls. No defaults or
model promotion. The two-group worker `1e32cc0af` is pushed and GH CPU
run `37957419289` has launched. Every candidate must preserve protected source
bytes, keep unrelated tensors identical to plain Q4, and retain nonzero Q4
payload. The unchanged F16 stage control and capture OFF/ON/OFF checks pass.
Both encoder gates fail. Proper frame-axis comparison (separate from unchanged
CLI layout) shows FFN preservation first failing layer 5, final relative-L2
0.016656; attention preservation first fails layer 1, final relative-L2
0.062321. Next this session claims two predeclared FFN projection isolation
arms in the same experimental worktree: hold attention at source precision,
then preserve FFN linear1 versus linear2. Retain common RNNT/prompt/preout
source guards, unchanged F16-first gates, physical source/unrelated-Q4 byte
checks and nonzero Q4 payload. No production quantization or acceptance claim. Worker `47e48bd5c` is
pushed; GH CPU `37964322326` completed. Both 926,182,304-byte candidates
retain 113,246,208 Q4 bytes but FAIL the unchanged encoder gate: linear1
source cosine 0.977366; linear2 source 0.882196. F16 still passes all three
stages and all 24 original layers; all five prior baselines and the PCM
reproduce byte-exactly. Capture OFF/ON/OFF remains exact in all seven arms.
Proper frame-axis first failures are layers 2 and 1; final relative L2 values
are 0.025868 and 0.059344. Full raw arrays, tensor byte/type/SHA audits,
original gold and worker are public and remote-SHA verified in
`docs/nemotron-encoder-projections-2026-10-09.json`. No promotion.
Completed three predeclared attention precision isolation
arms in the same `nemotron-quant` worktree, starting from the better
FFN-source baseline: preserve attention q/k/v versus out versus pos at
actual source precision, retaining existing RNNT/prompt/preout/FFN guards.
All unprotected tensors must stay byte-identical to plain Q4; minimum actual
Q4 payload is 28,311,552 bytes (the q/k/v-source arm). Keep original F16-first
three-stage gates, complete layer captures and OFF/ON/OFF controls. These are
mixed-precision diagnostics, with no speed or production Q4 shipping claim. Worker `7b9e0fc25` is pushed;
[GH CPU 37971678214](https://github.com/CrispStrobe/CrispASR/actions/runs/37971678214)
completed with all seven existing baselines plus the three attention arms.
Declarative selection checked against the actual prior physical tensor receipt:
q/k/v matches 72 additional matrices and retains 28,311,552 Q4 bytes; out/pos
each matches 24 matrices and retains 56,623,104 Q4 bytes. Terminal results are recorded below.
 Parakeet RNNT preserves joint/embed; GigaAM preserves
RNNT/prompt-front precision. Those small-head rules do not explain the
remaining Conformer drift, so isolate learned encoder groups before further
calibration or guards.
Worker `307bcf92d` explicitly reserves diagnostic output lifetimes; rerun
GH `37953858590` completed successfully, with all 24 F16 layers passing. Raw rejected probe and original gold publicly
archived: `docs/nemotron-layer-probe-2026-10-09.json`.
CUDA bundle GH `37943881572` SUCCESS at exact PR source `1977bd047`;
SHA-verified transfer is pinned in `docs/nemotron-frontend-gpu-2026-10-09.json`.
Kaggle `nemotron-frontend-validation` v1 initially ran worker `307bcf92d`, with
pulled source byte-identical. Actual native CUDA allocation, all three F16
stage gates, all 26 fresh/reused original words and long-turn streaming
packet/reset/flush gates were required. v1 ERROR before inference:
validator used an unversioned library name, but the bundle's SONAME is
`libcrispasr.so.1`. Full terminal log/source publicly archived and verified.
Worker `bbf9f165e` resolves the bundled SONAME and checks ELF loading before
model downloads; v2 COMPLETE/PASS on actual dual T4s, on the same immutable
source/build/thresholds. All three stage gates pass; encoder cosine minimum
0.99999088 and relative-L2 0.00028004. All 26 original words match in fresh
and reverse-reused sessions. Streaming presets 0/2/3 pass long-turn
packet-size token equality, byte-exact reset/repeat confidence, live tokens,
monotonic frames and idempotent final flush. No original streaming parity
or Q4 acceptance claim. Full public proof is pinned in the GPU diagnostic.
Remote worker source is byte-identical. All four local no-secrets tests pass
with the account environment explicitly set (156.584 seconds). All 26 archived
original NVIDIA attention masks independently confirm floor(samples/hop);
full failed-boundary arrays and logs publicly archived and SHA-verified:
`docs/nemotron-boundary-diagnostic-2026-10-09.json`.
Completed quant diagnosis:
`docs/nemotron-q4-precision-diagnostic-2026-10-09.json`.
Pinned original public proof:
`docs/nemotron-tts-reference-diagnostic-2026-10-09.json`.

Attention isolation is terminal: all three candidates fail the unchanged 0.999
encoder gate, despite F16 passing all three stages and all 24 frame-layer gates.
All seven previous baseline arrays and original PCM are byte-exact; tensor audits
confirm true Q4 payload and byte-identical protected source tensors. Closest
FFN+QKV-source candidate is 1,143,237,536 bytes with 28,311,552 actual Q4 bytes,
encoder cosine 0.998536; first proper-frame divergence is layer 21, final
relative L2 0.00946937. Out-source/pos-source arms at 1,070,885,792 bytes fail
0.996296/0.987063 respectively. No promotion. Receipt:
`docs/nemotron-encoder-attention-2026-10-09.json`.
Next claimed CPU experiment in the same isolated worktree: retain FFN+QKV source,
then separately retain all attention out or all position matrices, or retain
both only in late layers 20–23, 16–23, 12–23. These explicitly mostly-source
hybrids must retain at least 14,155,776 actual Q4 bytes; this payload floor is an
experimental scope constraint, not a change to any quality gate. Keep all ten
baselines, F16-first unchanged three-stage controls, tensor equality audits,
24-layer arrays and capture OFF/ON/OFF controls. No decoded/GPU promotion until
stage acceptance, and no claim that a near-F16-sized hybrid is a small Q4 model.

## CLAIMED 2026-10-09 — PR #492 salvage, OmniVoice cleanup and remaining performance work

Owner: this maintainer session, resumed 2026-10-09. Separate worktrees under
`/mnt/storage/crispasr/salvage-20261008/`: `mel`, `mimo-integration`,
`omnivoice`, `echo`, `profile`, `lm`; PR #492 diagnostics remain under
`/mnt/storage/crispasr/pr492-acceptance-20261008/worktree`. Do not modify the
original checkout or live service. Large scratch/models/proof stay on storage;
GH CPU and the one-account Kaggle real-GPU protocol apply. Every terminal GPU
run is archived before reuse, including rejected speech results; successful
actual-worker builds refresh their compiler cache. Thresholds stay unchanged.

- **MiMo/mel opt-ins integrated.** PR #519 merged as `aa061178d`; PR #520
  merged as `02e404538`, preserving the mel author's original commit. All
  applicable PR native/lint/binding and selected regression checks passed.
  Integrated native file hashes match accepted T4/CPU proof exactly. Both
  defaults remain OFF; measured gains and hardware limits are in HISTORY.
  Python array repair [PR #522](https://github.com/CrispStrobe/CrispASR/pull/522)
  also merged as `5360a96f6` after all 50 applicable CI checks PASS (one skip),
  including eight tests through the production shared library. Final combined-main native CI `37899170996` PASS on `892dca1a9`: all
  13 platform/unit/ASan/fuzz/dynamic-backend/codec/Vulkan jobs complete.
  Terminal receipt and log are archived on storage.
- **OmniVoice #518 — DSP exact, cloned speech rejected.** Draft
  [PR #521](https://github.com/CrispStrobe/CrispASR/pull/521) ports pinned
  upstream audio/text utilities with config opt-outs and processed-PCM cache
  semantics, preserving the params ABI. Native `a1ae00dff` passes 738 waveform
  and 18 text cases. Actual GH `37889514048` has identical generated codes and
  exact decode DSP waveform parity (max_abs 0.0); unconditioned raw/full/default
  marked speech all WER zero. Session raw/marked and CLI clones all substitute
  fox/dog -> fix/dig, WER 2/9=.2222, failing the unchanged <=.2 gate. Issue
  remains open; no native integration. Diagnostic worker `1b0ea317f`, GH
  `37891923274` COMPLETE: legacy and punctuation-only clones also fail at
  WER 2/9; reference/output cleanup fail at the same WER, with output-only
  codes identical. The rejected baseline error predates this DSP port.
  Fixed-audio cross-recognizer/resampling diagnostic is complete: Nemotron and
  Parakeet retain 2/9 substitutions across all three samplers; Qwen3 reads all
  four clones correctly. Neither output-only DSP nor resampling clears the
  unchanged primary gate. All 42 applicable code CI checks pass (one skip);
  full cloned-speech acceptance still fails. Original NVIDIA/F16 ASR retains
  the same substitutions. This session claims the missing original OmniVoice
  generation control in separate worktree
  `/mnt/storage/crispasr/salvage-20261008/omnivoice`: pinned upstream `08be0b4`
  and original model `c5fdb5cc`, immutable existing reference/text, seed 42 and
  32 steps. Compare baseline versus full preprocessing/postprocessing, archive
  complete PCM/codes and recognize with the pinned original NVIDIA F32 model.
  Run on GH CPU; no seed selection, gate relaxation or default promotion.
  Worker `381f3a3ae` is pushed; GH `37957953495` launched with pinned CPU
  torch/torchaudio 2.8.0, Transformers 5.19.0 and upstream package revision.
  It requires original generation capture OFF/ON PCM equality and a positive
  original-ASR JFK control before reporting cloned-speech WER. COMPLETE:
  both original arms fail at 2/9 WER, with exact capture controls and positive
  JFK control. Original model failure is evidence, not a relaxed native gate.

  No lucky-seed selection. Public proof:
  `docs/omnivoice-clone-failure-2026-10-09.json` and
  `docs/omnivoice-audio-utility-parity-2026-10-08.json`.
- **Echo 9B Q4 — public preparation complete, all candidates rejected.** GH
  `37851965549` produced/audited all four recipes publicly, remote size/SHA
  verified and all 177 source F32 sensitive tensors retained. Immutable
  preparation `afdef3d4b111aa89e45fc400a03ba31ff54ac1dd`. Public GPU retry v2
  (entry `845bf2e6d`, SDK `2790778aa`) ran on two T4s with an accepted F16
  control. All four recipes fail full stage/norm and exact complete-output
  gates. The middle-FFN recipe passes three Piper clips at WER zero but still
  fails strict full acceptance. No registry/default promotion. Receipts:
  `docs/index-echo-q4-preparation-2026-10-09.json` and
  `docs/index-echo-q4-rejection-2026-10-09.json`. Next audit first divergent
  tensors and calibration coverage before another quant recipe; plain
  `CRISPASR_IMATRIX_OUT` previously collected no decoder statistics. Branch
  `c247d0135` composes an opt-in collector with stage capture, retaining the
  original callback without calibration env vars. Actual collector sums/counts
  unit and fresh sm75 CUDA bundle PASS in GH `37901223622`, packaging `a03820a4f`.
  The actual CPU unit verifies GGUF activation sums/counts; GPU-less CUDA
  compilation does not establish GPU acceptance. Logs and bundle are archived
  on storage. Bundle checksum `894bf650161f4c672a2992da394cdb8d65cca69af2f17c5b960c52bd4b6826f9`,
  immutable tooling transfer `b284cbc8727e1f91904a4c2928e6991bd5bea938`.
  GPU worker `e4a64a9b8` runs unchanged full F16 acceptance with collection
  off/on, identical decoded receipts, and all 96 FFN matrix widths/counts.
  Its held-out statistics are explicitly unsuitable for quantization; a
  disjoint EN/ZH production calibration corpus is still needed. Kaggle
  `index-echo-calibration-smoke` v1 COMPLETE on two actual T4s. Both arms
  pass all stage/norm, full file translation, CLI/C ABI and roundtrip gates;
  decoded receipts are identical. 249 decoder matrices collected, all 96
  required FFN widths/counts valid. Held-out smoke statistics stay excluded
  from quantization. Full proof: `docs/index-echo-calibration-smoke-2026-10-09.json`. Disjoint 24 EN + 24 ZH CC0 corpus preparation is GH `37905840624`,
  worker `e35df5895`, stopped before preparation because hosted ffmpeg was
  absent; terminal log archived (no artifact emitted). Worker `861b82411`
  installs the missing tool; retry GH `37907323063` completed successfully:
  48 clips, 259.176 seconds, all PCM disjoint from held-out acceptance.
  Inputs and corpus selection unchanged. Publish the corpus and collect fresh
  actual GPU statistics before quantizing. Public corpus with verified checksum:
  `docs/index-echo-calibration-corpus-2026-10-09.json`. GPU collector
  `93f660df6`, Kaggle `index-echo-calibration-corpus` v1 COMPLETE on dual T4s,
  processed all 48 clips in one session and proved decoder CUDA assignment.
  249 decoder matrices collected; all 96 FFN matrices have valid widths,
  finite nonnegative nonzero statistics and positive counts. Full terminal
  logs, source and statistics publicly archived and remote-SHA verified.
  Fresh corpus imatrix SHA `4cb3e75e9ef3f09ed65860b172d5e0cfb994cca16544db31904e269d91cc9bb4`.
  No Q4 acceptance. This session claims the next CPU producer in the same
  `echo` worktree: imatrix-aware plain/FFN-guarded/middle recipes, exact
  source-F32 preservation, tensor audits and public experiment-only upload,
  followed by unchanged actual GPU acceptance. Worker `72aac76a5`,
  GH `37917151969`, completed calibrated plain/FFN-guarded/middle
  preparation. Public experiment receipt pin is
  `83ac9d70bd3eaec34cc1dc19dbff605ce5869d85` under
  `experiments/q4-imatrix-20261009/preparation.json`. Actual GPU acceptance
  of all three candidates is the next task in the claimed Echo worktree. All 96 FFN statistics and original-F32 floors are re-audited;
  imatrix loading/shape evidence is required. No candidate accepted. GPU
  worker `c150a6cb5`, Kaggle `index-echo-q4-imatrix` v1 ERROR after complete
  acceptance on actual dual T4s. F16 full control PASS; all three calibrated
  candidates REJECTED by stage/magnitude gates and full Chinese translation
  mismatches. Plain Q4 also fails CLI/C ABI and timestamps. Full terminal
  outputs archived; next localize decoder magnitude drift before another
  recipe. Pulled remote source is byte-identical. Full preparation proof/pins:
  `docs/index-echo-q4-imatrix-preparation-2026-10-09.json`. Release notes for already merged changes
  are drafted in `docs/release-notes-next.md`; no tag cut. Callback support merged as PR #523
  (`74a35b198`) after 42 applicable CI checks passed (one intentional skip). Build/launch pins: `docs/index-echo-calibration-build-2026-10-09.json`.
  Actual GPU calibration and coverage are complete; all three calibrated Q4
  candidates remain rejected. Next localize decoder magnitude drift before any
  new recipe; full candidate acceptance is still missing.

- **VoxCPM2 — step/rate repairs on branch; primary Q4 recognizer rejection isolated.**
  Profile branch forwards positive step overrides through native/C ABI/CLI,
  preserves ten-step default and returns actual native 48 kHz session PCM
  (old ABI decimated to 24 kHz while advertising 48). v4 five/six speech gates
  fail. v5 diagnostic crashed at an untyped ctypes setter; v6 uses the public
  pointer-width-safe setter and completes all four existing fused/per-step/
  batch/split Vulkan/CUDA paths, two models/texts/two byte-exact repeats.
  Six/sixteen primary Nemotron ASR cases pass; every short transcript repeats tokens.
  Disabling CFM fusion alone does not fix it. Native arithmetic/defaults stay
  unchanged. GH original-controls `37892473946` COMPLETE at worker `08be8c020`:
  pinned official Python BF16 full generation and native F16/Q8 CPU controls,
  same seed 2/ten steps/texts/repeats, full PCM/latent and ASR evidence.
  Official Python BF16 and native F16/Q8 short output all repeat sent in the
  Nemotron transcript; all three long controls pass. This does not establish
  whether the audio stutters or the recognizer/front door causes the repeats.
  Fixed-audio GH `37899546009` now shows Parakeet and Qwen3 read every one of
  22 VoxCPM2 controls exactly across all three samplers (132 transcripts),
  while Nemotron retains failures. Speech is recognizer-dependent; no synthesis
  defect or accepted replacement recognizer is established. Official NVIDIA
  ASR plus native F16/Q4 fresh/reused controls: GH `37902211301` stopped
  before inference because `librosa` was missing; all terminal logs/artifacts
  were archived. Retry GH `37904497983` adds pinned `librosa==0.11.0`, with
  source `397334025`, original weights, inputs and speech gates unchanged. COMPLETE: original
  NVIDIA and native F16 read all 22 VoxCPM2 controls correctly and have equal
  normalized words on all 26 cases. Published Q4 reads only 8/22 Vox controls;
  fresh/reused transcripts agree. This isolates the transcript repeats to
  native Nemotron Q4, not synthesis. Original NVIDIA retains all four OmniVoice
  clone substitutions. Public full proof: `docs/nemotron-tts-reference-diagnostic-2026-10-09.json`.

  Public proof/cache: `docs/voxcpm2-path-v6-diagnostic-2026-10-09.json`;
  earlier failed receipts remain in docs. Public model card now qualifies
  historical timings and discloses failed current GPU speech; no weights
  changed (`docs/voxcpm2-model-card-correction-2026-10-09.json`). Intel B390
  performance remains unvalidated by NVIDIA evidence; #461 stays open.
- **MiMo Q4 LM — first-divergence capture complete, no production fix.** GH
  `37855479658` compares byte-identical frozen layer-0 inputs across CPU
  flash/eager and BLAS controls. Q4 differences begin in layer-0 attention
  (relL2 .000146), amplify in MLP (.00183) and later layers. BLAS reduces
  flash/eager drift but original Python hidden/logit parity stays ~.965/.982,
  below acceptance. No BLAS/default promotion or original-checkpoint layer
  oracle claim. Raw proof: `docs/mimo-lm-first-divergence-2026-10-09.json`.

The unchanged secret scanner passes after account-placeholder repair
`731593dda`; #519/#520 checks passed and both PRs are merged.
PR #492's CANN RVQ/F16 embeddings and ggml NZ require actual Ascend evidence;
CUDA is not a substitute. Native/worker hashes on isolated branches match
accepted proof. v0.8.42 excludes these new changes: its release publication
is COMPLETE SUCCESS with 53 assets; full proof is in
`docs/release-v0.8.42-validation.json`. Storage maintenance is complete for the
verified inactive files, original source paths remain symlinks; local receipts
are under the task proof root. External reporter hardware retests and #456
remain pending/deferred; do not archive them as resolved.

Fixed-audio diagnostic GH `37899546009` COMPLETE (234 transcripts/26 clips),
source `43dc3194a`, pinned public proof in
`docs/tts-asr-frontdoor-diagnostic-2026-10-09.json`. The C ABI file loader is
miniaudio, not the shared CLI Kaiser path; the initial run `37898319808`
correctly rejected its residual tone level before transcripts and is archived.
No speech thresholds were relaxed. Python #522 fixes the independently known
aliasing/strided-buffer defect; it does not explain every speech failure.
Next original ASR diagnostic in profile worktree `397334025`, GH
`37902211301`: pinned NVIDIA Transformers 5.19 model-card offline generation
stopped before inference on a missing `librosa` dependency. Retry GH
`37904497983` adds `librosa==0.11.0` and is COMPLETE. Identical 16 kHz PCM,
native F16/Q4 plus fresh/reversed-reused sessions and full original
features/encoder/sequence evidence remain required. No recognizer promotion.



## VALIDATED 2026-10-08 — MiMo full CUDA speech acceptance and precision profile

Worktree: `/mnt/storage/crispasr/pr492-acceptance-20261008/worktree`, branch
`review/pr492-acceptance`. Kaggle `mimo-pr492-asr-cuda` v3 COMPLETE on one
Tesla T4/sm_75 at numeric `6b90353678f525e634c1a359355ae15c2c815ace`, packaging
`e77acb058`. All terminal artifacts are archived and the numerical arrays,
logs, transcripts and timing calls are public with an immutable pin/hash.
The full shared-library tokenizer matches all 2,208 independent official
encoder/quantizer codes in precise flash and eager modes; unchanged continuous
A/B gates pass (transformer relative L2 1.18811e-5, pooling 9.22843e-5).
Actual GGUF audit selects all 192 quantized codec matrices and none of 354 LM
matrices; dispatch logs and GPU PID prove execution. Earlier v1 missing-submodule
and v2 incorrect-prefix failures are preserved in their receipts, not accepted.

All 12 EN/ZH CLI/session outputs agree across default/precise/eager. All 64
full-inference profile calls retain exact decoded output, including warmups.
Same-binary/GPU/files ABBA profiling (2 warm + 6 measured per clip/process,
12 measured samples per clip/mode) gives EN 3.37865 → 3.61968s (+7.1%) and
ZH 1.45788 → 1.60141s (+9.8%). LM flash is unchanged between timed modes.
Sampled GPU0 process-memory peaks rise 4,932 → 4,952 MiB (+20 MiB), lower
bounds including load. Original Q4 codec/LM files stay 395,594,656 /
4,517,988,832 bytes. Decision: keep production defaults unchanged; no speed win.
Receipt: `docs/mimo-pr492-full-asr-cuda-2026-10-08.json`; successful full-build
cache refreshed at `0480221f9f7e5b16773ad8ad7e673cfc1afcdc94`.
Separate Q4 LM non-flash numerical, original-checkpoint and CANN hardware gaps
remain. PR #492 stays unmerged and excluded from v0.8.42.

Release gate COMPLETE at accepted main `032621174`: native `37831067687`
13/13 PASS, lint `37831067785` 10/10 PASS, full33 `37802900089` 33/33 PASS.
v0.8.42 published at annotated tag `3be8bcb4390b8080dc73f807b7abb34dc0aca9c3`
using the official bump script; all version files and the Dart changelog agree.
Comprehensive notes are retained in the tag and GitHub release. Binary assets
are building; verify per-platform publication before claiming packages available.
Receipt: `docs/release-v0.8.42-validation.json`. #492 remains excluded.


## OPEN 2026-10-08 — MiMo CPU BLAS follow-up validation

Worktree: `/mnt/storage/crispasr/pr492-acceptance-20261008/worktree`, branch
`review/pr492-acceptance`. The complete promotion oracle isolates CPU Q8
activation quantization. Test a diagnostic-only scheduler variant that adds
BLAS ahead of CPU for CPU tokenizer sessions, retaining the original Q4 file
and unchanged official-reference, attention A/B and exact RVQ gates. Hosted
`37792853553` at `2a4eda81d` PASSES: transformer A/B relative L2 1.04712e-7,
pooling 1.12548e-6, exact RVQ codes; official encoder/pooling <=3.05219e-7 /
6.10356e-6. Original file stays 395,594,656 bytes (zero size increase).
Paired tok_codes extraction takes 10.40/12.58s versus 6.08/6.61s on CPU:
1.71/1.90x slower, not a speedup or a default-flip justification. Full proof:
`docs/mimo-pr492-cpu-blas-2026-10-08.json`. Next validate placement, repeated
timing/memory, short EN/ZH and ARM before production use. No production
precision/default change has landed. No direct BLAS-module
symbol dependencies, GPU placement changes or reference tolerance changes.

## OPEN 2026-10-08 — PR #492 remaining acceptance

Latest full-library CUDA diagnostic: numeric `6b9035367`, harness `e77acb058`.
All 2,208 official tokenizer codes and continuous gates pass after scoped
precision controls, and all EN/ZH CLI/session outputs remain exact. That
control is 7.1–9.8% slower end to end: production defaults remain unchanged.
Receipt: `docs/mimo-pr492-full-asr-cuda-2026-10-08.json`.

Still unresolved: Q4 LM non-flash hidden/logit relative L2 6.1097% / 7.4700%
(F16 LM passes), first-divergence localization and original-checkpoint
precision scope. The separate `lm` worktree starts with diagnostic selected
LM-layer captures and a CPU activation-quantization control, without model
promotions or relaxed gates. CANN RVQ/F16 embedding and ggml NZ need actual
Ascend proof and PR-only/NZ-only/both attribution. The independent mel piece
is split into the opt-in branch above. Historical numerical/codec experiments
are archived in HISTORY; the full PR stays open and unmerged.


## OPEN 2026-10-08 — #483 reporter GPU comparison

The requested matched MMQ experiment is built and packaged. Both Windows jobs
pass in `37733919241`; the actual uploaded manifests pair locally with identical
source, ggml pin, CPU floor, PTX targets and CUDA runtime DLL hashes. The final
hosted pairing job also passes (`113183674377`); the complete workflow is green.
Packages, exact source and scope:
[experiment](docs/cuda126-mmq-experiment-2026-10-08.md) and its JSON receipt.
The reporter can now test the GTX1660/MX150. No release defaults or GPU speed
claims change; keep #483 open for actual hardware correctness/timing results.

## OPEN 2026-10-08 — final main checks and remaining triage

Orukeet CPU acceptance [37728843835](https://github.com/CrispStrobe/CrispASR/actions/runs/37728843835)
passed at `4b45fec39ff603ad68bad0435c733e41695bfe10`: fresh download,
cached short name, filename, explicit path and anonymous C ABI all produce
identical speech text. Released v0.8.41 reproduces #491; missing licence
acceptance refuses without downloading. Proof is in the triage receipt. #491 is closed with the hosted proof linked.

Integrated #496/#514 source and APT setup pass native CI `37704656972`,
regression `37704657102`, lint `37704656959` and five WASM builds
`37704657094` on `d754aa10a`; Node emits three note-binding assertions.
Lint Deep found an uninitialized Hikari probability, fixed in `4b45fec39`;
focused cppcheck passes. Full hosted rerun `37729063238` failed on a Hikari constructor virtual call.
Fixed in `f55c7bc8c`: constructor and reset share a nonvirtual helper. Focused
cppcheck 2.17.1 and C++ syntax pass; full pinned cppcheck 2.7 rerun
`37737578258` passes the full pinned cppcheck 2.7 run.
No new GPU performance claim. Cold proof: `/mnt/storage/crispasr/triage-20261007/`.
#490 character alignment is integrated with full real Arabic/Java proof. #483 has matched packages;
reporter GPU output/timing comparisons remain pending.
#492 MiMo/CANN remains unmerged pending numerical diagnosis; #516 is integrated.

## OPEN 2026-10-05 — live transcribe + translate: follow-ups

Shipped 2026-10-06 (#493) together with the ggml v0.26.0 sync — what landed
and how it was verified is in `HISTORY.md`; usage, model verdicts and
measurements are in `docs/streaming.md`. ⚠ Still true after the sync:
`GGML_PREC_F32` is now 10 (was 1), so a binding passing a raw `1` breaks
silently (not audited); `ggml_*_set_prec` are deprecated upstream (58 call
lines here); `tests/test-metal-pipeline-cache.mm` now executes both lifecycle cases on
actual macOS in shared/static builds (`37762956547`); speed is not established. Scheduler replay is wired through the standalone
`tools/ci-heavy/scheduler-replay` CMake target and its Vulkan workflow.

Open, in the order they would help:

1. **Re-measure on an idle machine.** Best runs were at load ~5-10.
2. **nemotron session** is real time on CPU now (67 ms per 320 ms chunk at
   load ~8) but CPU-bound: it fell 17-25 s behind when other jobs took the
   cores. The prompt kernel is a ggml graph on CPU now (1.7-2.4 s -> 45 ms
   per 15 s clip, same transcript), and the graph-reuse corruption is found
   and fixed (stale allocator addresses on re-allocation; reuse is exact now
   but saves nothing, so rebuild stays the default). Left: Metal is no faster
   than before (per-op overhead on ~2000 tiny nodes).
3. **Index-Translate-2B** works: the qwen35 loader now skips the appended
   MTP block. Left over:
   `qwen35moe` now skips the MTP block too (compiled; no MoE checkpoint
   run — the smallest is 35B); the interval-derived layer pattern equals the
   file's own `attention.recurrent_layers` on Index-Translate-2B, so the key
   is not read; the empty `<think>` block is now put into the prompt as the
   model's own template does (#498; it used to be generated, ~4 tokens per
   sentence); `test-chat-ggml` passes with Hy-MT2 and Index-Translate-2B (430 /
   420 assertions; one gemma-only assertion was relaxed).
4. The ggml bump was verified before merging: PR CI
   green (80 checks; CrispStrobe/ggml#5 green incl. Vulkan and the CUDA
   compile), and on a Kaggle GPU with CUDA seven backends pass outright
   (parakeet, canary, cohere, sensevoice, qwen3-asr, nemotron,
   moonshine-tiny) and index-echo-2b passes all 67 stages. wav2vec2 prints
   "…what ou can do…" there — identically on `main`, so not the bump.
   Kaggle suite: `transcript_format: srt` is honoured now, and three
   hand-padded GGUF pins (voxtral-mini-3b, parakeet-tdt_ctc-1.1b, mimo-asr)
   are real commits — `check-registry-urls.py` now HEADs every manifest pin.
   Left: wav2vec2's "ou" for "you" on Kaggle hardware only (not reproducible
   here; its WER gate is not loosened blind). canary's per-layer gate is 0.99 since this bump (x86
   layer 18 = 0.9977).
5. ~~VAD re-runs over the whole window every step~~: incremental Silero
   (each 32 ms frame scored once, state kept) — 30-86 → 3-6 ms per step.
   Other VAD models still re-scan.
5b. **Two speeds**: phase 1 (`--translate-revise`, slow re-translation of
   finished paragraphs) and phase 2 (`--translate-revise-asr`, a slower
   recogniser re-reads each finished utterance first) and phase 3 (in-place
   terminal view, `final_until_sentence`, `--translate-revise-backlog`) done.
   Left: a graphical viewer (browser / app) on the JSON events; tuning the
   backlog limit under real load. Open: does the slow pass delay the fast
   one on a shared GPU (measure at load < 3; the wait-for-idle guard is in).
6. **hikari** (sbintuitions/hikari-medium, English -> de/ja/ru simultaneous
   translation + English ASR) is ported: f16 equals the reference on jfk
   (every stage >= 0.9995 cosine, 161/161 stream steps, same text) and on a
   4 s cut (48/48). Left: (a) not real time on an M1 — 1.82 s per audio-second
   on Metal at load ~10; on CUDA (Kaggle) 0.19-0.26 s, i.e. 4-5x faster than
   real time, with q8_0 text-exact too; (b) q8_0 is token-exact on CPU-level text but on Metal changes
   one German sentence of a 27 s clip, q4_k breaks the encoder (cos 0.36) —
   ship f16 until a keep-rule is measured; (c) not published (no registry
   row), so no -m auto and no regression fixture.
7. Drafts: the part two consecutive drafts agree on is marked stable
   (normal text vs dimmed; JSON `stable`). On German speech with Opus-MT,
   whole drafts are rewritten at normalized erasure 1.17-1.48, the stable
   part at 0.16-0.18. Drafts now translate only the source words two
   partials agree on: whole-draft erasure 1.98/1.45/0.66 -> 0.71/0.64/0.05
   (3 pairs), ~12 % fewer correct early words in 2 of 3, half the translator
   calls. Tried and rejected: forcing the stable words as the next draft's
   fixed start (branch feat/draft-forced-prefix: no less erasure, worse
   drafts). Not done: a model trained to revise (StreamRevise).
8. moonshine-de ended its output at the first sentence-final pause. Fixed in
   the library: the German variants decode the stretches between pauses of
   >= 200 ms separately (file, VAD, live and C API: 11/11 sentences on the
   50 s clip, was 6-7/11). Cost: comma pauses as long as 200 ms also split
   (3 extra breaks on that clip). Windows: the live
   pipeline runs end to end on a Windows runner (#503). No timing here was taken on an idle
   machine (load 4-30 throughout).
9. Opus-MT: 24 directions hosted (de/en with fr, it, es, ar, he, tr; see
   docs/streaming.md), each f16 exact on 8 sentences; de↔tr goes through
   English (no model exists). de→ar ships f16 (its q8_0 matched 3/8). A multi-target checkpoint
   (`opus-mt-en-ROMANCE`, f16) matched the reference 8/8 greedy and beam 4,
   and `-tl es` selects the target; a relu checkpoint
   (`opus-mt-tc-big-gmw-gmw`, 437 MB f16, de↔en↔nl in one file) matched
   12/12 greedy and beam 4. gelu is the one activation branch never run,
   and nothing needs it: of 117 Helsinki-NLP checkpoints surveyed
   (2026-10-06) 92 are relu, 25 swish, none gelu.
   Quantisation: everything is quantised, measured — see the table in
   `examples/crispasr-quantize/main.cpp` (`CRISPASR_MARIAN_KEEP`).

## OPEN 2026-10-03 — Index-Echo Q4 quality follow-up

MioTTS and opt-in Echo scheduler source are integrated; see
[the integration report](docs/miotts-echo-integration-2026-10-03.md) and HISTORY.
Public Q4 preparation/transfer is now complete after the private staging
quota failures. Four decoder recipes retain all 177 source F32 tensors,
including 24 recurrent convolution matrices; acoustic tower/connector remain
original. Source pair is pinned at
`cstr/index-echo-9b-GGUF@dffbadf0f173446fee0364a0807803d2b2fb6f49`.

Actual two-T4 retry v2 accepts the F16 control but rejects all four Q4 recipes
on full stage/norm and complete decoded/subtitle gates. Middle-layer gate/up
Q4 passes all three Piper clips at WER zero; this does not establish complete
acceptance. See `docs/index-echo-q4-rejection-2026-10-09.json` for immutable
public proof and every recipe outcome. No candidate is promoted to registry.

Actual decoder collection is now integrated (PR #523), with disjoint EN/ZH
GPU calibration and all 96 required FFN statistics verified. The three calibrated
recipes still fail stage/magnitude and complete Chinese translation controls;
see `docs/index-echo-q4-imatrix-validation-2026-10-09.json`. First quantized
layers diverge at block 0 (plain/FFN-guarded) or block 4 (middle); final/logit
relative L2 remains 4–12.5%. Next localize this error before another recipe.
Retain existing stage/magnitude/cache, full-output and roundtrip gates; do not
retry rejected recipes unchanged or weaken gates. Task/proof is claimed above.

## OPEN 2026-09-30 — voxcpm2 follow-ups (#461, #478)

- **#461 (reporter retest 2026-10-08)**: on main `d754aa10a`, batched prefill
  is active (90.7 ms, no replay), but RTF remains 1.019 at eight steps with
  `voxcpm2-q8_0-locdit-f16.gguf`. VAE setup increased 118.1 → 313.4 ms,
  offsetting the prefill saving; this is not a controlled same-session A/B.
  Reporter now hears poorer quality at eight steps than ten. Ten remains the
  quality default. Earlier ~0.85–0.87 estimate was not achieved. Investigate
  VAE setup and current Vulkan LocDiT matmuls; no new kernel win is proven.
- **#478**: native AVX2/F16C CPU F16/Q8 RALM batching is now validated and default
  (see `HISTORY.md` / `PERFORMANCE.md`). Reporter Windows/249-position Khmer
  measurement remains external. Q4 batching stays opt-in: up to 1.83% state
  error and one extra-word readback; GPU and other CPU ISAs need their own proof.
- `CRISPASR_VOXCPM2_VAE_DW_SHIFT` is default-on for Vulkan only; CUDA/Metal unmeasured.

## DEFERRED 2026-09-24 — #456 nyra-forced-aligner

Deferred by the maintainer. Assessment posted on the issue: WavLM-large + a
Kaldi-style GMM-HMM (model.npz ~2 MB, decision trees, projections) + 41k-word
lexicon + beam Viterbi. Inference code MIT, but model files AND outputs (the
timestamps) are under the nyra health Non-Commercial Research License with
share-alike and a contractual-binding clause (3.3) -> would ship as a gated NC
GGUF; espeak-ng OOV fallback is GPL and cannot be bundled. WavLM exists in-tree
(MioCodec). Not started.

## CLAIMED 2026-09-26 — §247 shared scheduler profiler, then hidden-F16 audit

Worktree `.claude/worktrees/roadmap-445-438-337-perf`, branch
`feat/445-orukeet`. The prerequisite model and GPU stages are complete: Orukeet
and R2T2 (#445), both Hojo checkpoints (#438), and native gfx1100 Vulkan
Qwen3-TTS (#337) are on `main` with model or hardware proof. The remaining
sequence is strict:

1. move the FastConformer scheduler callback into `src/core/sched_prof.h` and
   expose one opt-in profiler usable by scheduler-based runtimes — **DONE**:
   `CRISPASR_SCHED_PROFILE=1` is wired through Canary CTC, Cohere, FireRed-ASR,
   Granite Speech, Moonshine, Moonshine Streaming and Paraformer; the legacy
   `CRISPASR_FC_PROFILE` switch still works;
2. run the profiler/metadata audit over quantized GGUF families, fix the largest
   hidden-F16 matmul offenders one at a time, and prove each A/B before defaulting.
   **Cohere DONE 2026-09-28:** the published Q4 artifact was requantized once;
   all 96 pointwise weights changed F16 -> Q8_0, no other tensor changed, the
   JFK transcript stayed identical, encoder profile improved 10242.59 ->
   9447.52 ms and process wall time improved 10.93 -> 10.16 s. A remote range
   read reconstructed the complete 1,289,178,240-byte GGUF and confirmed
   Q8_0=96. The successful build also refreshed `${KAGGLE_ACCOUNT}/crispasr-ccache`.
   **NOW:** apply the same artifact-time audit to FireRed-ASR's 32 retained F16
   pointwise matrices (~300 MiB). Granite and Moonshine have no large F16
   residuals; Paraformer's are mostly tiny FSMN kernels.

## OPEN — #412 Breeze TTS 2 follow-up: CFG multi-branch

The port shipped (issue closed; record in HISTORY "PLAN compaction 2026-09-30").
Voice Design and Voice Direction are REFUSED rather than downgraded: the cache
topology and branch index exist, per-branch prompt assembly and the logits
combine do not.

## OPEN (optional) — #416 Sidon quant: CUDA MMQ arm

Fixed and shipped in v0.8.32, issue closed (record in HISTORY). Only open:
`tools/kaggle/sidon-quant-cuda/` would answer whether CUDA MMQ (sm >= 61) shares
the broadcast-RPE defect; P100 draws cannot (MMQ off below DP4A).

## OPEN — #387 Quds follow-up

Shipped (record in HISTORY). Port the CTC head when the author uploads the .nemo.

## OPEN — canary follow-ups (after #375, record in HISTORY)

- Linear-resampler gap on 44.1/48 kHz compressed input through the glint decode
  paths (~28 vs ~38 dB after the decoder fix).
- Speed (M1 audit 2026-08-19: encoder compute-bound, decoder ~35 % of GPU wall):
  1. persistent decoder step graph (`canary_decode_step` rebuilds per token,
     ~4.2 ms/tok on Metal; `core_rnnt_ggml::Decoder` pattern, est. ~1.3x);
  2. chunk-batched encode/decode for long-form (NeMo runs batch_size=8).
  Both need byte-identical Metal+CPU A/B and CUDA validation before a default flip.

## OPEN 2026-08-19 — vibevoice-asr 7B answers "[Silence]" on time-stretched audio

The transcript-side damage is FIXED (`3b1bc0b2`, see HISTORY): `[Silence]` is a
Content value the MODEL emits and we no longer pass it through as transcript
text, so it cannot reach an SRT or suppress the empty-transcript warning.

Still open: why the 7B says it at all. Both members of the reporter's atempo pair
(`ko-test` stretched 3.26 s -> ~6 s at atempo 0.535 / 0.525) come back with no
utterance, on CPU and CUDA, in EVERY arm including one with all four #369 fixes
rolled back — so this is not something we introduced. The 1.5B BitNet checkpoint
transcribes the same two files, so it is specific to the 7B. A 2x time-stretch is
not exotic input, and a long recording containing a slow passage would lose it.

Next: dump `speech_features` for a stretched clip against its unstretched
original. The encoder is trustworthy now (cos 0.999926 vs upstream's own
modules), so if the conditioning matches, the divergence is the LM's. Also check
the prompt's duration string ("This is a 6.11 seconds audio") against the 46
speech tokens for an inconsistency the model could read as "mostly empty".

## OPEN 2026-08-19 — vibevoice-bitnet advertises caps its default output cannot support

Fallout from `51b99d1b`, recorded rather than fixed on the way past. The 1.5B now
gets its own plain-text instruction, and in that mode it returns prose rather
than the JSON array — which is what "plain text output" means upstream. So there
are no per-utterance timings and no speaker labels, while `vibevoice-bitnet`
still declares `CAP_DIARIZE` and `CAP_TIMESTAMPS_CTC`.

That is the same class of false claim as the `CAP_TEMPERATURE` removed in
`23107227`: a capability bit is a promise about output the framework then acts
on. Two defensible fixes and they need a decision, not a reflex:
  (a) drop both caps for the 1.5B — honest, and `--diarize` then warns; or
  (b) have the adapter switch to `CRISPASR_VIBEVOICE_ASR_PROMPT=json` when the
      user actually asks for diarization or timestamps, trading the non-English
      quality back for the structure they asked for.
(b) is friendlier but makes output quality depend on an unrelated flag, which is
the kind of thing that gets rediscovered as a bug later.

## OPEN 2026-08-18 — TQ2_0 has no Metal kernels: BitNet models are silent on GPU

`vibevoice-asr-bitnet-*` (TQ2_0 LM weights) yields an EMPTY transcript on Metal:

    ggml_metal_library_compile_pipeline: failed to compile pipeline:
      base = 'kernel_mul_mm_tq2_0_f32'
    Error: Function kernel_mul_mm_tq2_0_f32 was not found in the library

`ggml/src/ggml-metal/ggml-metal.metal` contains ZERO occurrences of `tq2_0` — no
`mul_mm`, no `mul_mv`, no dequant — and `ggml-metal-device.m` has no TQ2_0 entry
either. The type is not supported at all, yet a pipeline for it is still
requested, so it fails hard instead of falling back.

Same clip, same build, only the backend differs:

    ko-mic-cue-kept.wav   CPU (-ng) -> 내일 오전에 회의 자료 교육 보내주세요.
                          Metal     -> (nothing; pipeline compile error)

Impact: every Metal user of a TQ2_0 model gets silence. Ternary/BitNet GGUFs are
what people reach for on laptops, so this is the wrong platform to be missing.

Two questions before fixing: (a) why is a TQ2_0 matmul scheduled onto Metal when
the device declares no support — a type absent from the support switch should
route to CPU, so something is bypassing that; (b) whether the fix is a real
`kernel_mul_mm_tq2_0_f32` (check upstream ggml first — it may already exist) or
an explicit unsupported-declaration so the scheduler falls back cleanly. The
second is small and unbreaks the platform immediately.

Found while reproducing #369, and NOT that issue's cause: the reporter is on
Windows CPU/Vulkan and sees wrong-language output, not silence.

## OPEN 2026-08-05 — carried out of the #316 round-2 work

These are the loose ends from `619e74b6..4b875be0`. None block anything.

1. **fr/es have not been checked for either German defect.** Both were found by
   method, and the method transfers:
   (a) *citation stress* — `espeak_fr.tsv`/`espeak_es.tsv` were generated the
   same way (one word at a time), so they bake in the same isolation stress
   German had. `tools/gen-g2p-de-unstressed.py` is the generator; pointing it at
   another language is a few lines.
   (b) *out-of-vocabulary symbols* — scan the G2P output against the model's own
   `tokenizer.ggml.tokens`. Three lines of Python, and in German it found we
   were deleting `ʏ` out of every München. Worth running for **every**
   non-English Kokoro model we ship, not just fr/es.
2. **Regenerate `espeak_de.tsv` with `--tie`.** Our dictionary has no tie marks,
   so `core_phoneme`'s German `ts`→`ʦ` is a blanket rewrite that cannot tell an
   affricate from a compound seam. espeak emits `t^s` only for the affricate, so
   a tied dictionary makes the collapse exact — and would let
   `CRISPASR_KOKORO_DE_MISAKI_ALPHABET` be judged on its merits instead of on an
   approximation. Regenerate + re-upload to `cstr/g2p-dicts`.
3. **The German tied-alphabet collapse needs a listening test, or a kikiri-tts
   model to test against.** It matches the published recipe and made the ASR
   round-trip worse on the hui base we ship; the hypothesis is that that model
   predates that part of the recipe. `kikiri-tts/kikiri-german-{victoria,martin}`
   are explicitly "misaki 0.9.4 + espeak-ng" and would settle it.
4. **Five `tools/kaggle/*/kaggle_harness.py` bundles are gitignored** —
   `mimo-cuda-rvq-309`, `moss-tts-quants`, `streaming-diarize-300`,
   `whisper-ja-760M-convert`, `whisper-punc-308`. They exist only on the machine
   that made them, so a fresh clone (and the CI checkout) has **no** bundled
   fallback harness for those kernels — which is the exact failure
   `check-kaggle-harness-sync.py` was written to prevent, and it cannot see it
   because it only compares copies that are present. Either `git add -f` them or
   teach the checker to fail on a kernel dir with no bundle.
5. **English `that`/`read`/`used`/`object`/`console`/`use` need a POS tagger** —
   already recorded below with the measurement that says a full spaCy port buys
   0.34%. The cheap slice is a closed-class rule for `that` alone (~24 of the
   ~32 residual tokens, no model).

## OPEN follow-ups from #332 (Rust/Dart diarize ABI, landed 2026-08-05)

The landed part: `DiarizeMethod::FoxNose` + options exposed in the Rust crate
and Dart, and the real bug behind the report — the hand-maintained Rust and
Dart mirrors of the APPEND-ONLY `crispasr_diarize_opts_abi` were never updated
when #324 appended the FoxNose fields, so every `diarize_segments` call from
those bindings had the C side read 24 bytes past the caller's allocation.
Both mirrors now carry the 48-byte layout; `crispasr-sys` has a size/offset
layout test, the flutter smoke test pins the `DiarizeMethod` indexes, and the
c_api struct comment now lists every hand-written mirror to update on the next
append.

Still open from the issue's asks:

1. ~~**`crispasr_session_output_sample_rate()` (+ channels getters).**~~
   **DONE** (second #332 landing): `crispasr_session_output_sample_rate` /
   `input_channels` / `output_channels` in the C ABI (per-backend rate table
   mirroring the CLI adapters' `tts_sample_rate()`, 0 = no audio output),
   wired through Rust / Ruby / JS / Java / C# (the surfaces that expose
   `input_sample_rate`), pinned by `tests/test-session-abi-nulls.cpp` and
   documented in `docs/bindings.md` §"Session audio-format getters".
   **A new TTS backend must add its ctx to the getter's table** —
   `docs/contributing.md` §5b.3 records the duty. Go / Dart / Python don't
   expose `input_sample_rate` either; extend all four getters together there
   if anyone asks.
2. ~~**Same-benchmark DER for the pyannote+embedder path**~~ — **already
   existed**; the note here originally claimed the pyannote path had no DER
   on the shared benchmark, which was wrong. The cross-method table lives in
   the #326 NOW section below and in `docs/diarization-speakers.md` "#326":
   pyannote+embedder **7.81 %** vs foxnose **7.32 %** mean DER on the same 8
   VoxConverse dev files (whisper-tiny segments, 0.25 s collar), with the
   3.18 %-vs-7.32 % foxnose discrepancy explained there (own turns vs ASR
   segments as speech regions). Nothing left to run for #332; the estimator
   under-count remains #326's open accuracy item.

## NOW — #326 diarization: the count estimator is the last accuracy item

Landed: parallel 60 s chunked pyannote inference (e517273d), the `SPK_MASK`
powerset transposition worth 15 DER points (15aad6f8), and the over-clustering
fix (a719c89d). See `docs/diarization-speakers.md` "#326".

Measured end to end on the 8 VoxConverse dev files, whisper-tiny segments,
0.25 s collar, `tools/der_voxconverse.py`:

| path | mean DER |
|---|---|
| raw posteriors, no clustering (the chunking A/B harness only) | 33.37% |
| `--diarize-method pyannote --diarize-embedder auto` | **7.81%** |
| `--diarize-method foxnose` (#324) | 7.32% |

Reproduce either arm in one command:

    python tools/der_voxconverse.py --prepare <voxconverse>/data --audio-dir /tmp/vox
    python tools/der_voxconverse.py --audio-dir /tmp/vox --model ggml-tiny.bin \
        --args "--diarize --diarize-method pyannote --diarize-embedder auto"

### 2. OPEN — the speaker-count estimator now UNDER-counts

The cap no longer decides the count, but the BIC estimator that replaced it is
wrong on half the shard in the other direction:

    file    GT  hyp   DER%
    esrit    5    5    3.29
    fsaal    7    6    3.80   under
    jyirt    4    3    9.33   under
    mesob    4    4   16.90   <- worst file, count is RIGHT
    nnqfq    5    5    3.59
    rcxzg    4    4    9.71
    tiams    5    3    9.61   under
    willh    2    3    6.23   over

Two separate threads, and it matters not to conflate them:

  a. **Count.** 4 of 8 wrong, 3 of those under. HISTORY's estimator survey
     ("the estimator is not FRAGILE, it is BIASED") measured the same shape
     with margins of 6-14%, so this is not a tie-breaking problem and a
     silhouette tweak will not fix it. NME-SC was already tried and LOST
     (archived). Next candidate is calibrating the BIC penalty against
     embedding-window count, since the errors concentrate on short files.
  b. **mesob has the RIGHT count and the WORST DER (16.90%).** Nothing to do
     with counting — it is confusion within 4 correctly-estimated speakers.
     Diagnose separately; a count fix cannot touch it.

GATE for any count change: mean DER must beat 7.81% AND mesob must not regress.

## OPEN 2026-08-03 — stamp the kartoffel repos (needs ~15 GB free; this box has 11)

Unclaimed. Everything needed is built and proven; this is blocked on disk, not
on work.

**Do NOT stamp parler or csm.** Checked: the verdict table resolves `parler-tts`
and `csm` by BACKEND NAME (`crispasr_speaker_identity_models.h`), which is
already independent of the filename — renaming one changes nothing, so stamping
them moves 14 GB for no behavioural gain.

`kartoffel-orpheus-de-{natural,synthetic}` is the opposite case: one `orpheus`
backend serves several checkpoints, so the verdict keys on a filename substring
and a rename drops `natural` from real_person to unknown — silently losing an
Art. 50(4) disclosure. That is the case the stamp exists for.

**Why it is not done.** `stamp-speaker-identity.py` rewrites rather than patches,
so it needs source AND output live: 13.2 GB peak for the f16, 7 GB for the q8_0.
This machine has 11 GB on `/` and 5.9 GB on `/Volumes/backups`, and other agents
build here — filling the disk would take them down with it. Not worth a
rename-robustness gain.

**When there is room**, stream it one file at a time rather than
`snapshot_download`-ing 24 GB:

    for f in kartoffel-orpheus-de-natural-{q4_k,q8_0,f16}.gguf; do
      hf download cstr/kartoffel-orpheus-3b-german-natural-GGUF "$f" --local-dir src/
      python models/stamp-speaker-identity.py --input src/$f --output out/$f \
        --speaker-identity real_person \
        --evidence "card: fine-tuned primarily on natural human speech recordings; 19 speakers extracted from podcasts/lectures/OER"
      # verify tensors with a RAW BYTE comparison, then upload, then rm both
    done

The synthetic sibling takes `--speaker-identity synthetic` and the evidence
"card: trained on synthetic German speech with emotion and outburst control".

⚠ Verify with `.tobytes()`, not `np.array_equal` — the latter returns False
whenever NaNs are present and will call a byte-identical file corrupt.

**Scope narrowed after checking where the stamp actually buys anything.** The
verdict table resolves `parler-tts` and `csm` by BACKEND NAME
(`crispasr_speaker_identity_models.h`), which is already independent of the
filename — renaming one of those checkpoints changes nothing. Stamping them
would move 14 GB for no behavioural gain.

`kartoffel-orpheus-de-{natural,synthetic}` is the opposite case: one `orpheus`
backend serves several checkpoints, so the verdict keys on a filename substring
and a rename drops `natural` from real_person to unknown — silently losing an
Art. 50(4) disclosure. That is exactly what the stamp exists to fix, so those
two repos (24 GB) get it and the others do not. Everything needed is built and proven, this is bandwidth.

Stamp `crispasr.voice.speaker_identity` into the repos whose verdict is
established but which were left for size: `cstr/parler-tts-mini-v1.1-GGUF`
(real_person), `cstr/csm-1b-GGUF` (synthetic),
`cstr/kartoffel-orpheus-3b-german-{natural,synthetic}-GGUF` (real_person /
synthetic). Upload only, no runtime code:

    ./models/stamp-published-voices.sh <downloaded-dir>

It asks `crispasr --print-speaker-identity` per file and skips unknowns, so no
verdict is restated. Verify tensors are byte-identical afterwards with a RAW
BYTE comparison — `np.array_equal` returns False whenever NaNs are present and
will report a good file as corrupt.

## Why not the hash-chained log the sibling projects built

CrispTTS hash-chains its consent log (SHA-256 per line + a sibling `.anchor`
file, since a chain cannot detect truncation of its own tail; Art. 17 erasure
handled by re-chaining survivors plus a `[CHAIN-REBUILT]` marker). It is well
built. Porting it is still the wrong first move here, for four reasons:

1. **We would be chaining the wrong layer.** Our record identifies the voice by
   NAME (`voice=alice.wav`), never by content — there is no hash of the
   reference anywhere in the tree. That file can be swapped a minute later and
   the record still "verifies". A chain protects the SEQUENCE of records; if
   each record is an unbound assertion, a perfectly chained log of them proves
   nothing. CrispTTS's log is worth chaining *because* it already ties to a
   hash.
2. **The threat model does not support it.** This is a self-attestation BY the
   operator, who controls the binary, the file and the anchor. A chain does not
   defend against the party it is recording. It defends against a third party
   editing the file without the tooling — the narrow case. CrispTTS says
   "tamper-evidence, not tamper-proofing" and is right; the risk is readers
   hearing the stronger claim.
3. **Persisting more creates the liability.** CrispTTS had to build erasure,
   retention pruning and rebuild records BECAUSE they persist identifying data,
   then prove re-chaining cannot launder later edits. Every field stored is a
   field that must be deletable on request.
4. **Library vs application.** CrispTTS and Susurrus are applications. CrispASR
   is a library + CLI embedded by others (four bindings, an HTTP server,
   Wyoming). The right shape for an embedded component is to emit a clean
   structured record and let the host own durability; a bespoke chained log
   duplicates journald/CloudWatch/SIEM and is weaker than any of them.

Real tamper-resistance is a STORAGE decision — append-only permissions,
object-lock/WORM, shipping off-box — and the library must not pretend to
substitute for it.

### The work, in priority order

1. **`ref_sha256=` in every `[CONSENT]` line.** SHA-256 of the file the backend
   will actually open (`resolve_voice_path()` output), via the header-only
   `crispasr::sha` already vendored for C2PA. This is the change that turns the
   record from an assertion into evidence, and a hash carries far less
   data-protection weight than the recording or a name.
2. **Correlate the record to the output it authorised.** A per-process `run_id`
   in the `[CONSENT]` line and in the post-synthesis audit line, so a disputed
   clip can be walked back to the attestation. On the server add a request id —
   today a consent line and its output are unlinkable.
3. **`--consent-log <path>`, JSON Lines, default off.** Today the only route is
   redirecting stderr, which interleaves the record with model-load noise and
   progress output. A separable sink is what lets operators route it into
   infrastructure that IS append-only.
4. **Document the division of responsibility** in docs/eu-ai-act.md: the
   operator is the controller, this is their artefact, tamper-resistance is
   theirs to provide, and here is the field to key erasure on.

### Below the line, probably never in-library

Hash chaining. The one real case is the Docker server run as an appliance where
no host audit infrastructure exists and the operator wants to show a regulator
the log was not casually edited. Revisit only then, and only after (1).

⚠ Legal note: as read here, Art. 50(4) is a DISCLOSURE duty, which CrispASR
already discharges unconditionally through marking. The consent attestation goes
to GDPR lawful basis, where the accountability duty sits with the operator as
controller, not with the tool. Engineering judgement, not legal advice — put it
to counsel before it appears in a compliance claim.

## OPEN follow-ups from #300 / #308 (landed 2026-07-27, see HISTORY)

1. **C# and WASM bindings are source-only-verified.** No `dotnet` or emsdk on the
   Mac, so the `IntPtr`/`PtrToUtf8` marshalling in
   `bindings/csharp/CrispASR/NativeMethods.cs` and the two `emscripten.cpp` sites
   were not compiled. `bindings-csharp.yml` is the real check — watch that job.
2. **Three ASR backends still unaudited for `CAP_PUNCTUATION_NATIVE`**:
   `lfm2-audio`, `fastconformer-ctc`, `wav2vec2` — no local GGUFs. The CTC pair
   almost certainly needs the pass (unpunctuated by construction); `lfm2-audio`
   is an LLM decoder and is the likely one to need the flag. Method:
   `FIREREDPUNC_DEBUG=1 … | grep PUNCDBG` and read `in=` — do NOT use
   `--no-punctuation`, it strips after the fact and inverts the answer.
3. **Delete the duplicated fallback copies** (READY — this is "Ready to take" #1).

   Fourteen `.cpp`/`.h` files exist twice: once in `crisp_punc/src`,
   `crisp_lid/src`, `crisp_truecase/src` (what the shared libraries build, and
   what CrispEmbed consumes via `add_subdirectory`) and once in `src/` (what
   `src/CMakeLists.txt` builds when those directories are absent from a
   checkout). They are the same implementations twice.

   `tests/test-copies-in-sync.cpp` now makes the duplication *safe* — it byte-
   compares all 14 pairs and asserts the pair list is exhaustive — but safe is
   not the same as gone. Two copies had already drifted before that test existed
   (see HISTORY 2026-08-03), and #308's capitalisation fix was dead code for
   months for the same reason.

   **The task:** make the fallbacks unnecessary, then remove them.

   - Find out whether a partial checkout without `crisp_punc/`, `crisp_lid/`,
     `crisp_truecase/` is still a real distribution mode. `src/CMakeLists.txt`
     branches on their absence — check whether anything actually ships that way,
     or whether the submodules are always present now.
   - If nobody needs it: delete the `src/` copies, drop the CMake fallback
     branches, and delete `tests/test-copies-in-sync.cpp` with them. A guard for
     a hazard that no longer exists is dead weight.
   - If someone does need it: the copies must stay, and so must the test.
     Say who needs it, in this file, so the next person does not re-derive it.

   **Do not** simply delete the `src/` copies and see if the build goes green —
   the fallback branch only compiles when the shared directories are *missing*,
   so a normal build will pass either way. Test by moving the three directories
   aside and configuring from clean.


## OPEN — #316 Kokoro G2P English residue

Rounds 1-2 complete (HISTORY). Open, beyond the "carried out of #316" section:
- POS-dependent words (`that`/`read`/`used`/…): the tag-dependent remainder is
  0.34 %; the cheap slice is a closed-class rule for `that` (no spaCy port).
- misaki's reduced vowels `ᵊ`/`ᵻ` are not modelled (context rule needed).

## NOW — VAD + mel front-end parallelization campaign (#305 → fleet-wide)

Started from #305 (reporter: whisper-vad-asmr + firered-vad slow / single-core).
The recurring pattern: the model compute is fine (ggml threads / Accelerate BLAS),
but the **audio FRONT-END (STFT/mel/fbank) is a scalar single-threaded per-frame
loop** — and for long audio the mel can out-cost the encoder (measured in
whisper-vad). Parallelizing over the independent frame axis (per-thread FFT
scratch, reductions keep order) is **bit-identical** and a large win.

**DONE (per-backend, std::thread, gated + bit-identical):**
- whisper-vad-encdec — GPU move (3.3×) + parallel mel (~30% long-audio) + requant
  (b5eb7556 / 9f867909 / 8324719e). `CRISPASR_VAD_ENCDEC_*`.
- firered-vad — parallel DFSMN convs + fbank FFT, ~3.7× (8c1d6a10).
  `CRISPASR_FIRERED_VAD_SERIAL=1`.
- marblenet-vad — parallel mel front-end, ~1.6× (7b2f11bc).
  `CRISPASR_MARBLENET_VAD_SERIAL=1`.
- Audited silero (whisper.cpp native ggml, already threaded) + webrtc (subband
  GMM, no FFT) — no change needed.

**STATUS: the CPU front-end parallelization vein is MINED (2026-07-26).** Full sweep
done. Nothing clean+validatable-locally remains — do NOT keep sprinkling std::thread:
- `core/istft.h` (shared by outetts_wavtok/kokoro/cosyvoice3 +8 vocoders) is
  single-threaded BUT it is **overlap-add** — adjacent output frames write
  overlapping samples (data race, unlike the mel's disjoint writes), and n_fft is
  tiny for most consumers (kokoro=20, cosyvoice3=16) so the per-frame IRFFT is
  already cheap. Poor risk/reward — SKIP (would need per-thread out buffers + merge
  or a stride-coloring scheme for a marginal win).
  **2026-08-30 re-audit:** the "miocodec has a scalar FFT" line below was
  misattributed — `src/miocodec.cpp` contains no FFT of its own; its one
  `core_istft::istft` call sits behind `miocodec_extract_stage`, whose only
  caller is the diff harness (decode is a stub; `miotts.cpp:1770` calls
  core_istft directly). Measured miotts share: iSTFT ≈ 2.0% of synthesis CPU
  (n_fft=392/hop=98/T=1332 — 3.5 s of 176.6 s) — under the 5% gate, so still
  SKIP. For the record, a BIT-IDENTICAL parallel design DOES exist if a heavy
  consumer ever appears: parallelize only `irfft_hermitian` into a T×n_fft
  frame buffer (each frame self-contained, ~2 MB scratch at miotts sizes,
  ~99% of the stage) and keep the overlap-add strictly serial — the naive
  frame-parallel OLA is both a race and FP-non-associative, and per-thread
  buffers + ordered merge does NOT restore bit-identity, but the irfft split
  does by construction. Gate as `CRISPASR_ISTFT_SERIAL=1` if ever done.
- Own-mel backends NOT on core_mel (f5_tts, gemma4_e2b, titanet, chatterbox_s3gen,
  outetts_wavtok, ecapa_lid): their FFT runs ONCE on the reference clip or on TTS
  output where the DiT/decoder dominates — marginal fractions, not the
  per-long-audio bottleneck the ASR/VAD mel was. Not worth the churn.
- moonshine uses a shared mel helper (no local scalar loop of its own).
The remaining lever is TIER 2 (GPU ports) only — see below + the handover prompt.

**KEY FINDING that reframes the fleet rollout:** `core/mel.h::compute` (used by
~28 ASR/TTS backends: parakeet, canary, canary_ctc, nemotron, qwen3_asr, cohere,
glm_asr, granite_*, higgs_stt, ark_asr, voxtral/4b, moss_*, mini_omni2,
lfm2_audio, qwen3_tts, cosyvoice3, chatterbox, indextts, mimo, piano_transcription
…) ALREADY has a §176f parallel-STFT path — but it is **`#ifdef _OPENMP` only**,
and this macOS/AppleClang build has **`OpenMP_CXX_FLAGS=NOTFOUND`** (no libomp), so
it is **compiled out** on macOS (and any non-libomp build). That is why ZERO
backends set `allow_parallel_stft=true` — the flag is a no-op on the dev box. The
mel *projection* matmul in `compute()` is serial too (not even OpenMP-gated).

**TIER 1 — port core_mel STFT to std::thread (PORTABLE) — DONE (528d672f).** One
change to `src/core/mel.cpp`: portable std::thread STFT path (OpenMP kept when
`_OPENMP`), DEFAULT parallel with `CRISPASR_MEL_SERIAL=1` opt-out, threshold T≥256.
Bit-identical (disjoint power[] rows, same reduction order). The mel projection was
already Accelerate/BLAS-threaded, so the STFT was the sole single-threaded piece.
Validated parakeet-tdt-0.6b-ja (M1): STFT 1070-frame chunk 20.19→5.52 ms (~3.7×),
transcript BIT-IDENTICAL serial vs parallel. Lifts all ~28 core_mel backends.
Neutral for heavy-LLM-decoder ASR (mel is a tiny fraction); real win for
encoder-bound ASR (parakeet/canary/nemotron) and long audio. `allow_parallel_stft`
per-backend flag now redundant (kept for back-compat).

**TIER 2 — CPU-hardcoded backends that could go GPU (per-model, MAJOR effort — NOT
started; needs models + Kaggle validation).** `ggml_backend_cpu_init()` with no GPU
path, conv-heavy enough to benefit: **mel_band_roformer** (source sep — STRONGEST,
already promised as a GPU port under #296 below; but 1284 lines already
BLAS-optimized, model not local, and #296 was Kaggle-validated → a full
transformer+iSTFT ggml-graph port is a focused multi-hour project that must be
diff-harness + Kaggle validated, NOT doable+trustworthy locally), piano_transcription
(cblas=0 — not even BLAS yet; a cheaper first step is BLAS/threads before GPU),
pyannote_seg (already uses threads), openvoice2, TTS codecs (miocodec has a scalar
FFT — VAD-style parallelize; miotts/tada_encoder). Small classifiers (ecapa_lid,
lid_fasttext, bert_encoder, marblenet) stay CPU (launch-bound). Each needs a
ggml-graph port + Metal/Vulkan landmine handling + diff-harness validation. RECOMMEND
as a dedicated session per model (get the model, port, validate on Kaggle), starting
with mel_band_roformer (highest impact, already promised).

**TIER 3 — mined.** §176 runtime-opt campaign is 18/20 DONE; the 2 open are
<2% (measure-first). Do not chase.

## #296 mel-band-roformer — follow-ups PROMISED on the issue (do not drop)

`--separate` (mel-band-roformer) was ~24 min for 11 s on Linux/Windows (fast only
on macOS/Accelerate). Root cause: CPU-only forward with an Apple-only BLAS gate +
naive O(N²) iSTFT + scalar attention + redundant weight de-quant. FIXED on main to
**24 min → 56 s (~26×), cos=1.0** via: portable OpenBLAS `linear()`, FFT iSTFT,
attention→SGEMM + BLAS-thread-pin, per-layer weight hoist. Validated on Kaggle
(cos + per-stage profile each round). Replied on the issue
(`#296` comments 5062263388 / 5062697259).

**Explicitly PROMISED on the issue — must follow through:**
1. **OpenBLAS in the shipped binaries — AUDITED against the published v0.8.25
   artifacts, not the workflow source (`b0c548e7`).**
   - **Windows: DELIVERED.** `crispasr-windows-x86_64-cpu.zip` ships
     `openblas.dll` (1.78 MB) beside the exe. The reporter's platform has the
     fast path; this bullet's old "Windows CLI jobs set up NO BLAS" is stale.
   - **Linux: was BROKEN, and worse than a slow fallback.** apt's OpenBLAS is
     dynamic, so the binary carries `NEEDED libopenblas.so.0` while the tarball
     shipped only crispasr, crispasr-quantize and libc2pa_c.so — on a host
     without OpenBLAS it does not start at all (`error while loading shared
     libraries`). Read off the real tarball with objdump. Fixed by
     `scripts/bundle-openblas.sh` (the OpenBLAS half of `bundle-c2pa.sh`; RUNPATH
     is already `$ORIGIN`), wired into the x86_64 + arm64 Package steps.
   - **Both platforms could ship a degraded binary on a GREEN run** — the Windows
     vcpkg step is `continue-on-error` and CMake falls back to scalar silently.
     Both Package steps now assert the ARTIFACT and fail with `::error::`.
   ⚠ **UNVALIDATED until the next release run**: the bundling copy branch cannot
   execute on macOS. The no-op branch, the detection branch against the real
   shipped Linux binary, and the YAML parse were verified locally.
2. **GPU / ggml-graph port** (the real long-term fix, promised as "a GPU path is
   tracked"). Rewrite the forward (transformer + iSTFT) as a ggml graph → SIMD +
   threads + GPU everywhere, eliminating the scalar/BLAS/OpenMP scaffolding. Same
   Apple-only-BLAS pattern also slows **htdemucs** (the other `--separate`
   backend) on Linux/Windows — the ggml port is the template that fixes both.
   Needed for full-song latency (56 s/11 s still extrapolates to ~15 min/song).
   Validate with the mel-band-roformer diff harness (per-stage cos≥0.9995).

## #266 follow-up — hoist speaker orchestration into the library (PARKED, LOW)

The #266 rework (closed-roster, cluster-level speaker identification — DONE,
see `docs/speaker-db-clusters/PLAN.md` + HISTORY) left one parked item (F9):
the diarize → merge → global-cluster → identify orchestration lives in the CLI
(`crispasr_apply_global_speaker_stages()`, examples/cli/crispasr_run.cpp).
Session-ABI/bindings compose the primitives themselves; the server exposes no
speaker-db (deliberate). If a binding or the server ever needs the full named
pipeline, hoist the orchestration into src/ (the parakeet_orchestrate
pattern / multi-surface lesson) rather than duplicating it per surface. The
compliance invariants (consent + `--expect-speakers` closed roster + post-only)
must move with it — see `docs/diarization-speakers.md` §2.

Pending roadmap items. Each is self-contained with files, approach, and
effort estimate. Completed items have been moved to `HISTORY.md`.

> **Numbering convention:** `§N` refers to PLAN items (sections in this
> file). `#N` refers to GitHub issues on CrispStrobe/CrispASR. They are
> independent sequences and numbers may collide. When in doubt, PLAN
> items are always written as `§N` and GitHub issues as `#N`.

**Latest release: v0.8.32** (tag `v0.8.32`, + pub.dev `crispasr 0.8.32`). Release
notes live on each tag. Only the CURRENT version's `RELEASE_NOTES_v*.md` is kept
at the repo root — older ones are removed when the next is written, since each
tag carries its own copy and the GitHub release carries the published body.
`release.yml` hard-fails if the file for the tag being cut is absent.
The v0.8.18 → v0.8.20 train shipped in one session (2026-07-21): each patch was
a genuine fix that only surfaced on a real tag run — see "Recent completions"
and `LEARNINGS.md` "a green release job is not a shipped artifact".

**Recent completions (2026-07-22):**
- **#292 --max-new-tokens ignored** — SHIPPED. moss-diarize (and 9 more ASR
  backends) hardcoded the decode cap, so `--max-new-tokens` did nothing and a
  long single-pass truncated (reporter's 300 s file stopped at 164 s). Fixed
  across all 10: context `max_new_tokens` field defaulting to the old constant
  (no regression) + setter + cap/KV read it; CLI forwards only when
  `max_new_tokens_explicit`; session C-ABI forwards `s->max_new_tokens`.
  moonshine excluded (its 194 is an architectural short-form limit). Also added
  `chunk_id` to segments (part 2: diarize speaker labels are chunk-local).
  **CUDA-validated** on the reporter's exact backend (moss-diarize q4_k):
  `--max-new-tokens` 64→4096 raised output 216→366 words CPU / 232→382 CUDA,
  chunk_id 4 distinct, tabcnn CPU/CUDA parity 0 mismatches (6/6, ~18 min,
  `tools/kaggle/cuda-292-maxnewtokens/`). See `LEARNINGS.md` "a hardcoded decode
  cap" + memory [[kaggle-full-harness-regime]].

**Recent completions (2026-07-21):**
- **#290 canary-qwen long audio** — SHIPPED (v0.8.19). The backend declared
  `CAP_INTERNAL_CHUNKING` with no chunker (`src/canary_qwen.cpp` has zero
  chunking code vs 62 hits in parakeet.cpp), disabling BOTH dispatcher safety
  nets → one full-length encoder pass → O(T²) attention (384 MiB→10.2 GiB) and
  sparse output. Fix: drop the false capability. See `LEARNINGS.md`
  "a capability flag is a promise".
- **Lib-delivery bugs (CometBeat)** — SHIPPED (v0.8.19 rpath, v0.8.20 flat+gpu).
  6 of 7 lib bundles were unloadable as delivered: macOS baked the CI runner's
  build path into LC_RPATH; all 5 Linux bundles used `$ORIGIN/../../ggml/src`
  (one level too high). The old gate only checked deps were PRESENT, never that
  the loader could FIND them. New `tools/verify-lib-bundle.sh` relocates +
  dlopens; `tools/package-lib-bundle.sh` flattens to `lib/` + rewrites rpaths.
  See `LEARNINGS.md` "presence is not resolvability".
- **iOS shipping for the first time** — SHIPPED (v0.8.20). Added a
  `build-xcframework` job to release.yml (build.yml is tag-excluded by design);
  fixed `build-xcframework.sh` to include `libglint.a` in the combined archive.
- **#291 C# binding neglect** — SHIPPED (v0.8.20). VadSegments returned
  centiseconds while documented as seconds; added `CrispASR.Logging`; bound the
  7 task backends (tab/beats/chords/piano/pitch/separate/convert) in
  `SessionMusic.cs`; added the first-ever C# CI. C# was also missing from the
  contributing.md binding-parity list. See memory [[csharp-binding-neglect]].

**Recent completions (2026-07-17):**
- **Roadmap accuracy sweep** — audited the PLAN's OPEN/NOT-STARTED headers against
  the code; **~11 items were already shipped** but still marked open and are now
  corrected: §169 (qwen3-asr ChatML prompt), #128 (Piper TTS), §66 (pub.dev
  `crispasr 0.8.11`), python `_find_lib`, #60o (MTLBinaryArchive pipeline cache),
  §155 (CONV_TRANSPOSE_1D — all phases + Metal/Vulkan/CUDA col2im kernels), #58
  (MOSS-Audio-4B), #101 (OmniVoice), §229 (GGML_LLAMAFILE ON), plus §57/§106/§224/§247
  sub-items. See `LEARNINGS.md` "verify PLAN OPEN items against code".
- **#227 VAD boundary reuse** — SHIPPED (CLI `--vad-export`/`--vad-import` +
  server `vad_export`/`vad_import`); shared serializer, unit-tested.
- **#91 CLI parity** — `--offset-t`/`--duration` now honoured by every backend on
  both CLI + server (shared `core/audio_window.h`, unit-tested); `--print-confidence`
  fixed for non-whisper backends (was a parsed-but-dead flag).
- **#201 TADA on-the-fly voice cloning** — in-memory make-ref (no temp GGUF) on
  both the C-ABI/session and server/adapter surfaces, opt-in
  (`CRISPASR_TADA_WAV_CLONE=1`). Roundtrip gate pending (see below).
- **Docs** — new `docs/benchmarking.md` (§227 fair-measurement recipe); `-am`
  aligner aliases enumerated in `docs/cli.md` (§105).

**Recent completions (2026-07-11):**
- **#242 moss-diarize**: SHIPPED — joint ASR + diarization + timestamps, 0.9B model,
  diff harness 4/4 cos=1.0, GGUFs on HF, full 12-point checklist. See `HISTORY.md`.
- **#200 dots-tts PatchEncoder**: FIXED — added missing RoPE (theta=10K) + QK-norm;
  full pipeline now runs e2e (LLM → DiT → PEnc → vocoder → WAV), ASR roundtrip passes.
  Wired `--tts-steps` / `--tts-cfg-scale`. GPU already supported. See `HISTORY.md`.
- **#215e gallocr UAF audit**: DONE — all 19 `cached_*_gf` sites audited across
  the codebase. 7 backends fixed (canary, canary_ctc, kyutai_stt, moonshine_streaming,
  nemotron, paraformer, sensevoice). See `HISTORY.md`.
- **Generation-health gate**: DONE — `src/core/generation_health.h` with 5 checks
  + 16 unit tests. Non-breaking additive.
- **qwen3-tts-perf (#245)**: ANALYZED — profiled on CPU: dispatch overhead (build+
  reset+alloc = ~5ms) is <0.1% of per-frame cost (5000ms compute). The O15 path
  already caches the graph and uses a dedicated sched. Skip-realloc is broken on
  CUDA+Metal. The bottleneck is pure matmul compute; perf wins require GPU where
  the ~5ms overhead becomes significant. Handover removed.
- **Untrusted-input parser hardening**: SHIPPED — multi-agent security audit of the
  audio demuxers + GGUF loader found 6 DoS/OOB defects (MP4 stsz/stco/co64 count +
  co64 offset overflow, WebM lacing, WAV + AU size clamps, GGUF split mmap bounds),
  all fixed + ASan-validated. See `HISTORY.md` + `LEARNINGS.md`.

## Untrusted-input parser hardening — follow-ups

_Completed work archived to HISTORY.md (PLAN compaction 2026-07-17)._

**Still open:** Upstream the ggml empty-key fix to ggml-org (outbound public PR, left for a human)

**FIXED 2026-08-10 — the stb_vorbis SEGV `linux-fuzz-smoke` found.**
`vorbis_deinit` walked `comment_list_length` entries of a NULL `comment_list`.
`comment_list_length` is read straight from the file at
`examples/stb_vorbis.c:3660` and set BEFORE the array is allocated, so the
allocation-failure return one line later (an attacker-sized count makes
`setup_malloc` return NULL) left the two out of step. Reachable from the public
`crispasr_audio_load` on untrusted input.

⚠ An earlier CrispASR patch had already hardened the SIBLING path here — the
partially-filled array whose unassigned slots were freed as if valid — by
zeroing the allocation. It could not help when the allocation never happened.
Fixing the case you can see and leaving its neighbour is the recurring shape:
the guard now lives in `vorbis_deinit` (defends every path in, present and
future) rather than at the Nth caller, plus the length is reset on the error
path so the struct's invariant holds.

**Reproduced deterministically rather than waiting for the fuzzer.** The crash
input is 102 hand-crafted bytes — an Ogg page carrying a Vorbis ident header
and a comment header declaring `comment_list_length = 0x3FFFFFFF`. Kept as
`tests/fuzz/regressions/ogg-huge-comment-count.ogg`, and the smoke-fuzz job now
copies `tests/fuzz/regressions/` into its corpus, so every fixed crash becomes a
deterministic gate instead of a coin flip. Before/after on matching builds: 2
SEGV lines → 0.

⚠ `libcrispasr` is a SHARED library — the fuzz harness picks it up by rpath, so
"rebuild the harness" tests the new code with an old-looking binary. Rebuild the
dylib when doing a before/after, or the control silently becomes a second copy
of the experiment. (Cost me one invalid control here.)

The job also now uploads the crashing input on failure. It previously kept only
the stack trace, which for a stochastic job means the reproduction is gone.

**Noted, not fixed:** on macOS the same input flows past stb_vorbis into the
AudioToolbox fallback (`crispasr_at_decode` → `ExtAudioFileOpenURL`) and
libFuzzer reports an out-of-memory there. That is a resource limit inside
Apple's decoder on malformed input, not memory corruption, and that path does
not exist on the Linux CI. Worth a size sanity-check before handing a file to
AudioToolbox if anyone wants it.

## Diff-harness extensions (detail for "Ready to take" #5 and #6)

Both validate the TTS port against the Python reference. Neither is claimed.

### Diff-harness extension: per-step talker logits (#1, GPU-preferred)

**What:** Dump talker LLM logits at each generation step in both Python reference and C++ runtime, compare — validates the text-decoder input so the sampler is a faithful port over verified logits.

**How:** (a) add `talker_logits_step_N` capture to the Python reference dumper (`tools/reference_backends/qwen3_tts.py`) via a `generate`-time hook; (b) add matching C++ stage to `crispasr_diff_main.cpp`; (c) run on the TTS diff harness.

**C++ SIDE DONE 2026-08-10 (#337), and it settled the report.** Three levers,
which only work together:
`CRISPASR_QWEN3_TTS_GREEDY=1` (argmax), `CRISPASR_QWEN3_TTS_REPLAY_CODES=<file>`
(16 codec ids per frame — teacher forcing), and
`CRISPASR_QWEN3_TTS_DUMP_LOGITS=<dir>` (raw per-frame logits, before the
repetition penalty and suppress mask).

**Result: CPU vs Metal, fully pinned, worst cos 0.999870, mean 0.999940,
0/49 argmax disagreements.** Given identical history the backends agree at
every step, so the free-running divergence is entirely trajectory divergence
seeded by ~1e-4 arithmetic. That is the rigorous version of the #337 verdict;
the earlier free-running comparison suggested it but could not separate the two.

⚠ **Partial teacher forcing is a trap.** Replaying codebook 0 alone leaves the
15 residual codebooks sampled (`code_pred_generate_15` MUST sample — greedy
there is documented to produce silent output), so the per-frame input embedding
still diverges and the "teacher-forced" diff bottoms out at cos 0.849 — pure
artefact. Pin all 16 or measure nothing.

⚠ **Every per-synthesis dump must be tagged.** One `--tts` run generates twice
(the utterance, then the spoken AI disclaimer). `talker_%04d.f32` and
`generated_codes` were both frame-only names, so the disclaimer OVERWROTE the
utterance's dumps and a directory held two utterances with no way to tell.
Both now carry `_s%02d`. This produced one entirely bogus cross-backend table
before it was spotted — the tell was `argmax_gpu[k] == argmax_cpu[k-2]`.

**Python reference side DONE 2026-08-10.** `_hooks.capture_per_call()` written
(the module referenced an `_iter_capture` that never existed) — one capture per
call instead of first-call-only. `tools/reference_backends/qwen3_tts.py` now
emits `talker_logits_step0..15` and, at last, `generated_codes`: that stage had
been in DEFAULT_STAGES since the backend was written and was NEVER produced,
because the ids only exist inside `generate_voice_clone` (the outer call returns
audio). Captured by wrapping `tts.model.generate`.

Verified by running it: 17 tensors, `generated_codes (19, 16) int32` — exactly
the layout `CRISPASR_QWEN3_TTS_REPLAY_CODES` consumes — and
`talker_logits_step0 (1, 147, 3072)` = the PREFILL, with steps 1+ the AR steps
at `(1, 1, 3072)`. Note that indexing when diffing: step0 is not frame 0.

⚠ **Env:** the shared conda base has transformers 5.x; upstream Qwen3-TTS pins
4.57.3 and importing `qwen_tts` against 5.x dies on `check_model_inputs()`.
Do NOT downgrade the base. `tools/reference_envs/qwen3-tts/requirements.txt`
now scaffolds it; a `--system-site-packages` venv inherits torch and shadows
only transformers. `qwen_tts` comes from the clone at `~/code/Qwen3-TTS` via
PYTHONPATH, not pip.

**Prep landed 2026-08-10:** `qwen3_tts_sum_frame_embed()` factors the 16-codebook
embedding sum out of the AR loop so a harness entry point can build the SAME
per-step talker input without a second copy — duplicating it is how a harness
drifts from the runtime it checks, which is what #338 was. Verified
output-neutral: PCM byte-identical before/after (the WAV bytes differ only in
the C2PA/watermark metadata, which carries a per-run id — compare PCM, not the
container, when checking a refactor here).

**Still open, and here is the actual blocker.** A per-step talker input is
NOT just the codec-embedding sum:

    next_emb[step] = sum_{cb=0..15} embed_cb(frame[cb]) + trailing_text_hidden[step]

The `trailing` term is prompt-derived state computed from the synth text inside
the generate path. So a harness stage cannot simply prefill from the
reference's `talker_inputs_embeds` and then step — it also needs `trailing`,
which the reference does not currently dump and the runtime does not expose.
That dependency is why this stage does not exist yet; it is not just wiring.

Two ways forward, pick one before writing code:
1. Dump `trailing_text_hidden` as a reference stage too, and pass it into a
   `qwen3_tts_talker_logits_replay(embeds, n_tokens, codes16, n_frames,
   trailing, n_trail)` entry point. Most faithful, and it makes the
   dependency explicit in the archive.
2. Have the runtime construct the whole prompt itself from the same text +
   voice wav the reference used, and use the reference's
   `talker_inputs_embeds` ONLY as a structural gate (cos ≈ 1 before trusting
   any logits — the guide's input-alignment rule). Less plumbing, but it
   assumes the two prompt builders agree, which is the thing being tested.

Option 2's gate is worth having either way.

The superseded note: `CRISPASR_QWEN3_TTS_DUMP_LOGITS=<dir>`
writes raw per-frame talker logits (f32), dumped BEFORE the repetition penalty
and the suppress mask so a diff isolates the forward from the sampling policy.
It already paid for itself — it is what proved the reported "GPU miscompute"
was cos 0.99992 at frame 0, i.e. ordinary backend arithmetic amplified by the
AR loop. What remains is (a): the Python reference hook, which turns a
cross-BACKEND comparison into a cross-IMPLEMENTATION one against the blueprint.

**Test:** needs a TTS model (qwen3-tts or tada). 0.6B Q8_0 (941 MB) fits on VPS; TTS gen slow on CPU (~105x RTF) — use short "Hi." input, 2-3 frames.

### Diff-harness extension: replay-token dual-mode (#3, VPS-doable)

**What:** Dump Python's *sampled* token IDs and replay them in C++ (instead of re-sampling) so sampling-enabled downstream stages diff deterministically despite torch-vs-mt19937 RNG mismatch.

**How:** (a) Python dumper captures `sampled_token_ids` as a 1D int32 tensor in the reference GGUF; (b) C++ diff harness reads them and feeds the backend's step function instead of sampling; (c) compare downstream stages (codec, vocoder) vs the Python reference that used those tokens.

**Files:** `tools/reference_backends/<tts_backend>.py` + `crispasr_diff_main.cpp`.

## CometBeat handoff — singing-voice-conversion vocoders (OPEN, NOT STARTED)

Requested by the CometBeat `opus` (voice-svc) agent via its `docs/PLAN.md`
coordination note (read 2026-07-20). CometBeat is splitting a singing-voice-
conversion stack into pure-Dart (lightweight/offline) and native-via-CrispASR
halves. **We own the real-time-critical heavy vocoders**; they keep the
HuBERT/ContentVec encoder, Harvest F0, and a lightweight DDSP-SVC synth as the
web/offline fallback.

### §CB1 — RVC voice conversion — PORTED AND VALIDATED (packaging left)

Converter -> numpy spec -> ggml graphs -> convert() -> session C ABI, every
step at cos 1.00000000 against torch. `crispasr-diff rvc <model> <ref> <wav>`
reports 48 stages, including `convert_e2e` which runs the real
`rvc_svc_convert()` and reproduces the reference audio from ContentVec + F0 +
speaker id + replayed noise alone. Live test: 4 cases / 12825 assertions.

Deliberately NO CLI verb — the input is ContentVec features, which we do not
produce. `docs/bindings.md` documents the session surface.

ARTIFACTS: stored in the **PRIVATE** repo `cstr/rvc-svc-GGUF` (verified
`private: True` server-side before upload) — `rvc-40k-f32.gguf` (105 MB) plus
`rvc-ref.gguf` (23 MB), the 60-stage parity reference. That reference is the
valuable half: `crispasr-diff rvc <model> <ref> <wav>` re-runs all 48
comparisons with **no torch, no checkpoint and no RVC repo**.

REMAINING (packaging, not correctness):
- Registry entries. BLOCKED, and deliberately so on two counts: the checkpoint's
  licence is unscoped (converted `--license other`), AND the repo is private, so
  a registry URL would fail auto-download for anyone without auth. Scoping the
  base checkpoint's terms and re-stamping the tag comes first.
- F32 ONLY. An f16 GGUF converts but the graph is F32-only today (ggml_scale is
  F32-only; an F16 operand reaching ggml_add trips
  `GGML_ASSERT(src1->type == GGML_TYPE_F32)`). It REFUSES rather than producing
  subtly wrong audio, and is noted as such in the header.
- Dart/Flutter wrapper for `crispasr_session_convert*` (CometBeat is the
  consumer; the C ABI they need is done).
- Two agreed parameters were CORRECTED rather than implemented — `protect` is
  provably inert without the FAISS index, and `rms_mix_rate` needs the source
  waveform our seam never receives. See SVC_RECORD_SHAPES §9b.

### §CB1-history — original framing and the two findings that reshaped it

Contract CONFIRMED (see `docs/music-transcription/SVC_RECORD_SHAPES.md`).
Source traced; see `docs/music-transcription/RVC_BLUEPRINT.md`. Two findings
change the job before any code:

1. **The ask is ~3x bigger than "the vocoder".** The seam CometBeat wants
   (features + F0 + speaker -> audio) is `SynthesizerTrnMs768NSFsid.infer()`
   (`models.py:664`), which runs a **transformer encoder** (`enc_p`, with a
   `nn.Embedding(256, ...)` on the coarse pitch), a **normalizing flow**
   (4x coupling + flip, reversed), AND the NSF vocoder. Only the third is
   HiFi-GAN. **Ask CometBeat whether they want all three or only `dec`** — if
   only `dec`, they must send `z`, not ContentVec features, which changes the
   contract we just froze.
2. **Inference is STOCHASTIC, at two independent sites**: the latent sample
   `z_p = m_p + exp(logs_p) * randn * 0.66666` (`models.py:684`), and SineGen's
   random initial phase `rand_ini` (`:325`) plus additive noise (`:358`). So
   output is not reproducible run-to-run, waveform correlation is an invalid
   acceptance test, and the diff harness MUST replay the reference's noise
   (same pattern as `input_feat` in btc / `input_audio` in mel-band-roformer).
   Settle this before the graph, not after.

Numerical hazards already spotted (phase `cumsum` drift in f32, `F.interpolate`
mode, `upp = prod(upsample_rates)` derived not configured, `noise_convs`
stride schedule) are itemised in the blueprint.

### §CB1-old — RVC NSF-HiFi-GAN generator (original framing)

**What:** port the RVC NSF-HiFi-GAN generator to ggml/native-FFI. CometBeat
feeds us ContentVec features (from their `hubert.dart`), F0 (from RMVPE —
already done on their side) and a speaker id; we return converted audio.

**Seam:** they expect a `CrispasrSession.convert(...)`-style entry point. That
is a THIRD task-shaped surface after `--separate`/`--pitch`/`--chords`, so it
follows the same pattern: its own session entry points rather than riding on
transcribe(), plus the CLI dispatcher and the wasm/Go arms. Note the
`crispasr_detect_backend_from_gguf` trap — register the arch there too, not just
in the CLI (see the BTC entry in `docs/music-transcription/PLAN.md`).

**Blocking coordination — DRAFTED, awaiting their reply:** see
`docs/music-transcription/SVC_RECORD_SHAPES.md`, a concrete proposal for every
record shape (layout, dims, frame rate, F0 units, unvoiced encoding, speaker id,
return PCM) with reasons, so it can be accepted or amended rather than discussed
in the abstract. Items we have not yet verified against the RVC reference are
marked [UNVERIFIED] and must not be built against. The sharpest open question is
§3: who resamples F0 onto the feature timebase. The feature/F0 record shapes
must be agreed with the opus agent BEFORE their API freeze. Pin down, in writing: ContentVec feature
rate + dimensionality + dtype, F0 units (Hz vs cents vs MIDI) and hop, whether
F0 is voiced-masked, speaker-id encoding, and the sample rate of the returned
audio. Do this first — it is cheap now and expensive after the freeze.

**Licence:** RVC's own code is MIT, but the WEIGHTS in circulation are a mess
(many community models are of unclear provenance, and some RVC forks carry
non-commercial terms). Scope licences per-checkpoint before shipping any
registry entry, exactly as the music-transcription scoping pass did.

### §CB2 — Beatrice v2 (low-latency voice conversion)

**What:** port Beatrice v2; its low-latency design suits the native path.

**Licence: MIT — this entry previously said "custom/NON-COMMERCIAL" and that was
wrong.** `fierce-cats/beatrice-trainer` ships `LICENSE` = MIT (Copyright (c) 2024
Project Beatrice), and its README states in terms that *"このリポジトリ内の
ソースコードおよび**学習済みモデル**は MIT License のもとで公開されています"* —
the source **and the trained models**. So **no acceptance gate is needed**; a
`cc-by-nc-sa-4.0`-style tag here would have withheld permission the licence
actually grants.

Two real non-MIT signals exist but neither applies to this port: `beatrice.lib`
(the closed inference engine used by the VST "under permission") is irrelevant
because we port from the MIT source, not that binary; and the "Beatrice JVS
Corpus Edition" carve-out is a *different distribution*, not this repo.

**Feasibility: CONFIRMED — architecture and weights are both published.** An
earlier WebFetch summary of this same repo claimed it held "training scripts
only, not inference code or the model architecture". That summary was false;
`beatrice_trainer/__main__.py` is 4519 lines and defines the entire path
(`PhoneExtractor`, `PitchEstimator`, `VectorQuantizer`, `ConverterNetwork`,
`Vocoder`). Another instance of [[blueprint-summary-is-not-the-source]] — the
file listing settled in one call what the summary got backwards.

Weights (`assets/pretrained/`), all MIT, are **split per component** — the
obvious "load the checkpoint" assumption fails:

| file | contents |
|---|---|
| `122_checkpoint_03000000.pt` (14.7 MB) | `phone_extractor` **only** |
| `104_3_checkpoint_00300000.pt` (7.1 MB) | `pitch_estimator` **only** |
| `151_checkpoint_libritts_r_200_02750000.pt.gz` (153 MB) | `net_g` (177 tensors, the multi-speaker LibriTTS-R base) + `net_d` (discriminator, training-only → skip) |

**Scope is larger than §CB1, and the wire contract is different.**
`ConverterNetwork.forward` takes **raw waveform** (`x: [batch, 1, wav_length]`,
plus `target_speaker_id`, `formant_shift_semitone`, optional
`pitch_shift_semitone`) and returns 24 kHz audio. Beatrice therefore owns phone
extraction and pitch estimation itself — unlike RVC, it needs **no ContentVec
from the caller**, which simplifies CometBeat's side but means porting three
networks rather than one.

Non-obvious details to verify while reading (each a silent bug if assumed):

* `CausalConv1d` / `WSConv1d` — **weight-standardised** causal convs, not
  standard `nn.Conv1d`. The standardisation is part of the forward pass.
* `VectorQuantizer` is injected into `phone_extractor.head` via a **forward
  hook** (`enable_hook`). This session already lost time twice to hooks that
  silently never fire; the numpy spec must confirm the hook is active by
  asserting the quantised path *changes* the output, not by trusting it ran.
* `embed_quantized_pitch` is a **fixed sinusoidal** table (built in `__init__`,
  `requires_grad_(False)`) — it may or may not be present in the checkpoint, so
  the converter must rebuild it rather than assume it was saved.
* `key_value_speaker_embedding` is initialised with every speaker row **copied
  from row 0**, so speakers look identical until trained — an untrained-looking
  A/B is not necessarily a port bug.
* `self.melspectrograms` is **loss-only**; it is not on the inference path.
* Output rate is hardcoded 24000 with `hop_length = 24000/100` — the same 100 Hz
  frame rate §CB1 uses, so the record shapes carry over.

**Progress:** two of three components ported and validated.

| component | state |
|---|---|
| `PitchEstimator` | **DONE** — `crispasr-diff beatrice`, 30 stages + e2e, 0 failed |
| `PhoneExtractor` | **DONE** — `crispasr-diff beatrice-phone`, 69 stages + e2e, 0 failed |
| `ConverterNetwork` + `Vocoder` | blueprint read done; converter and graph NOT started |

**Next step:** the ConverterNetwork/Vocoder port. It is the largest of the three
and differs in kind from the two frozen extractors:

* The vocoder is a **source-filter / impulse-response synthesiser**, not a
  HiFi-GAN — a 512-tap IR per frame plus aperiodicity and post-filter, driven by
  `overlap_add`. That is why Beatrice is real-time-cheap.
* It uses **`WSConv1d`/`WSLinear`**, which neither ported component does, so the
  unbiased-variance trap (`torch.var_mean` is ddof=1, numpy defaults to 0 →
  uniform ~20 % weight mis-scale) becomes live for the first time.
* Its attention is **`CrossAttention`** over the speaker embedding, not
  `nn.MultiheadAttention`; the PhoneExtractor's `mha_subsequence()` does not
  transfer.
* `overlap_add` draws a **random initial phase**, so this component needs §CB1's
  injectable-noise discipline — and the injector must patch `torch.rand`, not
  just `randn_like`.
* The **lookahead alignment** (energy shifted 1 frame, quantised pitch and pitch
  features 2, all reflect-padded) is the highest-risk silent bug: getting it
  wrong misaligns content against pitch by 10–20 ms and no input-aligned
  per-stage check can see it.

Details in `docs/music-transcription/BEATRICE_BLUEPRINT.md`.

**Effort:** unestimated until the blueprint read is done, but larger than §CB1
(three networks, custom conv variants). Do NOT start before §CB1's record shapes
are agreed.

## Gemma-4 12B (gemma4_unified) ASR support (OPEN)

The remaining open item for full 12B support (a new converter map + backend audio path for the 640-dim unified
encoder) is a larger port, scoped but not started.

## crispasr-diff harness extensions — catch decode-policy / quality bugs (OPEN)

Status: config/adapter-parity guards DONE (`tests/test-tada-params.cpp` defaults-audit
now covers library + CLI + c_api; full ~40-adapter sweep clean, cosyvoice3/f5-tts session-config
bugs fixed). Generation-health header + unit tests DONE. Three extensions + one generalisation remain.

TODO (partial — status verified 2026-07-17):
1. **Per-step talker logits in the diff — DONE for the qwen3_tts exemplar.**
   `tools/reference_backends/qwen3_tts.py` captures `talker_logits` + `cp_step{0..14}_logits`;
   `_iter_capture.py` documents per-step talker_logits. Generalise to other TTS backends if/when
   a second consumer needs it.
2. **Wire generation-health checks into backends' live tests — still OPEN.** Shared header
   `src/core/generation_health.h` (check_not_empty / duration_plausibility / no_ngram_loop /
   not_truncated / tts_duration / trailing_silence) + `tests/test-generation-health.cpp` are
   done; still need per-backend live-test integration (needs models).
3. **Replay-token dual-mode reference — PARTIAL.** Noise-replay infra exists
   (`_iter_capture.py` writes `noise.bin` for C++ to replay); the *sampled-token* dual-mode
   (replay Python's argmax picks instead of re-sampling) is the remaining piece.
4. **Generalise the defaults-audit pattern across backends.** Per-backend table of
   (param → upstream default) checked against the params struct, so "knob declared but dead /
   default diverges from upstream" fails CI everywhere.

---

## #201 follow-up — generate a TADA voice ref from audio+transcript at query time (C-ABI + server DONE gated; roundtrip pending)

Switch-voice, offline `--make-ref`, `--align`, and CLI query-time inline cloning
(`--tts "…" --voice sample.wav --ref-text "…"`) all shipped.

**C-ABI / session half — DONE (opt-in, `feat/tada-201-server-abi`).** In-memory
make-ref, no temp GGUF:
- `tada_set_prompt_values()` — in-memory counterpart of `tada_load_prompt`
  (the latter now reuses it, so file + in-memory paths are identical).
- `tada_make_ref_from_pcm()` in `src/tada_tts.{h,cpp}` — `tada_encoder_encode`
  (validated) → `tada_set_prompt_values`. Provably equivalent to
  `write_ref_gguf()+load_prompt()` (same tensors), so no new graph math.
- `crispasr_session_set_voice(s, "ref.wav", "<transcript>")` decodes to 24 kHz,
  resolves encoder + language-matched aligner (explicit → next-to-model → cache),
  and bakes the prompt. `crispasr_session_tada_set_makeref_models()` sets the
  GGUF paths; Python `Session.set_voice(path, ref_text)` already routes here.
- **Gated default-OFF: `CRISPASR_TADA_WAV_CLONE=1`** — without it a `.wav` voice
  keeps the historical `-2` reject, so default behaviour is byte-identical.

**Remaining before flipping the gate on by default:**
- **Decoded-output roundtrip (HARD RULE #3)** — synth reference → set_voice(wav,
  text) → synth clone → ASR + `speaker-cosine(clone,ref) > cosine(baseline,ref)`
  via the Python `Session` API on `tada-1b` (+ `tada-encoder-f16.gguf` +
  `tada-aligner-en.gguf`). Not run on the dev box (memory-pressured).

**Server / adapter half — DONE (opt-in, same gate).** The HTTP server uses the
backend *adapter* (`crispasr_backend_tada.cpp`), a distinct surface from the
session C-ABI:
- `TadaBackend::clone_from_wav()` — gated (`CRISPASR_TADA_WAV_CLONE=1`) helper
  called from both `init()` (first voice) and `apply_request_voice()` (the
  server's per-request switch). Resolves the WAV against `--voice-dir` (bare
  name → `<dir>/<name>.wav` + companion `<dir>/<name>.txt` for ref-text, the
  qwen3-tts convention), resolves encoder + aligner (explicit → next-to-model →
  cache → auto-download), decodes to 24 kHz, and applies via
  `tada_make_ref_from_pcm` — no temp GGUF. Off/failed → the historical reject.
- `/v1/audio/speech` gained a `ref_text` body field (→ `tts_ref_text`). The
  existing `consent_attestation` gate already fires for a `.wav` voice.
- Docs: `docs/server.md` (ref_text field + updated tada voice note).

**Still OPEN (optional):** cache baked ref keyed by (audio hash, transcript) to
skip re-running the aligner on repeat requests.

**Files:** `examples/cli/crispasr_backend_tada.cpp`, `examples/cli/crispasr_server.cpp`,
`src/tada_tts.{h,cpp}`, `src/tada_encoder.*`. Aligner is language-specific
(`tada-aligner-<lang>.gguf`) — must match audio language. CLI helper
`tada_run_aligner_pipeline` is the reference implementation.

**Effort:** Medium-large. Pipeline is CLI-proven; work is server lifecycle +
per-request wiring + caching + the ~1.3 GB memory gate. Lower priority than the
shipped switch-voice half. Tracked on #201.

## §192 follow-up — native-Vulkan TADA garbled output FIXED (codec on CPU)

_Completed work archived to HISTORY.md (PLAN compaction 2026-07-17)._

**Still open:** GPU-native codec on RADV / chunked-decode design (deferred, conditional on a RADV user needing it)

## Cross-platform (Linux/x86 CPU) validation — DONE (green); follow-ups open

x86 CPU-only Linux validation of moss-transcribe / higgs-stt / ark-asr passed at
`dcc7e47b` (all verbatim, beam == greedy, Go LDFLAGS drift clean). Audit tool
`tools/check-backend-wiring.py` (`ccc04a02`) ships; 49 canonical PASS required.

Follow-ups (LOW, not blocking):
- [ ] Fix handover to `cmake --build build` (all targets) before `ctest -L unit` —
  VPS run only built `crispasr`/`crispasr-diff`, ran 2 unit tests. Or have VPS build all.
- [x] Install Go toolchain on VPS — DONE 2026-09-02: go1.23.4 at
  `/mnt/volume1/go-toolchain/go`, symlinked as `~/.local/bin/go` (on PATH).
- [ ] Optional: promote to a standing post-push Linux smoke (Routine/cron).
- [x] Orphaned ctest-label audit (2026-09-02, follow-up to the dead-`cli` find):
  cross-checked every `LABELS` in tests/CMakeLists.txt against every `-L` filter
  in .github/workflows + ci/. ONE real orphan found and fixed: `ci/run.sh` ran
  `ctest -L main` (upstream whisper.cpp's label — nothing here carries it), so
  both build.yml CPU jobs tested NOTHING while green; now `-L unit`, which also
  gives the unit suite its only Debug-build run. Labels that exist but are
  deliberately not CI-run (model-gated, local/live only): `live`, `base`,
  `small`, `medium`, `large`, `tiny;en`, `benchmark`, `integration`,
  `espeak;piper;tts` — do not "fix" these.

Multilingual + beam spot-checks (LOW, either machine):
- [ ] **#421 arm64 HiFT SIMDCONV failures** — the `-L unit` fix's first full-ci
  dispatch ran 1782 tests on arm64 (previously ZERO) and 4 fail: chatterbox +
  cosyvoice3 SIMDCONV scalar/SIMD identity (tests/test-*-hift-simdconv.cpp:66/68,
  instant assert). Possible silent vocoder divergence on arm64 Linux/Android.
  Unclaimed; details + log pointers on the issue.
- [x] moss-transcribe multilingual check — RESOLVED 2026-09-02, premise was
  WRONG: the upstream card says "intended for English automatic speech
  recognition" (this line had conflated it with the Diarize sibling). Ran the
  clips anyway (`samples/paraformer_zh.wav` + a bananamind-de synthesized
  German sentence — `audio_samples/` never existed): both come back as rough
  ENGLISH translations at rc 0, faithful to the model (prompt mirrors the
  upstream processor, instruction-less). Actioned: moss-transcribe now
  declares `sole_language()="en"` on the adapter + the session guard table,
  so `-l zh` is an explicit pre-dispatch rejection instead of silently wrong
  output, and `-l auto` short-circuits without an LID download (#227).
- [x] higgs/ark `-bs 4` noisy-clip beam test — DONE 2026-09-02 on Kaggle T4
  (`tools/kaggle/beam-noisy-ab/`, ${KAGGLE_ACCOUNT}/crispasr-beam-noisy-ab-higgs-ark v2):
  jfk + additive white noise at 10 dB and 5 dB SNR (seed 42), both backends,
  greedy vs `-bs 4`. Result: **WER 0.000 in every cell** — even 5 dB noise
  doesn't dent either model on this clip, and beam-4 wins nothing while
  costing ~3x wall (higgs 2s->7s, ark 4s->12s per clip). Greedy stays the
  right default. Residual (only if someone cares later): white noise is not
  accent — a FLEURS accented-speech rerun would need new fixtures; the
  harness takes any wav list.

Backend-wiring coverage gaps (LOW cleanup; re-list via `python tools/check-backend-wiring.py`):
- [x] **missing reference dumper** — RESOLVED by 2026-09-02: all five now exist
  in `tools/reference_backends/` (`fastconformer_ctc.py`, `wav2vec2.py`,
  `m2m100.py`, `kyutai_stt.py`, `gemma4_e2b.py`). Original caveat kept for
  the record (m2m100 text-only MT, gemma4-e2b
  shares gemma path, encoder components diff via host backends) — confirm per-backend
  before adding, not a blanket gap.

---

## moss-transcribe follow-ups (OPEN, LOW)

`moss-transcribe` backend (`OpenMOSS-Team/MOSS-Transcribe-preview-2B`) shipped
`9f3c5ede` — q4_k verbatim on jfk.wav, validated vs PyTorch ref via `crispasr-diff`.
#218 (greedy n-gram loop collapse + 30 s-seam dup) is fixed. Remaining optional work:

- **Publish f16 + q8_0** to `cstr/MOSS-Transcribe-preview-2B-GGUF` (q4_k + card live;
  f16/q8_0 held back for WLAN bandwidth). Re-stage from
  `/Volumes/backups/ai/moss-transcribe-preview-2b-{f16,q8_0}.gguf`, `hf upload-large-folder`
  into the existing repo. Both already produce the verbatim transcript locally.
- **GPU validation beyond Metal.** Metal (default) + CPU both verbatim; CUDA/Vulkan
  untested. LM reuses `core_attn::kv_self_attn` (covered by §192/#200 Vulkan F16-GQA guard);
  check the encoder's windowed `flash_attn_ext` + conv front-end on CUDA.
- **Multilingual eval.** Authors report 4.87 % avg WER; only English (jfk) validated here.
  Model is zh/en — spot-check a Chinese clip.

Note: encoder/adapter run F16 (cos ~0.98 vs f32 ref, byte-exact at layer 0 — pure F16
weight precision, not a bug). f32 encoder path not worth it since decode is verbatim.
Loop-fix opt-out: `CRISPASR_MOSS_TRANSCRIBE_NO_LOOPFIX=1`.

---

## #218 qwen3-asr long-audio root cause — quantized audio tower + prompt contract

_Completed work archived to HISTORY.md (PLAN compaction 2026-07-17)._

**Still open:** Broader eval to flip windowed-attn default / declare CAP_UNBOUNDED_INPUT (mechanism ready)

## #218 glm-asr long-form blueprint parity — instruction + multi-window

_Completed work archived to HISTORY.md (PLAN compaction 2026-07-17)._

**Still open:** Document that single-pass blueprint skips quiet leading audio (chunked covers more) in README

## #218 qwen3-family rebake + CUDA validation — kernel results

_Completed work archived to HISTORY.md (PLAN compaction 2026-07-17)._

**Still open:** Investigate mega-asr long-form at 4-bit vs bf16 blueprint (candidate follow-up)

## Priority ordering

| Priority | Item | Effort | Status |
|---|---|---|---|
| **HIGH** | [#221 Issue #89 hardening + v0.8.8](#221-issue-89-hardening--v088-release) | Medium | 5 steps: CI regression guard (a), server-path mirror (b), Vulkan sanity (c), q4_k registry/UX (d), release (e). |
| **DONE / LOW** | [§176 Runtime optimization pass](#176-runtime-optimization-pass--2026-06-20-audit) | Phased | 18/20 done. **2 OPEN (low-value, measure-first):** §176c device-resident KV (Dia measured ~1.2% of decode → DEFERRED; compute-bound decoders make this <2%), §176l Kyutai RVQ (genuinely scalar, but no local model to validate). Do not treat as HIGH. |
| **MEDIUM** | [#52 Qwen3-TTS](#52-qwen3-tts) — perf pass | Medium | talker + code_predictor + codec + ECAPA + codec_encoder done; step-4 perf pass open (~137 ms/frame → real-time). **O15 broken on CUDA and default-OFF** (`61c42bfb`) — main perf lever disabled; root cause is `ggml_set_rows` KV scatter or fixed-Lk causal mask on CUDA (crash on first code_pred call). Baseline O15=OFF: 27.4 ms/frame, WAV OK. |
| **MOSTLY DONE** | [#57 Commercial-friendly TTS expansion](#57-commercial-friendly-tts-backend-expansion) | Phased | Phases 1–3 + Turbo + native voice cloning shipped; #83 S3Gen fix landed. VoxCPM2, kugelaudio, gwen-tts, kartoffelbox-turbo, CosyVoice3 all shipped + registry-wired (verified 2026-07-17). **Remaining:** only the Darwin-TTS-1.7B-Cross / AMAImedia Qwen3-Darwin family unported. → HISTORY §82, upstream-prs/09–11. |
| **MEDIUM** | [#51c MiMo-V2.5-ASR F16 step decode](#51c-f16-step-decode) | Small | F16 step-decode validation blocked behind ≥32 GB box. Base runtime + Q4_K shipped → HISTORY §56. |
| **MOSTLY DONE** | [#58 MOSS-Audio-4B-Instruct](#58-moss-audio-4b-instruct) | Large | Runtime + GGUFs shipped, diff cos≥0.999, Kaggle P100 CUDA PASS. **Remaining:** flash-attn encoder, sweep transcript-extraction fix. → see HISTORY. |
| **MOSTLY DONE** | [§221 TADA encoder `--make-ref`](#221-tada-encoder---make-ref) | Medium | C++ encoder runtime + GGUF converters + diff harness shipped; GGUFs at `cstr/tada-encoder-GGUF`. **WIP:** cos_mean=0.94 parity (F16 precision), C++ BPE tokenizer for end-to-end `--make-ref`. → HISTORY §221. |
| **LOW** | [#95 IndexTTS Chinese TN binary alternative](#95-indextts-15-chinese-tn--binary-alternative-to-the-python-wetext-hook) | survey only | Python `INDEXTTS_TEXT_NORMALIZER` hook shipped. Hand-roll (#95a) is next *when* a user reports a digit/date prompt that breaks; OpenFST vendoring (#95b) only after #95a grows past ~5 cases. |
| **IN PROGRESS** | [#97 More Parakeet variants](#97-more-parakeet-variants) | Small per-variant | TDT/TDT+CTC + rnnt 0.6b/1.1b DONE. **parakeet-unified-en-0.6b** surveyed: 24L 600M Unified-FastConformer-RNNT, NOT converter-only — 8× subsampling (vs 4×) + Dynamic Chunked Convolutions are new; ~80% overlap with #81. Offline mode may work through existing converter; realtime-EOU blocked on #81 cache-aware streaming. |
| **LOW** | [#106 TEN-VAD](#106-ten-vad--low-latency-cross-platform-vad) | Small | Feasible VAD backend (C-compatible, 16 kHz / 10-16 ms frames, prebuilt libs + ONNX). License is the gate: Apache 2.0 plus extra no-compete / own-app-only conditions from Agora. |
| **MOSTLY DONE** | [#114 Long-form transcribe chunking-default ladder](#114-long-form-transcribe--make-chunkingstreamed-the-default-for-all-asr-backends-issue-89-follow-up) | Medium | Chunking/streamed-default shipped for all ASR backends + per-chunk AED re-injection + LCS dedup + word-snap. **Remaining:** EN-FLEURS retokenization artifacts (out of scope). → see HISTORY. |
| **MOSTLY DONE** | [#125 multi-backend bug sweep from montvid](#125-issue-125--multi-backend-bug-sweep-from-montvid-12-findings) | Medium | 12 findings; P1–P6b all DONE. **Remaining:** P0 Blackwell retest; mimo-asr `-np` empty-transcript retest. → see HISTORY. |
| **LOW** | [#127 Coverage gaps from 2026-05-26 sweep](#127-coverage-gaps-from-the-2026-05-26-overlap-save-sweep-close-out) | Small | (a) omniasr-llm DONE; (c) cohere-asr-ja DONE (repo `CKHO/cohere-asr-ja-GGUF`) — still needs JA fixture sweep for PERFORMANCE.md table. **(b) OPEN:** mimo-asr local test doesn't run in CI (4.2 GB Q4_K doesn't fit runner disk). |

**Open follow-ups from §79:**
- **#73 cohere long-form rerun.** flash_attn_ext shipped on canary + cohere (`193a736`). On JFK (~11 s)
  canary q8_0/q4_0 −17% under flash (win) but cohere q8_0/q4_0 is +11% (regress); F16 ties both.
  Before promoting flash as cohere's recommended path, validate on a multi-minute clip — if the
  crossover is workload-dependent, docs must recommend cast-on-read for short audio, flash for long.
  Until then PERFORMANCE.md notes flash as available-but-regresses-on-JFK for cohere.
- **encoder-decoder #69a** (canary, cohere, kyutai-stt). Cross-attention layout has no
  `<prefix><N>.*` block-tagged tensors; needs bespoke per-backend predicates. Own design problem.
- **#06 FA per-head mask (A1000 perf step).** Removes 72 CPU splits/chunk (per-head additive mask in
  `fattn.cu:423` + four kernel variants). ~300-500 LOC across `fattn.cu` / `fattn-common.cuh` /
  `fattn-mma-f16.cuh`. Expected ~10-15% wallclock on top of postsiglu. Follow the WDDM-warm bench
  protocol (LEARNINGS) for the new baseline before starting.

---

## §214 follow-up — chatterbox T3 batched-CFG (B=2) — deliver the win (OPEN)

Batched-CFG B=2 T3 decode shipped behind `CRISPASR_CHATTERBOX_T3_CFG_B2=1`
(default OFF); greedy-token bit-identical to legacy on CPU (all quants), GPU+F16,
GPU+quant. Path works — what's left is proving/delivering the speedup.
Files: `src/chatterbox.cpp` (`build_graph_t3_kv_b2`, `run_t3_kv_b2`,
`ensure_t3_b2_f16_weights`, decode loop ~§214). See HISTORY + PERFORMANCE §214.

TODO:
1. **Quiet-machine A/B + default-flip decision (HIGH, ~1 h).** CPU floor ~34 %
   was measured on a contended M1 — unreliable. Re-measure on a quiet host
   (alternating order, min-of-N, token-parity gate `CRISPASR_CHATTERBOX_TEMP=0`).
   **Flip default ON for the CFG path only if the win holds** (keep env + legacy
   path forever for bisection).
2. **Generalize B=2 to the other CFG backends — see §215.**

Closed/won't-pursue (do not reopen): cached/bucketed B=2 step graph (build+alloc
<1 % on both backends, and risks the §186 Metal `buffer is nil` crash); GPU B=2
as a speed win (GPU ~4× slower/step than CPU — its value is enabling T3-on-GPU +
GPU+quant F16-dequant, not beating the CPU default which stays production).

## §215 — batched-CFG (B=2) for the remaining TTS backends (OPEN)

Full-tree audit (§214, PERFORMANCE.md): only s3gen CFM + chatterbox T3 batch cond+uncond into one B=2 forward; every other CFG backend runs two sequential B=1 passes. Where the per-step forward is GPU-run, dispatch-bound, AND the dominant cost, apply the chatterbox-T3 B=2 pattern: batch over `ne[2]=2` for heavy GEMMs, split per-batch attention/KV-cache, tag `GGML_PREC_F32`, gate behind an env (default OFF), parity-gate vs the sequential path. **On GPU + quantized weights, dequant the batched weights q*→F16 GPU-resident** (s3gen `dequant_cfm_f16` / T3 `ensure_t3_b2_f16_weights` trick — Metal's batched `ne[2]=2` quant mat-vec misses the PREC_F32 exact-dot kernel and degenerates).

**General caveat (voxtral §93 lesson):** batched-CFG is a MODEST (~1.2–1.3×), GPU-dispatch-bound win. Before each port confirm the stage is GPU-run AND dispatch-bound AND dominant — measure, don't assume the T3 −42% transfers.

**Already resolved (see HISTORY):** §215a dia-*encoder* B=2 done; §215b tada — MEASURED NON-GOAL, do NOT port (talker only ~24% of loop, asymmetric graph paths); §215b bucket-floor follow-up SHIPPED backend-conditional default (`tada_default_bucket_min()`: Metal/CPU→64, CUDA/ROCm/Vulkan/WebGPU→512; `CRISPASR_TADA_BUCKET_MIN` overrides).

Open candidates, prioritized (high step count × dispatch-bound first):

1. **§215a dia decoder (HIGH).** `src/dia_tts.cpp run_dia_synth` — encoder already B=2, but the decoder AR loop still runs cond/uncond as two passes / two KV caches (`run_dia_decode_step`). Mirror chatterbox T3: B=2 decode-step graph, split per-batch KV write/read, F16-dequant on GPU+quant. Largest payoff — long AR loop, CFG every step.
2. **§215c zonos (MED).** `src/zonos_tts.cpp` keeps two KV caches (`kv_k`/`kv_k_uncond`), decodes sequentially (~L1740–1842). Batch AR backbone B=2, split dual-KV attention; keep dual-KV CFG + optional random speaker embed independent per batch.
3. **§215d voxcpm2 (MED).** `src/voxcpm2_tts.cpp` LocDiT runs `locdit_call` twice per ODE step (cond `mu` + zero-`mu`, ~L2752–2778). Diffusion (fewer steps) so lower payoff, but each LocDiT forward is heavy. Keep cfg-zero-star blend per batch.
4. **§215e f5 (MED, risky).** `src/f5_tts.cpp` runs `dit_forward` twice per CFG step (v_cond+v_uncond, ~L1563–1566). Same B=2-DiT shape as voxcpm2. §176h: a standalone B=2 DiT graph is correct but a runtime B=2 corrupted batch-1 (F5-runtime-specific, [[project_ggml_batched_fused_graph_alloc_bug]]) — run the parity gate especially carefully; may not be worth it.
5. **§215f cosyvoice3 (LOW — likely WON'T).** `src/cosyvoice3_tts.cpp` explicitly declined batching (~L3027–3030): 22-block diffusion twice/step, per-call overhead small vs forward. Only revisit if a profile shows dispatch overhead matters; else document as deliberate non-goal.
6. **§215g kugelaudio (BLOCKED).** `src/kugelaudio.cpp` CFG is a TODO (cfg_scale read but unused, negative path unimplemented). Implement sequential CFG first (correctness), then consider B=2.

**Shared infra:** factor the F16-dequant-of-matmul-weights helper (duplicated in `ensure_t3_b2_f16_weights` + s3gen `dequant_cfm_f16`) into a `core_*` helper only when a third backend needs it.

## §210 follow-up — shape-stable bucketed decode for remaining LLM/AR backends (CUDA-graph capture) (OPEN, CONDITIONAL)

Status: template landed in granite-speech (PR #207) — fixed-`Lk` KV bucket written via
`ggml_set_rows`, in-graph `ggml_argmax`, fused F16 embed. Unlocks CUDA-graph capture
(~9–13× decode on Ampere+, engages automatically in ggml-cuda for shape+pointer-stable graphs)
and Metal gallocr allocate-once. Survey done (LEARNINGS §210). **Done, no work:** granite_speech,
mimo_asr, dots_tts. irodori-tts DiT persistent graph implemented + byte-identical parity (default
OFF, gated `CRISPASR_IRODORI_PERSIST_GRAPH`).

### ⚠️ Cost/benefit — don't mass-port
- CUDA-graph win is gated to Ampere+ (sm_80+): `ggml-cuda.cu:4329`. Project's usual test GPUs
  **T4 (sm_75) / P100 (sm_60) are gated OUT** — only RTX 5090 / A100-class benefit.
- On Metal there is **no throughput win** (granite decode host-encode 1.8% / GPU 98%, GPU-bound
  Q4_K GEMVs); shape-stable rewrite buys only memory-pressure robustness, not speed.
- Each port is a MANUAL graph rewrite + byte-identical diff-harness validation — NOT delegable to
  agents (runtime graph code). Budget one focused session per backend.

**Port a backend ONLY when actually deployed on Ampere+ CUDA, and measure first**
(`CRISPASR_METAL_PROFILE` for host/GPU split; confirm no "disabling CUDA graphs" GGML_LOG_DEBUG line
on an Ampere+ GPU). Smaller decoders may have a larger host-encode fraction than granite's 1.8% — measure, don't assume.

### Templates (copy — already shape-stable)
- `src/granite_speech.cpp`: `granite_build_argmax_decode` (~L2474), `granite_dec_use_gallocr`
  (~L2362; gallocr on Metal/CPU, sched on CUDA so capture fires).
- `src/mimo_asr.cpp`: cached `step_t1_gf` + `step_t1_fixed_kv_len` (L211), set_rows scatter at
  runtime `kv_indices` (L958, L1026), skip-plan reuse (L1507–1522) — good set_rows reference.

### Candidates — growing-shape (`Lk = n_past + T`), naive per-step rebuild + sched_reset/alloc
ASR-LLM (prioritized by likely server deployment):
1. **voxtral** — `src/voxtral.cpp`, `Lk` L1019; rebuild + reset/alloc L1159/L1192/L1244.
2. **qwen3_asr** — `src/qwen3_asr.cpp`, `Lk` L1210.
3. **voxtral4b** — `src/voxtral4b.cpp`, `Lk` L1411 (decode via `core_greedy_decode`).
4. **gemma4_e2b** — `src/gemma4_e2b.cpp`, `Lk` L995 (`core_greedy_decode::run_with_probs_cb` ~L1631).
5. **glm_asr** — `src/glm_asr.cpp`, `Lk` L1322.
6. **higgs_stt** — `src/higgs_stt.cpp`, `Lk` L1061 (Qwen3-1.7B decoder).
7. **ark_asr** — `src/ark_asr.cpp`, `ark_build_decoder_graph` (L643) / `ark_run_decoder` (L729), `Lk` L674.
8. **lfm2_audio** — `src/lfm2_audio.cpp`, `Lk` L933; already on gallocr (L875) but growing-shape, not yet capturable.

TTS AR decoders (LOWER priority — heavier per-step compute = more GPU-bound, payoff diluted):
`csm_tts`, `indextts`, `bark_tts`, `moss_audio`, `mini_omni2` naive growing-shape. `qwen3_tts`,
`chatterbox` (T3), `tada_tts`, `parler_tts`, `vibevoice` already have per-backend perf work — audit individually.

### Per-backend recipe (mirror granite)
1. Allocate KV at `kv_max_ctx` once; pick fixed `bucket_len` (≤ cache cap).
2. Rewrite step graph to fixed `[0, bucket_len)` KV view; write new token K/V via `ggml_set_rows` at
   runtime index `n_past`. Mask `(bucket_len, 1)`, set host-side each step. Topology byte-identical across steps.
3. Move argmax in-graph (`ggml_argmax`); keep `logits` as graph output for callers.
4. Make embed capturable: if `token_embd` k-quant, in-graph `GET_ROWS` host-syncs and disables
   capture — fuse a F16 embed or pass a pre-computed F32 embed input (granite `fused_embed`).
5. Cache the cgraph; gate gallocr-vs-sched like `granite_dec_use_gallocr` (sched on CUDA/HIP so
   capture engages; force sched if a CPU layer split exists). Env opt-out.
6. Bound `n_past < bucket_len` (granite OOB guard `c5035969`).

### Validation gate (mandatory)
- Byte-identical transcript vs legacy path on jfk + fleurs_60s — no merge without it.
- Measure before/after: `CRISPASR_METAL_PROFILE` + per-step compute-µs accumulator like
  `CRISPASR_GRANITE_DEC_PROFILE`. On real Ampere+ confirm capture engages, A/B decode RTFx.
  M1 wall time is noise — gate on the instrumented per-step quantity.
- irodori DiT: flip its default ON (or gate `cc>=800`) only once a real Ampere A/B shows a win.

**Effort:** ~1 focused session per backend. Do highest-deployment ASR backend first; stop if its
measured Ampere+ CUDA A/B doesn't justify the next.

---

## §219 — more permissive audio input formats for crispasr_audio_load

_Completed work archived to HISTORY.md (PLAN compaction 2026-07-17)._

**Still open:** Speex / WavPack decoder support (LOW, only if a corpus needs them)

## §166 follow-up — WASM `asr*` session surface needs a build-verify

_Completed work archived to HISTORY.md (PLAN compaction 2026-07-17)._

**Still open:** Runtime asrTranscribe-equivalent smoke test in node/browser (LOW nicety)

## §175 Surgical DRY — share pure helpers across the CLI/library boundary

_Completed work archived to HISTORY.md (PLAN compaction 2026-07-17)._

**Still open:** language-instruction prompt template helper (revisit if a 3rd consumer appears)

## §177 VibeVoice #171 — remaining layer: RDNA4/RADV quantized-matmul path (OPEN)

All other legs fixed (server/CLI chunking §176 prefix guard, KV stride leak at
71f0639, quant recipe b36248c1+5c8add40 — three HF repos regenerated). Reporter
retested regenerated q8_0 (sha256-confirmed current HF file): still broken. Signature:
q8_0 breaks, f16 clean, same RX 9070 XT (RADV GFX1201, `int dot: 1`); Metal + MoltenVK
clean on the exact q8_0. Suspicion: RADV GFX1201 shader miscompile on the quantized-
matmul paths (int-dot MMQ/MMVQ, KHR_coopmat dequant) — shader logic proven fine
(forced int-dot + MMVQ clean on M1). Vendored ggml base 2026-05-05, no matching
upstream fix found.

TO DO:
- [ ] Reporter runs the knob matrix one-at-a-time on a broken sample (env-only, no
  rebuild): `GGML_VK_DISABLE_INTEGER_DOT_PRODUCT=1`, `GGML_VK_DISABLE_MMVQ=1`,
  `GGML_VK_DISABLE_COOPMAT=1`, `GGML_VK_DISABLE_F16=1`, anchor `CRISPASR_N_GPU_LAYERS=0`.
  Also Mesa upgrade / AMDVLK cross-check.
- [ ] Once one knob is confirmed → device-targeted safe default (RADV GFX12xx) in
  vendored `ggml-vulkan` + upstream report to ggml-org/llama.cpp with minimal repro.
- Note for thread: reporter's old `vibevoice-1.5b-bf16.gguf` predates `--include-decoder`;
  current f16 on `cstr/vibevoice-1.5b-GGUF` has the decoder tensors.

---

## 40. More Moonshine model variants

_Completed work archived to HISTORY.md (PLAN compaction 2026-07-17)._

**Still open:** moonshine-streaming-tiny/small/medium need new streaming runtime

## 97. More Parakeet variants

Runtime (`src/parakeet.cpp`) dispatches TDT/CTC/RNNT via GGUF flags; converter
(`models/convert-parakeet-to-gguf.py`) reads hparams from `model_config.yaml` +
cross-checks tensor shapes. Most FastConformer-encoder + TDT/CTC/RNNT checkpoints
are converter-only, no new C++. Shipped: tdt-0.6b-v2, tdt-1.1b, tdt_ctc-110m,
tdt_ctc-1.1b, rnnt-0.6b, rnnt-1.1b (all on HF, registry-wired). See HISTORY.

**Still open:**
- **`nvidia/parakeet-unified-en-0.6b`** — direct-zip extraction + GGUF conversion
  already WORK (v4/v5 Kaggle kernels; 1181 MB F16, same arch as standard parakeet:
  d_model=1024, n_layers=24, vocab=1025, pred=640, joint=640). CrispASR SIGABRTs
  because runtime assumes 4× subsampling but this model uses **8×** (3 strided
  convs, not 2). **Fix:** make `parakeet_build_pre_encode` in `src/parakeet.cpp`
  (or `core/fastconformer.h`) handle `subsampling_factor=8` — 3 Conv2d layers with
  strides [1,2] instead of 2. Tensor shapes already in GGUF; graph builder just
  needs the extra conv layer.
- **`nvidia/parakeet_realtime_eou_120m-v1`** — streaming + end-of-utterance head.
  Needs cache-aware FastConformer streaming (cf. #81 Nemotron) + an EOU head. Not
  converter-only.

**Won't do (unless a user asks):** `parakeet-ctc-0.6b-Vietnamese` — already
runtime-supported (CTC), known gap not active work.

See also #98 Hotwords (orthogonal; lights up biasing on every Parakeet variant
once CTC-WS trie lands).

## 98. Hotwords / contextual biasing (OPEN)

User-supplied vocabulary the ASR prefers when in doubt (names, jargon, product/place
names). Helps only the biased subset, but lift there is large. Phased so each phase covers
a family of backends. ~1 week total covering 9 of 14 backends. (Full upstream-support
survey → HISTORY.)

**Phase A — generic CTC-WS phrase-boost trie** (2–3 days). Covers parakeet-ctc / -tdt /
fastconformer-ctc / omniasr in one shot (model-agnostic on the logit stream).
- New shared helper `src/core/asr_context_bias.{h,cpp}` — Aho-Corasick trie over piece-id
  sequences, configurable per-phrase boost; emits a per-frame log-prob bias vector the
  CTC/TDT decoder shallow-fuses into argmax/beam scoring. Pure CPU, no ggml graph.
- Wire-in: `parakeet_ctc_decode` + `parakeet_tdt_decode` (`src/parakeet.cpp:999+`,
  `parakeet.cpp:1670` dispatch). Phrase tokenisation via the backend's existing
  SentencePiece model so users pass human-readable strings.
- CLI: `--hotwords "Acme Corp,Sandra Berenz,GPU-PB"` and/or `--hotwords-file <path>`
  (one phrase/line, optional `^N` boost suffix); env `CRISPASR_HOTWORDS=...` for the
  OpenAI-server path. ~250–400 LOC incl. beam-rescoring.
- Reference to mirror: NeMo CTC-WS notebook + TurboBias `BoostingTree` C++
  (`arxiv.org/html/2508.07014v1`).

**Phase B — `--hotwords` → LLM prompt-prefix helper** (1 day). Covers funasr, granite-plus,
voxtral, qwen3-asr (each already has a system-prompt path).
- Tiny `src/core/` helper renders a hotword list into each backend's prompt template
  (granite `Keywords: …`; funasr hotword token block; qwen3 free-text context). One template
  registry, one call site per backend.
- Wire-in: `funasr_transcribe_ex`, `granite_nle_transcribe`, voxtral, qwen3-asr. ~150 LOC +
  per-backend template strings.

**Phase C — parakeet TDT joint-net boost (Transducer-native)** (1–2 days). Per-step bias on
joint-net output when the partial hyp matches a trie prefix (mirrors NeMo MBS hotwords).
- DECISION-GATE: defer until Phase A is shipped + benchmarked; only do it if Phase A on TDT
  undershoots NeMo's reference numbers.

Out of scope: whisper `initial_prompt` (already upstream via `--initial-prompt`), MiMo
PromptASR (no upstream flag — park), cohere/moonshine/kyutai-stt/glm-asr (no upstream hook).

Validation: `tests/test_hotwords.py` — synthetic clip with a rare name (e.g. "Berenz")
through each Phase-A backend with/without `--hotwords Berenz`; assert unbiased misspells,
biased nails it. Phase B: assert prompt-prefix matches upstream Python byte-for-byte.

---

## funasr — perf follow-ups (LOW priority, not blocking)

Port ships with `ggml_flash_attn_ext` on encoder+adaptor (`FUNASR_NO_FA=1` to opt
out), fused QKV (DONE), and single-token embed fast path (`CRISPASR_FUNASR_EMBED_FAST`,
default ON, DONE §180). None affect correctness — pure throughput. Remaining:

- [x] **Per-step LLM decode graph cache — DONE 2026-09-02, but NOT as recorded
  here.** The literal design (`fixed_kv_len = kv_max_ctx`, one graph) was
  already scaffolded default-OFF in the file with an M1 3× regression note,
  and re-measured on x86 as **+69% decode CPU** — one wasted KV key costs
  ~0.28 ms/tok through the GQA repeat+cont, so a fixed-Lk graph loses far
  more to KV bandwidth than it saves in graph prep (~2.1 ms/tok). Shipped
  instead: **bucketed Lk, width 16** (`CRISPASR_FUNASR_STEP_BUCKET`, measured
  optimum; break-even w≈15), FIFO of 4 live graphs, padded slots masked to
  -inf (flash-attn skips fully masked positions → bit-identical by
  construction), default ON with `CRISPASR_FUNASR_STEP_CACHE=0` opt-out.
  Byte-identical 18/18 (3 clips × f16/q4_k × pre-change/off/on, re-verified
  independently), quant-bcast audit clean (0 hits, q4_k+q8_0, both arms).
  Net on x86: neutral within ±8% noise (decode is weight-bandwidth-bound;
  step prep was only 2–3.8 ms/tok here, not the ~30 ms the M1 note implied)
  while removing 93% of per-step graph prep — the win lives on platforms
  where graph construction is expensive (M1). qwen3_asr adoption: only with
  the bucketed variant, never fixed-Lk.
- [x] **Encoder graph cache — exact-T_lfr half DONE 2026-09-02** (default ON,
  `CRISPASR_FUNASR_ENC_CACHE=0` opt-out): repeat-length calls skip the
  7.5–23.9 ms graph build (cache hit = 0 ms; sched_alloc 2.4–6.2 ms still
  paid). The padded-bucket half is **dropped as designed**: the SANM FSMN
  branch is a width-11 depthwise conv over TIME (`core/sanm.h:128`), so a
  padded frame leaks into the last 5 real frames of all 70 blocks and
  zero-padding doesn't save it (LayerNorm of a zero row = bias ≠ 0) — a
  bit-identical version needs a time mask on V inside `core_sanm` — and the
  economics are upside down anyway (build is ~10–24 ms against a 2.3–6.1 s
  encoder; padding T_lfr 183→256 would spend seconds to save milliseconds).
- [ ] **Two-pass: CTC fast pass → Fun-ASR-Nano LLM rescore.** Upstream checkpoint has
  0 CTC tensors (LLM-style by choice); the only public trained CTC head is
  `csukuangfj/funasr-nano-with-ctc` (Apache-2.0, encoder+adaptor+CTC, no LLM, frozen
  encoder = upstream). Two patterns: (a) csukuangfj CTC head + upstream encoder/adaptor/LLM
  — single shared encoder forward, no vocab remap (cleaner, single-author trust);
  (b) fallback SenseVoice-Small fast pass (gold trust, extra encoder + vocab translate).
  - **Phase A (measure, no code):** tensor-list csukuangfj `model.pt` to confirm his
    encoder == upstream byte-identical; run his CTC head + upstream encoder+adaptor on
    a zh+en ground-truth set, measure CER/WER vs pure-LLM path. Need within ~3–5% rel
    for rescore net win. Write to LEARNINGS.md; proceed to B if in bounds, else use
    fallback (b).
  - **Phase B (impl):** grow `models/convert-funasr-to-gguf.py` to optionally pick up
    `ctc_decoder.*` from a with-ctc checkpoint → `funasr-nano-with-ctc-q4_k.gguf`,
    auto-download from `cstr/funasr-nano-with-ctc-GGUF` (mirror csukuangfj + attribution).
    Opt-in `CRISPASR_FUNASR_TWOPASS=1` (requires companion GGUF, else single-pass) +
    `--asr-rescore` CLI flag. One encoder forward forks into CTC + LLM heads: greedy
    CTC → per-frame probs; skip LLM if avg per-frame conf >0.95, else CTC top-K as
    LLM decode-prefix candidates. Expected 2–4× on high-conf clips, neutral on hard audio.

---

## §187 Cross-runtime embed fast path sweep

_Completed work archived to HISTORY.md (PLAN compaction 2026-07-17)._

**Still open:** granite_speech embed fast path (VPS bench blocked) + deepen bench stages / encoder graph cache

## §201 Kaggle CUDA backend failures — full sweep 2026-06-20 (OPEN)

Status: the 2026-06-20 full Kaggle GPU sweep (`tools/kaggle-benchmark-all-backends.py`, streamed to
`cstr/crispasr-kaggle-progress/full-backend-sweep/`) ran 59 backends; 10 apparent failures reduced to
a handful of genuine CUDA-path bugs. Most resolved (vibevoice/lfm2-audio §206/kugelaudio §209/
fastpitch+speecht5 §204/chatterbox §205 — all in HISTORY). Remaining open:

TODO (open):
- [ ] **f5-tts** — runs once given a reference voice but TIMEOUT at 120 s in re-test.
  Timeout half DONE (registry already carries `timeout_s=600` for f5-tts,
  `tools/test-all-backends.py` — ≥240 satisfied); the Kaggle re-run to settle
  pass-vs-stuck is still pending; passes on M1 Metal locally.
- [ ] **orpheus** (TTS) — fixed §215 (Metal + CPU bucket both ASR-roundtrip verbatim on M1), stays
  opt-in `CRISPASR_ORPHEUS_BUCKET=1` (~30% slower on M1 unified memory, may win on CUDA).
  **CUDA cross-check still pending** (Kaggle `${KAGGLE_ACCOUNT}/crispasr-orpheus-talker-cuda` end-to-end
  `orpheus_synthesize`).
- [ ] **chatterbox** (TTS) — 0-byte (~14 s) with `--voice <wav> --i-have-rights`; the #83 S3Gen GPU
  fix was Metal-validated — re-check the CUDA S3Gen path.
- [ ] **cosyvoice3** (TTS) — dies in 0.1 s even with a reference voice; passes on M1 Metal.
  Flow-matching + HiFT — cheapest to bisect. §205's mixed-radix FFT fix covers CosyVoice3's
  `n_fft=400` mel (same heap overflow as chatterbox) — likely resolved, needs Kaggle CUDA re-test.

**Method:** per-backend JSONs in the dataset have timing context; reproduce on a CUDA worker
(Kaggle T4/P100 or A1000) with `CRISPASR_VERBOSE=1` + `CRISPASR_<BACKEND>_DEBUG=1`. Several pass on
M1 Metal → CUDA-path-specific; cross-check Metal first. Small models also fit the 8 GB CPU-only VPS
where the diff harness can drive the fix if the bug reproduces on CPU (how §204/§206 were fixed).

---

## 51c. MiMo-V2.5-ASR F16 step decode — open (validation only)

Base runtime + Q4_K + fused-QKV shipped (HISTORY §56/§64); 51a mmap loader (§62)
and 51b step-decode KV reuse (§60) DONE. Only F16 step decode remains — **no code
change needed** (runtime is dtype-agnostic; mmap loader already wired). Blocked
behind ≥32 GB RAM: F16 working set (~16 GB) thrashes on this 16 GB box.

### TO DO
- On a 32+ GB box, validate JFK transcript byte-equality + decode speedup:
  ```
  CRISPASR_GGUF_MMAP=1 ./build-ninja-compile/bin/crispasr --backend mimo-asr \
    -m /path/to/mimo-asr-f16.gguf --codec-model /path/to/mimo-tokenizer-q4_k.gguf \
    -f samples/jfk.wav
  ```
- **Gate:** if F16 prefill hits ≥1× realtime as predicted, ship F16 as the
  recommended quant on `cstr/mimo-asr-GGUF` and demote Q4_K to memory-tight
  fallback. Until then both shipped, Q4_K default.

Effort: **0 LOC** (validation only).

## 52. Qwen3-TTS

_Completed work archived to HISTORY.md (PLAN compaction 2026-07-17)._

**Still open:** perf pass — clean quiet-machine FUSED_QKV bench (F16/Q4_K), fusing 15 cp steps into one graph

## 54-follow-up. granite-speech-4.1 plus speaker labels + word timestamps — open

Variants 4.1 / 4.1-plus / 4.1-nar shipped bit-exact on JFK → HISTORY
§61. Remaining: speaker labels + word-level timestamps for the `plus`
variant via chat_template (~50 LOC, template-only).

---

## 56. Kokoro multilingual phonemizer (espeak-ng)

In-process libespeak-ng phonemization (behind CMake `CRISPASR_WITH_ESPEAK_NG`
AUTO/ON/OFF, `CRISPASR_HAVE_ESPEAK_NG=1`) with popen fallback + LRU cache SHIPPED.
German voice cascade (Option 1/2a/2b), phonemizer diff harness, and cache-clear
ABI all DONE (see HISTORY). Remaining open: Mandarin tone numbers, Japanese kanji,
optional German stage-2 fine-tune.

### Open TODO

1. **German stage-2 fine-tune (optional, out of scope of this item).** Native
   German path already ships (dida-80b backbone + kikiri voicepacks, auto-routes
   when both `kokoro-82m-f16.gguf` and `kokoro-de-hui-base-f16.gguf` are in the
   same dir). For deployable single-speaker production quality, run Stage-2
   fine-tune on one HUI speaker (~half-day A40). Track separately if needed.

2. **Mandarin tone numbers.** espeak-ng emits digit-suffixed tones (`ni2χˈɑu2`)
   not in the 178-symbol kokoro IPA vocab → dropped at tokenization, losing tone.
   Investigate `--ipa=2` (no tone numbers) + separate tone embedding, or switch
   Mandarin G2P (e.g. `pypinyin`). Symptom is auto-captured by
   `tools/check_kokoro_phonemizer_parity.py`. Effort: ~an afternoon.

3. **Japanese kanji.** espeak-ng falls back to English for kanji (日本語 →
   "Chinese letter") inserting non-IPA `(en)…(ja)` markers. Pre-process kanji→kana
   with a Japanese frontend before espeak. MIT-clean approach: MeCab (BSD-3) +
   unidic-lite (MIT) morphological analysis → reading extraction → feed kana to
   espeak `ja`. NO kakasi/pykakasi (GPL-3, viral). Impl: libmecab C API via dlopen
   (like espeak) OR a shipped kanji→kana flat-file dict (simpler, less accurate on
   rare/compound; generate offline with fugashi/cutlet, both MIT). Effort: ~an
   afternoon.

Files: `src/kokoro.{h,cpp}`, `examples/cli/crispasr_backend_kokoro.cpp`.

## 57. Commercial-friendly TTS backend expansion (OPEN — Phase 1 done)

May 2026 sweep of high-traffic HF TTS models. Filter: **permissive license + reusable
architecture + reasonable effort**. Sequenced so each phase unlocks a family of finetunes
(e.g. Phase 3 Chatterbox stack also unlocks Phase 5's CFM solver).

License triage that drives ordering (candidates for later phases):

| ✅ Permissive (commercial OK) | ⚠️ Llama-3.2 community (OK w/ attribution) | ❌ Non-commercial — defer |
|---|---|---|
| Qwen3-TTS-{Base,CustomVoice} (Apache 2.0) | Orpheus-3B family + Kartoffel_Orpheus (llama3.2) | SebastianBodza/Kartoffelbox-v0.1 (CC-BY-NC-ND) |
| ResembleAI/chatterbox base (MIT) | HumeAI/tada-3b-ml (llama3.2) | marduk-ra/F5-TTS-German (CC-BY-NC) |
| SebastianBodza/Kartoffelbox_Turbo (CC-BY-4.0, gated) | | mlx-community/fish-audio-s2-pro (Fish-Audio Research) |
| oddadmix/lahgtna-chatterbox-v0/v1 (MIT) | | amphion/Vevo1.5 (CC-BY-NC-ND) |
| openbmb/VoxCPM2 (Apache 2.0) | | mlx-community/Voxtral-4B-TTS-2603 (CC-BY-NC; upstream Mistral Apache OK) |
| FINAL-Bench/Darwin-TTS-1.7B-Cross (Apache 2.0) | | |
| AMAImedia Qwen3-1.7B-TTS-Cross-Darwin AWQ (Apache 2.0) | | |
| g-group-ai-lab/gwen-tts-0.6B (MIT) | | |
| kugelaudio/kugelaudio-0-open (MIT) | | |

TO DO:
- ~~Resolve license gap before depending on CosyVoice 3~~ — MOOT: `src/cosyvoice3_tts.cpp`
  + registry entry already shipped. Likewise VoxCPM2, kugelaudio, gwen-tts, and
  kartoffelbox-turbo (German) are all ported + registry-wired (verified 2026-07-17).
- Remaining unported from the permissive column: **Darwin-TTS-1.7B-Cross** and the
  **AMAImedia Qwen3-1.7B-TTS-Cross-Darwin** family (no `src/*darwin*`, no registry entry).
- Phase 2+ ports otherwise not detailed here — scope from the permissive column when picking
  the next family.

Phase 1 — DONE (see HISTORY.md + git log).

---

## Ecosystem expansion (lower priority)

### Candidate new backends (from PazaBench, see HISTORY #30)

| Model | License | Approach | Priority |
|---|---|---|---|
| Wav2Vec2 Conformer | Apache-2.0 | Conformer attention variant | Medium |
| Qwen2-Audio 7B | Apache-2.0 | Whisper encoder + Qwen2 LLM | Medium |
| OmniASR larger (1B/3B/7B) | Apache-2.0 | Same converter, bigger models | Medium |
| NeMo Canary-Qwen-2.5b | Apache-2.0 | FastConformer + Qwen2.5 decoder | Medium |
| Paza / Phi-4 | MIT | 14B multimodal, defer to llama.cpp | Low |
| XiaomiMiMo/MiMo-V2.5-ASR | TBD (check) | LLM-style multimodal speech (Qwen3-ASR pattern) | Medium — user-requested #35 |
| google/gemma-4-E2B | Gemma terms | Conformer + Gemma 4 decoder | Medium — user-requested #35 |

From llama.cpp (MIT): Ultravox (Whisper enc + Llama 3.2), Gemma 4 Audio (Conformer,
chunked attn, streaming), LFM2-Audio (Conformer variant, position embeddings).

### Post-processing — remaining candidates

- [ ] **CT-Transformer (FunASR)** Apache-2.0 — Medium. SANM 3-layer (vocab 272727),
  zh+en, FunASR/RapidPunc production default.
  `modelscope/punc_ct-transformer_zh-cn-common-vadrealtime-vocab272727-pytorch`. SANM
  primitives already in CrispASR (`src/core/sanm.h`). New alias `ct-punc`; VAD-realtime
  variant emits per-segment punc for streaming.
- [ ] bert-restore-punctuation (MIT, en) — Low.
- [ ] xashru/punctuation (Apache-2.0, XLM-R+BiLSTM-CRF, 40+ langs) — Low.
  (FireRedPunc, fullstop, punctuate-all, PCS, all truecasers — DONE, see HISTORY.)

### Optimizations still open

| # | Optimization | Applies to | Expected gain | Status |
|---|---|---|---|---|
| O2 | Fused QKV pre-merge | LLM decoders | ~10-15% attn (GPU) | API ready in core/attention.h; CPU gain <1%, defer to GPU |
| O5 | Pipelined mel+encode | LLM backends, CPU | ~15-20% | TODO |
| O6 | Batched encoder (GPU) | All + GPU | 3-5x | TODO |
| O7 | Speculative decoding | LLM backends | 2-4x decode | TODO |
| O4 | Beam search for LLMs | Audio-LLM backends | Quality | DONE except mimo-asr, blocked on #115 |

Guidance: only move LARGE, REUSED matmuls onto ggml/GPU; persistent subgraphs per
decode step > one-off graphs; never dequant (native Q4_K matmul 9.3× faster than F32
OpenMP). Candidate from CrispEmbed: SentencePiece Viterbi DP optimal tokenizer.

### Audio format support (open)

- [ ] `.m4a`, `.mp4`, `.webm` crash with upstream ffmpeg integration — needs fix or
  robust fallback.
- [ ] `.aiff`, `.wma`, raw PCM not supported without pre-conversion. Consider bundling
  a lightweight M4A/AAC decoder or improving the ffmpeg path.

### Other

- [ ] **OmniASR-LLM beam search** — beam=2+ with N hypothesis KV caches.

---

## Publish language wrappers to package registries

**Status:** Dart wrapper is **published** on pub.dev (`crispasr 0.8.11`, manual publish — see §66). Rust + Python wrappers have publishable metadata + passing dry-runs but are **not yet on crates.io / PyPI** (both 404, verified 2026-07-17) — blocked on the one-time registry creds bootstrap in §66. All are thin FFI/ctypes shims over the C ABI in `src/crispasr_c_api.cpp` — they do NOT bundle the native lib (user must have `libcrispasr.{so,dylib,dll}` installed).

### TO DO — one-time registry setup (must happen before first `v*` tag)

1. **PyPI** — at https://pypi.org/manage/account/publishing/ add a pending publisher: owner `CrispStrobe`, repo `CrispASR`, workflow `release-wrappers.yml`, environment `pypi`. Then push any `v*` tag. (OIDC trusted-publishing, no token.)
2. **crates.io** — generate a token at https://crates.io/me, add as `CARGO_REGISTRY_TOKEN` repo secret. Publish order: `crispasr-sys` then `crispasr`.
3. **pub.dev** — at https://pub.dev/packages/crispasr/admin (after first manual publish/claim) enable automated publishing, tag pattern `v{{version}}`. Or first-publish locally via `dart pub publish` with owner creds.

### TO DO — Python library discovery
Update `_find_lib()` in `python/crispasr/_binding.py` to probe in order: (1) `$CRISPASR_LIB_PATH`; (2) `sys.prefix/lib/`; (3) Homebrew/Linux paths (`/opt/homebrew/lib`, `/usr/local/lib`, `/usr/lib`); (4) existing repo-relative fallbacks. If none found, raise `RuntimeError` linking to install docs.

### TO DO — release automation
`.github/workflows/release-wrappers.yml`, tag-triggered (`v*` only, not every commit), runs in parallel: `python -m build && twine upload` (PyPI OIDC); `cargo publish -p crispasr-sys && cargo publish -p crispasr` (crates.io); `dart pub publish --force` (pub.dev OIDC). Version bumps stay manual — bump `pyproject.toml`/`Cargo.toml`/`pubspec.yaml` together in the tag commit.

**Effort:** Low per wrapper.

### Deferred — bundled wheels for Python
After the pure-Python release is out and stable, add a `cibuildwheel` pipeline (manylinux2014 + macOS arm64/x64 + Windows) bundling `libcrispasr.*` via `auditwheel`/`delocate`/`delvewheel`. Same optional path for Rust (`crispasr-sys` vendoring native build like `tch-rs`/`onnxruntime-sys`). Defer.

## 59. Cross-binding C-ABI parity

**Status:** Session TTS API (incl. qwen3-tts variant routing) is wrapped across all 7 bindings (commit `65e0a61` + Dart follow-up). The non-Session ABI (~80 of the 136+ `crispasr_*` exports in `src/crispasr_c_api.cpp`) is still C-ABI-only or partially wrapped on most bindings. Rust + Python are canonical full-coverage; the diarize surface (segment-level + #107 P6 embedder/clustering/cache) also landed in Dart/Flutter + Go.

### When to do this
**Not now.** Open a capability×binding only when a concrete consumer asks (e.g. "Java VAD", "Go streaming"). Reference commits for the pattern: `4f476c3` (TTS surface sweep), `65e0a61` (variant detect).

### TO DO — capabilities reachable only from C-ABI / Rust / Python
Each = ~3-12 exports + an idiomatic result type per binding:
- **Forced alignment** — `crispasr_align_words`, `align_words_abi`, `align_result_*`.
- **Diarization (segment)** — `crispasr_diarize_segments[_abi]`. Missing in Java, Ruby, JS.
- **Diarization (embedder+clustering)** — `crispasr_speaker_embedder_*_abi`, `crispasr_speaker_cluster_abi`, `crispasr_pyannote_cache_*_abi` (#107 P6). Missing in Java, Ruby, JS.
- **Language ID** — `crispasr_detect_language[_pcm]`, `crispasr_lid_free_cache`.
- **VAD** — `crispasr_vad_segments`, `crispasr_compute_vad_slices`, `crispasr_stitch_vad_slices`, `crispasr_vad_remap_timestamp`, `crispasr_vad_free`.
- **Streaming** — `crispasr_stream_open/feed/get_text/flush/close`, `crispasr_stream_run_decode`.
- **Punctuation** — `crispasr_punc_init/process/free/free_text`.
- **Model registry** — `crispasr_registry_lookup[_abi]`, `registry_lookup_by_filename[_abi]`, `crispasr_detect_backend_from_gguf`.
- **Cache** — `crispasr_cache_dir_abi`, `crispasr_cache_ensure_file_abi`.

### TO DO — #107 diarize-pipeline follow-up (three bindings still have nothing wired)
- **Java** (`bindings/java/`) — JNI exposes only `crispasr_session_*` + `*speaker_name*`. Add JNI wrappers for `crispasr_diarize_segments_abi` + the 9 `crispasr_speaker_*_abi`/`crispasr_pyannote_cache_*_abi` exports + idiomatic helper class. ~250 LOC.
- **Ruby** (`bindings/ruby/`) — only `Session.transcribe`. Needs Ruby FFI diarize bindings. ~200 LOC.
- **JS/WASM** (`bindings/javascript/`) — no speaker surface; depends on the WASM build linking pyannote-seg/titanet/indextts_voc. Start with `crispasr_diarize_segments_abi` (no model deps beyond existing wasm whisper); defer embedder primitives.

**Effort:** ~150-300 LOC per binding. Suggested ordering once a consumer asks: (1) Streaming (Go/Java), (2) VAD+alignment (Dart/mobile), (3) Diarization+LID+punc, (4) Registry+cache.

### Follow-up — Rust binding directory location (low priority)
Optionally relocate `crispasr/` + `crispasr-sys/` (repo root) under `bindings/rust/` to match C-family bindings. **Do NOT rename the crates** (names are correct/idiomatic). Move both dirs together (relative `path = "../crispasr-sys"` dep, no workspace). Consumer-safe (crates.io resolves by name+version). Before moving, audit: (a) downstream repos using `git`+`path` dep on the subdir (CrispEmbed/CrisperWeaver), (b) internal CI/`scripts/`/`build_go` refs, (c) docs path refs. One deliberate commit. Not worth churn unless the root-dir ambiguity bothers.

## 65-residual. JS / emscripten word-accessor surface — open

Parent #65 (session-API word-confidence parity) shipped → HISTORY §65
(main batch + vibevoice / moonshine-streaming + gemma4-e2b token-prob
API + Go/Java/Ruby parity in `5534588` + `d963e3a`). Only residual:
JS/emscripten word accessors — leaving until a JS consumer asks (the
current JS binding is TTS-focused).

---

## 66. Wrapper publishing bootstrap — required before language registries can ship

**Status:** PARTIAL (verified live 2026-07-17). Auto-trigger silenced —
`tags: ['v*']` push trigger on `release-wrappers.yml` is COMMENTED OUT (failed on
every release since v0.5.0, confirmed v0.5.4 `gh run view 25248028443`). Workflow
stays on `workflow_dispatch` only.

Live registry state:
- **pub.dev `crispasr` — DONE**, latest **0.8.11** (manually published
  `dart pub publish --force`, see `~/code/pupdev.md` handover 2026-07-15). No
  bootstrap needed for the package to exist; only the pub.dev-admin "automated
  publishing" toggle remains if tag-triggered republish is wanted.
- **crates.io `crispasr` + `crispasr-sys` — DONE**, both **0.8.23** (manually
  published 2026-07-27 with `CRISPASR_LIB_DIR` set so the verification build
  takes build.rs path 1, no cmake). The account needed a one-time verified
  email first. crates.io consumers must link a pre-built lib
  (`CRISPASR_LIB_DIR`); the package does not vendor the C/C++ sources, so a
  from-source build needs the git dependency (build.rs hardened to say so).
- **PyPI `crispasr` — DONE**, latest **0.8.23** (also 0.8.22; both manually
  uploaded 2026-07-24, verified 2026-07-27 — the published 0.8.23 wheel is
  byte-identical to `python/crispasr/_binding.py`). Pure-Python `py3-none-any`
  wheel; the native `libcrispasr` is still installed separately by the user.

So all three registries (crates.io, PyPI, pub.dev) are now bootstrapped; only
the optional CI/OIDC auto-trigger remains (below).

### TO DO — bootstrap (one-time, needs repo admin creds)

1. **crates.io — ALREADY DONE** (both crates first-published 2026-07-27). The
   recipe used (needs a *verified email* on the account + a prebuilt lib so the
   verification build skips cmake):
   ```bash
   export CARGO_REGISTRY_TOKEN=...          # token from https://crates.io/me
   CRISPASR_LIB_DIR=/usr/local/lib cargo publish --manifest-path crispasr-sys/Cargo.toml --allow-dirty
   sleep 30
   CRISPASR_LIB_DIR=/usr/local/lib cargo publish --manifest-path crispasr/Cargo.toml --allow-dirty
   ```
   For tag-triggered CI, add `CARGO_REGISTRY_TOKEN` repo secret (Settings →
   Secrets → Actions) — CI build.rs will cmake from the checkout, so no
   `CRISPASR_LIB_DIR` needed there.

2. **PyPI — ALREADY DONE** (package first-published manually, `crispasr 0.8.22`
   + `0.8.23`, 2026-07-24). Only optional remaining step for tag-triggered
   republish: at https://pypi.org/manage/account/publishing/ create a pending
   publisher — Owner `CrispStrobe`, Repository `CrispASR`, Workflow
   `release-wrappers.yml`, Environment `pypi`. Manual `twine upload` of a bumped
   version also works without it.

3. **pub.dev (Dart) — ALREADY DONE** (first-published manually to `crispasr 0.8.11`).
   ~~`cd flutter/crispasr && dart pub get && dart pub publish`~~. Only remaining
   optional step: https://pub.dev/packages/crispasr/admin → enable Automated
   publishing, Repository `CrispStrobe/CrispASR`, Tag pattern `v{{version}}` — do
   this only if you want tag-triggered republish (manual `dart pub publish` works
   without it).

4. **Auto-trigger — DONE.** `release-wrappers.yml` now runs on `push: tags:
   ['v*']` and publishes Rust (crates.io) + Dart (pub.dev). Python is handled
   separately by **`release-python-wheels.yml`** (see below), so the PyPI job
   was removed from `release-wrappers.yml` to avoid a double upload.

### Python wheels — `release-python-wheels.yml`

Ships the model I recommended: bundled **CPU wheels → PyPI**, **GPU wheels → a
PEP 503 index on GitHub Pages** (`--extra-index-url .../whl/{cuda,vulkan}/`),
plus a pure-Python **sdist** fallback. It runs on `workflow_run` after
`release.yml` finishes and REUSES the `libcrispasr-<platform>[-cuda|-vulkan]`
bundles that release.yml attaches to the GitHub Release — no native rebuild.
`tools/stage_libs.py` copies the libs into the `crispasr` package,
`_binding.py:_find_lib()` probes the package dir first, wheels are retagged per
platform with `wheel tags`. CPU matrix: linux x86_64 + arm64, macOS arm64
(Metal), windows x86_64. GPU: CUDA (linux + windows) + Vulkan (windows),
carrying `+cuda`/`+vulkan` local versions. Verified end-to-end locally on
macOS-arm64 (install → `_find_lib` picks the bundled dylib → `CDLL` loads).

REGISTRY SECRETS (all set 2026-07-27 via `gh secret set`):
- `PYPI_API_TOKEN` (pre-existing), `CARGO_REGISTRY_TOKEN` (added — crates.io CI
  publish was otherwise skipped; `gh secret list` confirms both).

GPU INDEX HOSTING: Pages already serves `main`/root (legacy Jekyll, the README
landing). The GPU index is committed to `main` under `whl/` by the workflow
(Jekyll serves it at `.../whl/{cuda,vulkan}/`), leaving the README landing
untouched — no Pages reconfiguration. Index links point at Release-hosted
wheels, so main only carries tiny `index.html` files, regenerated from all
releases each run (`[skip ci]` commit, push-with-rebase-retry).

Assumption to verify on the first tagged run: linux wheels are labelled
`manylinux_2_28_*`; if a bundle needs newer glibc, bump the tag (`auditwheel
show`).

## 67. Deferred follow-ups carry-over (mid-May 2026 session)

- [ ] **60d F16 mimo-asr re-upload (HF).** HF F16 (`cstr/mimo-asr-GGUF`) is still legacy
  unfused layout; runtime fallback works but misses the 1.7× fused-QKV per-step decode.
  Needs a fresh BF16→F16 run (killed at 22 min on the 16 GB / 99%-full box, disk-thrash).
  Run on a 32+ GB box with non-99%-full external, then `tools/patch_mimo_asr_fuse_qkv.py`
  patches to fused layout (~5 min).
- [ ] **60e per-backend Q8_0 KV cosine validation.** `CRISPASR_KV_QUANT={f16,q8_0,q4_0}`
  wiring landed across 9 backends (defaults F16, bit-identical until opted in). Only
  mimo-asr diff-harness-validated at q8_0. Remaining 8 (qwen3_asr, voxtral, voxtral4b,
  granite_speech, gemma4_e2b, glm_asr, omniasr, orpheus, qwen3_tts) each need a
  `CRISPASR_KV_QUANT=q8_0 crispasr-diff <backend>` pass (≥0.98 gate) before any
  default-flip. ~5 min each, warm cache, zero code.
- [ ] **Vibevoice CUDA cache reuse re-test.** `backend_needs_fresh_pred_graph()` bypasses
  the pred-head graph cache on Metal+Vulkan+CUDA (CUDA on presumption). On a CUDA box run
  `CRISPASR_VIBEVOICE_REUSE_PRED_GRAPH=1`, confirm no `GGML_ASSERT(src_backend_id != -1)`.
  If clean → drop CUDA from bypass list, recover ~30% per-synthesis caching. If assert
  fires → keep gated off; proper fix = recompute view→backend mapping from
  `view_src->buffer` in `ggml_backend_sched_split_graph`.
- [ ] **SYCL/HIP/ROCm cache-bypass extension.** Same shape as CUDA; extend the
  `backend_needs_fresh_pred_graph()` prefix list when a report comes in or a maintainer
  audits the upstream sched reset path there.
- [ ] **Per-backend `MADV_RANDOM` post-prefill.** `core_gguf::mmap_advise_random()` exposed
  but unused; add one call between prefill and decode loop in `mimo_asr_transcribe` /
  `qwen3_asr_transcribe` / `voxtral_transcribe` etc. when a 32+ GB-box benchmark shows
  benefit (marginal on Q4_K; F16 is where it matters, unmeasurable on 16 GB).
- [ ] **Disk5 cleanup.** `/Volumes/backups` at 99%. Safe to delete local unfused
  `mimo-asr-q4_k.gguf` (superseded by `mimo-asr-q4_k.fused.gguf` + HF fused) once future
  A/B not needed.
- [ ] **CI legacy `build.yml`.** Legacy whisper.cpp matrix (triggers on nonexistent
  `branches: [master]` + `tags: v*`), failing every tag push since v0.4.x. Doesn't block
  (ci.yml/release.yml are the gates). Delete or repair after auditing whether any
  build-matrix combo isn't covered by new ci.yml.

---

## 70. Streaming TTS via chunked VAE decode (latency win, vibevoice / qwen3-tts)

**Effort:** Medium-large. **Status:** not started; design settled.

Goal: a `--stream` TTS mode that emits a 24 kHz PCM chunk every K AR steps
(stdout / HTTP) while the AR loop continues, dropping time-to-first-byte from full
wall-clock to "K AR steps + one chunked-VAE pass". NOT the Intel-Arc Vulkan
workgroup bug (already fixed via CPU fallback `31795a7` / `VIBEVOICE_VAE_BACKEND=cpu`).
geneing's `chunked_vibevoice.patch` (#52) nailed the chunk decomposition but
regressed on per-call `sched_reset`+`sched_alloc_graph` overhead — start there.

### Three pieces required (land in order)
1. **Persistent VAE compute-graph reused across chunks** (mostly mechanical; the
   overhead that killed geneing's prototype). Mirror qwen3-tts `O15` graph reuse
   (`src/qwen3_tts.cpp:1037`): build once at `Lk = max_chunk_latents`, pin topology,
   reuse cached gallocr plan; cost = one `set_rows`-style write op/call, not a
   rebuild. Benchmark to confirm the regression is gone before proceeding.
2. **Causal padding on the σ-VAE conv stack** — left-pad each chunk with previous
   chunk's tail, drop first L output samples, so chunk decode == full decode at
   boundaries (avoids phase artefacts). Ref: kokoro/voxtral4b causal-conv1d paths.
3. **Chunked transfer in HTTP TTS endpoint** — depends on #58 (`POST
   /v1/audio/speech`) landing first; wire cpp-httplib chunked-transfer for
   `Accept: audio/wav; chunked` or `stream=true`.

### Backends in scope
- **vibevoice** (σ-VAE) — primary, largest win (positioned as realtime backend).
- **qwen3-tts** codec decode — 12 Hz vocoder; already has `O15` graph reuse, extend
  to chunked output.
- **kokoro** iSTFTNet — straight-line generator, cleaner chunking but iSTFT inverse
  window has the same boundary-artefact problem.
- Skip orpheus (SNAC already emits 24 kHz PCM single-pass — no win).

### Files
`src/vibevoice.cpp` / `vibevoice_tts.cpp` (chunked decode + graph reuse + causal
pad); `examples/cli/crispasr_backend_vibevoice.cpp` (`--stream` stdout PCM);
`examples/cli/cli.cpp` (`--tts-stream` flag); `examples/server/server.cpp`
(chunked-transfer, after #58); `docs/tts.md`; `LEARNINGS.md` (per-call ggml graph
overhead trap + graph-reuse cure).

### Out of scope for v1
Multi-chunk look-ahead beyond one chunk; kokoro/orpheus chunking as separate items;
any change to AR decoding (only post-AR codec/VAE side is chunked).

## 75-followups. /v1/audio/speech OpenAI round 2 — open

Parent #75 round 1 shipped → HISTORY §81 (PR #63 merged + corrective
batch + 75a/75b + 75d chunking + 75c-opt-1 server-side speed
resampler). Remaining follow-ups: 75c-opt-2 (native-backend duration
knobs) and 75e (streaming / mp3 / upload).

Remaining gaps documented in follow-up items: 75c-opt-2 (native-backend duration knobs), 75e (streaming response, mp3/opus encoding, voice upload/delete).

---

## 81. Nemotron-Speech-Streaming-EN-0.6B — first cache-aware streaming-native ASR

_Completed work archived to HISTORY.md (PLAN compaction 2026-07-17)._

**Still open:** WER benchmarking on standard sets, GPU perf testing, decide whether streaming becomes default path

## 168. GPU scheduler migration — gallocr-only backends

_Completed work archived to HISTORY.md (PLAN compaction 2026-07-17)._

**Still open:** voxcpm2_tts sched migration; re-publish lfm2-audio GGUFs at Q5_K minimum quant

## 91. CrispASR CLI features missing from CrisperWeaver (OPEN — parity gaps)

CLI knobs CrispASR exposes that CrisperWeaver doesn't. None blocking; listed so
the next parity-pass audit doesn't re-discover them. (`--alt N` shipped — see
HISTORY.)

**TO DO (open parity gaps):**
- ~~`--offset-t MS` / `--duration MS`~~ — DONE on **both** dispatch surfaces
  (multi-surface trap — the server has its own slice loop, see
  [[multi-surface-dispatch-trap]]):
  - **CLI** (`crispasr_run.cpp process_one_input`, feat/offset-duration): windows
    the decoded PCM to `[offset, offset+duration)` before VAD/chunking and shifts
    reported segment/word/token timestamps back into original-audio time. Was
    whisper-internal only (`cli.cpp` → `wparams.offset_ms`). Offset-past-end exits
    cleanly. Verified on parakeet-ctc + jfk: timestamps land at 5.0–11.0 s;
    word-level JSON offsets shifted too.
  - **HTTP server** (`crispasr_server.cpp do_transcribe`,
    feat/server-offset-duration): `offset_t_ms` / `duration_ms` form fields were
    parsed but never applied. Same window + shift, applied after the per-slice
    diarize re-walk (which matches segments by the unshifted slice `t0_cs`).
    Verified live via `crispasr --server --backend moonshine` + curl:
    `offset_t_ms=5000` → seg 5.00–11.00; `offset_t_ms=4000&duration_ms=3000` →
    seg 4.00–7.00; offset past end → `{"text": ""}`.

  No session-C-ABI change needed: `crispasr_session_transcribe` is a low-level
  PCM-buffer primitive — windowing is the caller's concern there. Docs in
  `docs/cli.md` + `docs/server.md`. The window arithmetic (initially copy-pasted
  into both surfaces) was factored into `src/core/audio_window.h`
  (`core_audio_window::compute` + `trim`) and unit-tested (`tests/test-audio-window.cpp`,
  9 cases, model-free) so the CLI/server can't drift and the logic is CI-guarded.
  **Still open:** the Dart binding + CrisperWeaver UI rows (out of this repo's scope).
- Whisper decoder fallback knobs (`--word-thold`, `--entropy-thold`,
  `--logprob-thold`, `--no-speech-thold`, `--no-fallback`, `--temperature-inc`)
  — already in Dart binding's TranscribeOptions; just add UI rows + l10n in
  CrisperWeaver Advanced Options. ~half a day.
- Subtitle line formatting: **`--max-len` + `--split-on-punct` already work for all
  backends** (applied post-hoc via `crispasr_make_disp_segments(all_segs, max_len,
  split_on_punct)` in `crispasr_run.cpp`, verified 2026-07-17). Still whisper-only:
  **`--split-on-word`** (referenced only in `cli.cpp`, no `crispasr_run.cpp` hookup).
- `--carry-initial-prompt` — sticky vs reset initial prompt across segments.
  Edge case, ~1 hour.
- ~~`--print-confidence`~~ — DONE (feat/print-confidence-nonwhisper). The flag
  was advertised in `--help` and parsed but did **nothing** for non-whisper
  backends (only the whisper path in `cli.cpp` honoured it) — a silently-broken
  flag, not a missing feature. `crispasr_run.cpp` now prints each segment's
  tokens with an inline `word[NN%]` annotation after the transcript, via a new
  `crispasr_print_confidence` in `crispasr_output.cpp` (gated `else if` after
  the `--alt` printer so the two don't double up). Verified on parakeet-ctc
  (`ans[85%]`) and moonshine (`my[67%] ask[61%] ,[42%]` — low-confidence tokens
  now visible); no flag → single transcript line unchanged. Docs in `docs/cli.md`.
  (JSON/WTS exports already surfaced per-token `confidence`; this closes the
  stdout half.)
- Token suppression (`--suppress-nst`, `--suppress-regex`) — niche,
  whisper-specific; lowest priority.

**Deferred `--alt N` follow-ups (low priority, v1 covers common case):**
beam-search alt capture (siblings ≠ greedy alts, different capture + UX);
full word-level alt enumeration (BPE sub-word: only first content token gets
alts today); alt-picker popover widget test (CrisperWeaver Riverpod + l10n).

## 92. All-backend regression suite (nightly CI)

**Status:** 32 ASR + 21 TTS regression entries, 0 PLACEHOLDERs. CI at
`.github/workflows/regression.yml` runs nightly (04:00 UTC cron) + PR smoke-only.
Matrix: 22 ASR + 7 TTS = 29 backends. First nightly (2026-06-16) green after
fixing 6 failures. Architecture shipped in `tests/regression/`: `manifest.json`
(per-backend GGUF revision SHA + ref path + expected transcript + cosine
thresholds), `run_one.py` driver, `regression.yml`. Fixtures pinned in
`cstr/crispasr-regression-fixtures` (`fixtures.revision` SHA pins the whole set).

**Next steps (each ~1 h/backend):**
1. Flip `skip_diff` on backends where ref archives exist.
2. **Add parakeet-tdt-0.6b-v3** (English) — cache the `.nemo` source locally first
   (`nvidia/parakeet-tdt-0.6b-v3` not on the dev box yet).
3. **Add canary + cohere + kyutai-stt + moonshine** — all have reference modules in
   `tools/reference_backends/`.
4. **Add the TTS family** (kokoro, indextts, qwen3-tts, chatterbox, vibevoice) —
   need WAV-output checksums or ASR-roundtrip rather than transcript equality.
5. **Promote to release gating** — hook into `release.yml` pre-publish job so a
   regression aborts the tag.

## 95. IndexTTS-1.5 Chinese TN — binary alternative to the Python `wetext` hook (OPEN)

Today CrispASR ships `INDEXTTS_TEXT_NORMALIZER=<shell cmd>` + `tools/wetext-normalize.py`
(commit `1bfe7c5a`) — covers users who already have Python + wetext. No-Python deployments
(single-binary, Windows w/o Python, embedded) need a native path. Do NOT start speculatively.

**Gap:** default in-process `preprocess_indextts_text()` handles CJK char split, a subset of
`char_rep_map` punctuation, ASCII upper-case. Missing vs full `wetext.Normalizer(lang='zh',
operator='tn')`: Arabic-numeral→hanzi, pinyin tone-digit restoration, dates, times, currency,
phone numbers, math/measurements/fractions/percent, EN contractions in Chinese. This isn't
cosmetic — the model often fails to emit `stop_mel_token` on un-pronounceable digit inputs and
burns `max_mel_tokens=600`, so digit-containing prompts break without TN.

Options (in preference order — implement per trigger below, not all):
- **95a. Hand-roll high-leverage rules in C++** (recommended first). digit-string→hanzi for
  the 1-billion range (`零一二三四五六七八九` + `十百千万亿`), `年/月/日`, `点/分` time,
  pinyin tone-digit lookup. ~300–600 LOC, no deps, covers ~90 % of prompts. Extend
  `src/indextts.cpp:preprocess_indextts_text` with a `normalize_chinese_numbers()` pass +
  golden string-in/string-out tests. **Effort: ~1 day.**
- **95d. Tiny FST reader in own C++** — consume upstream `pengzhendong/wetext` `.fst` files
  (`fsts/zh/tn/tagger.fst` + `verbalizer.fst`, ~1 MB) without linking OpenFST. Parser +
  Tropical-semiring traverser + symbol-table loader + port of wetext `token_parser.py`
  (~200 LOC). Total ~500–800 LOC, zero deps. Byte-stable vs upstream except FST features not
  implemented. New `src/indextts_zh_tn.{h,cpp}`; invoke via `INDEXTTS_TEXT_NORMALIZER=native`.
  **Effort: 2–4 days.**
- **95b. Vendor `kaldifst` + OpenFST + ship compiled `.fst`** — byte-identical to upstream.
  `third_party/openfst` (~30–50 K LOC) + `third_party/kaldifst` (~5 K LOC), both Apache-2.0;
  real build-profile cost. Invoke via `INDEXTTS_TEXT_NORMALIZER=wetext`. **Effort: 3–5 days +
  ongoing submodule maintenance.** Only as fallback if 95d's FST gaps keep biting.
- **95c. PyInstaller-bundle the Python sidecar** — ~50–80 MB bloat, per-platform pipeline.
  Almost certainly not worth it; listed so nobody re-discovers it.

Not alternatives (already surveyed as insufficient): ICU `Transliterator`, `cn2an`/`pypinyin`
(no C++ port), HF `tokenizers` normalizers.

**DECISION-GATE — start only when one of:**
1. A user files a digit/date/pinyin-tone-digit prompt that breaks audio → do 95a, smallest
   rule set that fixes the reported case.
2. Hand-rolled list reaches 2–3 entries (use case confirmed alive) → 95d becomes the right
   next step rather than letting 95a grow into one-off rules.
3. Only if 95d's unimplemented FST features keep biting → fall back to 95b. Never vendor
   OpenFST speculatively.

---

## 102. RapidTP-Aligns — dedicated NN timestamp predictor (survey)

Status: survey-only. RapidAI ships `RapidTP-Aligns` as a standalone timestamp-only model that
predicts timestamps **from audio alone** (no ASR text). Would be a third path alongside our native
decoder timestamps (whisper/parakeet-TDT/canary/cohere/kyutai-stt) and CTC forced aligners
(`canary-ctc-aligner.gguf` / `qwen3-forced-aligner.gguf`, which need the text as input).

Value when: cross-backend timestamp consistency independent of which ASR ran; trustworthy segment
boundaries even when ASR text is wrong (diarization/VAD post-proc); single-forward-pass
end-of-utterance/silence for streaming.

TODO (open questions to resolve before starting):
1. What architecture upstream ships (README is one Chinese line; likely small Conformer-CTC over raw audio emitting frame-level boundary labels).
2. License + upstream weights (ModelScope-hosted? FunASR derivative?).
3. Quality vs our existing CTC aligners (`canary-ctc-aligner-q4_k.gguf` ~80 MB is fast+accurate; without a clear margin this is incremental).

**Trigger (stays survey-only until one fires):** a user reports the CTC aligner failing on a specific
audio class (heavy code-switch, multi-speaker overlap, music behind speech); OR we add a streaming
ASR endpoint needing sub-100-ms end-of-utterance prediction (currently a VAD silence heuristic).

---

## 104. Stateful frame-streaming TDT decode for parakeet long-form (issue #89)

**Priority: HIGH** — auto path (no `--vad`, no `--chunk-seconds`) tops out at
~82 % coverage on 60 s Japanese audio; users expect >95 %. Root cause is the TDT
decoder cold-starting each independent chunk (`kLongAudioFallbackChunkSeconds=30`
reinitializes the LSTM per chunk), losing 5–20 s of content per chunk interior —
not boundary stitching. Fix: keep TDT LSTM predictor state across frames like
NeMo's `BatchedFrameASRTDT` (stateful decoder, 4 s rolling buffer, running z-norm).

### TO DO

**Phase 1 — stateful TDT decode (core change, ~200–300 LOC; hot loop <50 lines,
LSTM state threading is the work).** Split API already exists (`parakeet.h`:
`parakeet_encode`, `parakeet_decode_frames`).
1. Add `parakeet_decode_frames_stateful` — like `parakeet_decode_frames` but
   accepts/returns LSTM hidden state (`parakeet_lstm_state`). The TDT loop at
   `parakeet.cpp:1003` already uses `state`; make it an in/out param instead of
   initializing to SOS.
2. Add streaming mel with running z-norm — maintain running mean/variance across
   frames (NeMo `get_norm_consts_per_frame`) via a new
   `parakeet_mel_streaming_context`.
3. Wire into `crispasr_run.cpp`: when long-audio fallback triggers and backend is
   parakeet/canary, use streaming decode (per 4 s frame: streaming-mel → encode →
   decode_frames_stateful(&lstm_state) → merge with LCS) instead of independent
   chunk transcription.

**Phase 2 — tuning (benchmarking).**
4. Frame-size sweep (1.6s, 2s, 4s, 8s) on benchmark corpus.
5. Running z-norm warmup: first frame per-frame z-norm, later frames EMA.
6. LCS delay tuning: `lcs_delay = (buffer - frame) / model_stride`.

### Files
- `src/parakeet.h` — add `parakeet_decode_frames_stateful`,
  `parakeet_mel_streaming_context`
- `src/parakeet.cpp` — LSTM state in/out, streaming-mel helper
- `src/core/mel.{h,cpp}` — running z-norm mode
- `examples/cli/crispasr_run.cpp` — streaming decode path in `process_one_input`
- `examples/cli/crispasr_backend_parakeet.cpp` — streaming transcribe method
- `tests/test-issue-89-long-audio-fallback.cpp` — pin streaming path activation

### Effort: Medium-Large.
### Success: `python tests/benchmark_asr.py --audio yt_60s.wav --backend parakeet-ja --settings auto` reports **coverage ≥ 95 %** on the #89 JA audio, without `--vad`.
### Trigger: immediate — #89 open, current fix is partial mitigation (prevents 0-output but doesn't match NeMo quality).

## 105. WhisperX word alignment models — wav2vec2 CTC zoo

_Completed work archived to HISTORY.md (PLAN compaction 2026-07-17)._

**Still open:** Benchmark new aligners vs canary-ctc. (`-am` aliases now documented
in `docs/cli.md` — canary-ctc, wav2vec2 [12 langs], fastconformer [18 langs],
qwen3-forced.)

## 106. TEN-VAD — low-latency cross-platform VAD

**Status:** feasible-but-BLOCKED-on-license; no code. Fourth VAD backend candidate
alongside Silero, FireRedVAD, MarbleNet, Whisper-VAD-EncDec. Upstream ships
cross-platform C bindings, prebuilt libs (Linux/macOS/Windows/Android/iOS/Web), and
an ONNX path; runtime target 16 kHz, 10/16 ms hop — fits our VAD surface. Aimed at
low-latency streaming turn detection, lighter than Silero.

**Decision-gate:** upstream license is Apache 2.0 **plus additional
no-compete/own-app-only conditions** — treat distribution as BLOCKED until legal
review or an explicit internal-only use case. Do NOT wire or ship until that's
resolved.

**Implementation plan (once unblocked):**
1. Decide prebuilt-native-lib path vs ONNX path (or both).
2. Add `ten-vad` alias in VAD registry + CLI so `--vad -vm ten-vad` works.
3. Add auto-download metadata for the chosen artifact(s); keep Silero default.
4. Run the boundary benchmark vs Silero + FireRedVAD on the short-gap /
   sentence-end test set.
5. Document 16 kHz-in sampling-rate handling (resample other inputs first).

**Trigger:** a user wants lower-latency/lower-footprint VAD than Silero, OR we want
a 4th native cross-platform backend — AND the license review clears (or an
internal-only path is confirmed).

## 125. Issue #125 — multi-backend bug sweep from montvid (mostly DONE — validation + longer-term threads open)

12 findings from user `montvid` (RTX PRO 6000 Blackwell sm_120, CUDA 12.6) on
CrispASR v0.6.10 `eaee2319`. All fix commits landed and M1-Metal-smoked; remaining
work is GPU validation + a few longer-term root-causes. Reports cached at
`/Volumes/backups/code/issue125-attachments/`.

**Open TODO items:**
- **P0 mimo-asr Blackwell segfault — hardening shipped (`a5a518c8`), needs GPU
  confirmation.** Real cause was `0f0f0793` (ggml-backend src-mutation
  log/restore), NOT the FA-mask commit. Ask montvid to rebuild from `95d74455`+
  and rerun `--backend mimo-asr -m auto --auto-download -f samples/jfk.wav -l en
  -np -nt`; expect the v0.6.9 reference transcript. If segfault persists: `gdb`
  backtrace; next suspects are Metal-debug commits leaking to CUDA (unlikely) or
  a Blackwell-specific ggml-cuda bug.
- **P1 funasr (longer-term):** root-cause the audio adaptor/encoder collapse on
  Blackwell CUDA (log `frames_spliced` in `funasr_init_from_file`; if 0 on 11 s
  JFK the adaptor is the failure; ideally diff adaptor output vs upstream FunASR
  Python ref). Loop guard + `-l` wiring already shipped.
- **P2 firered-asr (if it resurfaces):** the "JFK silence-only `<Sil>!` without
  --vad" subtask (suspected vocab/blank-id mismatch in auto-downloaded GGUF) was
  not reproduced locally; reopen only on a new report.
- **P4 gemma4-e2b (longer-term):** add init sanity logs `audio_soft_token_id`,
  `proj_dim` vs `d_model`, "audio projection weights found". Chunking + prefers_vad
  already shipped.
- **P6c kyutai-stt — DEFERRED:** streaming model on batch dispatcher is a
  footgun; a `--force-long-audio` cap is now UX-nice only (P6b bounded the
  wallclock). Defer until a user reports.

**Cross-finding open sweeps:**
- Wider GPU validation matrix before any future `ggml-cuda/fattn*` change: add
  mimo-asr, glm-asr, gemma4-e2b, voxtral, granite (multi-head audio-LLM backends).
- Do NOT submit `tools/upstream-prs/06-cuda-fa-perhead-mask.md` upstream until
  that wider matrix validation lands.
- Audit all 18 registered backends in `tools/test-all-backends.py` for silent
  staleness (4 were missing entries before this sweep).
- Defensive sweep of every `CAP_UNBOUNDED_INPUT` declaration — confirm the
  *encoder* is genuinely unbounded, not just the dispatcher's input shape.

**Completion triggers:** mimo-asr next release no longer segfaults on Blackwell
(montvid or same-GPU-class validation); 5-min EN clip clean on firered-asr /
omniasr-llm / gemma4-e2b.

Reporter contact: `montvid`, GitHub #125.

## 127. Coverage gaps from the 2026-05-26 overlap-save sweep close-out

Three small holes the sweep + #115 bisect surfaced. None are urgent; recording so the next contributor doesn't rediscover the same gaps.

### a. omniasr-llm — overlap-save bug status unknown

The original 5 min sweep and the 90 s rerun both came back `BOTH_EMPTY` for `omniasr-llm-300m-v2-q4_k.gguf`: default and `--chunk-overlap 0` both hit the 20 min per-pass wallclock on M1. Probably *slow*, possibly *also has the truncation bug like its sibling backends*. Can't tell without a faster box.

**Fix shape.** Re-run `./tools/check-overlap-save-bug.sh omniasr-llm` with `PER_RUN_TIMEOUT=2400` on a Linux x86 host (the VPS) or a Kaggle CPU kernel. If default produces materially less output than no-overlap, add to the opt-out list in `examples/cli/crispasr_chunk_context_gate.h`. If both produce the same content, mark VERIFIED-OK in the harness comment.

### b. mimo-asr — local test coverage is in place but doesn't run in CI

`tools/test-all-backends.py` has had a `mimo-asr` registry entry since 2026-05-02 (commit `2aeaf4c4`); the `test_transcribe` function explicitly handles `EMPTY` output (line 693-695). Locally the test would have caught PLAN #115's silent-empty regression at runtime — but CI doesn't run `test-all-backends.py` against large-model backends (mimo Q4_K is 4.2 GB, doesn't fit in the standard runner disk budget per pre-release), so the regression shipped in v0.6.10 anyway.

**Fix shape.** Either (a) Kaggle scheduled-CI workflow that runs the full `test-all-backends.py` against the 4 LLM-class backends (mimo, voxtral, gemma4-e2b, granite-4.1-2b) on each main push — patterns in `tools/kaggle/crispasr-regression.py` already handle the model-download + heartbeat parts; (b) cheaper, a documented `make smoke-llm-backends` target that release scripts run before tagging. (a) is more reliable; (b) is one afternoon of work.

### c. cohere-asr-ja-v0.1 — no benchmark numbers in PERFORMANCE.md

Issue #123 added the JA variant to the registry + README, but no row in any of `PERFORMANCE.md`'s cohere tables (the long-form coverage at line 1374, the cross-backend matrix at line 1535, the per-length wall-time table at line 1555). The English `cohere-transcribe` is benchmarked across multiple Japanese / English / multilingual clips; the JA fine-tune isn't.

**Fix shape.** Run the JA variant on the same fixture set the English one used (TedX / JSUT clips per the model card; `samples/jfk.wav` is English so won't exercise the JA tuning). Drop one extra row into each cohere table with the JA numbers. ~30 min of inference + table updates once the fixtures are downloaded.

---

## §136 — funasr CUDA !-loop fix (issue #125)

_Completed work archived to HISTORY.md (PLAN compaction 2026-07-17)._

**Still open:** File upstream ggml issue with minimal repro; revert weight split if fixed

## §138 SpeechT5 + Dia + Parler + FastPitch TTS stubs → working backends

**Status:** Dia CPU full speech is verified (2026-10-02); SpeechT5 still has a decoder content mismatch; Parler + FastPitch not started.

### SpeechT5 TTS (microsoft/speecht5_tts)
- Encoder verified (cos > 0.999 all 12 layers). Runtime `src/speecht5_tts.cpp` = encoder + decoder w/ KV cache + postnet + HiFi-GAN. GGUF `/mnt/storage/speecht5/speecht5-tts-f16.gguf` (300 MB). Converter `models/convert-speecht5-to-gguf.py`.
- **TO DO:** decoder content mismatch — validate decoder per-layer against the Python reference.

### Dia 1.6B TTS (nari-labs/Dia-1.6B)
- **DONE 2026-10-02:** fixed cross-attention Q/K RoPE, nucleus threshold inclusion and delayed-BOS timing. Independent pinned F32 source versus native F16 passes all 127 steps (cosine and magnitude); all 126 real-feedback inputs match, with the old BOS timing rejected by a negative control.
- F16 default and Q8 CPU full-speech roundtrips both pass WER 0. Q8 at 1/4/8 threads produces identical audio for each of two prompts. Model capacity replaces the 200-step cap; C ABI and CLI explicit limits/reset are verified. Shared CLI/server planning preserves the whole Dia dialogue.
- Proof: [full-generation receipt](docs/dia-full-generation-2026-10-02.json), and HISTORY. Q8 does not pass the strict F16 numerical threshold; its separate speech gate does pass. Metal compiles, but hosted Mac VMs supplied no usable physical GPU runtime proof.

### Parler TTS / FastPitch — NOT STARTED
- FastPitch: ~1000 LOC stub, converter exists, needs NeMo model.
- Parler: ~857 LOC stub w/ 3 TODOs, T5 encoder + DAC decoder.

## §139 Beam search — remaining ASR backends (issue #136 follow-up)

_Completed work archived to HISTORY.md (PLAN compaction 2026-07-17)._

**Still open:** mimo-asr beam (blocked on PLAN #115) and lfm2-audio beam (needs KV+conv save/restore) still API stubs

## §TTS-PROV: TTS AI-provenance compliance (watermark, C2PA, consent, disclaimer)

_Completed work archived to HISTORY.md (PLAN compaction 2026-07-17)._

**Still open:** AudioSeal diff-harness validation per status header

## 156. Permissive G2P phonemizer

Replaces GPLv3 espeak-ng static link with modular permissively-licensed
phonemization. **Phase 1+2 DONE (2026-06-07):** header-only `core/g2p_{en,de,fr,es}.h`
(LTS rules + IPA-dict loaders + neural GRU G2P for EN), `espeak_dlopen.h`,
`phonemizer.h/cpp` cascade with auto-download from HF `cstr/g2p-dicts`, wired into
`piper_tts.cpp`, 202 unit assertions + 4 live TTS→ASR roundtrips. Cascade:
pre-gen IPA dict → builtin CMUdict+neural+LTS → OLaPh MIT dicts → espeak dlopen →
espeak popen. See HISTORY for coverage table + phoneme-inventory fixes.

### Phase 3 — open
**a. More languages (need LTS rules; OLaPh MIT dicts already exist):** Portuguese,
Italian, Dutch, Swedish, Czech, Danish, Finnish, Polish. Japanese/Chinese/Korean
need rules from piper-plus. Optionally port piper-plus (ayutaz/piper-plus, MIT,
branch dev) FR (1197 lines) + ES (620 lines) — more thorough than ours (NFD norm,
PUA mapping, syllabification, stress). No German in piper-plus — our `g2p_de.h`
fills that.
**b. GGUF-embedded dicts (TTS.cpp pattern):** embed phonemizer rules /
CMUdict / neural weights per-model via `phonemizer.rules.keys` /
`phonemizer.rules.phonemes` arrays for zero runtime external deps.
**c. Gruut CRF (MIT, rhasspy/gruut):** dict + CRF G2P (18 MB SQLite + CRFsuite/BSD)
for higher-quality OOV (compounds/loanwords), langs de/en/fr/es/it/nl/pt/ru/sv/cs/
ar/fa/sw. Port: extract SQLite lexicon + CRFsuite model, write ~100-line C++
feature extractor, link libcrfsuite. Lower priority now OLaPh covers most words.
**d. Neural G2P weight distribution:** publish standalone `g2p_en.json` (MeloTTS v3
`melotts.g2p_en_json`, base64 ~4 KB); loader already in `g2p_en.h`.

### Files
`src/core/g2p_{en,de,fr,es}.h`, `src/espeak_dlopen.h`, `src/phonemizer.{h,cpp}`,
`tests/test-g2p-{en,de,fr,es}.cpp`, `tests/test-espeak-phonemize.cpp`,
`tests/test-piper-roundtrip.sh`.

## §167 — Beam search, MAES, and KV caching expansion

_Completed work archived to HISTORY.md (PLAN compaction 2026-07-17)._

**Still open:** §167e-h (granite-nle/mimo/moss/lfm2) beam need A/B validation on real models

## §176 Runtime optimization pass — 18/20 DONE, 2 open

16 items DONE at the 2026-06-20 audit (§176a,b,d–j,m,o–t); §176k + §176n later resolved.
Full audit → PERFORMANCE.md "Runtime optimization audit — 2026-06-20". Open:

#### §176c Migrate host-side KV to device-resident 4D tensors — OPEN but LOW-VALUE

**Do NOT implement without first measuring the KV round-trip fraction on the specific
backend** (`DIA_BENCH`-style instrument). Dia measured at ~1.2% of decode → DEFERRED,
not worth the non-bit-identical, high-risk device-KV rewrite (ggml KV write/read
ordering on Metal is the codebase's most bug-prone pattern). Compute-bound transformer
decoders are NOT KV-bandwidth-bound — the "dominant bottleneck" premise is wrong.
Genuinely-still-host-side backends (verified 2026-07-12 by code read):
- **SpeechT5** self-attn KV — host `std::vector<float>` that grows per step
  (`speecht5_tts.cpp:246-265`), re-uploaded whole every step (`:1035-1041,1084`).
  Cross-attn KV already device-resident (§202). Measure fraction first (likely <5%).
- **Pocket-TTS** self-attn KV — host, pre-sized to `max_seq` (doesn't grow,
  `pocket_tts.cpp:311-319`), but reordered past window re-uploaded per step
  (`:1207-1228`). Measure first.
- (Dia — measured ~1.2%, deferred. VoxCPM2/Parler/LFM2/KugelAudio already device-resident — do NOT re-chase.)
Approach if ever justified: IndexTTS/CSM 4D on-device `[head_dim, max_ctx, n_heads,
n_layers]` + `ggml_view_4d`/`ggml_cpy` writes; keep both paths gated; expect
low-single-digit-% ceiling. Effort: Medium/backend, expected <2% payoff — deprioritize.

#### §176k FireRed ASR self-attn — MOSTLY DONE, residual OPEN (LOW)

Shipped the real lever (env-gated matvec graph cache `CRISPASR_FIRERED_MATVEC_CACHE`,
default ON, bit-identical, pure Pareto). Residual: self-attn KV is a growing
`std::vector<float>` with scalar O(T²) scoring loop — for very long single-pass decodes
(hundreds of tokens) a pre-alloc 4D device KV + BLAS/ggml scoring could help there.
**Measure on a long clip before investing** — NOT the highest-impact lever for typical
clips. File: `src/firered_asr.cpp`.

#### §176l Kyutai STT: vectorized RVQ encode — MOSTLY DONE, remaining = default-flip

Fast path shipped gated: `rvq_encode_group` (`kyutai_stt.cpp:768`) calls
`core_rvq::encode_euclidean_per_stage` when `CRISPASR_KYUTAI_RVQ_FAST=1` (**default OFF**);
non-uniform dim / failure falls back to scalar. Helper + extraction/transpose unit-tested
(`test-core-rvq`, LABELS unit) — codes identical to scalar reference. Only end-to-end on a
real model is unverified.
- [ ] **To flip default:** run kyutai on a Kyutai STT GGUF (clip via
  `tools/kaggle/kyutai-stt-2.6b-convert`) with flag on vs off, assert emitted RVQ codes
  byte-identical, then measure speedup. Until then scalar stays default + reference.
  Effort: Small (one-clip validation). Files: `src/kyutai_stt.cpp`, `src/core/rvq.{h,cpp}`.

---

## §ARK — ARK-ASR-3B support (⚠️ EXPERIMENTAL / WIP; branch feat/arkasr-3b)

_Completed work archived to HISTORY.md (PLAN compaction 2026-07-17)._

**Still open:** CUDA validation (only Metal validated); nice-to-have

## llama.cpp comparison — perf-tricks to adopt (ANALYSIS / OPEN)

Status: comparative study (2026-07-03) of what CrispASR shares with the ggml-org ecosystem
(whisper.cpp + llama.cpp `libmtmd`). Framing: we're *siblings* on ggml — kernel-level wins
(flash-attn, CUDA MMQ/tensor cores, CPU tinyBLAS/sgemm, Metal `mul_mm`, Vulkan coopmat, k-quants)
are inherited free by keeping ggml sync current; llama.cpp's real advantages are at the
orchestration layer (imatrix, mixed-precision quant CLI, speculative decode, continuous batching).
Full comparison tables archived to HISTORY. Actionable adoption items:

### Tier 1 — real gaps
1. **imatrix + per-tensor quant overrides — DONE, one loose end.** `crispasr-quantize` takes
   `--imatrix <file>` + iq4_nl/iq4_xs + requant-from-q8_0, and `--tensor-type <regex>=<type>`
   (llama.cpp-parity). Producer `src/crispasr_imatrix.{h,cpp}` (set `CRISPASR_IMATRIX_OUT`) installed
   on decode scheduler of the major ASR-LLM backends; A/B harness `tools/imatrix_ab.py` gates on
   transcript **CER** (cosine only tiebreaks). CC0 calib set at `cstr/crispasr-imatrix-calib`
   (+ `tools/imatrix-calib/`); Kaggle kernel `tools/kaggle/imatrix-quant/`. See `docs/quantize.md`.
   **OPEN:** wire the collector into the remaining (non-ASR-decoder) backends if we ever want imatrix
   there. **Corpus rule:** diversity + in-distribution + language coverage is decisive — a
   narrow/mismatched corpus makes imatrix *worse*; use CC0 Common Voice (clean license).
2. **FA default-on audit.** Upstream flipped flash-attn to baseline. Re-check our two known FA bugs —
   batched-FA corruption (§176h) and the Vulkan GQA-REPEAT-f16 path — against the current
   `FLASH_ATTN_EXT` in vendored ggml; some may already be fixed upstream.
3. **Keep ggml sync current — kernel wins are free there.** Verify pinned ggml is recent enough for
   Metal `mul_mm` (M1 encoder prefill), Metal FA with head_size_k≠head_size_v (#12612 — matters for
   partial-RoPE ASR encoders: ark rot32/hd64, higgs, parakeet), CPU RMSNorm+MUL fusion, MMQ, tinyBLAS.

### Tier 2 — conditional on serving story
4. **Symmetric quantized-KV (q8_0)** for Qwen-family audio-LLM decoders (ark, higgs-stt,
   moss-transcribe). Near-lossless, shrinks decoder KV. Must be symmetric K==V type or it silently
   drops off the fused FA path. Validate against per-head KV-stride bugs (#171) first.
5. **Continuous batching** if the server ever needs concurrent transcription. Bespoke single-stream
   decode can't multi-slot today; #171 shows per-slot KV isolation is a real hazard.
6. **Prompt-lookup / n-gram speculative decoding** for AR ASR heads (transcripts echo their own
   context → high acceptance). Heed §161: the draft must reproduce the target's exact realization or generation derails.

**Don't bother:** i-quants (low-bit breaks audio backbones; slow codebook decode on compute-bound
TTS/ASR), KV defrag (deprecated upstream; single-pass audio doesn't fragment), paged attention (not merged upstream).

**Don't converge to them on:** model breadth (~60 vs ~8 archs), per-model long-audio chunking
(overlap-merge beats their fixed-30 s), diffusion/flow TTS with batched CFG, neural diarization, voice cloning, single-file arch-autodetect UX.

### Verify-in-tree caveats before mirroring anything
- `ggml_rope_ext` arg order / `GGML_ROPE_TYPE_*` enum values vs pinned `ggml/include/ggml.h` (revised upstream).
- Current speculative-decode flag namespace (`--spec-*` vs older `--model-draft/--draft-max`).
- mtmd mel params (128 HTK bins) + LFM2-Audio/MiniCPM-o audio status are DeepWiki-sourced — confirm from `clip.cpp` if load-bearing.

---

## 221. Issue #89 hardening + v0.8.8 release

The #89 fix (VAD slice cap + per-slice single-pass + gap-fill) is shipped and
manually verified (see HISTORY + LEARNINGS). This section makes it durable and
gets it released. All subitems OPEN.

### 221a. CI regression guard for the JA long-form path — HIGH, small
Coverage is protected only by manual runs; next parakeet refactor could silently
re-break #89. Drive a live test from the reazon baseball fixture
(`cstr/crispasr-regression-fixtures` → `parakeet-tdt-0.6b-ja/reazon_baseball_14s/
audio.wav`): concatenate ×3 (42.2 s — crosses 30 s auto-chunk + 12 s slice cap),
transcribe via the **session ABI** (carries cap + gap-fill), assert (a) 岡本 ≥3×,
(b) last timestamp reaches the third repetition, (c) a byte floor. Wire into
`tests/` + `tests/env-live-tests.sh` (`CRISPASR_MODEL_PARAKEET_JA`,
`CRISPASR_FIXTURE_PARAKEET_JA`).

### 221b. HTTP server path audit + mirror — HIGH, small-medium
`crispasr_server.cpp` has its own slice loop (`crispasr_compute_audio_slices`
~line 410) separate from the CLI dispatcher and session ABI. Audit whether a JA
parakeet request via `/v1/audio/transcriptions` gets the slice cap + gap-fill; if
not (expected), mirror the policy — consult the backend's
`vad_slice_cap_seconds()` like `crispasr_run.cpp`, or route server parakeet
handling through the session-ABI path.

### 221c. Vulkan sanity run — MEDIUM, small
#89 reporter is on AMD/Vulkan (encoder instability worse there). Fix is
policy-level (slicing) so should transfer, but confirm once via MoltenVK on M1
(`GGML_VULKAN=ON` build, `VK_ICD_FILENAMES` gotcha — see LEARNINGS): yt_60s
default run should land ≈97 % recall like Metal.

### 221d. q4_k registry/UX guard — MEDIUM, small
parakeet-ja q4_k TDT decode is degenerate (repetition loop; pre-existing) while
CTC over the same file is clean. Two guards: (1) check what
`-m auto --model-name parakeet-ja` resolves to in
`src/crispasr_model_registry.cpp` — if q4_k, point at q8_0 (TDT byte-identical to
F16); (2) stderr hint when a JA parakeet GGUF with ≤q4 weights loads in TDT mode:
suggest `--parakeet-decoder ctc` or the q8_0 file.

### 221e. v0.8.8 release — after a–d
Everything since v0.8.7: #89 series (slice cap, gap-fill, session mirror, tools,
CTC-head GGUFs), #218 moss fixes, #217 --align-only docs. Per dev-guide release
process: notes from `git log v0.8.7..HEAD --oneline --no-merges` into
RELEASE_NOTES_v0.8.8.md; `scripts/bump-version.sh 0.8.8`; push main; wait green
CI; push tag; `gh release create` with notes; then remove the repo-root notes file.

## §222 Aligner model expansion — permissive-license fleet (OPEN)

_Completed work archived to HISTORY.md (PLAN compaction 2026-07-17)._

**Still open:** parakeet-ja CTC aligner upload, first-word t0 clamp, OWSM-CTC eval

## §224 Issue #222 follow-ups — silero LID ggml graph (DONE) + diarize/firered CPU perf (OPEN)

_Completed work archived to HISTORY.md (PLAN compaction 2026-07-17)._

**Still open:** Vulkan RADV verify, TitaNet batch/F16, chatterbox_campplus scalar cure.
(openvoice2 + firered_vad scalar cures already DONE — both use Accelerate `cblas_sgemm`
gated by `CRISPASR_OV2_FORCE_SCALAR` / `CRISPASR_FIRERED_VAD_FORCE_SCALAR`, verified 2026-07-17.)

## §226 irodori-tts GPU + codec GGUF fix

_Completed work archived to HISTORY.md (PLAN compaction 2026-07-17)._

**Still open:** baseline JA generation quality WIP, DiT graph caching perf follow-up

## §227 starling comparison — CUDA decode-loop optimization options (OPEN)

sims1253/starling (CUDA-graph ASR inference, RTX 5090, bf16) benchmarks CrispASR and overlaps
our fleet (granite-speech, parakeet-tdt-v3, MOSS-Transcribe, qwen3-asr, ARK-ASR-3B,
higgs-audio-v3-stt, cohere-transcribe — several via our cstr/* GGUFs). Their claim: stock
transformers decode is launch-bound (GPU ~10% busy) → capturing decode steps + multi-step
token loops into CUDA graphs gives 27–1180× vs 3–66× stock, byte-identical, WER-verified.
Repo cloned to ~/code/starling (read 2026-07-06).

**Benchmark-fairness (cheap, reputational):** their CrispASR adapter
(`benchmarks/engines.py:722`, `scripts/bench_qwen3_crispasr.py`) times ONE full CLI subprocess
per clip — incl. process start + multi-GB F16 GGUF disk load + CUDA weight upload every rep —
while starling/stock numbers exclude model load and use warm reps. Cold-start seconds get
labeled engine speed. No crispasr numbers published yet, but the columns could appear anytime.
- [ ] Contact author: benchmark `crispasr-server` (resident model, matches their server mode)
      or parse our stderr phase timings; offer setup help / a resident-mode adapter PR.
      (**The doc to point them at now exists — see below.**)
- [x] **DONE** — `docs/benchmarking.md`: the fair-measurement contract (measure transcribe
      time, not cold start), three methods (server resident-model, in-process
      Session/ctypes = the apples-to-apples path, CLI-with-parsed-stderr-line),
      proof-of-work rules (non-zero exit/empty = FAIL; scale check; warmup + median +
      absolute ms), identical-load discipline, required reporting fields, and the
      phase-timing env vars (`CRISPASR_VERBOSE`, `CRISPASR_<BACKEND>_BENCH`,
      `CRISPASR_METAL_PROFILE`, `CRISPASR_FC_PROFILE`). Links the existing
      `tools/benchmark_asr_engines.README.md` rather than duplicating it, and is linked
      from README's doc index so third-party benchers actually find it. Key fact it
      documents: the CLI/server stderr line `transcribed Xs audio in Ys (Zx realtime)`
      already excludes model load (timer starts after init/decode/VAD) — so benchers
      should parse it instead of wrapping the process in `time`.

**DECISION-GATE — option 0 first, gate options 1–4 on it:** one Kaggle T4/P100 run measuring
GPU-busy% during granite/qwen3-asr AR decode (`nvidia-smi dmon` or nsys). If ggml decode is
already 60%+ busy, options 2–4 are duds (cf. §210 Metal ICB 1.8%, vibevoice CPU cache A/B 0%).
Only if ~10–20% busy (starling's stock baseline) → proceed.

Optimization options (ordered by evidence):
1. **Keep per-step graphs capture-friendly** (mostly done, free). ggml-cuda already captures
   stable per-step graphs. Audit other AR backends where a decode loop shares a sched with
   helper graphs (EOS classifiers, connectors) — alternating graphs on one sched defeats
   gallocr + capture. (#171 per-purpose dedicated scheds already fixed pred head / per-KV-path
   LM steps.)
2. **Multi-step token-loop unroll** (the real starling edge, CUDA-only). Unroll K greedy steps
   into one graph: in-graph `GGML_OP_ARGMAX` → get_rows embedding feedback → static per-step KV
   positions; host sync per-K instead of per-token; EOS checked per block; byte-exact for
   greedy; stable topology per (n_past bucket, K). Substantial engineering, EXACTLY the
   cached-graph minefield of #171/#184/#220 (invariant: one graph per sched, or
   last-allocated-only). **Do not start before option 0 numbers exist.**
3. **Self-speculative draft from CTC head (granite only).** Draft from granite's encoder CTC
   head, verify with the LLM — no extra model; we already have granite CTC infra. Caveat: their
   batched spec at B≥16 loses (0.76×); B=1 spec is the win.
4. **Fused RMSNorm/SwiGLU steps** — ggml-cuda already fuses some; only audit if option 0 shows
   launch-bound decode with capture ON.

Anti-options (skip — negative results transfer): INT8 weight-only quant on CUDA decode (SLOWER,
launch-bound not bandwidth-bound; quant is a CPU/Metal win only); shape-bucketed graph caches
without eviction (per-clip capture cost depresses RTFx at high shape diversity); torch.compile
encoder fusion (not byte-exact — fp32 upcast + BatchNorm amplification).

## §229 transcribe.cpp perf audit — vs handy-computer/transcribe.cpp (OPEN)

Static code read (not benchmarks) of the direct ASR peer `handy-computer/transcribe.cpp`
(ggml C/C++ STT, heavy roster overlap). They have the more performant core ASR
*engine* (CPU, flash, AR decoder, on-target encoder, features, streaming latency)
via uniform shared-infra; we win quantization QUALITY (imatrix, unmatched),
VAD-gated long-form robustness, and breadth (ASR+TTS). Full scoreboard + file:line
findings in HISTORY. Every port gated on decoded-output A/B (§176/§210/§227 rule).

### Actions (ranked)
0. **GATE on measurement (do first).** One M1 + one T4/CPU encoder-RTF A/B for
   wins 1-2. Port each only if it moves the encoder ≥5%.
1. [x] **Force `GGML_LLAMAFILE ON`** — DONE (verified 2026-07-17). `CMakeLists.txt:199`
   `set(GGML_LLAMAFILE_DEFAULT ON)` overrides ggml's default-OFF (`ggml/CMakeLists.txt:113`),
   landed with the §232 work. Note §232's closed-notes call the net effect "neutral on
   Q4_K/x86" but it stays ON as a low-risk default.
2. [ ] **Metal (+Vulkan) `GGML_OP_NORM_AFFINE` kernel** — our fused norm op
   (`ggml.h:500`, `ggml.c:3162`, used 7×/block in `core/fastconformer.h` + `core/sanm.h`)
   has ZERO Metal/Vulkan impl → falls to CPU with GPU↔CPU copies in the hot loop
   (~336/canary encode). Add the kernel + `supports_op` case
   (`ggml-metal-device.m:1392-1393`), OR gate `core_conformer`/`core_sanm` to plain
   `ggml_norm`+mul+add on Metal/Vulkan. Gate on M1 canary encode A/B.
3. [ ] **In-graph argmax + used-prefix-only KV snapshot** for cohere/AR decode.
   Ours (`src/cohere.cpp`) does full-vocab device→host + host max/softmax every
   token (`:2606,2614`). Beam search is the slow form: N sequential single-token
   forwards (`core/beam_decode.h:412-441`) each preceded by a full-KV deep-copy
   regardless of `n_past` (`core/attention.h:230-284`) — the #161 driver. Fix:
   in-graph `ggml_argmax` + 1-int readback (their `arch/cohere/model.cpp:1211-1213`);
   snapshot only the `n_past` prefix.
4. [ ] **Flash head-dim padding helper + real per-backend flash-capability gate.**
   Ours = ~40 scattered per-model `p.flash_attn` + reactive `getenv` guards, no
   head-dim padding, Turing sm_75 −9-10% fusion regression unhandled at model layer
   (fold in the opt-out from memory [[project_flash_attn_turing_regression]]). Ref
   their `pad_head_dim` (`arch/moonshine_streaming/decoder.cpp:37`).
5. [ ] **Consolidate per-backend pow2 FFT into one shared non-pow2 frontend.** Ours
   is per-backend pow2-only (`core/mel.h:33`), backends ship duplicate hand-rolled
   copies (`voxtral.cpp:446` "cargo-cult", `core/fft.h:6-8`). (Optional: switch
   miniaudio linear resample `crispasr_audio.cpp:406` → windowed-sinc if fidelity
   matters.)
6. [ ] (stretch) Utterance-batched encoder + batched multi-utterance decode for
   offline throughput — large, and the cached-graph minefield of #171/#184/§227.2.
   Do NOT start before action 0.
7. [ ] (cleanup) Bucket-table quant tensor classifier (their
   `tools/transcribe-quantize/policy.cpp:50-278`) to replace our per-arch if-ladder
   (`examples/crispasr-quantize/main.cpp:559-684`) — refactor, not a perf change.

Note: our single-stream conformer core is actually leaner (fused norm_affine,
fused GLU, load-time BN fold, zero-copy strided rel_shift `fastconformer.h:43-47`) —
port that rel_shift trick INTO them / keep it; §176 encoder-graph cache confirmed a
dud (keep `CRISPASR_PARAKEET_ENC_CACHE` OFF).

## §232 qwen3-tts code predictor perf — fuse 15 graph dispatches into 1 (#245, OPEN)

_Completed work archived to HISTORY.md (PLAN compaction 2026-07-17)._

**Still open:** optional CPU direct-path rescue (avoid lm_head slot blit)

## §235 next perf targets — triage after the §232 qwen3-tts sweep (2026-07-11)

_Completed work archived to HISTORY.md (PLAN compaction 2026-07-17)._

**Still open:** CV3 flow-steps default 10->6 flip pending human listen; transducer GPU decode pending parallel CUDA verdict

## §234 omnivoice — persistent step graphs + silence root cause

_Completed work archived to HISTORY.md (PLAN compaction 2026-07-17)._

**Still open:** voice-clone roundtrip validation, output-gain/clipping check, denoise token, rerun parity kernel

## §232 transcribe.cpp parity — close the RTF gap (mostly CLOSED; residual items OPEN)

**Status:** Engine is competitive (CA wins/ties most of 11 shared models vs
transcribe.cpp on Kaggle P100). Big losses (RNNT/TDT CPU-cblas decode, moonshine
CPU-only-from-CLI bug) are FIXED and default-flipped. GPU-forwarding audit done
across all CLI adapters. Remaining gaps are either architectural (moonshine-streaming)
or measurement/validation, all GPU-hardware-gated (Kaggle P100). Full history +
A/B tables → HISTORY.

### Open TODO

- **[HIGH, LARGE] Moonshine-streaming 18× loss — banded/blocked windowed attention.**
  Bottleneck is O(T²) sliding-window masked flash-attn (dense T_enc×T_enc F16 mask/layer
  ×6, wl=16/wr=4), NOT frame-by-frame encoding (encoder runs once, `moonshine_streaming.cpp:808`;
  masks can't be dropped — LEARNING 17 degenerate output). Need a banded flash kernel
  (no native banded flash in ggml). Own campaign. Keep `--backend moonshine` for offline.

- **[MED] Extend persistent ggml decode to beam/RNNT/maes paths.** Greedy TDT/RNNT decode
  is ported+flipped, but `parakeet_rnnt_decode`, `parakeet_tdt_beam_decode`, `*_maes_decode`,
  nemotron beam still call cblas `predictor_step`/`joint_step` beam_size× per step. Reuse
  `core_rnnt_ggml::Decoder` (one Decoder serves all hypotheses, state passed per call).
  Low risk, proven pattern. Needs a parakeet-rnnt model to validate the RNNT path.

- **[MED, GPU] dia TTS — widen default beyond Metal.** Default GPU on Metal only; CUDA/Vulkan
  opt-in (`DIA_TTS_GPU=1`). get_rows contiguity crash fixed (`65a5d30c`) but CUDA greedy
  tokens DIVERGE from CPU at step 0 (M1 Metal matched, argmax 568). TODO: ASR-roundtrip the
  CUDA-generated audio (the real HARD-RULE-#3 test) — if intelligible it's benign FP → widen
  default; if garbled, bisect the decode graph for another non-contiguous/precision-sensitive op.
  `load_weights_split` (enc+DAC→GPU, decoder→CPU) is the fallback.

- **[MED] Re-run P100 competitive scoreboard (v16).** After decode flips + moonshine GPU-forwarding
  fix (`d46839ca`), re-measure parakeet/nemotron/moonshine TOTAL RTF vs transcribe.cpp; update
  `docs/performance.md`. Run v16 kernel to confirm moonshine P100 CA ~0.012-0.015 (would make CA lead 6–5).
  Kernels: `tools/kaggle/parakeet-ggml-decode-ab/` (P100, exercises persistent path),
  `tools/kaggle/gpu-pin-ab`. Measurement, not a code change.

- **[LOW] In-graph argmax for transducer greedy path.** 2 int32 readbacks vs 8198-logit readback,
  greedy no-hotword no-sampling path only (hotword-bias/temperature still need full readback).
  Minor now that persistent-graph decode won. `parakeet.cpp` predictor/joint step.

- **[LOW] Parakeet same-version A/B (Fix 5).** CA uses parakeet-tdt-0.6b-v3 (25 EU langs) vs TC's
  v2 (EN-only) — the residual gap may be the model. Download v2 GGUF, benchmark both, note in docs.
  Model local: `parakeet-tdt-0.6b-v3-q4_k.gguf` (467 MB). Quick, VPS.

### Decision gates / notes
- No more optimisations on VPS — all remaining wins need GPU hardware (Kaggle) to validate+measure.
- Flip a GPU default only if transcript/WER parity holds AND GPU decode < cblas decode ON THE TARGET
  platform. Metal is excluded from the RNNT/TDT ggml-decode flip (Accelerate cblas beats it there;
  P100 win is a slow-OpenBLAS effect — LEARNING 34). Overrides: `PARAKEET/NEMOTRON_GGML_DECODE=1/0`,
  `RNNT_GGML_PERSTEP`, `CRISPASR_{PARAFORMER,M2M100,T5}_GPU=1/0`, `MOONSHINE_ALL_GPU=1`,
  `MOONSHINE_ENC_ATTN=manual`, `DIA_TTS_GPU=1/0`.
- Closed (no action): moonshine decode hybrid placement, moonshine encoder manual-attn (slower,
  opt-in), f5_tts GPU (7.8× base fix; cheap levers tapped out — needs Metal-capture profiling),
  paraformer/m2m100/t5 GPU (flipped GPU on CUDA/Vulkan), titanet/diarize (Accelerate-BLAS, GPU port
  poor EV), GGML_LLAMAFILE ON (neutral on Q4_K/x86, kept as default).

## Runtime speedup roadmap (2026-07-11 cross-repo sweep)

Status: full ASR+TTS+codec+pipeline re-verification 2026-07-11 (see PERFORMANCE.md "Runtime
Optimization Audit — Re-verification (2026-07-11)"). **Gate: every item needs a target GGUF (q8_0
preferred, to isolate from q4_k quant noise) + before/after parity + latency — do NOT land a perf
change on a compile-only check.** Note: most per-model TTS "flash not wired" claims are false — flash
reaches Orpheus/OuteTTS/Zonos/TADA/Chatterbox/CSM via shared `core_attn::kv_self_attn`
(`src/core/attention.h:665,903`). Real flash gaps are only manual-`soft_max` backends (dia, speecht5,
parler) and structurally-can't-flash relpos models (melotts, piper).

### Maps onto existing tracked items (don't duplicate)
- Decode-step graph cache for remaining LLM/AR backends → **§210 follow-up** (shape-stable bucketed
  decode / CUDA-graph capture). Templates: qwen3-tts Lk-bucket, granite §210 gallocr, mimo `step_t1_gf`.
- Batched-CFG (B=2) for remaining TTS → **§215**. Un-migrated diffusion/DiT targets: f5, dots,
  kugelaudio, pocket (+ dia/speecht5/parler once they get device KV). Respect Metal quant-B=2 gotcha
  (dequant batched-against weights q*→F16 once) and item-24 (don't CPU-batch a GPU pipeline).
- gallocr cross-call UAF audit → **#215e** / encoder-graph-cache removal (#235). Encoder-graph caching stays OFF (measured dud + GPU UAF).

### Shared cross-repo Tier-1 levers (coordinate with CrispEmbed PLAN.md)
- Decode-step graph cache — same design as CrispEmbed Tier-1 #1; CrispASR is further along (§210 CUDA-graph-capture template).
- ggml-metal ICB replay — Apple-side equivalent of §210's CUDA-graph capture; ggml-metal has no ICB path. Depends on a stable per-step graph. Shared ggml submodule — do once, both repos benefit.

### Genuinely-new gaps (not yet tracked)
| P | Area | Gap | File |
|---|---|---|---|
| P0 | melotts / piper | Scalar O(H·T²·D) relpos attention (can't flash — additive bias); HiFi-GAN 17.9s of 26.3s VPS total. Needs manual-attn ggml graph or BLAS | melotts.cpp, piper_tts.cpp |
| P0 | voxcpm2_tts | CPU-only (Metal SIGSEGV); manual per-step host KV re-upload | `voxcpm2_tts.cpp:106-111` |
| P0 | openvoice2 | 16-layer WaveNet + ref-encoder Conv2d/GRU scalar CPU | openvoice2.cpp |
| P1 | voxtral/voxtral4b enc, mimo LLM dec | Attention not on flash_attn_ext (manual soft_max) | voxtral4b.cpp, mimo_asr.cpp |
| P1 | firered/glm/funasr/qwen3/omniasr/mimo | Beam = replay; add KV snapshot pool (canary/moonshine/kyutai template) | — |
| P2 | scalar CPU hotpaths | RNN-T LSTM pred+joint; granite cpu_linear+depthwise; rvq encode; istft IRFFT; titanet mel; diarize `apply_xcorr` | core/rvq.cpp, core/istft.h, titanet.cpp:740 |
| P2 | parakeet/nemotron | Batched sgemm decode opt-in default-OFF — validate + flip on | `CRISPASR_TDT_BATCH` |
| P3 | threading | Hardcoded default-4 threads in ~90 sites; adopt whisper-core's `min(4, hw)` | crispasr_c_api.cpp |
| P3 | pyannote | Runs per-slice not once-over-audio (#107); RNNoise recreates state+resamplers/call | crispasr_diarize.cpp |

(Struck P0 firered_asr — self-attn KV already cached, remaining O(T²) is scalar attention *scoring*
that often loses on M1/CPU — and P2 align_wav2vec2_ctc — now on the §176e resident cache — dropped as
non-gaps; detail in HISTORY.)

---

## §247 roll #81 techniques out to the other runtimes (OPEN)

Techniques shipped for the FastConformer family (fastconformer.h consumers: parakeet,
canary, canary_ctc, canary_qwen, lfm2_audio, nemotron); roll each out where it applies.

- [ ] **(1) F16-weights-in-quantized-GGUF audit** (the 35%-of-encoder trap): converters
  storing matmul-consumed weights 3D/1D get them skipped by the quantizer's 2D rule → run
  on the ~6×-slower CPU F16 mul_mat path. Method: list tensors with `GGUFReader`, flag F16
  tensors feeding `mul_mat`; or run per-node profiler and look for `MUL_MAT f16` high-%
  rows. Suspects: cohere-transcribe, firered-asr, granite/conformer_ibm, sanm/paraformer
  conv, moonshine, TTS vocoders (hifigan/seanet/dac k=1 — qwen3-tts FASTCONV fixed the cast
  side, not storage). Fix: quantizer carve-out (+Q8_0 floor + idempotency) + load-time
  repack via `core_conformer::repack_conv_pw_q8` + fleet requant kernel
  (`tools/kaggle/fc-pw-requant`).
- [x] **(2) Generalize the per-node profiler**: move `cc_prof_cb` (sched eval callback,
  aggregates by op+src-type+shape, `CRISPASR_FC_PROFILE=1`) to `src/core/sched_prof.h` so
  the audit targets share it. `CRISPASR_SCHED_PROFILE=1` now covers Canary CTC,
  Cohere, FireRed-ASR, Granite Speech, Moonshine, Moonshine Streaming and Paraformer;
  the callback reports relative shares because forcing one split per node adds dispatch overhead.
- [ ] **(3) Fused QKV** (`core_conformer::fuse_qkv` is tensor-generic): bit-identical ~free
  win wherever Q/K/V share an input. Already deployed on ~10 backends
  (parakeet/canary/canary_qwen/canary_ctc/lfm2_audio via `core_conformer::fuse_qkv`
  under `CRISPASR_FC_FUSED_QKV`; voxtral/voxtral4b/qwen3_asr/higgs_stt/qwen3_tts have
  their own env-gated fused impls; nemotron deliberately opts out). **Still open** for
  the remaining named targets: whisper encoders, cohere, firered_asr, granite_speech,
  sanm/paraformer, AED decoder self-attn.
- [ ] **(4) Strided flash inputs**: grep `ggml_cont` feeding `flash_attn_ext` across src/ —
  the kernel reads strided views (llama.cpp does); each cont is a full tensor copy/layer/pass.
- [ ] **(5) Manual-attn-on-CUDA gate**: any backend whose flash mask has ne[2]>1 (per-head
  bias) silently falls back to CPU on CUDA — `GGML_SCHED_DEBUG=2` on a CUDA box shows flash
  nodes on CPU splits. Check cohere-transcribe (Shaw rel-pos) first. Reuse `fc_gpu_manual_attn`
  + BlockParams.manual_attn.
- [ ] **(6) -inf pad masking + bucketed persistent graphs** (`CRISPASR_FC_BUCKET`): reusable
  base for batched inference + CUDA-graph capture; correct padding for future streaming/batching
  (finite mask constants get overrun by pad garbage — LEARNINGS 2026-07-12).
- [ ] **(7) Q8_0 floor for decode-critical tensors in sub-8-bit quants** (quantizer): conv pw
  done; consider same floor for other high-sensitivity small tensors flagged by future A/Bs.

Suggested order: (2) profiler → (1) audit sweep with it (one Kaggle CPU kernel over the
registry, collect `MUL_MAT f16` % per backend) → fix top offenders → (4)/(3) mechanical
wins alongside → (5) after the §246 CUDA per-stage data.

---

## §248 — audio.cpp competitive benchmark + gap analysis (IN PROGRESS)

**Context:** #274 flagged [0xShug0/audio.cpp](https://github.com/0xShug0/audio.cpp) as
a comparable C++ inference engine. Audit performed 2026-07-19. They share ~6 ASR and ~12
TTS models with us; the rest is non-overlapping. CrispASR is broader on ASR (43 vs ~6)
and TTS (48 vs ~12) but audio.cpp covers categories we don't touch yet.

### (A) Head-to-head benchmark (shared models)

Must-compare on Kaggle (GPU + CPU kernels):

| Model | audio.cpp claim | Our backend | Metric |
|-------|----------------|-------------|--------|
| Voxtral Realtime | RTF 0.089 (11.2×), 15.7× Q8_0 | `voxtral4b` | RTF, WER on librispeech-test-clean |
| VibeVoice TTS 1.5B | 5.15× realtime (93.9 min podcast in 18.2 min) | `vibevoice-tts` / `vibevoice-1.5b` | RTF, MOS-proxy (ASR roundtrip) |
| Qwen3-ASR | (no claim) | `qwen3` | WER, RTF |
| Higgs Audio STT | (no claim) | `higgs-stt` | WER, RTF |
| Nemotron 3.5 | (no claim) | `nemotron` | WER, RTF |
| Chatterbox TTS | (no claim) | `chatterbox` | RTF, ASR roundtrip |
| IndexTTS2 | (no claim) | `indextts` | RTF, ASR roundtrip |
| OmniVoice | (no claim) | `omnivoice` | RTF, ASR roundtrip |

**Approach:** build audio.cpp on Kaggle (their CMake), run their CLI on the same WAV
files we use, collect wall-clock + output text, compare WER/RTF side-by-side.

### (B) Models they have that we lack — new backends to port

All licenses verified 2026-07-19. Only open-licensed models listed (Vevo2 is
CC-BY-NC-4.0 = non-commercial, skip; Stable Audio is Stability Community License =
commercial-restricted, skip).

**Source separation (new category):**
- [x] **HTDemucs** — hybrid transformer demucs, music/voice separation. **MIT** (Meta).
  **DONE (2026-07-19, VPS+Kaggle).** ~2720 lines C++. **Full parity** with Python.
  12-point checklist verified. All ops implemented:
  STFT→CaC→norm→encoder(Conv2d+DConv+GLU+freq_emb)→channel_up→
  2D/1D sin pos emb→LayerNorm→CrossTransformer(5 layers, self+cross attn)→
  channel_down→decoder(skip+GLU+ConvTranspose2d+time_decoder)→
  CaC unmask→iSTFT+time_denorm→per-source stereo PCM.
  All wired: CLI adapter + factory + GGUF auto-detect + model registry
  (Q4_K default) + `--separate` dispatch + C API session + Python binding +
  Go LDFLAGS. GGUFs on HF (`cstr/htdemucs-GGUF`): F16 (81 MB), Q8_0 (53 MB),
  Q4_K (38 MB). Kaggle: 21 reference stages validated. VPS (8 GB) can't run
  inference (swap pressure); validated on Kaggle (16 GB).
- [x] **Mel-Band RoFormer** — frequency-band source separation. **MIT**.
  The #274 reporter specifically mentioned this. Do both — they're complementary
  (HTDemucs = 4-stem, RoFormer = vocal/instrumental).
  **TAKEN** (M1/Metal session, 2026-07-19). Picked for this box because it is
  **non-autoregressive** — no sampling, so the diff harness gives deterministic
  per-stage cos verdicts with no torch-vs-mt19937 RNG mismatch (unlike an AR TTS
  port), and the model is small enough to run the Python reference and the C++
  port in 16 GB. No NeMo dependency (NeMo import is broken on this Mac).
  Regime: read the Python blueprint line-by-line → `tools/reference_backends/
  mel_band_roformer.py` dumper → converter → C++ backend → per-stage diff →
  **acceptance = decoded-output roundtrip** (separated stems judged by SDR/ASR
  on the vocal stem, not by cos alone — HARD RULE #3).
  > **Coordination with the in-flight HTDemucs work (same category):** HTDemucs
  > is unchecked above but IS being worked (converter `a6a447587` + 21-stage
  > reference dumper `60ada0a06`). We share the new `--task separate` surface
  > (CLI flag, stem output/WAV writing, backend-capability bit). **Whoever lands
  > that scaffolding first owns it; the second builds on it rather than adding a
  > parallel one.** Per-backend files (`src/mel_band_roformer.*`, converter,
  > reference dumper, registry/CMake entries) are additive and conflict-free.

**Voice conversion (new category):**
- [ ] **Seed-VC** — zero-shot voice conversion. **GPL-3.0** (Plachta/Seed-VC on HF).
  Encoder + flow-matching + vocoder. Medium effort. GPL is fine for our Apache-2.0
  project as long as the model weights are loaded at runtime (not statically linked).
  **Scoped 2026-07-19:** Tiny XLSR variant = 142 MB (25M params), fits 8GB VPS (~3 GB
  RAM with PyTorch). Architecture: XLSR→CAMPPlus speaker emb→length regulator→DiT
  UViT flow-matching (30 ODE steps)→HiFi-GAN vocoder. Risk: flow-matching requires
  noise pinning for deterministic diffs; FunASR+ModelScope deps are heavy. Repo archived.

**TTS:**
- [ ] **Supertonic 3** — claims 200×+ realtime on CUDA. **OpenRAIL-M** (Supertone/
  supertonic-3 on HF). Permissive (attribution + responsible use).
  **Scoped 2026-07-19:** ~400 MB total (4 ONNX components: text_encoder 36 MB,
  duration_predictor 4 MB, vector_estimator 257 MB, vocoder 101 MB). Non-autoregressive
  flow-matching (5-12 steps). 44.1 kHz output, 31 languages, 10 voices. Runs on
  Raspberry Pi (~800 MB RAM). ONNX-only distribution = custom ref approach needed
  (dump ONNX intermediate tensors, no PyTorch hooks). Deterministic diffs. **VPS-ready.**
- [x] **MioTTS** — voice cloning TTS. **Apache-2.0** (Aratako/MioTTS-0.6B, Qwen3-based).
  **CODEC PARITY ACHIEVED** (2026-07-19). Full pipeline: Qwen3 LLM (28L, 1024d, GQA
  16/8, head_dim=128, vocab=164480) + MioCodec-25Hz-24kHz (FSQ → wave_prenet → conv →
  ResNet → AdaLN-Zero decoder → ResNet → iSTFT → 24kHz waveform).
  **Parity results (diff harness):**
  - FSQ dequant: cos=1.0, wave_prenet: cos=1.0, audio: cos=0.999 (F32)
  - Q8_0 (723 MB): audio cos=0.954 — acceptable for TTS
  - Q4_K (397 MB): audio cos=0.069 — codec needs F16 for quality
  - LLM forward: argmax matches Python (tokens correct), tail-cos lower
  **Remaining (non-blocking for audio generation):**
  - Tokenizer integration (Qwen3 BPE + ChatML) for `miotts_synthesize()`
  - CLI adapter (`crispasr_backend_miotts.cpp`) + registry + 12-point checklist
  - Mixed quantization (LLM=Q4_K + codec=F16) for optimal quality/size
  - HF upload: F16 + Q8_0 GGUFs with README
  - Voice cloning: global encoder (reference audio → 128-d embedding)

**Audio codec:**
- [x] **MioCodec v2** — standalone audio codec. **MIT** (confirmed on HF + GitHub).
  **TAKEN** (VPS session, 2026-07-19). 133M params, ~530 MB F32. Architecture:
  WavLM-base+ encoder → FSQ quantizer (1 codebook, 12800 vocab, 25 Hz) → iSTFT
  decoder. Encode+decode is the full pipeline (no external vocoder for v2). Also does
  standalone voice conversion (swap global embedding). ~1.2 GB RAM — fits easily.
  Unblocks MioTTS port (shared codec). Deterministic diffs (feed-forward, no sampling).
  Regime: read Python → converter → backend → per-stage diff → roundtrip (encode→decode
  parity with original audio).

**Music generation:**
- [ ] **ACE-Step 1.5** — music generation. **Apache-2.0**. 3.5B params, 8.3 GB.
  **Scoped 2026-07-19:** mT5 text encoder + 3.5B linear transformer flow-matching +
  DCAE latent codec + vocoder. Kaggle-only (weights alone exceed 8GB RAM). 4 min max.
- [ ] **HeartMuLa** — music generation. **Apache-2.0** (HeartMuLa/HeartMuLa-oss-3B).
  **Scoped 2026-07-19:** 4B params, ~16 GB F32. AR LM + HeartCodec (12.5 Hz). Kaggle-only.

**Diarization:**
- [ ] **Sortformer** — NVIDIA dedicated diarization. **NOT Apache-2.0** — streaming v2
  is CC-BY-4.0, v2.1 is NVIDIA Open Model License. ~470 MB, 4-speaker hard cap.
  **Scoped 2026-07-19:** End-to-end (no VAD/clustering pipeline). Fast-Conformer 18L +
  Transformer 18L → per-frame 4-speaker labels. Community ONNX exports exist. ~750 MB
  RAM via ONNX, ~125 MB Q4_K GGUF. VPS-feasible but: (a) license isn't Apache-2.0,
  (b) 4-speaker cap vs pyannote/MOSS unlimited, (c) need quality eval first.

**Skipped (license incompatible):**
- ~~Vevo2~~ — CC-BY-NC-4.0 (non-commercial only)
- ~~Stable Audio 3~~ — Stability Community License (commercial requires separate agreement)

### (C) Priority

1. **Benchmark first** — head-to-head on shared models (Kaggle GPU kernel). If we're
   slower on Voxtral/VibeVoice, fix perf before adding scope.
2. **Source separation** — HTDemucs + Mel-Band RoFormer (both MIT, #274's ask).
3. **TTS** — Supertonic 3 (speed benchmark), MioTTS (voice cloning, Apache-2.0).
4. **Voice conversion** — Seed-VC (GPL-3.0, runtime-loaded weights = OK).
5. **Music generation** — ACE-Step + HeartMuLa (both Apache-2.0, new category).
6. **Diarization** — Sortformer (evaluate vs moss-diarize first).

---

## §249 WebRTC VAD — remaining follow-ups (backend DONE 2026-07-19, see HISTORY)

- [ ] `tools/sync_go_cgo_ldflags.py` — add `-lwebrtc-vad` (CI will flag)
- [ ] Dedicated live test comparing segment boundaries vs Silero/FireRed

## §250 — Music transcription backends (piano_transcription + Basic Pitch)

**Context:** CometBeat TRANSCRIPTION_SOTA_HANDOFF.md lists these as CrispASR
ggml port targets. New category: audio → note events (MIDI).

**Models:**
- [x] **piano_transcription_inference** (ByteDance/Kong, Apache-2.0). CRNN: 4×
  (4-layer Conv2d + 2-layer BiGRU) sub-networks (frame/onset/offset/velocity),
  88-key output at 100fps. ~172 MB checkpoint, ~86 MB F16 GGUF. **TAKEN** (VPS
  session, 2026-07-19).
- [x] **Basic Pitch** (Spotify, Apache-2.0). Polyphonic, instrument-agnostic
  audio → note events. **DONE (2026-08-30, VPS).** The network is tiny (~40k
  conv weights, 112 KB F16 GGUF) — the model is really its front end.
  Weights come from the `nmp.onnx` that ships inside the package
  (`basic_pitch/saved_models/icassp_2022/`, sha256 `2c3c1d14…59a0ec`): the ONNX
  has the BatchNorms already folded AND carries the nnAudio CQT kernels as
  initializers, so `models/convert-basic-pitch-to-gguf.py` copies them
  bit-for-bit instead of reimplementing `scipy.signal.firwin2`.
  **`src/core/cqt.h` could NOT be reused** — it is the direct-kernel librosa CQT
  (zero pad, `1+n/hop` frames); Basic Pitch trained on nnAudio **CQT2010v2**
  (one top-octave kernel bank, 9 octaves of recursive x2 decimation, reflect
  padding, `sqrt(lengths)` rescale). New `src/core/cqt2010v2.h`; `cqt.h` is
  untouched so BTC parity is unaffected.
  Parity vs `tools/reference_backends/basic_pitch.py` (onnxruntime), synthetic
  polyphonic clip: **every stage cos = 1.000000** — audio window, CQT magnitude,
  NormalizedLog, harmonic stack, all three heads, and the stitched full-file
  posteriorgrams. End-to-end note events **27/27 exact** (start, end, MIDI,
  velocity) at F32; F16 shifts 2 of 27 note ENDS by one frame at a
  threshold boundary. jfk.wav (16 kHz → resampled, so the resampler differs from
  librosa's): all stages ≥ 0.9991, note events 11/11 exact at both F16 and F32.
  Wired: converter + `src/basic_pitch.{h,cpp}` + CLI adapter + `--piano`
  dispatch (routes on GGUF arch) + arch map + registry row + `crispasr-diff`
  branch. **GGUF not yet uploaded** — registry row points at
  `cstr/basic-pitch-GGUF`, build locally until then.
  Not ported: pitch bends (`get_pitch_bends`) and MIDI file writing — the CLI
  emits note events, same shape as `piano-transcription`.
- [ ] **MT3** (Google, Apache-2.0). Seq2seq multi-instrument. Feasibility check
  DONE 2026-08-30 → **GO**, `docs/music-transcription/mt3-feasibility.md`.
  Corrections to the old line: the checkpoint is **171.6 MB** (60 M params, ~120
  MB F16 — verified against the live GCS listing), not "1 GB+", and the T5X/zarr
  checkpoint decodes with stdlib+numpy (no JAX/t5x/TensorStore). ~80% of
  `src/t5_translate.cpp` reuses directly; the gotcha is MT3 uses sinusoidal
  ABSOLUTE positions (`FixedEmbed`, network.py:180/225 — zero
  relative_attention_bias params in the checkpoint), so the T5 runtime needs a
  second positional branch. Gate on note-level F1 vs `openmirlab/mt3-infer`
  (numpy ref dumper required — mt3-infer excludes Magenta MT3 deps), NOT cosine:
  the risk is the tie-section cross-segment note stitching. ~8 working days.

**New CLI surface:** `--task transcribe-music` / `--backend piano-transcription`
→ MIDI output file.

---

## §251 — Music-analysis infrastructure + the remaining CometBeat roster

**Context:** §250 covers note-event models. This covers the shared DSP those
models need, plus the roster items still absent. Status reply written back to
CometBeat: `mus-textbook/docs/TRANSCRIPTION_CRISPASR_STATUS.md`.

**Already shipped, for the record** (so nobody re-ports them): CREPE F0
(`src/crepe.cpp`, `--pitch`, Dart + WASM + pub.dev `crispasr 0.8.16`), HTDemucs
and Mel-Band RoFormer separation, piano_transcription (§250).

### Phase 0 — shared DSP (blocks the rest)

- [x] **`core/cqt.h`** — constant-Q transform. **DONE** (`src/core/cqt.h`,
  `tests/test-core-cqt.cpp` 726 assertions, `tools/cqt_librosa_parity.py`).
  Direct time-domain Brown kernels, not librosa's recursive downsampling.
  Measured vs librosa 0.11.0 at BTC params: **per-frame shape correlation median
  0.9999** (mean 0.9721, min 0.1136), **peak-bin exact match 97.6%**. The mean is
  dragged down by exactly 3 transition frames + the tail frame, where librosa's
  per-octave group delay differs from our uniform centring — steady state agrees
  to 0.9999, and for 10 s BTC segments that latency is immaterial. Frame count
  differs by one at the tail; align on the leading edge. **CometBeat can reuse
  `tools/cqt_librosa_parity.py` as their Dart CQT oracle** — it is the exact
  parity harness their handover asks them to budget for.
  ⚠️ Cost is O(n_bins · N_k)/frame; N_k ≈ 24k samples at the bottom bin. Fine
  offline at hop 2048, NOT a real-time path — add a sparse spectral-kernel
  variant behind a flag if one is ever needed.
- [x] **`core/gru.h`** — uni/bidirectional GRU. **DONE**, validated against a
  real `torch.nn.GRU` fixture: **max_abs < 1e-5**
  (`tools/gen_gru_reference.py` → `CRISPASR_GRU_REF` → `tests/test-core-gru.cpp`;
  the parity case skips without the fixture so CI needs no torch/network).
  Two traps documented inline because both yield plausible-but-wrong output:
  (1) the reset gate multiplies the RECURRENT TERM `r*(W_hn h + b_hn)`, not
  `W_hn(r*h)` as the textbook GRU does; (2) `b_ih`/`b_hh` CANNOT be pre-summed
  the way core/lstm.h folds them, since `b_hn` sits inside the `r` product.
  (piano_transcription landed with its own BiGRU; fold it in when convenient,
  do NOT refactor it blind.)
- [ ] **`core/stft.h`** — forward STFT. `core/istft.h` covers only the inverse.
  htdemucs and mel-band-roformer each carry a private copy already, so a third
  consumer makes it the third duplicate. ⚠️ **This refactors two SHIPPED
  backends** — requires byte-identical stem output on both before it lands, or
  it ships gated. Lower priority than cqt/gru precisely because of that risk.

### Phase 1 — models

- [ ] **RMVPE** (MIT, lj1995/VoiceConversionWebUI). Robust vocal F0; **same
  360-bin salience as CREPE**, so it reuses `crepe_decode_local_average`
  unchanged and drops into the same `--pitch` surface. CometBeat's vet
  correction is important and correct: the *shipped ONNX is a pure conv U-Net
  with NO recurrent layer* (the paper's GRU is absent), so it ports cleanly.
  Input is a 128-bin mel — nearly free for us via `core/mel.h`. 361 MB f32 →
  ~180 MB q8_0 / ~90 MB q4_k. Wire as a second capacity behind `--pitch`.
- [x] **BTC chord recognition** — SHIPPED 2026-07-20. `--chords` + session C ABI
  + wasm; `crispasr-diff btc` 13/13 at cos 1.000000 (f16 and f32); 98.6-99.2 %
  `mir_eval` agreement with the torch reference on real music. Weights are
  CC-BY-NC-SA and ship behind `--accept-license`; published as
  `cstr/btc-chords-GGUF`. `core/cqt.h` landed and was NOT the last blocker --
  the real bugs were a missing `scale=True` and a chunked-vs-continuous
  front-end mismatch. See `docs/music-transcription/PLAN.md`.
- [x] **Basic Pitch** — see §250. DONE 2026-08-30. Note: it needed a SECOND CQT
  (`src/core/cqt2010v2.h`, nnAudio CQT2010v2) — `core/cqt.h` stays the librosa
  direct-kernel one for BTC/TabCNN.
- [ ] **MT3** — feasibility memo on T5X/JAX checkpoint conversion BEFORE any C++.

### Phase 2 — surfaces

- [x] **Bind `crispasr_session_separate*` in the Dart package** — SHIPPED
  (`flutter/crispasr/lib/src/crispasr.dart:4031`, `separate()` returning
  `List<Stem>`). NOTE: until 4eccc60cb this reached htdemucs ONLY -- the session
  ABI had no mel-band-roformer arm, so the MIT vocals/instrumental model (the
  CLI default) was unreachable from Dart. Now both.
- [ ] **CREPE perf**: per-node profile (never done — the RTF numbers came from
  FLOP arithmetic, not measurement) and sweep `kBatch` (64 was a guess).
- [ ] **CUDA validation** for the CREPE path and both ggml conv fixes
  (`ggml_conv_1d` batch reshape, `ggml_conv_1d_dw` batch support). Everything so
  far is M1/Metal only, and LEARNING 35 says Metal-correct is not CUDA-correct.
- [ ] **File the upstream ggml PR** drafted at
  `tools/upstream-prs/24-conv-1d-batch-reshape.md`.

## §252 — CosyVoice3 native-Vulkan: fp32-accumulation lever (LOW priority, likely dead end)

Background (see LEARNINGS "Backend miscomputes my pipeline ≠ op X is broken" +
`project_304_cosyvoice3_vulkan_native_lm_vs_flow`): CV3 is CPU-routed under Vulkan
because native synthesis is noise. This was FULLY root-caused, on a real Tesla
P100: it is **not** a broken ggml op (test-backend-ops passes every flow/HiFT op
on Vulkan — IM2COL, NORM, MUL_MAT, ROPE, …) and **not** my gallocr dispatch
(`FORCE_GALLOCR` on CPU is bit-identical to the scheduler). It is **aggregate
precision sensitivity**: Vulkan's in-tolerance per-op accumulation deltas (fp16 vs
CPU fp32) compound across the 22-layer DiT × 6 CFM Euler steps and are amplified
by the log-mel HiFT vocoder → flow mel cosine(cpu,vk)=0.961 → garbage. The LM
(shallow-per-token) is bit-correct on Vulkan (512/512 greedy tokens).

- [ ] **Lever (low priority, low odds):** force **fp32 accumulation** in the
  ggml-vulkan matmul/conv path for the flow + HiFT (e.g. `GGML_PREC_F32` /
  coopmat-f32-accum, or a build/env knob) and re-measure the flow mel cosine and
  the audio ASR round-trip on a real NVIDIA Vulkan device (Kaggle P100). If cosine
  climbs to ≳0.999 and audio is intelligible, native Vulkan (or at least a
  flow+HiFT-on-Vulkan hybrid) becomes viable. **Why it's likely a dead end:** every
  op already passes test-backend-ops *within tolerance*, so fp32-accum may not
  shrink the per-op delta enough to stop the CFM+vocoder amplification; and it is
  deep upstream ggml-vulkan shader work. Do NOT invest unless someone independently
  wants native-Vulkan CV3 speed badly enough to accept the shader effort.
- Groundwork already committed (branch `fix/304-cosyvoice3-se`, gated, NOT merged):
  LM single-backend gallocr (proves the LM runs native-Vulkan), hybrid
  HiFT-on-CPU infra, and diagnostics `CRISPASR_COSYVOICE3_{GREEDY,DUMP_MEL,
  DUMP_HIFT,FORCE_GALLOCR,HIFT_ON_GPU}`. Repro kernels:
  `tools/kaggle/cv3-vulkan-{isolate,convtest}`. A real self-recursion crash in the
  `cv3_sched_*` shims (CPU/Metal SIGSEGV) was fixed on that branch; the shipped
  release was never affected (shims are branch-only).
- Default stays the shipped all-CPU route under Vulkan — correct, and the right
  answer unless the lever above ever pays off.

## Nemotron late-layer isolation launch checkpoint 2026-10-09

Experimental worker `3bb2b58b0f20c65b8cd0c876d363754a5948ced2` is pushed on
`fix/nemotron-rnnt-quant`. GH CPU run
[37981088899](https://github.com/CrispStrobe/CrispASR/actions/runs/37981088899)
is queued at that exact source. All ten baselines plus five new predeclared
arms run with unchanged F16-first gates and capture controls. Actual prior
physical tensor metadata verifies remaining Q4 bytes: out/position isolation
14,155,776 each; late 20/16/12 guards 23,592,960 / 18,874,368 / 14,155,776.
The previous attention-isolation raw arrays, complete tensor ledger, original
gold and exact worker are public and remote-SHA verified in
`docs/nemotron-encoder-attention-2026-10-09.json`; the tracked receipt keeps
compact type counts and all 24 layer/norm metrics, while the full physical
ledger stays in the public proof. No new model/default promotion or speed claim.

This checkpoint passes all four full secret-scan tests with no skips
(93.963 seconds) and all 227 documentation anchors across 82 files.

Windows CUDA package pairing is complete: both CUDA 12.8 and 12.6 runtime ZIPs
and shared-library archives match all three actual DLL hashes in their CLI
manifests. A real 12.8 CLI/12.6 library-runtime mismatch is rejected for the
expected reason. GH CPU 37980464852 is SUCCESS; no Windows GPU inference claim.
Receipt: `docs/windows-cuda-package-pair-2026-10-09.json`.

## CLAIMED 2026-10-09 — Decode original controls for stage-passing Nemotron hybrid

Worktree: `/mnt/storage/crispasr/salvage-20261008/nemotron-quant`, experimental
branch `fix/nemotron-rnnt-quant`; native diagnostic branch is not an integration PR.
GH CPU 37981088899 is SUCCESS at `3bb2b58b0`: all ten prior baseline arrays and
original PCM are byte-exact, F16 passes all three unchanged stage and 24 layer
gates. Four new late-layer/position-source arms fail the unchanged encoder gate.
FFN+QKV+out-source / position-Q4_K passes all three (encoder cosine 0.999842)
and all 24 original frame-layer gates, final relative L2 0.00309294. Physical
size 1,179,413,408 bytes, actual Q4 payload 14,155,776 bytes. Most weights are
source precision; this is an 8.3% size reduction from F16, not a small Q4 or
speed claim. No production model/default promotion. Receipt:
`docs/nemotron-encoder-late-layer-2026-10-09.json`.
Next reproduce its exact SHA/type/byte ledger on GH CPU, retain F16-first
strict gates, and run all 26 pinned original transcripts plus 22 VoxCPM2 words
in fresh and reverse-reused sessions. Include a plain-Q4 negative stage control.
Only retain the candidate in a workflow artifact if every original decoded and
state control passes; then it can be copied to a public experimental immutable
model path for actual GPU parity/streaming controls. No original model compute,
conversion, quantization or native compilation on Kaggle. No release quant pin
before full acceptance. All scratch/artifacts stay on storage/GH.

Original decoded-control worker `b93d60bfe93da80ebac1841df0835277b12ec10c`
is pushed; GH CPU [37987517614](https://github.com/CrispStrobe/CrispASR/actions/runs/37987517614)
is queued at that exact source. Reproduction must match candidate SHA
`84cefe0658e8d87fc55a704406f92c48224118f916f3d5b47a94817b7e315f28`,
size 1,179,413,408 and Q4 payload 14,155,776 bytes, with source-guard byte
identity and plain-Q4 tensor equality. F16 and candidate must pass unchanged
stages before decoded testing; the known plain-Q4 negative must reject.
Every positive must match all 26 original transcripts, all 22 VoxCPM2 controls,
and reverse-reused fresh outputs before retaining the candidate in the artifact.
No model/default promotion; public full stage/layer proof is pinned in the receipt.


Plugin-discovery correction checkpoint: `37a90a37c9871ff4d4cf915993acec2b734293cd`
explicitly preserves the UTF-8 byte representation across C++17/C++20;
standalone warning-as-error compilation and non-ASCII path assertions pass
under both standards. The actual staged C ABI CI regression now uses a
non-ASCII package path. Failed hosted job 113979995896 logs are retained in
storage. Old CUDA build 37977686803 was canceled after its checkout was observed;
replacement no-publication [37988408052](https://github.com/CrispStrobe/CrispASR/actions/runs/37988408052)
is queued at the corrected source. Freeze the branch until its checkout is
observed; no candidate package/GPU acceptance yet.
The original decoded-control worker's full four-test secret scan passes with
no skips (137.633 seconds); decoded run 37987517614 remains queued.

Corrected plugin source passes all four no-secrets tests with the account
environment set, no skips (122.715 seconds). Documentation checkpoint resolves
all 227 internal anchors. Candidate CI 37988394079 and CUDA package
37988408052 are queued; decoded Q4 run 37987517614 is queued.


## CLAIMED 2026-10-10 — Corrected package and stage-passing hybrid CUDA controls

Continue in `/mnt/storage/crispasr/salvage-20261008/release-cuda-hardware`.
Corrected source `37a90a37c9871ff4d4cf915993acec2b734293cd` passes the
actual staged C ABI regression through CPU-only and GPU-requested routes from
a non-ASCII package path, with no loader overrides. GH CUDA build 37988408052
is SUCCESS. Download artifact 11648407040 to storage, audit exact source and
archive/dependency hashes, mirror the exact tar at a public immutable HF path,
then launch version 4 against unchanged F16 stages, original transcripts and
native streaming controls on the single authorized Kaggle account.
Original decoded-control GH 37987517614 is SUCCESS. Verify full receipts and
retained candidate SHA before publishing it under a public experimental model
path. Extend hardware controls only after its CPU acceptance is independently
confirmed; retain F16-first, unchanged stage gates and all decoded/state checks.
No model/default promotion, original model compute, quantization or compilation
on Kaggle. Archive every terminal kernel output before repushing.
PR #528 Docker checks failed on Docker Hub HTTP 429 before compilation;
failed jobs rerun without changing source or acceptance gates. Other checks,
including the actual dynamic C ABI regression, pass.


CUDA package worker `759b421362f17dfa7d50da3958ee138211c31ad8` is pushed,
version `2026-10-10.release-package.4`. Syntax/diff checks and all four secret
checks pass with no skips (166.763 seconds). The exact corrected archive
passes dependency closure and actual local C ABI CPU-plugin initialization on
both CPU/GPU-requested routes after extraction completes. Public byte-identical
mirror DATASET `cstr/crispasr-regression-fixtures@67b9fb956ee5498be8f97567d27c2a607155bb4d`,
path `experimental/release-package-37988408052/crispasr-linux-x86_64-cuda.tar.gz`,
remote SHA `a38c111037c8a6ae7b46c0a9bd1553271890daf72a42d893b74df76ccf7de79d`
is verified. Single-account scan of 30 recent kernels found no active sessions.
Predeclared launch retains unchanged F16 stages/transcripts/state/streaming and
requires actual packaged CUDA plugin loading plus allocations on its own PID.
No new package GPU acceptance until terminal proof.

## OPEN 2026-10-10 — Contributor PR corrections after review

Inspect exact heads `08ff8c5c492031ea94a42964f8fa21cd11dd2691` and
`5fce2351533617e35bba13711ac3319dbe870e08`, including all comments/reviews,
full diffs and CI. #529 has 49 successful checks and one intentional skip;
new two-branch guidance lacks original model parity and changes existing
exported C APIs. #530 fork workflows required maintainer approval; approved
only unchanged checks after reading code. Review the sample-rate fixes,
reference-buffer lifetime and failure-state contracts before integration.
No new contributor PR merge or acceptance from generic build checks alone.


Nemotron original decoded control is independently verified: F16 and the exact
1,179,413,408-byte position-Q4 hybrid pass all unchanged stages, 26/26 original
word controls, 22/22 VoxCPM2 controls and all 26 fresh/reverse-reused outputs.
Known plain-Q4 negative rejects (rc 6). Candidate SHA matches the stage proof;
all protected tensors equal the F16/source ledger and all unprotected tensors
equal plain Q4. Full proof plus experimental candidate are public at DATASET
`cstr/crispasr-regression-fixtures@545fbcf4615470b62aa6bca840c7444949acfa00`,
under `experimental/nemotron-position-q4-20261010/`, with remote SHA verified.
Receipt: `docs/nemotron-q4-original-controls-2026-10-10.json`.
Claimed next worker: `tools/kaggle/nemotron-position-q4-validation` in the
storage hardware worktree. Reuse exact corrected CUDA package controls with
F16 first, then pinned candidate, save separate full stage arrays/decoded/state/
streaming results, and accept only if both complete. Gate quantized hardware
validation below compute capability 6.1 as inconclusive; do not fish for hardware.
No CPU build/quant/reference work on Kaggle and no production quant pin.


Corrected real CUDA package version 4 is terminal COMPLETE on dual Tesla T4
with CUDA runtime 12.8/driver 580.178.04. Exact archive/source/plugin hashes and
pulled worker bytes agree. Actual C ABI loads packaged CPU/CUDA plugins and
allocates 1,330 MiB on its own PID. All three unchanged F16 stage gates pass
(encoder .99999088, relative L2 .000280039); 26 original fresh/reused controls
and native streaming presets 0/2/3 pass. Public full arrays/output/source proof
DATASET `cstr/crispasr-regression-fixtures@ce47022770e44caa5da782490ed7b7b6466388ef`,
path `experimental/release-package-37988408052/cuda-package-hardware-proof.zip`,
SHA `def2692c3663fc6f11da2d0d8863c98c11d2361bd9c861f99232b4b9e7789504`.
All 50 applicable PR #528 checks now pass (one intentional skip); old Docker
HTTP 429 failures passed unchanged-source reruns. Accepted for integration,
not yet final release-tip matrix validation.
Candidate GPU worker `2edfd1c5a` is pushed on `probe/release-cuda-hardware`.
Each F16/candidate arm runs in a fresh process, so cached allocations cannot
satisfy another model's proof. Syntax/diff and all four secret checks pass,
no skips (53.975 seconds). Predeclared kernel
`${KAGGLE_ACCOUNT}/crispasr-nemotron-position-q4-validation` remains prepared
for launch; both arms must pass separately before a combined verdict.

Contributor reviews posted: [#529](https://github.com/CrispStrobe/CrispASR/pull/529#pullrequestreview-5477583473)
requests preserved exported C API and original guided model parity; [#530](https://github.com/CrispStrobe/CrispASR/pull/530#pullrequestreview-5477583717)
requests atomic Breeze reference setters, owned Moss/OpenVoice buffers,
propagated SNAC load errors and regenerated CLI capability data. Twelve pending
fork workflow runs were approved after the full patch review; CI is running.
Do not merge these contributor PRs based on generic build checks alone.


PR #528 merged as `cf9f61c29a6ec4f19291f12611ce6bee9ff0add3`; accepted
package hardware evidence is archived in HISTORY and its receipt. The fresh
process candidate kernel version 1 is launched; pulled source is byte-identical
to worker `2edfd1c5a`, entry hash recorded in its CPU proof receipt. All 30 recent
single-account kernels were terminal before launch. No candidate GPU acceptance
or production/default promotion yet.
Next claim: full no-publication release dry-run matrix on this integrated main
source, now using immutable checkouts from merged #527. Native plugin source
is identical to the accepted package; verify all final-tip artifacts/packages
and main CI before bumping/tagging. Contributor #529/#530 changes remain excluded.


Integrated full no-publication release matrix [38024994245](https://github.com/CrispStrobe/CrispASR/actions/runs/38024994245)
is queued at immutable event source `f51dc777018483a6719ff32159141c513e70e542`.
Both input tag and job filter are empty, so all packages are requested and
publication is disabled. No v0.8.43 bump/tag has been made.
PR #530 CI now reports clang-format 18 violations in its C ABI source; remaining
build/model checks are running or queued. Exact review findings remain recorded
in the posted request-changes reviews; no contributor patch was merged.

Integrated package/GPU documentation passes all four secret-scan tests with
the account environment set, no skips (112.083 seconds). Documentation
anchors pass (227). Subsequent checkpoint changes only status/wording; the
release matrix and candidate GPU run remain pending, with no publication.


## CLAIMED 2026-10-10 — Contributor TTS ownership and compatibility corrections

Owner: this maintainer session. Worktree
`/mnt/storage/crispasr/salvage-20261008/contributor-tts-fixes`, branch
`fix/contributor-tts-contracts`. Start from reviewed PR #530 head
`5fce2351533617e35bba13711ac3319dbe870e08`, preserve contributor commits,
and correct atomic reference state, Moss/OpenVoice ownership, SNAC error
propagation and generated capabilities/style. Preserve/compose #529's existing
exported API and reference+instruction paths in a separate integration step.
Require meaningful failure/replacement/lifetime controls before accepting fixes;
model changes still require original diff and decoded/clone controls on GH,
with heavy GPU work only on the authorized Kaggle account. No merge or release
claim from generic builds alone. The Nemotron fresh-process candidate GPU run
is terminal COMPLETE; verify full arrays, source/hash and decoded/state receipts
and publish its proof before recording acceptance.


Nemotron candidate GPU version 1 is terminal COMPLETE and independently
verified. Both F16/candidate arms pass full original stage arrays, all 26
normalized-word controls, exact fresh/reused state and native streaming
presets 0/2/3. Distinct PIDs allocate 1330/1228 MiB through the packaged CUDA
C ABI. Public proof DATASET @77235249ea4436b63dfd41b2048752c84aee789b,
`experimental/nemotron-position-q4-20261010/cuda-hardware-proof.zip`,
SHA `1bc7b09cc33932125151cc9fc2eec7c66851dc4b5339f71f023c8f9a5c9e7e55`,
6,279,694 bytes. Receipt: `docs/nemotron-position-q4-gpu-2026-10-10.json`.
This completes the launched candidate hardware controls, not production/default
promotion or speed proof. Contributor fixes remain active in the claimed
separate worktree; final integrated release matrix remains pending.


Contributor corrections are prepared in draft [PR #531](https://github.com/CrispStrobe/CrispASR/pull/531),
source `c222540e1cdef04d011807040a44030ff8ba9228`. Original #530 commits
are preserved and current main merged. Four actual-dispatch failure/ownership
contracts reject original #530 source and pass fixed code, including 100 Moss
replacements and 100 OpenVoice conversions, failed allocation/encode, Breeze
reference/provenance rollback and SNAC errors. The OpenVoice rate-getter path
also repairs #530's observed linux-unit failure. Formatting 18/syntax/diff
and all four tracked secret checks pass (50.743 seconds, no skips).
Hosted [GH 38026088284](https://github.com/CrispStrobe/CrispASR/actions/runs/38026088284)
builds actual CLI/shared library, generates capability/feature tables and
checks CLI/C ABI parity; tables must be copied from its terminal artifact
and rechecked before acceptance. Earlier queued dispatch 38026075171 was
cancelled before work to use Ubuntu 24's packaged clang-format 18.
Receipt: `docs/contributor-tts-contracts-2026-10-10.json`. No contributor
merge, original guided-stage or actual decoded/clone acceptance yet. #529
exported-API compatibility and reference+instruction composition still pending.
Integrated release matrix 38024994245 has 12 successful jobs and no failures
at this checkpoint; immutable source stays f51dc7770. No bump/tag yet.


## CLAIMED 2026-10-11 — Session UTF-8 and Canary alignment crash

Owner: this maintainer session. Separate worktrees under
`/mnt/storage/crispasr/salvage-20261008/issue533-utf8` and
`/mnt/storage/crispasr/salvage-20261008/issue532-canary`, branches
`fix/session-byte-bpe-utf8` and `fix/canary-ctc-allocation`. Whole new issues
#533/#532 and all comments read before starting. Reproduce split-byte BPE
through native session result getters/bindings; retain complete valid text and
existing timestamps/probabilities. Trace Canary CTC allocation on changed/reused
alignment shapes, keep original diff/decoded controls and use GH for model work.
No gate relaxation or speculative backend changes.
Contributor TTS hosted build 38026088284 is SUCCESS, as is immutable full
release matrix 38024994245 (36 successful jobs, no failures). Download terminal
artifacts, apply generated capability tables to #531 and validate its final
head before integration. Read all live open issue/PR discussions, including
#529's updated contributor head, before further fixes/answers.

### 2026-10-11 launch checkpoint — assembled UTF-8 and CTC allocation

- #533 fix is draft [PR #534](https://github.com/CrispStrobe/CrispASR/pull/534),
  source `0bbd31921`. Production finalization/getters compiled into a native
  shared fixture reproduce 24 strict-decoder errors before the fix and pass
  all four tests afterward. Normalize assembled strings before callbacks and
  publication; timing/confidence preserved. All four secret tests pass without
  skips (56.347 s). Full native CI is running. Reporter Metal/recording and the
  reason for malformed decoded bytes remain unmeasured; no decoding-parity claim.
- #531 hosted contract worker **38026088284 SUCCESS** at `c222540e1`, artifact
  **11661638474**. All four contracts, actual native output-rate dispatch, and
  all 136 compiled CLI/C ABI capability rows pass. Exact generated header and
  feature matrices applied at `e1e504536`; final-head CI is running. This CPU
  build does not compile optional ONNX adapters. Model plain/clone and #529
  original guided acceptance remain pending; PR stays draft.
- Immutable full release dry run **38024994245 SUCCESS**: all 36 jobs pass at
  event source `f51dc777018483a6719ff32159141c513e70e542`, with no publication.
  This does not validate later #531/#534 native changes. No bump/tag yet.
- #532 actual-model CPU allocation probe is prepared on
  `fix/canary-ctc-allocation`. It will run native CTC logits for 11/4/33/90/11 s,
  bucket OFF/128, with exact fresh/reused short-shape controls and full saved
  logits/hashes. It is a runtime replay diagnostic, not original-model parity
  or reproduction of the unavailable German recording/device. No speculative
  runtime fix has been applied before reproduction.
- Full bodies/comments of all 12 live issues and five prior live PRs have been
  re-read. #529 author head is now `3ebd2a561`; review of the new diff and guided
  proof remains in progress. Reporter-dependent hardware results and rejected
  OmniVoice/Echo/MiMo gates remain open, without threshold/default changes.

## CLAIMED 2026-10-11 — Breeze guided acceptance must reject missing evidence

Worktree `/mnt/storage/crispasr/salvage-20261008/pr529-guided-validation`,
branch `fix/breeze-guided-validation`, preserving contributor head `3ebd2a561`.
The updated #529 comparator discovers only available reference directories and
inherits a missing-stage path that reports but does not fail. A negative control
with only `guided/plain/codes.npy` in the reference and an empty C++ directory
returns success. Require all eight requested cases and required stage inventory,
then add deletion/missing-case controls before original guided/model validation.
CPU Canary allocation probe launched: GH **38119476469**, immutable source
`170f7e6289bdbb65e30a5004e7c76e396cc5c664`, currently queued; no result claim.

### 2026-10-11 terminal local checkpoint — strict guided gate

The #529 validation repair is `3de24603eb4b4326842013c5dd3c31116bf26045`,
fast-forwarded to the author's opted-in maintainer-editable PR branch after
all eight regression tests and all four secret tests pass (no skips, 47.924 s).
All contributor commits are retained. Six missing/empty/error controls fail on
the original comparator; the complete matrix and mismatch controls also pass
as tests of the repaired gate. `docs/breeze-guided-validation-2026-10-11.json`
records the exact scope. This establishes gate correctness, not actual original
model or decoded guidance acceptance. Current fork PR CI is pending.

#531 final-table head `e1e504536` passes the full four-test secret scan without
skips (145.072 s); final native CI pending. #534 has no failed checks so far,
but remains draft until final native CI completes. Canary actual-model probe
**38119476469** remains queued; no runtime fix or model result is claimed.
Issues #532/#533 and PRs #529/#531 have received precise progress/proof replies.
No pending/rejected model experiment has been promoted and no release tag cut.

## CLAIMED 2026-10-11 — Generated C ABI capability drift

Worktree `/mnt/storage/crispasr/salvage-20261008/caps-drift`, branch
`fix/backend-caps-drift`. Actual immutable release CLI lists 136 backends but its
C ABI table omits `hikari` and `moonshine-tiny-de-dattazigzag`. The generated
header repair must be independent of #531's pending synthesis/model scope.
Regenerate from the actual native-identical packaged CLI, preserve baseline
capabilities, and add non-skipping actual CLI/shared-library comparison plus
pinned-format generator drift checks in the CI job that really builds both.
Do not claim raw backend aliases are canonical capability-table keys.

Canary CPU probe **38119476469 SUCCESS** at `170f7e6289`: all ten actual-model
11/4/33/90/11-second bucket OFF/128 allocations succeed; all saved arrays are
finite, shapes/hash verified locally, and returning 11-second logits are exact.
No crash reproduced and no runtime fix; reporter assertion/device/audio still
needed. Full proof publicly pinned at dataset commit `2cfe9f89b0d64a3450e03023b87f5178fbdc949a`.
Immutable release CUDA CLI/quantizer startup/dependency closure and real WASM
exports/getters/piano/readme checks pass. CUDA plugin bytes differ from the
previous T4-validated archive, so this archive has no new physical GPU claim.

### 2026-10-11 independent capability fix checkpoint

Regenerated the baseline table from the immutable actual CLI: exactly two new
canonical rows, Hikari and German Dattazigzag Moonshine. The original full
packaged library fails both row lookups. Compiling the unchanged production
capability ABI functions against the regenerated table passes all 136 CLI rows
and size/error contracts; this local fixture is not a full project build.
The dedicated fix adds actual CLI/shared-library validation and pinned
clang-format-18 generator checking to linux-unit, without a missing-binary
skip. Full hosted native build remains the acceptance gate.

The actual Canary CPU allocation receipt and immutable CUDA/WASM package
inspection receipt are now archived in docs. No Canary runtime change, original
model parity, reporter-recording reproduction, new package GPU acceptance or
release publication is inferred. #529/#531/#534 native unit jobs pass; other
checks remain pending and the two synthesis PRs still need model acceptance.

## CLAIMED 2026-10-11 — Pinned original Breeze guidance reference on GH CPU

Worktree `/mnt/storage/crispasr/salvage-20261008/pr529-guided-validation`,
branch `fix/breeze-guided-validation`. Port delivery of the contributor's
reference dumper to heavy-cpu without changing inference or matrix tolerances.
Pin original source `58ec70ce5fa4cc361bdebf77ec40d1365da00ab2` and model
`3e28c5151381a722f1d8661b4118c298caa77aa4`; retain all eight guided cases,
seed 42, BF16/eager, 24 greedy frames and complete stage arrays. GH CPU only:
no Kaggle CPU fallback, no shared-environment package installation, no fixture
overwrite. Save immutable provenance/hashes and fail incomplete/nonfinite dumps.
Once complete, compare actual native #529 output with strict crispasr-diff;
decoded full-speech/ASR controls remain separate acceptance requirements.

Pinned reference worker `3ac8eedae3e1ce48fd6fa4002b7f3e91fc00c95d`
passes syntax/extraction checks, all eight strict comparator controls and all
four secret tests with no skips (106.897 s). GH CPU run
[38121758868](https://github.com/CrispStrobe/CrispASR/actions/runs/38121758868)
is dispatched at that exact head; currently queued. Source/model/codec delivery
uses GH scratch, never this VPS's RAM or a Kaggle CPU fallback. No new original
reference, native parity or speech acceptance is claimed until terminal output
is downloaded and independently verified. The worker is on the maintainer's
validation branch; the contributor PR still retains head `3de24603e`.

## CLAIMED 2026-10-11 — Native Breeze guided parity and frame-budget audit

Continue in `/mnt/storage/crispasr/salvage-20261008/pr529-guided-validation`.
Original GH reference run 38121758868 is SUCCESS; independently verified 145
complete finite arrays and all eight cases from artifact 11695878891. The scale-0
original cases emit 31 frames despite the declared 24-frame configuration;
other cases emit 24. Audit upstream stopping-length setup before negative-prompt
swap, preserve this unadjusted receipt, and require an explicitly matched frame
budget for native comparison. Do not silently truncate arrays or weaken exact
code/shape gates. Add GH CPU native F16/codec controls against pinned references,
retain strict cosine/magnitude/argmax/code and reused-context checks. #534/#535
have all applicable checks green and are merged; synthesis PRs remain pending.

## Completed investigation 2026-10-11 — #533 reporter clips and missing continuation byte

Worktree `/mnt/storage/crispasr/salvage-20261008/issue533-byte-preservation`,
branch `fix/qwen3-byte-preservation`. Read both new reporter comments in full:
byte 0xA0 vanishes deterministically from 造/加, with two linked 30-second WAVs.
Trace byte-level decoding versus Unicode whitespace cleanup, prove a controlled
before/after regression, and run actual clips through C ABI on GH CPU/Metal
where supported. Preserve the merged UTF-8 publication contract as defense;
fix the byte-loss cause and keep model/platform acceptance claims separate.

## CLAIMED 2026-10-11 — PR #536 reference transcript cache invalidation

Owner: this maintainer session. Worktree
`/mnt/storage/crispasr/salvage-20261008/pr536-voice-cache`, branch
`fix/bt2-voice-cache`. Whole PR #536 and its unique changes against #529 read.
Preserve contributor commits; reproduce a sidecar-only transcript edit retaining
the WAV mtime, then include the sidecar identity in the shared reference key.
Cover real production-header behavior, explicit transcript precedence, missing
files and retry contracts. Model/server acceptance still depends on original
Breeze parity and decoded roundtrips; no merge or release from cache tests alone.
PR #537 is reviewed; physical Metal Q8/F16 acceptance remains pending.

### 2026-10-11 current integration and validation checkpoint

PR #534 is merged as `a38d66428f4f89cfac8702ef7be65e52eea84104`: all
54 applicable checks pass, plus the hosted full native build/2,008 unit tests
and four strict ctypes controls. PR #535 is merged as
`3e4bea709f3d7be225e73d6faf9d2eec00e69633`: all 42 applicable checks pass,
plus actual full CLI/shared-library discovery across 136 canonical rows.
Release notes now include both; no bump or tag.

Original unadjusted Breeze reference GH 38121758868 is SUCCESS: 145 finite
arrays independently hash/shape/type verified; all eight cases retained.
Scale-zero generates 31 frames because HF computes max_length before the
negative-prompt swap. The seven-frame prompt-length difference explains the
excess. Public unchanged proof is pinned at fixture dataset revision
`bb8c9394f77aa8062eac258a1bf26e925ea2fb6c`,
`breeze-guided-20261011/unadjusted-oracle-proof.zip`, SHA256
`0bcd5ecd956315d3ccecd0f1ebdb49e08051af14c21265199542a534a8515a8a`.
Explicit active-prompt max_length preserves inference while imposing the same
24-frame budget; combined original/native GH 38146159521 at `ac7227125`
is running. No truncation, shape/code waiver or decoded-speech acceptance.

PR #538 (`0b9da8d127e793f7130d1ceeffc010da644ec089`) follows #534:
ASCII-only whitespace classification preserves byte-level UTF-8 continuation
bytes during loop cleanup. Injected classifier controls reproduce three
original failures, then all five controls pass. Actual reporter clips are
queued on GH macOS 38147247663 and Linux 38147249399. Actual macOS 26/M4
classifier behavior and model acceptance remain unproven.

PR #536 is based on #529 and remains pending original/model/server gates.
A production-header sidecar-only cache regression fails its contributor head;
adding sidecar identity when no explicit transcript overrides it passes all
seven local policy controls. Full hosted unit tests are still required.
PR #537 removes redundant contiguous copies of already-contiguous Q/K/V
weight-row views. Reporter Metal Q8 and F16 A/B claims are read; immutable
input/output and source provenance are needed for hardware acceptance.


### 2026-10-11 hardware-worker checkpoint

The first #533 macOS job 38147247663 failed before compilation/model loading:
Actions' shallow checkout lacked the pinned baseline commit. Receipt is archived
locally; no inference or Metal result is inferred. The queued identical Linux
job 38147249399 was cancelled before wasting work. Worker `7e3f768b8` fetches
full baseline SHA and verifies its header SHA256. An empty-repository fetch
control and cached reread pass, all five byte-preservation controls pass, and
all four secret checks pass without skips (46.909 s). Replacement macOS
38148680663 and Linux 38148682586 jobs are queued at that immutable head.

#537 worker `d473b82060d1a0fe6264d40deebdbaeeb8d73303` is pushed separately
from the contributor PR; GH macOS 38148987616 is queued. Source A/B allows
exactly the three CONT differences and explanatory comment; unchanged ggml,
pinned F16/Q8 model SHA and retained deterministic stereo input. It forces the
fused CPU graph so the changed nodes are actually covered. Local preflight
and exact/changed-sample/scale-error controls pass. Full four-test secret scan
passes without skips (43.363 s). Exact CPU/F16 Metal stems and original Q8
Metal abort/finite candidate numerical gates remain pending; no original
Python parity, reporter music-corpus, speed or hardware acceptance claim.

### 2026-10-11 terminal Breeze result and next boundary diagnostic

GH 38146159521 is FAILED at source `ac7227125`: all eight matched-budget
original cases have complete 24-frame captures. All prompt/negative-prompt IDs
match; frame-zero backbone cosine 0.999675–0.999994, norm ratio 0.9969–1.0014,
and argmax pass. All native reused-context/scale-zero/near-one controls pass.
Exact audio codes FAIL on 8/8 cases: first differences range from frame zero
codebook 5 (design-s1) to frame four codebook 5 (plain/design-s0). No acceptance,
tolerance relaxation or numerical-rounding explanation is inferred.

Independently verified all 145 original and 28 native arrays. Full failed proof
is public at fixture dataset revision `27896a1fab69d45c8b246eb659823f92a5e0e458`,
`breeze-guided-20261011/matched-budget-failed-parity-proof.zip`, SHA256
`13d5eed11955e07f2e5fc2f681df123c1b2b35d2f11998416571dbf962d54f33`,
5,896,927 bytes; anonymous download hash verified. Compact receipt is
`docs/breeze-guided-parity-2026-10-11.json`.

Diagnostic `9144c2796` adds all 27 existing TE layer outputs per text segment
to crispasr-diff and scores oracle-fed TE/projection/backbone/depth stages at
unchanged strict gates before re-scoring the full guided matrix. It reuses the
pinned original archive, avoiding another reference generation. Original full
generation/codec stages remain separate, never silently accepted as a subset.
All eight strict comparator controls and four secret checks pass without skips
(50.452 s); all 145 archived original arrays pass preflight. GH CPU 38150047103
is queued. Source remains on the maintainer validation branch, not #529's fork.

#536 macOS job 114495588783 failed: libc++ filesystem-clock count has no
unambiguous std::to_string overload. Follow-up `1bc158ed9` on the contributor
branch serializes seconds plus subsecond nanoseconds using standard chrono
representations, retaining timestamp precision. All seven local policy
controls and four secret checks pass without skips (50.088 s); hosted CI pending.
Reporter-clip macOS 38148680663 is now actually running; Linux and #537
Metal A/B remain queued. No new hardware acceptance or merge/tag.

### 2026-10-11 actual macOS UTF-8 root cause proved

Reporter-clip GH 38148680663 is SUCCESS at `7e3f768b8` on macOS 14.8.9
arm64, actual locale `en_US.UTF-8`. Real libc std::isspace classifies both
0xA0 and 0x85 as whitespace. The original production cleanup corrupts the
provided 造/加 contexts; candidate preserves their bytes exactly despite the
same classifier. Both reporter 30-second WAVs pass actual Q4_K C ABI inference:
strict raw-token and result/word UTF-8, no U+FFFD, and 造/加 counts preserved
through cleanup. Clip 01 retains one 造; clip 16 retains all ten 加 generated
by this CPU run. These counts do not claim the reporter's GPU transcript matches.

The actual Metal probe returns NONE. This is macOS CPU/locale proof, not M4/
macOS26 or GPU inference acceptance. #538's shared post-decode byte-loss
cause is proved; final native CI still pending. Linux clip check remains queued.
Full proof is public at fixture dataset revision
`2cf3d11e58d88da740bbfcc7ec2167bb72db718d`,
`qwen3/locale-utf8-20261011/proof.zip`, SHA256
`c1b81aa064d50bee54d276b975a27af2aab6db71ea5b3fb6e034cdd36f059552`,
1,553,292 bytes, anonymous remote hash verified. Compact receipt:
`docs/qwen3-locale-utf8-2026-10-11.json`. #537 A/B is now running but cannot
be advertised as physical Metal acceptance without its actual device probe.

### 2026-10-11 HTDemucs integration complete

PR #537 merged as `ad7e91172f4bee46d2e8c7ea1a1765acb625f4b0`: all 35
applicable checks pass, two intentional skips. Independent hosted fused CPU
F16/Q8 outputs match byte-for-byte; every stem cosine/norm ratio 1, max error 0.
The contributor supplied a complete pinned Apple M4/macOS26.6.2 receipt:
original Q8 Metal exit134, candidate exit0; F16 Metal stems exact, Q8 Metal/
CPU cosine minimum0.999495 and magnitude ratio0.99953–1.00000. These are
attributed reporter hardware results; hosted Mac's Metal probe is NONE.
Source/input/models/stem hashes and limits are archived in
`docs/htdemucs-metal-views-2026-10-11.json`; full public independently hashed
proof is fixture dataset `684d1488e2833fc0448d7b01a344670743e0cbbb`,
`htdemucs/metal-row-views-20261011/proof.zip`, SHA256
`cd0944fdb42cde993c1ea4c95202b922b5dd00dbfa2e4074033b6b6d0ebd9298`.
Release notes and HISTORY include the fix. No music-corpus/speed claim.
#538 remaining CI, #529 numeric diagnosis and #536 server/model gates remain
pending; no release tag.

### 2026-10-11 Breeze boundary capture: final-slot convention corrected

GH 38150047103 is terminal FAILED at `9144c2796`: 105/107 oracle-fed
boundaries pass; all backbone/depth logits and all 16 stepwise frame-zero
codes pass. The two failures compare the original normalized final TE state
with the native pre-normalization residual (norm ratios 1702.6/1729.1).
Both original final slots are byte-identical to their respective normalized
hidden outputs. Native normalized outputs pass at cosine 0.999966/0.999938.
A separately copied archived-array control, explicitly relabelling only those
two slots while retaining raw residuals, passes 107/107 unchanged strict
gates. This is a diagnostic control, not a rebuilt native run.

The CLI harness now emits HF-compatible final normalized slots and retains
`te_seg{k}_final_pre_norm` separately. Runtime math and all generation gates
remain unchanged. All eight generation cases still fail exact codes; full
speech/ASR remains untested. Receipt: `docs/breeze-boundaries-2026-10-11.json`.
The corrected native harness will be rerun on the maintainer branch.
#536 portable timestamp fix now passes the hosted macOS build. #538 CI is
still settling without failures; no release tag has been cut.

Corrected harness checkpoint `aab781e77` is pushed on the maintainer branch.
Eight strict comparator controls pass; all four no-secrets tests pass with
zero skips (151.742 s). Rebuild/recheck GH 38152301380 is queued, reusing the
pinned original archive. No new reference weights or relaxed gates.

### CLAIMED 2026-10-11 — Breeze first differing depth decision

Continue in the existing `pr529-guided-validation` worktree and maintainer
branch. Pinned upstream source is read through depth forward, generation
preparation, cache updates and CFG/full-prefix paths. Add a design-s1 frame-zero
probe using the preserved original code history: original BF16 full-prefix and
cached depth forwards, a separately labelled FP32 depth-only promotion control,
and native F16 logits with original versus native backbone hidden states.
Preserve original fixture hashes, prompt parity, raw logits, ranks/gaps and
finite arrays. Native history-forcing is a diagnostic entry point; production
sampling remains unchanged. All CPU model/build work belongs on GH CI.
Do not replace canonical BF16 captures or relax exact-code/speech gates.

Breeze depth diagnostic source `f4c8cf0c0` is pushed; GH CPU 38154207238
is queued. It uses sequential original/native processes and retains six
variants' raw logits, ranks/gaps, cosine and norms under the same original
history. Original prompt IDs and prefill logits must be byte-exact against
the pinned archive before the probe proceeds. Production sampling defaults
are unchanged. Native runtime history controls are queued, not yet proven.
C++ syntax/Python compilation pass; all eight strict comparator controls,
90 model-free decision rows (including near-tie and 2x scale controls), and
four secret tests pass with zero skips (46.970 s). Launch receipt:
`docs/breeze-depth-probe-launch-2026-10-11.json`. Corrected full boundary
run 38152301380 is still in progress.

Linux reporter-clip run 38148682586 is SUCCESS at `7e3f768b8`: both actual
C ABI clips retain all 造/加 occurrences with strict UTF-8 and no replacement
characters. glibc C.UTF-8 does not treat 0xA0/0x85 as space, so this is a
regression control, not a Linux reproduction of the macOS libc defect.
Full proof is public, anonymously hash-verified, and pinned in
`docs/qwen3-locale-utf8-linux-2026-10-11.json`. No physical GPU claim.
#538 has 39 passing checks and six pending; merge awaits completion.

Raw boundary diagnostic proof is public at fixture revision
`dfcbfe37c52d7dab09f10c1c2617728b75380eb5`,
`breeze-guided-20261011/raw-boundary-diagnostic-proof.zip`, SHA256
`97afdcf4eed295075db5fdcb4ceff0c0d0a0f003306528fd81774f54fc605bdc`,
8,377,800 bytes. All 138 native arrays independently pass SHA/shape/type/finite
verification; anonymous archive download hash matches. The archive preserves
the raw 105/107 stage failure and excludes the relabelled control directory.
Main documentation secret scan passes all four checks with zero skips (60.565 s).

Corrected native rebuild 38152301380 is terminal FAILED at `aab781e77`:
all 107/107 oracle-fed boundaries pass unchanged gates; full guided code
parity still fails 8/8. The harness normalization fix is independently
confirmed by a rebuilt binary, not just archived relabelling. All 140 native
arrays independently match SHA/shape/type/finite receipts. Depth decision
probe 38154207238 remains queued; no decoded speech/ASR acceptance yet.

Full corrected rebuild proof is public at fixture revision
`e980fcac568798b62a4a1ffb26c7c272499e89be`,
`breeze-guided-20261011/corrected-boundary-rebuild-proof.zip`, SHA256
`ba5b995c8197c96d9197a0dc5a5d7e9c51593903b18aec5a1515fa87e781e0cb`,
8,575,560 bytes, anonymous download hash verified. Both the 107-stage
pass and unchanged eight-case generation failure are retained.

## CLAIMED 2026-10-11 — Compose pending Breeze synthesis surfaces

Worktree `/mnt/storage/crispasr/salvage-20261008/breeze-synthesis-composition`,
branch `test/breeze-synthesis-composition`. Compose current main, the maintainer
#529 diagnostic branch, #531 ownership/rate fixes and #536 per-request voices
without merging pending model features to main. Read-only merge audit finds a
C ABI conflict: #529 passes instruction but no reference; #531 passes reference
but no instruction. Resolve by passing both in one synthesis call. Test actual
C ABI dispatch with plain/reference/instruction/combined parameters and existing
ownership/failure guards. Preserve contributor histories and work on storage.
Source checkpoint and model-free checks precede GH model/ASR/server controls.
Existing depth probe branch stays pinned; no speech acceptance or release from
this composition alone.

### 2026-10-11 integration checkpoint

PR #538 merged as `7bcdbdfe1dc0f3e00a310fca1a9f521162b5d1d3` after
45 applicable checks passed, one intentional skip. Release notes and HISTORY
include the locale-dependent byte-loss fix and actual macOS/Linux C ABI proof.
Reporter M4/macOS26 confirmation remains pending; no physical GPU claim.
Earlier pending-CI entries above are superseded by this checkpoint.

Breeze composition source `a59772ba7f510518c9a704992d993faedcccff0e` is
pushed on `test/breeze-synthesis-composition`; full integration CI
[38156532213](https://github.com/CrispStrobe/CrispASR/actions/runs/38156532213)
is queued. It preserves all three contributor histories and passes 7 actual
compiled dispatch controls, 4 ownership/failure controls, 4 UTF-8 publication
controls, 8 strict comparator controls and 4 no-secrets tests (zero skips).
The dispatch tests replay the actual production function against backend stubs:
parent #529 fails 5/7 and parent #531 fails 4/7; composed source passes 7/7,
including plain/clone/design/direction, reused contexts, failure and optional
output count. Full C API syntax check passes with Breeze enabled and C2PA's source include path.
These are contract controls, not real-model CLI/C ABI/server or speech acceptance.
Receipt: `docs/breeze-synthesis-composition-2026-10-11.json`.
Depth diagnostic 38154207238 has started on GH CPU; its pinned branch remains
unchanged. Strict generated-code parity still fails all eight cases despite
107/107 oracle-fed stages passing; no merge of pending synthesis PRs or tag.

Final tracked documentation checkpoint passes all four no-secrets tests, zero
skips (53.812 seconds), and 227 internal anchors across 82 Markdown files.

## CLAIMED 2026-10-11 — Breeze backbone precision boundary probe

Continue in `/mnt/storage/crispasr/salvage-20261008/pr529-guided-validation`,
branch `fix/breeze-guided-validation`. Depth probe 38154207238 completed
successfully and all 106 arrays independently match hashes, shapes, dtypes and
finite receipts. Original BF16 cached/full-prefix and depth-only promoted FP32
all preserve all 15 design-s1 frame-zero picks. Native depth with original hidden
also preserves all 15. Native backbone from either original or native embeddings
flips only codebook 5: original 193 ranks second to 936, gap 0.016719/0.017995.
This localizes the first flip to backbone-state differences, not depth cache.
Capture original BF16 raw post-layer states and a separately labelled FP32
backbone promotion using identical BF16-rounded weights/embeddings; compare
native per-layer states and normalized final hidden. Retain original fixture,
prompt/logit equality, capture-off/on equality, scales and all raw arrays.
No canonical replacement, tolerance change or speech acceptance. GH CPU only.

Composition GH 38156532213 has passed Linux unit, Windows, macOS, iOS, Android
and AMR g++ jobs. Linux unit rebuilt actual source and passed all 2,019 CTests,
plus seven mode, four ownership and UTF-8 controls. Remaining jobs still run.
Fresh full Release dry run 38156833241 pins `3fdc1838dc1b95f517a3862a9379b24d20d6c0c1`;
no tag or publication requested. Packaging and actual fresh-archive GPU checks
remain pending. Earlier queue statuses are superseded by these observations.

Verified depth-decision receipt and public original arrays/logs are archived in
`docs/breeze-depth-decision-2026-10-11.json`. Public fixture revision
`dcaddf36f66d4932ec6f5391ec0e8239cc5cc61a`, proof SHA256
`8d10e90a0dc8a3da5ff6028753ce2b8ca641e2254feb5ca3a9049f3667d128fd`,
1,140,836 bytes; anonymous remote hash verified. All 106 array hashes/shapes/
dtypes/finite values independently verified. Backbone diagnostic worker passes
Python compilation, eight strict comparator controls and four no-secrets tests
with zero skips (51.214 seconds). It adds no production math changes.

Backbone probe source `82de8d9f5` is pushed on the maintainer validation branch;
GH CPU [38158522850](https://github.com/CrispStrobe/CrispASR/actions/runs/38158522850)
is queued with `--backbone-boundaries`. BF16 captures remain canonical; raw
post-block residuals and normalized hidden have separate slots. Native arrays
start as NaN so missing captures fail; capture OFF/ON/OFF must preserve final
hidden and logits exactly. No speech or release acceptance from this probe.

Queued backbone run 38158522850 was cancelled before execution after review
caught whole-module BF16 restoration casting originally FP32 buffers. Fixed
source `e7219be4b` restores every original parameter/buffer dtype separately.
A real tiny PyTorch module with BF16 parameters and mixed BF16/FP32 buffers
fails byte/dtype preservation under the old actual source block and passes
under the corrected block. All four no-secrets tests pass with zero skips
(47.462 seconds). Replacement GH CPU
[38158762302](https://github.com/CrispStrobe/CrispASR/actions/runs/38158762302)
is queued; no model result is claimed for either queued version.

PR #536 now passes all 51 applicable checks, one intentional skip. This does
not waive combined real-model CLI/C ABI/server ASR/speaker validation. Pending
Breeze synthesis features remain outside main/release. Integration CI has seven
passing jobs and the immutable release dry run has nine; no failures observed,
but remaining jobs are queued/running. Final tracked proof/docs scan passes
four tests with zero skips (55.208 seconds); all 227 anchors resolve.

## CLAIMED 2026-10-11 — Breeze session sampler wiring and decoded surfaces

Continue in `/mnt/storage/crispasr/salvage-20261008/breeze-synthesis-composition`,
branch `test/breeze-synthesis-composition`. Audit found Breeze open nested
inside CA_HAVE_CONFUCIUS4_TTS and runtime seed/temperature/max-token session
setters missing despite native hooks. Repair the independent compile guard,
forward open seed and public setter calls, observe guards fail on old actual
source and pass on repaired dispatch. Preserve all pending contributor histories;
no merge/default/quant change. Then run full compiled C ABI plus CLI/server
plain/clone/design/direction on GH CPU with pinned models, retained PCM,
ASR and closed-loop speaker controls. Model-backed evidence must distinguish
contract/repeatability, decoded speech and unresolved exact generated codes.
All source/model/build/proof work stays on storage; heavy models run on GH.

### 2026-10-11 Breeze sampler source checkpoint

Source `a2f58338ef9a86dfee114cdd5cd610dfe7b556c6` on the separate composition
branch fixes the independent Breeze open guard, open seed forwarding, runtime
seed/temperature/token-limit dispatch and checkpoint-default limit reset.
The documented `--tts-instruct` alias now parses alongside `--instruct`.
Eight actual compiled controls pass (old code fails 6/8), seven mode controls
pass, four ownership/failure controls pass, both C++ syntax checks pass and
all four no-secrets checks pass with zero skips (55.242 seconds).
Full current-source integration CI is
[38159958466](https://github.com/CrispStrobe/CrispASR/actions/runs/38159958466).
It supersedes the older partial integration run after source changes; that
run's eight passing builds and 2,019 Linux CTests remain attributed to its old
source. Real CLI/C ABI/server speech worker
[38159960891](https://github.com/CrispStrobe/CrispASR/actions/runs/38159960891)
is queued. It retains all PCM/ASR/speaker evidence for plain, clone, design,
direction, seeded reuse, negative-alone equivalence and unknown-voice recovery.
It uses the independently parity-tested Nemotron F16 recognizer, requires a
known JFK control first, then unchanged WER <= .2 and closed-loop clone speaker
similarity improvement. No speech result claimed yet, no gate waiver.
Receipt: `docs/breeze-session-sampler-2026-10-11.json`.
Backbone probe 38158762302 remains queued. Release dry run has twenty passing
jobs and no failure observed; remaining packaging/platform checks are pending.

### 2026-10-11 actual fresh CPU release packages

Downloaded the actual Linux CPU CLI/shared archives from immutable Release
38156833241, source `3fdc1838dc1b95f517a3862a9379b24d20d6c0c1`. CLI starts
and reports that source/version 0.8.42. Quantizer reaches its documented usage
path (exit 1 for insufficient positional arguments; startup only). Both CLI
executables use $ORIGIN and all seven ELF dependency closures pass, including
bundled OpenBLAS/OpenMP/Fortran/C2PA. Relocated shared-library load passes and
all 17 library closures resolve internally. Actual CLI/shared C ABI capability
comparison passes all 136 canonical rows and size/error contracts. Actual
whisper_version=0.8.42 and C API version=0.7.0 (deliberate stable ABI); session,
result, instruction and additive Nemotron preencode exports are present.
Loader overrides were cleared. Archives/log hashes and limitations are in
`docs/release-cpu-package-2026-10-11.json`. No models/GPU were exercised, no
new tag, and the full platform matrix is still pending.

Fresh CPU package raw logs and archive/contract receipts are public at fixture
revision `9c76e78e281c01fa0782c63bb45bf718551ba0b2`,
`release/38156833241/cpu-package-proof.zip`, SHA256
`234bcc11071e9f4bd520afa6e0f1dfe1559b4bec84d6e18c916c66774f8293af`,
81,689 bytes; anonymous remote hash verified. Final tracked proof/docs scan
passes all four tests with zero skips (52.341 seconds); public archive text
was also checked for credentials and the account name before upload.

## CLAIMED 2026-10-11 — independent Breeze session sampler repair

Isolate the existing-backend sampler fixes from the pending guidance/voice
composition in a new storage worktree, branch `fix/breeze-session-sampler`.
Main source confirms the same nested Confucius compile guard and missing
open-seed/runtime temperature/seed/token-limit dispatch. Preserve checkpoint
defaults, restore metadata token limit when clearing an override, and add
the already documented `--tts-instruct` CLI alias. No pending guidance,
reference ownership or server-cache feature is included. Compile the actual
source contract before/after (old failures retained), register its eight
controls in CTest and CI, and require normal full integration CI before merge.
The queued combined model speech proof remains separately attributed to its
composition source; these contract checks do not establish speech parity.

### 2026-10-11 independent sampler PR checkpoint

[PR #539](https://github.com/CrispStrobe/CrispASR/pull/539), source
`af26991cf37e6af14276a1ee5d4e124995811bc5`, isolates the existing-backend
Breeze opening/sampler fixes from the pending synthesis composition. All
eight compiled actual-source controls pass after six failed against unchanged
main; four no-secrets checks pass with zero skips (48.462 seconds), complete
C ABI/native Breeze C++ syntax and clang-format 18 pass. Full integration
38162567542 is queued, required before merge. Original failed/control raw logs
remain in storage; hashes and exact scope are in
`docs/breeze-independent-sampler-2026-10-11.json`. No model/speech acceptance
attributed to these recording-stub contracts. Backbone 38158762302 and
composition speech 38159960891 are still queued. Release 38156833241 has
24 successful jobs and no failure observed; Apple framework compilation has
finished and packaging continues. No tag/default/quant promotion.

## CLAIMED 2026-10-11 — original Breeze decoded speech comparison

Prepare a separate storage worktree/branch `test/breeze-original-speech` from
the pinned synthesis composition. The native CLI/C ABI/server speech worker
is queued; its receipt alone cannot establish matching original Python decoded
behavior. Prepare a GH CPU oracle that consumes its immutable artifact, checks
source/model/reference hashes, and generates plain/clone/design/direction using
the pinned upstream BF16 runtime, identical text/reference WAV, seed, temperatures,
top-k, repetition penalty and frame budget. Retain actual original prompts,
codes, PCM and all ASR transcripts. Release original model memory before using
the already verified actual CPU Python release archive and pinned F16 recognizer.
Require the independent JFK recognizer control and unchanged WER <= .2 for
both original and native cases; failures stay failures. Keep original/existing
exact-code diagnostics separate; no gate waiver or model/synthesis acceptance.
Heavy model work stays on GH, hardware work on Kaggle; no local large models.
Do not dispatch until the native worker artifact exists. Preserve all queued
workers' source branches unchanged until their checkout is observed.

### 2026-10-11 original-equivalence failure and speech-oracle checkpoint

Backbone 38158762302 failed before any native comparison: its fresh original
BF16 eager CPU logits failed the unchanged byte equality against the immutable
original fixture. Prompt equality had passed. Earlier depth 38154207238 passed
this control; the logged Torch/Torchaudio/NumPy/Transformers versions match.
No CPU/model cause can be established from the original failed run because
the assertion preceded array retention. Raw failed logs are now durable/public
at fixture revision `e4bdba818982aa579afb2114968eb36ddd246210`, proof SHA
`38cfde451e27e000db287f4c9ee80d4162a4b7d1308dccc67c667fc2a9481126`;
anonymous remote hash verified. Repair `5732be84fd9dc16911b3d317aada27820c18f8a5`
retains both arrays/scale/errors plus CPU/runtime/checkpoint hashes before the
unchanged assertion. Actual exact/one-ULP helper controls pass; four secrets
checks pass with zero skips (85.034 seconds). Version-pinned retry
38163903250 is queued. No native result claimed or gate waived.
Receipt: `docs/breeze-original-equivalence-2026-10-11.json`.

Original speech comparison worker `d89e854049bea3deb2cd691a8490cdd0b0229adf`
is prepared on `test/breeze-original-speech`: pinned original BF16 full
generation, identical native reference WAV/text/sampling settings, four modes,
retained prompts/codes/PCM and eight original/native ABI ASR roundtrips, each
requiring WER <= .2 after an independent JFK control. The actual verified CPU
Python release package and Nemotron F16 recognizer run after original model
process exit. Syntax/help/unchanged WER controls and pinned loader compilation
pass; four no-secrets checks pass, zero skips (44.972 seconds). Do not dispatch
until native 38159960891 uploads its artifact; that run has now started.
Receipt: `docs/breeze-original-speech-plan-2026-10-11.json`.
PR #539 remains queued; Release 38156833241 has 27 successful jobs and no
failure observed. Some Windows CUDA artifacts have arrived; both Windows
12.8/12.6 pairs and the fresh Linux CUDA hardware archive are not yet complete.
No release tag, default/quant promotion or synthesis feature merge.

## CLAIMED 2026-10-11 — Fresh release CUDA archive hardware validation

Reuse the existing prebuilt-package GPU worker in the separate storage worktree
`release-cuda-hardware`, preserving the old proof and source. Pin the actual
Linux CUDA archive from release dry run 38156833241 at source 3fdc1838dc, verify
its original artifact ZIP and tar SHA256, and publish a byte-identical immutable
public fixture mirror before dispatching. Keep all existing original-transcript,
stage, streaming, state reuse and actual CUDA allocation gates. No compilation
or CPU reference work on Kaggle. Archive terminal prior logs before repushing;
respect the single account and two-session cap. Fresh hardware acceptance remains
pending until the exact new archive passes; old archive results do not transfer.

Native Breeze speech run 38159960891 passed actual C ABI/CLI/server surface
contracts and all 20 recognition controls (reference included), then failed
loading the speaker encoder because webrtcvad imports missing pkg_resources.
No speaker acceptance is claimed. Retry 38169287241 keeps source a2f58338ef and
all gates unchanged, adding setuptools==80.9.0 to the worker dependencies.
Fresh Windows CUDA 12.8/12.6 paired-package proof is dispatched as 38169253465.

Breeze backbone retry 38163903250 succeeded at 5732be84f: original prompt/logits
are byte-exact; all 222 arrays independently match hashes, shapes, dtypes and
finite values. 116 comparisons cover all 28 backbone layers plus final norm,
two embedding sources, and canonical BF16 versus FP32-promoted arithmetic.
Native with oracle embeddings has minimum cosine 0.999999821 against FP32
and 0.999952674 against BF16. This supports arithmetic precision as a contributor
without resolving eight-case full-generation mismatch or proving its sole cause.
Full verified proof is public; details: docs/breeze-backbone-boundaries-2026-10-11.json.
Native speech raw 20 PCM arrays also independently match SHA/shape/finite controls;
C ABI, CLI and reused server WER is 0–0.1, reference WER 1/9. Speaker gate stopped
on missing pkg_resources and remains pending, not passed. Details:
docs/breeze-native-speech-2026-10-11.json.

Fresh release dry run 38156833241 is SUCCESS, all 35 build/validation jobs
green plus publishing bookkeeping with empty tag (nothing published). Exact
Linux CUDA tar 857a79c767ec91f406cceb114e4b330eb462b5c0aaee1e9fb3bda8fe758ae8c8
passes on dual T4s with CUDA runtime 12.8, driver 580.178.04. Actual packaged C ABI
loads its CUDA plugin and allocates 1330 MiB; three F16 stages, 26 original
NVIDIA transcript controls, reverse reused state and nine streaming cases pass.
Six full arrays and all original transcripts independently verified. Windows
12.8/12.6 CLI/runtime/library SHA pairing and mixed-version negative control pass
in GH 38169253465. Public proof and limitations are pinned in
docs/release-fresh-cuda-package-2026-10-11.json and
docs/release-fresh-windows-pairs-2026-10-11.json. No tag or pending PR promotion.

Native Breeze retry 38169287241 is SUCCESS at unchanged a2f58338ef: all 20
retained PCM arrays independently match SHA/length/finite controls, every ASR
check passes and the independent JFK recognizer has zero WER. Closed-loop
clone/reference cosine 0.811606526 exceeds baseline/reference 0.449814618.
Speaker model SHA and actual source/model/codec pins are retained. Full
public proof is anonymously hash verified. This proves the composition's
native speech surfaces on these fixed cases, not original full-generation
parity or every backend in #530/#531. Original decoded speech comparison
38172313354 is dispatched from d89e854049 with this successful native artifact;
its original/native eight ASR checks remain pending.

## CLAIMED 2026-10-11 — Trace backbone precision into depth decisions

In the existing storage diagnostic worktree pr529-guided-validation, add an
opt-in precision-decision control to the existing teacher-forced probe. Feed
the captured original FP32-promoted backbone hidden state into original BF16
and FP32 depth decoders and into the native depth graph under the exact same
original code history. Preserve all existing canonical captures, strict
original byte controls, defaults and generation gates. Retain each new raw
logit array, decision/rank/gap, cosine and magnitude. This tests whether the
backbone precision difference itself can reproduce the first divergent
codebook choice; it does not replace canonical BF16 or imply acceptance.
Use GH heavy CPU, no local model execution and no Kaggle CPU work.

Independent Breeze sampler PR #539 is MERGED as 913266271cf after 56 passing
checks and one intentional skip. Release notes and history now include only
this accepted existing-backend wiring; #529/#530/#531/#536 remain pending.
Original speech run 38172313354 failed before any generation: generic HF
repetition penalty indexed mixed text prompt ID 262146 into the 2052-entry
audio vocabulary. Repair uses the pinned upstream generated-backbone-only
penalty helper via the supported custom logits processor API, keeping effective
penalty 1.1 and excluding prompt/reference and depth codebooks. No checkpoint
or upstream source is modified. Actual tiny Torch/HF controls reproduce the
old crash and prove prompt exclusion, CB0-only history, signs, unique duplicates,
stateless reuse and invalid-batch rejection. Original full-generation output
and all unchanged speech gates still require a fresh GH run.
Precision decision probe is dispatched as 38172976584 at fd2f9b2fa; canonical
90 logger decisions are unchanged in same-runtime A/B, missing captures reject,
45 new logger routes pass. These are logger controls, not new model evidence.

## CLAIMED 2026-10-11 — Prepare isolated v0.8.43 release candidate

Use a new storage worktree and release/v0.8.43-candidate branch from accepted
main including #539 and comprehensive notes. Run the repository bump-version
script once for 0.8.43, preserving its annotated local tag, then push only the
candidate branch and dispatch the complete empty-tag release dry run at that
exact event source. Do not publish the tag until candidate CI/package checks
and the exact candidate CUDA archive hardware test pass. Old 3fdc1838 archive
acceptance does not transfer to the sampler change or version-bumped candidate.
Pending feature PRs and rejected quantization candidates are excluded.

Isolated v0.8.43 candidate is frozen at eb633d908120cc5f44b8af8cd4aeb1191f16bfd8,
branch release/v0.8.43-candidate. Repository bump script ran once; required
Flutter changelog was added before validation. All version fields agree, all
227 doc anchors pass, and four secret tests pass with zero skips (45.769 s).
Annotated v0.8.43 tag exists locally only and has NOT been pushed. Candidate
CI 38173748841 and full empty-tag release dry run 38173750598 are dispatched.
Do not reuse old 3fdc1838 archive acceptance; inspect these candidate artifacts
and rerun actual CUDA hardware controls before publication. Main remains
0.8.42 until candidate integration. Receipt:
docs/release-candidate-v0.8.43-2026-10-11.json.
Original speech repair 080a0fe5a is dispatched as 38173377874 with the unchanged
complete native speech artifact 38169287241; four secret tests pass with zero
skips (204.863 s). Precision-decision run 38172976584 is in progress. No new
original decoded or alternate-state model result is claimed yet.

Precision-decision run 38172976584 FAILED at its unchanged original-only
byte control before native comparison. All eight arrays independently verify.
Original logits cosine 0.999991075, norm ratio 1.001262460, relative L2
0.004412063, max error 0.0625; 1469 values differ, argmax 404 unchanged.
Package versions and all checkpoint SHA256s match successful 38163903250.
That run used AMD EPYC 7763/PyTorch AVX2; this failure used Intel Xeon
Platinum 8573C/PyTorch AVX512. Prompt IDs/mask match exactly; already-assembled
original embeddings differ (relative L2 0.013288). CPU dispatch is a hypothesis,
not a proven cause. Full failed arrays/provenance are public and anonymously
hash-verified in docs/breeze-reference-dispatch-2026-10-11.json. Next useful
control is default versus explicit AVX2 dispatch on the SAME GH host before
retrying native precision decisions. Preserve both results and every gate.
Release candidate excludes the pending feature and is unchanged at eb633d908.

Original/native decoded speech run 38173377874 is SUCCESS at
080a0fe5a17a32cf9515e8077d76856257ad0689. All four original codec/PCM pairs
and all eight original/native PCM roundtrips independently pass hash/finite
checks. Original plain/clone/design/direction WER: .2/.1/0/0; native:
.1/0/0/.1. Unchanged WER maximum .2, independent JFK recognizer WER 0.
Full original and native arrays, transcripts, executed loader and provenance
are publicly archived and anonymously SHA256 verified. See
docs/breeze-original-speech-plan-2026-10-11.json. This establishes fixed-control
decoded speech quality, not exact generated-code parity, GPU portability,
streaming speed or acceptance of every backend in the pending feature PRs.

Same-host original CPU-dispatch A/B is dispatched as 38174972934 at
0fbdbeff8040cf3bc6192ce14cfca2964c266627. Fresh default and AVX2-capped
processes run the unchanged original-only probe, seed 42 before loading,
retain all failed arrays, and require both immutable byte controls to pass.
No local model execution or native acceptance. Candidate eb633d908 is
unchanged; release jobs are building, with no failures at this checkpoint.

2026-10-12 checkpoint: candidate CI 38173748841 SUCCESS (13 jobs); release
dry run 38173750598 SUCCESS (35 build/validation jobs plus empty-tag bookkeeping;
nothing published). Continue the already claimed exact-candidate CPU/Python,
Windows CUDA-pair and actual CUDA hardware checks against eb633d908. Repin
existing package worker v6 to its byte-identical candidate CI archive, preserving
all original stage/26 recognition/9 streaming controls. Archive v5 before push.

CPU dispatch diagnostic 38174972934 FAILED before model loading in BOTH arms:
runpy bootstrap omitted the script directory, so sibling breeze_guided_reference
cannot import. No reference arrays were produced; no CPU precision conclusion.
Repair the bootstrap import path and test sibling imports in actual fresh
processes before retrying the already claimed same-host experiment. Preserve
both failed logs and every immutable byte gate.

Candidate eb633d908 actual CPU CLI/library/Python packages PASS: dependency
closure (glibc <= 2.34), relocated load, all 136 CLI/C ABI capability rows
including error/buffer contracts, version 0.8.43 / stable C ABI 0.7.0, required
exports, binding byte equality and automatic bundled-library discovery with
loader overrides cleared. Full proof is public and anonymously hash verified
in docs/release-candidate-v0.8.43-2026-10-11.json. Candidate Windows 12.8/12.6
CLI/runtime pair and mixed-runtime rejection run 38198054235 is in progress.

Actual candidate CUDA CI ZIP 11719048109 and tar hashes are verified. Public
byte-identical mirror is pinned; worker 26a369a7d v6 is pushed to Kaggle and
RUNNING. Source pull is byte equal; v5 terminal logs were archived first, all
20 recent kernels were terminal (zero active). Same three stage/magnitude,
26 fresh/reused speech and nine streaming packet/reset/flush gates remain.
No build, model conversion or CPU-only job on Kaggle. Release tag stays local.

CPU dispatch bootstrap repair 7bcf259d2 adds sibling import path before runpy.
Actual fresh-process controls reproduce the old failure and prove new import,
argv, seeded RNG and selected AVX2 behavior; four secret checks pass, zero skips.
Retry 38198367653 is queued with all original byte gates unchanged. The failed
38174972934 logs remain archived; its zero-array result gives no precision
conclusion. No issues/PR heads or reporter follow-ups changed since last audit.

Candidate Windows package pair run 38198054235 is SUCCESS: both actual 12.8
and 12.6 CLI manifests, runtime ZIPs and shared-library archives have matching
three-DLL SHA256s; mixed 12.8 CLI / 12.6 runtime is correctly rejected. This
is package consistency, not physical Windows GPU validation. All 227 anchors
pass; four secret checks pass with zero skips (121.399 s). GPU v6 remains
RUNNING; repaired CPU dispatch retry remains queued at this checkpoint.

Candidate CUDA v6 is COMPLETE and PASS on dual T4 (actual archive source
eb633d908, plugin SHA byte-matches extracted CI archive; C ABI loads bundled
CUDA plugin and owns 1330 MiB). All six full stage arrays independently verify
cosine and magnitudes: encoder cosine .999990880489, relative L2 .00028003938;
26 original word controls and fresh/reverse-reused output equality pass; nine
streaming packet/reset/flush controls pass. Full immutable hardware/Windows
proof is public and anonymously hash verified. All candidate requirements are
satisfied; integrate frozen source and push the annotated bump-script tag.
Tagged rebuilds must be tracked separately from validated dry-run archive pins.
No pending feature, quant pin or default enters the release. CPU precision
diagnostic 38198367653 remains queued independently of this accepted release.

Frozen candidate integrated into main as d71153547a6ed76b5d5ec3d9881a7c5dd1a9b284.
Only four maintainer documentation files differ from validated eb633d908;
production sources and versioned files are identical. Annotated v0.8.43 tag
is pushed at eb633d908. Draft release with comprehensive notes exists; tagged
release run 38199002944 is queued. Keep draft until final assets are complete;
final release notes body is prepared with immutable proof links, and must be
restored after workflow notes attachment. Candidate archive hashes and tagged
rebuild hashes are separate provenance. Version sync and 227 anchors pass;
four secret tests pass with zero skips (50.370 s).

Same-host CPU control 38198367653 SUCCESS: all 106 arrays independently verify
hashes/shapes/dtypes/finite values; 53 default/AVX2 pairs byte-identical; both
original prompt/logit byte gates pass. Both paths were AVX2 on AMD EPYC 7763,
so this gives no AVX512 contrast and does not explain the Intel mismatch. Full
immutable successful and failed-bootstrap proof is public. Native not executed.

## CLAIMED 2026-10-12 — Execute precision decisions under explicit AVX2 control

Add a small GH CPU entrypoint selecting AVX2 for PyTorch/oneDNN/MKL before
Torch import, seeding initialization identically to the successful same-host
control, and invoking the unchanged breeze_depth_probe.py --precision-decisions.
Preserve default probe, canonical BF16 fixture, full original byte control and
every native decision/generation gate. Record complete arrays/provenance and
strict failures; no native math/default or model pin changes. This extends the
existing first-near-tie precision investigation, not a cause or parity claim.
Actual fresh-process controls must verify path/argv/seed/dispatch before GH.

Explicit AVX2 precision-decision worker 8ed327bc53f8eb9d86df62ecacec54908175b6a1
is dispatched as 38199379678. Wrapper selects PyTorch/oneDNN/MKL AVX2 before
Torch import; optional seed 42 is inherited by the fresh original process and
recorded in provenance. Ordinary probe behavior and every strict original
byte/native decision gate are unchanged. Actual model-free reference controls
stop before first download and prove default RNG preservation / opt-in seed;
fresh entrypoint control proves selected AVX2, inheritance and argv. Four
secret checks pass, zero skips (49.997 s); no local model execution.
Tag release 38199002944 is building with no failures at checkpoint; v0.8.43
release remains draft awaiting final assets and final notes restoration.

Release publication correction: the existing per-job softprops actions publish
the initially created draft as soon as assets land. v0.8.43 is now PUBLIC with
the first two assets, and tagged run 38199002944 is still building. Prior
statements that it remains draft are superseded. This follows the existing
incremental release workflow; candidate validation remains complete, and final
tagged-asset completion/hash verification and final notes restoration remain.


Explicit AVX2 precision run 38199379678 SUCCESS. All 268 full arrays and 135
codebook decisions independently verify; canonical original prompt/logits are
byte-exact. Original-only promoted backbone plus FP32 depth reproduces the
first native codebook-5 flip 193 → 936. Promoting depth alone, or using BF16
depth with the promoted backbone, retains 193. Under exactly the same promoted
hidden state, original FP32 and native depth agree on all 15 picks; minimum
full-distribution cosine .99999975436. Joint arithmetic sensitivity is sufficient
for this isolated flip. This does not explain every free-running mismatch,
accept the eight failed generation cases, or establish the Intel mismatch cause.
Full immutable proof is public and anonymously SHA256 verified; canonical
fixtures, generation gates, defaults and production math remain unchanged.

## CLAIMED 2026-10-12 — Breeze composition CUDA speech validation

Build the already CPU-tested pending-feature composition a2f58338 with CUDA on
GH CI, including CLI, C ABI, server and diff tools. Use a probe-only packaging
branch; no production graph or default changes. Kaggle is only for actual GPU
execution: verify bundled plugin/device allocation, F16 conditioning and reuse,
decoded ASR roundtrips, stage magnitudes/cosines and quant controls where pinned
fixtures permit. Archive all terminal evidence and retain strict failures.
This evidence supports review of pending #529/#531/#536; no feature acceptance
or merge follows merely from a green build or the isolated precision control.


Published tagged CPU CLI, shared library and Python archives independently
verify against GitHub asset SHA256/size, and pass dependency closure, relocated
library load, C ABI 0.7.0, 136 capability/error contracts and automatic packaged
Python backend discovery. Actual CLI reports 0.8.43 / eb633d90. Immutable
public proof is anonymously hash verified; tagged rebuild hashes are distinct
from the validated dry-run candidate. CUDA/Windows builds remain in progress,
without failed jobs; final matrix/asset completion and notes restoration remain.


Breeze composition CUDA package dispatched as GH run 38205813889 at source
e4c8461b900084afe5320ea2d734d3d86dfa644c, based on CPU-tested a2f58338.
Probe-only reuse of the existing CUDA workflow builds sm75 / toolkit 12.4.1,
CLI with server support, shared ABI, diff, quantizer and matching bindings;
user-space runtime is bundled, driver/stub is excluded. Build is queued,
no GPU/model acceptance claimed. Local Python syntax/diff pass; four secret
checks pass with zero skips (63.289 s). Main receipt/claim/precision/public
CPU asset pins also pass 227 anchors and four secret checks, zero skips
(116.727 s). Exact artifact/public mirror pins are required before GPU launch.


CUDA composition run 38205813889 is compiling after successful toolkit setup.
Probe worker 9f2c9c8a9 is prepared: unchanged 20 speech/ASR/reuse controls, actual
per-process CUDA allocation witnesses, GPU speaker encoder, full original stage
arrays/cosine/magnitude controls and separately strict guided-generation gate.
Nonidentical guided codes keep the worker failed even if speech/stages pass.
Archive pins deliberately remain BUILD_PENDING and prevent launch before model
downloads; no Kaggle execution or quant acceptance claimed. Worker syntax/diff
and four secret tests pass, zero skips (52.065 s). v0.8.43 tagged release has
35 public assets and no failed jobs; six remaining build jobs are in progress.
Comprehensive public release notes retain the final 17,509-character body.

## CLAIMED 2026-10-12 — Dolphin noncausal checkpoint bug #540

Read the complete issue (no comments). Audit both encoder depthwise convolutions
against pinned upstream source and the non-streaming checkpoint configuration.
Implement causal/noncausal padding and CSGU normalization order correctly, with
actual ggml controls that fail on the previous causal-only behavior. Use a
separate storage worktree and GH CPU for conversion, original per-stage dumps,
diff and decoded speech on both non-streaming and existing streaming models.
Do not alter the already tagged v0.8.43 release or accept a model from a build.


Dolphin fix 014f20755 is pushed as draft PR #541. Original upstream CSGU and
actual encoder-layer merge convolution generate twelve native ggml controls:
12/12 pass (max error 1.79e-7); old causal-only mutation fails six noncausal
cases, padding-before-normalization mutation fails three noncausal CSGU cases.
Full original/native/mutation proof is public and anonymously hash verified.
Source syntax/clang18/index/227 anchors and four secret tests pass, zero skips
(49.679 s). Both immutable original checkpoints/configs are pinned; GH CPU
38207601849 is queued for conversion, fresh Chinese/JFK original dumps, diff,
complete native per-stage cosines/magnitudes and automatic C ABI speech.
Keep draft/issue open until that evidence; private reporter clip/Kylin not
reproduced. The frozen v0.8.43 tag is unchanged. A new learning and its generated
index are on the fix branch, alongside the actual native CTest controls.

Tagged Windows builds completed. Actual tagged CUDA 12.8/12.6 pair validation
is dispatched as 38207603649; pending. Remaining release builds are Linux CUDA
CLI/shared and HIP CLI, no failures at checkpoint. Breeze CUDA composition
38205813889 continues compiling; worker 63f4ef80c additionally isolates Torch
speaker inference in a separate GPU process, requiring the native C ABI to
load the bundled CUDA runtime/BLAS files. Four secret tests pass, zero skips
(47.728 s); immutable archive pins remain pending and prevent Kaggle launch.

## CLAIMED 2026-10-12 — Final tagged CUDA CLI/shared package hardware validation

All 35 build/validation jobs in tagged release run 38199002944 passed; final
publish succeeded and all 50 public assets are present. Restore comprehensive
final notes after workflow attachment and mark the release latest. Windows
12.8/12.6 actual tagged package pair run 38207603649 succeeded independently.

Independently verify actual published CUDA CLI and shared-library tars against
GitHub asset digests and byte-identical CI ZIP contents, then create immutable
public mirrors. The final CLI CUDA plugin differs from the candidate; the
final shared-library plugin also differs from the CLI plugin. Run both actual
packages on Kaggle, in separate fresh processes, retaining the unchanged three
full stage arrays/cosine/magnitude gates, 26 original speech/reuse controls and
nine streaming reset/packet/flush controls for each. No archive/hash substitution
or untested hardware acceptance. Archive terminal v6 logs before repushing,
check the single account has at most two active sessions and pull back exact
submitted source. Large assets/evidence stay on storage. No production changes.


v0.8.43 final tagged workflow 38199002944 is SUCCESS, all 50 assets public;
comprehensive 17,509-character release notes restored after publication. Actual
tagged Windows CUDA 12.8/12.6 CLI/runtime/library pairing 38207603649 passes,
including mixed-runtime rejection. Both actual Linux CUDA archives independently
match CI and GitHub digests and immutable anonymous public mirrors. Their GPU
plugins differ from the hardware-tested candidate and each other, so prepared
v7 validates both in fresh processes; hardware acceptance remains pending.

Breeze CUDA packaging retry 38209137738 SUCCESS at 557e2cc7, sm60/sm75. First
run compiled successfully but omitted an unversioned library alias; repaired
builder preserves SONAME aliases. Queued sm75-only intermediate retry canceled
before execution. Exact archive verification/pinning precedes GPU launch.

Dolphin full checkpoint retry 38208703707 passes all 16 nonstreaming Chinese
crispasr-diff stages and exact original decoded raw speech. Automatic Python
Session opening then fails: converter writes dolphin but shared architecture
table omits it. b87235519 adds the entry and C ABI regression probes, preserves
partial native receipts, and dispatches full two-model/two-sample retry
38211719379. No full checkpoint or reporter-machine acceptance yet.

## CLAIMED 2026-10-12 — Canary #532 actual CUDA allocation/reuse probe

Refresh complete issue/comments (no new reporter data). Existing CPU lengths
11/4/33/90/11 s, buckets 0/128 all pass but do not exercise the GPU allocator.
Use a separate storage probe worktree and exact verified tagged CUDA archive,
pinned 392 MB Q4_K aligner, and GPU-only Kaggle execution after a slot is free.
Capture all ten complete logits arrays, returning-short fresh/reuse equality,
actual process GPU allocation and exact packaged CUDA-library byte witness.
Compare saved CPU arrays only as CPU/GPU diagnostics, not original parity.
Keep any allocation failure; no speculative runtime fix or claim of replaying
the private German/VAD/diarization recording. No new native build is needed.


Final tagged CUDA worker v8 COMPLETE PASS on dual T4s, CLI and shared archive
in separate fresh processes. Independently verified 12 full arrays, 52
original word cases, 18 streaming controls, exact archive/plugin byte pins.
Public complete GPU/Windows proof a5c5642f / tagged-cuda-windows-package-proof.zip
SHA fddbd06d970fe7fa199cb5aa1e55cd47970843cf04bc2a155e2b8b649751251d.
Release notes now describe actual tagged bytes, all 50 published assets.

Breeze exact CI ZIP CRC and tar SHA/size, all 25 bundle files (including
SONAME aliases), and anonymous immutable mirror verified. v1 GPU guard
correctly fails before speech because host CUDA12.8 replaced bundled12.4.
Archived failure, repaired process-start library isolation; v2 d22fa7caa
submitted and source pull byte equality PASS. Speech/stage/generation pending.
Canary CUDA probe 78f0c269d submitted with source pull byte equality PASS;
independent processes for cached bucket values 0/128; acceptance pending.


Dolphin #541 is ready for review after full CPU 38211719379 SUCCESS and
independent recomputation of 128 complete arrays / 64 stage comparisons.
All four decoded original raw transcripts and automatic Session paths pass.
Full anonymously verified proof: 25c38157 / dolphin-20261012/full-checkpoint-proof.zip,
SHA e17f4758518b7edb319d5f45bfa23480d05b25082fdad2a7cd90004db7c21fc7.
Remaining general PR CI gates before merge; published v0.8.43 unchanged.

Canary CUDA v1 COMPLETE: all ten 11/4/33/90/11-second arrays finite, exact
fresh/reused returning shape, both buckets allocate on actual T4. Independent
array/hash/cosine/norm verification complete and public proof 182a2bd4,
SHA b9e61df6f1044a4c97be573d74d9e0c3253260c1027cf30337b1da7997194522.
No crash reproduced or original parity accepted. CPU/GPU Q4 min cosine .981333
and argmax differences are retained. v2 probes consecutive same-bucket graph
reuse at 11/11.05/11 seconds in each independent bucket process.

Closed shipped/resolved #481, #478, #533 after full report/comment/code audit.
Remaining exact reporter hardware/performance limits documented in replies.
Release-next draft now starts after published v0.8.43; historical comprehensive
notes remain in RELEASE_NOTES_v0.8.43.md and publication in HISTORY.md.


Dolphin #541 merged 065ae1fdd after all 45 applicable general checks pass
(one intentional skip); #540 closed. Full two-model/two-sample original proof
and automatic C ABI validation accepted. No change to frozen v0.8.43 tag.


## CLAIMED 2026-10-12 — Breeze direct-init CUDA registration order

Completed CUDA worker v4 proves 20 speech/interface controls and 107 original
stages pass, but all eight strict guided-code gates fail. Guided process
allocates only 868 MiB while stage process allocates 6482 MiB: code audit finds
Breeze selects the GPU before core CPU initialization registers dynamic
plugins; the later codec runs on CUDA while Breeze stays on CPU. Fix the
Breeze initialization order in the separate composition worktree, record
the actual selected Breeze backend, and build/test on GH CI plus Kaggle.
Require that backend witness for both stage and guided processes; preserve
all original fixture, cosine/norm, exact-code and speech gates. Archive the
completed mixed-device v4 separately and never call it full CUDA generation.


Canary CUDA cached follow-up 9132234df COMPLETE PASS: sixteen full logits
arrays independently verified, including six consecutive same-bucket cases
11/11.05/11 seconds. Initial ten arrays byte-equal prior GPU probe, returning
and fresh/reused outputs exact. Public proof 29d14881 / cached CUDA proof,
SHA bff8914b2d881d023ecb4b56d66ea827c11e50aa6c4dd33aa11be2c55ba4c389.
Reported allocation crash still not reproduced; Q4 CPU/GPU differences remain
diagnostic, not original parity. #532 stays open awaiting assertion/device/clip.

Closed shipped/resolved #488 (hotword system turn), #482 (AudioSeal graph
capacity) and #484 (Intel SIMD release packaging), with published evidence
and exact private-hardware/recording limitations in each issue reply.
Current audit: six open issues, six open PRs; no new contributor heads.


Breeze v4 COMPLETE: independently verified 163 arrays, all 107 stages
(min cosine .9996649853; norm ratio .995455–1.003287), 20 speech/interface
cases (max WER .111111), speaker clone/reference .894997 versus baseline
.527019. Strict code generation 0/8 still FAIL. Direct-init code audit
identifies CPU Breeze plus CUDA codec in the guided dump; no full CUDA guided
acceptance. Public immutable mixed-device proof edf3eb38, SHA
1a61be46acbff0602f30181cbc86a8592ace624ebeccc5bd65ad4069254dd6ab.
Registration-order fix 435c68486 queued GH build 38219077361; original
gates retained and next GPU worker must witness Breeze's own CUDA backend.


## CLAIMED 2026-10-12 — All-eight original Breeze arithmetic control

The isolated design-s1 frame-zero probe explains one argmax flip when both
original backbone and depth arithmetic are promoted to F32. Extend that
original-only diagnostic to all eight full 24-frame guided generations on
GH CPU with explicit AVX2 dispatch and initialization seed 42. Keep original
BF16-rounded weights, text/codec computation and canonical fixtures; promote
backbone/depth/LM-head arithmetic only, explicitly casting backbone inputs.
Compare complete generated grids to immutable canonical and saved CPU/GPU
native diagnostics without changing any acceptance gate or fixture. This
control may explain rounding amplification; it cannot certify the port.


Independent production registration-order fix is draft #542, head 7a33fe272;
CUDA composition build 38219077361 at 435c68486 remains queued. Worker guard
036ab9fb5 now demands explicit Breeze CUDA compute-backend names plus >4 GiB
resident allocation in both native dumps, so CUDA codec alone cannot pass.
Original-only all-eight arithmetic control 31c338218 queued GH CPU
38219796806 with pinned package versions, explicit AVX2 and seed 42. Neither
queued run is acceptance. Canary #532 and Breeze #529/#536 updated with
immutable complete proof links and exact scope/failure limitations.


Breeze corrected CUDA build 38219077361 SUCCESS, exact tar SHA
b1306a24ef0f460c0ed84bbcfa07712af1eecab442f8b58a37256b2eb196b3e7,
all 25 bundled file hashes/aliases and anonymous mirror bed22cc2 verified.
Worker v5 b5199d969 submitted; source-pull bytes identical, actual GPU running.
Requires Breeze's own CUDA backend witness; all old strict gates remain.

All-eight original joint-F32 arithmetic control 38219796806 SUCCESS, 145
complete arrays independently verified. All eight promoted originals differ
from canonical BF16; design-s1 matches the native CPU's ENTIRE 24x16 grid.
Other seven promoted-original/native grids still differ: this explains one
complete case, not all guided failures. No fixture/default/gate changed.
Public control proof fc241f7a, SHA
d58605dc03e78b426bd8b032fbcd452ca5a73ce5c05d08231b6d011f28cff130.


## CLAIMED 2026-10-12 — Original Breeze F16 KV-storage control

The full joint-F32 original control reproduces native design-s1 exactly, but
not seven other grids. Native backbone/depth KV storage is F16. Run the same
original-only control with explicit F16 rounding on F32 DynamicCache writes,
preserving F32 attention math and BF16 text/codec caches. Apply the same
rounding to depth attention K/V in its uncached CFG forwards, without changing
that original inference flow. Capture actual cache-write and attention counts
and rounding-error witness; compare all eight complete
code grids. GH CPU only, no canonical fixture or acceptance-gate change.


Breeze v5 COMPLETE: actual Breeze CUDA0 6482/6924 MiB in stage/guided
dumps, all reuse/scale self-checks pass. Independent verification: 163 arrays,
107 original stages, 20 speech/interface controls; all 20 PCM arrays byte-equal
prior real-CUDA run. Speaker .894997/.527019 unchanged. Strict guided
codes 0/8 still FAIL and the worker remains red; no feature acceptance.
Full immutable CUDA proof 48cff19f, SHA
4c368fa9158c6abc4a6fb06c583bcad5119bd695b7be6960e4ed5179fd03f304.
Independent #542 initialization repair awaits general CI; hardware evidence
is the composition package, with identical device/weight-load source prefix
and explicit separate branch-cache implementation. Original-only F16 KV
control c9575d6be queued GH CPU 38223133652; all 4 secret checks pass, zero
skips (97.452 s), helper rounding invariants and eight strict comparator
controls pass. No default, fixture, math or acceptance gate changed.


## CLAIMED 2026-10-12 — Original Breeze full-inference F32 control

F32 transformer/F16 KV control independently verifies 145 arrays and real
rounding (37,632 cache writes, 47,700 depth forwards). It reproduces 2/8
native CPU grids, 3/8 actual CUDA grids; canonical BF16 exact gate stays 0/8.
Remaining plain/direction first divergence could originate in BF16 text/audio
embedding arithmetic retained by these controls. Run an original-only full
inference F32 control on GH CPU, preserving BF16-rounded loaded weights and
the unused training codec plus external audio tokenizer. Leave canonical
fixtures, generation recipe and gates unchanged; record module dtypes and
compare every complete grid and magnitude/cosine. No native math/default edit.


PR #542 MERGED `f8eb4325e`: 36 applicable checks pass, two intentional skips.
Main CI is queued/running; actual GPU composition scope remains explicit.
Original F16 KV control COMPLETE: 145 independently verified arrays; 2/8 CPU,
3/8 actual CUDA, 0/8 canonical BF16 exact grids. Public proof `b826a537`, SHA
`38b3977e0caf693639eb7bdeb1d5e0b636854cc48a925123d3e7c1c04baaffb2`.
Whole-inference F32 control 38227576695 remains running; no gate/default change.


## CLAIMED 2026-10-12 — Refresh PR #531 and protect Supertonic speaker getters

Audit found the contributor Supertonic getters dereference null contexts and
accept negative indices through the direct API. Refresh #531 against main,
preserving contributor commits and previously merged UTF-8/Breeze fixes; add
weight-free source-compiled failure controls before correcting those guards.
No model arithmetic, quantization or defaults change. Keep broader acceptance
limitations and run hosted general CI on the resulting exact head.


Whole-inference F32 original control COMPLETE (38227576695, source 95330ca08):
145 arrays independently verified; 5/8 native CPU and 5/8 actual CUDA complete
24x16 grids match, 0/8 canonical BF16. Parameter bytes 10,628,513,410 ->
13,706,837,634, all active inference parameters F32, unused training codec
preserved. Final precision isolation proof `b3d1ef591`, SHA
`66990b07a0d6c9e3e0fd481bbbfb525afdad5f4b710896c6d758620444f781e8`,
anonymous full ZIP verified. No fixture/default/math/gate change; remaining
three grids differ, so no claim of complete integer generation parity.


## CLAIMED 2026-10-12 — Restore complete generated capability coverage in #531

Refreshed #531 has 52 passing checks, two intentional skips, and one ONNX
acceptance failure: CPU-only feature generation omitted optional Moonshine
ONNX rows. Run the contributor generator worker with the pinned ONNX SDK,
verify complete CLI/C ABI rows, and apply generated outputs unchanged. Retain
required wiring checks; no manual table edits or model/gate changes. Hosted
unit tier proves UTF-8 cleanup and all speaker/reference contracts pass.


## CLAIMED 2026-10-12 — Refresh validated Breeze contributor composition

Prepare a separate integration worktree from the hardware-tested composition,
preserving #529/#530/#536 contributor commits. Merge current main and corrected
#531, resolve CLI/C ABI sampler/reference/instruction compatibility, then run
source-compiled contracts and hosted CLI/library/general CI. Retain canonical
BF16 exact-code diagnostics; assess stage magnitudes and the already recorded
original/native decoded roundtrips under the authoritative development method.
No arithmetic, quantization, recipe or default change to force exact AR codes.
Keep unrelated model quality reports separately pending.


## CLAIMED 2026-10-12 — Review and validate PR #544 CANN selection/build coverage

Whole PR description, comments and three contributor commits read at head
`b80272710ab378c6fd821c3c7142dceabc65a131`. Use the separate storage worktree
`/mnt/storage/crispasr/salvage-20261008/cann-pr544`. Fix its formatting gate,
check requested CANN device selection with production-source controls, and run
the opt-in x86_64/910B and arm64/310P container compile jobs on GitHub. Preserve
contributor history; compilation is not Ascend hardware/model acceptance.


## UPDATE 2026-10-12 — Refreshed integration proof and CANN review

PR #531 head `d9fa1574e` corrects the unit job's native-only roster mismatch by
fetching the pinned ONNX SDK. Hosted linux-unit run `38251502074`, job
`114818732081`, passes exact generation (140 rows), actual CLI/C ABI
capabilities/size/error contracts, all 2009 weight-free CTests, and ownership,
speaker, locale, sampler and UTF-8 suites. All 48 applicable checks passed, one intentional skip; #531 merged as `d73e39dc8c3f68b485e31b06c26d1351f41654f4`. GitHub also marked its preserved contributor ancestor #530 merged; the model-quality reports in that PR remain unresolved.

PR #543 head `986ac0aba` includes the CI repair and pinned refreshed CUDA
worker v6. Build `38236488844` / native source `e6b4ce362` runs on actual T4s:
107 original stage comparisons and 20 speech/interface roundtrips pass; all
20 speech arrays match the earlier GPU proof byte-for-byte. Independently
verified complete arrays, magnitude bounds, own-process CUDA allocation,
speaker controls and unchanged 0/8 canonical-code diagnostic are archived in
`docs/breeze-cuda-integration-package-2026-10-12.json`. Worker ERROR reflects
only that unchanged exact-code assertion. No new fixture, default or gate.
Integration remains draft pending its remaining general CI; #531 is now merged. Broader #530 model-quality reports remain separately unresolved.

PR #544 maintainer head `cfff1abb0` preserves all three contributor commits,
fixes CLI formatting, adds shared-library CANN compilation, and passes ten
production-source routing controls (previous helper fails four). The dedicated
manual `cann-build.yml` avoids duplicating general platform jobs. A targeted
x86_64/910B rerun of `38252679792` uses byte-identical production files/job
steps from source `76017342e`; all four targets and ggml-cann compiled successfully. The image digest and full log hash are recorded in `docs/cann-pr544-compile-2026-10-12.json`. A targeted ARM64/310P rerun is queued; normal PR CI remains pending. No physical Ascend/model acceptance is claimed.


## UPDATE 2026-10-12 — Combined-source CI and isolated CANN matrix

Breeze #543 is refreshed against merged main `dd05b5e26` at
`3042d533a4b6f4f03c172c6328fdcdd0431d2553`. Only documentation changes;
all four Breeze runtime files remain byte-identical to the completed T4 proof.
Seven source-compiled session-mode controls, 227 documentation anchors and all
four staged secret checks pass without skips (73.634 s). The completed GPU
receipt is linked from the refreshed integration receipt; general CI is queued.

CANN #544 is refreshed at `4149be6261b599a1a4e64fdc1d1e13e14454d801`,
including the merged TTS fixes, complete ONNX catalogue and unit CI. Production
CLI/routing source is unchanged. Ten routing scenarios, four ownership/rate and
three speaker controls pass, plus anchors/index and all four secret checks
without skips (69.502 s). The isolated current-source compile run is
`38266299769`: ARM64/310P job `114869220977`, x86_64/910B job
`114869221253`, both queued. Validation commit `03e605b64447228a193cb72a9565a54ee60baf70`
changes only the registered `ci.yml` to the identical two-job manual CANN
workflow. This avoids dispatching the general platform matrix while GitHub
requires a newly named workflow to exist on main first. Its staged four secret
checks pass without skips (51.200 s). The older-source run was archived and its
still-queued ARM job cancelled; completed x86 compilation remains retained.
Receipt: `docs/cann-pr544-compile-2026-10-12.json`.

GitHub's incident `93njd414rpbx` affected Actions queuing from 12:21 UTC; GitHub
reported mitigation at 14:36 UTC. Our refreshed runs still show queued rather
than failed. No refreshed merge, Ascend inference/performance claim or new tag.


## CLAIMED 2026-10-12 — Dia short-text silence and trustworthy reference capture

Investigate #530's exact F16/DAC prompt, "We are committed to making it secure."
Work in a storage worktree. Audit the pinned official old-checkpoint inference
flow and both CLI/C ABI generation. The existing Dia reference backend installs
unversioned upstream and reconstructs speaker-tag input outside generate();
verify those assumptions against the actual blueprint before trusting its dumps.
Add a pinned, complete short-prompt diagnostic with native/reference code and
PCM captures, independent ASR and cosine/magnitude checks where aligned.
Keep existing full-speech gates and sampling defaults unchanged. Patch runtime
only after the earliest divergence or a source-compiled failing control identifies
a defect. Heavy models/builds run on GH CI or actual Kaggle GPU; large local
artifacts stay on /mnt/storage.

### 2026-10-12 portable Dia sampler candidate ready for hosted validation

Main claim `24b1453c9` precedes implementation. The documented MCG31 stream plus
scalar F32 exponential race reproduces all 11,034 canonical original draws,
all 1,226 decoder transitions and the final 1,210-frame grid in the actual
original feedback loop. Both original-probability and actual native-filter
arms agree exactly. Noise cosine is 0.9999999999999962 and relative norm error
3.2631726210041734e-9; final F32 bits are not byte-identical to MKL vector log.
The fixed row-split negative differs on 9,090 draws. Full weight-free replay
artifacts and exact executed worker/header sources are retained under storage.

`CRISPASR_DIA_PORTABLE_RNG=1` now gates the candidate at the shared native
synthesis loop: one draw of all head/vocabulary noise per step, including
masked tokens. Existing default sampling, greedy behavior, model pins and
neural graphs remain unchanged. The diagnostic original-model bridge uses
that same header and filter. Source-only runtime compilation, clang18 format,
Python syntax and six tiny original/native contracts pass (14.834 s).
The candidate-specific contract is additionally checked against compiled
unmodified 75fcdc70f sampler source as a rejecting negative control.

Next exact-source hosted checks: `dia_portable_random_control.py` repeats the
complete weight-free original/native-filter witness; then
`dia_portable_speech_control.py` requires real original-neural/DAC canonical
codes and original/native C ABI/CLI ASR roundtrips with the validated +1,312-byte
critical-output DAC export. Full-model work runs on GitHub CI, not this VPS.
Applicable contributing audit: shared library implementation reaches existing
CLI/C-ABI/server/bindings through their synthesis calls; no new backend,
architecture, CMake library, model registry, converter schema or quantizer rule
is introduced. No speech, GPU, Q4 or complete model acceptance is claimed, and
this experimental runtime branch remains unmerged pending real output proof.
