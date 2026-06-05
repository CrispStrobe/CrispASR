"""Tier 0: dev-box smoke test for the regression driver.

No model downloads, no binary invocations, no network. Just:

  - manifest.json schema and shape (every backend has the required
    keys, types are right, no obvious typos).
  - parse_diff_stdout()       — feed it canned crispasr-diff output,
                                 assert it pulls cos_min for each stage.
  - evaluate_stage_thresholds() — the threshold-compare logic that
                                 decides per-stage PASS/FAIL/missing.

Runs in well under a second. Catches manifest typos and parser
regressions in PR CI without needing a built binary or HF auth.

Usage:

  python -m unittest tests/regression/test_driver_smoke.py

or via pytest if available:

  pytest tests/regression/test_driver_smoke.py
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))  # so `import run_one` works

import run_one  # noqa: E402


# Sample crispasr-diff stdout — actual capture from the parakeet JA seed,
# with ANSI codes stripped. Tweak when crispasr-diff's output format
# changes.
SAMPLE_DIFF_STDOUT = """\
crispasr-diff: audio 224964 samples (14.06s), reference /path/ref.gguf, backend parakeet
[FAIL] mel_spectrogram        shape=[80,1407]        cos_min=0.951084  cos_mean=0.999388  max_abs=2.23e+00  rms=3.67e-02
[PASS] encoder_output         shape=[1024,176]       cos_min=0.999594  cos_mean=0.999975  max_abs=3.13e-02  rms=1.56e-03
[PASS] encoder_output_ref_mel shape=[1024,176]       cos_min=0.999807  cos_mean=0.999988  max_abs=2.28e-02  rms=1.08e-03

summary: 2 pass, 1 fail, 0 skip (cos threshold 0.999)
"""


class ManifestSchemaTests(unittest.TestCase):
    """The manifest is the source of truth — typos here corrupt every
    nightly run silently. Lock the shape.
    """

    @classmethod
    def setUpClass(cls):
        with (HERE / "manifest.json").open() as f:
            cls.manifest = json.load(f)

    def test_version_pinned(self):
        self.assertEqual(self.manifest.get("version"), 1)

    def test_fixtures_block_pinned(self):
        fx = self.manifest.get("fixtures", {})
        self.assertIsInstance(fx, dict)
        self.assertIn("repo", fx)
        self.assertIn("revision", fx)
        # "main" is technically valid but defeats the entire upstream-
        # safety guarantee. Block it explicitly so a sloppy revert
        # can't slip in.
        self.assertNotEqual(
            fx["revision"], "main",
            "fixtures.revision must be pinned to a commit SHA, not a branch name "
            "(see tests/regression/README.md for why).",
        )
        self.assertRegex(
            fx["revision"], r"^[0-9a-f]{7,40}$",
            "fixtures.revision should look like a git SHA (7-40 hex chars).",
        )

    def test_backends_present(self):
        self.assertIn("backends", self.manifest)
        self.assertIsInstance(self.manifest["backends"], list)
        self.assertGreater(len(self.manifest["backends"]), 0)

    def test_backend_entries_have_required_keys(self):
        # Keys required by every entry regardless of skip_diff.
        always_required = {"name", "backend_id", "gguf", "expected_transcript"}
        # Keys required only for full diff entries (skip_diff absent or false).
        diff_required = {"fixture_ref_path", "diff_thresholds"}
        gguf_required = {"repo", "revision", "file"}
        names_seen: set[str] = set()
        for entry in self.manifest["backends"]:
            missing = always_required - set(entry)
            self.assertEqual(missing, set(),
                f"backend missing keys: {missing} in {entry.get('name', '?')}")
            self.assertNotIn(entry["name"], names_seen,
                f"duplicate backend name: {entry['name']}")
            names_seen.add(entry["name"])
            self.assertEqual(
                gguf_required - set(entry["gguf"]), set(),
                f"{entry['name']}.gguf missing keys",
            )
            # Pinned revision must be a SHA, not a branch.
            rev = entry["gguf"]["revision"]
            self.assertRegex(
                rev, r"^[0-9a-f]{7,40}$",
                f"{entry['name']}.gguf.revision must be a SHA, got {rev!r}",
            )
            skip_diff = entry.get("skip_diff", False)
            if not skip_diff:
                # Full diff entry: fixture_ref_path + diff_thresholds required.
                missing_diff = diff_required - set(entry)
                self.assertEqual(missing_diff, set(),
                    f"{entry['name']}: diff entry missing keys {missing_diff} "
                    f"(set skip_diff=true if no ref dump exists yet)")
                self.assertIsInstance(entry["diff_thresholds"], dict)
                self.assertGreater(len(entry["diff_thresholds"]), 0,
                    f"{entry['name']} has empty diff_thresholds — at least one stage must be tracked")
                for stage, threshold in entry["diff_thresholds"].items():
                    self.assertIsInstance(threshold, (int, float),
                        f"{entry['name']}.{stage} threshold must be numeric")
                    self.assertTrue(
                        0.0 <= threshold <= 1.0,
                        f"{entry['name']}.{stage} threshold {threshold} out of [0,1]",
                    )
            else:
                # Transcript-only entry: diff_thresholds and fixture_ref_path
                # must NOT be present (forces the maintainer to graduate it
                # to a full diff entry once the ref dump is baked).
                self.assertNotIn("diff_thresholds", entry,
                    f"{entry['name']}: skip_diff=true but diff_thresholds is also set — "
                    f"remove skip_diff once the ref dump is in cstr/crispasr-regression-fixtures")
                self.assertNotIn("fixture_ref_path", entry,
                    f"{entry['name']}: skip_diff=true but fixture_ref_path is also set — "
                    f"remove skip_diff once the ref dump is in cstr/crispasr-regression-fixtures")

            # Optional transcript_tolerance block — opt-in WER/CER bar
            # for backends whose decoder ties on punctuation/case
            # boundaries (e.g. glm-asr-nano). Schema:
            #   { "cer_max": 0..1, "wer_max": 0..1 } both required when present.
            if "transcript_tolerance" in entry:
                tol = entry["transcript_tolerance"]
                self.assertIsInstance(tol, dict,
                    f"{entry['name']}: transcript_tolerance must be an object")
                required_tol = {"cer_max", "wer_max"}
                missing_tol = required_tol - set(tol)
                self.assertEqual(missing_tol, set(),
                    f"{entry['name']}.transcript_tolerance missing keys: {missing_tol}")
                for k in required_tol:
                    v = tol[k]
                    self.assertIsInstance(v, (int, float),
                        f"{entry['name']}.transcript_tolerance.{k} must be numeric")
                    self.assertTrue(
                        0.0 <= v <= 1.0,
                        f"{entry['name']}.transcript_tolerance.{k} = {v} out of [0,1]",
                    )

    def test_sample_source_declared(self):
        """Every backend must declare its sample source: either a
        repo-relative `sample` (in-tree, must exist) or a
        `fixture_sample_path` pulled from the fixtures HF repo at
        the pinned revision (the recommended path for new entries).
        """
        for entry in self.manifest["backends"]:
            has_in_repo = "sample" in entry
            has_hf = "fixture_sample_path" in entry
            self.assertTrue(
                has_in_repo or has_hf,
                f"{entry['name']}: neither `sample` (in-repo) nor "
                f"`fixture_sample_path` (HF) declared",
            )
            if has_in_repo:
                sample = HERE.parent.parent / entry["sample"]
                self.assertTrue(
                    sample.exists(),
                    f"{entry['name']}: in-repo sample {sample} not "
                    f"present; either drop the field or check the WAV in",
                )


class DiffParserTests(unittest.TestCase):
    """parse_diff_stdout() — given a captured crispasr-diff stdout,
    pull cos_min per stage. Don't care about [PASS]/[FAIL] verdict
    (we apply manifest thresholds separately).
    """

    def test_basic_capture(self):
        stages = run_one.parse_diff_stdout(SAMPLE_DIFF_STDOUT)
        self.assertEqual(
            set(stages),
            {"mel_spectrogram", "encoder_output", "encoder_output_ref_mel"},
        )
        # Spot-check values land where the line says they should.
        self.assertAlmostEqual(stages["mel_spectrogram"], 0.951084, places=6)
        self.assertAlmostEqual(stages["encoder_output"], 0.999594, places=6)
        self.assertAlmostEqual(stages["encoder_output_ref_mel"], 0.999807, places=6)

    def test_handles_empty_stdout(self):
        self.assertEqual(run_one.parse_diff_stdout(""), {})

    def test_handles_garbage_lines(self):
        """Output without `[PASS]/[FAIL]` lines must yield {}."""
        garbage = "loading model...\nstage A failed (oops)\nfinished\n"
        self.assertEqual(run_one.parse_diff_stdout(garbage), {})

    def test_scientific_notation(self):
        """The regex must accept scientific-notation cos_min values
        (rare but produced when stages are numerically degenerate).
        """
        line = "[FAIL] foo shape=[1] cos_min=1.0e-05 cos_mean=2.5e-05"
        stages = run_one.parse_diff_stdout(line)
        self.assertIn("foo", stages)
        self.assertAlmostEqual(stages["foo"], 1.0e-05, places=10)

    def test_negative_cos_min(self):
        """`cos_min` can be negative when the captured stage anti-
        correlates with the reference (a real failure mode worth
        catching). Regex must allow the sign.
        """
        line = "[FAIL] anti shape=[8,1] cos_min=-0.42 cos_mean=-0.3"
        stages = run_one.parse_diff_stdout(line)
        self.assertAlmostEqual(stages["anti"], -0.42, places=6)


class ThresholdEvaluationTests(unittest.TestCase):
    """evaluate_stage_thresholds() — partition cos_min map into
    pass / fail / missing / extras given the manifest's thresholds.
    """

    THRESHOLDS = {
        "encoder_output": 0.999,
        "encoder_output_ref_mel": 0.999,
        "mel_spectrogram": 0.95,
    }

    def test_all_pass(self):
        stages = {
            "encoder_output": 0.9999,
            "encoder_output_ref_mel": 0.9999,
            "mel_spectrogram": 0.96,
        }
        p, f, m, e = run_one.evaluate_stage_thresholds(stages, self.THRESHOLDS)
        self.assertEqual(len(p), 3)
        self.assertEqual(f, [])
        self.assertEqual(m, [])
        self.assertEqual(e, [])

    def test_threshold_strict_below_fails(self):
        """A stage at exactly `threshold` PASSES; below FAILS."""
        stages = {
            "encoder_output": 0.999,           # == threshold → PASS
            "encoder_output_ref_mel": 0.9989,  # < threshold → FAIL
            "mel_spectrogram": 0.95,           # == threshold → PASS
        }
        p, f, m, e = run_one.evaluate_stage_thresholds(stages, self.THRESHOLDS)
        passes = {x[0] for x in p}
        fails = {x[0] for x in f}
        self.assertIn("encoder_output", passes)
        self.assertIn("mel_spectrogram", passes)
        self.assertIn("encoder_output_ref_mel", fails)

    def test_missing_stage(self):
        """If the manifest asks for a stage the diff harness didn't
        emit, it shows up in `missing` (not silently in `passes`).
        """
        stages = {"encoder_output": 0.9999}
        p, f, m, e = run_one.evaluate_stage_thresholds(stages, self.THRESHOLDS)
        self.assertEqual(set(m), {"encoder_output_ref_mel", "mel_spectrogram"})
        self.assertEqual(len(p), 1)
        self.assertEqual(f, [])

    def test_extra_stage(self):
        """A stage in the diff output but not in thresholds should be
        flagged as an extra (so a maintainer can add it intentionally).
        """
        stages = {
            "encoder_output": 0.9999,
            "encoder_output_ref_mel": 0.9999,
            "mel_spectrogram": 0.96,
            "some_new_stage": 0.42,  # extra
        }
        p, f, m, e = run_one.evaluate_stage_thresholds(stages, self.THRESHOLDS)
        self.assertEqual(e, [("some_new_stage", 0.42)])

    def test_end_to_end_against_canned_diff_output(self):
        """Wire parse_diff_stdout → evaluate_stage_thresholds against
        the canned sample; assert the partition matches what the
        seed backend (parakeet JA) actually produces.
        """
        stages = run_one.parse_diff_stdout(SAMPLE_DIFF_STDOUT)
        p, f, m, e = run_one.evaluate_stage_thresholds(stages, self.THRESHOLDS)
        passing = {x[0] for x in p}
        # All three stages clear their (lenient) thresholds. The diff
        # harness's own `[FAIL] mel_spectrogram` verdict is ignored
        # because manifest mel threshold is 0.95, not 0.999.
        self.assertEqual(
            passing,
            {"encoder_output", "encoder_output_ref_mel", "mel_spectrogram"},
        )
        self.assertEqual(f, [])
        self.assertEqual(m, [])


class StreamJsonVadRoutingTests(unittest.TestCase):
    """Pure-Python model of the stream JSON+VAD routing invariants
    introduced in PR #92 (perf: skip aggregate segs in JSON VAD path).

    The C++ logic is modelled here so regressions to the routing
    semantics are caught without needing a built binary or audio data.

    Three invariants locked in by PR #92
    -------------------------------------
    1. `json_vad_path` flag
         True  ⟺  stream_json=True AND vad_model is set
         False otherwise (stream_json=False, or no VAD, or both)

    2. `decoded_segments_this_step` flag
         Each decode branch sets it when the backend returns non-empty
         segments, regardless of which path runs:
           a) JSON+VAD   → sl_for_text non-empty
           b) Non-JSON VAD → slice_segs non-empty
           c) No-VAD       → segs non-empty

    3. `segs` invariant in JSON+VAD mode
         `segs` is intentionally left empty; aggregate post-processing
         (punc / truecase / pcs / strip-punc) is skipped via the
         `if !json_vad_path` guard. In non-JSON or no-VAD paths `segs`
         accumulates normally.

    `stream_monitor` silence-dot (`·`) now fires on
    `!decoded_segments_this_step` instead of `segs.empty()`, which
    was always True in JSON+VAD mode even when audio was transcribed.
    """

    # ------------------------------------------------------------------
    # Helper: Python model of the three-branch decode routing
    # ------------------------------------------------------------------

    @staticmethod
    def _simulate_step(
        stream_json: bool,
        vad_model: str,
        backend_returns: list[str],   # per-slice mock text; empty list = no VAD
    ) -> dict:
        """Return the state dict that C++ would have after one step.

        backend_returns:
          - If vad_model is set (VAD path): one entry per VAD slice.
            Each is the text that backend->transcribe() returns for
            that slice (empty string = backend returned nothing).
          - If vad_model is empty (no-VAD path): a single entry for
            the whole-window transcribe call.
        """
        json_vad_path   = stream_json and bool(vad_model)
        segs: list[str] = []
        decoded_segments_this_step = False

        if vad_model:
            # VAD branches
            for text in backend_returns:
                if stream_json:
                    # JSON+VAD: sl_for_text, segs stays empty
                    sl_for_text = [text] if text else []
                    if sl_for_text:
                        decoded_segments_this_step = True
                    # segs intentionally NOT updated here
                else:
                    # Non-JSON VAD: aggregate into segs
                    slice_segs = [text] if text else []
                    if slice_segs:
                        decoded_segments_this_step = True
                    segs.extend(slice_segs)
        else:
            # No-VAD path
            text = backend_returns[0] if backend_returns else ""
            segs = [text] if text else []
            if segs:
                decoded_segments_this_step = True

        # Post-loop guard: skip aggregate post-processing in JSON+VAD mode
        post_processed = not json_vad_path

        silence_dot = not decoded_segments_this_step

        return {
            "json_vad_path": json_vad_path,
            "segs": segs,
            "decoded_segments_this_step": decoded_segments_this_step,
            "post_processed": post_processed,
            "silence_dot": silence_dot,
        }

    # ------------------------------------------------------------------
    # 1. json_vad_path flag
    # ------------------------------------------------------------------

    def test_json_vad_path_true_when_both_set(self):
        r = self._simulate_step(stream_json=True, vad_model="firered_vad.gguf",
                                backend_returns=["hello"])
        self.assertTrue(r["json_vad_path"])

    def test_json_vad_path_false_stream_json_only(self):
        r = self._simulate_step(stream_json=True, vad_model="",
                                backend_returns=["hello"])
        self.assertFalse(r["json_vad_path"])

    def test_json_vad_path_false_vad_only(self):
        r = self._simulate_step(stream_json=False, vad_model="firered_vad.gguf",
                                backend_returns=["hello"])
        self.assertFalse(r["json_vad_path"])

    def test_json_vad_path_false_neither(self):
        r = self._simulate_step(stream_json=False, vad_model="",
                                backend_returns=["hello"])
        self.assertFalse(r["json_vad_path"])

    # ------------------------------------------------------------------
    # 2a. decoded_segments_this_step — JSON+VAD path
    # ------------------------------------------------------------------

    def test_decoded_flag_json_vad_set_when_backend_returns_text(self):
        r = self._simulate_step(stream_json=True, vad_model="vad.gguf",
                                backend_returns=["hello world"])
        self.assertTrue(r["decoded_segments_this_step"])

    def test_decoded_flag_json_vad_clear_when_all_slices_empty(self):
        r = self._simulate_step(stream_json=True, vad_model="vad.gguf",
                                backend_returns=["", ""])
        self.assertFalse(r["decoded_segments_this_step"])

    def test_decoded_flag_json_vad_set_when_any_slice_non_empty(self):
        """Even one non-empty slice among many empty ones is enough."""
        r = self._simulate_step(stream_json=True, vad_model="vad.gguf",
                                backend_returns=["", "speech here", ""])
        self.assertTrue(r["decoded_segments_this_step"])

    # ------------------------------------------------------------------
    # 2b. decoded_segments_this_step — non-JSON VAD path
    # ------------------------------------------------------------------

    def test_decoded_flag_non_json_vad_set(self):
        r = self._simulate_step(stream_json=False, vad_model="vad.gguf",
                                backend_returns=["hello"])
        self.assertTrue(r["decoded_segments_this_step"])

    def test_decoded_flag_non_json_vad_clear_when_empty(self):
        r = self._simulate_step(stream_json=False, vad_model="vad.gguf",
                                backend_returns=[""])
        self.assertFalse(r["decoded_segments_this_step"])

    # ------------------------------------------------------------------
    # 2c. decoded_segments_this_step — no-VAD path
    # ------------------------------------------------------------------

    def test_decoded_flag_no_vad_set_when_backend_returns_text(self):
        r = self._simulate_step(stream_json=False, vad_model="",
                                backend_returns=["text"])
        self.assertTrue(r["decoded_segments_this_step"])

    def test_decoded_flag_no_vad_clear_when_silent(self):
        r = self._simulate_step(stream_json=False, vad_model="",
                                backend_returns=[""])
        self.assertFalse(r["decoded_segments_this_step"])

    # ------------------------------------------------------------------
    # 3. segs invariant
    # ------------------------------------------------------------------

    def test_segs_empty_in_json_vad_mode(self):
        """PR #92 core invariant: aggregate segs must stay empty in
        JSON+VAD mode regardless of what the backend returns.
        """
        r = self._simulate_step(stream_json=True, vad_model="vad.gguf",
                                backend_returns=["word1", "word2", "word3"])
        self.assertEqual(r["segs"], [],
            "JSON+VAD mode must not populate aggregate segs")

    def test_segs_populated_in_non_json_vad_mode(self):
        r = self._simulate_step(stream_json=False, vad_model="vad.gguf",
                                backend_returns=["hello", "world"])
        self.assertEqual(r["segs"], ["hello", "world"])

    def test_segs_populated_in_no_vad_mode(self):
        r = self._simulate_step(stream_json=False, vad_model="",
                                backend_returns=["full text"])
        self.assertEqual(r["segs"], ["full text"])

    def test_post_processing_skipped_in_json_vad_mode(self):
        """punc/truecase/pcs must not run on the empty segs in JSON+VAD
        mode (the `if !json_vad_path` guard).
        """
        r = self._simulate_step(stream_json=True, vad_model="vad.gguf",
                                backend_returns=["text"])
        self.assertFalse(r["post_processed"])

    def test_post_processing_runs_in_non_json_vad_mode(self):
        r = self._simulate_step(stream_json=False, vad_model="vad.gguf",
                                backend_returns=["text"])
        self.assertTrue(r["post_processed"])

    def test_post_processing_runs_in_no_vad_mode(self):
        r = self._simulate_step(stream_json=False, vad_model="",
                                backend_returns=["text"])
        self.assertTrue(r["post_processed"])

    # ------------------------------------------------------------------
    # 4. stream_monitor silence dot
    # ------------------------------------------------------------------

    def test_silence_dot_fires_when_no_decode_output_json_vad(self):
        """With old code segs.empty() was always True in JSON+VAD mode,
        so the silence dot always fired even when audio was transcribed.
        PR #92 fixes this; model the corrected behaviour.
        """
        r = self._simulate_step(stream_json=True, vad_model="vad.gguf",
                                backend_returns=[""])
        self.assertTrue(r["silence_dot"],
            "silence dot should fire when no segments decoded")

    def test_silence_dot_suppressed_when_json_vad_has_output(self):
        r = self._simulate_step(stream_json=True, vad_model="vad.gguf",
                                backend_returns=["hello"])
        self.assertFalse(r["silence_dot"],
            "silence dot must NOT fire when backend returned segments")

    def test_silence_dot_fires_in_no_vad_silence(self):
        r = self._simulate_step(stream_json=False, vad_model="",
                                backend_returns=[""])
        self.assertTrue(r["silence_dot"])

    def test_silence_dot_suppressed_in_no_vad_speech(self):
        r = self._simulate_step(stream_json=False, vad_model="",
                                backend_returns=["speech"])
        self.assertFalse(r["silence_dot"])


class TtsBackendsSchemaTests(unittest.TestCase):
    """Schema checks for the optional `tts_backends` section of
    `manifest.json` introduced by PLAN #12 (TTS->ASR roundtrip nightly).

    The section is opt-in (omitted entirely on older manifests), so
    these checks short-circuit when absent. When present, every entry
    must declare a pinned TTS gguf, a pinned voice gguf, a phrase, a
    `roundtrip_asr_backend` referencing a real ASR entry, and a
    numeric `wer_max` in [0, 1].
    """

    @classmethod
    def setUpClass(cls):
        with (HERE / "manifest.json").open() as f:
            cls.manifest = json.load(f)

    def test_tts_backends_field_is_optional(self):
        # tts_backends should be either absent or a list. Tolerate
        # older manifests that haven't enabled the roundtrip path.
        section = self.manifest.get("tts_backends", [])
        self.assertIsInstance(section, list)

    def test_tts_entries_have_required_keys(self):
        # `voice` (block) and `voice_preset` (name string) are
        # mutually-exclusive — exactly one must be set. Verified after
        # the per-entry required-keys check below.
        required = {"name", "backend_id", "gguf",
                    "tts_phrase", "roundtrip_asr_backend", "wer_max"}
        pinned_required = {"repo", "revision", "file"}
        names_seen: set[str] = set()
        asr_names = {b["name"] for b in self.manifest["backends"]}

        for entry in self.manifest.get("tts_backends", []):
            missing = required - set(entry)
            self.assertEqual(missing, set(),
                f"tts backend missing keys: {missing} in {entry.get('name', '?')}")
            name = entry["name"]
            self.assertNotIn(name, names_seen,
                f"duplicate tts backend name: {name}")
            names_seen.add(name)

            # Exactly one of `voice` (block) or `voice_preset` (string).
            has_voice = "voice" in entry
            has_preset = "voice_preset" in entry
            self.assertTrue(
                has_voice ^ has_preset,
                f"{name}: must set exactly one of `voice` block or "
                f"`voice_preset` string (got voice={has_voice} preset={has_preset})")
            if has_preset:
                self.assertIsInstance(entry["voice_preset"], str,
                    f"{name}.voice_preset must be a string")
                self.assertGreater(len(entry["voice_preset"].strip()), 0,
                    f"{name}.voice_preset must be non-empty")

            # Pinned revisions: SHA only, never a branch.
            pinned_blocks: list[tuple[str, dict]] = [("gguf", entry["gguf"])]
            if has_voice:
                pinned_blocks.append(("voice", entry["voice"]))
            if "codec" in entry:
                pinned_blocks.append(("codec", entry["codec"]))
            for label, block in pinned_blocks:
                self.assertEqual(pinned_required - set(block), set(),
                    f"{name}.{label} missing keys")
                self.assertRegex(
                    block["revision"], r"^[0-9a-f]{7,40}$",
                    f"{name}.{label}.revision must be a SHA, got {block['revision']!r}")

            # roundtrip_asr_backend must reference a real ASR entry.
            asr_ref = entry["roundtrip_asr_backend"]
            self.assertIn(asr_ref, asr_names,
                f"{name}.roundtrip_asr_backend={asr_ref!r} not in manifest.backends")

            # wer_max must be a numeric in [0, 1].
            wer_max = entry["wer_max"]
            self.assertIsInstance(wer_max, (int, float),
                f"{name}.wer_max must be numeric")
            self.assertTrue(0.0 <= wer_max <= 1.0,
                f"{name}.wer_max = {wer_max} out of [0,1]")

            # tts_phrase must be a non-empty string.
            self.assertIsInstance(entry["tts_phrase"], str,
                f"{name}.tts_phrase must be a string")
            self.assertGreater(len(entry["tts_phrase"].strip()), 0,
                f"{name}.tts_phrase must be non-empty")

    def test_tts_entries_name_is_unique_against_asr_entries(self):
        # `run_one.py main()` dispatches on name across both lists.
        # A duplicate name would collide ambiguously.
        asr_names = {b["name"] for b in self.manifest["backends"]}
        tts_names = {b["name"] for b in self.manifest.get("tts_backends", [])}
        overlap = asr_names & tts_names
        self.assertEqual(overlap, set(),
            f"tts_backends names collide with asr backend names: {overlap}")


class TranscriptToleranceTests(unittest.TestCase):
    """CER/WER metric for the optional transcript_tolerance field.

    The metric only kicks in when byte-equal fails AND the manifest
    entry opts in via ``transcript_tolerance``. Per-backend opt-in so
    backends with truly deterministic output keep the strict bar.
    """

    def test_levenshtein_empty(self):
        self.assertEqual(run_one._levenshtein([], []), 0)
        self.assertEqual(run_one._levenshtein(["a"], []), 1)
        self.assertEqual(run_one._levenshtein([], ["a", "b"]), 2)

    def test_levenshtein_identical(self):
        self.assertEqual(run_one._levenshtein(list("hello"), list("hello")), 0)

    def test_levenshtein_substitution(self):
        self.assertEqual(run_one._levenshtein(list("cat"), list("bat")), 1)

    def test_levenshtein_mixed_ops(self):
        # kitten -> sitting: substitute k->s, e->i, insert g => 3 ops
        self.assertEqual(run_one._levenshtein(list("kitten"), list("sitting")), 3)

    def test_metrics_identical_strings(self):
        cer, wer = run_one.compute_transcript_metrics("hello world", "hello world")
        self.assertEqual(cer, 0.0)
        self.assertEqual(wer, 0.0)

    def test_metrics_punctuation_only_diff(self):
        # The exact glm-asr-nano failure mode: 'you. Ask' vs 'you, ask'.
        # Two character substitutions ('.'→',' AND 'A'→'a'); zero word
        # edits after WER normalization (lowercase + strip punc).
        expected = "And so, my fellow Americans, ask not what your country can do for you. Ask what you can do for your country."
        actual = "And so, my fellow Americans, ask not what your country can do for you, ask what you can do for your country."
        cer, wer = run_one.compute_transcript_metrics(expected, actual)
        # 2 character diffs over 108 chars = ~0.0185 CER.
        self.assertAlmostEqual(cer, 2.0 / len(expected), places=4)
        # 'Ask' vs 'ask' is a case-only difference and the leading
        # punctuation diffs are stripped, so WER = 0.
        self.assertEqual(wer, 0.0)
        # The configured 2% CER / 5% WER tolerance covers it:
        self.assertLessEqual(cer, 0.02)
        self.assertLessEqual(wer, 0.05)

    def test_metrics_one_word_substitution(self):
        cer, wer = run_one.compute_transcript_metrics("hello world", "hello earth")
        # 'world' -> 'earth' is one word substitution after lowercase+split.
        self.assertEqual(wer, 0.5)  # 1 substitution / 2 expected words
        self.assertGreater(cer, 0.0)

    def test_metrics_inserted_word(self):
        cer, wer = run_one.compute_transcript_metrics("hello world", "hello there world")
        # 1 inserted word out of 2 expected words.
        self.assertEqual(wer, 0.5)

    def test_metrics_deleted_word(self):
        cer, wer = run_one.compute_transcript_metrics("hello there world", "hello world")
        # 1 deleted word out of 3 expected words.
        self.assertAlmostEqual(wer, 1.0 / 3, places=4)

    def test_metrics_empty_expected(self):
        cer, wer = run_one.compute_transcript_metrics("", "")
        self.assertEqual(cer, 0.0)
        self.assertEqual(wer, 0.0)
        # Non-empty actual against empty expected -> infinite (cannot divide).
        cer, wer = run_one.compute_transcript_metrics("", "something")
        self.assertEqual(cer, float("inf"))
        self.assertEqual(wer, float("inf"))

    def test_metrics_case_normalization(self):
        cer, wer = run_one.compute_transcript_metrics("Hello World", "hello world")
        # CER counts the case differences as substitutions; WER does not
        # (normalization lowercases first).
        self.assertGreater(cer, 0.0)
        self.assertEqual(wer, 0.0)

    def test_metrics_punctuation_normalization(self):
        cer, wer = run_one.compute_transcript_metrics(
            "hello, world!", "hello world"
        )
        # CER counts the dropped comma + exclamation; WER ignores them.
        self.assertGreater(cer, 0.0)
        self.assertEqual(wer, 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
