#!/usr/bin/env python3
"""Actual CLI + Python Session/C ABI contract for the Mini BS RoFormer port.

Set CRISPASR_MINI_BS_ROFORMER_MODEL to a converted model. Missing model input
is a CTest skip (77); a supplied but unusable model is a failure. The model is
renamed before opening so the C ABI must use general.architecture.
"""
from __future__ import annotations

import argparse
import ctypes
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import wave

ROOT = Path(__file__).resolve().parents[1]
STEMS = ["bass", "drums", "other", "vocals"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--crispasr", required=True, type=Path)
    parser.add_argument("--library", required=True, type=Path)
    parser.add_argument("--model", default=os.environ.get("CRISPASR_MINI_BS_ROFORMER_MODEL", ""))
    parser.add_argument("--frames", type=int, default=12347,
                        help="input frames per channel; use 352800 for a complete trained eight-second chunk")
    args = parser.parse_args()
    if args.frames <= 0:
        parser.error("--frames must be positive")
    if not args.model:
        print("SKIP: set CRISPASR_MINI_BS_ROFORMER_MODEL for actual-model surface validation")
        return 77
    model = Path(args.model).resolve(strict=True)
    binary = args.crispasr.resolve(strict=True)
    library = args.library.resolve(strict=True)

    import numpy as np
    sys.path.insert(0, str(ROOT / "python"))
    from crispasr import Session, detect_backend_from_gguf

    roster = subprocess.run([str(binary), "--list-backends-json"], check=True,
                            capture_output=True, text=True)
    backends = {row["name"]: row for row in json.loads(roster.stdout)["backends"]}
    assert "separate" in backends["mini-bs-roformer"]["caps"]
    assert "auto-download" in backends["mini-bs-roformer"]["caps"]
    assert "mini-bs-roformer" in Session.available_backends(lib_path=str(library))

    with tempfile.TemporaryDirectory(prefix="mini-bsr-surface-") as tmp:
        work = Path(tmp)
        neutral = work / "weights.gguf"
        try:
            neutral.symlink_to(model)
        except OSError:
            shutil.copyfile(model, neutral)
        assert detect_backend_from_gguf(str(neutral), lib_path=str(library)) == "mini-bs-roformer"

        # Deliberately asymmetric stereo and a non-hop-aligned length catch
        # channel swaps, truncation, and accidental mono-ASR routing.
        frames = args.frames
        time = np.arange(frames, dtype=np.float32) / 44100
        pcm16 = np.stack([0.18 * np.sin(2 * np.pi * 220 * time),
                          0.11 * np.sin(2 * np.pi * 659 * time)], axis=1)
        pcm16 = np.round(pcm16 * 32767).astype("<i2")
        audio = pcm16.astype(np.float32) / 32768
        wav = work / "fixture.wav"
        with wave.open(str(wav), "wb") as out:
            out.setnchannels(2)
            out.setsampwidth(2)
            out.setframerate(44100)
            out.writeframes(pcm16.tobytes())

        with Session(str(neutral), lib_path=str(library), n_threads=2, use_gpu=False) as session:
            assert session.backend == "mini-bs-roformer"
            stems = session.separate(audio)
            assert list(stems) == STEMS, list(stems)
            for name, pcm in stems.items():
                assert pcm.shape == (frames * 2,), (name, pcm.shape)
                assert np.isfinite(pcm).all(), name
            native = session._lib
            assert native.crispasr_session_separate_n_stems(session._handle) == 4
            assert native.crispasr_session_separate_sample_rate(session._handle) == 44100
            assert native.crispasr_session_separate_stem_name(session._handle, -1) is None
            assert native.crispasr_session_separate_stem_name(session._handle, 4) is None
            count = ctypes.c_int(999)
            assert not native.crispasr_session_separate_stem(session._handle, 4, ctypes.byref(count))
            assert count.value == 0
            # A repeated separation replaces the native result. Existing Python
            # arrays must remain owned copies, and repeated output must agree.
            snapshots = {name: value.copy() for name, value in stems.items()}
            repeated = session.separate(audio)
            for name in STEMS:
                np.testing.assert_array_equal(stems[name], snapshots[name])
                np.testing.assert_allclose(repeated[name], snapshots[name], rtol=1e-5, atol=1e-6)

            # Keep the graph geometry identical while replacing the PCM, then
            # revisit the first input. A graph that retains freed RoPE/input
            # storage or overwrites an input across stem decoders fails this
            # cold/warm comparison even if its first forward is correct.
            changed_audio = np.ascontiguousarray(audio[::-1, ::-1] * np.float32(0.7))
            changed = session.separate(changed_audio)
            returned = session.separate(audio)
            for name in STEMS:
                assert np.isfinite(changed[name]).all(), name
                np.testing.assert_allclose(returned[name], snapshots[name], rtol=1e-5, atol=1e-6)
                np.testing.assert_array_equal(stems[name], snapshots[name])

        with Session(str(neutral), lib_path=str(library), n_threads=2, use_gpu=False) as fresh:
            changed_cold = fresh.separate(changed_audio)
            for name in STEMS:
                assert np.isfinite(changed_cold[name]).all(), name
                np.testing.assert_allclose(changed[name], changed_cold[name], rtol=1e-5, atol=1e-6)
                assert changed[name].flags.owndata, name

        output = work / "stems"
        output.mkdir()
        subprocess.run([str(binary), "--separate", "--no-gpu", "-t", "2", "-m", str(neutral),
                        "-f", str(wav), "--stems", "bass,vocals", "--sep-output-dir", str(output)],
                       check=True, timeout=600)
        assert sorted(p.name for p in output.iterdir()) == ["fixture_bass.wav", "fixture_vocals.wav"]
        for name in ("bass", "vocals"):
            with wave.open(str(output / f"fixture_{name}.wav"), "rb") as result:
                assert (result.getnchannels(), result.getsampwidth(), result.getframerate(), result.getnframes()) == (
                    2, 2, 44100, frames)
                decoded = np.frombuffer(result.readframes(frames), dtype="<i2").astype(np.float32) / 32768
            # int16 serialization rounds/clamps; the two surfaces otherwise run
            # the same model and input waveform.
            np.testing.assert_allclose(decoded, np.clip(stems[name], -1, 1), rtol=0, atol=1e-4)
    print("PASS: CLI selection/WAV geometry, renamed-model Session auto-detection, four stems, "
          "cold/warm input reuse, repeat ownership")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
