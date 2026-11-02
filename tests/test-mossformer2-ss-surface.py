#!/usr/bin/env python3
"""Actual CLI + Python Session/C ABI contract for the MossFormer2_SS_16K port.

Set CRISPASR_MOSSFORMER2_SS_MODEL to a converted model. Missing model input is
a CTest skip (77); a supplied but unusable model is a failure. The model is
renamed before opening so the C ABI must use general.architecture.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import wave

ROOT = Path(__file__).resolve().parents[1]
STEMS = ["spk1", "spk2"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--crispasr", required=True, type=Path)
    parser.add_argument("--library", required=True, type=Path)
    parser.add_argument("--model", default=os.environ.get("CRISPASR_MOSSFORMER2_SS_MODEL", ""))
    parser.add_argument("--frames", type=int, default=20000, help="16-kHz input samples (> 32000 = segmented path)")
    args = parser.parse_args()
    if not args.model or not Path(args.model).is_file():
        print("SKIP: set CRISPASR_MOSSFORMER2_SS_MODEL to an existing converted GGUF for surface validation")
        return 77
    model = Path(args.model).resolve(strict=True)
    binary = args.crispasr.resolve(strict=True)
    library = args.library.resolve(strict=True)

    import numpy as np
    sys.path.insert(0, str(ROOT / "python"))
    from crispasr import Session, detect_backend_from_gguf

    roster = subprocess.run([str(binary), "--list-backends-json"], check=True, capture_output=True, text=True)
    backends = {row["name"]: row for row in json.loads(roster.stdout)["backends"]}
    assert "separate" in backends["mossformer2-ss"]["caps"]
    assert "auto-download" not in backends["mossformer2-ss"]["caps"]
    assert "mossformer2-ss" in Session.available_backends(lib_path=str(library))

    with tempfile.TemporaryDirectory(prefix="mf2ss-surface-") as tmp:
        work = Path(tmp)
        neutral = work / "weights.gguf"
        try:
            neutral.symlink_to(model)
        except OSError:
            shutil.copyfile(model, neutral)
        assert detect_backend_from_gguf(str(neutral), lib_path=str(library)) == "mossformer2-ss"

        frames = args.frames
        t = np.arange(frames) / 16000
        mono = 0.2 * np.sin(2 * np.pi * 140 * t) + 0.1 * np.sin(2 * np.pi * 280 * t)
        mono += np.where(np.fmod(t, 0.5) < 0.25, 0.15 * np.sin(2 * np.pi * (900 + 300 * t) * t), 0.0)
        pcm16 = np.round(mono * 32767).astype("<i2")
        audio = pcm16.astype(np.float32) / 32768
        wav = work / "mixture.wav"
        with wave.open(str(wav), "wb") as out:
            out.setnchannels(1)
            out.setsampwidth(2)
            out.setframerate(16000)
            out.writeframes(pcm16.tobytes())
        stereo = np.ascontiguousarray(np.stack([audio, audio], axis=1).reshape(-1))

        with Session(str(neutral), lib_path=str(library), n_threads=4, use_gpu=False) as session:
            assert session.backend == "mossformer2-ss"
            stems = session.separate(stereo)
            assert list(stems) == STEMS, list(stems)
            native = session._lib
            assert native.crispasr_session_separate_n_stems(session._handle) == 2
            assert native.crispasr_session_separate_sample_rate(session._handle) == 16000
            assert native.crispasr_session_separate_stem_name(session._handle, 2) is None
            for name, pcm in stems.items():
                assert pcm.shape == (frames * 2,), (name, pcm.shape)
                assert np.isfinite(pcm).all(), name
                # Mono speakers are duplicated into both interleaved channels.
                np.testing.assert_array_equal(pcm[0::2], pcm[1::2])
            snapshots = {name: value.copy() for name, value in stems.items()}
            repeated = session.separate(stereo)
            for name in STEMS:
                np.testing.assert_array_equal(stems[name], snapshots[name])
                np.testing.assert_array_equal(repeated[name], snapshots[name])

        output = work / "stems"
        output.mkdir()
        subprocess.run([str(binary), "--separate", "--no-gpu", "-t", "4", "-m", str(neutral), "-f", str(wav),
                        "--sep-output-dir", str(output)], check=True, timeout=900)
        assert sorted(p.name for p in output.iterdir()) == ["mixture_spk1.wav", "mixture_spk2.wav"]
        for name in STEMS:
            with wave.open(str(output / f"mixture_{name}.wav"), "rb") as result:
                assert (result.getnchannels(), result.getsampwidth(), result.getframerate(), result.getnframes()) == (
                    1, 2, 16000, frames)
                decoded = np.frombuffer(result.readframes(frames), dtype="<i2").astype(np.float32) / 32768
            np.testing.assert_allclose(decoded, np.clip(stems[name][0::2], -1, 1), rtol=0, atol=1e-4)
    print("PASS: CLI spk1/spk2 WAVs, renamed-model Session auto-detection, mono-duplicated stereo stems, repeat")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
