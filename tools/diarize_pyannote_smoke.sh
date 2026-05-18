#!/usr/bin/env bash
# Reproducible live smoke test for --diarize-method pyannote (issue #107).
#
# Builds a synthetic 2-speaker WAV by concatenating samples/jfk.wav (real
# JFK speech, mono 16 kHz) with a TTS-generated reading of the same text
# at a different sample rate (auto-resampled). Then runs the C++
# test-diarize-pyannote-live target against that fixture and the
# auto-cached pyannote-seg-3.0.gguf model.
#
# Why this fixture: two genuinely different voices reading similar
# content gives the segmentation net unambiguous turn boundaries; it
# exercises the within-pass scoring logic without needing access to
# gated multi-speaker corpora (AMI/VoxConverse/CallHome).
#
# NOTE: this validates the IN-PROCESS C++ API (single full-audio forward
# pass → multiple speaker IDs). The CLI's per-slice diarize flow has a
# separate cross-slice stitching limitation that is NOT exercised by
# this script — see #107 for the remaining Phase 2 work.

set -euo pipefail

repo_root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$repo_root"

build_dir="${BUILD_DIR:-build-ninja-compile}"
fixture_dir="${FIXTURE_DIR:-/Volumes/backups/ai/crispasr-test-fixtures}"
model_dir="${MODEL_DIR:-/Volumes/backups/ai/crispasr-models/pyannote-seg-3.0}"

wav="${fixture_dir}/two-speakers-jfk-tts.wav"
model="${model_dir}/pyannote-seg-3.0.gguf"

mkdir -p "$fixture_dir" "$model_dir"

# 1) Synthesize the 2-speaker fixture if missing.
if [[ ! -f "$wav" ]]; then
    spk_b="${SPEAKER_B_WAV:-$HOME/.cache/crispasr/cb_baker_jfk.wav}"
    if [[ ! -f "$spk_b" ]]; then
        echo "ERROR: speaker-B sample not found at $spk_b" >&2
        echo "       set SPEAKER_B_WAV to any single-speaker mono WAV" >&2
        exit 1
    fi
    echo "[smoke] synthesizing 2-speaker fixture → $wav"
    python - <<PY
import wave, struct
def load(path):
    with wave.open(path,'rb') as w:
        return w.getframerate(), [int.from_bytes(w.readframes(w.getnframes())[i:i+2],'little',signed=True)
                                  for i in range(0, w.getnframes()*2, 2)]
def resample(s, src, dst):
    if src == dst: return s
    r = dst/src; n = int(len(s)*r); out = []
    for i in range(n):
        x = i/r; i0 = int(x); i1 = min(i0+1, len(s)-1); a = x - i0
        out.append(int(s[i0]*(1-a) + s[i1]*a))
    return out
sa, a = load('samples/jfk.wav')
sb, b = load('$spk_b')
b16 = resample(b, sb, 16000)
silence = [0]*(16000//2)
mix = a + silence + b16 + silence + a + silence + b16
with wave.open('$wav','wb') as w:
    w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000)
    w.writeframes(b''.join(struct.pack('<h', max(-32768,min(32767,s))) for s in mix))
print(f'wrote {len(mix)/16000:.1f}s 2-speaker WAV')
PY
fi

# 2) Fetch the pyannote-seg GGUF if missing.
if [[ ! -f "$model" ]]; then
    echo "[smoke] downloading pyannote-seg-3.0.gguf → $model"
    curl -fsSL -o "$model" \
        "https://huggingface.co/cstr/pyannote-v3-segmentation-GGUF/resolve/main/pyannote-seg-3.0.gguf"
fi

# 3) Build the live test target.
echo "[smoke] building test-diarize-pyannote-live"
cmake --build "$build_dir" --target test-diarize-pyannote-live >/dev/null

# 4) Run it. The test asserts ≥2 distinct speaker labels surface.
echo "[smoke] running"
CRISPASR_TEST_DIARIZE_WAV="$wav" \
CRISPASR_TEST_DIARIZE_MODEL="$model" \
    "$build_dir/bin/test-diarize-pyannote-live" --success
