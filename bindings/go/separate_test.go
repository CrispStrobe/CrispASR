package whisper_test

// Source separation (#359).
//
// Reported as "there is no clear way to utilize models like htdemucs from
// libraries". The C ABI has always had a five-function separation surface, and
// Python and Dart bound it — Go and Rust did not. So the only separation-shaped
// verb a Go caller could find was SpeechToSpeech, which is not what these
// models do and which fails with "no audio produced".
//
// Gated on SEPARATE_MODEL, a path to a separation GGUF (mel-band-roformer,
// mini-bs-roformer or htdemucs). Without it these skip on a model-free checkout.

import (
	"math"
	"os"
	"testing"

	whisper "github.com/CrispStrobe/CrispASR/bindings/go"
	assert "github.com/stretchr/testify/assert"
)

const separateModelEnv = "SEPARATE_MODEL"

func openSeparateSession(t *testing.T) *whisper.CrispasrSession {
	t.Helper()
	path := os.Getenv(separateModelEnv)
	if path == "" {
		t.Skip("Skipping test, " + separateModelEnv + " not set")
	}
	if _, err := os.Stat(path); err != nil {
		t.Skip("Skipping test, separation model not found: " + path)
	}
	s, err := whisper.SessionOpen(path, 0)
	if err != nil {
		t.Fatalf("NewSession(%s): %v", path, err)
	}
	return s
}

func TestSeparateReturnsNamedStems(t *testing.T) {
	assert := assert.New(t)
	s := openSeparateSession(t)
	defer s.Close()

	// The rate accessor is what a caller needs BEFORE they can feed anything
	// in, and its absence is why the reporter saw 0.
	sr := s.SeparateSampleRate()
	assert.Greater(sr, 0, "SeparateSampleRate must be known once loaded")

	// 2 s of interleaved stereo at the model's own rate — a quiet tone rather
	// than silence, so a passthrough backend is still exercised.
	nFrames := sr * 2
	pcm := make([]float32, 0, nFrames*2)
	for i := 0; i < nFrames; i++ {
		v := float32(math.Sin(float64(i)/float64(sr)*220.0*2*math.Pi) * 0.2)
		pcm = append(pcm, v, v)
	}

	stems, err := s.Separate(pcm)
	assert.NoError(err)
	assert.NotEmpty(stems, "expected at least one stem")
	for _, st := range stems {
		assert.NotEmpty(st.Name, "every stem is named")
		assert.Equal(0, len(st.PCM)%2, "stem %s must be interleaved stereo", st.Name)
		assert.NotEmpty(st.PCM, "stem %s came back empty", st.Name)
		for _, v := range st.PCM {
			if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
				t.Fatalf("stem %s has a non-finite sample", st.Name)
			}
		}
	}
	t.Logf("separate: %d stems at %d Hz", len(stems), sr)
}

func TestSeparateRejectsNonStereoInput(t *testing.T) {
	assert := assert.New(t)
	s := openSeparateSession(t)
	defer s.Close()

	// The C API counts PER-CHANNEL frames; fewer than one frame is a caller
	// error rather than something to hand the model.
	_, err := s.Separate(nil)
	assert.Error(err)
}
