package whisper_test

import (
	"os"
	"testing"
)

const (
	ModelPath  = "../../models/ggml-small.en.bin"
	SamplePath = "../../samples/jfk.wav"
)

func requireModel(t *testing.T) {
	t.Helper()
	if _, err := os.Stat(ModelPath); os.IsNotExist(err) {
		t.Skip("Skipping test, model not found:", ModelPath)
	}
}

func requireSample(t *testing.T) {
	t.Helper()
	if _, err := os.Stat(SamplePath); os.IsNotExist(err) {
		t.Skip("Skipping test, sample not found:", SamplePath)
	}
}
