//go:build probe

package whisper

import (
	"math"
	"testing"
)

func TestProbeUltraAlignmentPrecisionAndOwnership(t *testing.T) {
	probeReset()
	words, err := AlignWordsUltra("base.gguf", "head.gguf", "你好", []float32{1, 3, 5}, 7)
	if err != nil {
		t.Fatal(err)
	}
	if len(words) != 1 || words[0].Text != "你好" || math.Abs(words[0].Start-0.012345) > 1e-12 || math.Abs(words[0].End-0.067891) > 1e-12 {
		t.Fatalf("lost refined precision or native text: %+v", words)
	}
	if probeCallsGet() != 1 || probeResultFreesGet() != 1 {
		t.Fatal("native call/result ownership")
	}
}

func TestProbeUltraAlignmentFailureAndEmpty(t *testing.T) {
	for _, mode := range []int{1, 2, 3} {
		probeReset()
		probeModeSet(mode)
		words, err := AlignWordsUltra("base.gguf", "head.gguf", "你好", []float32{1, 3, 5}, 7)
		if mode == 1 {
			if err != nil || len(words) != 0 {
				t.Fatalf("empty result: %v %v", words, err)
			}
		} else if err == nil {
			t.Fatalf("accepted failure mode %d", mode)
		}
		wantFrees := 1
		if mode == 2 {
			wantFrees = 0
		}
		if probeCallsGet() != 1 || probeResultFreesGet() != wantFrees {
			t.Fatalf("mode %d leaked or freed null", mode)
		}
	}
	probeReset()
}

func TestProbeUltraAlignmentInvalidInput(t *testing.T) {
	cases := []struct {
		base, head, text string
		pcm              []float32
		threads          int
	}{
		{"", "head.gguf", "你好", []float32{1}, 7},
		{"base.gguf", "", "你好", []float32{1}, 7},
		{"base.gguf", "head.gguf", "", []float32{1}, 7},
		{"base.gguf", "head.gguf", "你好", nil, 7},
		{"base.gguf", "head.gguf", "你好", make([]float32, 120*16000+1), 7},
		{"base\x00.gguf", "head.gguf", "你好", []float32{1}, 7},
		{"base.gguf", "head\x00.gguf", "你好", []float32{1}, 7},
		{"base.gguf", "head.gguf", "你\x00好", []float32{1}, 7},
	}
	for _, tc := range cases {
		probeReset()
		_, err := AlignWordsUltra(tc.base, tc.head, tc.text, tc.pcm, tc.threads)
		if err == nil || probeCallsGet() != 0 {
			t.Errorf("invalid input reached native code: %q %q %q", tc.base, tc.head, tc.text)
		}
	}
	// The C count is signed 32-bit even when Go int is 64-bit.
	if ^uint(0)>>63 != 0 {
		probeReset()
		largeThreads := int64(1)<<32 + 7
		_, err := AlignWordsUltra("base.gguf", "head.gguf", "a", []float32{1}, int(largeThreads))
		if err == nil || probeCallsGet() != 0 {
			t.Error("thread count truncated at FFI")
		}
	}
}
