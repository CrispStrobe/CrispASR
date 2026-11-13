//go:build probe

package whisper

import "testing"

func TestProbeVADSegmentsSeconds(t *testing.T) {
	pcm := []float32{1, 3, 5}
	frees := probeVADSegmentsFrees()
	spans, err := VADSegments("vad", pcm, 16000, .375, 123, 456, 2)
	if err != nil {
		t.Fatal(err)
	}
	want := []VADSpan{{T0: 1.255, T1: 2.5025}, {T0: 3.0075, T1: 9.0375}}
	if len(spans) != len(want) {
		t.Fatalf("got %d spans, want %d", len(spans), len(want))
	}
	for i := range want {
		if spans[i] != want[i] {
			t.Fatalf("span %d = %+v, want seconds %+v", i, spans[i], want[i])
		}
	}
	if probeVADSegmentsLive() != 0 || probeVADSegmentsFrees() != frees+1 {
		t.Fatal("native spans were not freed exactly once")
	}
	for _, model := range []string{"empty", "error"} {
		got, err := VADSegments(model, pcm, 16000, .375, 123, 456, 2)
		if (err != nil) != (model == "error") || len(got) != 0 {
			t.Fatalf("%s: got %v, %v", model, got, err)
		}
	}
	if probeVADSegmentsLive() != 0 || probeVADSegmentsFrees() != frees+1 || spans[0] != want[0] {
		t.Fatal("empty/error result cleanup or copied spans changed")
	}
}
