package whisper

/*
#include <stdlib.h>
#include "crispasr_session.h"
*/
import "C"

import (
	"errors"
	"math"
	"strings"
	"unsafe"
)

// UltraAlignedWord holds a refined word interval in seconds. Floating-point
// times retain the local head's precision beyond the generic AlignedWord's
// integer centiseconds.
type UltraAlignedWord struct {
	Text  string
	Start float64
	End   float64
}

// AlignWordsUltra runs experimental Qwen Ultra local-head forced alignment.
// Both converted base and local-head GGUF paths are required. Audio must be
// mono 16 kHz PCM, nonempty and at most 120 seconds long. A native failure
// returns an error; a successful empty alignment returns an empty slice.
// The local head runs on CPU. This does not qualify additional languages or GPUs.
func AlignWordsUltra(alignerModel, headModel, transcript string, pcm []float32, nThreads int) ([]UltraAlignedWord, error) {
	for _, text := range []string{alignerModel, headModel, transcript} {
		if text == "" || strings.IndexByte(text, 0) >= 0 {
			return nil, errors.New("alignment paths and transcript must be nonempty and contain no NUL")
		}
	}
	if len(pcm) == 0 || len(pcm) > 120*16000 {
		return nil, errors.New("Ultra alignment requires between 1 and 1920000 mono 16 kHz samples")
	}
	if int64(nThreads) < math.MinInt32 || int64(nThreads) > math.MaxInt32 {
		return nil, errors.New("alignment thread count exceeds int32 range")
	}
	base := C.CString(alignerModel)
	defer C.free(unsafe.Pointer(base))
	head := C.CString(headModel)
	defer C.free(unsafe.Pointer(head))
	text := C.CString(transcript)
	defer C.free(unsafe.Pointer(text))
	result := C.crispasr_align_words_ultra_abi(base, head, text, (*C.float)(unsafe.Pointer(&pcm[0])), C.int32_t(len(pcm)), C.int32_t(nThreads))
	if result == nil {
		return nil, errors.New("Ultra alignment failed")
	}
	defer C.crispasr_ultra_align_result_free(result)
	count := int(C.crispasr_ultra_align_result_n_words(result))
	if count < 0 {
		return nil, errors.New("invalid native alignment word count")
	}
	words := make([]UltraAlignedWord, count)
	for i := range words {
		words[i] = UltraAlignedWord{
			Text:  C.GoString(C.crispasr_ultra_align_result_word_text(result, C.int(i))),
			Start: float64(C.crispasr_ultra_align_result_word_start_ms(result, C.int(i))) / 1000,
			End:   float64(C.crispasr_ultra_align_result_word_end_ms(result, C.int(i))) / 1000,
		}
	}
	return words, nil
}
