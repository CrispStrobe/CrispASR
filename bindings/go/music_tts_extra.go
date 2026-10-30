package whisper

/*
#include <stdlib.h>
#include <string.h>
typedef struct crispasr_session CrispasrSession;
#include <crispasr_session.h>
#include <crispasr.h>

extern void goPcmStreamCallback(float* pcm, int n_samples, int is_final, uintptr_t user);

static void pcmStreamTrampoline(const float* pcm,int n,int final,void* user){goPcmStreamCallback((float*)pcm,n,final,(uintptr_t)user);}
static int callSynthesizeStreaming(struct crispasr_session* s, const char* text, uintptr_t user) {
    return crispasr_session_synthesize_streaming(s, text, pcmStreamTrampoline, (void*)user);
}
*/
import "C"
import (
	"errors"
	"fmt"
	"math"
	"runtime/cgo"
	"strings"
	"unsafe"
)

// --- Piano ---

type PianoNote struct {
	OnsetMs  float32
	OffsetMs float32
	MidiNote int
	Velocity int
}

type PianoResult struct {
	Notes      []PianoNote
	Programs   []int
	SampleRate int
}

func (s *CrispasrSession) Piano(pcm []float32) (*PianoResult, error) {
	s.nativeHandle()
	if s.handle == nil {
		return nil, errors.New("session is closed")
	}
	if len(pcm) == 0 {
		return nil, errors.New("Piano: empty PCM")
	}
	n := C.crispasr_session_piano((*C.struct_crispasr_session)(unsafe.Pointer(s.nativeHandle())), (*C.float)(unsafe.Pointer(&pcm[0])), C.int(len(pcm)))
	if n < 0 {
		return nil, fmt.Errorf("Piano failed (rc=%d)", int(n))
	}
	if n == 0 {
		return &PianoResult{Notes: nil, Programs: nil, SampleRate: int(C.crispasr_session_piano_sample_rate((*C.struct_crispasr_session)(unsafe.Pointer(s.nativeHandle()))))}, nil
	}
	var outN C.int
	ptr := C.crispasr_session_piano_notes((*C.struct_crispasr_session)(unsafe.Pointer(s.nativeHandle())), &outN)
	count := int(outN)
	if count < 0 {
		return nil, fmt.Errorf("Piano: negative count %d", count)
	}
	if ptr == nil || count == 0 {
		if ptr == nil && count != 0 {
			return nil, errors.New("Piano: notes pointer is nil")
		}
		return &PianoResult{Notes: nil, Programs: nil, SampleRate: int(C.crispasr_session_piano_sample_rate((*C.struct_crispasr_session)(unsafe.Pointer(s.nativeHandle()))))}, nil
	}
	if count > (1<<30)/4 {
		return nil, fmt.Errorf("Piano: count overflow %d", count)
	}
	flat := unsafe.Slice((*float32)(unsafe.Pointer(ptr)), count*4)
	notes := make([]PianoNote, count)
	for i := 0; i < count; i++ {
		notes[i] = PianoNote{
			OnsetMs:  flat[i*4+0],
			OffsetMs: flat[i*4+1],
			MidiNote: int(flat[i*4+2]),
			Velocity: int(flat[i*4+3]),
		}
	}
	var outN2 C.int
	ptr2 := C.crispasr_session_piano_note_programs((*C.struct_crispasr_session)(unsafe.Pointer(s.nativeHandle())), &outN2)
	if ptr2 == nil {
		return nil, errors.New("Piano: programs pointer is nil")
	}
	if int(outN2) != count {
		return nil, fmt.Errorf("Piano notes/programs length mismatch %d vs %d", count, int(outN2))
	}
	progsFlat := unsafe.Slice((*int32)(unsafe.Pointer(ptr2)), count)
	progs := make([]int, count)
	for i, v := range progsFlat {
		progs[i] = int(v)
	}
	return &PianoResult{Notes: notes, Programs: progs, SampleRate: int(C.crispasr_session_piano_sample_rate((*C.struct_crispasr_session)(unsafe.Pointer(s.nativeHandle()))))}, nil
}

func (s *CrispasrSession) PianoSampleRate() int {
	s.nativeHandle()
	if s.handle == nil {
		return 0
	}
	return int(C.crispasr_session_piano_sample_rate((*C.struct_crispasr_session)(unsafe.Pointer(s.nativeHandle()))))
}

func (s *CrispasrSession) PianoNNotes() int {
	s.nativeHandle()
	if s.handle == nil {
		return 0
	}
	return int(C.crispasr_session_piano_n_notes((*C.struct_crispasr_session)(unsafe.Pointer(s.nativeHandle()))))
}

// --- Chords ---

type ChordSpan struct {
	StartMs    float32
	EndMs      float32
	Label      int
	Confidence float32
	Name       string
}

type ChordResult struct {
	Spans     []ChordSpan
	VocabSize int
}

func (s *CrispasrSession) Chords(pcm []float32, sampleRate int) (*ChordResult, error) {
	s.nativeHandle()
	if s.handle == nil {
		return nil, errors.New("session is closed")
	}
	if len(pcm) == 0 {
		return nil, errors.New("Chords: empty PCM")
	}
	if sampleRate <= 0 {
		return nil, errors.New("Chords: sampleRate must be positive")
	}
	n := C.crispasr_session_chords((*C.struct_crispasr_session)(unsafe.Pointer(s.nativeHandle())), (*C.float)(unsafe.Pointer(&pcm[0])), C.int(len(pcm)), C.int(sampleRate))
	if n < 0 {
		return nil, fmt.Errorf("Chords failed (rc=%d)", int(n))
	}
	if n == 0 {
		return &ChordResult{Spans: nil, VocabSize: int(C.crispasr_session_chords_vocab_size((*C.struct_crispasr_session)(unsafe.Pointer(s.nativeHandle()))))}, nil
	}
	var outN C.int
	ptr := C.crispasr_session_chords_spans((*C.struct_crispasr_session)(unsafe.Pointer(s.nativeHandle())), &outN)
	count := int(outN)
	if count < 0 {
		return nil, fmt.Errorf("Chords: negative count %d", count)
	}
	if ptr == nil || count == 0 {
		if ptr == nil && count != 0 {
			return nil, errors.New("Chords: spans pointer is nil")
		}
		return &ChordResult{Spans: nil, VocabSize: int(C.crispasr_session_chords_vocab_size((*C.struct_crispasr_session)(unsafe.Pointer(s.nativeHandle()))))}, nil
	}
	if count > (1<<30)/4 {
		return nil, fmt.Errorf("Chords: count overflow %d", count)
	}
	flat := unsafe.Slice((*float32)(unsafe.Pointer(ptr)), count*4)
	spans := make([]ChordSpan, count)
	for i := 0; i < count; i++ {
		namePtr := C.crispasr_session_chords_span_name((*C.struct_crispasr_session)(unsafe.Pointer(s.nativeHandle())), C.int(i))
		name := ""
		if namePtr != nil {
			name = C.GoString(namePtr)
		}
		spans[i] = ChordSpan{
			StartMs:    flat[i*4+0],
			EndMs:      flat[i*4+1],
			Label:      int(flat[i*4+2]),
			Confidence: flat[i*4+3],
			Name:       name,
		}
	}
	return &ChordResult{Spans: spans, VocabSize: int(C.crispasr_session_chords_vocab_size((*C.struct_crispasr_session)(unsafe.Pointer(s.nativeHandle()))))}, nil
}

func (s *CrispasrSession) ChordsVocabSize() int {
	s.nativeHandle()
	if s.handle == nil {
		return 0
	}
	return int(C.crispasr_session_chords_vocab_size((*C.struct_crispasr_session)(unsafe.Pointer(s.nativeHandle()))))
}

// --- Beats ---

type BeatEvent struct {
	TimeS      float32
	IsDownbeat bool
}

type BeatResult struct {
	Events     []BeatEvent
	TempoBPM   float32
	SampleRate int
}

func (s *CrispasrSession) Beats(pcm []float32, sampleRate int) (*BeatResult, error) {
	s.nativeHandle()
	if s.handle == nil {
		return nil, errors.New("session is closed")
	}
	if len(pcm) == 0 {
		return nil, errors.New("Beats: empty PCM")
	}
	if sampleRate <= 0 {
		return nil, errors.New("Beats: sampleRate must be positive")
	}
	n := C.crispasr_session_beats((*C.struct_crispasr_session)(unsafe.Pointer(s.nativeHandle())), (*C.float)(unsafe.Pointer(&pcm[0])), C.int(len(pcm)), C.int(sampleRate))
	if n < 0 {
		return nil, fmt.Errorf("Beats failed (rc=%d) — sample rate mismatch or no beat arm", int(n))
	}
	if n == 0 {
		return &BeatResult{Events: nil, TempoBPM: float32(C.crispasr_session_beats_tempo_bpm((*C.struct_crispasr_session)(unsafe.Pointer(s.nativeHandle())))), SampleRate: int(C.crispasr_session_beats_sample_rate((*C.struct_crispasr_session)(unsafe.Pointer(s.nativeHandle()))))}, nil
	}
	var outN C.int
	ptr := C.crispasr_session_beats_events((*C.struct_crispasr_session)(unsafe.Pointer(s.nativeHandle())), &outN)
	count := int(outN)
	if count < 0 {
		return nil, fmt.Errorf("Beats: negative count %d", count)
	}
	if ptr == nil {
		if count != 0 {
			return nil, errors.New("Beats: events pointer is nil")
		}
		return &BeatResult{Events: nil, TempoBPM: float32(C.crispasr_session_beats_tempo_bpm((*C.struct_crispasr_session)(unsafe.Pointer(s.nativeHandle())))), SampleRate: int(C.crispasr_session_beats_sample_rate((*C.struct_crispasr_session)(unsafe.Pointer(s.nativeHandle()))))}, nil
	}
	if count == 0 {
		return &BeatResult{Events: nil, TempoBPM: float32(C.crispasr_session_beats_tempo_bpm((*C.struct_crispasr_session)(unsafe.Pointer(s.nativeHandle())))), SampleRate: int(C.crispasr_session_beats_sample_rate((*C.struct_crispasr_session)(unsafe.Pointer(s.nativeHandle()))))}, nil
	}
	if count > (1<<30)/2 {
		return nil, fmt.Errorf("Beats: count overflow %d", count)
	}
	flat := unsafe.Slice((*float32)(unsafe.Pointer(ptr)), count*2)
	events := make([]BeatEvent, count)
	for i := 0; i < count; i++ {
		events[i] = BeatEvent{TimeS: flat[i*2+0], IsDownbeat: flat[i*2+1] != 0}
	}
	return &BeatResult{Events: events, TempoBPM: float32(C.crispasr_session_beats_tempo_bpm((*C.struct_crispasr_session)(unsafe.Pointer(s.nativeHandle())))), SampleRate: int(C.crispasr_session_beats_sample_rate((*C.struct_crispasr_session)(unsafe.Pointer(s.nativeHandle()))))}, nil
}

func (s *CrispasrSession) BeatsSampleRate() int {
	s.nativeHandle()
	if s.handle == nil {
		return 0
	}
	return int(C.crispasr_session_beats_sample_rate((*C.struct_crispasr_session)(unsafe.Pointer(s.nativeHandle()))))
}

func (s *CrispasrSession) BeatsTempoBPM() float32 {
	s.nativeHandle()
	if s.handle == nil {
		return 0
	}
	return float32(C.crispasr_session_beats_tempo_bpm((*C.struct_crispasr_session)(unsafe.Pointer(s.nativeHandle()))))
}

// --- Tab ---

type TabResult struct {
	Emissions   []float32 // flat [frames*strings*classes]
	Frames      int
	Strings     int
	Classes     int
	SilentClass int
	FramePeriod float32
	OpenMidi    []int
}

func (s *CrispasrSession) Tab(pcm []float32, sampleRate int) (*TabResult, error) {
	s.nativeHandle()
	if s.handle == nil {
		return nil, errors.New("session is closed")
	}
	if len(pcm) == 0 {
		return nil, errors.New("Tab: empty PCM")
	}
	if sampleRate <= 0 {
		return nil, errors.New("Tab: sampleRate must be positive")
	}
	n := C.crispasr_session_tab((*C.struct_crispasr_session)(unsafe.Pointer(s.nativeHandle())), (*C.float)(unsafe.Pointer(&pcm[0])), C.int(len(pcm)), C.int(sampleRate))
	if n < 0 {
		return nil, fmt.Errorf("Tab failed (rc=%d)", int(n))
	}
	if n == 0 {
		// successful empty result
		return &TabResult{Emissions: nil, Frames: 0, Strings: 0, Classes: 0, SilentClass: int(C.crispasr_session_tab_silent_class((*C.struct_crispasr_session)(unsafe.Pointer(s.nativeHandle())))), FramePeriod: float32(C.crispasr_session_tab_frame_period((*C.struct_crispasr_session)(unsafe.Pointer(s.nativeHandle())))), OpenMidi: nil}, nil
	}
	var outFrames, outStrings, outClasses C.int
	ptr := C.crispasr_session_tab_emissions((*C.struct_crispasr_session)(unsafe.Pointer(s.nativeHandle())), &outFrames, &outStrings, &outClasses)
	frames, strs, cls := int(outFrames), int(outStrings), int(outClasses)
	if frames < 0 || strs < 0 || cls < 0 {
		return nil, fmt.Errorf("Tab: negative dimensions %d/%d/%d", frames, strs, cls)
	}
	if ptr == nil {
		if frames == 0 || strs == 0 || cls == 0 {
			return &TabResult{Emissions: nil, Frames: frames, Strings: strs, Classes: cls, SilentClass: int(C.crispasr_session_tab_silent_class((*C.struct_crispasr_session)(unsafe.Pointer(s.nativeHandle())))), FramePeriod: float32(C.crispasr_session_tab_frame_period((*C.struct_crispasr_session)(unsafe.Pointer(s.nativeHandle())))), OpenMidi: nil}, nil
		}
		return nil, errors.New("Tab: emissions pointer is nil")
	}
	// Checked multiplication to avoid overflow on 32-bit
	if frames > 0 && strs > 0 && cls > 0 {
		if frames > (1<<30)/strs || frames*strs > (1<<30)/cls {
			return nil, fmt.Errorf("Tab: dimensions overflow %d*%d*%d", frames, strs, cls)
		}
	}
	total := frames * strs * cls
	if total == 0 {
		return &TabResult{Emissions: nil, Frames: frames, Strings: strs, Classes: cls, SilentClass: int(C.crispasr_session_tab_silent_class((*C.struct_crispasr_session)(unsafe.Pointer(s.nativeHandle())))), FramePeriod: float32(C.crispasr_session_tab_frame_period((*C.struct_crispasr_session)(unsafe.Pointer(s.nativeHandle())))), OpenMidi: nil}, nil
	}
	emissions := make([]float32, total)
	copy(emissions, unsafe.Slice((*float32)(unsafe.Pointer(ptr)), total))
	openMidi := make([]int, strs)
	for i := 0; i < strs; i++ {
		openMidi[i] = int(C.crispasr_session_tab_string_open_midi((*C.struct_crispasr_session)(unsafe.Pointer(s.nativeHandle())), C.int(i)))
	}
	return &TabResult{
		Emissions:   emissions,
		Frames:      frames,
		Strings:     strs,
		Classes:     cls,
		SilentClass: int(C.crispasr_session_tab_silent_class((*C.struct_crispasr_session)(unsafe.Pointer(s.nativeHandle())))),
		FramePeriod: float32(C.crispasr_session_tab_frame_period((*C.struct_crispasr_session)(unsafe.Pointer(s.nativeHandle())))),
		OpenMidi:    openMidi,
	}, nil
}

// --- Pitch ---

type PitchFrame struct {
	TimeMs     float32
	F0Hz       float32
	VoicedProb float32
}

func (s *CrispasrSession) Pitch(pcm []float32, hopMs float32) ([]PitchFrame, error) {
	s.nativeHandle()
	if s.handle == nil {
		return nil, errors.New("session is closed")
	}
	if len(pcm) == 0 {
		return nil, errors.New("Pitch: empty PCM")
	}
	n := C.crispasr_session_pitch((*C.struct_crispasr_session)(unsafe.Pointer(s.nativeHandle())), (*C.float)(unsafe.Pointer(&pcm[0])), C.int(len(pcm)), C.float(hopMs))
	if n < 0 {
		return nil, fmt.Errorf("Pitch failed (rc=%d)", int(n))
	}
	if n == 0 {
		return nil, nil
	}
	var outN C.int
	ptr := C.crispasr_session_pitch_frames((*C.struct_crispasr_session)(unsafe.Pointer(s.nativeHandle())), &outN)
	count := int(outN)
	if count < 0 {
		return nil, fmt.Errorf("Pitch: negative count %d", count)
	}
	if ptr == nil {
		if count == 0 {
			return nil, nil
		}
		return nil, errors.New("Pitch: frames pointer is nil")
	}
	if count == 0 {
		return nil, nil
	}
	if count > (1<<30)/3 {
		return nil, fmt.Errorf("Pitch: count overflow %d", count)
	}
	flat := unsafe.Slice((*float32)(unsafe.Pointer(ptr)), count*3)
	frames := make([]PitchFrame, count)
	for i := 0; i < count; i++ {
		frames[i] = PitchFrame{TimeMs: flat[i*3+0], F0Hz: flat[i*3+1], VoicedProb: flat[i*3+2]}
	}
	// copy before free/next call can replace
	cp := make([]PitchFrame, count)
	copy(cp, frames)
	return cp, nil
}

func (s *CrispasrSession) PitchSampleRate() int {
	s.nativeHandle()
	if s.handle == nil {
		return 0
	}
	return int(C.crispasr_session_pitch_sample_rate((*C.struct_crispasr_session)(unsafe.Pointer(s.nativeHandle()))))
}

// --- Convert (RVC) ---

func (s *CrispasrSession) ConvertContentDim() int {
	s.nativeHandle()
	if s.handle == nil {
		return 0
	}
	return int(C.crispasr_session_convert_content_dim((*C.struct_crispasr_session)(unsafe.Pointer(s.nativeHandle()))))
}

func (s *CrispasrSession) ConvertNSpeakers() int {
	s.nativeHandle()
	if s.handle == nil {
		return 0
	}
	return int(C.crispasr_session_convert_n_speakers((*C.struct_crispasr_session)(unsafe.Pointer(s.nativeHandle()))))
}

func (s *CrispasrSession) ConvertSampleRate() int {
	s.nativeHandle()
	if s.handle == nil {
		return 0
	}
	return int(C.crispasr_session_convert_sample_rate((*C.struct_crispasr_session)(unsafe.Pointer(s.nativeHandle()))))
}

func (s *CrispasrSession) ConvertInterChannels() int {
	s.nativeHandle()
	if s.handle == nil {
		return 0
	}
	return int(C.crispasr_session_convert_inter_channels((*C.struct_crispasr_session)(unsafe.Pointer(s.nativeHandle()))))
}

func (s *CrispasrSession) ConvertUpsampleProduct() int {
	s.nativeHandle()
	if s.handle == nil {
		return 0
	}
	return int(C.crispasr_session_convert_upsample_product((*C.struct_crispasr_session)(unsafe.Pointer(s.nativeHandle()))))
}

func (s *CrispasrSession) Convert(content []float32, nFrames int, f0Hz []float32, speakerID int, noiseZp []float32, noiseSine []float32) ([]float32, error) {
	s.nativeHandle()
	if s.handle == nil {
		return nil, errors.New("session is closed")
	}
	contentDim := s.ConvertContentDim()
	if contentDim <= 0 {
		return nil, errors.New("Convert: could not query content dim (is this a conversion model?)")
	}
	if nFrames <= 0 {
		return nil, errors.New("Convert: nFrames must be positive")
	}
	if len(content) != nFrames*contentDim {
		return nil, fmt.Errorf("Convert: content length %d != nFrames %d * contentDim %d", len(content), nFrames, contentDim)
	}
	if len(f0Hz) != nFrames {
		return nil, fmt.Errorf("Convert: f0 length %d != nFrames %d", len(f0Hz), nFrames)
	}
	nSpeakers := s.ConvertNSpeakers()
	if speakerID < 0 || speakerID >= nSpeakers {
		return nil, fmt.Errorf("Convert: speaker_id %d out of range [0,%d)", speakerID, nSpeakers)
	}
	interCh := s.ConvertInterChannels()
	upsample := s.ConvertUpsampleProduct()
	if interCh < 0 || upsample < 0 {
		return nil, fmt.Errorf("Convert: negative inter_channels %d or upsample_product %d", interCh, upsample)
	}
	if interCh > 0 && nFrames > (1<<30)/interCh {
		return nil, fmt.Errorf("Convert: inter_channels overflow %d*%d", interCh, nFrames)
	}
	if upsample > 0 && nFrames > (1<<30)/upsample {
		return nil, fmt.Errorf("Convert: upsample overflow %d*%d", nFrames, upsample)
	}
	if noiseZp != nil {
		expZp := interCh * nFrames
		if len(noiseZp) != expZp {
			return nil, fmt.Errorf("Convert: noise_zp length %d != inter_channels %d * nFrames %d", len(noiseZp), interCh, nFrames)
		}
	}
	if noiseSine != nil {
		expSine := nFrames * upsample
		if len(noiseSine) != expSine {
			return nil, fmt.Errorf("Convert: noise_sine length %d != nFrames %d * upsample_product %d", len(noiseSine), nFrames, upsample)
		}
	}
	// Validate via checked products (prevent overflow — though values are small)
	var zpPtr *C.float
	if noiseZp != nil {
		zpPtr = (*C.float)(unsafe.Pointer(&noiseZp[0]))
	}
	var sinePtr *C.float
	if noiseSine != nil {
		sinePtr = (*C.float)(unsafe.Pointer(&noiseSine[0]))
	}
	var contentPtr *C.float
	if len(content) > 0 {
		contentPtr = (*C.float)(unsafe.Pointer(&content[0]))
	}
	var f0Ptr *C.float
	if len(f0Hz) > 0 {
		f0Ptr = (*C.float)(unsafe.Pointer(&f0Hz[0]))
	}
	nOut := C.crispasr_session_convert((*C.struct_crispasr_session)(unsafe.Pointer(s.nativeHandle())), contentPtr, C.int(nFrames), f0Ptr, C.int(speakerID), zpPtr, sinePtr)
	if nOut <= 0 {
		return nil, fmt.Errorf("Convert failed (rc=%d)", int(nOut))
	}
	var outN C.int
	ptr := C.crispasr_session_convert_audio((*C.struct_crispasr_session)(unsafe.Pointer(s.nativeHandle())), &outN)
	if ptr == nil || int(outN) <= 0 {
		return nil, errors.New("Convert: audio pointer is nil")
	}
	count := int(outN)
	audio := make([]float32, count)
	copy(audio, unsafe.Slice((*float32)(unsafe.Pointer(ptr)), count))
	return audio, nil
}

// --- TTS extras ---

func (s *CrispasrSession) SetTTSCfgScale(scale float32) error {
	s.nativeHandle()
	if s.handle == nil {
		return errors.New("session is closed")
	}
	rc := C.crispasr_session_set_tts_cfg_scale((*C.struct_crispasr_session)(unsafe.Pointer(s.nativeHandle())), C.float(scale))
	if rc == -2 {
		return fmt.Errorf("SetTTSCfgScale not supported by backend (rc=%d)", int(rc))
	}
	if rc != 0 {
		return fmt.Errorf("SetTTSCfgScale failed (rc=%d)", int(rc))
	}
	return nil
}

func (s *CrispasrSession) TadaSetMakerefModels(encoder, aligner string) error {
	s.nativeHandle()
	if s.handle == nil {
		return errors.New("session is closed")
	}
	var cEnc, cAli *C.char
	if encoder != "" {
		cEnc = C.CString(encoder)
		defer C.free(unsafe.Pointer(cEnc))
	}
	if aligner != "" {
		cAli = C.CString(aligner)
		defer C.free(unsafe.Pointer(cAli))
	}
	rc := C.crispasr_session_tada_set_makeref_models((*C.struct_crispasr_session)(unsafe.Pointer(s.nativeHandle())), cEnc, cAli)
	if rc != 0 {
		return fmt.Errorf("TadaSetMakerefModels failed (rc=%d)", int(rc))
	}
	return nil
}

func DisclaimerText() string {
	ptr := C.crispasr_session_disclaimer_text()
	if ptr == nil {
		return ""
	}
	return C.GoString(ptr)
}

func (s *CrispasrSession) GetDisclaimerPCM() ([]float32, error) {
	s.nativeHandle()
	if s.handle == nil {
		return nil, errors.New("session is closed")
	}
	var n C.int
	ptr := C.crispasr_session_get_disclaimer_pcm((*C.struct_crispasr_session)(unsafe.Pointer(s.nativeHandle())), &n)
	if int(n) <= 0 {
		if ptr != nil {
			C.crispasr_pcm_free(ptr)
		}
		return nil, errors.New("GetDisclaimerPCM: no audio (call before SetVoice?)")
	}
	if ptr == nil {
		return nil, errors.New("GetDisclaimerPCM: no audio (call before SetVoice?)")
	}
	defer C.crispasr_pcm_free(ptr)
	count := int(n)
	audio := make([]float32, count)
	copy(audio, unsafe.Slice((*float32)(unsafe.Pointer(ptr)), count))
	return audio, nil
}

func (s *CrispasrSession) LastSynthError() string {
	s.nativeHandle()
	if s.handle == nil {
		return ""
	}
	ptr := C.crispasr_session_last_synth_error((*C.struct_crispasr_session)(unsafe.Pointer(s.nativeHandle())))
	if ptr == nil {
		return ""
	}
	return C.GoString(ptr)
}

func (s *CrispasrSession) SetVoiceSamples(pcm []float32, sampleRate int, refText string) error {
	s.nativeHandle()
	if s.handle == nil {
		return errors.New("session is closed")
	}
	if len(pcm) == 0 {
		return errors.New("SetVoiceSamples: empty PCM")
	}
	if sampleRate <= 0 {
		return errors.New("SetVoiceSamples: invalid sampleRate")
	}
	var cRef *C.char
	if refText != "" {
		cRef = C.CString(refText)
		defer C.free(unsafe.Pointer(cRef))
	}
	rc := C.crispasr_session_set_voice_samples((*C.struct_crispasr_session)(unsafe.Pointer(s.nativeHandle())), (*C.float)(unsafe.Pointer(&pcm[0])), C.int(len(pcm)), C.int(sampleRate), cRef)
	if rc != 0 {
		return fmt.Errorf("SetVoiceSamples failed (rc=%d)", int(rc))
	}
	return nil
}

// --- Streaming TTS ---

//export goPcmStreamCallback
func goPcmStreamCallback(pcm *C.float, nSamples C.int, isFinal C.int, userData C.uintptr_t) {
	if userData == 0 {
		return
	}
	h := cgo.Handle(userData)
	ctx := h.Value().(*streamCtx)
	if isFinal < 0 {
		return
	}
	if ctx.err != nil {
		return
	}
	if pcm == nil || int(nSamples) <= 0 {
		ctx.err = errors.New("SynthesizeStreaming: empty chunk from native")
		return
	}
	// Copy PCM before callback returns — owned by the call
	src := unsafe.Slice((*float32)(unsafe.Pointer(pcm)), int(nSamples))
	cp := make([]float32, len(src))
	copy(cp, src)
	// Call user callback; transport panic as error
	func() {
		defer func() {
			ctx.owner.callbackActive.Store(false)
			if r := recover(); r != nil {
				ctx.err = fmt.Errorf("SynthesizeStreaming callback panic: %v", r)
			}
		}()
		ctx.owner.callbackActive.Store(true)
		ctx.cb(cp, isFinal != 0)
	}()
}

type streamCtx struct {
	owner *CrispasrSession
	cb    func(pcm []float32, isFinal bool)
	err   error
}

func (s *CrispasrSession) SynthesizeStreaming(text string, cb func(pcm []float32, isFinal bool)) error {
	s.nativeHandle()
	if s.handle == nil {
		return errors.New("session is closed")
	}
	if cb == nil {
		return errors.New("SynthesizeStreaming: nil callback")
	}
	ctext := C.CString(text)
	defer C.free(unsafe.Pointer(ctext))
	ctx := &streamCtx{cb: cb, owner: s}
	h := cgo.NewHandle(ctx)
	defer h.Delete()
	rc := C.callSynthesizeStreaming((*C.struct_crispasr_session)(unsafe.Pointer(s.nativeHandle())), ctext, C.uintptr_t(h))
	if ctx.err != nil {
		return ctx.err
	}
	if rc != 0 {
		return fmt.Errorf("SynthesizeStreaming failed (rc=%d)", int(rc))
	}
	return nil
}

// --- Audio helpers ---

func AudioLoad(path string, targetRate int) ([]float32, int, error) {
	cpath := C.CString(path)
	defer C.free(unsafe.Pointer(cpath))
	var outPcm *C.float
	var outSamples, outRate C.int
	var rc C.int
	if targetRate > 0 {
		rc = C.crispasr_audio_load_at_rate(cpath, C.int(targetRate), &outPcm, &outSamples, &outRate)
	} else {
		rc = C.crispasr_audio_load(cpath, &outPcm, &outSamples, &outRate)
	}
	defer C.crispasr_audio_free(outPcm)
	if rc != 0 {
		return nil, 0, fmt.Errorf("AudioLoad failed (rc=%d)", int(rc))
	}
	if outSamples < 0 || (outSamples > 0 && outPcm == nil) {
		return nil, 0, errors.New("AudioLoad: invalid native buffer")
	}
	if int(outSamples) == 0 {
		return nil, int(outRate), nil
	}
	count := int(outSamples)
	pcm := make([]float32, count)
	copy(pcm, unsafe.Slice((*float32)(unsafe.Pointer(outPcm)), count))
	return pcm, int(outRate), nil
}

func AudioLoadStereo(path string) (left, right []float32, samples, sampleRate, channels int, err error) {
	cpath := C.CString(path)
	defer C.free(unsafe.Pointer(cpath))
	var outLeft, outRight *C.float
	var outSamples, outRate, outCh C.int
	rc := C.crispasr_audio_load_stereo(cpath, &outLeft, &outRight, &outSamples, &outRate, &outCh)
	defer C.crispasr_audio_free(outLeft)
	if outRight != outLeft {
		defer C.crispasr_audio_free(outRight)
	}
	if rc != 0 {
		err = fmt.Errorf("AudioLoadStereo failed (rc=%d)", int(rc))
		return
	}
	if outSamples < 0 || (outSamples > 0 && (outLeft == nil || outRight == nil)) {
		err = errors.New("AudioLoadStereo: invalid native buffer")
		return
	}
	if outSamples == 0 {
		samples = 0
		sampleRate = int(outRate)
		channels = int(outCh)
		return
	}
	samples = int(outSamples)
	sampleRate = int(outRate)
	channels = int(outCh)
	count := samples
	left = make([]float32, count)
	right = make([]float32, count)
	copy(left, unsafe.Slice((*float32)(unsafe.Pointer(outLeft)), count))
	copy(right, unsafe.Slice((*float32)(unsafe.Pointer(outRight)), count))
	return
}

func AudioResample(pcm []float32, sourceRate, targetRate int) ([]float32, error) {
	if sourceRate <= 0 || targetRate <= 0 {
		return nil, errors.New("AudioResample: invalid rates")
	}
	if len(pcm) == 0 {
		return nil, nil
	}
	var outPcm *C.float
	var outSamples C.int
	rc := C.crispasr_audio_resample((*C.float)(unsafe.Pointer(&pcm[0])), C.int(len(pcm)), C.int(sourceRate), C.int(targetRate), &outPcm, &outSamples)
	defer C.crispasr_audio_free(outPcm)
	if rc != 0 {
		return nil, fmt.Errorf("AudioResample failed (rc=%d)", int(rc))
	}
	if outSamples < 0 || (outSamples > 0 && outPcm == nil) {
		return nil, errors.New("AudioResample: invalid native buffer")
	}
	if outSamples == 0 {
		return nil, nil
	}
	count := int(outSamples)
	out := make([]float32, count)
	copy(out, unsafe.Slice((*float32)(unsafe.Pointer(outPcm)), count))
	return out, nil
}

// Native allocations must be freed even when their returned size is invalid.
func copyNativeContainer(ptr *C.uchar, size C.size_t) ([]byte, error) {
	if ptr != nil {
		defer C.crispasr_c2pa_free(ptr)
	}
	if ptr == nil || size == 0 || uint64(size) > math.MaxInt32 {
		return nil, errors.New("invalid native container pointer or copy length")
	}
	return C.GoBytes(unsafe.Pointer(ptr), C.int(size)), nil
}

func PcmToWav(pcm []float32, sampleRate int) ([]byte, error) {
	if len(pcm) == 0 {
		return nil, errors.New("PcmToWav: empty PCM")
	}
	if sampleRate <= 0 {
		return nil, errors.New("PcmToWav: invalid sampleRate")
	}
	var outLen C.size_t
	ptr := C.crispasr_pcm_to_wav((*C.float)(unsafe.Pointer(&pcm[0])), C.int(len(pcm)), C.int(sampleRate), &outLen)
	return copyNativeContainer(ptr, outLen)

}

func C2paSign(data []byte, format, certPath, keyPath string) ([]byte, error) {
	if len(data) == 0 {
		return nil, errors.New("C2paSign: empty data")
	}
	if format == "" {
		return nil, errors.New("C2paSign: format required")
	}
	if (certPath == "") != (keyPath == "") {
		return nil, errors.New("certificate and key must be paired")
	}
	if strings.IndexByte(format, 0) >= 0 || strings.IndexByte(certPath, 0) >= 0 || strings.IndexByte(keyPath, 0) >= 0 {
		return nil, errors.New("signing options must not contain NUL")
	}
	cFormat := C.CString(format)
	defer C.free(unsafe.Pointer(cFormat))
	var cCert, cKey *C.char
	if certPath != "" {
		cCert = C.CString(certPath)
		defer C.free(unsafe.Pointer(cCert))
	}
	if keyPath != "" {
		cKey = C.CString(keyPath)
		defer C.free(unsafe.Pointer(cKey))
	}
	var outLen C.size_t
	ptr := C.crispasr_c2pa_sign((*C.uchar)(unsafe.Pointer(&data[0])), C.size_t(len(data)), cFormat, cCert, cKey, &outLen)
	return copyNativeContainer(ptr, outLen)

}

// WatermarkLoadModel already declared in crispasr_session.go — not redeclared here

func FreeAlignerCache() {
	C.crispasr_free_aligner_cache()
}

func (s *CrispasrSession) SetPcmSampleRate(rate int) error {
	s.nativeHandle()
	if s.handle == nil {
		return errors.New("session is closed")
	}
	rc := C.crispasr_session_set_pcm_sample_rate((*C.struct_crispasr_session)(unsafe.Pointer(s.nativeHandle())), C.int(rate))
	if rc != 0 {
		return fmt.Errorf("SetPcmSampleRate failed (rc=%d)", int(rc))
	}
	return nil
}

func (s *CrispasrSession) SetParakeetAttContext(left, right int) error {
	s.nativeHandle()
	if s.handle == nil {
		return errors.New("session is closed")
	}
	rc := C.crispasr_session_set_parakeet_att_context((*C.struct_crispasr_session)(unsafe.Pointer(s.nativeHandle())), C.int(left), C.int(right))
	if rc != 0 {
		return fmt.Errorf("SetParakeetAttContext failed (rc=%d)", int(rc))
	}
	return nil
}

func (s *CrispasrSession) SetAudioCtx(audioCtx int) error {
	s.nativeHandle()
	if s.handle == nil {
		return errors.New("session is closed")
	}
	rc := C.crispasr_session_set_audio_ctx((*C.struct_crispasr_session)(unsafe.Pointer(s.nativeHandle())), C.int(audioCtx))
	if rc != 0 {
		return fmt.Errorf("SetAudioCtx failed (rc=%d)", int(rc))
	}
	return nil
}

func (s *CrispasrSession) SetHikariRefreshStride(stride int) error {
	s.nativeHandle()
	if s.handle == nil {
		return errors.New("session is closed")
	}
	rc := C.crispasr_session_set_hikari_refresh_stride((*C.struct_crispasr_session)(unsafe.Pointer(s.nativeHandle())), C.int(stride))
	if rc != 0 {
		return fmt.Errorf("SetHikariRefreshStride failed (rc=%d)", int(rc))
	}
	return nil
}

func (s *CrispasrSession) SetGrammarStrict(strict bool) error {
	s.nativeHandle()
	if s.handle == nil {
		return errors.New("session is closed")
	}
	v := 0
	if strict {
		v = 1
	}
	rc := C.crispasr_session_set_grammar_strict((*C.struct_crispasr_session)(unsafe.Pointer(s.nativeHandle())), C.int(v))
	if rc != 0 {
		return fmt.Errorf("SetGrammarStrict failed (rc=%d)", int(rc))
	}
	return nil
}

func (s *CrispasrSession) ScoreTexts(pcm []float32, language, initialPrompt string, texts []string) ([]float32, []int, error) {
	s.nativeHandle()
	if s.handle == nil {
		return nil, nil, errors.New("session is closed")
	}
	if len(pcm) == 0 || len(pcm) > math.MaxInt32 || len(texts) > math.MaxInt32 {
		return nil, nil, errors.New("invalid PCM or candidate count")
	}
	if strings.IndexByte(language, 0) >= 0 || strings.IndexByte(initialPrompt, 0) >= 0 {
		return nil, nil, errors.New("scoring options must not contain NUL")
	}
	for _, text := range texts {
		if strings.IndexByte(text, 0) >= 0 {
			return nil, nil, errors.New("candidate must not contain NUL")
		}
	}
	if len(texts) == 0 {
		return nil, nil, errors.New("ScoreTexts: empty texts")
	}
	var pcmPtr *C.float
	if len(pcm) > 0 {
		pcmPtr = (*C.float)(unsafe.Pointer(&pcm[0]))
	}
	var cLang, cPrompt *C.char
	if language != "" {
		cLang = C.CString(language)
		defer C.free(unsafe.Pointer(cLang))
	}
	if initialPrompt != "" {
		cPrompt = C.CString(initialPrompt)
		defer C.free(unsafe.Pointer(cPrompt))
	}
	cTexts := make([]*C.char, len(texts))
	for i, t := range texts {
		cTexts[i] = C.CString(t)
		defer C.free(unsafe.Pointer(cTexts[i]))
	}
	outLogprobs := make([]float32, len(texts))
	outNTokens := make([]int32, len(texts))
	rc := C.crispasr_session_score_texts((*C.struct_crispasr_session)(unsafe.Pointer(s.nativeHandle())), pcmPtr, C.int(len(pcm)), cLang, cPrompt, (**C.char)(unsafe.Pointer(&cTexts[0])), C.int(len(texts)), (*C.float)(unsafe.Pointer(&outLogprobs[0])), (*C.int)(unsafe.Pointer(&outNTokens[0])))
	if rc != 0 {
		return nil, nil, fmt.Errorf("ScoreTexts failed (rc=%d)", int(rc))
	}
	nTokens := make([]int, len(texts))
	for i, v := range outNTokens {
		nTokens[i] = int(v)
	}
	return outLogprobs, nTokens, nil
}

func SetGpuBackend(name string) {
	var cName *C.char
	if name != "" {
		cName = C.CString(name)
		defer C.free(unsafe.Pointer(cName))
	}
	C.crispasr_set_gpu_backend(cName)
}

func DetectBackendsFromGGUF(path string) ([]string, error) {
	cpath := C.CString(path)
	defer C.free(unsafe.Pointer(cpath))
	buf := make([]C.char, 512)
	n := C.crispasr_detect_backends_from_gguf(cpath, &buf[0], C.int(len(buf)))
	if n <= 0 {
		return nil, fmt.Errorf("DetectBackendsFromGGUF failed (rc=%d)", int(n))
	}
	s := C.GoString(&buf[0])
	var out []string
	for _, line := range splitLines(s) {
		if line != "" {
			out = append(out, line)
		}
	}
	return out, nil
}

func CApiVersion() string {
	ptr := C.crispasr_c_api_version()
	if ptr == nil {
		return ""
	}
	return C.GoString(ptr)
}

func DartHelpersVersion() string {
	ptr := C.crispasr_dart_helpers_version()
	if ptr == nil {
		return ""
	}
	return C.GoString(ptr)
}

func (s *CrispasrSession) InputChannels() int {
	s.nativeHandle()
	if s.handle == nil {
		return 0
	}
	return int(C.crispasr_session_input_channels((*C.struct_crispasr_session)(unsafe.Pointer(s.nativeHandle()))))
}

func (s *CrispasrSession) OutputChannels() int {
	s.nativeHandle()
	if s.handle == nil {
		return 0
	}
	return int(C.crispasr_session_output_channels((*C.struct_crispasr_session)(unsafe.Pointer(s.nativeHandle()))))
}

func ResetProgress() { C.crispasr_reset_progress() }

// Streamed polling helpers
func GetStreamedSegmentCount() int { return int(C.crispasr_get_streamed_segment_count()) }
func ResetStreamedSegments()       { C.crispasr_reset_streamed_segments() }
func GetStreamedTokenCount() int   { return int(C.crispasr_get_streamed_token_count()) }

// DrainStreamedTokens copies the native NUL-separated token buffer. Go strings
// preserve raw UTF-8 fragments. Serialize polling with session inference/close:
// the C API's buffer belongs to the session that most recently streamed.
func DrainStreamedTokens() ([]string, error) {
	var n C.int
	ptr := C.crispasr_drain_streamed_tokens(&n)
	if n < 0 || (n > 0 && ptr == nil) {
		return nil, errors.New("invalid native streamed token buffer")
	}
	if n == 0 {
		return nil, nil
	}
	out := make([]string, 0, int(n))
	base := unsafe.Pointer(ptr)
	for i := 0; i < int(n); i++ {
		token := C.GoString((*C.char)(base))
		out = append(out, token)
		base = unsafe.Add(base, len(token)+1)
	}
	return out, nil
}
func ResetStreamedTokens() { C.crispasr_reset_streamed_tokens() }

func splitLines(s string) []string {
	var out []string
	start := 0
	for i, c := range s {
		if c == '\n' {
			out = append(out, s[start:i])
			start = i + 1
		}
	}
	if start < len(s) {
		out = append(out, s[start:])
	}
	return out
}
