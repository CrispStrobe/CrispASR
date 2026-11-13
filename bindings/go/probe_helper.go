//go:build probe

package whisper

/*
#cgo CFLAGS: -I${SRCDIR}/../../include -I${SRCDIR}/../../ggml/include
#include <stdlib.h>
#include "crispasr_session.h"
#include "crispasr.h"
#include "crispasr_chat.h"
typedef struct crispasr_session CrispasrSession;
void probe_mode_set(int m);
int probe_vad_segments_live(void);
int probe_vad_segments_frees(void);
void probe_speaker_mode(int);
int probe_speaker_live(void);
void probe_streamed_ready(int ready);
void probe_legacy_expect(const char*, const char*);
void probe_legacy_mode(int);
int probe_legacy_calls(void);
int probe_parallel_by_ref_calls(void);
int probe_dtw_enabled(void);
int probe_legacy_context_flash(void);
void probe_legacy_tokens_reset(void);
void probe_legacy_tokens_mutate(void);
int probe_legacy_alt_text_calls(void);

int probe_dtw_preset(void);
int probe_dtw_top(void);
int probe_rust_legacy_contexts(void);
void probe_text_mode(int mode);
int probe_text_live(void);
int probe_text_contexts(void);
void probe_db_mode(int mode);
int probe_db_live(void);
int probe_db_calls(void);
void probe_titanet_mode(int mode);
int probe_titanet_live(void);
int probe_live_get(void);
int probe_calls_get(void);
int probe_result_frees_get(void);
void probe_reset(void);
void probe_mutate(void);
void probe_utility_mode(int mode);
int probe_utility_live(void);
int probe_stream_closes_get(void);
*/
import "C"
import (
	"testing"
	"unsafe"
)

func probeModeSet(m int)       { C.probe_mode_set(C.int(m)) }
func probeLiveGet() int        { return int(C.probe_live_get()) }
func probeCallsGet() int       { return int(C.probe_calls_get()) }
func probeResultFreesGet() int { return int(C.probe_result_frees_get()) }
func probeReset()              { C.probe_reset() }
func probeMutate()             { C.probe_mutate() }

func newProbeSession(t *testing.T) *CrispasrSession {
	t.Helper()
	h := (*C.CrispasrSession)(unsafe.Pointer(uintptr(0x100000001)))
	return &CrispasrSession{handle: h}
}

func probeUtilityMode(mode int) { C.probe_utility_mode(C.int(mode)) }
func probeUtilityLive() int     { return int(C.probe_utility_live()) }
func probeStreamCloses() int    { return int(C.probe_stream_closes_get()) }

func probeTitaNetMode(mode int) { C.probe_titanet_mode(C.int(mode)) }
func probeTitaNetLive() int     { return int(C.probe_titanet_live()) }

func probeDbMode(mode int) { C.probe_db_mode(C.int(mode)) }
func probeDbLive() int     { return int(C.probe_db_live()) }
func probeDbCalls() int    { return int(C.probe_db_calls()) }

func probeTextMode(mode int) { C.probe_text_mode(C.int(mode)) }
func probeTextLive() int     { return int(C.probe_text_live()) }
func probeTextContexts() int { return int(C.probe_text_contexts()) }

func probeStreamedReady(ready int) { C.probe_streamed_ready(C.int(ready)) }

func probeLegacyExpect(prompt, model string) {
	p, m := C.CString(prompt), C.CString(model)
	defer C.free(unsafe.Pointer(p))
	defer C.free(unsafe.Pointer(m))
	C.probe_legacy_expect(p, m)
}
func probeLegacyMode(mode int)        { C.probe_legacy_mode(C.int(mode)) }
func probeParallelByRefCalls() int    { return int(C.probe_parallel_by_ref_calls()) }
func probeLegacyCalls() int           { return int(C.probe_legacy_calls()) }
func newProbeLegacyContext() *Context { return (*Context)(unsafe.Pointer(uintptr(0x100000001))) }

func probeDTWState() (bool, int, int) {
	return C.probe_dtw_enabled() != 0, int(C.probe_dtw_preset()), int(C.probe_dtw_top())
}
func probeLegacyContexts() int { return int(C.probe_rust_legacy_contexts()) }

func newProbeLegacyContextForWorker(worker int) *Context {
	return (*Context)(unsafe.Pointer(uintptr(0x100020000) + uintptr(worker)*16))
}

func probeLegacyContextFlash() bool { return C.probe_legacy_context_flash() != 0 }

func probeLegacyTokensReset()      { C.probe_legacy_tokens_reset() }
func probeLegacyTokensMutate()     { C.probe_legacy_tokens_mutate() }
func probeLegacyAltTextCalls() int { return int(C.probe_legacy_alt_text_calls()) }

func probeSpeakerMode(mode int) { C.probe_speaker_mode(C.int(mode)) }
func probeSpeakerLive() int     { return int(C.probe_speaker_live()) }

func probeVADSegmentsLive() int  { return int(C.probe_vad_segments_live()) }
func probeVADSegmentsFrees() int { return int(C.probe_vad_segments_frees()) }
