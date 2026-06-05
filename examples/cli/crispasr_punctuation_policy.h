// crispasr_punctuation_policy.h — decide when to auto-enable FireRedPunc.

#pragma once

#include "crispasr_backend.h"
#include "whisper_params.h"

// Auto-enable punctuation restoration only for backends that produce
// unpunctuated text by default and don't expose a native punctuation toggle.
// Backends that already emit punctuation should not get a second pass.
inline bool crispasr_should_auto_enable_punctuation(uint32_t caps, const whisper_params& params) {
    return params.punctuation && params.punc_model.empty() && !(caps & CAP_PUNCTUATION_TOGGLE) &&
           !(caps & CAP_PUNCTUATION_NATIVE);
}
