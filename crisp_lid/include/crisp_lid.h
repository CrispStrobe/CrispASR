// crisp_lid.h — shared text-based language identification library.
//
// Two backends auto-detected from GGUF architecture:
//   lid-fasttext — GlotLID / LID-176 (fastText supervised)
//   lid-cld3     — Google CLD3 (compact, Apache-2.0)
//
// Used by CrispASR (post-ASR LID) and CrispEmbed (OCR language routing).

#ifndef CRISP_LID_H
#define CRISP_LID_H

// Re-export the dispatch API — this is the public interface.
#include "../src/text_lid_dispatch.h"

#endif // CRISP_LID_H
