#ifndef CRISPASR_PUNC_LOADER_H
#define CRISPASR_PUNC_LOADER_H

// CLI-layer alias for the shared `--punc-model` resolver. The pure resolution
// table now lives in src/crispasr_punc_model.h so the C-ABI session layer
// (src/crispasr_c_api.cpp) can share it too; this header just re-exports it for
// the CLI one-shot path (crispasr_run.cpp) and the HTTP server
// (crispasr_server.cpp), which include it by this name.
#include "crispasr_punc_model.h"

#endif // CRISPASR_PUNC_LOADER_H
