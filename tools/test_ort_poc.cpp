// test_ort_poc.cpp — lokaler POC-Test für den CrispASR-ORT-Umbau (PR Option B).
// Vergleicht TitaNet GGUF vs. ONNX (gleiche WAV -> Embeddings sollten ~identisch
// sein) und testet das pyannote-seg ONNX-Modell (Frame-Anzahl, Sprachaktivität).
#include "titanet.h"
#include "pyannote_seg.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static std::vector<float> load_wav_pcm16_mono(const char* path, int* sr_out) {
    std::vector<float> out;
    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); return out; }
    char hdr[44];
    if (fread(hdr, 1, 44, f) != 44) { fclose(f); return out; }
    unsigned sr = *(unsigned*)(hdr + 24);
    unsigned nch = *(unsigned short*)(hdr + 22);
    unsigned bits = *(unsigned short*)(hdr + 34);
    if (sr_out) *sr_out = (int)sr;
    fseek(f, 0, SEEK_END);
    long len = ftell(f);
    fseek(f, 44, SEEK_SET);
    if (bits == 16) {
        int n = (int)((len - 44) / 2);
        std::vector<short> raw(n);
        if (fread(raw.data(), 2, n, f) != (size_t)n) { fclose(f); return out; }
        out.resize(n / nch);
        for (int i = 0; i < (int)out.size(); i++)
            out[i] = raw[i * nch] / 32768.0f;
    }
    fclose(f);
    return out;
}

int main(int argc, char** argv) {
    if (argc < 5) {
        fprintf(stderr, "usage: %s <wav1> <wav2> <titanet.gguf> <titanet.onnx> [pyannote.onnx]\n", argv[0]);
        return 1;
    }
    int sr = 0;
    auto pcm1 = load_wav_pcm16_mono(argv[1], &sr);
    auto pcm2 = load_wav_pcm16_mono(argv[2], &sr);
    if (pcm1.empty() || pcm2.empty()) { fprintf(stderr, "no audio loaded\n"); return 1; }
    printf("audio1: %d samples @ %d Hz (%.2f s)\n", (int)pcm1.size(), sr, pcm1.size() / (float)sr);
    printf("audio2: %d samples @ %d Hz (%.2f s)\n", (int)pcm2.size(), sr, pcm2.size() / (float)sr);

    struct titanet_context* gg = titanet_init(argv[3], 4);
    struct titanet_context* on = titanet_init(argv[4], 4);
    if (!gg || !on) {
        fprintf(stderr, "titanet init failed (gg=%p on=%p)\n", (void*)gg, (void*)on);
        return 1;
    }
    std::vector<float> g1(192), g2(192), o1(192), o2(192);
    int ng1 = titanet_embed(gg, pcm1.data(), (int)pcm1.size(), g1.data());
    int ng2 = titanet_embed(gg, pcm2.data(), (int)pcm2.size(), g2.data());
    int no1 = titanet_embed(on, pcm1.data(), (int)pcm1.size(), o1.data());
    int no2 = titanet_embed(on, pcm2.data(), (int)pcm2.size(), o2.data());
    auto cos = [](const std::vector<float>& a, const std::vector<float>& b) {
        float dot = 0, na = 0, nb = 0;
        for (size_t i = 0; i < a.size(); i++) { dot += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i]; }
        return dot / (std::sqrt(na) * std::sqrt(nb) + 1e-12f);
    };
    printf("\nGGUF:  %d/%d/%d/%d dims (norm 1.0 = %s)\n", ng1, ng2, no1, no2,
           (ng1 == 192 && ng2 == 192 && no1 == 192 && no2 == 192) ? "ok" : "FEHLER");
    printf("cos(GGUF sr1, GGUF sr2) = %.4f  (gleiche Person ~1, verschiedene deutlich <)\n", cos(g1, g2));
    printf("cos(ONNX sr1, ONNX sr2) = %.4f  (gleiche Person ~1, verschiedene deutlich <)\n", cos(o1, o2));
    printf("cos(GGUF sr1, ONNX sr1) = %.4f  (Modell-Aequivalenz, ~1 ideal)\n", cos(g1, o1));
    printf("cos(GGUF sr2, ONNX sr2) = %.4f\n", cos(g2, o2));

    if (argc > 5) {
        struct pyannote_seg_context* pc = pyannote_seg_init(argv[5], 4);
        if (!pc) { fprintf(stderr, "pyannote init failed\n"); return 1; }
        int T = 0;
        float* probs = pyannote_seg_run(pc, pcm1.data(), (int)pcm1.size(), &T);
        printf("pyannote ONNX: %d frames (%.2f s audio -> %.1f ms/frame)\n", T,
               pcm1.size() / 16000.0f, T ? pcm1.size() / 16000.0f / T * 1000.0f : 0.0f);
        if (probs) {
            // Sprachaktivität: max über Klassen 1..6 > max(0.5 vs silence-Klasse)
            int speech = 0;
            for (int t = 0; t < T; t++) {
                const float* row = probs + (size_t)t * 7;
                float p_sil = row[0];
                float p_spk = row[1];
                for (int k = 2; k < 7; k++) p_spk = std::max(p_spk, row[k]);
                if (p_spk > p_sil) speech++;
            }
            printf("pyannote ONNX: %d/%d Frames mit Sprachaktivitaet\n", speech, T);
            free(probs);
        }
        pyannote_seg_free(pc);
    }

    titanet_free(gg);
    titanet_free(on);
    return 0;
}
