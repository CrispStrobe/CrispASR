// voxtral4b-main — CLI for Mistral Voxtral-Mini-4B-Realtime-2602.
//
// Usage:
//   voxtral4b-main -m voxtral-mini-4b-realtime.gguf -f audio.wav [-t 4] [-l en]
//   voxtral4b-main -m model.gguf -f audio.wav -am aligner.gguf -timestamps

#include "voxtral4b.h"
#include "canary_ctc.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>
#include <chrono>

static bool load_wav_16k_mono(const std::string & path, std::vector<float> & out) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { fprintf(stderr, "cannot open %s\n", path.c_str()); return false; }
    auto read32 = [&](uint32_t & v) { f.read((char*)&v, 4); };
    auto read16 = [&](uint16_t & v) { f.read((char*)&v, 2); };
    char riff[4]; f.read(riff, 4); uint32_t riff_size; read32(riff_size);
    char wave[4]; f.read(wave, 4);
    if (memcmp(riff,"RIFF",4)!=0||memcmp(wave,"WAVE",4)!=0) return false;
    uint16_t afmt=0,nchan=0,bps=0; uint32_t sr=0; std::vector<uint8_t> data;
    while (f) {
        char cid[4]; f.read(cid,4); uint32_t csz; read32(csz); if(!f) break;
        if (!memcmp(cid,"fmt ",4)) {
            read16(afmt); read16(nchan); read32(sr);
            uint32_t br; read32(br); uint16_t ba; read16(ba); read16(bps);
            if (csz > 16) f.seekg(csz-16, std::ios::cur);
        } else if (!memcmp(cid,"data",4)) {
            data.resize(csz); f.read((char*)data.data(), csz); break;
        } else f.seekg(csz, std::ios::cur);
    }
    if (afmt!=1||bps!=16) return false;
    if (sr!=16000) { fprintf(stderr, "%s: need 16kHz (got %u)\n", path.c_str(), sr); return false; }
    const int16_t * pcm = (const int16_t *)data.data();
    size_t ns = data.size()/2;
    if (nchan == 1) {
        out.resize(ns); for (size_t i=0;i<ns;i++) out[i]=pcm[i]/32768.0f;
    } else {
        size_t nf = ns/nchan; out.resize(nf);
        for (size_t i=0;i<nf;i++) { float s=0; for(int c=0;c<nchan;c++) s+=pcm[i*nchan+c]/32768.0f; out[i]=s/nchan; }
    }
    return true;
}

static std::vector<std::string> tokenise_words(const std::string & text) {
    std::vector<std::string> out;
    std::string cur;
    for (char c : text) {
        if (c==' '||c=='\n'||c=='\t'||c=='\r') { if (!cur.empty()) { out.push_back(cur); cur.clear(); } }
        else cur += c;
    }
    if (!cur.empty()) out.push_back(cur);
    return out;
}

static void print_usage(const char * prog) {
    fprintf(stderr,
        "\nusage: %s -m MODEL.gguf -f AUDIO.wav [options]\n\n"
        "options:\n"
        "  -m  FNAME        voxtral-4b GGUF model (required)\n"
        "  -f  FNAME        input audio, 16 kHz mono WAV (required)\n"
        "  -t  N            threads (default: 4)\n"
        "  -l  LANG         language hint (default: en)\n"
        "  -n  N            max new tokens (default: 512)\n"
        "  -am FNAME        CTC aligner GGUF for timestamps\n"
        "  -timestamps      enable word-level timestamps (requires -am)\n"
        "  -np              suppress stderr info\n\n", prog);
}

int main(int argc, char ** argv) {
    std::string model_path, audio_path, aligner_path, lang = "en";
    int n_threads = 4, max_new = 512;
    bool timestamps = false, no_prints = false;

    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if      (a=="-m"  && i+1<argc) model_path   = argv[++i];
        else if (a=="-f"  && i+1<argc) audio_path   = argv[++i];
        else if (a=="-t"  && i+1<argc) n_threads    = std::atoi(argv[++i]);
        else if (a=="-l"  && i+1<argc) lang         = argv[++i];
        else if (a=="-n"  && i+1<argc) max_new      = std::atoi(argv[++i]);
        else if (a=="-am" && i+1<argc) aligner_path = argv[++i];
        else if (a=="-timestamps")     timestamps    = true;
        else if (a=="-np")             no_prints     = true;
        else if (a=="-h"||a=="--help") { print_usage(argv[0]); return 0; }
        else { fprintf(stderr, "unknown: %s\n", a.c_str()); print_usage(argv[0]); return 1; }
    }
    if (model_path.empty()||audio_path.empty()) { fprintf(stderr,"missing -m or -f\n"); return 1; }
    if (timestamps && aligner_path.empty()) { fprintf(stderr,"-timestamps requires -am\n"); return 1; }

    std::vector<float> samples;
    if (!load_wav_16k_mono(audio_path, samples)) return 2;
    if (!no_prints) fprintf(stderr, "audio: %.2f s (%zu samples)\n", samples.size()/16000.0, samples.size());

    auto cp = voxtral4b_context_default_params();
    cp.n_threads = n_threads;
    cp.verbosity = no_prints ? 0 : 1;
    auto * ctx = voxtral4b_init_from_file(model_path.c_str(), cp);
    if (!ctx) { fprintf(stderr, "init failed\n"); return 3; }

    auto t0 = std::chrono::steady_clock::now();

    // Mel
    int n_mels=0, T_mel=0;
    float * mel = voxtral4b_compute_mel(ctx, samples.data(), (int)samples.size(), &n_mels, &T_mel);
    if (!mel) { fprintf(stderr,"mel failed\n"); voxtral4b_free(ctx); return 4; }
    auto t_mel = std::chrono::steady_clock::now();
    if (!no_prints) fprintf(stderr, "mel: %d × %d  (%.0f ms)\n", n_mels, T_mel,
            std::chrono::duration<double,std::milli>(t_mel-t0).count());

    // Encoder
    int N_enc=0, pdim=0;
    float * audio_embeds = voxtral4b_run_encoder(ctx, mel, n_mels, T_mel, &N_enc, &pdim);
    free(mel);
    if (!audio_embeds) { fprintf(stderr,"encoder failed\n"); voxtral4b_free(ctx); return 5; }
    auto t_enc = std::chrono::steady_clock::now();
    if (!no_prints) fprintf(stderr, "encoder: %d frames × %d dim  (%.0f ms)\n", N_enc, pdim,
            std::chrono::duration<double,std::milli>(t_enc-t_mel).count());

    // Build prompt: BOS + STREAMING_PAD × (32 + delay_tokens)
    // The 4B Realtime model ADDS adapter output to token embeddings (not splice).
    const int delay_tokens = 6;  // 480ms default (6 × 80ms per token)
    const int prompt_pad = 32 + delay_tokens;
    int T_prompt = 1 + prompt_pad;  // BOS + pads

    // But we also need to cover all N_enc audio frames. The prompt length
    // should be at least N_enc so every audio frame has a decoder position.
    if (T_prompt < N_enc) T_prompt = N_enc;
    if (!no_prints) fprintf(stderr, "prompt: %d tokens (BOS + %d STREAMING_PAD, %d audio frames)\n",
                            T_prompt, T_prompt - 1, N_enc);

    // Build token IDs: BOS at pos 0, STREAMING_PAD everywhere else
    std::vector<int32_t> ids(T_prompt);
    ids[0] = 1;  // BOS
    for (int i = 1; i < T_prompt; i++) ids[i] = 32;  // STREAMING_PAD

    // Embed tokens, then ADD audio embeddings (element-wise)
    float * text_embeds = voxtral4b_embed_tokens(ctx, ids.data(), T_prompt);
    if (!text_embeds) { free(audio_embeds); voxtral4b_free(ctx); return 6; }
    for (int i = 0; i < std::min(N_enc, T_prompt); i++) {
        for (int j = 0; j < pdim; j++) {
            text_embeds[(size_t)i * pdim + j] += audio_embeds[(size_t)i * pdim + j];
        }
    }
    free(audio_embeds);

    // KV cache + prefill
    if (!voxtral4b_kv_init(ctx, 4096)) { free(text_embeds); voxtral4b_free(ctx); return 7; }
    voxtral4b_kv_reset(ctx);
    auto t_pf0 = std::chrono::steady_clock::now();
    int n_t=0, vocab=0;
    float * logits = voxtral4b_run_llm_kv(ctx, text_embeds, T_prompt, 0, &n_t, &vocab);
    auto t_pf1 = std::chrono::steady_clock::now();
    if (!logits) { free(text_embeds); voxtral4b_free(ctx); return 8; }
    free(text_embeds);
    if (!no_prints) fprintf(stderr, "prefill: %.0f ms\n",
            std::chrono::duration<double,std::milli>(t_pf1-t_pf0).count());

    int next=0; { float mx=-1e30f; for(int k=0;k<vocab;k++) if(logits[k]>mx){mx=logits[k];next=k;} }
    free(logits);

    // Greedy decode
    const int EOS = 2;
    std::vector<int32_t> gen; gen.push_back(next);
    auto t_dec0 = std::chrono::steady_clock::now();
    int n_past = T_prompt;
    while ((int)gen.size() < max_new && gen.back() != EOS) {
        int32_t last = gen.back();
        float * tail = voxtral4b_embed_tokens(ctx, &last, 1);
        if (!tail) break;
        float * lg = voxtral4b_run_llm_kv(ctx, tail, 1, n_past, nullptr, nullptr);
        free(tail); if (!lg) break;
        n_past++;
        int nx=0; float mx=-1e30f;
        for(int k=0;k<vocab;k++) if(lg[k]>mx){mx=lg[k];nx=k;}
        free(lg); gen.push_back(nx);
    }
    auto t_dec1 = std::chrono::steady_clock::now();
    double dec_ms = std::chrono::duration<double,std::milli>(t_dec1-t_dec0).count();
    if (!no_prints)
        fprintf(stderr, "decode: %zu tokens in %.0f ms (%.0f ms/token)\n",
                gen.size()-1, dec_ms, dec_ms/std::max((size_t)1,gen.size()-1));

    // Decode tokens
    std::string transcript;
    for (auto id : gen) {
        if (id == EOS) break;
        int len=0;
        const uint8_t * bytes = voxtral4b_token_text(ctx, id, &len);
        if (bytes && len > 0) transcript.append((const char*)bytes, len);
    }
    auto t_total = std::chrono::steady_clock::now();
    if (!no_prints) fprintf(stderr, "total: %.0f ms\n",
            std::chrono::duration<double,std::milli>(t_total-t0).count());

    // Timestamps (optional CTC aligner)
    if (timestamps) {
        canary_ctc_context_params acp = canary_ctc_context_default_params();
        acp.n_threads = n_threads;
        canary_ctc_context * actx = canary_ctc_init_from_file(aligner_path.c_str(), acp);
        if (actx) {
            float * ctc_logits = nullptr; int T_ctc=0, V_ctc=0;
            if (canary_ctc_compute_logits(actx, samples.data(), (int)samples.size(),
                                          &ctc_logits, &T_ctc, &V_ctc) == 0) {
                auto words = tokenise_words(transcript);
                if (!words.empty()) {
                    std::vector<canary_ctc_word> aw(words.size());
                    std::vector<const char *> wp(words.size());
                    for (size_t i=0;i<words.size();i++) wp[i]=words[i].c_str();
                    if (canary_ctc_align_words(actx, ctc_logits, T_ctc, V_ctc,
                                               wp.data(), (int)words.size(), aw.data()) == 0) {
                        for (const auto & w : aw)
                            printf("[%8.2fs → %8.2fs]  %s\n", w.t0/100.0, w.t1/100.0, w.text);
                        free(ctc_logits); canary_ctc_free(actx); voxtral4b_free(ctx); return 0;
                    }
                }
                free(ctc_logits);
            }
            canary_ctc_free(actx);
        }
    }

    printf("%s\n", transcript.c_str());
    voxtral4b_free(ctx);
    return 0;
}
