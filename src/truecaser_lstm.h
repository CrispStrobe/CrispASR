#ifndef TRUECASER_LSTM_H
#define TRUECASER_LSTM_H

#ifdef __cplusplus
extern "C" {
#endif

struct truecaser_lstm_context;

// Load a BiLSTM truecaser binary model.
struct truecaser_lstm_context* truecaser_lstm_init(const char* model_path);

// Apply truecasing to text. Returns newly allocated string (caller frees).
char* truecaser_lstm_process(struct truecaser_lstm_context* ctx, const char* text);

// Free context.
void truecaser_lstm_free(struct truecaser_lstm_context* ctx);

#ifdef __cplusplus
}
#endif

#endif // TRUECASER_LSTM_H
