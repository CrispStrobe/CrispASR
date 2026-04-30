package io.github.ggerganov.whispercpp.params;

/** Available sampling strategies */
public enum WhisperSamplingStrategy {
    /** similar to OpenAI's GreedyDecoder */
    CRISPASR_SAMPLING_GREEDY,

    /** similar to OpenAI's BeamSearchDecoder */
    CRISPASR_SAMPLING_BEAM_SEARCH
}
