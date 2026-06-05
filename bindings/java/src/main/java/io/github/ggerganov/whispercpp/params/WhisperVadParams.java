package io.github.ggerganov.whispercpp.params;

import com.sun.jna.Structure;

import java.util.Arrays;
import java.util.List;

/**
 * Maps to C struct whisper_vad_params.
 */
public class WhisperVadParams extends Structure {

    public float threshold;
    public int min_speech_duration_ms;
    public int min_silence_duration_ms;
    public float max_speech_duration_s;
    public int speech_pad_ms;
    public float samples_overlap;

    @Override
    protected List<String> getFieldOrder() {
        return Arrays.asList(
            "threshold",
            "min_speech_duration_ms",
            "min_silence_duration_ms",
            "max_speech_duration_s",
            "speech_pad_ms",
            "samples_overlap"
        );
    }

    public static class ByValue extends WhisperVadParams implements Structure.ByValue {}
}
