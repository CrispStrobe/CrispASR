package io.github.ggerganov.whispercpp;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIfEnvironmentVariable;

import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Smoke test for the JNA microphone binding (PLAN #62/#62d).
 *
 * <p>Opens a real capture device, runs ~1s, and asserts the audio
 * thread fired the user callback at least 10 times. Catches the
 * common JNI-thread-attach failure modes that otherwise crash CI
 * silently:
 *   - Callback never holds a strong ref → GC'd → JVM segfault
 *   - Audio thread never detached → JVM thread table leaks
 *   - Mic.close() never stops the device → callbacks fire on a
 *     freed handle and corrupt memory
 *
 * <p>Gated behind {@code CRISPASR_MIC_TEST=1} because CI runners
 * typically have no microphone and the device-open call would fail.
 */
class MicSmokeTest {

    @Test
    @EnabledIfEnvironmentVariable(named = "CRISPASR_MIC_TEST", matches = "1")
    void micFiresCallbacks() throws Exception {
        AtomicInteger callbacks = new AtomicInteger(0);
        AtomicInteger totalSamples = new AtomicInteger(0);

        try (Mic mic = Mic.open(16000, 1, pcm -> {
            callbacks.incrementAndGet();
            totalSamples.addAndGet(pcm.length);
        })) {
            mic.start();
            Thread.sleep(1000);
            mic.stop();
        }

        // 16 kHz mono with miniaudio's 256-4096 sample buffers ⇒ at
        // least 4-60 callbacks per second. 10 is a safe floor.
        assertTrue(callbacks.get() >= 10,
                "expected ≥10 mic callbacks in 1s, got " + callbacks.get());
        assertTrue(totalSamples.get() > 8000,
                "expected ≥0.5s of audio, got " + totalSamples.get() + " samples");
    }

    @Test
    @EnabledIfEnvironmentVariable(named = "CRISPASR_MIC_TEST", matches = "1")
    void defaultDeviceNameIsAvailable() {
        String name = Mic.defaultDeviceName();
        assertNotNull(name);
        // Empty is allowed (headless host with no input device); we
        // just want to confirm the symbol resolves without crashing.
        System.out.println("default capture device: '" + name + "'");
    }
}
