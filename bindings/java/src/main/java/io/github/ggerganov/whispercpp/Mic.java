package io.github.ggerganov.whispercpp;

import com.sun.jna.Native;
import com.sun.jna.Pointer;

/**
 * Library-level microphone capture handle (PLAN #62d). Wraps the C-ABI
 * {@code crispasr_mic_*} functions which delegate to miniaudio's
 * {@code ma_device} (Core Audio on macOS, ALSA/PulseAudio on Linux,
 * WASAPI on Windows).
 *
 * <p>The user callback fires on miniaudio's audio thread. JNA
 * auto-attaches a JNIEnv to that thread on first call; we explicitly
 * detach when the callback returns ({@link Native#detach Native.detach(true)})
 * so the audio thread doesn't keep a JNIEnv pinned across firings.
 *
 * <p>Threading: keep work in the user callback short and non-blocking.
 * For streaming ASR, push samples into a ring buffer and drain on a
 * dedicated worker thread; if you call into a {@link CrispasrSession.Stream}
 * directly from the callback, decode latency will block the audio
 * thread and the driver will eventually drop frames.
 *
 * <p>Use as a try-with-resources block:
 * <pre>{@code
 * try (Mic mic = Mic.open(16000, 1, pcm -> queue.offer(pcm.clone()))) {
 *     mic.start();
 *     // ... pump samples elsewhere ...
 *     mic.stop();
 * }
 * }</pre>
 */
public final class Mic implements AutoCloseable {

    /** User-supplied callback. Receives a copy of the audio buffer (safe to retain). */
    public interface Listener {
        void onAudio(float[] pcm);
    }

    private Pointer handle;
    // Strong reference to the JNA Callback so it isn't GC'd while the
    // C side is calling it. The C audio thread holds no Java reference.
    @SuppressWarnings("unused")
    private final CrispasrSession.MicCallback nativeCallback;
    private boolean started;

    private Mic(Pointer handle, CrispasrSession.MicCallback nativeCallback) {
        this.handle = handle;
        this.nativeCallback = nativeCallback;
    }

    /**
     * Open a microphone capture device. Handle is NOT started — call
     * {@link #start()} to begin capture.
     *
     * @param sampleRate target rate in Hz; 16000 matches every ASR backend
     * @param channels   1 (mono) recommended
     * @param listener   per-buffer callback; receives a copy of the audio
     */
    public static Mic open(int sampleRate, int channels, final Listener listener) {
        if (listener == null) throw new IllegalArgumentException("listener is required");

        CrispasrSession.MicCallback cb = new CrispasrSession.MicCallback() {
            @Override
            public void invoke(Pointer pcm, int nSamples, Pointer userdata) {
                try {
                    if (pcm != null && nSamples > 0) {
                        float[] copy = pcm.getFloatArray(0, nSamples);
                        listener.onAudio(copy);
                    }
                } catch (Throwable t) {
                    // Never propagate — JNA marshals exceptions via the
                    // audio thread, and miniaudio doesn't expect that.
                    System.err.println("Mic.Listener threw: " + t);
                } finally {
                    // Detach this thread's JNIEnv when the callback
                    // returns. Audio threads are owned by miniaudio,
                    // not the JVM — without this, JVM thread tables
                    // grow on every device restart.
                    Native.detach(true);
                }
            }
        };

        Pointer p = CrispasrSession.Lib.INSTANCE.crispasr_mic_open(sampleRate, channels, cb, null);
        if (p == null) throw new IllegalStateException("crispasr_mic_open failed");
        return new Mic(p, cb);
    }

    /** Begin capture. Throws on driver error. */
    public void start() {
        if (handle == null) throw new IllegalStateException("mic is closed");
        int rc = CrispasrSession.Lib.INSTANCE.crispasr_mic_start(handle);
        if (rc != 0) throw new IllegalStateException("crispasr_mic_start failed (rc=" + rc + ")");
        started = true;
    }

    /** Stop capture. Idempotent. The callback may still fire briefly while the driver drains. */
    public void stop() {
        if (handle == null || !started) return;
        CrispasrSession.Lib.INSTANCE.crispasr_mic_stop(handle);
        started = false;
    }

    /** Free the handle + release the device. Implies {@link #stop()}. */
    @Override
    public void close() {
        if (handle == null) return;
        stop();
        CrispasrSession.Lib.INSTANCE.crispasr_mic_close(handle);
        handle = null;
    }

    /** Default capture-device name, or empty string if no input device is available. */
    public static String defaultDeviceName() {
        String s = CrispasrSession.Lib.INSTANCE.crispasr_mic_default_device_name();
        return s == null ? "" : s;
    }
}
