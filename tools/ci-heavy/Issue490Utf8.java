import io.github.ggerganov.whispercpp.CrispasrSession;

// The native guard validates incoming bytes as well as the returned UTF-8 text.
public final class Issue490Utf8 {
    public static void main(String[] args) {
        CrispasrSession.AlignedWord[] words = CrispasrSession.alignWords(
            "/tmp/\u0645\u0648\u062f\u064a\u0644.gguf", "\u0628", new float[] {0.25f}, 325, 4);
        if (words.length != 1 || !words[0].text.equals("\u0628") || words[0].t0 != 325 || words[0].t1 != 327
                || words[0].characters.length != 1 || !words[0].characters[0].text.equals("\u0628")
                || words[0].characters[0].t0 != 325 || words[0].characters[0].t1 != 327)
            throw new AssertionError("UTF-8 text or centisecond times changed");
        System.out.println("JAVA_BIDIRECTIONAL_UTF8_PASS");
    }
}
