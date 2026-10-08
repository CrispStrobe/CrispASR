import io.github.ggerganov.whispercpp.CrispasrSession;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Base64;

// Run against the exact PCM16 WAV and transcript used by the Python/CLI gates.
public final class Issue490Characters {
    private static String encoded(String text) {
        return Base64.getEncoder().encodeToString(text.getBytes(StandardCharsets.UTF_8));
    }
    public static void main(String[] args) throws Exception {
        byte[] wav = Files.readAllBytes(Paths.get(args[1]));
        ByteBuffer bytes = ByteBuffer.wrap(wav).order(ByteOrder.LITTLE_ENDIAN);
        // soundfile's PCM16 fixture has the canonical 44-byte RIFF header.
        if (bytes.getInt(40) != wav.length - 44) throw new AssertionError("Unexpected WAV layout");
        float[] pcm = new float[(wav.length - 44) / 2];
        for (int i = 0; i < pcm.length; i++) pcm[i] = bytes.getShort(44 + 2 * i) / 32768.0f;
        String transcript = new String(Files.readAllBytes(Paths.get(args[2])), StandardCharsets.UTF_8).trim();
        CrispasrSession.AlignedWord[] words = CrispasrSession.alignWords(args[0], transcript, pcm, 325, 4);
        StringBuilder output = new StringBuilder();
        int characters = 0;
        for (CrispasrSession.AlignedWord word : words) {
            output.append("W\t").append(encoded(word.text)).append('\t').append(word.t0).append('\t').append(word.t1).append('\n');
            for (CrispasrSession.AlignedCharacter cp : word.characters) {
                output.append("C\t").append(encoded(cp.text)).append('\t').append(cp.t0).append('\t').append(cp.t1).append('\n');
                characters++;
            }
        }
        if (words.length == 0 || characters < 50) throw new AssertionError("Missing measured spans");
        Files.write(Paths.get(args[3]), output.toString().getBytes(StandardCharsets.UTF_8));
        System.out.println("JAVA_ARABIC_CTC_CHARACTER_PASS: " + words.length + " words, " + characters + " characters");
    }
}
