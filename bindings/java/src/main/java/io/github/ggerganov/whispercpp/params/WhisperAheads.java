package io.github.ggerganov.whispercpp.params;
import com.sun.jna.*;
import java.util.Arrays;
import java.util.List;

public class WhisperAheads extends Structure {
    /**
     * Number of alignment heads. C declares this as `size_t` (8 bytes on
     * every 64-bit platform — Win64 LLP64, Linux/macOS LP64). JNA's
     * `NativeLong` was the original type here, but `NativeLong` follows
     * C `long` semantics (4 bytes on Win64), which is a 4-byte ABI
     * mismatch against `size_t` and corrupts the struct layout — the
     * matching `dtw_mem_size` mismatch in WhisperContextParams was
     * causing `whisper_init_from_file_with_params` to read past the
     * Java-marshalled struct and JNA to throw `Invalid memory access`
     * once the windows build matrix actually ran (issue #70 bindings-
     * java cascade). The Java `long` primitive is always 8 bytes, so
     * matches `size_t` on every 64-bit target — and bindings-java only
     * ships x64 artifacts, so 32-bit divergence isn't reachable.
     */
    public long n_heads;

    public Pointer heads;

    public WhisperAheads() {
        super();
    }

    /**
     * Create alignment heads from an array of WhisperAhead objects
     */
    public void setHeads(WhisperAhead[] aheadsArray) {
        this.n_heads = aheadsArray.length;

        int structSize = aheadsArray[0].size();
        Memory mem = new Memory(structSize * aheadsArray.length);

        for (int i = 0; i < aheadsArray.length; i++) {
            aheadsArray[i].write();
            byte[] buffer = aheadsArray[i].getPointer().getByteArray(0, structSize);
            mem.write(i * structSize, buffer, 0, buffer.length);
        }

        this.heads = mem;
    }

    @Override
    protected List<String> getFieldOrder() {
        return Arrays.asList("n_heads", "heads");
    }

    public static class ByReference extends WhisperAheads implements Structure.ByReference {}

    public static class ByValue extends WhisperAheads implements Structure.ByValue {}
}
