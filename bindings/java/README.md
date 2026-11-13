# Java JNA bindings for CrispASR

This package provides Java JNA bindings for crispasr. They have been tested on:

  * <strike>Darwin (OS X) 12.6 on x64_64</strike>
  * Ubuntu on x86_64
  * Windows on x86_64

The "low level" bindings are in `WhisperCppJnaLibrary`. The most simple usage is as follows:

JNA will attempt to load the `crispasr` shared library from:

- jna.library.path
- jna.platform.library
- ~/Library/Frameworks
- /Library/Frameworks
- /System/Library/Frameworks
- classpath

```java
import io.github.ggerganov.whispercpp.WhisperCpp;

public class Example {

    public static void main(String[] args) {
        
        WhisperCpp whisper = new WhisperCpp();
        try {
            // By default, models are loaded from ~/.cache/whisper/ and are usually named "ggml-${name}.bin"
            // or you can provide the absolute path to the model file.
            whisper.initContext("../ggml-base.en.bin"); 
            WhisperFullParams.ByValue whisperParams = whisper.getFullDefaultParams(WhisperSamplingStrategy.CRISPASR_SAMPLING_BEAM_SEARCH); 
            
            // custom configuration if required      
            //whisperParams.n_threads = 8;
            whisperParams.temperature = 0.0f;
            whisperParams.temperature_inc = 0.2f;
            //whisperParams.language = "en";
                            
            float[] samples = readAudio(); // divide each value by 32767.0f
            List<WhisperSegment> whisperSegmentList = whisper.fullTranscribeWithTime(whisperParams, samples);
            
            for (WhisperSegment whisperSegment : whisperSegmentList) {

                long start = whisperSegment.getStart();
                long end = whisperSegment.getEnd();

                String text = whisperSegment.getSentence();
                    
                System.out.println("start: "+start);
                System.out.println("end: "+end);
                System.out.println("text: "+text);
                
            }
    
        } catch (IOException e) {
            e.printStackTrace();
        } finally {
            whisper.close();
        }
        
     }
}
```

## Chat / LLM

`io.github.ggerganov.whispercpp.chat` binds the `crispasr_chat_*` C ABI
(`include/crispasr_chat.h`) — text in, text out, separate from the ASR surface
above and usable on its own.

```java
import io.github.ggerganov.whispercpp.chat.*;
import java.util.Arrays;
import java.util.List;

List<ChatMessage> turns = Arrays.asList(
        ChatMessage.system("You are terse."),
        ChatMessage.user("Name three primes."));

try (ChatSession s = ChatSession.open("/models/gemma-3-1b-it-Q4_K_M.gguf",
        new ChatOpenParams().nCtx(4096))) {

    System.out.println(s.templateName() + ", " + s.nCtx() + " tokens");
    System.out.println(s.countTokens(turns) + " tokens of prompt");

    // One shot.
    System.out.println(s.generate(turns, new ChatGenerateParams().maxTokens(128)));

    // Or streamed. Chunks are whole characters; concatenated they equal the
    // one-shot text. The listener must not call back into the same session.
    s.generateStream(turns, new ChatGenerateParams().maxTokens(128), System.out::print);
}
```

Points worth knowing:

- **Pass the WHOLE conversation on every call.** The session compares the
  templated prompt against the tokens it already holds and decodes only what is
  new; passing just the latest turn re-prefills from scratch.
- **Cancellation.** `setAbortCallback` takes a predicate that returns **true to
  CONTINUE** — the polarity of the C callback and of the ASR side's
  encoder-begin callback. A cancelled call throws `ChatAbortedException` and
  leaves the session as if freshly opened, so no `reset()` is needed.
- **Params keep the ABI defaults.** `new ChatGenerateParams().maxTokens(64)`
  leaves `temperature` at 0.8, not 0.0.
- **`maxTokens(0)`** selects the ABI default of 256, not "generate nothing" —
  use `prefillOnly(true)` for that.
- **`memoryEstimate` over-reports on purpose.** It bills the KV cache at the
  full attention width, so on a grouped-query model the figure comes out well
  above the real working set — the safe direction for a "will this fit?"
  pre-flight guard. See its javadoc for the measured factor.
- **EU AI Act Art. 50.** This is synthetic text generation and nothing marks it.
  `ChatSession.aiDisclosureText()` is the canonical "you are talking to an AI"
  wording; show it visibly, and mark the output machine-readably yourself.

The end-to-end cases in `ChatSessionTest` need a GGUF chat model and are gated
on `CRISPASR_CHAT_TEST_MODEL`, self-skipping when it is unset:

```bash
CRISPASR_CHAT_TEST_MODEL=/models/gemma-3-1b-it-Q4_K_M.gguf ./gradlew test
```

## Shared auxiliary operations

`CrispasrSession.transcribeVad(pcm, rate, model, language, options)` accepts
`CrispasrSession.VadOptions`; the original three-argument overload keeps native
defaults. Durations are milliseconds, chunk length is seconds, and the language
applies only to this transcription.

`session.translateText(text, sourceLanguage, targetLanguage)` copies and frees
native UTF-8. `CrispasrSession.enhanceAudioRnnoise(pcm48kMono)` returns independent
PCM or throws when the native build lacks RNNoise. Use
`new CrispasrUtilities.TextProcessor(model, TextProcessorKind.PUNCTUATION)`
for standalone punctuation; the existing boolean constructor still selects
truecase or PCS.

`CrispasrAuxiliary.SpeakerDb(directory, expectedNamesCsv, consentAttested)`
provides `count`, `match`, `close` and static `enroll`. Negative native scores
are no match; enrollment writes to disk, so reopen to refresh the roster.
See the [callable ledger](../../docs/wrapper-callable-coverage.md) for reviewed
families and PLAN.md for remaining wrapper/platform/model gates.

## Building & Testing

In order to build, you need to have the JDK 8 or higher installed. Run the tests with:

```bash
git clone https://github.com/ggml-org/crispasr.git
cd crispasr/bindings/java

./gradlew build
```

You need to have the `crispasr` library in your [JNA library path](https://java-native-access.github.io/jna/4.2.1/com/sun/jna/NativeLibrary.html). On Windows the dll is included in the jar and you can update it:

```bash
copy /y ..\..\build\bin\Release\crispasr.dll build\generated\resources\main\win32-x86-64\crispasr.dll
```


## License

The license for the Java bindings is the same as the license for the rest of the crispasr project, which is the MIT License. See the `LICENSE` file for more details.


## Legacy parameter and context ownership

`WhisperCpp` passes context and decoding parameters through native pointer
exports. Context initialization forwards GPU/flash/device and every exposed DTW
setting, including custom heads and memory size. Custom head arrays are copied
and kept alive by the loaded model; later configuration changes affect a future
initialization. Native failure during replacement leaves the model closed;
invalid options or parameter-allocation failure preserve the existing context.

`getContextDefaultParams` and `getFullDefaultParams` return independent Java-owned
objects. Do not free their storage with `Native.free` or native parameter-free
functions. Temporary native defaults are freed internally with matching native
destructors. Repeated model close is harmless, and explicit reinitialization is
supported. Decoding after close fails. Native strings use UTF-8.

Overlapping context operations and close/reinitialization from a native callback
raise `IllegalStateException`. Callback functions and parameter storage stay
reachable through native inference, including foreign callback threads. Do not
mutate the parameter object or its borrowed pointer fields while inference runs.
The legacy full-parameter struct requires a 64-bit JVM; 32-bit legacy decode is
explicitly refused. Actual target/model validation remains in the repository
PLAN.md, alongside the generic `CrispasrSession` API and remaining callable audit.

### Experimental Ultra local-head alignment

```java
CrispasrSession.UltraAlignedWord[] words = CrispasrSession.alignWordsUltra(
    "qwen3-forced-aligner-ultra-f16.gguf", "ultra-local-head.gguf",
    "Words to align", pcm16khzMono, 4);
// words[i].text is owned UTF-8 text; start and end are double seconds.
```

This separate method preserves the local head's fractional-millisecond timing.
Audio is limited to 120 seconds. It requires a native library exporting the
Ultra local-head C ABI and both converted GGUF files; see the
[experimental guide](../../docs/qwen3-ultra-local-head-port.md) for conversion
and the bounded model-validation scope.
