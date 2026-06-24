import os
import json
import shlex  
import subprocess
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EXE_PATH = os.path.join(BASE_DIR, "crispasr.exe")
CONFIG_FILE = os.path.join(BASE_DIR, "config.json")

class ScrollableFrame(ttk.Frame):
    def __init__(self, container, *args, **kwargs):
        super().__init__(container, *args, **kwargs)
        canvas = tk.Canvas(self, borderwidth=0, highlightthickness=0)
        scrollbar = ttk.Scrollbar(self, orient="vertical", command=canvas.yview)
        self.scrollable_frame = ttk.Frame(canvas)

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        canvas.bind_all("<MouseWheel>", lambda e: canvas.yview_scroll(int(-1*(e.delta/120)), "units"))
        
        canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        self.inner = self.scrollable_frame

class CrispasrFullGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("CrispASR Professional Console (Multi-Track Env Injector)")
        self.root.geometry("1300x900")
        
        self.config = self.load_config()
        self.vars = {} 
        
        
        # Field Description: (Category, Label Name (with flag), Command Flag (ENV: prefix for env variables), Data Type, Default Value, Default Checked, Detailed Explanation)
        self.param_definitions = [
            # ========================= 1. General & Core =========================
            ("1. General & Core", "Help (--help)", "-h", "bool", "", False, "Show this help message and exit."),
            ("1. General & Core", "Version (--version)", "--version", "bool", "", False, "Print build info (version, git SHA, supported backends) and exit."),
            ("1. General & Core", "Diagnostics (--diagnostics)", "--diagnostics", "bool", "", False, "Print full diagnostics (build + env + GPU enumeration) and exit."),
            ("1. General & Core", "Model Path (-m)", "-m", "file", "auto", False, "Core model path. Set to 'auto' to automatically download a default model for the chosen backend."),
            ("1. General & Core", "Inference Backend (--backend)", "--backend", "str", "qwen3", False, "Specify the inference backend (e.g., whisper, parakeet, canary, qwen3)."),
            ("1. General & Core", "Audio File (-f)", "-f", "file", "", False, "Input audio source file path."),
            ("1. General & Core", "Language (-l)", "-l", "str", "auto", False, "Spoken language code (ISO-639-1, e.g., en, zh). Set to 'auto' for auto-detect."),
            ("1. General & Core", "Threads (-t)", "-t", "int", "4", False, "Number of threads to use during computation."),
            ("1. General & Core", "Processors (-p)", "-p", "int", "1", False, "Number of processors to use during computation."),
            ("1. General & Core", "Device ID (-dev)", "-dev", "int", "0", False, "Allocated GPU device ID (default: 0)."),
            ("1. General & Core", "Disable GPU (-ng)", "-ng", "bool", "", False, "Force disable GPU, utilizing pure CPU for inference."),
            ("1. General & Core", "GPU Backend (--gpu-backend)", "--gpu-backend", "str", "auto", False, "Force GPU backend architecture: cuda | vulkan | metal | cpu."),
            ("1. General & Core", "Enable Flash Attn (-fa)", "-fa", "bool", "", False, "Enable flash attention to reduce VRAM usage and increase speed."),
            ("1. General & Core", "Disable Flash Attn (-nfa)", "-nfa", "bool", "", False, "Disable flash attention (for compatibility with older hardware)."),
            ("1. General & Core", "Auto Download (--auto-download)", "--auto-download", "bool", "", False, "Auto-download missing models without prompting."),
            ("1. General & Core", "Cache Directory (--cache-dir)", "--cache-dir", "dir", "", False, "Override the default auto-download cache directory (default ~/.cache/crispasr/)."),
            ("1. General & Core", "HF Repo Pull (-hfr)", "-hfr", "str", "", False, "Fetch model from arbitrary HuggingFace repo (OWNER/REPO[:FILE])."),
            ("1. General & Core", "HF Target File (-hff)", "-hff", "str", "", False, "Filename within the HuggingFace repo (alternative to the shorthand in -hfr)."),
            ("1. General & Core", "OpenVINO Device (-oved)", "-oved", "str", "CPU", False, "The OpenVINO device used for encode inference."),

            # ========================= 2. Output & Format =========================
            ("2. Output & Format", "Output TXT (-otxt)", "-otxt", "bool", "", False, "Output transcription result in a plain text file without timestamps."),
            ("2. Output & Format", "Output SRT (-osrt)", "-osrt", "bool", "", False, "Output result in a standard SRT subtitle file."),
            ("2. Output & Format", "Output VTT (-ovtt)", "-ovtt", "bool", "", False, "Output result in a WebVTT subtitle file."),
            ("2. Output & Format", "Output CSV (-ocsv)", "-ocsv", "bool", "", False, "Output result in a CSV file (includes start, end time, and text)."),
            ("2. Output & Format", "Output JSON (-oj)", "-oj", "bool", "", False, "Output result in a JSON file."),
            ("2. Output & Format", "Output Full JSON (-ojf)", "-ojf", "bool", "", False, "Include word-level timestamps and token features in the JSON file."),
            ("2. Output & Format", "Output LRC (-olrc)", "-olrc", "bool", "", False, "Output result in a dynamic LRC lyric file with timestamps."),
            ("2. Output & Format", "Output Karaoke Script (-owts)", "-owts", "bool", "", False, "Output a base script used for generating karaoke highlight videos."),
            ("2. Output & Format", "Karaoke Font Path (-fp)", "-fp", "file", "/System/Library/Fonts/Supplemental/Courier New Bold.ttf", False, "Absolute path to a monospace font for karaoke video generation."),
            ("2. Output & Format", "Output File Prefix (-of)", "-of", "str", "", False, "Custom output file path and name (without file extension)."),
            ("2. Output & Format", "Max Segment Length (-ml)", "-ml", "int", "0", False, "Maximum segment display length in characters (0 for unlimited)."),
            ("2. Output & Format", "Split on Punctuation (-sp)", "-sp", "bool", "", False, "Force split subtitle lines at sentence-ending punctuation for readability."),
            ("2. Output & Format", "Split on Word (-sow)", "-sow", "bool", "", False, "Split based on word boundaries rather than token boundaries."),
            ("2. Output & Format", "Quiet Console (-np)", "-np", "bool", "", False, "Do not print anything to the console other than the final results."),
            ("2. Output & Format", "Print Colors (-pc)", "-pc", "bool", "", False, "Print using different colors in the console based on token confidence."),
            ("2. Output & Format", "No Timestamps (-nt)", "-nt", "bool", "", False, "Do not print timestamps prefix when outputting results in console."),

            # ========================= 3. Segmentation & Hotwords =========================
            ("3. Segmentation & Hotwords", "Time Offset (-ot)", "-ot", "int", "0", False, "Processing start time offset in milliseconds."),
            ("3. Segmentation & Hotwords", "Process Duration (-d)", "-d", "int", "0", False, "Duration of audio to process in milliseconds."),
            ("3. Segmentation & Hotwords", "Blind Chunk Length (-ck)", "-ck", "int", "30", False, "Fallback chunk size in seconds when VAD is disabled."),
            ("3. Segmentation & Hotwords", "Chunk Overlap (--chunk-overlap)", "--chunk-overlap", "float", "3.0", False, "Overlap context in seconds at chunk boundaries to prevent clipping words."),
            ("3. Segmentation & Hotwords", "LCS Deduplication (--lcs-dedup)", "--lcs-dedup", "str", "auto", False, "Sub-word longest common subsequence (LCS) dedup across chunk boundaries (auto|on|off)."),
            ("3. Segmentation & Hotwords", "LCS Min Length (--lcs-min-length)", "--lcs-min-length", "int", "1", False, "Minimum LCS duplicate length required to act on deduplication."),
            ("3. Segmentation & Hotwords", "Hotwords List (--hotwords)", "--hotwords", "long_text", "", False, "Comma-separated list of hotwords to boost recognition of domain-specific terminology (supports multiplier syntax like Name^3.0)."),
            ("3. Segmentation & Hotwords", "Hotwords File (--hotwords-file)", "--hotwords-file", "file", "", False, "Read hotwords from a local text file (one per line)."),
            ("3. Segmentation & Hotwords", "Hotwords Boost (--hotwords-boost)", "--hotwords-boost", "float", "2.0", False, "Log probability boost multiplier for hotword tokens (default: 2.0)."),

            # ========================= 4. Sampling & Decoding =========================
            ("4. Sampling & Decoding", "Temperature (-tp)", "-tp", "float", "0.00", False, "The decoding sampling temperature (0 = greedy decoding; >0 enables polynomial sampling)."),
            ("4. Sampling & Decoding", "Random Seed (--seed)", "--seed", "int", "0", False, "RNG seed for sampling. A fixed seed ensures bit-level deterministic audio/text results."),
            ("4. Sampling & Decoding", "Beam Size (-bs)", "-bs", "int", "5", False, "Beam search width (default: Whisper 5, others 1)."),
            ("4. Sampling & Decoding", "Max New Tokens (-n)", "-n", "int", "512", False, "Max new tokens generated per LLM backend forward pass."),
            ("4. Sampling & Decoding", "Frequency Penalty (--frequency-penalty)", "--frequency-penalty", "float", "0.00", False, "Penalize repeated generated tokens on AR backends to break repetitive loops."),
            ("4. Sampling & Decoding", "Parakeet Decoder (--parakeet-decoder)", "--parakeet-decoder", "str", "tdt", False, "Select specific decoding path (ctc | tdt | maes)."),
            ("4. Sampling & Decoding", "Word Probability Threshold (-wt)", "-wt", "float", "0.01", False, "Word timestamp probability threshold."),
            ("4. Sampling & Decoding", "Entropy Threshold (-et)", "-et", "float", "2.40", False, "Entropy threshold indicating a decoder failure to trigger fallback."),
            ("4. Sampling & Decoding", "Suppress Non-Speech (-sns)", "-sns", "bool", "", False, "Force suppress meaningless non-speech tokens."),
            ("4. Sampling & Decoding", "Suppress Regex (--suppress-regex)", "--suppress-regex", "str", "", False, "Regular expression matching tokens to explicitly suppress."),
            ("4. Sampling & Decoding", "GBNF Grammar (--grammar)", "--grammar", "file", "", False, "Provide a GBNF grammar file to guide and enforce the large model's output structure."),
            ("4. Sampling & Decoding", "Initial Prompt (--prompt)", "--prompt", "long_text", "", False, "Provide preceding context prompt to guide the model's tone or recognize proper nouns."),

            # ========================= 5. Alignment & Language =========================
            ("5. Alignment & Language", "Detect Language Only (-dl)", "-dl", "bool", "", False, "Exit immediately after automatically detecting the language."),
            ("5. Alignment & Language", "LID Backend (--lid-backend)", "--lid-backend", "str", "whisper", False, "Language-detect (LID) backend: whisper | silero | ecapa | firered."),
            ("5. Alignment & Language", "LID Model (--lid-model)", "--lid-model", "file", "", False, "LID model path (default auto-downloads ggml-tiny.bin)."),
            ("5. Alignment & Language", "LID on Transcript (--lid-on-transcript)", "--lid-on-transcript", "str", "", False, "Post-ASR text LID: Re-evaluate spoken language on the generated pure text."),
            ("5. Alignment & Language", "CTC Aligner Model (-am)", "-am", "file", "", False, "CTC aligner GGUF path used to generate word-level timestamps for LLM backends."),
            ("5. Alignment & Language", "Force CTC Aligner (-falign)", "-falign", "bool", "", False, "Use the CTC aligner's word timestamps even when the backend produces native ones."),
            ("5. Alignment & Language", "No Auto Aligner (--no-auto-aligner)", "--no-auto-aligner", "bool", "", False, "Skip the implicit automatic aligner mounting for backends like Canary."),
            ("5. Alignment & Language", "DTW Alignment (-dtw)", "-dtw", "file", "", False, "Compute ultra-fine token-level timestamps model path."),

            # ========================= 6. Translation & Multilingual =========================
            ("6. Translation & Multilingual", "Translate to English (-tr)", "-tr", "bool", "", False, "(ASR Feature) Translate recognized audio directly from source language to English text."),
            ("6. Translation & Multilingual", "Source Language (-sl)", "-sl", "str", "", False, "Explicitly specify the source audio language (overrides auto-detect)."),
            ("6. Translation & Multilingual", "Target Language (-tl)", "-tl", "str", "", False, "Specify the final output foreign language target for arbitrary translation models (like Canary)."),
            ("6. Translation & Multilingual", "Disable Punctuation (--no-punctuation)", "--no-punctuation", "bool", "", False, "Disable punctuation generation for models like Canary / Cohere."),
            ("6. Translation & Multilingual", "Punctuation Model (--punc-model)", "--punc-model", "file", "", False, "Post-processing punctuation recovery model: auto | firered | fullstop | punctuate-all."),
            ("6. Translation & Multilingual", "Truecase Model (--truecase-model)", "--truecase-model", "file", "", False, "Model to recover uppercase English letters for entirely lowercase results."),
            ("6. Translation & Multilingual", "Text Translation Input (--text)", "--text", "long_text", "", False, "(Text-to-Text Feature) Input plain text to be translated by models like m2m100."),
            ("6. Translation & Multilingual", "Translation Source Lang (--tr-sl)", "--tr-sl", "str", "", False, "Translator-stage source language, used alongside --text."),
            ("6. Translation & Multilingual", "Translation Target Lang (--tr-tl)", "--tr-tl", "str", "", False, "Translator-stage target language, used alongside --text."),
            ("6. Translation & Multilingual", "Translate Max Tokens (--translate-max-tokens)", "--translate-max-tokens", "int", "256", False, "Max output tokens cap for the pure text translation stage."),

            # ========================= 7. Diarization (Speaker Separation) =========================
            ("7. Diarization", "Enable Diarization (-di)", "-di", "bool", "", False, "Global switch: Separate audio according to the speaker."),
            ("7. Diarization", "Diarize Method (--diarize-method)", "--diarize-method", "str", "", False, "Algorithmic path: energy | xcorr | vad-turns | sherpa | pyannote | ecapa."),
            ("7. Diarization", "Diarize Embedder (--diarize-embedder)", "--diarize-embedder", "file", "off", False, "Speaker-embedding model used to cluster local tracks into globally stable speaker IDs."),
            ("7. Diarization", "Cluster Merge Threshold (--diarize-cluster-threshold)", "--diarize-cluster-threshold", "float", "0.50", False, "Cosine merge threshold for clustering (higher = more distinct/difficult to merge)."),
            ("7. Diarization", "Max Speakers (--diarize-max-speakers)", "--diarize-max-speakers", "int", "8", False, "Hard cap on cluster count for embedding algorithms."),
            ("7. Diarization", "Sherpa Binary (--sherpa-bin)", "--sherpa-bin", "file", "", False, "Path to the external sherpa-onnx-offline-speaker-diarization executable."),
            ("7. Diarization", "Sherpa Segment Model (--sherpa-segment-model)", "--sherpa-segment-model", "file", "", False, "Sherpa's Pyannote segmentation ONNX model."),
            ("7. Diarization", "Sherpa Embedding Model (--sherpa-embedding-model)", "--sherpa-embedding-model", "file", "", False, "Sherpa's speaker embedding feature extraction ONNX model."),

            # ========================= 8. VAD Silence Detection =========================
            ("8. VAD Silence Detection", "Enable VAD (--vad)", "--vad", "bool", "", False, "Crucial Optimization: Enable neural-network-based silence and non-speech skipping mechanism."),
            ("8. VAD Silence Detection", "VAD Model (-vm)", "-vm", "file", "", False, "VAD model path, or built-in alias ('firered', 'silero')."),
            ("8. VAD Silence Detection", "VAD Threshold (-vt)", "-vt", "float", "0.50", False, "VAD recognition threshold; probability must exceed this to count as human speech."),
            ("8. VAD Silence Detection", "Min Speech Duration (-vspd)", "-vspd", "int", "250", False, "Sounds shorter than this (ms) are considered noise or coughs and are erased."),
            ("8. VAD Silence Detection", "Min Silence Duration (-vsd)", "-vsd", "int", "100", False, "Pause between utterances exceeding this (ms) triggers a physical audio split."),
            ("8. VAD Silence Detection", "Max Speech Duration (-vmsd)", "-vmsd", "str", "", False, "Force cap on max continuous speech slice (seconds) to prevent OOM on extreme long sounds."),
            ("8. VAD Silence Detection", "VAD Speech Padding (-vp)", "-vp", "int", "30", False, "Padding buffer added to edges of VAD slices (ms) to prevent swallowing first syllables."),
            ("8. VAD Silence Detection", "VAD Samples Overlap (-vo)", "-vo", "float", "0.10", False, "Overlap context (seconds) shared between two separated consecutive audio blocks."),

            # ========================= 9. Streaming & Network Server =========================
            ("9. Streaming & Server", "HTTP Server Mode (--server)", "--server", "bool", "", False, "Run the program persistently as an HTTP server exposing OpenAI-compatible REST APIs."),
            ("9. Streaming & Server", "Bind IP (--host)", "--host", "str", "127.0.0.1", False, "Server bind network address (0.0.0.0 allows public access)."),
            ("9. Streaming & Server", "Listen Port (--port)", "--port", "int", "8080", False, "Server API listening port."),
            ("9. Streaming & Server", "WebSocket Port (--ws-port)", "--ws-port", "int", "-1", False, "Real-time WebSocket ASR streaming port (-1 to disable, 0 for HTTP port + 1)."),
            ("9. Streaming & Server", "Wyoming Port (--wyoming-port)", "--wyoming-port", "int", "", False, "Wyoming protocol TCP port compatible with Home Assistant Assist."),
            ("9. Streaming & Server", "Skip Warmup (--no-warmup)", "--no-warmup", "bool", "", False, "Skip startup virtual inference warmup (workaround for specific GPU driver crashes)."),
            ("9. Streaming & Server", "API Keys (--api-keys)", "--api-keys", "str", "", False, "Comma-separated server API authorization tokens."),
            ("9. Streaming & Server", "Streaming Mode (--stream)", "--stream", "bool", "", False, "Read raw s16le PCM audio stream real-time from stdin."),
            ("9. Streaming & Server", "Capture Microphone (--mic)", "--mic", "bool", "", False, "Capture data directly from the system default microphone (implies --stream)."),
            ("9. Streaming & Server", "Continuous Live Mode (--live)", "--live", "bool", "", False, "Continuous live transcription mode without interruption."),
            ("9. Streaming & Server", "Stream Step Size (--stream-step)", "--stream-step", "int", "3000", False, "Audio buffer chunk size in ms for streaming reception."),
            ("9. Streaming & Server", "Stream Context Window (--stream-length)", "--stream-length", "int", "10000", False, "Rolling context window cap (ms) kept in memory during streaming."),
            ("9. Streaming & Server", "Finalize on Silence (--stream-final-on-silence-ms)", "--stream-final-on-silence-ms", "int", "800", False, "Trailing silence (ms) that promotes a local partial sentence to a finalized text."),

            # ========================= 10. TTS Synthesis & Speech-to-Speech =========================
            ("10. TTS & S2S", "TTS Synthesize Text (--tts)", "--tts", "long_text", "", False, "Synthesize input TEXT to speech (requires a backend with CAP_TTS ability)."),
            ("10. TTS & S2S", "Play Audio Direct (--tts-play)", "--tts-play", "bool", "", False, "Play the synthesized audio directly through the default system speaker."),
            ("10. TTS & S2S", "Play Audio Device ID (--tts-play-device)", "--tts-play-device", "int", "-1", False, "Paired with play direct, specifies the hardware device index (default: -1)."),
            ("10. TTS & S2S", "TTS Output File (--tts-output)", "--tts-output", "str", "tts_output.wav", False, "Output WAV file path for the synthesized audio (default: tts_output.wav)."),
            ("10. TTS & S2S", "Speech-to-Speech Mode (--s2s)", "--s2s", "bool", "", False, "Speech-to-speech mode: input audio directly, output the LLM's vocal response."),
            ("10. TTS & S2S", "S2S Output File (--s2s-output)", "--s2s-output", "file", "", False, "Output WAV path for generated audio responses in S2S mode."),
            ("10. TTS & S2S", "Voice Clone/Reference (--voice)", "--voice", "file", "", False, "TTS Zero-shot cloning: Supply a reference WAV file or GGUF preset voice pack."),
            ("10. TTS & S2S", "Confirm Copyright (--i-have-rights)", "--i-have-rights", "bool", "", False, "[CRITICAL LEGAL] Required for WAV cloning; attests you hold consent/rights to the cloned speaker."),
            ("10. TTS & S2S", "Skip Spoken Disclaimer (--no-spoken-disclaimer)", "--no-spoken-disclaimer", "bool", "", False, "Skip the audible 'This is AI generated audio' disclaimer prefix on cloned output."),
            ("10. TTS & S2S", "Clone Reference Text (--ref-text)", "--ref-text", "long_text", "", False, "Provide the actual text spoken in the reference WAV audio to greatly boost clone realism."),
            ("10. TTS & S2S", "Natural Style Instruct (--instruct)", "--instruct", "long_text", "", False, "Natural-language description to guide the voice's emotion, speed, or speaking style."),
            ("10. TTS & S2S", "Codec Companion Model (--codec-model)", "--codec-model", "file", "", False, "Neural codec (Tokenizer) GGUF companion model required by TTS."),
            ("10. TTS & S2S", "Voice Profiles Dir (--voice-dir)", "--voice-dir", "dir", "", False, "Server: directory containing preset .wav / .gguf voice profiles to load."),
            ("10. TTS & S2S", "Max TTS Input Chars (--tts-max-input-chars)", "--tts-max-input-chars", "int", "4096", False, "Server: cap on single TTS request input synthesis length (0 = no cap)."),
            ("10. TTS & S2S", "G2P Dictionary (--g2p-dict)", "--g2p-dict", "file", "olaph", False, "G2P dictionary source (olaph, open-dict, or text file) to correct pronunciation."),
            ("10. TTS & S2S", "Attached Chat Model (--chat-model)", "--chat-model", "file", "", False, "Server: Attach a large text GGUF model to provide plain text chat completion interfaces."),
            
            # ========================= 11. Performance & Env Variables (KV & Tuning) =========================
            ("11. Performance & Env", "KV Cache Quantization", "ENV:CRISPASR_KV_QUANT", "str", "", False, "Specify context cache data type (f16, q8_0, q4_0). q4_0 saves 75% KV VRAM, highly recommended."),
            ("11. Performance & Env", "K Cache Quantization", "ENV:CRISPASR_KV_QUANT_K", "str", "", False, "Specify Key quantization level independently (Keys are fragile, recommend at least q8_0)."),
            ("11. Performance & Env", "V Cache Quantization", "ENV:CRISPASR_KV_QUANT_V", "str", "", False, "Specify Value quantization level independently (Values tolerate heavy loss, q4_0 is fine)."),
            ("11. Performance & Env", "KV Cache on CPU", "ENV:CRISPASR_KV_ON_CPU", "str", "", False, "Set to 1 to force context cache allocation into system memory when VRAM is exhausted."),
            ("11. Performance & Env", "GPU Offload Layers", "ENV:CRISPASR_N_GPU_LAYERS", "str", "", False, "Number of Transformer layers to place on GPU. Mix with CPU compute if VRAM is insufficient."),
            ("11. Performance & Env", "MMAP Memory Mapping", "ENV:CRISPASR_GGUF_MMAP", "str", "1", False, "Set to 1 to enable (default) or 0 to disable. Loads model via pointer mapping to save massive memory."),
            ("11. Performance & Env", "Qwen Codec Chunk Size", "ENV:QWEN3_TTS_CODEC_CHUNK", "str", "150", False, "Limit max parallel frames (default 150) for the codec to prevent OOM when synthesizing long text."),
            ("11. Performance & Env", "Qwen Codec on GPU", "ENV:QWEN3_TTS_CODEC_GPU", "str", "", False, "Set to 1 to accelerate Qwen3-TTS acoustic feature decoder using the GPU."),
            ("11. Performance & Env", "Qwen Codec on CPU", "ENV:QWEN3_TTS_CODEC_CPU", "str", "", False, "Set to 1 to forcefully offload codec out of the GPU into system RAM, preserving VRAM in extreme cases."),
            ("11. Performance & Env", "Qwen Skip Ref Decode", "ENV:QWEN3_TTS_SKIP_REF_DECODE", "str", "", False, "Set to 0 to disable optimization. Default is on, skipping redundant ref audio decodes to boost speed by 50%."),
            ("11. Performance & Env", "Chatterbox Batch Accel", "ENV:CRISPASR_CHATTERBOX_T3_CFG_B2", "str", "1", False, "Set to 1. Merges unconditional & conditional generation into a Batch=2 operation, significantly boosting GPU speed."),
            ("11. Performance & Env", "Cohere Legacy Attention", "ENV:CRISPASR_COHERE_LEGACY_SA", "str", "1", False, "Set to 1. Revert to legacy compatibility path if you experience severe performance degradation using Cohere models."),
        ]

        self.create_widgets()
        self.apply_config()

    def load_config(self):
        if os.path.exists(CONFIG_FILE):
            try:
                with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except: pass
        return {}

    def save_config_only(self):
        config_to_save = {}
        for flag, data in self.vars.items():
            if data["vtype"] == "long_text":
                val = data["val_widget"].get("1.0", "end-1c")
            else:
                val = data["val_widget"].get()
            config_to_save[flag] = {"use": data["use"].get(), "val": val}
            
        config_to_save["EXTRA_CLI"] = self.extra_cli_text.get("1.0", "end-1c")
        config_to_save["EXTRA_ENV"] = self.extra_env_text.get("1.0", "end-1c")
        try:
            with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
                json.dump(config_to_save, f, indent=4)
            messagebox.showinfo("State Dump Complete","All interactive panel configurations and environment mappings have been serialized to config.json!")
        except Exception as e:
            messagebox.showerror("Serialization Error", f"IO break: {str(e)}")

    def browse_file(self, widget, is_dir=False):
        if is_dir:
            path = filedialog.askdirectory(title="Select Target Directory")
        else:
            path = filedialog.askopenfilename(title="Select Target File")
        if path:
            widget.delete(0, tk.END)
            widget.insert(0, path)

    def create_widgets(self):
        # 顶部指挥按钮区
        top_frame = ttk.Frame(self.root, padding=10)
        top_frame.pack(fill=tk.X)
        self.save_btn = ttk.Button(top_frame, text="💾 Save Configuration Only (Do not run)", command=self.save_config_only)
        self.save_btn.pack(side=tk.LEFT, padx=5)
        self.run_btn = ttk.Button(top_frame, text="▶ Inject Config & Start Inference Backend", command=self.run_command, style="Accent.TButton")
        self.run_btn.pack(side=tk.LEFT, padx=5)
        ttk.Label(top_frame, text="* Unchecked checkboxes will be ignored by the engine. Column boundaries are resizable").pack(side=tk.LEFT, padx=10)

        # 构建主控 Tab 网格
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        self.tabs = {}
        
        for tab_name, label_name, flag, vtype, default_val, default_enable, help_text in self.param_definitions:
            if tab_name not in self.tabs:
                frame_wrap = ScrollableFrame(self.notebook)
                self.notebook.add(frame_wrap, text=tab_name)
                parent = frame_wrap.inner
                self.tabs[tab_name] = parent
                
                # 配置权重
                parent.columnconfigure(0, weight=0, minsize=240)
                parent.columnconfigure(1, weight=1, minsize=300)
                parent.columnconfigure(2, weight=2, minsize=350)
                
                ttk.Label(parent, text="[Enable Lock] Function Name (Flag / ENV)", font=('Arial', 9, 'bold')).grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
                ttk.Label(parent, text="Parameter Value / Long Text Input", font=('Arial', 9, 'bold')).grid(row=0, column=1, sticky=tk.W, padx=5, pady=5)
                ttk.Label(parent, text="Technical Principle Description", font=('Arial', 9, 'bold')).grid(row=0, column=2, sticky=tk.W, padx=5, pady=5)

            parent = self.tabs[tab_name]
            row_idx = parent.grid_size()[1]
            
            # [列0] 启用复选框与技术标签
            use_var = tk.BooleanVar(value=default_enable)
            chk = ttk.Checkbutton(parent, text=label_name, variable=use_var)
            chk.grid(row=row_idx, column=0, sticky=tk.W, padx=5, pady=5)
            
            # [列1] 参数值填入槽
            input_widgets = []
            if vtype == "bool":
                val_widget = ttk.Entry(parent)
                val_widget.insert(0, "[Boolean Trigger: Effective upon check]")
                val_widget.grid(row=row_idx, column=1, sticky=(tk.W, tk.E), padx=5, pady=5)
                val_widget.config(state=tk.DISABLED)
                
            elif vtype == "long_text":
                val_widget = tk.Text(parent, height=3, wrap="word", font=("Arial", 9))
                val_widget.insert("1.0", str(default_val))
                val_widget.grid(row=row_idx, column=1, sticky=(tk.W, tk.E), padx=5, pady=5)
                input_widgets.append(val_widget)
                
            elif vtype in ["file", "dir"]:
                frame_file = ttk.Frame(parent)
                val_widget = ttk.Entry(frame_file)
                val_widget.insert(0, str(default_val))
                val_widget.pack(side=tk.LEFT, fill=tk.X, expand=True)
                is_dir_flag = (vtype == "dir")
                btn = ttk.Button(frame_file, text="Open...", width=8, command=lambda v=val_widget, d=is_dir_flag: self.browse_file(v, d))
                btn.pack(side=tk.RIGHT, padx=2)
                frame_file.grid(row=row_idx, column=1, sticky=(tk.W, tk.E), padx=5, pady=5)
                input_widgets.extend([val_widget, btn])
                
            else:
                val_widget = ttk.Entry(parent)
                val_widget.insert(0, str(default_val))
                val_widget.grid(row=row_idx, column=1, sticky=(tk.W, tk.E), padx=5, pady=5)
                input_widgets.append(val_widget)
                
            # [列2] 中文深度解析只读框
            desc_text = tk.Text(parent, height=3 if vtype == "long_text" else 2, wrap="word", bg=self.root.cget("background"), bd=0, fg="#2A2A2A", font=("Arial", 9))
            desc_text.insert("1.0", help_text)
            desc_text.config(state=tk.DISABLED)
            desc_text.grid(row=row_idx, column=2, sticky=(tk.W, tk.E), padx=15, pady=5)

            # 加入数据字典阵列
            self.vars[flag] = {
                "use": use_var,
                "val_widget": val_widget,
                "vtype": vtype,
                "widgets": input_widgets
            }
            
            # 灰度联动事件（关锁则输入框失效）
            def _make_trace_func(f_key=flag):
                def _trace_toggle(*args):
                    is_on = self.vars[f_key]["use"].get()
                    new_state = tk.NORMAL if is_on else tk.DISABLED
                    for w in self.vars[f_key]["widgets"]:
                        w.config(state=new_state)
                return _trace_toggle
            
            use_var.trace_add("write", _make_trace_func())
            _make_trace_func()() 

        # ================= 终极越权操作区 (双轨输入) =================
        adv_frame = ttk.LabelFrame(self.root, text="Advanced Custom Injection Area", padding=10)
        adv_frame.pack(fill=tk.BOTH, expand=False, padx=10, pady=5)
        
        adv_frame.columnconfigure(0, weight=1)
        adv_frame.columnconfigure(1, weight=1)
        
        # 左侧：自定义 CLI 参数
        ttk.Label(adv_frame, text="Additional CLI Payload (Safely parses quotes, supports multi-line):").grid(row=0, column=0, sticky=tk.W)
        self.extra_cli_text = tk.Text(adv_frame, height=3, font=("Consolas", 9), wrap="word")
        self.extra_cli_text.grid(row=1, column=0, sticky=(tk.W, tk.E), padx=5, pady=5)
        
        # 右侧：自定义环境变量
        ttk.Label(adv_frame, text="Custom Environment Variables (KEY=VALUE format, one per line):").grid(row=0, column=1, sticky=tk.W)
        self.extra_env_text = tk.Text(adv_frame, height=3, font=("Consolas", 9), wrap="none")
        self.extra_env_text.grid(row=1, column=1, sticky=(tk.W, tk.E), padx=5, pady=5)

        # 底部侦听终端
        log_frame = ttk.LabelFrame(self.root, text="System Engine Feedback Monitor (stdout / stderr)", padding=5)
        log_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        self.log_text = tk.Text(log_frame, bg="#0E0E0E", fg="#4CAF50", font=("Consolas", 10), height=10)
        scrollbar = ttk.Scrollbar(log_frame, command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=scrollbar.set)
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    def apply_config(self):
        for flag, data in self.vars.items():
            if flag in self.config:
                saved = self.config[flag]
                if isinstance(saved, dict):
                    data["use"].set(saved.get("use", False))
                    v = str(saved.get("val", ""))
                    if data["vtype"] == "long_text":
                        data["val_widget"].delete("1.0", tk.END)
                        data["val_widget"].insert("1.0", v)
                    else:
                        data["val_widget"].delete(0, tk.END)
                        data["val_widget"].insert(0, v)
                        
        self.extra_cli_text.insert("1.0", self.config.get("EXTRA_CLI", ""))
        self.extra_env_text.insert("1.0", self.config.get("EXTRA_ENV", ""))

    def log(self, message):
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)

    def run_command(self):
        if not os.path.exists(EXE_PATH):
            messagebox.showerror("Engine Missing", f"Core engine not detected. Please place this console in the same directory:\n{EXE_PATH}")
            return
            
        self.save_config_only() # 执行前先静默备份

        cmd = [EXE_PATH]
        # 获取纯净操作系统环境变量副本，准备注入高阶参数
        env_dict = os.environ.copy()
        
        # 1. 组装预设面板的指令与环境
        for flag, data in self.vars.items():
            if data["use"].get():
                if data["vtype"] == "long_text":
                    input_val = data["val_widget"].get("1.0", "end-1c").strip()
                else:
                    input_val = data["val_widget"].get().strip()

                if flag.startswith("ENV:"):
                    env_key = flag.split("ENV:")[1]
                    if input_val:
                        env_dict[env_key] = input_val
                else:
                    if data["vtype"] == "bool":
                        cmd.append(flag)
                    else:
                        if input_val:
                            cmd.extend([flag, input_val])
                            
        # 2. 注入自定义环境变量 (KEY=VALUE)
        custom_env_lines = self.extra_env_text.get("1.0", "end-1c").strip().split('\n')
        for line in custom_env_lines:
            line = line.strip()
            if line and '=' in line and not line.startswith('#'):
                k, v = line.split('=', 1)
                env_dict[k.strip()] = v.strip()
                    
        # 3. 注入附加 CLI 参数 (使用 shlex 安全拆分引号)
        extra_cli_str = self.extra_cli_text.get("1.0", "end-1c").strip()
        if extra_cli_str:
            try:
                # 使用 shlex.split 保证即使参数里有带空格的引号字符串也能正确解析
                parsed_args = shlex.split(extra_cli_str)
                cmd.extend(parsed_args)
            except Exception as e:
                messagebox.showerror("Parameter Parsing Error", f"Unclosed quotes or syntax error in additional CLI parameters:\n{str(e)}")
                return

        self.log_text.delete(1.0, tk.END)
        # 将本次调用的临时环境变量过滤显示出来
        custom_envs = {k: v for k, v in env_dict.items() if k not in os.environ or os.environ[k] != v}
        env_str = " ".join([f"{k}=\"{v}\"" for k, v in custom_envs.items()])
        
        self.log(f"[*Engine Call Protocol Loaded*]\n[ENV] {env_str}\n[CMD] {' '.join(cmd)}\n" + "="*110)
        self.run_btn.config(state=tk.DISABLED, text="Connecting to Compute...")
        threading.Thread(target=self.execute_process, args=(cmd, env_dict), daemon=True).start()

    def execute_process(self, cmd, run_env):
        try:
            # 挂载自定义的环境变量执行子进程
            process = subprocess.Popen(
                cmd,
                env=run_env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, 
                text=True,
                encoding='utf-8',
                errors='replace', 
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
            )

            for line in iter(process.stdout.readline, ''):
                self.root.after(0, self.log, line.strip())

            process.wait()
            self.root.after(0, self.log, f"\n[Link Disconnected] Engine exited safely, return code {process.returncode}。")
        except Exception as e:
            self.root.after(0, self.log, f"\n[Fatal Breakpoint] Failed to launch core framework: {str(e)}")
        finally:
            self.root.after(0, lambda: self.run_btn.config(state=tk.NORMAL, text="▶ Inject configuration and start the inference backend"))

if __name__ == "__main__":
    root = tk.Tk()
    style = ttk.Style(root)
    if 'clam' in style.theme_names():
        style.theme_use('clam')
    app = CrispasrFullGUI(root)
    root.mainloop()