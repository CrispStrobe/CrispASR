import os
import json
import shlex  # 新增：用于安全解析带引号的命令行参数
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
        self.root.title("CrispASR 专业控制台 (多轨环境注入版)")
        self.root.geometry("1300x900")
        
        self.config = self.load_config()
        self.vars = {} 
        
        # 字段说明: (所属栏目, 参数功能名(带英文), 命令行标志(ENV:开头表示环境变量), 数据类型, 默认值, 是否默认勾选, 详细解释)
        self.param_definitions = [
            # ========================= 1. 通用与核心设置 =========================
            ("1. 通用与核心", "帮助 (--help)", "-h", "bool", "", False, "显示帮助信息并退出。"),
            ("1. 通用与核心", "版本 (--version)", "--version", "bool", "", False, "打印构建版本信息、Git SHA及支持的后端并退出。"),
            ("1. 通用与核心", "诊断 (--diagnostics)", "--diagnostics", "bool", "", False, "输出完整的诊断信息（构建、环境、GPU枚举）并退出。"),
            ("1. 通用与核心", "主模型路径 (-m)", "-m", "file", "auto", False, "核心模型路径。填写 'auto' 将根据所选后端自动下载默认模型。"),
            ("1. 通用与核心", "推理后端 (--backend)", "--backend", "str", "qwen3", False, "强制指定特定后端（如 whisper, parakeet, canary, qwen3）。"),
            ("1. 通用与核心", "音频文件 (-f)", "-f", "file", "", False, "输入的音频源文件路径。"),
            ("1. 通用与核心", "语言 (-l)", "-l", "str", "auto", False, "输入音频的语种代码（ISO-639-1，如 en, zh）。填写 'auto' 为自动检测。"),
            ("1. 通用与核心", "线程数 (-t)", "-t", "int", "4", False, "计算时使用的 CPU 线程数量。"),
            ("1. 通用与核心", "处理器数 (-p)", "-p", "int", "1", False, "计算时并行的处理器状态数量。"),
            ("1. 通用与核心", "设备 ID (-dev)", "-dev", "int", "0", False, "分配的 GPU 设备 ID（默认：0卡）。"),
            ("1. 通用与核心", "禁用 GPU (-ng)", "-ng", "bool", "", False, "强制禁用 GPU，使用纯 CPU 进行推理。"),
            ("1. 通用与核心", "GPU 后端 (--gpu-backend)", "--gpu-backend", "str", "auto", False, "强制指定 GPU 后端架构：cuda | vulkan | metal | cpu。"),
            ("1. 通用与核心", "开启 Flash Attn (-fa)", "-fa", "bool", "", False, "启用 Flash Attention 加速技术，降低显存占用并提升速度。"),
            ("1. 通用与核心", "禁用 Flash Attn (-nfa)", "-nfa", "bool", "", False, "强制禁用 Flash Attention（用于老旧硬件兼容）。"),
            ("1. 通用与核心", "自动下载 (--auto-download)", "--auto-download", "bool", "", False, "发现缺少依赖模型时静默自动下载，不再提示。"),
            ("1. 通用与核心", "缓存目录 (--cache-dir)", "--cache-dir", "dir", "", False, "覆盖模型自动下载的默认缓存目录（默认 ~/.cache/crispasr/）。"),
            ("1. 通用与核心", "HF 仓库拉取 (-hfr)", "-hfr", "str", "", False, "从指定的 HuggingFace 仓库拉取模型 (OWNER/REPO[:FILE])。"),
            ("1. 通用与核心", "HF 目标文件 (-hff)", "-hff", "str", "", False, "配合 -hfr，指明 HF 仓库内的特定文件名。"),
            ("1. 通用与核心", "OpenVINO设备 (-oved)", "-oved", "str", "CPU", False, "the OpenVINO device used for encode inference (使用 OpenVINO 进行编码推理时的设备)。"),

            # ========================= 2. 输出与格式 =========================
            ("2. 输出与格式", "输出 TXT (-otxt)", "-otxt", "bool", "", False, "将转录结果输出为无时间戳的纯文本文件。"),
            ("2. 输出与格式", "输出 SRT (-osrt)", "-osrt", "bool", "", False, "将转录结果输出为标准 SRT 字幕文件。"),
            ("2. 输出与格式", "输出 VTT (-ovtt)", "-ovtt", "bool", "", False, "将转录结果输出为 WebVTT 字幕文件。"),
            ("2. 输出与格式", "输出 CSV (-ocsv)", "-ocsv", "bool", "", False, "将结果输出为 CSV 格式表格（包含起始时间、结束时间和文本）。"),
            ("2. 输出与格式", "输出 JSON (-oj)", "-oj", "bool", "", False, "将结果输出为 JSON 格式文件。"),
            ("2. 输出与格式", "输出完整 JSON (-ojf)", "-ojf", "bool", "", False, "在 JSON 中包含逐词时间戳 (words) 和 Token 数据等完整特征。"),
            ("2. 输出与格式", "输出 LRC (-olrc)", "-olrc", "bool", "", False, "输出带时间戳的 LRC 动态歌词文件。"),
            ("2. 输出与格式", "输出卡拉OK脚本 (-owts)", "-owts", "bool", "", False, "输出用于生成卡拉 OK 高亮视频的底层脚本文件。"),
            ("2. 输出与格式", "卡拉OK字体路径 (-fp)", "-fp", "file", "/System/Library/Fonts/Supplemental/Courier New Bold.ttf", False, "卡拉 OK 视频所使用的等宽字体绝对路径。"),
            ("2. 输出与格式", "输出文件前缀 (-of)", "-of", "str", "", False, "自定义输出文件路径与名称（不要带扩展名）。"),
            ("2. 输出与格式", "最大分段字符 (-ml)", "-ml", "int", "0", False, "每个字幕显示的最高字符长度（0 为不限制）。"),
            ("2. 输出与格式", "按标点断句 (-sp)", "-sp", "bool", "", False, "遇到句末标点符号强制切分字幕行，保持字幕易读性。"),
            ("2. 输出与格式", "按词切分 (-sow)", "-sow", "bool", "", False, "基于词汇边界而不是 Token 边界进行切分。"),
            ("2. 输出与格式", "静默控制台 (-np)", "-np", "bool", "", False, "除了最终结果，抑制所有进度与 stderr 输出。"),
            ("2. 输出与格式", "彩色高亮 (-pc)", "-pc", "bool", "", False, "在控制台输出中根据 Token 置信度使用不同颜色进行高亮。"),
            ("2. 输出与格式", "隐藏时间戳 (-nt)", "-nt", "bool", "", False, "控制台打印结果时隐藏前缀的时间戳标识。"),

            # ========================= 3. 分段与热词 =========================
            ("3. 分段与热词", "时间偏移 (-ot)", "-ot", "int", "0", False, "处理音频的起始时间偏移量（毫秒）。"),
            ("3. 分段与热词", "处理总时长 (-d)", "-d", "int", "0", False, "限制仅处理指定的音频总时长（毫秒）。"),
            ("3. 分段与热词", "盲切块长度 (-ck)", "-ck", "int", "30", False, "禁用 VAD 时，模型强制回退的固定音频切块时长（秒）。"),
            ("3. 分段与热词", "盲切块重叠 (--chunk-overlap)", "--chunk-overlap", "float", "3.0", False, "切块边界的重叠上下文时长（秒），缓解吞字。"),
            ("3. 分段与热词", "LCS 重叠纠删 (--lcs-dedup)", "--lcs-dedup", "str", "auto", False, "跨切块边界的亚词级最长公共子串 (LCS) 去重 (auto|on|off)。"),
            ("3. 分段与热词", "LCS 最小触发长 (--lcs-min-length)", "--lcs-min-length", "int", "1", False, "触发 LCS 纠删所需的最小重复长度。"),
            ("3. 分段与热词", "热词列表 (--hotwords)", "--hotwords", "long_text", "", False, "逗号分隔的热词列表，用于增强领域专有名词的识别率（支持后缀提升如 Name^3.0）。"),
            ("3. 分段与热词", "热词文件 (--hotwords-file)", "--hotwords-file", "file", "", False, "从本地文本文件读取热词（每行一个）。"),
            ("3. 分段与热词", "热词权重 (--hotwords-boost)", "--hotwords-boost", "float", "2.0", False, "热词 Token 的对数概率增强倍数（默认 2.0）。"),

            # ========================= 4. 采样与解码 =========================
            ("4. 采样与解码", "采样温度 (-tp)", "-tp", "float", "0.00", False, "解码采样温度（0 = 贪婪解码，最准；>0 启用多项式采样以提升多样性）。"),
            ("4. 采样与解码", "随机种子 (--seed)", "--seed", "int", "0", False, "采样随机数种子。固定种子可确保相同提示产生比特级一致的音频/文本结果。"),
            ("4. 采样与解码", "波束宽度 (-bs)", "-bs", "int", "5", False, "集束搜索 (Beam Search) 的宽度（默认：Whisper 5，其他 1）。"),
            ("4. 采样与解码", "最大生成 Token (-n)", "-n", "int", "512", False, "限制 LLM 后端单次生成的最大 Token 数量。"),
            ("4. 采样与解码", "频率惩罚 (--frequency-penalty)", "--frequency-penalty", "float", "0.00", False, "惩罚自回归模型生成的重复 Token，打断复读现象。"),
            ("4. 采样与解码", "Parakeet解码器 (--parakeet-decoder)", "--parakeet-decoder", "str", "tdt", False, "选择具体的解码路线 (ctc | tdt | maes)。"),
            ("4. 采样与解码", "词级概率阈值 (-wt)", "-wt", "float", "0.01", False, "词级别时间戳的判定概率阈值。"),
            ("4. 采样与解码", "解码失败熵阈值 (-et)", "-et", "float", "2.40", False, "判定解码失败并触发回退的熵阈值。"),
            ("4. 采样与解码", "非语音抑制 (-sns)", "-sns", "bool", "", False, "强制抑制/过滤无意义的非语音 Token。"),
            ("4. 采样与解码", "正则词过滤 (--suppress-regex)", "--suppress-regex", "str", "", False, "利用正则表达式匹配并过滤特定生成的词汇。"),
            ("4. 采样与解码", "GBNF 语法指导 (--grammar)", "--grammar", "file", "", False, "提供 GBNF 语法文件，强制约束大模型的输出结构。"),
            ("4. 采样与解码", "初始提示词 (--prompt)", "--prompt", "long_text", "", False, "提供前置上下文提示词，引导模型识别语境或专有名词。"),

            # ========================= 5. 对齐与语种 =========================
            ("5. 对齐与语种", "仅探测语言 (-dl)", "-dl", "bool", "", False, "自动检测输入音频语种后即刻退出。"),
            ("5. 对齐与语种", "LID 语种后端 (--lid-backend)", "--lid-backend", "str", "whisper", False, "音频语种识别 (LID) 提供者：whisper | silero | ecapa | firered。"),
            ("5. 对齐与语种", "LID 探测模型 (--lid-model)", "--lid-model", "file", "", False, "LID 语种识别模型路径（默认自动下载 ggml-tiny.bin）。"),
            ("5. 对齐与语种", "转录文本 LID (--lid-on-transcript)", "--lid-on-transcript", "str", "", False, "ASR 后处理：对生成的纯文本再次进行语种识别复核。"),
            ("5. 对齐与语种", "CTC 对齐模型 (-am)", "-am", "file", "", False, "针对缺乏原生词级时间戳的大模型，外挂 CTC 对齐器 GGUF。"),
            ("5. 对齐与语种", "强制 CTC 对齐 (-falign)", "-falign", "bool", "", False, "即使主模型自带时间戳，也强制使用外挂 CTC 结果覆写。"),
            ("5. 对齐与语种", "禁用自动对齐 (--no-auto-aligner)", "--no-auto-aligner", "bool", "", False, "阻止 Canary 等后端在请求时间戳时自动隐式挂载对齐器。"),
            ("5. 对齐与语种", "DTW 对齐计算 (-dtw)", "-dtw", "file", "", False, "计算 DTW 级超精细时间戳的模型路径。"),

            # ========================= 6. 翻译与多语言 =========================
            ("6. 翻译与多语言", "翻译至英文 (-tr)", "-tr", "bool", "", False, "（ASR功能）将识别出的音频源语言直接翻译成英文文字。"),
            ("6. 翻译与多语言", "听力源语言 (-sl)", "-sl", "str", "", False, "明确指定源音频的语言（覆盖自动探测）。"),
            ("6. 翻译与多语言", "翻译目标语言 (-tl)", "-tl", "str", "", False, "针对具备任意翻译能力的模型（如 Canary），指定最终输出的外语语种。"),
            ("6. 翻译与多语言", "禁用标点符号 (--no-punctuation)", "--no-punctuation", "bool", "", False, "禁用 Canary / Cohere 等模型的标点符号生成。"),
            ("6. 翻译与多语言", "标点修补模型 (--punc-model)", "--punc-model", "file", "", False, "后处理标点恢复模型：auto | firered | fullstop | punctuate-all。"),
            ("6. 翻译与多语言", "大写恢复模型 (--truecase-model)", "--truecase-model", "file", "", False, "针对全小写结果恢复英文字母大写的模型。"),
            ("6. 翻译与多语言", "文本翻译输入 (--text)", "--text", "long_text", "", False, "（Text-to-Text 功能）输入需要被 m2m100 等纯文本翻译模型翻译的文字内容。"),
            ("6. 翻译与多语言", "文本源语种 (--tr-sl)", "--tr-sl", "str", "", False, "配合 --text，明确输入文本的语种。"),
            ("6. 翻译与多语言", "文本目标语种 (--tr-tl)", "--tr-tl", "str", "", False, "配合 --text，明确要翻译至的目标语种。"),
            ("6. 翻译与多语言", "翻译最大 Token (--translate-max-tokens)", "--translate-max-tokens", "int", "256", False, "限制纯文本翻译阶段输出的最大 Token 数量。"),

            # ========================= 7. 说话人分离 (Diarization) =========================
            ("7. 说话人分离", "启用分离 (-di)", "-di", "bool", "", False, "通用分离开关：将音频按说话人切分。"),
            ("7. 说话人分离", "分离算法 (--diarize-method)", "--diarize-method", "str", "", False, "算法路线：energy | xcorr | vad-turns | sherpa | pyannote | ecapa。"),
            ("7. 说话人分离", "声纹聚类模型 (--diarize-embedder)", "--diarize-embedder", "file", "off", False, "用于提取说话人全局声纹特征的嵌入模型（确保长音频 ID 一致性）。"),
            ("7. 说话人分离", "聚类合并阈值 (--diarize-cluster-threshold)", "--diarize-cluster-threshold", "float", "0.50", False, "声纹余弦相似度合并阈值（越高越难判定为同一人）。"),
            ("7. 说话人分离", "最大说话人数 (--diarize-max-speakers)", "--diarize-max-speakers", "int", "8", False, "硬性限制声纹聚类的最大人群数量。"),
            ("7. 说话人分离", "Sherpa 二进制 (--sherpa-bin)", "--sherpa-bin", "file", "", False, "外部调用的 sherpa-onnx-offline-speaker-diarization 可执行程序路径。"),
            ("7. 说话人分离", "Sherpa 分割图 (--sherpa-segment-model)", "--sherpa-segment-model", "file", "", False, "Sherpa 使用的 Pyannote 语音切分 ONNX 模型。"),
            ("7. 说话人分离", "Sherpa 识人图 (--sherpa-embedding-model)", "--sherpa-embedding-model", "file", "", False, "Sherpa 使用的提取声纹特征 ONNX 模型。"),

            # ========================= 8. VAD 静音检测 =========================
            ("8. VAD 静音检测", "启用 VAD 检测 (--vad)", "--vad", "bool", "", False, "极其关键的优化选项：开启基于神经网络的静音与非语音跳过机制。"),
            ("8. VAD 静音检测", "VAD 核心模型 (-vm)", "-vm", "file", "", False, "VAD 模型路径，或内置标识符（'firered', 'silero'）。"),
            ("8. VAD 静音检测", "人声置信度 (-vt)", "-vt", "float", "0.50", False, "VAD 识别阈值，超过该概率才判定为人类语音。"),
            ("8. VAD 静音检测", "最小语音时长 (-vspd)", "-vspd", "int", "250", False, "低于此毫秒数的声音被视为底噪或咳嗽抹除。"),
            ("8. VAD 静音检测", "断句静音长 (-vsd)", "-vsd", "int", "100", False, "说话间隙的停顿超过此毫秒即判定为两句话进行物理切割。"),
            ("8. VAD 静音检测", "最大切片容忍 (-vmsd)", "-vmsd", "str", "", False, "强制设定最长连续语音切片时长（秒），防止极端长音造成 OOM。"),
            ("8. VAD 静音检测", "语音边缘填充 (-vp)", "-vp", "int", "30", False, "VAD 切割时，向边缘多保留的缓冲时长（毫秒），防止吞咬字首。"),
            ("8. VAD 静音检测", "相邻上下文黏连 (-vo)", "-vo", "float", "0.10", False, "输入模型前，两个被切开的相邻音频块互相包含的重叠时长（秒）。"),

            # ========================= 9. 流式与网络服务 =========================
            ("9. 流式与网络", "HTTP 服务器模式 (--server)", "--server", "bool", "", False, "将程序驻留为后端服务，暴露兼容 OpenAI 标准的 REST API 接口。"),
            ("9. 流式与网络", "绑定 IP (--host)", "--host", "str", "127.0.0.1", False, "服务器监听绑定的网络地址（0.0.0.0 允许公网访问）。"),
            ("9. 流式与网络", "监听端口 (--port)", "--port", "int", "8080", False, "服务器 API 监听的服务端口。"),
            ("9. 流式与网络", "WebSocket 端口 (--ws-port)", "--ws-port", "int", "-1", False, "实时 WebSocket ASR 流媒体端口（-1 禁用，0 为 HTTP 端口 +1）。"),
            ("9. 流式与网络", "Wyoming 端口 (--wyoming-port)", "--wyoming-port", "int", "", False, "兼容 Home Assistant Assist 的专用 TCP 端口。"),
            ("9. 流式与网络", "跳过显卡预热 (--no-warmup)", "--no-warmup", "bool", "", False, "服务器启动时跳过虚拟推理预热，规避某些缺陷 GPU 驱动导致死机。"),
            ("9. 流式与网络", "授权密钥 (--api-keys)", "--api-keys", "str", "", False, "API 访问权限令牌，用逗号分隔多个 Key。"),
            ("9. 流式与网络", "实时流媒体 (--stream)", "--stream", "bool", "", False, "挂起并从 stdin 实时读取原始 s16le PCM 音频数据流。"),
            ("9. 流式与网络", "捕获麦克风 (--mic)", "--mic", "bool", "", False, "直接从系统默认麦克风捕获数据（隐式开启 --stream）。"),
            ("9. 流式与网络", "无限直播同传 (--live)", "--live", "bool", "", False, "不间断实时语音识别模式。"),
            ("9. 流式与网络", "流式接收步长 (--stream-step)", "--stream-step", "int", "3000", False, "流式接收时音频缓冲切块的大小（毫秒）。"),
            ("9. 流式与网络", "流式滚存窗口 (--stream-length)", "--stream-length", "int", "10000", False, "流式处理时保留在内存中的过去音频上下文长度上限（毫秒）。"),
            ("9. 流式与网络", "触发最终定案 (--stream-final-on-silence-ms)", "--stream-final-on-silence-ms", "int", "800", False, "超过此毫秒的静默即宣告当前局部句段（partial）转为确定定稿（final）。"),

            # ========================= 10. TTS 合成与语音互译 =========================
            ("10. TTS与S2S", "TTS 合成文本 (--tts)", "--tts", "long_text", "", False, "将输入的文字内容合成为语音（需要加载具备 CAP_TTS 能力的后端）。"),
            ("10. TTS与S2S", "扬声器直出 (--tts-play)", "--tts-play", "bool", "", False, "合成后直接通过系统默认设备播放音频。"),
            ("10. TTS与S2S", "指定播放设备 (--tts-play-device)", "--tts-play-device", "int", "-1", False, "配合直出功能，指定硬件设备 ID（默认 -1）。"),
            ("10. TTS与S2S", "TTS 输出文件 (--tts-output)", "--tts-output", "str", "tts_output.wav", False, "合成音频的输出文件路径（默认保存为 tts_output.wav）。"),
            ("10. TTS与S2S", "语音至语音互译 (--s2s)", "--s2s", "bool", "", False, "Speech-to-speech 模式：直接输入音频，输出大模型的回复音频。"),
            ("10. TTS与S2S", "互译输出文件 (--s2s-output)", "--s2s-output", "file", "", False, "S2S 模式下，生成的对面回复音频保存路径。"),
            ("10. TTS与S2S", "音色克隆/参考 (--voice)", "--voice", "file", "", False, "TTS 零样本克隆：传入真人参考 WAV 文件，或 GGUF 预置语音包。"),
            ("10. TTS与S2S", "版权确认授权 (--i-have-rights)", "--i-have-rights", "bool", "", False, "【关键法规】使用真实 WAV 进行音色克隆时必须勾选，声明你拥有克隆授权。"),
            ("10. TTS与S2S", "跳过免责口播 (--no-spoken-disclaimer)", "--no-spoken-disclaimer", "bool", "", False, "禁止系统在生成的录音开头添加“这是 AI 生成音频”的口头免责声明。"),
            ("10. TTS与S2S", "克隆参考文本 (--ref-text)", "--ref-text", "long_text", "", False, "提供参考 WAV 音频中实际朗读的文本，极大幅度提升克隆逼真度。"),
            ("10. TTS与S2S", "自然语言风格指令 (--instruct)", "--instruct", "long_text", "", False, "通过自然语言向模型描述声音的情感、语速或说话风格设定。"),
            ("10. TTS与S2S", "伴生编解码模型 (--codec-model)", "--codec-model", "file", "", False, "TTS 依赖的神经编解码器（Codec / Tokenizer）伴飞模型 GGUF。"),
            ("10. TTS与S2S", "音巢词库目录 (--voice-dir)", "--voice-dir", "dir", "", False, "作为服务器时，存放并加载大量预置 .wav / .gguf 音色的文件夹路径。"),
            ("10. TTS与S2S", "防爆字数上限 (--tts-max-input-chars)", "--tts-max-input-chars", "int", "4096", False, "服务器模式下，单次 TTS 请求允许合成的极限字符数。"),
            ("10. TTS与S2S", "发音拼读教案 (--g2p-dict)", "--g2p-dict", "file", "olaph", False, "G2P 字典来源（olaph, open-dict 或自定义文本），纠正发音。"),
            ("10. TTS与S2S", "外挂聊天模型 (--chat-model)", "--chat-model", "file", "", False, "在服务器模式中挂载大型文本 GGUF 模型，提供纯文本闲聊接口。"),
            
            # ========================= 11. 性能与环境变量 (KV & Tuning) =========================
            ("11. 性能与环境", "KV 缓存量化", "ENV:CRISPASR_KV_QUANT", "str", "", False, "指定上下文缓存的数据类型 (f16, q8_0, q4_0)。q4_0 可节省 75% 的 KV 显存，极推荐。"),
            ("11. 性能与环境", "K 缓存独立量化", "ENV:CRISPASR_KV_QUANT_K", "str", "", False, "单独指定 Key 的量化等级（Key 较脆弱，建议至少 q8_0）。"),
            ("11. 性能与环境", "V 缓存独立量化", "ENV:CRISPASR_KV_QUANT_V", "str", "", False, "单独指定 Value 的量化等级（Value 容错高，可承受 q4_0）。"),
            ("11. 性能与环境", "KV 溢写至 CPU", "ENV:CRISPASR_KV_ON_CPU", "str", "", False, "填 1 开启。在显存耗尽时，将缓存强制分配在系统内存中。"),
            ("11. 性能与环境", "GPU 卸载层数", "ENV:CRISPASR_N_GPU_LAYERS", "str", "", False, "指定放在 GPU 上的 Transformer 层数。不足以装下整个模型时可进行 CPU 混合运算。"),
            ("11. 性能与环境", "MMAP 内存映射", "ENV:CRISPASR_GGUF_MMAP", "str", "1", False, "填 1 开启（默认），填 0 关闭。通过指针映射载入模型，节省巨量内存。"),
            ("11. 性能与环境", "Qwen 编解码分块", "ENV:QWEN3_TTS_CODEC_CHUNK", "str", "150", False, "限制编解码最大并行帧数（默认150）。将其调小可解决合成长文时显存暴涨 OOM 问题。"),
            ("11. 性能与环境", "Qwen 编解码 GPU", "ENV:QWEN3_TTS_CODEC_GPU", "str", "", False, "填 1 开启。让 Qwen3-TTS 的声学特征解码器使用 GPU 进行加速。"),
            ("11. 性能与环境", "Qwen 编解码 CPU", "ENV:QWEN3_TTS_CODEC_CPU", "str", "", False, "填 1 开启。强行将编解码移出显卡转交系统内存，极端情况保全显存。"),
            ("11. 性能与环境", "Qwen 跳过参考解码", "ENV:QWEN3_TTS_SKIP_REF_DECODE", "str", "", False, "填 0 禁用优化。默认开启，可跳过无意义的参考音频解码耗时，提速 50%。"),
            ("11. 性能与环境", "Chatterbox 批处加速", "ENV:CRISPASR_CHATTERBOX_T3_CFG_B2", "str", "1", False, "填 1 开启。将无条件与条件生成合并为 Batch=2 运算，大幅提升 GPU 推理速度。"),
            ("11. 性能与环境", "Cohere 旧版注意力", "ENV:CRISPASR_COHERE_LEGACY_SA", "str", "1", False, "填 1 开启。若使用 Cohere 模型时性能严重劣化卡顿，开启此项退回兼容路径。"),
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
            messagebox.showinfo("状态存根完成", "所有交互面板的配置数据与环境映射已被序列化到 config.json！")
        except Exception as e:
            messagebox.showerror("序列化错误", f"IO 阻断: {str(e)}")

    def browse_file(self, widget, is_dir=False):
        if is_dir:
            path = filedialog.askdirectory(title="映射目标目录")
        else:
            path = filedialog.askopenfilename(title="绑定目标文件")
        if path:
            widget.delete(0, tk.END)
            widget.insert(0, path)

    def create_widgets(self):
        # 顶部指挥按钮区
        top_frame = ttk.Frame(self.root, padding=10)
        top_frame.pack(fill=tk.X)
        self.save_btn = ttk.Button(top_frame, text="💾 仅序列化配置 (不执行)", command=self.save_config_only)
        self.save_btn.pack(side=tk.LEFT, padx=5)
        self.run_btn = ttk.Button(top_frame, text="▶ 注入配置并启动推理后端", command=self.run_command, style="Accent.TButton")
        self.run_btn.pack(side=tk.LEFT, padx=5)
        ttk.Label(top_frame, text="* 未启用（未勾选）的复选框将被内核逻辑静默忽略。各列边界可自由拉伸。").pack(side=tk.LEFT, padx=10)

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
                
                ttk.Label(parent, text="【启用锁】 功能名 (Flag / ENV)", font=('微软雅黑', 9, 'bold')).grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
                ttk.Label(parent, text="参数值/长文本输入区", font=('微软雅黑', 9, 'bold')).grid(row=0, column=1, sticky=tk.W, padx=5, pady=5)
                ttk.Label(parent, text="技术原理解析", font=('微软雅黑', 9, 'bold')).grid(row=0, column=2, sticky=tk.W, padx=5, pady=5)

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
                val_widget.insert(0, "[布尔型触发器：勾选即刻生效]")
                val_widget.grid(row=row_idx, column=1, sticky=(tk.W, tk.E), padx=5, pady=5)
                val_widget.config(state=tk.DISABLED)
                
            elif vtype == "long_text":
                val_widget = tk.Text(parent, height=3, wrap="word", font=("微软雅黑", 9))
                val_widget.insert("1.0", str(default_val))
                val_widget.grid(row=row_idx, column=1, sticky=(tk.W, tk.E), padx=5, pady=5)
                input_widgets.append(val_widget)
                
            elif vtype in ["file", "dir"]:
                frame_file = ttk.Frame(parent)
                val_widget = ttk.Entry(frame_file)
                val_widget.insert(0, str(default_val))
                val_widget.pack(side=tk.LEFT, fill=tk.X, expand=True)
                is_dir_flag = (vtype == "dir")
                btn = ttk.Button(frame_file, text="打开...", width=8, command=lambda v=val_widget, d=is_dir_flag: self.browse_file(v, d))
                btn.pack(side=tk.RIGHT, padx=2)
                frame_file.grid(row=row_idx, column=1, sticky=(tk.W, tk.E), padx=5, pady=5)
                input_widgets.extend([val_widget, btn])
                
            else:
                val_widget = ttk.Entry(parent)
                val_widget.insert(0, str(default_val))
                val_widget.grid(row=row_idx, column=1, sticky=(tk.W, tk.E), padx=5, pady=5)
                input_widgets.append(val_widget)
                
            # [列2] 中文深度解析只读框
            desc_text = tk.Text(parent, height=3 if vtype == "long_text" else 2, wrap="word", bg=self.root.cget("background"), bd=0, fg="#2A2A2A", font=("微软雅黑", 9))
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
        adv_frame = ttk.LabelFrame(self.root, text="高级自定义注入区 (Advanced Injection)", padding=10)
        adv_frame.pack(fill=tk.BOTH, expand=False, padx=10, pady=5)
        
        adv_frame.columnconfigure(0, weight=1)
        adv_frame.columnconfigure(1, weight=1)
        
        # 左侧：自定义 CLI 参数
        ttk.Label(adv_frame, text="附加 CLI 载荷 (安全解析引号，支持换行):").grid(row=0, column=0, sticky=tk.W)
        self.extra_cli_text = tk.Text(adv_frame, height=3, font=("Consolas", 9), wrap="word")
        self.extra_cli_text.grid(row=1, column=0, sticky=(tk.W, tk.E), padx=5, pady=5)
        
        # 右侧：自定义环境变量
        ttk.Label(adv_frame, text="自定义环境变量 (KEY=VALUE 格式，每行一个):").grid(row=0, column=1, sticky=tk.W)
        self.extra_env_text = tk.Text(adv_frame, height=3, font=("Consolas", 9), wrap="none")
        self.extra_env_text.grid(row=1, column=1, sticky=(tk.W, tk.E), padx=5, pady=5)

        # 底部侦听终端
        log_frame = ttk.LabelFrame(self.root, text="系统内核反馈监控 (stdout / stderr)", padding=5)
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
            messagebox.showerror("内核迷失", f"未探测到核心引擎，请将本控制台置于同级目录:\n{EXE_PATH}")
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
                messagebox.showerror("参数解析错误", f"附加 CLI 参数的引号未能闭合或存在语法错误:\n{str(e)}")
                return

        self.log_text.delete(1.0, tk.END)
        # 将本次调用的临时环境变量过滤显示出来
        custom_envs = {k: v for k, v in env_dict.items() if k not in os.environ or os.environ[k] != v}
        env_str = " ".join([f"{k}=\"{v}\"" for k, v in custom_envs.items()])
        
        self.log(f"[*引擎呼叫协议已装填*]\n[ENV] {env_str}\n[CMD] {' '.join(cmd)}\n" + "="*110)
        self.run_btn.config(state=tk.DISABLED, text="高能运算接入中...")
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
            self.root.after(0, self.log, f"\n[链路断开] 内核安全退出，返回代码 {process.returncode}。")
        except Exception as e:
            self.root.after(0, self.log, f"\n[致命断点] 无法拉起内核框架: {str(e)}")
        finally:
            self.root.after(0, lambda: self.run_btn.config(state=tk.NORMAL, text="▶ 注入配置并启动推理后端"))

if __name__ == "__main__":
    root = tk.Tk()
    style = ttk.Style(root)
    if 'clam' in style.theme_names():
        style.theme_use('clam')
    app = CrispasrFullGUI(root)
    root.mainloop()