# AI Chat Companion (AI 语音聊天伴侣)

这是一个基于 Python 的命令行 AI 语音聊天伴侣应用。它集成了阿里云 DashScope (通义千问) 的实时语音识别 (STT)、大语言模型 (LLM) 和实时语音合成 (TTS) 技术，提供流畅的端到端语音交互体验。

## ✨ 主要特性

*   **实时语音交互**: 支持流式语音识别和流式语音合成，实现低延迟的自然对话。
*   **智能打断 (Barge-in)**: 在 AI 说话时，用户可以随时插话打断，AI 会立即停止当前回复并倾听新的输入。
*   **长期记忆**: 自动总结对话历史并持久化保存到本地，AI 能够记住之前的聊天内容，实现连贯的长期陪伴。
*   **声音克隆与设计**:
    *   支持通过音频文件克隆指定声音。
    *   支持通过文本提示词 (Prompt) 设计全新的声音。
*   **自定义人设**: 可以随时通过命令修改 AI 的系统提示词 (System Prompt)，改变 AI 的性格和行为方式。

## 🛠️ 技术栈

*   **语言**: Python 3
*   **核心依赖**:
    *   `dashscope`: 阿里云大模型 API SDK
    *   `pyaudio`: 音频输入输出处理
    *   `requests`: HTTP 请求处理
*   **使用的模型**:
    *   STT: `qwen3-asr-flash-realtime`
    *   LLM: `qwen3.5-flash` (支持多模态)
    *   TTS: `qwen3-tts-vd-realtime-2026-01-15`
    *   Voice: `qwen-voice-enrollment`, `qwen-voice-design`

## 🚀 快速开始

### 1. 环境准备

确保你的系统已安装 Python 3，并且安装了麦克风和扬声器设备。
由于使用了 `pyaudio`，在某些系统上可能需要先安装底层音频库（例如 Ubuntu 上的 `portaudio19-dev`，macOS 上的 `brew install portaudio`）。

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 配置 API Key

本项目依赖阿里云 DashScope API。你需要获取一个 API Key，并将其设置为环境变量 `DASHSCOPE_API_KEY`。

**Windows (CMD):**
```cmd
set DASHSCOPE_API_KEY=你的_API_KEY
```

**Windows (PowerShell):**
```powershell
$env:DASHSCOPE_API_KEY="你的_API_KEY"
```

**Linux / macOS:**
```bash
export DASHSCOPE_API_KEY="你的_API_KEY"
```

### 4. 运行应用

```bash
python main.py
```

## 💬 使用说明

启动应用后，程序会自动开启麦克风监听，你可以直接对着麦克风说话与 AI 进行交流。

此外，你还可以在命令行输入以下指令进行控制：

*   `/exit`: 安全退出程序，并保存当前对话记忆。
*   `/system [prompt]`: 更新 AI 的系统提示词（人设）。例如：`/system 你是一个傲娇的二次元少女`。
*   `/voice-clone [path]`: 从本地音频文件克隆声音。例如：`/voice-clone ./my_voice.wav`。
*   `/voice-design [prompt]|[preview_text]`: 通过文本描述设计声音。`preview_text` 是可选的试听文本。例如：`/voice-design 一个温柔的年轻女性声音|你好，我是你的新助手`。
*   **直接输入文本**: 如果你不方便说话，也可以直接在命令行输入文本按回车发送给 AI。

## 📁 项目结构

*   `main.py`: 程序主入口，负责协调各个模块和处理命令行交互。
*   `stt_module.py`: 语音识别模块，处理麦克风录音和流式 STT。
*   `llm_module.py`: 大语言模型模块，处理对话生成和上下文总结。
*   `tts_module.py`: 语音合成模块，处理流式 TTS 和音频播放。
*   `voice_module.py`: 声音定制模块，处理声音克隆和声音设计。
*   `memory_module.py`: 记忆管理模块，负责对话历史的保存和加载。
*   `settings.json`: 本地配置文件，保存当前的声音 ID 和系统提示词等设置。
*   `memory/`: 存放历史对话记忆 (JSON 格式) 的目录。
*   `voice/`: 存放声音设计生成的试听音频文件的目录。
