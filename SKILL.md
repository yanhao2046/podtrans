---
name: podtrans
description: 播客音频ASR转录工具。将MP3/WAV音频通过本地FunASR模型转换为带词级时间戳的JSON和SRT字幕。当用户说"转录音频"、"语音转文字"、"播客转录"、"transcribe"时触发。
argument-hint: "[audio_file] [output_dir]"
---

# PodTrans - 播客ASR转录

将播客音频通过本地 FunASR 模型转录为带词级时间戳的结构化文本。

## 使用方式

```bash
# 转录单个文件
python3 ~/.claude/skills/podtrans/podtrans_core.py audio.mp3 ./output

# Python API
from podtrans import transcribe_audio
result = transcribe_audio("podcast.mp3", output_dir="./output")
```

## 功能

- 中文语音识别（paraformer-zh）
- 自动标点恢复（ct-punc）
- 词级时间戳（fa-zh）
- 按语义分段（标点切分 + 时长控制）
- 输出 JSON + SRT 两种格式

## 输出格式

- **JSON**: 包含 metadata、segments（含词级时间戳）、full_text
- **SRT**: 标准字幕格式，可用 Aegisub 等工具查看

## 已知限制

- FunASR 时间戳存在累积漂移（约 5s/10min），不适合需要精确时间戳的场景
- 如需高精度时间戳（如口癖剪辑），建议使用 [filler_detect](https://github.com/YourUsername/filler_detect) 的 Qwen3+VAD 方案
- 首次运行需下载模型（约 1GB）
