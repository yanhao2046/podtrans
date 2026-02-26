<div align="center">

[English](README.md) | 中文

</div>

# PodTrans

基于 FunASR 的本地播客转录工具，将音频文件转换为带词级时间戳的文字。

## 功能

- 中文语音识别（paraformer-zh）
- 自动标点恢复（ct-punc）
- 词级时间戳（fa-zh）
- 语义分段（基于标点 + 时长限制）
- 输出格式：JSON + SRT

## 快速开始

```bash
pip install funasr modelscope torch torchaudio ffmpeg-python

python podtrans_core.py podcast.mp3 ./output
```

## Python API

```python
from podtrans import transcribe_audio

result = transcribe_audio("podcast.mp3", output_dir="./output")
# -> {"json_path": "...", "srt_path": "...", "segments_count": 42, "duration": 3600.5}
```

## 输出格式

### JSON

```json
{
  "metadata": {"duration": 3600.5, "model": "paraformer-zh", "language": "zh"},
  "segments": [
    {
      "id": 0, "start": 0.0, "end": 3.5,
      "text": "...",
      "words": [{"word": "...", "start": 0.0, "end": 0.4}]
    }
  ],
  "full_text": "..."
}
```

### SRT

标准 SubRip 字幕格式，可用 Aegisub 等字幕编辑器打开。

## 已知限制

- **时间戳漂移**：FunASR 存在累积漂移（约 5s/10分钟），文字准确率不受影响，但时间戳在长音频中会逐渐偏移。
- **不适合精确剪辑**：如需精确时间戳（如删除口癖词），建议使用 [filler_detect](https://github.com/yanhao2046/filler_detect)，它采用 Qwen3+VAD 方案实现高精度对齐。
- 首次运行需下载约 1GB 模型。

## 关联项目

**[filler_detect](https://github.com/yanhao2046/filler_detect)** — 播客口癖识别与粗剪工具。

PodTrans 和 filler_detect 最初设计为两阶段管线（ASR + 口癖检测）。开发中发现 FunASR 时间戳漂移不适合精确剪辑，filler_detect 因此自建了 Qwen3 ASR 方案。两个项目现在各自独立：

| 工具 | 适用场景 | ASR 引擎 |
|------|----------|----------|
| **PodTrans** | 转录、字幕、全文检索 | FunASR (paraformer-zh) |
| **filler_detect** | 口癖剪辑、精确音频编辑 | Qwen3 + VAD |

两者的 JSON 输出格式兼容（相同的 segments + words 结构）。

## 环境要求

- Python 3.9+
- ffmpeg（系统依赖）
- 约 5GB 磁盘空间（模型）
- GPU 可选（支持 CUDA、Apple MPS、CPU 回退）

## 许可证

MIT
