# PodTrans - 播客ASR转录工具 PRD

## 1. 项目概述

### 1.1 目标
本地播客音频转录工具，将 MP3/WAV 音频通过 FunASR 模型转换为带词级时间戳的结构化文本（JSON + SRT）。

### 1.2 定位
PodTrans 是一个**独立的 ASR 转录工具**，适用于：
- 播客内容转文字、生成字幕
- 语音内容的全文检索和归档
- 作为其他音频处理工具的 ASR 前端

> **注意**：PodTrans 最初设计为播客口癖剪辑管线的第一阶段（ASR转录）。在开发过程中，发现 FunASR 存在时间戳漂移问题（详见第 7 节），导致下游项目 [filler_detect](https://github.com/YourUsername/filler_detect) 最终采用了自带 ASR 的 Qwen3+VAD 方案。PodTrans 现在作为独立转录工具维护。

### 1.3 核心要求
- 纯本地运行，无需联网（模型首次下载除外）
- 支持中文播客场景（普通话 + 少量英文）
- 输出精确到词级的时间戳
- 文本按语义分段，便于阅读
- 支持 30-60 分钟的音频

---

## 2. 技术方案

### 2.1 ASR 模型组合

| 模型 | 配置 | 用途 |
|------|------|------|
| **paraformer-zh** | `model="paraformer-zh"` | 中文语音识别（非流式） |
| **fsmn-vad** | `vad_model="fsmn-vad"` | 语音活动检测，切分长音频 |
| **ct-punc** | `punc_model="ct-punc"` | 标点恢复 |
| **fa-zh** | `timestamp_model="fa-zh"` | 词级时间戳 |

**选择理由**：
- `paraformer-zh` 是阿里达摩院针对中文优化的模型，中文识别率高于 Whisper
- 非流式版本精度更高，适合离线处理
- `ct-punc` 解决中文 ASR 无标点问题
- `fa-zh` 提供词级时间戳

**关键参数**：
```python
vad_kwargs={"max_single_segment_time": 30000}  # VAD最大片段30秒
batch_size_s=60  # 动态batch
device="mps"  # 自动检测：cuda > mps > cpu
```

### 2.2 依赖

```bash
pip install funasr modelscope torch torchaudio ffmpeg-python
# 系统依赖：ffmpeg
```

---

## 3. 核心概念

### 3.1 FunASR 输出结构

```json
{
  "text": "大家好，欢迎收听。今天我们要聊的是...",
  "timestamp": [
    [380, 600],
    [600, 720]
  ]
}
```

- `text`: 完整转录文本（已由 ct-punc 添加标点）
- `timestamp`: 每个字符的时间戳 `[[start_ms, end_ms], ...]`
- **没有** `sentence_info` 字段，需要基于标点自行切分

### 3.2 标点恢复（ct-punc）

原始音频没有标点，ct-punc 根据上下文语义预测标点位置：
```
输入：["大家", "好", "欢迎", "收听"]
输出："大家好，欢迎收听。"
```

### 3.3 词级时间戳的价值

| 层级 | 精度 | 用途 |
|------|------|------|
| 片段时间戳 | 句子级 | 大致定位 |
| **词级时间戳** | 每个词 | 精确定位每个词的起止时间 |

词级时间戳使得下游工具可以精确操作单个词（如删除口癖词、高亮关键词）。

---

## 4. 分段策略

### 4.1 实现方式

1. **词级切分**：将 `timestamp` 与 `text` 逐字对齐
2. **标点切分**：遇到 `。？！.?!…` 时结束当前片段
3. **合并优化**：短片段（<3s）合并，长片段（>15s）强制切断

### 4.2 示例

```
原始文本："大家好，欢迎收听本期播客。今天我们聊AI。"

分段结果：
1. "大家好，欢迎收听本期播客。" (0.0s-3.5s)
2. "今天我们聊AI。" (3.8s-6.2s)
```

---

## 5. 输出格式

### 5.1 JSON（主输出）

```json
{
  "metadata": {
    "duration": 3600.5,
    "model": "paraformer-zh",
    "language": "zh",
    "processed_at": "2026-02-23T10:00:00",
    "segments_count": 42
  },
  "segments": [
    {
      "id": 0,
      "start": 0.0,
      "end": 3.5,
      "text": "大家好，欢迎收听本期播客。",
      "words": [
        {"word": "大", "start": 0.0, "end": 0.4},
        {"word": "家", "start": 0.4, "end": 0.9}
      ]
    }
  ],
  "full_text": "大家好，欢迎收听本期播客。..."
}
```

### 5.2 SRT（字幕格式）

```srt
1
00:00:00,000 --> 00:00:03,500
大家好，欢迎收听本期播客。
```

---

## 6. 实现文件

| 文件 | 说明 |
|------|------|
| `podtrans_core.py` | 核心转录模块（PodcastTranscriber 类） |
| `__init__.py` | 公开 API（transcribe_audio, batch_transcribe） |
| `requirements.txt` | Python 依赖 |
| `SKILL.md` | Claude Code Skill 注册 |

### 6.1 公开 API

```python
from podtrans import transcribe_audio, batch_transcribe

# 单文件转录
result = transcribe_audio("podcast.mp3", output_dir="./output")
# -> {"json_path": "...", "srt_path": "...", "segments_count": 42, "duration": 3600.5}

# 批量转录
results = batch_transcribe("./audio_dir", output_dir="./output", pattern="*.mp3")
```

### 6.2 命令行

```bash
python podtrans_core.py audio.mp3 ./output
```

---

## 7. 已知限制与经验教训

### 7.1 FunASR 时间戳漂移

**现象**：FunASR 的词级时间戳存在累积漂移，约 5 秒/10 分钟。

**影响**：
- 对转录文本本身无影响（文字准确率 >95%）
- 对 SRT 字幕的影响可接受（偏移几秒，人工阅读无问题）
- **对精确剪辑场景影响严重**（如按时间戳删除口癖词，会删错位置）

**发现过程**：
在开发下游口癖剪辑工具 filler_detect 时，发现使用 PodTrans 输出的时间戳进行音频剪辑，删除率几乎为 0%。经分析确认是 FunASR 时间戳漂移导致。filler_detect 最终改用 Qwen3+ForcedAligner 方案，实现了 100% 删除率。

**建议**：
- 如果只需要文字转录和字幕，PodTrans 完全够用
- 如果需要精确时间戳操作（剪辑、对齐），建议使用 filler_detect 的 Qwen3 方案

### 7.2 其他已知问题

| 问题 | 原因 | 状态 |
|------|------|------|
| `device="auto"` 报错 | FunASR 不支持 auto 参数 | ✅ 已修复（`_get_device()`） |
| segments 为空 | 误用 sentence_info 字段 | ✅ 已修复（改用 text+timestamp） |
| 缺少 torchaudio | requirements.txt 遗漏 | ⚠️ 需补充 |
| 首次下载模型慢 | 模型约 1GB | 提示用户预下载 |
| 长音频内存压力 | 模型 + 音频同时加载 | VAD 自动切分缓解 |

---

## 8. 与 filler_detect 的关系

PodTrans 和 [filler_detect](https://github.com/YourUsername/filler_detect) 是同一播客处理工具链中的两个独立项目：

```
┌──────────────────────────────────────────────────────┐
│ 播客处理工具链                                         │
│                                                       │
│  ┌─────────┐         ┌──────────────┐                │
│  │ PodTrans│         │ filler_detect│                │
│  │         │         │              │                │
│  │ FunASR  │         │ Qwen3 + VAD  │ ← 自带ASR      │
│  │ 转录    │         │ 口癖检测     │                │
│  │ JSON+SRT│         │ 音频剪辑     │                │
│  └─────────┘         └──────────────┘                │
│                                                       │
│  适用：转录、字幕      适用：口癖剪辑、精确操作         │
└──────────────────────────────────────────────────────┘
```

- **PodTrans**：通用 ASR 转录，适合生成字幕、全文检索
- **filler_detect**：专注口癖检测和剪辑，自带高精度 ASR（Qwen3），不依赖 PodTrans

两者的 JSON 输出格式兼容（segments + words 结构），理论上可以互相使用对方的输出。

---

## 9. POC 验证结果

### 9.1 测试环境
- 音频：10 分钟中文播客（44MB MP3）
- 设备：Apple MPS (MacBook Pro M1)
- 模型：paraformer-zh + fsmn-vad + ct-punc + fa-zh

### 9.2 性能数据

| 指标 | 结果 |
|------|------|
| 处理耗时 | 24.4 秒（RTF ≈ 0.04） |
| 生成片段 | 54 个（平均 11.1 秒） |
| JSON 大小 | 238KB |
| 识别准确率 | >95% |

---

## 10. 验收标准

### 功能验收

- [x] 支持 MP3 格式音频输入
- [x] 输出 JSON 包含完整 metadata
- [x] 每个 segment 包含词级时间戳
- [x] 分段符合"按标点 + 15秒上限"策略
- [x] 同步输出 SRT 格式文件
- [x] 纯本地运行

### 质量验收

- [x] 中文识别准确率 > 95%（清晰音频）
- [ ] 词级时间戳误差 < 0.3 秒 — **未达标**（存在累积漂移，详见第 7.1 节）
- [x] 分段长度主要分布在 5-15 秒区间

---

## 附录：术语表

| 术语 | 解释 |
|------|------|
| ASR | Automatic Speech Recognition，自动语音识别 |
| VAD | Voice Activity Detection，语音活动检测 |
| ct-punc | Chinese Text Punctuation，中文标点恢复模型 |
| fa-zh | Forced Alignment Chinese，中文强制对齐（时间戳） |
| FunASR | 阿里达摩院开源的语音识别框架 |
| SRT | SubRip Text，字幕文件格式 |
| RTF | Real-Time Factor，处理时间与音频时长的比值 |

---

*最后更新：2026-02-26 — 补充时间戳漂移限制，重新定位为独立工具*
