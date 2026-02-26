# PodTrans

Local podcast transcription tool using FunASR. Converts audio files to text with word-level timestamps.

## Features

- Chinese speech recognition (paraformer-zh)
- Automatic punctuation (ct-punc)
- Word-level timestamps (fa-zh)
- Semantic segmentation (punctuation-based + duration limits)
- Output: JSON + SRT

## Quick Start

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

## Output Format

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

Standard SubRip subtitle format, viewable in Aegisub or any subtitle editor.

## Known Limitations

- **Timestamp drift**: FunASR has cumulative timestamp drift (~5s per 10 minutes). Text accuracy is unaffected, but timestamps become less reliable for longer audio.
- **Not suitable for precise editing**: If you need accurate timestamps for audio editing (e.g., removing filler words), use [filler_detect](https://github.com/YourUsername/filler_detect) which uses Qwen3+VAD for high-precision alignment.
- First run downloads ~1GB of models.

## Related Project

**[filler_detect](https://github.com/YourUsername/filler_detect)** — Podcast filler word detection and rough-cut tool.

PodTrans and filler_detect were originally designed as a two-stage pipeline (ASR + filler detection). During development, FunASR's timestamp drift made it unsuitable for precise audio editing, so filler_detect built its own Qwen3-based ASR pipeline. Both projects now work independently:

| Tool | Best For | ASR Engine |
|------|----------|------------|
| **PodTrans** | Transcription, subtitles, full-text search | FunASR (paraformer-zh) |
| **filler_detect** | Filler word removal, precise audio editing | Qwen3 + VAD |

Their JSON output formats are compatible (same segments + words structure).

## Requirements

- Python 3.9+
- ffmpeg (system dependency)
- ~5GB disk space for models
- GPU optional (supports CUDA, Apple MPS, CPU fallback)

## License

MIT
