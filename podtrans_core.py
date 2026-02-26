#!/usr/bin/env python3
"""
Podcast Transcription Core Module (FunASR Version)
播客语音转录核心模块

基于 FunASR 框架：
- paraformer-zh: 中文语音识别
- ct-punc: 标点恢复
- fa-zh: 词级时间戳
- fsmn-vad: 语音活动检测
"""

import os
import re
import json
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple
from datetime import datetime


@dataclass
class WordTimestamp:
    """词级时间戳"""
    word: str
    start: float
    end: float


@dataclass
class TranscriptionSegment:
    """转录片段"""
    id: int
    start: float
    end: float
    text: str
    words: List[WordTimestamp]

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "start": self.start,
            "end": self.end,
            "text": self.text,
            "words": [{"word": w.word, "start": w.start, "end": w.end} for w in self.words]
        }

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass
class TranscriptionResult:
    """转录结果"""
    segments: List[TranscriptionSegment]
    full_text: str
    duration: float
    model: str
    language: str
    processed_at: str

    def to_dict(self) -> Dict:
        return {
            "metadata": {
                "duration": self.duration,
                "model": self.model,
                "language": self.language,
                "processed_at": self.processed_at,
                "segments_count": len(self.segments)
            },
            "segments": [s.to_dict() for s in self.segments],
            "full_text": self.full_text
        }

    def save_json(self, output_path: Path) -> Path:
        """保存为JSON文件"""
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
        return output_path

    def save_srt(self, output_path: Path) -> Path:
        """保存为SRT字幕格式"""
        def format_time(seconds: float) -> str:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            secs = int(seconds % 60)
            millis = int((seconds % 1) * 1000)
            return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

        with open(output_path, "w", encoding="utf-8") as f:
            for seg in self.segments:
                f.write(f"{seg.id + 1}\n")
                f.write(f"{format_time(seg.start)} --> {format_time(seg.end)}\n")
                f.write(f"{seg.text}\n\n")
        return output_path


class PodcastTranscriber:
    """播客转录器 - FunASR版本"""

    def __init__(
        self,
        model_size: str = "paraformer-zh",
        device: str = "auto",
        language: str = "zh"
    ):
        """
        初始化转录器

        Args:
            model_size: ASR模型 (paraformer-zh, SenseVoiceSmall)
            device: 运行设备 (cpu, cuda:0, auto)
            language: 语言代码 (zh, auto)
        """
        self.model_size = model_size
        self.device = device
        self.language = language
        self._model = None

    def _get_device(self) -> str:
        """获取实际设备"""
        if self.device == "auto":
            import torch
            if torch.cuda.is_available():
                return "cuda:0"
            elif torch.backends.mps.is_available():
                return "mps"
            else:
                return "cpu"
        return self.device

    def _load_model(self):
        """懒加载FunASR模型"""
        if self._model is None:
            try:
                from funasr import AutoModel
            except ImportError:
                raise ImportError(
                    "funasr not installed. "
                    "Run: pip install funasr"
                )

            device = self._get_device()
            print(f"Loading FunASR models...")
            print(f"  ASR: {self.model_size}")
            print(f"  VAD: fsmn-vad")
            print(f"  PUNC: ct-punc")
            print(f"  TIMESTAMP: fa-zh")
            print(f"  Device: {device}")

            # 根据模型选择配置
            if self.model_size == "paraformer-zh":
                self._model = AutoModel(
                    model="paraformer-zh",
                    vad_model="fsmn-vad",
                    punc_model="ct-punc",
                    timestamp_model="fa-zh",
                    vad_kwargs={"max_single_segment_time": 30000},
                    device=device
                )
            elif self.model_size == "SenseVoiceSmall":
                self._model = AutoModel(
                    model="SenseVoiceSmall",
                    vad_model="fsmn-vad",
                    punc_model="ct-punc",
                    timestamp_model="fa-zh",
                    vad_kwargs={"max_single_segment_time": 30000},
                    device=self.device
                )
            else:
                # 通用配置
                self._model = AutoModel(
                    model=self.model_size,
                    device=device
                )

        return self._model

    def transcribe(
        self,
        audio_path: Path,
        merge_strategy: str = "sentence_with_limit",
        max_segment_duration: float = 15.0
    ) -> TranscriptionResult:
        """
        转录音频文件

        Args:
            audio_path: 音频文件路径
            merge_strategy: 分段策略 (sentence_with_limit / raw)
            max_segment_duration: 最大片段时长（秒）

        Returns:
            TranscriptionResult 转录结果
        """
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        model = self._load_model()

        print(f"Transcribing: {audio_path.name}")

        # 执行转录
        result = model.generate(
            input=str(audio_path),
            batch_size_s=60,
            language=self.language if self.language != "auto" else None
        )

        if not result or len(result) == 0:
            raise RuntimeError("Transcription failed: empty result")

        # 解析FunASR输出
        raw_result = result[0]

        # 提取基本信息
        full_text = raw_result.get("text", "")
        timestamp = raw_result.get("timestamp", [])

        # 构建词级时间戳
        words = self._build_word_timestamps(full_text, timestamp)

        # 基于标点切分句子构建片段
        raw_segments = self._split_by_punctuation(words)

        # 应用分段策略
        if merge_strategy == "sentence_with_limit":
            segments = self._merge_segments_by_sentence(
                raw_segments,
                max_duration=max_segment_duration
            )
        else:
            segments = raw_segments

        # 重新编号
        for i, seg in enumerate(segments):
            seg.id = i

        # 计算总时长
        total_duration = segments[-1].end if segments else 0.0

        return TranscriptionResult(
            segments=segments,
            full_text=full_text,
            duration=round(total_duration, 3),
            model=self.model_size,
            language=self.language,
            processed_at=datetime.now().isoformat()
        )

    def _build_word_timestamps(
        self,
        text: str,
        timestamp: List[List[int]]
    ) -> List[WordTimestamp]:
        """
        从FunASR输出构建词级时间戳

        FunASR timestamp格式: [[start_ms, end_ms], [start_ms, end_ms], ...]
        """
        words = []

        # 简单分词（按字/词分割）
        # FunASR的timestamp通常对应字符级别
        chars = list(text)

        for i, (char, ts) in enumerate(zip(chars, timestamp)):
            if len(ts) >= 2:
                start_ms, end_ms = ts[0], ts[1]
                words.append(WordTimestamp(
                    word=char,
                    start=round(start_ms / 1000, 3),
                    end=round(end_ms / 1000, 3)
                ))

        return words

    def _split_by_punctuation(
        self,
        words: List[WordTimestamp]
    ) -> List[TranscriptionSegment]:
        """
        基于标点切分词列表，构建片段
        """
        if not words:
            return []

        segments = []
        current_words = []
        current_id = 0

        for word in words:
            current_words.append(word)

            # 遇到标点结束当前片段
            if word.word in ('。', '？', '！', '.', '?', '!', '…'):
                if current_words:
                    text = ''.join(w.word for w in current_words)
                    seg = TranscriptionSegment(
                        id=current_id,
                        start=current_words[0].start,
                        end=current_words[-1].end,
                        text=text,
                        words=current_words.copy()
                    )
                    segments.append(seg)
                    current_id += 1
                    current_words = []

        # 处理剩余未分段的词
        if current_words:
            text = ''.join(w.word for w in current_words)
            seg = TranscriptionSegment(
                id=current_id,
                start=current_words[0].start,
                end=current_words[-1].end,
                text=text,
                words=current_words.copy()
            )
            segments.append(seg)

        return segments

    def _build_raw_segments(
        self,
        sentences: List[Dict],
        all_words: List[WordTimestamp]
    ) -> List[TranscriptionSegment]:
        """
        从FunASR的sentence_info构建片段（备用方法）
        """
        segments = []
        word_idx = 0

        for sent_idx, sent in enumerate(sentences):
            text = sent.get("text", "")
            start = sent.get("start", 0) / 1000  # ms to s
            end = sent.get("end", 0) / 1000

            # 提取该句子的词
            sent_words = []
            char_count = len(text)

            for _ in range(char_count):
                if word_idx < len(all_words):
                    sent_words.append(all_words[word_idx])
                    word_idx += 1

            segments.append(TranscriptionSegment(
                id=sent_idx,
                start=round(start, 3),
                end=round(end, 3),
                text=text,
                words=sent_words
            ))

        return segments

    def _merge_segments_by_sentence(
        self,
        segments: List[TranscriptionSegment],
        max_duration: float = 15.0,
        min_duration: float = 3.0
    ) -> List[TranscriptionSegment]:
        """
        按句子合并片段，同时控制时长

        策略：
        1. 优先按句子边界（句号、问号、感叹号）切分
        2. 但单片段不超过 max_duration
        3. 超过则强制切分
        """
        if not segments:
            return []

        merged = []
        current = None

        for seg in segments:
            if current is None:
                current = TranscriptionSegment(
                    id=0,
                    start=seg.start,
                    end=seg.end,
                    text=seg.text,
                    words=seg.words.copy()
                )
                continue

            # 检查是否应合并
            new_duration = seg.end - current.start
            ends_with_punctuation = self._is_sentence_end(current.text)

            should_merge = (
                new_duration < max_duration and
                not ends_with_punctuation
            )

            if should_merge:
                # 合并
                current.end = seg.end
                current.text += seg.text
                current.words.extend(seg.words)
            else:
                # 保存当前，开始新片段
                merged.append(current)
                current = TranscriptionSegment(
                    id=len(merged),
                    start=seg.start,
                    end=seg.end,
                    text=seg.text,
                    words=seg.words.copy()
                )

        if current:
            merged.append(current)

        return merged

    def _is_sentence_end(self, text: str) -> bool:
        """判断是否句子结尾"""
        text = text.strip()
        if not text:
            return False
        sentence_endings = ('。', '？', '！', '.', '?', '!', '…')
        return text.endswith(sentence_endings)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("用法: python podtrans_core.py <audio_file> [output_dir]")
        sys.exit(1)

    audio_file = Path(sys.argv[1])
    output_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("./transcripts")

    transcriber = PodcastTranscriber()
    result = transcriber.transcribe(audio_file)

    output_dir.mkdir(exist_ok=True)
    base_name = audio_file.stem

    result.save_json(output_dir / f"{base_name}.json")
    result.save_srt(output_dir / f"{base_name}.srt")

    print(f"\n转录完成:")
    print(f"  片段数: {len(result.segments)}")
    print(f"  时长: {result.duration:.1f}s")
    print(f"  输出: {output_dir}")
