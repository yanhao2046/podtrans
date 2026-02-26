"""
PodTrans - 播客ASR转录 Skill

功能：
- 本地ASR转录 (FunASR)
- 词级时间戳
- 按语义分段
- 输出JSON/SRT

用法：
    from skills.podtrans import transcribe_audio
    result = transcribe_audio("podcast.mp3", output_dir="./output")
"""

from pathlib import Path
from typing import Union, Dict, Optional


def transcribe_audio(
    audio_path: Union[str, Path],
    output_dir: Union[str, Path] = "./transcripts",
    model_size: str = "paraformer-zh",
    device: str = "auto"
) -> Dict:
    """
    转录音频文件为带时间戳的文字

    Args:
        audio_path: 音频文件路径 (mp3/wav/m4a)
        output_dir: 输出目录
        model_size: ASR模型 (paraformer-zh, SenseVoiceSmall, etc.)
        device: 运行设备 (cuda:0, cpu, auto)

    Returns:
        {
            "json_path": "...",
            "srt_path": "...",
            "segments_count": 42,
            "duration": 3600.5
        }
    """
    from .podtrans_core import PodcastTranscriber

    transcriber = PodcastTranscriber(
        model_size=model_size,
        device=device
    )

    result = transcriber.transcribe(audio_path)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    base_name = Path(audio_path).stem

    # 保存JSON
    json_path = output_dir / f"{base_name}.json"
    result.save_json(json_path)

    # 保存SRT
    srt_path = output_dir / f"{base_name}.srt"
    result.save_srt(srt_path)

    return {
        "json_path": str(json_path),
        "srt_path": str(srt_path),
        "segments_count": len(result.segments),
        "duration": result.duration
    }


def batch_transcribe(
    audio_dir: Union[str, Path],
    output_dir: Union[str, Path] = "./transcripts",
    pattern: str = "*.mp3",
    **kwargs
) -> list:
    """
    批量转录目录中的音频文件

    Args:
        audio_dir: 音频文件目录
        output_dir: 输出目录
        pattern: 文件匹配模式 (默认 *.mp3)
        **kwargs: 传递给 transcribe_audio 的其他参数

    Returns:
        转录结果列表
    """
    from .podtrans_core import PodcastTranscriber

    audio_dir = Path(audio_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    transcriber = PodcastTranscriber(**kwargs)

    audio_files = list(audio_dir.glob(pattern))
    results = []

    for audio_file in audio_files:
        print(f"\n[{len(results)+1}/{len(audio_files)}] {audio_file.name}")
        try:
            result = transcriber.transcribe(audio_file)

            base_name = audio_file.stem
            json_path = output_dir / f"{base_name}.json"
            srt_path = output_dir / f"{base_name}.srt"

            result.save_json(json_path)
            result.save_srt(srt_path)

            results.append({
                "file": str(audio_file),
                "json": str(json_path),
                "srt": str(srt_path),
                "segments": len(result.segments),
                "duration": result.duration
            })
        except Exception as e:
            print(f"  错误: {e}")
            results.append({
                "file": str(audio_file),
                "error": str(e)
            })

    return results


# 向后兼容
__all__ = ["transcribe_audio", "batch_transcribe", "PodcastTranscriber"]
