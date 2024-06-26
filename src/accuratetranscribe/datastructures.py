from typing import List, Dict
from dataclasses import dataclass
from datetime import timedelta


@dataclass
class WhisperSegment:
    id: int
    seek: int
    start: int
    end: int
    text: str
    tokens: List[int]
    temperature: float
    average_logprob: float
    compression_ratio: float
    no_speech_prob: float


@dataclass
class WhisperOutput:
    segments: List[WhisperSegment]

@dataclass
class TranscriptSegment:
    speaker: str
    content: str

@dataclass
class Transcript:
    segments: Dict[str, TranscriptSegment]
