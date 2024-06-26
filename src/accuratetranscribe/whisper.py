from pydub import AudioSegment

from . import utils
from .openai import OpenAIClient
from .datastructures import WhisperOutput

client = OpenAIClient()
openai = client.get_openai()


def transcribe_audio_segment(audio_segment: AudioSegment) -> WhisperOutput:
    """
    Call Whisper transcription for a short sample of audio.
    """
    buffer = utils.buffer_audio(audio_segment)
    transcription: WhisperOutput = openai.audio.transcriptions.create(
        model="whisper-1",
        file=buffer,
        language="en",
        response_format="verbose_json",
        temperature=0.1,
    ).segments
    return transcription

