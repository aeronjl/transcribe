import time

from . import utils
from .openai import OpenAIClient

client = OpenAIClient()
openai = client.get_openai()

def transcribe_audio_segment(audio_segment):
    """
    Call Whisper transcription for a short sample of audio.
    """
    
    buffer = utils.buffer_audio(audio_segment) 
    
    t = time.time()
    transcription = openai.audio.transcriptions.create(
        model="whisper-1",
        file=buffer,
        language="en",
        response_format="verbose_json",
        temperature=0.2
    )
    elapsed = time.time() - t
    print(f"Transcribed audio segment. Elapsed: {elapsed} seconds.")
    
    return transcription.segments