from . import utils
from .openai import OpenAIClient

client = OpenAIClient()
openai = client.get_openai()

def transcribe_audio_segment(audio_segment):
    """
    Call Whisper transcription for a short sample of audio.
    """
    buffer = utils.buffer_audio(audio_segment) 
    transcription = openai.audio.transcriptions.create(
        model="whisper-1",
        file=buffer,
        language="en",
        response_format="verbose_json",
        temperature=0,
        prompt="The audio provided may have moments of silence, do not make up words to fill in extended times of silence."
        
    )    
    return transcription.segments