from .openai import OpenAIClient
import time

client = OpenAIClient()
openai = client.get_openai()

def transcribe_audio_segment(audio_buffer):
    """
    Call Whisper transcription for a short sample of audio.
    """
    
    t = time.time()
    transcription = openai.audio.transcriptions.create(
        model="whisper-1",
        file=audio_buffer,
        language="en",
        response_format="verbose_json",
        temperature=0.2
    )
    elapsed = time.time() - t
    print(f"Transcribed audio segment. Elapsed: {elapsed} seconds.")
    
    return transcription.segments