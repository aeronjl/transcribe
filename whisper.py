from .openai import OpenAIClient

client = OpenAIClient()
openai = client.get_openai()

def transcribe_audio_segment(audio_buffer):
    """
    Call Whisper transcription for a short sample of audio.
    """
    
    transcription = openai.audio.transcriptions.create(
        model="whisper-1",
        file=audio_buffer,
        language="en",
        response_format="verbose_json",
        temperature=0.2
    )
    
    return transcription.segments