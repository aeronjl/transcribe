import os
from .utils import convert_audio_to_wav, segment_audio, transcribe_audio_segments

def transcribe_audio(input_filepath, save=True):
    filename, extension = os.path.splitext(input_filepath)
    print(f"Filename: {filename}, Extension: {extension}")
    if not extension:
        raise ValueError(f"No file extension found for {input_filepath}")
    
    convert_audio_to_wav(filename, extension)
    audio_segments = segment_audio(f"{filename}.wav", 100000)
    combined_transcript_segments, combined_processed_chunks = transcribe_audio_segments(audio_segments, filename=filename)
    
    return combined_transcript_segments, combined_processed_chunks