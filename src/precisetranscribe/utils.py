"""This module provides utilities for transcription."""

import datetime
import ffmpeg
import numpy as np
import json
import os
from io import BytesIO
from pydub import AudioSegment
import tiktoken
from . import whisper, gpt
from .openai import OpenAIClient
from typing import Tuple, Optional
from contextlib import contextmanager
import uuid
import tempfile

encoding = tiktoken.get_encoding("cl100k_base")

@contextmanager
def temporary_file(suffix: Optional[str]):
    """Context manager for creating temporary files."""
    temp_dir = tempfile.gettempdir()
    temp_file = os.path.join(temp_dir, f"{uuid.uuid4()}{suffix or ''}")
    try:
        yield
    finally:
        if os.path.exists(temp_file):
            os.remove(temp_file)

def convert_to_wav(input_file):
    with temporary_file() as temp_input, temporary_file('.wav') as temp_output:
        # Write the input file to a temporary file
        with open(temp_input, 'wb') as f:
            f.write(input_file.getvalue())
            
        try:
            stream = ffmpeg.input(temp_input)
            stream = ffmpeg.output(stream, temp_output, acodec='pcm_s16le', ac=1, ar='16k')
            ffmpeg.run(stream, overwrite_output=True, capture_stdout=True, capture_stderr=True)
            
            # Read the output file
            with open(temp_output, 'rb') as f:
                return f.read()
        except ffmpeg.Error as e:
            print('stdout:', e.stdout.decode('utf8'))
            print('stderr:', e.stderr.decode('utf8'))
            return None
    
def segment_audio(wav_data, segment_duration: int) -> list[AudioSegment]:
    """
    Segments an audio file into smaller chunks.
    """
    def check_segment_end_time(proposed_end_time, max_total_duration):
        return proposed_end_time if proposed_end_time < max_total_duration else max_total_duration
    
    audio_to_segment = AudioSegment.from_wav(BytesIO(wav_data))
    audio_duration = len(audio_to_segment)
    n_segments = int(np.ceil(audio_duration / segment_duration))
    audio_segments = []
    for segment_index in range(n_segments):
        start_time = segment_index * segment_duration
        proposed_end_time = segment_duration + (segment_index * segment_duration)
        end_time = check_segment_end_time(proposed_end_time, audio_duration)
        if end_time - start_time < 100: # Whisper won't accept segments less than 100ms
            break
        audio_segments.append(audio_to_segment[start_time:end_time])
    
    return audio_segments

def save_whisper_output(combined_transcript_segments, filename):
    """
    
    """
    with open(f'{filename}_whisper_output.json', 'w') as json_file:
        json.dump(combined_transcript_segments, json_file)
    return None

def load_whisper_output(filepath):
    with open(f"{filepath}_whisper_output.json", "r") as f:
        whisper_output = json.load(f)
    return whisper_output

def save_processed_transcript(combined_processed_chunks, filename) -> None:
    """
    """
    with open(f'{filename}_final_output.json', 'w') as json_file:
        json.dump(combined_processed_chunks, json_file)
    return None

def convert_to_timestamp(seconds):
    td = datetime.timedelta(seconds=seconds)

    hours, remainder = divmod(td.total_seconds(), 3600)
    minutes,seconds = divmod(remainder, 60)

    timestamp = "{:02}:{:02}:{:02}".format(int(hours), int(minutes), int(seconds))
    return timestamp

def buffer_audio(audio):
    buffer = BytesIO()
    buffer.name = "buffer.wav"
    audio.export(buffer, format="wav")
    buffer.seek(0)
    return buffer

def renumber_transcribed_audio_segment(transcribed_audio_segment, number_to_start_from=0, time_to_start_from=0):
    for phrase_index, phrase in enumerate(transcribed_audio_segment):
        phrase['id'] = number_to_start_from + phrase_index + 1
        phrase['start'] = phrase['start'] + time_to_start_from
        phrase['end'] = phrase['end'] + time_to_start_from
    return transcribed_audio_segment

def chunk_transcript_to_token_limit(transcript_segments, token_limit=1200):
    text_buffer = ''
    chunks = []
    for index, segment in enumerate(transcript_segments):    
        text_buffer += segment['text']
        token_count = len(encoding.encode(text_buffer))
        
        if index == len(transcript_segments) - 1:
            chunks.append(text_buffer)
            text_buffer = ''
            break
        
        if token_count > token_limit:
            if text_buffer.strip().endswith('.'):
                chunks.append(text_buffer)
                text_buffer = ''
                    
    return chunks, len(chunks)

def generate_system_prompt(speakers=None):
    
    if speakers == None:
        speaker_prompt = "at least one respondent"
    elif speakers == 1:
        speaker_prompt = "one respondent"
    else:
        speaker_prompt = f"{speakers} respondents"
        
    system_prompt = f"""
    You are a helpful assistant whose job it is to label a transcript according to who is speaking.
    You will see a transcript from a conversation between an interviewer and {speaker_prompt}.
    Reorganise and label the transcript so it is clear who is speaking. Guess the name of the respondent from the context where possible.
    Remove filler words and phrases without changing the meaning of the transcript.
    Only add necessary punctuation such as periods, commas, and capitalization, and use only the context provided.
    Return your response as a JSON.
    """
    
    examples = """
    Example 1:

    Hi there, how are you, you know, doing today Emily? I'm um fine, thank you. Great. I'm going to show you some you know marketing materials. Is that okay? I mean, yes. No problem.

    {
        "1" : {
            "Speaker" : "Interviewer",
            "Content" : "Hi there, how are you doing today Emily?"
        },
        "2" : {
            "Speaker" : "Respondent 1 (Emily)",
            "Content" : "I'm fine, thank you!"
        },
        "3" : {
            "Speaker" : "Interviewer",
            "Content" : "Great. I'm going to show you some marketing materials. Is that okay?"
        }
        "4" : {
            "Speaker" : "Respondent 1 (Emily)",
            "Content" : "Yes. No problem."
        }
    }

    Example 2:

    It's hard because, you know, there are so many, um, things to consider. I mean, you know, it's not easy. I mean, it's not easy at all. I see. Thank you, Doctor. For our next exercise we're going to look at some headline statements. Take a look at these and tell me what you think.

    {
        "1" : {
            "Speaker" : "Respondent 1",
            "Content" : "It's hard because there are so many things to consider. It's not easy. It's not easy at all."
        },
        "2": {
            "Speaker" : "Interviewer",
            "Content" : "I see. Thank you, Doctor. For our next exercise we're going to look at some headline statements. Take a look at these and tell me what you think."
        }
    }
    """
    
    system_prompt = system_prompt + examples
    
    return system_prompt