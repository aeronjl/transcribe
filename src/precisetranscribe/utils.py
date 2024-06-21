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
        
def split_filepath(filepath: str) -> Tuple[str, str]:
    filename, extension = os.path.splitext(filepath)
    print(f"Filename: {filename}, Extension: {extension}")
    return filename, extension  

def convert_input_to_wav(input_file):
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
    
    wav_data = wav_data.seek(0)
    
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

def save_final_output(combined_processed_chunks, filename):
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
        
def load_system_prompt():
    module_path = os.path.abspath(__file__)
    prompt_path = os.path.join(os.path.dirname(module_path), "data", "prompt.txt")
    
    with open(prompt_path, "r") as f:
        system_prompt = f.read()
    return system_prompt

def process_transcription(chunks):
    system_prompt = load_system_prompt()
    processed_chunks = []
    
    prompt = system_prompt
    for index, chunk in enumerate(chunks):
        completion = gpt.process_transcription(chunk, prompt)
        processed_chunk = json.loads(completion.choices[0].message.content)
        processed_chunks.append(processed_chunk)
        chunk_items = list(processed_chunk.items())
        
        print(f"Processed chunk at index {index}")
        print(f"Finish reason: {completion.choices[0].finish_reason}")
        
        if index > 0:
            prompt = system_prompt + f"""
            ---
            You are continuing on from a previous transcription which ended as follows:

            "{chunk_items[-4][0]}" : {chunk_items[-4][1]},
            "{chunk_items[-3][0]}" : {chunk_items[-3][1]},
            "{chunk_items[-2][0]}" : {chunk_items[-2][1]},
            "{chunk_items[-1][0]}" : {chunk_items[-1][1]}
            """
    
    # Combine the processed chunks        
    combined_processed_chunks =  {key: value for d in processed_chunks for key, value in d.items()}
    return combined_processed_chunks