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

encoding = tiktoken.get_encoding("cl100k_base")

def convert_audio_to_wav(filename, filetype):
    stream = ffmpeg.input(f"{filename}{filetype}")
    stream = ffmpeg.output(stream, f"{filename}.wav")
    ffmpeg.run(stream, quiet=True, overwrite_output=True)
    return None
    
def segment_audio(audio_file, segment_duration):
    def check_segment_end_time(proposed_end_time, max_total_duration):
        return proposed_end_time if proposed_end_time < max_total_duration else max_total_duration
    audio_to_segment = AudioSegment.from_file(audio_file, format="wav")
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

def convert_to_timestamp(seconds):
    td = datetime.timedelta(seconds=seconds)

    hours, remainder = divmod(td.total_seconds(), 3600)
    minutes,seconds = divmod(remainder, 60)

    timestamp = "{:02}:{:02}:{:02}".format(int(hours), int(minutes), int(seconds))
    return timestamp

def buffer_audio_segment_and_transcribe(audio_segment, index):
    buffer = BytesIO()
    buffer.name = "buffer.wav"
    audio_segment.export(buffer, format="wav")
    buffer.seek(0)
    transcribed_audio_segment = whisper.transcribe_audio_segment(buffer)
    if index == 0:
        last_text_segment_id = transcribed_audio_segment[-1]['id']
        pass
    else:
        for index, text_segment in enumerate(transcribed_audio_segment):
            text_segment['id'] = last_text_segment_id + index + 1
        last_text_segment_id = transcribed_audio_segment[-1]['id']
    return transcribed_audio_segment

def chunk_transcript_to_token_limit(transcript_segments, token_limit=1200):
    for index, segment in enumerate(transcript_segments):
        if index == 0:
                text_buffer = ''
                chunks = []
            
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
    for index, chunk in enumerate(chunks):
        if index == 0:
            completion = gpt.process_transcription(chunk, system_prompt)
        else:
            annotated_system_prompt = system_prompt + f"""
            ---
            You are continuing on from a previous transcription which ended as follows:

            "{chunk_items[-4][0]}" : {chunk_items[-4][1]},
            "{chunk_items[-3][0]}" : {chunk_items[-3][1]},
            "{chunk_items[-2][0]}" : {chunk_items[-2][1]},
            "{chunk_items[-1][0]}" : {chunk_items[-1][1]}
            """
            completion = gpt.process_transcription(chunk, annotated_system_prompt)
        print(f"Processed chunk at index {index}")
        print(f"Finish reason: {completion.choices[0].finish_reason}")
        
        processed_chunk = json.loads(completion.choices[0].message.content)
        processed_chunks.append(processed_chunk)
        chunk_items = list(processed_chunk.items())
        
    combined_processed_chunks =  {key: value for d in processed_chunks for key, value in d.items()}
    return combined_processed_chunks

def transcribe_audio_segments(audio_segments, filename, save=True):
    n_segments = len(audio_segments)
    print(f"Transcribing {n_segments} audio segments. Estimated time: {n_segments * 10} seconds.")
    
    transcribed_audio_segments = []
    for index, audio_segment in enumerate(audio_segments):
        transcribed_audio_segment = buffer_audio_segment_and_transcribe(audio_segment, index)
        transcribed_audio_segments.append(transcribed_audio_segment)

    for index, segment in enumerate(transcribed_audio_segments):
        if index == 0:
            last_end_time = 0
        else:
            previous_index = index - 1
            last_end_time = transcribed_audio_segments[previous_index][-1]['end']
        for phrase_index, phrase in enumerate(segment):
            transcribed_audio_segments[index][phrase_index]['start'] = phrase['start'] + last_end_time
            transcribed_audio_segments[index][phrase_index]['end'] = phrase['end'] + last_end_time
            
    combined_transcript_segments = [item for sublist in transcribed_audio_segments for item in sublist]
    
    for index, segment in enumerate(combined_transcript_segments):
        combined_transcript_segments[index]['start'] = convert_to_timestamp(segment['start'])
        combined_transcript_segments[index]['end'] = convert_to_timestamp(segment['end'])
        
    if save:
        with open(f'{filename}_whisper_output.json', 'w') as json_file:
            json.dump(combined_transcript_segments, json_file)
            
    chunks, n_transcript_chunks = chunk_transcript_to_token_limit(combined_transcript_segments, token_limit=1200)    
    
    print(f"Processing {n_transcript_chunks} transcript chunks. Estimated time: {n_transcript_chunks * 30} seconds.")  
    
    combined_processed_chunks = process_transcription(chunks)
    if save:
        with open(f'{filename}_final_output.json', 'w') as json_file:
            json.dump(combined_processed_chunks, json_file)
    
    return combined_transcript_segments, combined_processed_chunks
    
    
    