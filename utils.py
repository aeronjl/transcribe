"""
"""

import argparse
import datetime
import ffmpeg
import numpy as np
import json
import os
from io import BytesIO
from openai import OpenAI
from pydub import AudioSegment
import tiktoken

client = OpenAI()
encoding = tiktoken.get_encoding("cl100k_base")

def convert_audio_to_wav(filename, filetype):
    stream = ffmpeg.input(f"{filename}{filetype}")
    stream = ffmpeg.output(stream, f"{filename}.wav")
    ffmpeg.run(stream, overwrite_output=True)
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


def transcribe_audio_segment(audio_buffer):
    """
    Call Whisper transcription for a short sample of audio.
    """
    
    transcription = client.audio.transcriptions.create(
        model="whisper-1",
        file=audio_buffer,
        language="en",
        response_format="verbose_json",
        temperature=0.2
    )
    
    return transcription.segments

def convert_to_timestamp(seconds):
    td = datetime.timedelta(seconds=seconds)

    hours, remainder = divmod(td.total_seconds(), 3600)
    minutes,seconds = divmod(remainder, 60)

    timestamp = "{:02}:{:02}:{:02}".format(int(hours), int(minutes), int(seconds))
    return timestamp

def transcribe_audio_segments(audio_segments, filename, save=True):
    transcribed_audio_segments = []
    for index, audio_segment in enumerate(audio_segments):
        buffer = BytesIO()
        buffer.name = "buffer.wav"
        audio_segment.export(buffer, format="wav")
        buffer.seek(0)
        transcribed_audio_segment = transcribe_audio_segment(buffer)
        if index == 0:
            last_text_segment_id = transcribed_audio_segment[-1]['id']
            pass
        else:
            for index, text_segment in enumerate(transcribed_audio_segment):
                text_segment['id'] = last_text_segment_id + index + 1
            last_text_segment_id = transcribed_audio_segment[-1]['id']
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
            
    for index, segment in enumerate(combined_transcript_segments):
        if index == 0:
            text_buffer = ''
            chunks = []
        
        text_buffer += segment['text']
        token_count = len(encoding.encode(text_buffer))
        
        if index == len(combined_transcript_segments) - 1:
            chunks.append(text_buffer)
            text_buffer = ''
            break
        
        if token_count > 1200:
            if text_buffer.strip().endswith('.'):
                chunks.append(text_buffer)
                text_buffer = ''
                
    system_prompt = """
    You are a helpful assistant whose job it is to label a transcript according to who is speaking.
    You will see a transcript from a conversation between an interviewer and three respondents.
    Reorganise and label the transcript so it is clear who is speaking. Guess the name of the respondent from the context where possible.
    Remove filler words and phrases without changing the meaning of the transcript.
    Only add necessary punctuation such as periods, commas, and capitalization, and use only the context provided.
    Return your response as a JSON.

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
                
    processed_chunks = []
    for index, chunk in enumerate(chunks):
        if index == 0:
            completion = client.chat.completions.create(
                model="gpt-4o",
                response_format={ "type" : "json_object" },
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt
                    },
                    {
                        "role": "user",
                        "content": chunk
                    }
                ],
                temperature=0.2
            )
        else:
            annotated_system_prompt = system_prompt + f"""
            ---
            You are continuing on from a previous transcription which ended as follows:

            "{chunk_items[-4][0]}" : {chunk_items[-4][1]},
            "{chunk_items[-3][0]}" : {chunk_items[-3][1]},
            "{chunk_items[-2][0]}" : {chunk_items[-2][1]},
            "{chunk_items[-1][0]}" : {chunk_items[-1][1]}
            """

            completion = client.chat.completions.create(
                model="gpt-4o",
                response_format={ "type" : "json_object" },
                messages=[
                    {
                        "role": "system",
                        "content": annotated_system_prompt
                    },
                    {
                        "role": "user",
                        "content": chunk
                    }
                ],
                temperature=0.2
            )
        print(f"Processed chunk at index {index}")
        print(f"Finish reason: {completion.choices[0].finish_reason}")
        
        processed_chunk = json.loads(completion.choices[0].message.content)
        processed_chunks.append(processed_chunk)
        chunk_items = list(processed_chunk.items())
        
    combined_processed_chunks =  {key: value for d in processed_chunks for key, value in d.items()}

    if save:
        with open(f'{filename}_final_output.json', 'w') as json_file:
            json.dump(combined_processed_chunks, json_file)
    
    return combined_transcript_segments, combined_processed_chunks
    
    
    