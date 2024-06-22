import json
import re
from concurrent.futures import ThreadPoolExecutor

from . import utils, whisper, gpt

def transcribe_audio_segment(audio_segment):
    return whisper.transcribe_audio_segment(audio_segment)

def transcribe_audio(wav_data):
    audio_segments = utils.segment_audio(wav_data, 100000)
    n_segments = len(audio_segments)
    print(f"Transcribing {n_segments} audio segments. Estimated time: {n_segments * 10} seconds.")
    print(f"Total audio duration: {sum(len(segment) for segment in audio_segments) / 1000} seconds")

    with ThreadPoolExecutor() as executor:
        futures = []
        for i, segment in enumerate(audio_segments):
            print(f"Submitting segment {i+1}/{n_segments} for transcription (duration: {len(segment)/1000} seconds)")
            future = executor.submit(transcribe_audio_segment, segment)
            futures.append(future)

        transcribed_audio_segments = []
        for i, future in enumerate(futures):
            result = future.result()
            if result:
                transcribed_audio_segments.extend(result)
                print(f"Completed transcription of segment {i+1}. First text: {result[0]['text'][:50]}...")
            else:
                print(f"Warning: Segment {i+1} returned no transcription")

    # Renumber and adjust timings
    current_id = 0
    current_time = 0
    for segment in transcribed_audio_segments:
        segment['id'] = current_id
        current_id += 1
        
        segment['start'] += current_time
        segment['end'] += current_time
        current_time = segment['end']
        
        segment['start'] = utils.convert_to_timestamp(segment['start'])
        segment['end'] = utils.convert_to_timestamp(segment['end'])
    
    print(f"Total transcribed segments: {len(transcribed_audio_segments)}")
    if transcribed_audio_segments:
        print(f"First transcribed segment: {transcribed_audio_segments[0]['text'][:50]}...")
        print(f"Last transcribed segment: {transcribed_audio_segments[-1]['text'][-50:]}")
    
    return transcribed_audio_segments

def remove_excessive_repetitions(text, max_repetitions=3):
    words = text.split()
    result = []
    repeat_count = 0
    last_word = None
    for word in words:
        if word == last_word:
            repeat_count += 1
            if repeat_count <= max_repetitions:
                result.append(word)
        else:
            repeat_count = 1
            result.append(word)
        last_word = word
    return ' '.join(result)

def truncate_content(content, max_length=500):
    if len(content) > max_length:
        return content[:max_length] + "..."
    return content

def clean_and_parse_json(raw_json):
    # Remove any trailing commas in the JSON string
    cleaned_json = re.sub(r',\s*}', '}', raw_json)
    cleaned_json = re.sub(r',\s*]', ']', cleaned_json)
    
    # Parse the JSON
    try:
        parsed_json = json.loads(cleaned_json)
        
        # Clean up each content field
        for key, value in parsed_json.items():
            if isinstance(value, dict) and 'Content' in value:
                value['Content'] = remove_excessive_repetitions(value['Content'])
                value['Content'] = truncate_content(value['Content'])
        
        return parsed_json
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}")
        return None

def process_chunk(chunk, system_prompt, previous_chunk=None):
    prompt = system_prompt
    if previous_chunk:
        chunk_items = list(previous_chunk.items())
        prompt += "\n---\nYou are continuing on from a previous transcription which ended as follows:\n\n"
        prompt += "\n".join(f'"{k} : {json.dumps(v)},' for k, v in chunk_items)
    
    completion = gpt.process_transcription(chunk, prompt)
    
    # Clean and parse the JSON output
    cleaned_json = clean_and_parse_json(completion.choices[0].message.content)
    
    if cleaned_json:
        return cleaned_json
    else:
        # If parsing fails, return a simplified error response
        return {"error": "JSON parsing failed", "raw_content": completion.choices[0].message.content[:1000]}  # Truncate raw content to 1000 characters


def process_whisper_transcription(transcribed_audio_segments, speakers=None):
    chunks, n_transcript_chunks = utils.chunk_transcript_to_token_limit(transcribed_audio_segments, token_limit=1200)    
    print(f"Processing {n_transcript_chunks} transcript chunks. Estimated time: {n_transcript_chunks * 30} seconds.")
        
    system_prompt = gpt.generate_system_prompt(speakers)
    
    processed_chunks = {}
    previous_chunk = None
    for chunk in chunks:
        processed_chunk = process_chunk(chunk, system_prompt, previous_chunk)
        processed_chunks.update(processed_chunk)  # Merge the new chunk into the accumulated dictionary
        previous_chunk = processed_chunk
        
    return processed_chunks