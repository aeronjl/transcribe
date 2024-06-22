import datetime
from difflib import SequenceMatcher
import re
import json
from typing import List, Dict, Any, Tuple

import tiktoken

from . import gpt

# Constants
ENCODING = tiktoken.get_encoding("cl100k_base")

def renumber_transcribed_audio_segment(
    transcribed_audio_segment: List[Dict[str, Any]],
    number_to_start_from: int = 0,
    time_to_start_from: float = 0
) -> List[Dict[str, Any]]:
    for phrase_index, phrase in enumerate(transcribed_audio_segment):
        phrase['id'] = number_to_start_from + phrase_index + 1
        phrase['start'] = phrase['start'] + time_to_start_from
        phrase['end'] = phrase['end'] + time_to_start_from
    return transcribed_audio_segment

def chunk_transcript_to_token_limit(
    transcript_segments: List[Dict[str, Any]],
    token_limit: int = 1200
) -> Tuple[List[str], int]:
    text_buffer = ''
    chunks = []
    for index, segment in enumerate(transcript_segments):    
        text_buffer += segment['text']
        token_count = len(ENCODING.encode(text_buffer))
        
        if index == len(transcript_segments) - 1:
            chunks.append(text_buffer)
            text_buffer = ''
            break
        
        if token_count > token_limit:
            if text_buffer.strip().endswith('.'):
                chunks.append(text_buffer)
                text_buffer = ''
                    
    return chunks, len(chunks)

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
    chunks, n_transcript_chunks = chunk_transcript_to_token_limit(transcribed_audio_segments, token_limit=1200)    
    print(f"Processing {n_transcript_chunks} transcript chunks. Estimated time: {n_transcript_chunks * 30} seconds.")
        
    system_prompt = gpt.generate_system_prompt(speakers)
    
    processed_chunks = {}
    previous_chunk = None
    for chunk in chunks:
        processed_chunk = process_chunk(chunk, system_prompt, previous_chunk)
        
        # Merge the new chunk into the accumulated dictionary
        # Use the highest existing key + 1 as the starting point for new keys
        start_key = max(map(int, processed_chunks.keys() or ['0'])) + 1
        for i, (key, value) in enumerate(processed_chunk.items(), start=start_key):
            processed_chunks[str(i)] = value
        
        previous_chunk = processed_chunk
    
    return processed_chunks

def similarity(a, b):
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()

def time_to_seconds(time_str):
    t = datetime.datetime.strptime(time_str, "%H:%M:%S")
    return t.hour * 3600 + t.minute * 60 + t.second

def align_timestamps(processed_transcript, whisper_transcript):
    aligned_transcript = {}
    whisper_entries = sorted([entry for entry in whisper_transcript if entry['text'].strip()], key=lambda x: x['start'])
    
    last_matched_index = -1
    last_end_time = "00:00:00"

    for key, item in processed_transcript.items():
        content = item['Content']
        
        # Consider only entries after the last matched one
        candidate_entries = whisper_entries[last_matched_index + 1:]
        
        # Find the best match among candidates
        best_match = max(candidate_entries, key=lambda x: similarity(content, x['text']))
        best_match_index = whisper_entries.index(best_match)
        
        # Check if the match makes sense time-wise (within 5 minutes of the last end time)
        if time_to_seconds(best_match['start']) - time_to_seconds(last_end_time) > 300:
            # If not, find the closest match in time
            best_match = min(candidate_entries, key=lambda x: abs(time_to_seconds(x['start']) - time_to_seconds(last_end_time)))
            best_match_index = whisper_entries.index(best_match)
        
        start_time = best_match['start']
        
        # Find the end time by looking at the next entry's start time
        if best_match_index + 1 < len(whisper_entries):
            end_time = whisper_entries[best_match_index + 1]['start']
        else:
            end_time = best_match['end']
        
        aligned_transcript[key] = {
            'Speaker': item['Speaker'],
            'Content': content,
            'Start': start_time,
            'End': end_time
        }
        
        last_matched_index = best_match_index
        last_end_time = end_time
    
    return aligned_transcript

def format_entry(entry):
    speaker = entry['Speaker']
    start = entry['Start']
    end = entry['End']
    content = entry['Content']
    
    formatted = f"{speaker}\n{start} - {end}\n\n{content}\n\n"
    return formatted

def export_transcript(transcript, filename='transcript'):
    with open(f'data/transcripts/{filename}.txt', 'w') as f:
        for key in sorted(transcript.keys(), key=int):  # Sort keys numerically
            entry = transcript[key]
            formatted_entry = format_entry(entry)
            f.write(formatted_entry)
            
# Main execution
if __name__ == "__main__":
    pass