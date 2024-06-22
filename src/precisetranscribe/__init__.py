import json
from concurrent.futures import ThreadPoolExecutor, as_completed
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
    
def process_chunk(chunk, system_prompt, previous_chunk=None):
    prompt = system_prompt
    if previous_chunk:
        chunk_items = list(previous_chunk.items())
        prompt += "\n---\nYou are continuing on from a previous transcription which ended as follows:\n\n"
        prompt += "\n".join(f'"{k} : {json.dumps(v)},' for k, v in chunk_items)
    
    completion = gpt.process_transcription(chunk, prompt)
    
    try:
        return json.loads(completion.choices[0].message.content)
    except json.JSONDecodeError as e:
        print("Error decoding JSON from GPT response:")
        print(f"Error message: {str(e)}")
        print("GPT response content:")
        print(completion.choices[0].message.content)
        print("\nAttempting to fix JSON...")
        
        # Attempt to fix common JSON issues
        fixed_content = completion.choices[0].message.content
        fixed_content = fixed_content.replace("'", '"')  # Replace single quotes with double quotes
        fixed_content = fixed_content.strip()  # Remove leading/trailing whitespace
        if not fixed_content.startswith('{'): 
            fixed_content = '{' + fixed_content
        if not fixed_content.endswith('}'):
            fixed_content += '}'
        
        try:
            return json.loads(fixed_content)
        except json.JSONDecodeError:
            print("Failed to fix JSON. Returning raw content.")
            return {"error": "JSON parsing failed", "raw_content": completion.choices[0].message.content}


def process_whisper_transcription(transcribed_audio_segments, speakers=None):
    # Rearrange the transcript segments into chunks which fit the token limit
    chunks, n_transcript_chunks = utils.chunk_transcript_to_token_limit(transcribed_audio_segments, token_limit=1200)    
    print(f"Processing {n_transcript_chunks} transcript chunks. Estimated time: {n_transcript_chunks * 30} seconds.")
        
    system_prompt = utils.generate_system_prompt(speakers)
    
    processed_chunks = []
    previous_chunk = None
    for chunk in chunks:
        processed_chunk = process_chunk(chunk, system_prompt, previous_chunk)
        processed_chunks.append(processed_chunk)
        previous_chunk = processed_chunk
        
    processed_transcript = {key: value for d in processed_chunks for key, value in d.items()}
    return processed_transcript