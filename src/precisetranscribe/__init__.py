import json
from . import utils, whisper, gpt

def transcribe_audio(wav_data):
    """
    """
    audio_segments = utils.segment_audio(wav_data, 100000)
    n_segments = len(audio_segments)
    print(f"Transcribing {n_segments} audio segments. Estimated time: {n_segments * 10} seconds.")
    transcribed_audio_segments = []
    for index, audio_segment in enumerate(audio_segments):
        print(f"Transcribing audio segment {index + 1}/{n_segments}")
        
        transcribed_audio_segment = whisper.transcribe_audio_segment(audio_segment)
        if index == 0:
            # No need to renumber the first segment
            last_text_segment_id = transcribed_audio_segment[-1]['id']
            last_end_time = transcribed_audio_segment[-1]['end']
        else:
            transcribed_audio_segment = utils.renumber_transcribed_audio_segment(transcribed_audio_segment, number_to_start_from=last_text_segment_id)
            last_text_segment_id = transcribed_audio_segment[-1]['id']
            last_end_time = transcribed_audio_segment[-1]['end']
        transcribed_audio_segments.append(transcribed_audio_segment)

    combined_transcript_segments = [item for sublist in transcribed_audio_segments for item in sublist]

    # Convert the start and end times to timestamps
    for index, segment in enumerate(combined_transcript_segments):
        combined_transcript_segments[index]['start'] = utils.convert_to_timestamp(segment['start'])
        combined_transcript_segments[index]['end'] = utils.convert_to_timestamp(segment['end'])  
        
    return combined_transcript_segments
    
def process_whisper_transcription(transcribed_audio_segments, speakers=None):
    # Rearrange the transcript segments into chunks which fit the token limit
    chunks, n_transcript_chunks = utils.chunk_transcript_to_token_limit(transcribed_audio_segments, token_limit=1200)    
    print(f"Processing {n_transcript_chunks} transcript chunks. Estimated time: {n_transcript_chunks * 30} seconds.")
        
    system_prompt = utils.generate_system_prompt(speakers)
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
    processed_transcript =  {key: value for d in processed_chunks for key, value in d.items()}
    return processed_transcript