import os
import json
from . import utils, whisper


def clear_console():
    os.system('cls' if os.name == 'nt' else 'clear')
    
def transcribe_audio(input_file, save=True):
    """
    """
    # Split the input filepath into filename and extension
    #filename, extension = utils.split_filepath(input_filepath)
    #if not extension:
    #    raise ValueError(f"No file extension found for {input_filepath}")
    
    # Convert the input file to WAV format
    utils.convert_input_to_wav(input_file)
    
    # Segment the audio file into smaller chunks for transcribing
    audio_segments = utils.segment_audio(input_file, 100000)
    n_segments = len(audio_segments)
    
    clear_console()
    
    print(f"Transcribing {n_segments} audio segments. Estimated time: {n_segments * 10} seconds.")
    
    # Transcribe the audio segments
    transcribed_audio_segments = []
    for index, audio_segment in enumerate(audio_segments):
        print(f"Transcribing audio segment {index + 1}/{n_segments}")
        buffer = utils.buffer_audio(audio_segment)
        transcribed_audio_segment = whisper.transcribe_audio_segment(buffer)
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
        
    if save:
        # Save the transcribed audio segments to a JSON file
        with open(f'{filename}_whisper_output.json', 'w') as json_file:
            json.dump(combined_transcript_segments, json_file)
    
    # Chunk the transcript segments to a token limit
    chunks, n_transcript_chunks = utils.chunk_transcript_to_token_limit(combined_transcript_segments, token_limit=1200)    

    clear_console()
    
    # Process the transcription chunks with GPT-4o
    print(f"Processing {n_transcript_chunks} transcript chunks. Estimated time: {n_transcript_chunks * 30} seconds.")  
    combined_processed_chunks = utils.process_transcription(chunks)
    
    # Save the processed output to a JSON file
    if save:
        with open(f'{filename}_final_output.json', 'w') as json_file:
            json.dump(combined_processed_chunks, json_file)

    return combined_transcript_segments, combined_processed_chunks