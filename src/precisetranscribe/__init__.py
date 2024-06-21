from . import utils, whisper

def transcribe_audio(wav_data):
    """
    Transcribes audio data from a WAV file in segmented chunks, processes each segment,
    and saves the results as JSON files.

    Args:
    - wav_data (bytes): Audio data in WAV format.
    - save (bool, optional): Whether to save output JSON files (default: True).

    Returns:
    - tuple: A tuple containing two lists:
        - combined_transcript_segments (list): List of all transcribed segments with start/end times.
        - combined_processed_chunks (list): List of all processed transcript chunks.

    Notes:
    - Uses utility functions from 'utils' for segmenting audio, buffering, renumbering segments,
      converting times to timestamps, chunking transcripts, and processing with GPT-4o.
    - Requires 'whisper' for actual transcription and 'utils' for various utility functions.

    Example usage:
    >>> audio_data = load_wav_file('audio.wav')
    >>> transcriptions, processed_chunks = transcribe_audio(audio_data)
    """
    
    # Segment the audio file into smaller chunks for transcribing
    audio_segments = utils.segment_audio(wav_data, 100000)
    n_segments = len(audio_segments)
    
    # os.system('cls' if os.name == 'nt' else 'clear')
    
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
    
    # Chunk the transcript segments to a token limit
    chunks, n_transcript_chunks = utils.chunk_transcript_to_token_limit(combined_transcript_segments, token_limit=1200)    

    # os.system('cls' if os.name == 'nt' else 'clear')
    
    # Process the transcription chunks with GPT-4o
    print(f"Processing {n_transcript_chunks} transcript chunks. Estimated time: {n_transcript_chunks * 30} seconds.")  
    combined_processed_chunks = utils.process_transcription(chunks)

    return combined_transcript_segments, combined_processed_chunks