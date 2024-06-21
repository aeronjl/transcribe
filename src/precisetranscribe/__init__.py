from . import utils, whisper

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
        """
        """
        # Rearrange the transcript segments into chunks which fit the token limit
        chunks, n_transcript_chunks = utils.chunk_transcript_to_token_limit(transcribed_audio_segments, token_limit=1200)    

        print(f"Processing {n_transcript_chunks} transcript chunks. Estimated time: {n_transcript_chunks * 30} seconds.")  
        processed_transcript = utils.process_transcription(chunks, speakers)
        # os.system('cls' if os.name == 'nt' else 'clear')
        return processed_transcript