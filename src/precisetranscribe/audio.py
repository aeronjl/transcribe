import os
from io import BytesIO
from typing import BinaryIO, List, Optional
from concurrent.futures import ThreadPoolExecutor

import ffmpeg
import numpy as np
from pydub import AudioSegment

from . import utils, whisper

def convert_to_wav(input_file: BinaryIO) -> Optional[bytes]:
    """Convert input audio to WAV format."""
    input_file.seek(0, 2)
    file_size = input_file.tell()
    input_file.seek(0)
    print(f"Input file size: {input_file.tell()} bytes")
    print(f"Input file type: {type(input_file)}")
    print(f"Input file mode: {input_file.mode}")
    
    with utils.temporary_file() as temp_input, utils.temporary_file('.wav') as temp_output:
        print(f"Temporary input file: {temp_input}")
        print(f"Temporary output file: {temp_output}")
        
        # Write the input file to a temporary file
        try:
            with open(temp_input, 'wb') as f:
                input_data = input_file.read()
                f.write(input_data)
                print(f"Wrote {len(input_data)} bytes to temporary input file")
        except Exception as e:
            print(f"Error writing to temporary input file: {str(e)}")
            return None
            
        try:
            # Get the duration of the input file
            probe = ffmpeg.probe(temp_input)
            duration = float(probe['streams'][0]['duration'])
            print(f"Input file duration: {duration} seconds")
            
            stream = ffmpeg.input(temp_input)
            stream = ffmpeg.output(stream, temp_output, acodec='pcm_s16le', ac=1, ar='16k')
            ffmpeg.run(stream, overwrite_output=True, capture_stdout=True, capture_stderr=True)
            
            # Read the output file
            with open(temp_output, 'rb') as f:
                wav_data = f.read()
                print(f"Converted WAV size: {len(wav_data)} bytes")
            
            # Get the duration of the output file
            probe = ffmpeg.probe(temp_output)
            out_duration = float(probe['streams'][0]['duration'])
            print(f"Output file duration: {out_duration} seconds")
            
            if abs(duration - out_duration) > 0.1:
                print(f"Warning: Input and output durations differ by {abs(duration - out_duration)} seconds")
                
            return wav_data
        except ffmpeg.Error as e:
            print('stdout:', e.stdout.decode('utf8'))
            print('stderr:', e.stderr.decode('utf8'))
        except Exception as e:
            print(f'An unexpected error occurred: {str(e)}')
        return None

def prepare_audio(filename: str) -> Optional[bytes]:   
    if not os.path.exists(filename):
        print(f"Error: File '{filename}' does not exist.")
    else:
        print(f"File '{filename}' found.")

        try:
            with open(filename, "rb") as input_file:
                wav_data = convert_to_wav(input_file)
            
            if wav_data is None:
                print("Error: convert_to_wav returned None.")
            else:
                print(f"Successfully converted file. WAV data length: {len(wav_data)} bytes")
        except Exception as e:
            print(f"An error occurred: {str(e)}")
    return wav_data

def segment_audio(wav_data: bytes, segment_duration: int) -> List[AudioSegment]:
    """
    Segments an audio file into smaller chunks.
    """
    def check_segment_end_time(proposed_end_time, max_total_duration):
        return proposed_end_time if proposed_end_time < max_total_duration else max_total_duration
    
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

def transcribe_audio_segment(audio_segment):
    return whisper.transcribe_audio_segment(audio_segment)

def transcribe_audio(wav_data):
    audio_segments = segment_audio(wav_data, 100000)
    n_segments = len(audio_segments)
    print(f"Transcribing {n_segments} audio segments. Total audio duration: {sum(len(segment) for segment in audio_segments) / 1000} seconds")

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

# Main execution
if __name__ == "__main__":
    pass