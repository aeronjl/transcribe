import argparse
import os
from . import transcribe as ts

def main():
    parser = argparse.ArgumentParser(description="Transcribe an audio file")
    parser.add_argument("--input", "-i", type=str, help="Input file path")
    args = parser.parse_args()
    
    input_file = args.input
    filename, extension = os.path.splitext(input_file)
    print(f"Filename: {filename}, Extension: {extension}")
    if not extension:
        raise ValueError(f"No file extension found for {input_file}")
    
    ts.convert_audio_to_wav(filename, extension)
    audio_segments = ts.segment_audio(f"{filename}.wav", 100000)
    combined_transcript_segments, combined_processed_chunks = ts.transcribe_audio_segments(audio_segments, filename=filename)

if __name__ == "__main__":
    main()