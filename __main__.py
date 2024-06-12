import argparse
import os
from . import utils as ts
from . import transcribe_audio

def main():
    parser = argparse.ArgumentParser(description="Transcribe an audio file")
    parser.add_argument("--input", "-i", type=str, help="Input file path")
    args = parser.parse_args()
    
    input_file = args.input
    
    combined_transcript_segments, combined_processed_chunks = transcribe_audio(input_file)

if __name__ == "__main__":
    main()