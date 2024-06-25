import argparse
import os

from . import audio, file, text

def main():
    parser = argparse.ArgumentParser(description="Transcribe an audio file")
    parser.add_argument("--input", "-i", type=str, help="Input file path")
    args = parser.parse_args()
    
    input_file = args.input
    
    wav_data = audio.prepare_audio(input_file)
    
    whisper_output = audio.transcribe_audio(wav_data)
    file.save_whisper_output(whisper_output, f"data/transcripts/{os.path.splitext(input_file)[0]}")
    
    processed_transcript = text.process_whisper_transcription(whisper_output, speakers=2)
    file.save_processed_transcript(processed_transcript, f"data/transcripts/{os.path.splitext(input_file)[0]}")
    
    aligned_transcript = text.align_timestamps(processed_transcript, whisper_output)
    file.save_aligned_transcript(aligned_transcript, f"data/transcripts/{os.path.splitext(input_file)[0]}")

    text.export_transcript(aligned_transcript, filename=f"data/transcripts/{os.path.splitext(input_file)[0]}")

if __name__ == "__main__":
    main()
