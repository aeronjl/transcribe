import os
from io import BytesIO
from typing import BinaryIO, List, Optional
from concurrent.futures import ThreadPoolExecutor

import ffmpeg
import numpy as np
from pydub import AudioSegment

from . import utils, whisper
from .datastructures import WhisperOutput


def convert_to_wav(input_file: BinaryIO) -> Optional[bytes]:
    """Convert input audio to WAV format."""
    try:
        input_file.seek(0, 2)
        # file_size = input_file.tell()
        input_file.seek(0)
        print(f"Input file size: {input_file.tell()} bytes")
        print(f"Input file type: {type(input_file)}")
        print(f"Input file mode: {input_file.mode}")

        with (
            utils.temporary_file() as temp_input,
            utils.temporary_file(".wav") as temp_output,
        ):
            print(f"Temporary input file: {temp_input}")
            print(f"Temporary output file: {temp_output}")

            # Write the input file to a temporary file
            try:
                with open(temp_input, "wb") as f:
                    input_data = input_file.read()
                    f.write(input_data)
                    print(f"Wrote {len(input_data)} bytes to temporary input file")
            except Exception as e:
                print(f"Error writing to temporary input file: {str(e)}")
                return None

            # Get the duration of the input file
            probe = ffmpeg.probe(temp_input)
            duration = float(probe["streams"][0]["duration"])
            print(f"Input file duration: {duration} seconds")

            stream = ffmpeg.input(temp_input)
            stream = ffmpeg.output(
                stream, temp_output, acodec="pcm_s16le", ac=1, ar="16k"
            )
            ffmpeg.run(
                stream, overwrite_output=True, capture_stdout=True, capture_stderr=True
            )

            # Read the output file
            with open(temp_output, "rb") as f:
                wav_data = f.read()
                print(f"Converted WAV size: {len(wav_data)} bytes")

            # Get the duration of the output file
            probe = ffmpeg.probe(temp_output)
            out_duration = float(probe["streams"][0]["duration"])
            print(f"Output file duration: {out_duration} seconds")

            if abs(duration - out_duration) > 0.1:
                print(
                    f"Warning: Input and output durations differ by {abs(duration - out_duration)} seconds"
                )

            return wav_data

    except ffmpeg.Error as e:
        print("An FFmpeg error occurred:")
        print("stdout:", e.stdout.decode("utf8"))
        print("stderr:", e.stderr.decode("utf8"))
    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")
    return None


def prepare_audio(filename: str) -> bytes:
    """
    Prepare an audio file for transcription by converting it to WAV format.

    Args:
        filename(str): The path to the audio file.

    Returns:
        bytes: The WAV data.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file conversion fails.
        Exception: For any other error.
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Error: File '{filename}' does not exist.")

    try:
        with open(filename, "rb") as input_file:
            wav_data = convert_to_wav(input_file)

        if wav_data is None:
            raise ValueError("Error: convert_towav() returned None.")

        print(f"Successfully converted file. WAV data length: {len(wav_data)} bytes")
        return wav_data
    except Exception as e:
        raise Exception(f"Error preparing audio file: {str(e)}")


def segment_audio(
    wav_data: bytes, segment_duration: int, overlap_duration: int = 2000
) -> List[AudioSegment]:
    """
    Segments an audio file into smaller chunks with overlap to improve transcription accuracy.

    Args:
        wav_data: The WAV audio data to segment
        segment_duration: Duration of each segment in milliseconds
        overlap_duration: Duration of overlap between segments in milliseconds (default: 2000ms)

    Returns:
        List of audio segments with overlap
    """
    audio_to_segment = AudioSegment.from_wav(BytesIO(wav_data))
    audio_duration = len(audio_to_segment)
    print(f"Total audio duration: {audio_duration / 1000:.2f} seconds")

    # Calculate effective segment duration (accounting for overlap)
    effective_segment_duration = segment_duration - overlap_duration

    # Calculate number of segments needed
    # Ensure we create at least one segment even for very short audio
    if audio_duration <= segment_duration:
        # For short audio, just use the entire thing
        n_segments = 1
    else:
        # Otherwise calculate based on effective duration
        n_segments = max(
            1,
            int(
                np.ceil(
                    (audio_duration - overlap_duration) / effective_segment_duration
                )
            ),
        )

    print(
        f"Segmenting into {n_segments} segments of ~{segment_duration/1000:.1f}s each with {overlap_duration/1000:.1f}s overlap"
    )

    audio_segments = []
    for segment_index in range(n_segments):
        # Calculate segment start and end times
        start_time = segment_index * effective_segment_duration
        end_time = min(start_time + segment_duration, audio_duration)

        # Handle special case for last segment
        if segment_index == n_segments - 1:
            # Make sure the last segment includes all remaining audio
            end_time = audio_duration
            # Adjust start time to maintain segment_duration if possible
            if end_time - segment_duration > 0:
                start_time = max(0, end_time - segment_duration)

        # Ensure minimum segment length
        if end_time - start_time < 1000:
            print(
                f"Skipping segment {segment_index+1} because it's too short ({(end_time-start_time)/1000:.2f}s)"
            )
            continue

        segment = audio_to_segment[start_time:end_time]
        audio_segments.append(segment)
        print(
            f"Created segment {segment_index+1}: {start_time/1000:.2f}s - {end_time/1000:.2f}s (duration: {len(segment)/1000:.2f}s)"
        )

    return audio_segments


def transcribe_audio(wav_data: bytes) -> WhisperOutput:
    """Transcribe audio data using a two-phase approach for accurate timestamps.

    First phase: Get precise timestamps across the entire audio
    Second phase: Process the content with speaker identification

    Args:
        wav_data (bytes): WAV audio data to transcribe

    Returns:
        WhisperOutput: List of transcribed segments with accurate timestamps
    """
    # Use appropriate segment sizes for better timestamp accuracy
    segment_duration_ms = 40 * 1000  # 40 seconds
    overlap_duration_ms = 4000  # 4 seconds overlap (10% of segment)

    print(f"Phase 1: Transcribing audio with accurate timestamps")

    # Prepare audio segments with overlap
    audio_to_segment = AudioSegment.from_wav(BytesIO(wav_data))
    total_audio_duration = len(audio_to_segment) / 1000  # in seconds
    print(f"Total audio duration: {total_audio_duration:.2f} seconds")

    # Instead of segmenting first, get a full transcription to use as a reference
    # for timestamp calibration
    print("Getting full audio transcription for timestamp reference...")
    full_transcription = []

    # For very long audio, we still need to segment, but we'll do it differently
    if total_audio_duration > 60 * 60:  # If longer than 1 hour
        print(
            f"Long audio detected ({total_audio_duration/60:.1f} minutes). Using segmented approach."
        )

        # 1. Create segments with substantial overlap
        segments = segment_audio(wav_data, segment_duration_ms, overlap_duration_ms)
        n_segments = len(segments)

        # 2. Calculate accurate base time for each segment
        segment_base_times = []
        for i in range(n_segments):
            effective_duration = segment_duration_ms - overlap_duration_ms
            base_time = (i * effective_duration) / 1000  # in seconds
            segment_base_times.append(base_time)

        # 3. Transcribe each segment with proper time calibration
        with ThreadPoolExecutor() as executor:
            futures = []
            for i, segment in enumerate(segments):
                print(f"Submitting segment {i+1}/{n_segments} for transcription")
                future = executor.submit(whisper.transcribe_audio_segment, segment)
                futures.append((i, future))

            for i, future in sorted(futures, key=lambda x: x[0]):
                result = future.result()
                if not result:
                    print(f"Warning: No transcription for segment {i+1}")
                    continue

                base_time = segment_base_times[i]

                # If not the first segment, we need special handling for the overlap region
                overlap_handling_range = 0
                if i > 0:
                    overlap_handling_range = overlap_duration_ms / 1000  # seconds

                for j, segment in enumerate(result):
                    # Skip segments at the start of non-first chunks that are likely duplicates
                    if i > 0 and segment["start"] < overlap_handling_range:
                        # Only include if it appears to be a new segment not covered by previous chunk
                        # This is determined by looking at when the previous chunk ended
                        if (
                            full_transcription
                            and base_time + segment["start"]
                            < full_transcription[-1]["end"] + 1.0
                        ):
                            continue

                    # Calibrate the timestamps
                    absolute_start = base_time + segment["start"]
                    absolute_end = base_time + segment["end"]

                    # Create a new properly timed segment
                    new_segment = segment.copy()
                    new_segment["start"] = absolute_start
                    new_segment["end"] = absolute_end

                    # Add it to our results
                    full_transcription.append(new_segment)
    else:
        # For shorter audio, transcribe the entire thing at once
        print("Audio is short enough to transcribe at once for best timestamp accuracy")
        full_result = whisper.transcribe_audio_segment(audio_to_segment)
        full_transcription = full_result

    # Sort all segments by start time
    full_transcription.sort(key=lambda x: x["start"])

    # Deduplicate segments based on start time and content similarity
    print("Deduplicating segments...")
    if full_transcription:
        deduplicated = [full_transcription[0]]

        for i in range(1, len(full_transcription)):
            curr = full_transcription[i]
            prev = deduplicated[-1]

            # Check for similar start times (within 1 second)
            if abs(curr["start"] - prev["start"]) < 1.0:
                # Check for similar content
                from difflib import SequenceMatcher

                similarity = SequenceMatcher(
                    None, curr["text"].lower(), prev["text"].lower()
                ).ratio()

                # If very similar, keep the one with more content
                if similarity > 0.5:
                    if len(curr["text"]) > len(prev["text"]):
                        # Replace previous with current
                        deduplicated[-1] = curr
                    continue

            # Check for overlaps and resolve them
            if curr["start"] < prev["end"]:
                # If significant overlap, adjust the previous end time
                overlap_amount = prev["end"] - curr["start"]
                if overlap_amount > 0.3:  # If overlap is more than 0.3 seconds
                    # Split the difference for the overlap
                    middle_point = (prev["end"] + curr["start"]) / 2
                    prev["end"] = middle_point
                    curr["start"] = middle_point

            # Add to deduplicated list
            deduplicated.append(curr)

        full_transcription = deduplicated

    # Validate and fix any remaining timestamp issues
    print("Validating timestamps...")
    if len(full_transcription) > 1:
        for i in range(1, len(full_transcription)):
            # Ensure timestamps are strictly increasing
            if full_transcription[i]["start"] <= full_transcription[i - 1]["end"]:
                # Set a small gap between segments (30ms)
                full_transcription[i]["start"] = full_transcription[i - 1]["end"] + 0.03

            # Ensure each segment has a reasonable duration
            min_duration = 0.1  # 100ms minimum
            if (
                full_transcription[i]["end"] - full_transcription[i]["start"]
                < min_duration
            ):
                full_transcription[i]["end"] = (
                    full_transcription[i]["start"] + min_duration
                )

    # Re-number all segments
    for i, segment in enumerate(full_transcription):
        segment["id"] = i

    print(
        f"Final transcript contains {len(full_transcription)} segments with calibrated timestamps"
    )
    if full_transcription:
        print(
            f"First segment: \"{full_transcription[0]['text'][:50]}...\" "
            + f"[{format_timestamp(full_transcription[0]['start'])} - {format_timestamp(full_transcription[0]['end'])}]"
        )
        print(
            f"Last segment: \"{full_transcription[-1]['text'][-50:]}\" "
            + f"[{format_timestamp(full_transcription[-1]['start'])} - {format_timestamp(full_transcription[-1]['end'])}]"
        )

    return full_transcription


def format_timestamp(seconds):
    """Format seconds as HH:MM:SS.MS"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds_remainder = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{seconds_remainder:06.3f}"


# Main execution
if __name__ == "__main__":
    pass

# Timestamp Handling Strategy:
# ---------------------------
# 1. Audio Segmentation:
#    - The full audio is divided into overlapping segments (default 30s with 3s overlap)
#    - Each segment's absolute start time in the original audio is calculated
#
# 2. Parallel Transcription:
#    - Each segment is transcribed by Whisper in parallel
#    - Whisper provides relative timestamps within each segment
#
# 3. Timestamp Calculation:
#    - For each transcribed segment, we add the segment's start time to Whisper's relative timestamps
#    - This converts relative timestamps to absolute timestamps in the original audio
#
# 4. Boundary Handling:
#    - In overlap regions, we prioritize segments from the first half of the overlap
#    - Segments that start in the second half of an overlap are skipped (to avoid duplicates)
#    - Segments that extend too far into the next segment's territory are truncated
#
# 5. Deduplication:
#    - We check for segments with overlapping timestamps and similar content
#    - If found, we merge them to eliminate duplicate transcriptions
#
# This approach ensures accurate timestamps while handling segment boundaries properly.
