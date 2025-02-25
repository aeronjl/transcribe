import datetime
from difflib import SequenceMatcher
import re
import json
from typing import List, Dict, Any, Tuple, Union, Optional

import tiktoken

from . import gpt
from .datastructures import WhisperOutput, Transcript
from . import utils

# Constants
ENCODING = tiktoken.get_encoding("cl100k_base")


def renumber_transcribed_audio_segment(
    transcribed_audio_segment: List[Dict[str, Any]],
    number_to_start_from: int = 0,
    time_to_start_from: float = 0,
) -> List[Dict[str, Any]]:
    for phrase_index, phrase in enumerate(transcribed_audio_segment):
        phrase["id"] = number_to_start_from + phrase_index + 1
        phrase["start"] = phrase["start"] + time_to_start_from
        phrase["end"] = phrase["end"] + time_to_start_from
    return transcribed_audio_segment


def chunk_transcript_to_token_limit(
    transcript_segments: List[Dict[str, Any]], token_limit: int = 1200
) -> Tuple[List[str], int]:
    """
    Splits transcript segments into chunks based on token limit.
    Improved to ensure NO content is lost in chunking process and
    there's sufficient overlap between chunks for better context.
    """
    text_buffer = ""
    chunks = []
    segment_buffer = []
    accumulated_segments = []

    # Track which segments have been processed to ensure none are missed
    processed_segment_ids = set()

    OVERLAP_TOKEN_COUNT = 200  # Ensure 200 tokens of overlap between chunks

    for segment in transcript_segments:
        # Store the segment and update text buffer
        segment_buffer.append(segment)
        accumulated_segments.append(segment)
        new_text = text_buffer + segment["text"]
        token_count = len(ENCODING.encode(new_text))

        # If adding this segment exceeds the token limit, create a chunk
        if token_count > token_limit and text_buffer:  # Only if we already have content
            # Add the current buffer as a chunk
            chunks.append(" ".join([s["text"] for s in segment_buffer[:-1]]))

            # Record processed segment IDs
            for s in segment_buffer[:-1]:
                processed_segment_ids.add(s["id"])

            # Find overlap point - keep last few segments for context in next chunk
            overlap_segments = []
            overlap_tokens = 0
            for s in reversed(segment_buffer[:-1]):
                segment_tokens = len(ENCODING.encode(s["text"]))
                if overlap_tokens + segment_tokens > OVERLAP_TOKEN_COUNT:
                    break
                overlap_segments.insert(0, s)
                overlap_tokens += segment_tokens

            # Reset buffers but keep the current segment and overlap segments
            segment_buffer = overlap_segments + [segment]
            text_buffer = " ".join([s["text"] for s in segment_buffer])
        else:
            # Update text buffer
            text_buffer = new_text

    # Add any remaining content as the final chunk
    if segment_buffer:
        chunks.append(" ".join([s["text"] for s in segment_buffer]))
        for s in segment_buffer:
            processed_segment_ids.add(s["id"])

    # Ensure we've processed all segments
    if len(processed_segment_ids) != len(transcript_segments):
        missed_segments = [
            s for s in transcript_segments if s["id"] not in processed_segment_ids
        ]
        print(
            f"Warning: {len(missed_segments)} segments were not processed. Adding them to final chunk."
        )
        missed_text = " ".join([s["text"] for s in missed_segments])
        if chunks:
            chunks[-1] += " " + missed_text
        else:
            chunks.append(missed_text)

    return chunks, len(chunks)


def remove_excessive_repetitions(text, max_repetitions=3):
    words = text.split()
    result = []
    repeat_count = 0
    last_word = None
    for word in words:
        if word == last_word:
            repeat_count += 1
            if repeat_count <= max_repetitions:
                result.append(word)
        else:
            repeat_count = 1
            result.append(word)
        last_word = word
    return " ".join(result)


def extract_key_value_pairs(text: str) -> Dict[str, Dict[str, str]]:
    pattern = r'"(\d+)":\s*{\s*"Speaker":\s*"([^"]+)",\s*"Content":\s*"([^"]+)"\s*}'
    matches = re.findall(pattern, text)
    return {match[0]: {"Speaker": match[1], "Content": match[2]} for match in matches}


def clean_and_parse_json(
    raw_json: str,
) -> Union[Dict[str, Any], Dict[str, Union[str, Dict[str, Any]]]]:
    # Remove any trailing commas in the JSON string
    cleaned_json = re.sub(r",\s*}", "}", raw_json)
    cleaned_json = re.sub(r",\s*]", "]", cleaned_json)

    # Ensure all brackets and quotes are closed
    open_curly = cleaned_json.count("{")
    close_curly = cleaned_json.count("}")
    open_square = cleaned_json.count("[")
    close_square = cleaned_json.count("]")

    cleaned_json += "}" * (open_curly - close_curly)
    cleaned_json += "]" * (open_square - close_square)

    # Ensure the JSON starts and ends with curly braces
    if not cleaned_json.startswith("{"):
        cleaned_json = "{" + cleaned_json
    if not cleaned_json.endswith("}"):
        cleaned_json += "}"

    # Parse the JSON
    try:
        parsed_json = json.loads(cleaned_json)

        # Clean up each content field
        for key, value in parsed_json.items():
            if isinstance(value, dict) and "Content" in value:
                value["Content"] = remove_excessive_repetitions(value["Content"])

        return parsed_json
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}")

        # Fallback: Try to extract valid key-value pairs
        extracted_data = extract_key_value_pairs(cleaned_json)

        if extracted_data:
            return {
                "partial_parse": True,
                "extracted_data": extracted_data,
                "error": str(e),
                "raw_content": cleaned_json[
                    :1000
                ],  # Truncate raw content to 1000 characters
            }
        else:
            return {
                "error": "JSON parsing failed",
                "error_message": str(e),
                "raw_content": cleaned_json[
                    :1000
                ],  # Truncate raw content to 1000 characters
            }


def process_chunk(
    chunk: Dict, system_prompt: str, previous_chunk: Optional[Dict] = None
) -> Dict:
    """
    Process a chunk of transcript text using the GPT model.
    Enhanced to better handle overlapping contexts and ensure complete coverage.
    """
    prompt = system_prompt

    if previous_chunk:
        # Add more context from the previous chunk
        chunk_items = list(previous_chunk.items())
        prompt += "\n---\nYou are continuing on from a previous transcription which ended as follows:\n\n"

        # Include the last few items from the previous chunk as context
        context_items = chunk_items[-4:] if len(chunk_items) > 4 else chunk_items
        for k, v in context_items:
            prompt += f'"{k}" : {json.dumps(v)},\n'

        prompt += "\nPlease continue the transcription, preserving ALL content and maintaining the conversation flow."

    # Process with GPT
    try:
        completion: Dict = gpt.process_transcription(chunk, prompt)

        # Clean and parse the JSON output
        cleaned_json: Optional[Dict] = clean_and_parse_json(
            completion.choices[0].message.content
        )

        if cleaned_json:
            # Detect if any content is missing by comparing input and output word counts
            input_words = len(chunk.split())
            output_words = sum(
                len(item.get("Content", "").split())
                for item in cleaned_json.values()
                if isinstance(item, dict)
            )

            # If there's a significant discrepancy, log a warning
            if (
                output_words < input_words * 0.7
            ):  # Allow for some reduction due to filler word removal
                print(
                    f"Warning: Possible content loss. Input: {input_words} words, Output: {output_words} words"
                )

            return cleaned_json
    except Exception as e:
        print(f"Error in process_chunk: {str(e)}")
        return {
            "error": f"Processing failed: {str(e)}",
            "raw_content": chunk[:1000],  # Include part of the raw content
        }

    # Fallback for any other issues
    print("Warning: JSON parsing failed in process_chunk")
    return {
        "error": "JSON parsing failed",
        "raw_content": (
            completion.choices[0].message.content[:1000]
            if "completion" in locals()
            else chunk[:1000]
        ),
    }


def process_whisper_transcription(whisper_transcript: WhisperOutput, speakers=None):
    """
    Process the whisper transcript, breaking it into chunks and processing each chunk with GPT.
    Preserves the original timestamps from the Whisper output.

    Args:
        whisper_transcript: The raw Whisper output with timestamps
        speakers: Number of speakers to identify or None for auto-detection

    Returns:
        Processed transcript with speaker labels and original timestamps
    """
    # Create a mapping of original timestamp information
    timestamp_map = {}
    for segment in whisper_transcript:
        segment_id = str(segment["id"])
        timestamp_map[segment_id] = {
            "start": segment["start"],
            "end": segment["end"],
            "text": segment["text"],
        }

    # Break the transcript into chunks
    chunks, n_transcript_chunks = chunk_transcript_to_token_limit(
        whisper_transcript, token_limit=1000  # Reduced token limit for better handling
    )
    print(
        f"Processing {n_transcript_chunks} transcript chunks. Estimated time: {n_transcript_chunks * 30} seconds."
    )
    print(f"Total words in transcript: {sum(len(chunk.split()) for chunk in chunks)}")

    # Generate the system prompt for GPT
    system_prompt = gpt.generate_system_prompt(speakers)

    # Process each chunk
    transcript = {}
    previous_chunk = None
    total_entries_before_merge = 0

    # Track segment IDs to preserve timestamp information
    segment_id_map = {}  # Maps GPT output entry keys to original segment IDs

    for chunk_idx, chunk in enumerate(chunks):
        # Process this chunk
        processed_chunk = process_chunk(chunk, system_prompt, previous_chunk)

        # Find the segment IDs that correspond to this chunk's text
        # This is a best-effort matching of GPT-processed text back to original segments
        for entry_key, entry in processed_chunk.items():
            content = entry.get("Content") or entry.get("content")
            if not content:
                continue

            # Find best matching original segment
            best_match_id = None
            best_similarity = 0

            for seg_id, seg_info in timestamp_map.items():
                # Skip segments already assigned
                if seg_id in segment_id_map.values():
                    continue

                # Calculate text similarity
                similarity = SequenceMatcher(
                    None, content.lower(), seg_info["text"].lower()
                ).ratio()

                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match_id = seg_id

            # If we found a decent match, record the mapping
            if best_match_id and best_similarity > 0.5:
                segment_id_map[entry_key] = best_match_id

        # Count entries in this chunk
        chunk_entry_count = len(processed_chunk.keys())
        total_entries_before_merge += chunk_entry_count
        print(
            f"Chunk {chunk_idx+1}/{n_transcript_chunks} processed: {chunk_entry_count} entries"
        )

        # Better merging strategy: Use a consistent numbering scheme
        # Find the highest key in the transcript
        if transcript:
            start_key = max(map(int, transcript.keys())) + 1
        else:
            start_key = 1

        # Add entries from this chunk to the transcript with new keys
        for i, (_, value) in enumerate(processed_chunk.items(), start=0):
            key = str(start_key + i)
            transcript[key] = value

        # Use this chunk as context for the next one
        previous_chunk = processed_chunk

    print(
        f"Processing complete. Input: {len(whisper_transcript)} segments -> {total_entries_before_merge} entries before merge -> {len(transcript)} final entries"
    )

    # Now ensure all transcript entries have timestamp data
    # For entries without a direct match, estimate based on surrounding entries
    for key in sorted(transcript.keys(), key=int):
        if key in segment_id_map:
            # We have a direct match to an original segment
            original_seg_id = segment_id_map[key]
            if original_seg_id in timestamp_map:
                # Add timestamp information
                transcript[key]["original_start"] = timestamp_map[original_seg_id][
                    "start"
                ]
                transcript[key]["original_end"] = timestamp_map[original_seg_id]["end"]

    return transcript


def similarity(a, b):
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def time_to_seconds(time_str):
    t = datetime.datetime.strptime(time_str, "%H:%M:%S")
    return t.hour * 3600 + t.minute * 60 + t.second


def clean_text(text: str) -> str:
    """Clean text for comparison by removing punctuation, extra whitespace, and normalizing case."""
    # Remove punctuation and normalize whitespace
    text = re.sub(r"[^\w\s]", "", text.lower())
    text = " ".join(text.split())
    # Remove common filler words that might cause mismatches
    filler_words = {"um", "uh", "ah", "er", "like", "you know", "i mean"}
    words = text.split()
    words = [w for w in words if w not in filler_words]
    return " ".join(words)


def find_best_segment_match(
    content: str,
    whisper_segments: List[Dict],
    start_idx: int,
    window_size: int = 300,  # Increased window size
    min_similarity: float = 0.15,  # Lowered threshold further
) -> Tuple[Optional[Dict], Optional[int], float]:
    """
    Find the best matching segment in whisper_segments for the given content.
    Uses a sliding window approach to improve efficiency.
    """
    if not whisper_segments or start_idx >= len(whisper_segments):
        return None, None, 0.0

    # Clean the content for better matching
    clean_content = clean_text(content)

    # Try different content variations for more robust matching
    content_variations = [
        clean_content,
        # First 10 words
        (
            " ".join(clean_content.split()[:10])
            if len(clean_content.split()) > 10
            else clean_content
        ),
        # First 20 words
        (
            " ".join(clean_content.split()[:20])
            if len(clean_content.split()) > 20
            else clean_content
        ),
        # Last 10 words
        (
            " ".join(clean_content.split()[-10:])
            if len(clean_content.split()) > 10
            else clean_content
        ),
        # Middle portion (if long enough)
        (
            " ".join(clean_content.split()[10:30])
            if len(clean_content.split()) > 30
            else clean_content
        ),
        # First sentence
        (clean_content.split(".")[0] if "." in clean_content else clean_content),
    ]

    # Limit search window but ensure we don't go out of bounds
    end_idx = min(start_idx + window_size, len(whisper_segments))

    best_match = None
    best_idx = None
    best_sim = 0.0

    # First pass: try to find an exact match or high similarity match
    for i in range(start_idx, end_idx):
        segment = whisper_segments[i]
        clean_segment_text = clean_text(segment["text"])

        # Try all content variations for matching
        for variant in content_variations:
            sim = similarity(clean_segment_text, variant)
            if sim > best_sim:
                best_sim = sim
                best_match = segment
                best_idx = i

                # If we found a very good match, return immediately
                if sim > 0.8:
                    return best_match, best_idx, best_sim

    # If we didn't find a good match, try checking more segments
    if best_sim < min_similarity:
        # First try searching from the beginning
        broader_search_window = min(len(whisper_segments), window_size * 2)
        for i in range(0, broader_search_window):
            if i >= start_idx and i < end_idx:
                continue  # Skip segments we already checked

            segment = whisper_segments[i]
            clean_segment_text = clean_text(segment["text"])

            # Try all content variations for matching
            for variant in content_variations:
                sim = similarity(clean_segment_text, variant)
                if sim > best_sim:
                    best_sim = sim
                    best_match = segment
                    best_idx = i

        # If still no good match, try combining consecutive segments
        if best_sim < min_similarity:
            # Try combining 2-3 consecutive segments to find a match
            for i in range(0, len(whisper_segments) - 1):
                # Try combining 2 segments
                if i + 1 < len(whisper_segments):
                    combined_text = clean_text(
                        whisper_segments[i]["text"]
                        + " "
                        + whisper_segments[i + 1]["text"]
                    )
                    for variant in content_variations:
                        sim = similarity(combined_text, variant)
                        if sim > best_sim:
                            best_sim = sim
                            # Create a merged segment
                            best_match = {
                                "start": whisper_segments[i]["start"],
                                "end": whisper_segments[i + 1]["end"],
                                "text": whisper_segments[i]["text"]
                                + " "
                                + whisper_segments[i + 1]["text"],
                            }
                            best_idx = i

                # Try combining 3 segments
                if i + 2 < len(whisper_segments):
                    combined_text = clean_text(
                        whisper_segments[i]["text"]
                        + " "
                        + whisper_segments[i + 1]["text"]
                        + " "
                        + whisper_segments[i + 2]["text"]
                    )
                    for variant in content_variations:
                        sim = similarity(combined_text, variant)
                        if sim > best_sim:
                            best_sim = sim
                            # Create a merged segment
                            best_match = {
                                "start": whisper_segments[i]["start"],
                                "end": whisper_segments[i + 2]["end"],
                                "text": (
                                    whisper_segments[i]["text"]
                                    + " "
                                    + whisper_segments[i + 1]["text"]
                                    + " "
                                    + whisper_segments[i + 2]["text"]
                                ),
                            }
                            best_idx = i

    # If similarity is still below threshold, return None
    if best_sim < min_similarity:
        return None, None, 0.0

    return best_match, best_idx, best_sim


def merge_overlapping_segments(segments: List[Dict]) -> Dict:
    """Merge overlapping Whisper segments into a single segment."""
    if not segments:
        return None

    # Ensure timestamps are reasonable
    max_duration = 300  # Maximum reasonable duration for a single segment (5 minutes)
    start = segments[0]["start"]
    end = segments[-1]["end"]

    # If the duration is unreasonable, try to fix it
    if end - start > max_duration:
        # Look for a more reasonable endpoint
        for i in range(len(segments) - 1, -1, -1):
            if segments[i]["end"] - start <= max_duration:
                end = segments[i]["end"]
                segments = segments[: i + 1]
                break
        else:
            # If we can't find a reasonable endpoint, limit to max duration
            end = start + max_duration

    # Combine text with proper spacing
    text = segments[0]["text"]
    for segment in segments[1:]:
        if not text.endswith(".") and not text.endswith("?") and not text.endswith("!"):
            text += " "
        text += segment["text"]

    return {"start": start, "end": end, "text": text}


def align_timestamps(
    processed_transcript: Dict, whisper_transcript: List[Dict]
) -> Dict:
    """
    Align processed transcript with original timestamps.
    Prioritizes using preserved timestamp information from the processing step.

    Args:
        processed_transcript: The processed transcript with speaker identification
        whisper_transcript: The original Whisper output with accurate timestamps

    Returns:
        Dict: The aligned transcript with accurate timestamps
    """
    if not isinstance(processed_transcript, dict) or not isinstance(
        whisper_transcript, list
    ):
        raise TypeError(
            f"Invalid input types: processed_transcript is {type(processed_transcript)}, "
            f"whisper_transcript is {type(whisper_transcript)}"
        )

    print(
        f"Starting align_timestamps with {len(processed_transcript)} processed entries "
        f"and {len(whisper_transcript)} whisper entries"
    )

    aligned_transcript = {}
    entries_with_timestamps = 0
    entries_requiring_matching = 0

    # First pass: Use preserved timestamps where available
    for key, item in processed_transcript.items():
        if not isinstance(item, dict):
            continue

        content = item.get("Content") or item.get("content")
        speaker = item.get("Speaker") or item.get("speaker")

        if not content:
            continue

        # Check if this entry has preserved timestamp information
        if "original_start" in item and "original_end" in item:
            aligned_transcript[key] = {
                "Speaker": speaker,
                "Content": content,
                "Start": item["original_start"],
                "End": item["original_end"],
            }
            entries_with_timestamps += 1

    # Second pass: For entries without timestamps, estimate based on surrounding entries
    sorted_keys = sorted(processed_transcript.keys(), key=int)
    timestamp_keys = sorted([k for k in aligned_transcript.keys()], key=int)

    for key in sorted_keys:
        if key in aligned_transcript:
            continue  # Already processed

        item = processed_transcript[key]
        content = item.get("Content") or item.get("content")
        speaker = item.get("Speaker") or item.get("speaker")

        if not content:
            continue

        # Find nearest entries with timestamps before and after
        prev_key = None
        next_key = None

        for ts_key in timestamp_keys:
            if int(ts_key) < int(key):
                prev_key = ts_key
            elif int(ts_key) > int(key):
                next_key = ts_key
                break

        # Calculate position based on available reference points
        if prev_key and next_key:
            # Interpolate between two known points
            prev_end = aligned_transcript[prev_key]["End"]
            next_start = aligned_transcript[next_key]["Start"]

            # Estimate duration based on content length (approx. 15 chars per second)
            content_chars = len(content)
            estimated_duration = max(1.0, content_chars / 15)

            # Position proportionally in the gap
            total_gap = next_start - prev_end
            if total_gap <= 0:
                # If timestamps are too close or overlapping, place right after prev
                start_time = prev_end + 0.05
            else:
                # Proportionally divide the gap based on content size
                start_time = prev_end + (total_gap / 3)  # Place in first third of gap

            end_time = min(next_start - 0.05, start_time + estimated_duration)

        elif prev_key:
            # Only have a previous entry with timestamp
            prev_end = aligned_transcript[prev_key]["End"]
            start_time = prev_end + 0.1

            # Estimate duration
            content_chars = len(content)
            estimated_duration = max(1.0, content_chars / 15)
            end_time = start_time + estimated_duration

        elif next_key:
            # Only have a next entry with timestamp
            next_start = aligned_transcript[next_key]["Start"]

            # Estimate duration
            content_chars = len(content)
            estimated_duration = max(1.0, content_chars / 15)
            end_time = next_start - 0.1
            start_time = max(0, end_time - estimated_duration)

        else:
            # No reference points, use fallback to original matching algorithm
            # This is unlikely now but provides a safety net
            best_match, _, _ = find_best_segment_match(
                content,
                whisper_transcript,
                0,
                window_size=len(whisper_transcript),
                min_similarity=0.1,
            )

            if best_match:
                start_time = best_match["start"]
                end_time = best_match["end"]
            else:
                # Last resort: estimate position based on key number
                key_number = int(key)
                total_keys = len(processed_transcript)
                if not whisper_transcript:
                    # Default values if no reference available
                    start_time = key_number * 5.0
                    end_time = start_time + 5.0
                else:
                    # Distribute based on total audio duration
                    total_duration = whisper_transcript[-1]["end"]
                    position_ratio = key_number / total_keys
                    start_time = total_duration * position_ratio
                    end_time = start_time + min(5.0, total_duration / total_keys)

        # Add the aligned entry
        aligned_transcript[key] = {
            "Speaker": speaker,
            "Content": content,
            "Start": start_time,
            "End": end_time,
        }
        entries_requiring_matching += 1

    print(f"Finished aligning {len(aligned_transcript)} entries.")
    print(f"  - {entries_with_timestamps} entries used preserved timestamps")
    print(f"  - {entries_requiring_matching} entries required timestamp estimation")

    return aligned_transcript


def format_entry(entry):
    speaker = entry["Speaker"]
    start = utils.convert_to_timestamp(entry["Start"])
    end = utils.convert_to_timestamp(entry["End"])
    content = entry["Content"]

    formatted = f"{speaker} [{start} - {end}]\n{content}\n\n"
    return formatted


def export_transcript(transcript, filename="transcript"):
    with open(f"data/transcripts/{filename}.txt", "w") as f:
        for key in sorted(transcript.keys(), key=int):  # Sort keys numerically
            entry = transcript[key]
            formatted_entry = format_entry(entry)
            f.write(formatted_entry)


# Main execution
if __name__ == "__main__":
    pass
