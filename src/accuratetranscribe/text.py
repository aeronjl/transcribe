import datetime
from difflib import SequenceMatcher
import re
import json
from typing import List, Dict, Any, Tuple, Union, Optional

import tiktoken

from . import gpt

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
    text_buffer = ""
    chunks = []
    for index, segment in enumerate(transcript_segments):
        text_buffer += segment["text"]
        token_count = len(ENCODING.encode(text_buffer))

        if index == len(transcript_segments) - 1:
            chunks.append(text_buffer)
            text_buffer = ""
            break

        if token_count > token_limit:
            if text_buffer.strip().endswith("."):
                chunks.append(text_buffer)
                text_buffer = ""

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
    prompt = system_prompt
    if previous_chunk:
        chunk_items = list(previous_chunk.items())
        prompt += "\n---\nYou are continuing on from a previous transcription which ended as follows:\n\n"
        prompt += "\n".join(f'"{k} : {json.dumps(v)},' for k, v in chunk_items)

    completion: Dict = gpt.process_transcription(chunk, prompt)

    # Clean and parse the JSON output
    cleaned_json: Optional[Dict] = clean_and_parse_json(
        completion.choices[0].message.content
    )

    if cleaned_json:
        return cleaned_json
    else:
        print("Warning: JSON parsing failed in process_chunk")
        return {
            "error": "JSON parsing failed",
            "raw_content": completion.choices[0].message.content[:1000],
        }


def process_whisper_transcription(transcribed_audio_segments, speakers=None):
    chunks, n_transcript_chunks = chunk_transcript_to_token_limit(
        transcribed_audio_segments, token_limit=1200
    )
    print(
        f"Processing {n_transcript_chunks} transcript chunks. Estimated time: {n_transcript_chunks * 30} seconds."
    )

    system_prompt = gpt.generate_system_prompt(speakers)

    processed_chunks = {}
    previous_chunk = None
    for chunk in chunks:
        processed_chunk = process_chunk(chunk, system_prompt, previous_chunk)

        # Merge the new chunk into the accumulated dictionary
        # Use the highest existing key + 1 as the starting point for new keys
        start_key = max(map(int, processed_chunks.keys() or ["0"])) + 1
        for i, (key, value) in enumerate(processed_chunk.items(), start=start_key):
            processed_chunks[str(i)] = value

        previous_chunk = processed_chunk

    return processed_chunks


def similarity(a, b):
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def time_to_seconds(time_str):
    t = datetime.datetime.strptime(time_str, "%H:%M:%S")
    return t.hour * 3600 + t.minute * 60 + t.second


def align_timestamps(
    processed_transcript: Dict, whisper_transcript: List[Dict]
) -> Dict:
    if not isinstance(processed_transcript, dict) or not isinstance(
        whisper_transcript, list
    ):
        raise TypeError(
            f"Invalid input types: processed_transcript is {type(processed_transcript)}, whisper_transcript is {type(whisper_transcript)}"
        )

    print(
        f"Starting align_timestamps with {len(processed_transcript)} processed entries and {len(whisper_transcript)} whisper entries"
    )

    aligned_transcript = {}
    current_whisper_index = 0

    for key, item in processed_transcript.items():
        if not isinstance(item, dict):
            print(f"Warning: Item for {key} is not a dictionary: {item}")
            continue

        content = item.get("Content") or item.get("content")
        if not content:
            print(f"Warning: 'Content' key not found for {key} in item {item}")
            continue

        speaker = item.get("Speaker") or item.get("speaker")
        if content is None:
            print(f"Warning: No 'Content' found for key {key}: {item}")
            continue

        # Find the best match among candidates
        best_match = None
        best_similarity = -1
        for i in range(current_whisper_index, len(whisper_transcript)):
            whisper_entry = whisper_transcript[i]
            current_similarity = similarity(content, whisper_entry.get("text", ""))
            if current_similarity > best_similarity:
                best_similarity = current_similarity
                best_match = whisper_entry
                current_whisper_index = i

        if best_match is None:
            print(f"Warning: No match found for key {key}")
            continue

        aligned_transcript[key] = {
            "Speaker": speaker,
            "Content": content,
            "Start": best_match["start"],
            "End": best_match["end"],
        }

        current_whisper_index += 1

    print(f"Finished aligning {len(aligned_transcript)} entries.")
    return aligned_transcript


def format_entry(entry):
    speaker = entry["Speaker"]
    start = entry["Start"]
    end = entry["End"]
    content = entry["Content"]

    formatted = f"{speaker}\n{start} - {end}\n\n{content}\n\n"
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
