import json
from typing import List, Dict, Any


def save_whisper_output(
    combined_transcript_segments: List[Dict[str, Any]], filepath: str
) -> None:
    """ """
    with open(f"{filepath}_whisper_output.json", "w") as json_file:
        json.dump(combined_transcript_segments, json_file, indent=4)
    return None


def load_whisper_output(filepath: str) -> List[Dict[str, Any]]:
    with open(f"{filepath}_whisper_output.json", "r") as f:
        whisper_output = json.load(f)
    return whisper_output


def load_processed_transcript(filepath: str) -> Dict[str, Any]:
    with open(f"{filepath}_final_output.json", "r") as f:
        processed_transcript = json.load(f)
    return processed_transcript


def save_processed_transcript(
    combined_processed_chunks: Dict[str, Any], filepath: str
) -> None:
    """ """
    with open(f"{filepath}_final_output.json", "w") as json_file:
        json.dump(combined_processed_chunks, json_file, indent=4)
    return None


def save_aligned_transcript(aligned_transcript: Dict[str, Any], filepath: str) -> None:
    """Save the aligned transcript with timestamps."""
    with open(f"{filepath}_aligned_transcript.json", "w") as f:
        json.dump(aligned_transcript, f, indent=2)
    return None


def load_aligned_transcript(filepath: str) -> Dict[str, Any]:
    """Load an aligned transcript with timestamps."""
    with open(f"{filepath}_aligned_transcript.json", "r") as f:
        aligned_transcript = json.load(f)
    return aligned_transcript


# Main execution
if __name__ == "__main__":
    pass
