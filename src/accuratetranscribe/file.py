import json
from typing import List, Dict, Any

def save_whisper_output(combined_transcript_segments: List[Dict[str, Any]], filepath: str) -> None:
    """
    
    """
    with open(f'{filepath}_whisper_output.json', 'w') as json_file:
        json.dump(combined_transcript_segments, json_file, indent=4)
    return None

def load_whisper_output(filepath: str) -> List[Dict[str, Any]]:
    with open(f"{filepath}_whisper_output.json", "r") as f:
        whisper_output = json.load(f)
    return whisper_output

def save_processed_transcript(combined_processed_chunks: List[Dict[str, Any]], filepath: str) -> None:
    """
    """
    with open(f'{filepath}_final_output.json', 'w') as json_file:
        json.dump(combined_processed_chunks, json_file, indent=4)
    return None

def save_aligned_transcript(aligned_transcript, filepath) -> None:
    with open(f'{filepath}_aligned_transcript.json', 'w') as f:
        json.dump(aligned_transcript, f, indent=2)
    return None

# Main execution
if __name__ == "__main__":
    pass