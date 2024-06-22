import json
from typing import List, Dict, Any

def save_whisper_output(combined_transcript_segments: List[Dict[str, Any]], filename: str) -> None:
    """
    
    """
    with open(f'{filename}_whisper_output.json', 'w') as json_file:
        json.dump(combined_transcript_segments, json_file, indent=4)
    return None

def load_whisper_output(filepath: str) -> List[Dict[str, Any]]:
    with open(f"{filepath}_whisper_output.json", "r") as f:
        whisper_output = json.load(f)
    return whisper_output

def save_processed_transcript(combined_processed_chunks: List[Dict[str, Any]], filename: str) -> None:
    """
    """
    with open(f'{filename}_final_output.json', 'w') as json_file:
        json.dump(combined_processed_chunks, json_file, indent=4)
    return None

# Main execution
if __name__ == "__main__":
    pass