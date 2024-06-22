from typing import List, Dict, Any, Tuple

import tiktoken

# Constants
ENCODING = tiktoken.get_encoding("cl100k_base")

def renumber_transcribed_audio_segment(
    transcribed_audio_segment: List[Dict[str, Any]],
    number_to_start_from: int = 0,
    time_to_start_from: float = 0
) -> List[Dict[str, Any]]:
    for phrase_index, phrase in enumerate(transcribed_audio_segment):
        phrase['id'] = number_to_start_from + phrase_index + 1
        phrase['start'] = phrase['start'] + time_to_start_from
        phrase['end'] = phrase['end'] + time_to_start_from
    return transcribed_audio_segment

def chunk_transcript_to_token_limit(
    transcript_segments: List[Dict[str, Any]],
    token_limit: int = 1200
) -> Tuple[List[str], int]:
    text_buffer = ''
    chunks = []
    for index, segment in enumerate(transcript_segments):    
        text_buffer += segment['text']
        token_count = len(ENCODING.encode(text_buffer))
        
        if index == len(transcript_segments) - 1:
            chunks.append(text_buffer)
            text_buffer = ''
            break
        
        if token_count > token_limit:
            if text_buffer.strip().endswith('.'):
                chunks.append(text_buffer)
                text_buffer = ''
                    
    return chunks, len(chunks)

# Main execution
if __name__ == "__main__":
    pass